// Qwen3.5 0.8B / 2B engine: GGUF -> kernel row layout -> persistent decode launches.
//
// Only this file knows about the GGUF.  The kernel gets the exact same Q8_0 weights as
// llama.cpp, with the fp16 scales copied bit-for-bit (they are moved to the end of each
// row but never rounded).  Norm tensors are used as stored: llama.cpp's converter already
// added the HF +1 offset to every *.norm.weight except linear_attn.norm.
#include "qwen35.h"
#include "gguf_parser.h"
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q35_GRID 40
#define Q35_MAX_BATCH 32
#define Q35_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[qwen35] CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    return -1; } } while (0)

typedef struct {
    float *conv_hist;
    float *state;
    void  *k_cache;
    void  *v_cache;
} q35_slot_t;

struct q35_model {
    gguf_context_t *gguf;
    int hid, inter, vocab, n_layers, nq, nkv, hd, rot, lin_h, lin_k, lin_v;
    int ctx_model;
    int bos, eos;
    int prepared;
    q35_params_t params;
    void **dev_allocs;
    int n_dev_allocs;
};

struct q35_engine {
    q35_model *m;
    int ctx, nslots;
    q35_slot_t *slots;
    int epoch;
    float *rope_cos, *rope_sin;
    unsigned long long *xl, *projl, *corel, *attnl, *actl, *lml;
    float *attn_part;
    unsigned *attn_cnt;
    int32_t *d_forced;
    int *d_slot_ids;
    int *d_positions;
    void *vecf;
    unsigned long long *d_tokens;
    int *d_info;
    float *d_logits;
    int logits_cap;
};

static int arch_is_qwen35(const char *path) {
    gguf_context_t *g = tuff_gguf_parse_file(path);
    if (!g) return 0;
    const gguf_kv_t *kv = tuff_gguf_get_kv(g, "general.architecture");
    const int ok = kv && kv->type == GGUF_TYPE_STRING && strcmp(kv->val.str, "qwen35") == 0;
    tuff_gguf_free(g);
    return ok;
}

static const uint8_t *tensor_data(gguf_context_t *g, const char *name, const gguf_tensor_info_t **info,
                                  int ne0, int ne1, uint32_t type) {
    const gguf_tensor_info_t *t = tuff_gguf_get_tensor(g, name);
    if (!t) {
        fprintf(stderr, "[qwen35] missing tensor %s\n", name);
        return NULL;
    }
    if (ne0 > 0 && (int)t->dims[0] != ne0) {
        fprintf(stderr, "[qwen35] %s: ne0 %llu != %d\n", name, (unsigned long long)t->dims[0], ne0);
        return NULL;
    }
    if (ne1 > 0 && (int)t->dims[1] != ne1) {
        fprintf(stderr, "[qwen35] %s: ne1 %llu != %d\n", name, (unsigned long long)t->dims[1], ne1);
        return NULL;
    }
    if (t->type != type) {
        fprintf(stderr, "[qwen35] %s: type %u != %u\n", name, t->type, type);
        return NULL;
    }
    *info = t;
    return g->tensor_data_blob + t->offset;
}

// The kernels read their params from one __constant__ copy.  Every launch path goes through here, so
// the 4 KB upload is skipped when the params are byte-identical to the last ones uploaded.
static q35_params_t g_last_params;
static int g_have_last_params = 0;
static int set_params_cached(const q35_params_t *p) {
    if (g_have_last_params && memcmp(&g_last_params, p, sizeof(*p)) == 0) return 0;
    if (q35_mk_set_params(p) != 0) { g_have_last_params = 0; return -1; }
    g_last_params = *p;
    g_have_last_params = 1;
    return 0;
}

static void *dev_alloc(q35_model *m, size_t bytes, int zero) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) return NULL;
    if (zero) cudaMemset(p, 0, bytes);
    m->dev_allocs = realloc(m->dev_allocs, sizeof(void *) * (m->n_dev_allocs + 1));
    m->dev_allocs[m->n_dev_allocs++] = p;
    return p;
}

// Repack `rows` Q8_0 rows of length K into dst at row offset `row_off`.
// GGUF block = {fp16 d; int8 q[32]}; kernel row = K bytes of (q+128), then K/32 fp16 d.
static void repack_into(unsigned char *dst, size_t row_off, const unsigned char *src, int rows, int K) {
    const int nb = K / 32;
    const size_t rb = (size_t)K + (size_t)nb * 2;
    const size_t sb = (size_t)nb * 34;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const unsigned char *s = src + (size_t)r * sb;
        unsigned char *d = dst + (size_t)(row_off + r) * rb;
        for (int b = 0; b < nb; b++) {
            const unsigned char *q = s + (size_t)b * 34 + 2;
            unsigned char *dq = d + (size_t)b * 32;
            for (int j = 0; j < 32; j++) dq[j] = (unsigned char)((int)(signed char)q[j] + 128);
            d[K + (size_t)b * 2 + 0] = s[(size_t)b * 34 + 0];
            d[K + (size_t)b * 2 + 1] = s[(size_t)b * 34 + 1];
        }
    }
}

// Repack one Q8_0 matrix (signed q bytes) into the batch GEMV fragment layout
// [group][kt][544].  `group_off` is the first 16-row group written; rows must be a
// multiple of 16.  Within a 544-byte tile: for h in {0,1} (k16 half of the Q8 block),
// q in {0,1} (k columns 0..7 / 8..15 of the half), lane L: 4 bytes
// (r0,c0)(r0,c0+1)(r1,c0)(r1,c0+1); 16 fp16 block scales at +512.
static void repack_frag_into(unsigned char *dst, size_t group_off, const unsigned char *src,
                             int rows, int K) {
    const int nb = K / 32;
    const size_t sb = (size_t)nb * 34;
    const int ng = rows / 16;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ng; g++) {
        for (int kt = 0; kt < nb; kt++) {
            unsigned char *gt = dst + ((group_off + (size_t)g) * nb + kt) * 544;
            for (int h = 0; h < 2; h++)
                for (int qq = 0; qq < 2; qq++)
                    for (int lane = 0; lane < 32; lane++) {
                        const int r0 = lane >> 2, c0 = (lane & 3) * 2, r1 = r0 + 8;
                        const int kb = kt * 32 + h * 16 + qq * 8 + c0;
                        unsigned char *o = gt + ((h * 2 + qq) * 32 + lane) * 4;
                        o[0] = src[(size_t)(g * 16 + r0) * sb + (size_t)kt * 34 + 2 + h * 16 + qq * 8 + c0];
                        o[1] = src[(size_t)(g * 16 + r0) * sb + (size_t)kt * 34 + 2 + h * 16 + qq * 8 + c0 + 1];
                        o[2] = src[(size_t)(g * 16 + r1) * sb + (size_t)kt * 34 + 2 + h * 16 + qq * 8 + c0];
                        o[3] = src[(size_t)(g * 16 + r1) * sb + (size_t)kt * 34 + 2 + h * 16 + qq * 8 + c0 + 1];
                        (void)kb;
                    }
            for (int r = 0; r < 16; r++) {
                const unsigned char *sp = src + (size_t)(g * 16 + r) * sb + (size_t)kt * 34;
                gt[512 + 2 * r + 0] = sp[0];
                gt[512 + 2 * r + 1] = sp[1];
            }
        }
    }
}

static void *upload_concat_frag(q35_model *m, const char **names, const int *rows, int n, int K) {
    const int nb = K / 32;
    size_t total_groups = 0;
    for (int i = 0; i < n; i++) total_groups += (size_t)rows[i] / 16;
    const size_t bytes = total_groups * nb * 544;
    unsigned char *h = (unsigned char *)malloc(bytes);
    if (!h) return NULL;
    size_t goff = 0;
    for (int i = 0; i < n; i++) {
        const gguf_tensor_info_t *ti = NULL;
        const uint8_t *src = tensor_data(m->gguf, names[i], &ti, K, rows[i], 8);
        if (!src) { free(h); return NULL; }
        repack_frag_into(h, goff, src, rows[i], K);
        goff += (size_t)rows[i] / 16;
    }
    void *d = dev_alloc(m, bytes, 0);
    if (d) cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    free(h);
    return d;
}

static void *upload_matrix_frag(q35_model *m, const char *name, int K, int rows) {
    const gguf_tensor_info_t *ti = NULL;
    const uint8_t *src = tensor_data(m->gguf, name, &ti, K, rows, 8);
    if (!src) return NULL;
    const size_t bytes = (size_t)rows * K + (size_t)rows * (K / 32) * 2;
    unsigned char *h = (unsigned char *)malloc(bytes);
    if (!h) return NULL;
    repack_frag_into(h, 0, src, rows, K);
    void *d = dev_alloc(m, bytes, 0);
    if (d) cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// gate/up fragment layout: 16-row group g holds gate rows 8g..8g+7 in rows 0..7 and the
// matching up rows in rows 8..15, so the MMA output has the gate/up pair in one lane.
static void *upload_gate_up_frag(q35_model *m, const char *gate, const char *up, int K, int rows) {
    const gguf_tensor_info_t *tg = NULL, *tu = NULL;
    const uint8_t *g = tensor_data(m->gguf, gate, &tg, K, rows, 8);
    const uint8_t *u = tensor_data(m->gguf, up, &tu, K, rows, 8);
    if (!g || !u) return NULL;
    const int nb = K / 32;
    const size_t sb = (size_t)nb * 34;
    const int ng = rows / 8;                       // one output group per 8 FFN rows
    const size_t bytes = (size_t)ng * nb * 544;
    unsigned char *h = (unsigned char *)malloc(bytes);
    if (!h) return NULL;
#pragma omp parallel for schedule(static)
    for (int grp = 0; grp < ng; grp++) {
        for (int kt = 0; kt < nb; kt++) {
            unsigned char *gt = h + ((size_t)grp * nb + kt) * 544;
            for (int hh = 0; hh < 2; hh++)
                for (int qq = 0; qq < 2; qq++)
                    for (int lane = 0; lane < 32; lane++) {
                        const int r0 = lane >> 2, c0 = (lane & 3) * 2;
                        unsigned char *o = gt + ((hh * 2 + qq) * 32 + lane) * 4;
                        const uint8_t *sg = g + (size_t)(grp * 8 + r0) * sb + (size_t)kt * 34 + 2;
                        const uint8_t *su = u + (size_t)(grp * 8 + r0) * sb + (size_t)kt * 34 + 2;
                        const int off = hh * 16 + qq * 8 + c0;
                        o[0] = sg[off]; o[1] = sg[off + 1];
                        o[2] = su[off]; o[3] = su[off + 1];
                    }
            for (int r = 0; r < 8; r++) {
                const unsigned char *spg = g + (size_t)(grp * 8 + r) * sb + (size_t)kt * 34;
                const unsigned char *spu = u + (size_t)(grp * 8 + r) * sb + (size_t)kt * 34;
                gt[512 + 2 * r + 0] = spg[0]; gt[512 + 2 * r + 1] = spg[1];
                gt[512 + 2 * (r + 8) + 0] = spu[0]; gt[512 + 2 * (r + 8) + 1] = spu[1];
            }
        }
    }
    void *d = dev_alloc(m, bytes, 0);
    if (d) cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    free(h);
    return d;
}

static void *upload_matrix(q35_model *m, const char *name, int K, int rows) {
    const gguf_tensor_info_t *ti = NULL;
    const uint8_t *src = tensor_data(m->gguf, name, &ti, K, rows, 8);
    if (!src) return NULL;
    const size_t rb = (size_t)K + (size_t)(K / 32) * 2;
    unsigned char *h = (unsigned char *)malloc(rb * rows);
    if (!h) return NULL;
    repack_into(h, 0, src, rows, K);
    void *d = dev_alloc(m, rb * rows, 0);
    if (d) cudaMemcpy(d, h, rb * rows, cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// Interleave two [rows,K] matrices as gate0, up0, gate1, up1, ...
static void *upload_gate_up(q35_model *m, const char *gate, const char *up, int K, int rows) {
    const gguf_tensor_info_t *tg = NULL, *tu = NULL;
    const uint8_t *g = tensor_data(m->gguf, gate, &tg, K, rows, 8);
    const uint8_t *u = tensor_data(m->gguf, up, &tu, K, rows, 8);
    if (!g || !u) return NULL;
    const size_t rb = (size_t)K + (size_t)(K / 32) * 2;
    unsigned char *h = (unsigned char *)malloc(rb * 2 * rows);
    if (!h) return NULL;
    repack_into(h, 0, g, rows, K);
    repack_into(h, (size_t)rows, u, rows, K);
    // h currently [gate rows][up rows]; interleave into a second buffer.
    unsigned char *hi = (unsigned char *)malloc(rb * 2 * rows);
    if (!hi) { free(h); return NULL; }
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const int r1 = r;
        (void)r1;
        memcpy(hi + (size_t)(2 * r) * rb, h + (size_t)r * rb, rb);
        memcpy(hi + (size_t)(2 * r + 1) * rb, h + (size_t)(rows + r) * rb, rb);
    }
    void *d = dev_alloc(m, rb * 2 * rows, 0);
    if (d) cudaMemcpy(d, hi, rb * 2 * rows, cudaMemcpyHostToDevice);
    free(h);
    free(hi);
    return d;
}

static void *upload_concat(q35_model *m, const char **names, const int *rows, int n, int K) {
    const size_t rb = (size_t)K + (size_t)(K / 32) * 2;
    int total = 0;
    for (int i = 0; i < n; i++) total += rows[i];
    unsigned char *h = (unsigned char *)malloc(rb * total);
    if (!h) return NULL;
    int off = 0;
    for (int i = 0; i < n; i++) {
        const gguf_tensor_info_t *ti = NULL;
        const uint8_t *src = tensor_data(m->gguf, names[i], &ti, K, rows[i], 8);
        if (!src) { free(h); return NULL; }
        repack_into(h, off, src, rows[i], K);
        off += rows[i];
    }
    void *d = dev_alloc(m, rb * total, 0);
    if (d) cudaMemcpy(d, h, rb * total, cudaMemcpyHostToDevice);
    free(h);
    return d;
}

static float *upload_f32(q35_model *m, const char *name, int n) {
    const gguf_tensor_info_t *ti = NULL;
    const uint8_t *src = tensor_data(m->gguf, name, &ti, 0, 0, 0);
    if (!src) return NULL;
    if ((int)ti->nelements != n) {
        fprintf(stderr, "[qwen35] %s: %llu elements != %d\n", name,
                (unsigned long long)ti->nelements, n);
        return NULL;
    }
    float *d = (float *)dev_alloc(m, sizeof(float) * n, 0);
    if (d) cudaMemcpy(d, src, sizeof(float) * n, cudaMemcpyHostToDevice);
    return d;
}

static int kv_u32(gguf_context_t *g, const char *k, int def) {
    return (int)tuff_gguf_get_kv_u32(g, k, (uint32_t)def);
}

int q35_model_is_arch(const char *path) { return arch_is_qwen35(path); }

q35_model *q35_model_load(const char *path) {
    if (!arch_is_qwen35(path)) return NULL;
    gguf_context_t *g = tuff_gguf_parse_file(path);
    if (!g) return NULL;
    q35_model *m = (q35_model *)calloc(1, sizeof(*m));
    if (!m) { tuff_gguf_free(g); return NULL; }
    m->gguf = g;
    m->hid = kv_u32(g, "qwen35.embedding_length", 0);
    m->inter = kv_u32(g, "qwen35.feed_forward_length", 0);
    m->n_layers = kv_u32(g, "qwen35.block_count", 0);
    m->nq = kv_u32(g, "qwen35.attention.head_count", 0);
    m->nkv = kv_u32(g, "qwen35.attention.head_count_kv", 0);
    m->hd = kv_u32(g, "qwen35.attention.key_length", 0);
    m->rot = kv_u32(g, "qwen35.rope.dimension_count", 0);
    m->lin_h = kv_u32(g, "qwen35.ssm.group_count", 0);
    m->lin_k = kv_u32(g, "qwen35.ssm.state_size", 0);
    m->lin_v = kv_u32(g, "qwen35.ssm.inner_size", 0) / (m->lin_h ? m->lin_h : 1);
    m->ctx_model = kv_u32(g, "qwen35.context_length", 4096);
    m->vocab = 0;
    const gguf_tensor_info_t *te = tuff_gguf_get_tensor(g, "token_embd.weight");
    if (te) m->vocab = (int)te->dims[1];
    m->bos = -1;
    m->eos = (int)tuff_gguf_get_kv_u32(g, "tokenizer.ggml.eos_token_id", 248046);
    if (m->hid <= 0 || m->inter <= 0 || m->n_layers != Q35_MAX_LAYERS || m->nq != Q35_NQ ||
        m->nkv != Q35_NKV || m->hd != Q35_HD || m->rot != Q35_ROT || m->lin_h != Q35_LIN_H ||
        m->lin_k != Q35_LIN_DK || m->lin_v != Q35_LIN_DV || m->vocab != Q35_VOCAB ||
        (m->hid != 1024 && m->hid != 2048) || (m->hid == 1024 ? m->inter != 3584 : m->inter != 6144)) {
        fprintf(stderr, "[qwen35] unsupported shape hid=%d inter=%d layers=%d vocab=%d\n",
                m->hid, m->inter, m->n_layers, m->vocab);
        q35_model_free(m);
        return NULL;
    }
    return m;
}

void q35_model_free(q35_model *m) {
    if (!m) return;
    for (int i = 0; i < m->n_dev_allocs; i++) cudaFree(m->dev_allocs[i]);
    free(m->dev_allocs);
    if (m->gguf) tuff_gguf_free(m->gguf);
    free(m);
}

gguf_context_t *q35_model_gguf(q35_model *m) { return m ? m->gguf : NULL; }
int q35_model_vocab(const q35_model *m) { return m ? m->vocab : 0; }
int q35_model_bos(const q35_model *m) { return m ? m->bos : -1; }
int q35_model_eos(const q35_model *m) { return m ? m->eos : -1; }

int q35_prepare(q35_model *m, int *max_context_len) {
    (void)max_context_len;
    if (!m || m->prepared) return m ? 0 : -1;
    const int HID = m->hid, INTER = m->inter;
    q35_params_t *P = &m->params;
    memset(P, 0, sizeof(*P));
    P->hid = HID;
    P->inter = INTER;

    for (int i = 0; i < Q35_MAX_LAYERS; i++) {
        char nm[128];
        q35_layer_ptrs_t *L = &P->layer[i];
        L->is_attn = ((i + 1) % 4 == 0);
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", i);
        L->in_norm = upload_f32(m, nm, HID);
        snprintf(nm, sizeof nm, "blk.%d.post_attention_norm.weight", i);
        L->post_norm = upload_f32(m, nm, HID);
        if (!L->in_norm || !L->post_norm) return -1;

        if (L->is_attn) {
            const char *names[3];
            int rows[3] = {Q35_NQ * 2 * Q35_HD, Q35_NKV * Q35_HD, Q35_NKV * Q35_HD};
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", i); names[0] = strdup(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", i); names[1] = strdup(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", i); names[2] = strdup(nm);
            L->w_in = upload_concat(m, names, rows, 3, HID);
            for (int k = 0; k < 3; k++) free((void *)names[k]);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", i);
            L->w_out = upload_matrix(m, nm, Q35_NQ * Q35_HD, HID);
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", i);
            L->q_norm = upload_f32(m, nm, Q35_HD);
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", i);
            L->k_norm = upload_f32(m, nm, Q35_HD);
            {
                const char *fn[3];
                int fr[3] = {Q35_NQ * 2 * Q35_HD, Q35_NKV * Q35_HD, Q35_NKV * Q35_HD};
                char b1[128], b2[128], b3[128];
                snprintf(b1, sizeof b1, "blk.%d.attn_q.weight", i);
                snprintf(b2, sizeof b2, "blk.%d.attn_k.weight", i);
                snprintf(b3, sizeof b3, "blk.%d.attn_v.weight", i);
                fn[0] = b1; fn[1] = b2; fn[2] = b3;
                L->w_in_f = upload_concat_frag(m, fn, fr, 3, HID);
                char b4[128];
                snprintf(b4, sizeof b4, "blk.%d.attn_output.weight", i);
                L->w_out_f = upload_matrix_frag(m, b4, Q35_NQ * Q35_HD, HID);
            }
            if (!L->w_in || !L->w_out || !L->q_norm || !L->k_norm || !L->w_in_f || !L->w_out_f) return -1;
        } else {
            // Fused input projection rows: qkv(6144) | z(2048) | beta(16) | alpha(16).
            const char *names[4];
            int rows[4] = {Q35_CONV_DIM, Q35_LIN_V, Q35_LIN_H, Q35_LIN_H};
            snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", i); names[0] = strdup(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", i); names[1] = strdup(nm);
            snprintf(nm, sizeof nm, "blk.%d.ssm_beta.weight", i); names[2] = strdup(nm);
            snprintf(nm, sizeof nm, "blk.%d.ssm_alpha.weight", i); names[3] = strdup(nm);
            L->w_in = upload_concat(m, names, rows, 4, HID);
            L->w_in_f = upload_concat_frag(m, names, rows, 4, HID);
            for (int k = 0; k < 4; k++) free((void *)names[k]);
            snprintf(nm, sizeof nm, "blk.%d.ssm_out.weight", i);
            L->w_out = upload_matrix(m, nm, Q35_LIN_V, HID);
            L->w_out_f = upload_matrix_frag(m, nm, Q35_LIN_V, HID);
            snprintf(nm, sizeof nm, "blk.%d.ssm_conv1d.weight", i);
            L->conv_w = upload_f32(m, nm, Q35_CONV_DIM * Q35_CONV_K);
            snprintf(nm, sizeof nm, "blk.%d.ssm_a", i);
            L->ssm_a = upload_f32(m, nm, Q35_LIN_H);
            snprintf(nm, sizeof nm, "blk.%d.ssm_dt.bias", i);
            L->dt_bias = upload_f32(m, nm, Q35_LIN_H);
            snprintf(nm, sizeof nm, "blk.%d.ssm_norm.weight", i);
            L->gdn_norm = upload_f32(m, nm, Q35_LIN_DV);
            if (!L->w_in || !L->w_out || !L->conv_w || !L->ssm_a || !L->dt_bias || !L->gdn_norm) return -1;
        }
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", i);
        char nm2[128];
        snprintf(nm2, sizeof nm2, "blk.%d.ffn_up.weight", i);
        char nm3[128];
        snprintf(nm3, sizeof nm3, "blk.%d.ffn_down.weight", i);
        L->w_gu = upload_gate_up(m, nm, nm2, HID, INTER);
        L->w_gu_f = upload_gate_up_frag(m, nm, nm2, HID, INTER);
        L->w_down = upload_matrix(m, nm3, INTER, HID);
        L->w_down_f = upload_matrix_frag(m, nm3, INTER, HID);
        if (!L->w_gu || !L->w_down || !L->w_gu_f || !L->w_down_f) return -1;
        // state is allocated per slot by the engine
    }

    P->embed = upload_matrix(m, "token_embd.weight", HID, Q35_VOCAB);
    P->lm_head = P->embed;
    P->embed_f = upload_matrix_frag(m, "token_embd.weight", HID, Q35_VOCAB);
    P->final_norm = upload_f32(m, "output_norm.weight", HID);
    if (!P->embed || !P->embed_f || !P->final_norm) return -1;
    P->max_seq = m->ctx_model;
    m->prepared = 1;
    return 0;
}

static int alloc_rope(q35_engine *e, int max_seq) {
    const size_t n = (size_t)max_seq * (Q35_ROT / 2);
    float *hc = (float *)malloc(sizeof(float) * n);
    float *hs = (float *)malloc(sizeof(float) * n);
    if (!hc || !hs) { free(hc); free(hs); return -1; }
    const double base = 10000000.0;
    const int half = Q35_ROT / 2;
    for (int p = 0; p < max_seq; p++) {
        for (int i = 0; i < half; i++) {
            const double inv = pow(base, -2.0 * i / Q35_ROT);
            const double a = (double)p * inv;
            hc[(size_t)p * half + i] = (float)cos(a);
            hs[(size_t)p * half + i] = (float)sin(a);
        }
    }
    e->rope_cos = (float *)dev_alloc(e->m, sizeof(float) * n, 0);
    e->rope_sin = (float *)dev_alloc(e->m, sizeof(float) * n, 0);
    if (!e->rope_cos || !e->rope_sin) { free(hc); free(hs); return -1; }
    cudaMemcpy(e->rope_cos, hc, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(e->rope_sin, hs, sizeof(float) * n, cudaMemcpyHostToDevice);
    free(hc); free(hs);
    return 0;
}

q35_engine *q35_engine_init(q35_model *m, int ctx, int n_slots) {
    if (!m || !m->prepared || ctx <= 0 || n_slots <= 0) return NULL;
    q35_engine *e = (q35_engine *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->m = m;
    e->ctx = ctx;
    e->nslots = n_slots;
    e->epoch = 1;
    if (alloc_rope(e, ctx) != 0) { q35_engine_free(e); return NULL; }

    // One set of activation buffers; the kernel is launched for one batch at a time.
    const size_t vec = (size_t)((m->inter > m->hid ? m->inter : m->hid) > 2048 ? (m->inter > m->hid ? m->inter : m->hid) : 2048) + 64;
    e->xl = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_MAX_BATCH * m->hid, 1);
    e->projl = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_PROJ_MAX, 1);
    e->corel = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_LIN_V, 1);
    e->attnl = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_NQ * Q35_HD, 1);
    e->actl = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_MAX_BATCH * (size_t)m->inter, 1);
    e->lml = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_GRID * Q35_MAX_BATCH * 2, 1);
    e->attn_part = (float *)dev_alloc(m, sizeof(float) * Q35_MAX_BATCH * Q35_NKV * Q35_MAX_SPLITS * 4 * Q35_PART_STRIDE, 1);
    e->attn_cnt = (unsigned *)dev_alloc(m, sizeof(unsigned) * Q35_MAX_BATCH * Q35_NKV, 1);
    {
        // f16 batch-GEMV activations: [k/8][n-tile(4)][lane][2 halves] = K*32 bytes
        const size_t kb = (size_t)(m->inter > m->hid ? m->inter : m->hid);
        e->vecf = dev_alloc(m, kb * 64, 1);
    }
    (void)vec;
    if (!e->xl || !e->projl || !e->corel || !e->attnl || !e->actl || !e->lml || !e->attn_part ||
        !e->attn_cnt || !e->vecf) {
        q35_engine_free(e);
        return NULL;
    }
    e->slots = (q35_slot_t *)calloc((size_t)n_slots, sizeof(q35_slot_t));
    if (!e->slots) { q35_engine_free(e); return NULL; }
    const size_t n_conv = (size_t)Q35_CONV_DIM * Q35_CONV_RING;
    const size_t n_state = (size_t)Q35_LIN_H * Q35_LIN_DV * Q35_LIN_DK;
    const size_t n_kv = (size_t)Q35_NKV * ctx * Q35_HD;
    // All slots live in one allocation per state kind, so `slot0_layer_ptr + slot*stride`
    // is valid for every slot (the kernel cannot chase an array of pointers).
    float *conv_all = (float *)dev_alloc(m, sizeof(float) * n_conv * 18 * n_slots, 1);
    float *state_all = (float *)dev_alloc(m, sizeof(float) * n_state * 18 * n_slots, 1);
    void *k_all = dev_alloc(m, (size_t)2 * n_kv * 6 * n_slots, 1);
    void *v_all = dev_alloc(m, (size_t)2 * n_kv * 6 * n_slots, 1);
    if (!conv_all || !state_all || !k_all || !v_all) { q35_engine_free(e); return NULL; }
    for (int s = 0; s < n_slots; s++) {
        q35_slot_t *sl = &e->slots[s];
        sl->conv_hist = conv_all + (size_t)s * n_conv * 18;
        sl->state = state_all + (size_t)s * n_state * 18;
        sl->k_cache = (char *)k_all + (size_t)s * 2 * n_kv * 6;
        sl->v_cache = (char *)v_all + (size_t)s * 2 * n_kv * 6;
    }
    // forced | slot_ids | positions in one block, so a batch step uploads its inputs with one copy
    e->d_forced = (int32_t *)dev_alloc(m, sizeof(int32_t) * 3 * Q35_MAX_BATCH, 0);
    e->d_slot_ids = e->d_forced ? (int *)(e->d_forced + Q35_MAX_BATCH) : NULL;
    e->d_positions = e->d_forced ? (int *)(e->d_forced + 2 * Q35_MAX_BATCH) : NULL;
    e->d_tokens = (unsigned long long *)dev_alloc(m, sizeof(unsigned long long) * Q35_MAX_BATCH, 1);
    e->d_info = (int *)dev_alloc(m, sizeof(int) * 8, 1);
    e->logits_cap = Q35_MAX_BATCH;
    e->d_logits = (float *)dev_alloc(m, sizeof(float) * Q35_VOCAB * (size_t)e->logits_cap, 0);
    if (!e->d_forced || !e->d_slot_ids || !e->d_positions || !e->d_tokens ||
        !e->d_info || !e->d_logits) {
        q35_engine_free(e);
        return NULL;
    }
    return e;
}

void q35_engine_free(q35_engine *e) {
    if (!e) return;
    // Device buffers are owned by the model's allocation list; free the host buffers.
    free(e->slots);
    free(e);
}

void q35_engine_reset(q35_engine *e, int slot) {
    if (!e || slot < 0 || slot >= e->nslots) return;
    q35_slot_t *s = &e->slots[slot];
    cudaMemset(s->conv_hist, 0, sizeof(float) * (size_t)Q35_CONV_DIM * Q35_CONV_RING * 18);
    cudaMemset(s->state, 0, sizeof(float) * (size_t)Q35_LIN_H * Q35_LIN_DV * Q35_LIN_DK * 18);
    cudaMemset(s->k_cache, 0, (size_t)2 * (size_t)Q35_NKV * e->ctx * Q35_HD * 6);
    cudaMemset(s->v_cache, 0, (size_t)2 * (size_t)Q35_NKV * e->ctx * Q35_HD * 6);
}

static void build_params(q35_engine *e, int slot, q35_params_t *P) {
    *P = e->m->params;
    q35_slot_t *s = &e->slots[slot];
    int gd = 0, at = 0;
    for (int i = 0; i < Q35_MAX_LAYERS; i++) {
        if (!P->layer[i].is_attn) {
            P->layer[i].conv_hist = s->conv_hist + (size_t)gd * Q35_CONV_DIM * Q35_CONV_RING;
            P->layer[i].state = s->state + (size_t)gd * Q35_LIN_H * Q35_LIN_DV * Q35_LIN_DK;
            gd++;
        } else {
            P->layer[i].k_cache = (uint16_t *)s->k_cache + (size_t)at * Q35_NKV * e->ctx * Q35_HD;
            P->layer[i].v_cache = (uint16_t *)s->v_cache + (size_t)at * Q35_NKV * e->ctx * Q35_HD;
            at++;
        }
    }
    P->rope_cos = e->rope_cos;
    P->rope_sin = e->rope_sin;
    P->xl = e->xl; P->projl = e->projl; P->corel = e->corel; P->attnl = e->attnl;
    P->actl = e->actl; P->lml = e->lml; P->attn_part = e->attn_part; P->attn_cnt = e->attn_cnt;
    P->max_seq = e->ctx;
    P->hid = e->m->hid;
    P->inter = e->m->inter;
    P->nslots = e->nslots;
    P->vecf = e->vecf;
    for (int i = 0; i < Q35_MAX_LAYERS; i++) {
        if (!P->layer[i].is_attn) {
            // Per-slot stride spans all 18 GDN layers, not one layer: the engine's slot
            // arrays are [18][layer_size] and a layer pointer points inside slot 0.
            P->layer[i].conv_stride = 18LL * Q35_CONV_DIM * Q35_CONV_RING;
            P->layer[i].state_stride = 18LL * Q35_LIN_H * Q35_LIN_DV * Q35_LIN_DK;
        } else {
            P->layer[i].kv_stride = 6LL * Q35_NKV * e->ctx * Q35_HD;
        }
    }
}

static int run_steps(q35_engine *e, int slot, const int32_t *tokens, int n, int pos0,
                     int32_t *argmax, float *logits_host) {
    if (!e || slot < 0 || slot >= e->nslots || n <= 0 || n > 16) return -1;
    if (pos0 < 0 || pos0 + n > e->ctx) return -2;
    q35_params_t P;
    build_params(e, slot, &P);
    if (set_params_cached(&P) != 0) return -3;
    cudaMemcpy(e->d_forced, tokens, sizeof(int32_t) * n, cudaMemcpyHostToDevice);
    cudaMemset(e->d_tokens, 0, sizeof(unsigned long long) * n);
    cudaMemset(e->d_info, 0, sizeof(int) * 2);

    q35_args_t A;
    memset(&A, 0, sizeof(A));
    A.forced = e->d_forced;
    A.n_forced = n;
    // All logits are needed only when the caller asks for them; a plain prefill chunk
    // streams the LM head for its last position only.
    A.lm_from = logits_host ? 0 : (n - 1);
    A.n_steps = n;
    A.start_pos = pos0;
    A.epoch_base = (unsigned)e->epoch;
    A.tokens_out = e->d_tokens;
    A.logits_out = logits_host ? e->d_logits : NULL;
    A.inv_temperature = 0.f;
    A.out_info = e->d_info;
    A.hid = e->m->hid;
    A.n_stop = 0;
    static unsigned long long *prof = NULL;
    if (getenv("Q35_PROF")) {
        if (!prof) prof = (unsigned long long *)dev_alloc(e->m, sizeof(unsigned long long) * 256, 1);
        A.prof = prof;
    }

    if (getenv("Q35_DEBUG")) fprintf(stderr, "[qwen35] launch n=%d pos=%d lm_from=%d epoch=%u\n", n, pos0, A.lm_from, A.epoch_base);
    int rc = q35_mk_launch(&A, Q35_GRID, NULL);
    if (getenv("Q35_DEBUG")) fprintf(stderr, "[qwen35] launch returned %d\n", rc);
    if (rc != 0) {
        fprintf(stderr, "[qwen35] launch failed: %s\n", cudaGetErrorString((cudaError_t)rc));
        return -4;
    }
    int info[8] = {0};
    cudaMemcpy(info, e->d_info, sizeof(int) * 4, cudaMemcpyDeviceToHost);
    if (info[0] != n) {
        fprintf(stderr, "[qwen35] kernel stopped early: %d/%d steps\n", info[0], n);
        return -5;
    }
    if (getenv("Q35_PROF") && A.prof) {
        unsigned long long h[256];
        cudaMemcpy(h, A.prof, sizeof(h), cudaMemcpyDeviceToHost);
        int nops = 24 * 5 + 1;
        double sum[6] = {0};
        for (int i = 0; i < nops; i++) {
            int opi = i + 1;
            int type = (opi == nops - 1) ? 5 : opi % 5;
            sum[type] += (double)(h[i + 1] - h[i]) / 1e6;
        }
        fprintf(stderr, "[qwen35] prof ms: in=%.3f core=%.3f out=%.3f gu=%.3f down=%.3f lm=%.3f total=%.3f\n",
                sum[0], sum[1], sum[2], sum[3], sum[4], sum[5],
                (double)(h[nops] - h[0]) / 1e6);
    }
    e->epoch = info[1] & 0xffffffff;
    if (e->epoch == 0) e->epoch = 1;

    if (logits_host) {
        cudaMemcpy(logits_host, e->d_logits, sizeof(float) * (size_t)Q35_VOCAB * n, cudaMemcpyDeviceToHost);
    }
    if (argmax) {
        for (int s = 0; s < n; s++) {
            const float *lg = logits_host ? logits_host + (size_t)s * Q35_VOCAB : NULL;
            if (!lg && s == n - 1) {
                unsigned long long key = 0;
                cudaMemcpy(&key, e->d_tokens + s, sizeof(key), cudaMemcpyDeviceToHost);
                argmax[s] = (int32_t)(0xFFFFFFFFu - (uint32_t)key);
                continue;
            }
            if (!lg) { argmax[s] = -1; continue; }
            int best = 0;
            for (int i = 1; i < Q35_VOCAB; i++) if (lg[i] > lg[best]) best = i;
            argmax[s] = best;
            if (getenv("Q35_DEBUG") && s == n - 1) {
                unsigned long long key = 0;
                cudaMemcpy(&key, e->d_tokens + s, sizeof(key), cudaMemcpyDeviceToHost);
                int ktok = (int)(0xFFFFFFFFu - (uint32_t)key);
                if (ktok != best) fprintf(stderr, "[qwen35] token path mismatch: kernel=%d logits=%d\n", ktok, best);
            }
        }
    }
    return 0;
}

int q35_engine_step(q35_engine *e, int slot, int32_t token, int pos, int32_t *argmax, float *logits_host) {
    return run_steps(e, slot, &token, 1, pos, argmax, logits_host);
}

int q35_engine_steps(q35_engine *e, int slot, const int32_t *tokens, int n, int pos0,
                     int32_t *argmax, float *logits_host) {
    return run_steps(e, slot, tokens, n, pos0, argmax, logits_host);
}

int q35_engine_decode_batch(q35_engine *e, const int *slots, const int *positions,
                            const int32_t *tokens, int n, int32_t *argmax, float *logits_host) {
    if (!e || n <= 0 || n > Q35_MAX_BATCH) return -1;
    // The batch path is used at every tier, including a single active slot (G4).
    int bt = 1;
    while (bt < n) bt <<= 1;
    if (bt > 32) return -1;

    q35_params_t P;
    build_params(e, 0, &P);
    if (set_params_cached(&P) != 0) return -3;
    {
        int32_t in[3 * Q35_MAX_BATCH];
        memcpy(in, tokens, sizeof(int32_t) * n);
        memcpy(in + Q35_MAX_BATCH, slots, sizeof(int) * n);
        memcpy(in + 2 * Q35_MAX_BATCH, positions, sizeof(int) * n);
        cudaMemcpy(e->d_forced, in, sizeof(int32_t) * (2 * Q35_MAX_BATCH + n), cudaMemcpyHostToDevice);
    }
    cudaMemset(e->d_tokens, 0, sizeof(unsigned long long) * n);
    cudaMemset(e->d_info, 0, sizeof(int) * 8);

    q35_args_t A;
    memset(&A, 0, sizeof(A));
    A.forced = e->d_forced;
    A.slot_ids = e->d_slot_ids;
    A.positions = e->d_positions;
    A.n_entries = n;
    A.n_forced = n;
    A.n_steps = 1;
    A.lm_from = 0;
    A.epoch_base = (unsigned)e->epoch;
    A.tokens_out = e->d_tokens;
    A.logits_out = logits_host ? e->d_logits : NULL;
    A.out_info = e->d_info;
    A.hid = e->m->hid;
    A.n_stop = 0;
    A.prof_step = getenv("Q35_DBG_OPS") ? atoi(getenv("Q35_DBG_OPS")) : 0;
    {
        // test hook (kernel built with -DQ35_DBG_STALL): make launch number N stall one CTA
        static int stall_cnt = 0;
        const char *sl = getenv("Q35_DBG_STALL_LAUNCH");
        A.lm_from = (sl && ++stall_cnt == atoi(sl)) ? 1 : 0;
    }
    static unsigned long long *bprof = NULL;
    if (getenv("Q35_PROF")) {
        if (!bprof) bprof = (unsigned long long *)dev_alloc(e->m, sizeof(unsigned long long) * 256, 1);
        A.prof = bprof;
    }
    int rc = q35_mk_launch_batch(&A, bt, Q35_GRID, NULL);
    if (rc != 0) {
        fprintf(stderr, "[qwen35] batch launch failed: %s\n", cudaGetErrorString((cudaError_t)rc));
        return -4;
    }
    cudaError_t kerr = cudaDeviceSynchronize();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[qwen35] batch kernel error: %s\n", cudaGetErrorString(kerr));
        return -6;
    }
    if (getenv("Q35_PROF") && A.prof) {
        // CTA 0 stamps every op end: per op type, time between consecutive stamps (CTA-0 view),
        // averaged over the launches of this process; printed every 64 launches.
        static double acc[6], tot, pro[6], amix;
        static long cnt;
        unsigned long long h[256];
        cudaMemcpy(h, A.prof, sizeof(h), cudaMemcpyDeviceToHost);
        const int nops = 24 * 5 + 1;
        for (int i = 1; i <= nops; i++) {
            const int type = (i == nops) ? 5 : (i - 1) % 5;   // in, mixer, out, gate/up, down, lm
            acc[type] += (double)(h[i] - h[i - 1]) / 1e3;
            if (type == 1 && ((i - 1) / 5) % 4 == 3) amix += (double)(h[i] - h[i - 1]) / 1e3;   // attention-layer mixer
            if (type != 1) pro[type] += (double)(h[128 + i - 1] - h[i - 1]) / 1e3;   // prologue part
        }
        tot += (double)(h[nops] - h[0]) / 1e3;
        if (++cnt % 64 == 0)
            fprintf(stderr, "[qwen35] batch prof us/step (avg of %ld, n=%d): in=%.0f mixer=%.0f out=%.0f gu=%.0f down=%.0f lm=%.0f total=%.0f\n",
                    cnt, n, acc[0] / cnt, acc[1] / cnt, acc[2] / cnt, acc[3] / cnt, acc[4] / cnt, acc[5] / cnt, tot / cnt);
        if (cnt % 64 == 0)
            fprintf(stderr, "[qwen35] batch prof prologue-part us/step: in=%.0f out=%.0f gu=%.0f down=%.0f lm=%.0f (rest of each op = GEMV + epilogue + barrier)\n",
                    pro[0] / cnt, pro[2] / cnt, pro[3] / cnt, pro[4] / cnt, pro[5] / cnt);
        if (cnt % 64 == 0)
            fprintf(stderr, "[qwen35] batch prof mixer split us/step: gdn=%.0f attn=%.0f\n", (acc[1] - amix) / cnt, amix / cnt);
    }
    int info[8] = {0};
    cudaMemcpy(info, e->d_info, sizeof(int) * 8, cudaMemcpyDeviceToHost);
    if (getenv("Q35_DEBUG")) fprintf(stderr, "[qwen35] batch info done=%d wd=%d op=%d blk=%d\n", info[0], info[2], info[3], info[4]);
    {
        // A timed-out wait means the kernel consumed stale data: a hard error, never silent output.
        // The record lives in device globals (reset every launch) so it is exact even when the CTA
        // that writes out_info finished before another CTA timed out.
        int wblk = -1;
        unsigned wep = 0;
        const int wk = q35_mk_watchdog_fired(&wblk, &wep);
        if (wk != 0) {
            const long long op = (long long)(int)(wep - (unsigned)A.epoch_base);
            fprintf(stderr, "[qwen35] WATCHDOG batch: %s timed out after 100 ms (kind %d, block %d, epoch %u = op ~%lld of the launch, n=%d)\n",
                    wk == 3 ? "vecf-arrival wait" : wk == 2 ? "ll_wait" : "wait", wk, wblk, wep, op, n);
            e->epoch += 4096;   // never let stale tags from the aborted launch satisfy a later wait
            return -7;
        }
    }
    if (info[0] != 1) { fprintf(stderr, "[qwen35] batch kernel did not complete\n"); e->epoch += 4096; return -5; }
    e->epoch = info[1] & 0xffffffff;
    if (e->epoch == 0) e->epoch = 1;
    if (logits_host) {
        cudaMemcpy(logits_host, e->d_logits, sizeof(float) * (size_t)Q35_VOCAB * n, cudaMemcpyDeviceToHost);
    }
    if (getenv("Q35_DUMP")) {
        const size_t vb = (size_t)(e->m->inter > e->m->hid ? e->m->inter : e->m->hid) * 64;
        unsigned char *hv = (unsigned char *)malloc(vb);
        cudaMemcpy(hv, e->vecf, vb, cudaMemcpyDeviceToHost);
        FILE *f = fopen("/tmp/q35_vecf.bin", "wb");
        fwrite(hv, 1, vb, f); fclose(f);
        unsigned long long *hp = (unsigned long long *)malloc(sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_PROJ_MAX);
        cudaMemcpy(hp, e->projl, sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_PROJ_MAX, cudaMemcpyDeviceToHost);
        f = fopen("/tmp/q35_projl.bin", "wb");
        fwrite(hp, sizeof(unsigned long long), Q35_MAX_BATCH * Q35_PROJ_MAX, f); fclose(f);
        free(hv); free(hp);
        unsigned long long *hx = (unsigned long long *)malloc(sizeof(unsigned long long) * Q35_MAX_BATCH * e->m->hid);
        cudaMemcpy(hx, e->xl, sizeof(unsigned long long) * Q35_MAX_BATCH * e->m->hid, cudaMemcpyDeviceToHost);
        f = fopen("/tmp/q35_xl.bin", "wb");
        fwrite(hx, sizeof(unsigned long long), Q35_MAX_BATCH * e->m->hid, f); fclose(f);
        free(hx);
        unsigned long long *hc = (unsigned long long *)malloc(sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_LIN_V);
        cudaMemcpy(hc, e->corel, sizeof(unsigned long long) * Q35_MAX_BATCH * Q35_LIN_V, cudaMemcpyDeviceToHost);
        f = fopen("/tmp/q35_corel.bin", "wb");
        fwrite(hc, sizeof(unsigned long long), Q35_MAX_BATCH * Q35_LIN_V, f); fclose(f);
        free(hc);
        unsigned long long *ha = (unsigned long long *)malloc(sizeof(unsigned long long) * Q35_MAX_BATCH * e->m->inter);
        cudaMemcpy(ha, e->actl, sizeof(unsigned long long) * Q35_MAX_BATCH * e->m->inter, cudaMemcpyDeviceToHost);
        f = fopen("/tmp/q35_actl.bin", "wb");
        fwrite(ha, sizeof(unsigned long long), Q35_MAX_BATCH * e->m->inter, f); fclose(f);
        free(ha);
    }
    if (argmax) {
        unsigned long long keys[Q35_MAX_BATCH];
        cudaMemcpy(keys, e->d_tokens, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) argmax[i] = (int32_t)(0xFFFFFFFFu - (uint32_t)keys[i]);
    }
    return 0;
}
