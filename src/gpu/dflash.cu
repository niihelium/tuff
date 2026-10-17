// DFlash2 block drafter on the GPU (see include/dflash.h). Mirrors llama.cpp's DFlash2 graph:
// encoder fc + RMSNorm -> K/V injection; decoder = 6 x [RMSNorm, dynamic conv-in, q/k/v, q/k
// head-RMSNorm, NEOX RoPE (theta 1e7, 128 dims), non-causal attention over the sliding-window
// context + the block, o-proj, conv-out, residual, RMSNorm, conv-in, SwiGLU FFN, conv-out,
// residual] -> output RMSNorm -> target lm_head -> top-16 + selector lattice (host walk).
extern "C" {
#include "gguf_parser.h"
}
#include "state.h"
#include "dflash.h"
#include <vector>
#include <time.h>

#define DF_BLOCK_MAX 8
#define DF_TOPK 16

typedef struct {
    const void *q, *k, *v, *o, *gate, *up, *down, *aconv, *fconv;   // device Q8_0
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *aconv_base, *fconv_base;  // device f32
} df_layer_t;

static struct {
    int active;
    gguf_context_t *g;
    const model_t *tm;
    int n_layer, dim, n_head, n_kv, hd, ffn, window, block, rank, topk, conv_k, conv_g;
    int taps[DFLASH_MAX_TAPS], n_taps;
    float eps, theta;
    int32_t mask_id;
    df_layer_t L[8];
    const void *fc;              // device Q8_0 [2048 rows x 16384]
    const float *enc_norm, *out_norm;
    const void *sel_hidden;      // device Q8_0 [256 rows x 2048]
    const uint8_t *sel_prev, *sel_next;   // host Q8_0 rows [vocab][256]
    // state
    __half *kc, *vc;             // [n_layer][window][n_kv*hd] context ring
    int    *ring_pos;            // host: position held by each ring slot (-1 empty)
    int     ctx_hi;              // highest injected position + 1
    float  *feat;                // device [PEND_MAX][n_taps*dim]
    float  *stage;               // device [2][n_taps*dim]: the verify columns' taps (fixed for graphs)
    int     pend_pos[DFLASH_PEND_MAX];
    int     n_pend;
    // scratch
    float *x, *xn, *t1, *t2, *dyn, *qb, *kb, *vb, *att, *ffg, *ffu, *logits, *gate;
    float *kblk, *vblk;
    block_q8_1_c *xq;
    float *d_tv; int *d_ti;      // top-k results [T][16]
    float *pin_tv; int *pin_ti; float *pin_gate;
    int *d_pos; int *pin_pos;
    float *h_emb;                // pinned [T][dim]
    double ms; unsigned long long n;
} D;

// ---------------------------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------------------------
static const gguf_tensor_info_t *df_t(const char *name) {
    const gguf_tensor_info_t *t = tuff_gguf_get_tensor(D.g, name);
    if (!t) { fprintf(stderr, "[DFLASH] missing tensor %s\n", name); exit(EXIT_FAILURE); }
    return t;
}
static const void *df_up(const char *name, size_t *bytes_acc) {
    const gguf_tensor_info_t *t = df_t(name);
    void *d;
    CUDA_CHECK(cudaMalloc(&d, t->size_bytes));
    CUDA_CHECK(cudaMemcpy(d, D.g->tensor_data_blob + t->offset, t->size_bytes, cudaMemcpyHostToDevice));
    *bytes_acc += t->size_bytes;
    return d;
}

extern "C" int dflash_load(const char *path, const model_t *target) {
    if (D.active || !path || !*path) return D.active;
    D.g = tuff_gguf_parse_file(path);
    if (!D.g) { fprintf(stderr, "[DFLASH] cannot parse %s\n", path); return 0; }
    D.tm = target;
    D.n_layer = (int)tuff_gguf_get_kv_u32(D.g, "dflash.block_count", 6);
    D.dim     = (int)tuff_gguf_get_kv_u32(D.g, "dflash.embedding_length", 2048);
    D.ffn     = (int)tuff_gguf_get_kv_u32(D.g, "dflash.feed_forward_length", 6144);
    D.n_head  = (int)tuff_gguf_get_kv_u32(D.g, "dflash.attention.head_count", 32);
    D.n_kv    = (int)tuff_gguf_get_kv_u32(D.g, "dflash.attention.head_count_kv", 8);
    D.hd      = (int)tuff_gguf_get_kv_u32(D.g, "dflash.attention.key_length", 128);
    D.window  = (int)tuff_gguf_get_kv_u32(D.g, "dflash.attention.sliding_window", 2048);
    D.block   = (int)tuff_gguf_get_kv_u32(D.g, "dflash.block_size", 8);
    D.rank    = (int)tuff_gguf_get_kv_u32(D.g, "dflash.selector_rank", 256);
    D.topk    = (int)tuff_gguf_get_kv_u32(D.g, "dflash.selector_top_k", 16);
    D.conv_k  = (int)tuff_gguf_get_kv_u32(D.g, "dflash.conv_kernel_size", 2);
    D.conv_g  = (int)tuff_gguf_get_kv_u32(D.g, "dflash.conv_group_size", 16);
    D.eps     = tuff_gguf_get_kv_f32(D.g, "dflash.attention.layer_norm_rms_epsilon", 1e-6f);
    D.theta   = tuff_gguf_get_kv_f32(D.g, "dflash.rope.freq_base", 1e7f);
    D.mask_id = (int32_t)tuff_gguf_get_kv_u32(D.g, "tokenizer.ggml.mask_token_id", 248077);
    const gguf_kv_t *tl = tuff_gguf_get_kv(D.g, "dflash.target_layers");
    D.n_taps = 0;
    if (tl && tl->type == GGUF_TYPE_ARRAY) {
        for (uint64_t i = 0; i < tl->val.arr.len && D.n_taps < DFLASH_MAX_TAPS; ++i) {
            int v = (tl->val.arr.item_type == GGUF_TYPE_UINT32 || tl->val.arr.item_type == GGUF_TYPE_INT32)
                    ? ((const int32_t*)tl->val.arr.data)[i] : (int)((const int64_t*)tl->val.arr.data)[i];
            D.taps[D.n_taps++] = v;
        }
    }
    if (D.n_taps != DFLASH_MAX_TAPS || D.topk != DF_TOPK || D.block > DF_BLOCK_MAX || D.n_layer > 8 ||
        D.dim != (int)target->embedding_length || D.conv_k != 2) {
        fprintf(stderr, "[DFLASH] unsupported drafter geometry (taps %d, topk %d, block %d)\n", D.n_taps, D.topk, D.block);
        return 0;
    }
    size_t bytes = 0;
    char nm[128];
    for (int l = 0; l < D.n_layer; ++l) {
        df_layer_t *L = &D.L[l];
#define UP(field, suffix) snprintf(nm, sizeof nm, "blk.%d." suffix, l); L->field = (decltype(L->field))df_up(nm, &bytes)
        UP(q, "attn_q.weight"); UP(k, "attn_k.weight"); UP(v, "attn_v.weight"); UP(o, "attn_output.weight");
        UP(gate, "ffn_gate.weight"); UP(up, "ffn_up.weight"); UP(down, "ffn_down.weight");
        UP(aconv, "attn_conv_proj.weight"); UP(fconv, "ffn_conv_proj.weight");
        UP(attn_norm, "attn_norm.weight"); UP(ffn_norm, "ffn_norm.weight");
        UP(q_norm, "attn_q_norm.weight"); UP(k_norm, "attn_k_norm.weight");
        UP(aconv_base, "attn_conv_base"); UP(fconv_base, "ffn_conv_base");
#undef UP
    }
    D.fc = df_up("fc.weight", &bytes);
    D.enc_norm = (const float*)df_up("enc.output_norm.weight", &bytes);
    D.out_norm = (const float*)df_up("output_norm.weight", &bytes);
    D.sel_hidden = df_up("selector_hidden.weight", &bytes);
    D.sel_prev = D.g->tensor_data_blob + df_t("selector_predecessor.weight")->offset;
    D.sel_next = D.g->tensor_data_blob + df_t("selector_successor.weight")->offset;

    const int kvd = D.n_kv * D.hd, T = DF_BLOCK_MAX, V = (int)target->vocab_size;
    const size_t kv_bytes = (size_t)D.n_layer * D.window * kvd * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&D.kc, kv_bytes));
    CUDA_CHECK(cudaMalloc(&D.vc, kv_bytes));
    bytes += 2 * kv_bytes;
    D.ring_pos = (int*)malloc((size_t)D.window * sizeof(int));
    CUDA_CHECK(cudaMalloc(&D.feat, (size_t)DFLASH_PEND_MAX * D.n_taps * D.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.stage, (size_t)2 * D.n_taps * D.dim * sizeof(float)));
    bytes += (size_t)DFLASH_PEND_MAX * D.n_taps * D.dim * sizeof(float);
    const int qd = D.n_head * D.hd;
    const int big = D.ffn > qd ? D.ffn : qd;
    float **f[] = { &D.x, &D.xn, &D.t1, &D.t2, &D.att, &D.qb, &D.ffg, &D.ffu };
    for (float **p : f) { CUDA_CHECK(cudaMalloc(p, (size_t)T * (big > D.dim ? big : D.dim) * sizeof(float))); }
    CUDA_CHECK(cudaMalloc(&D.dyn, (size_t)T * 2 * 2 * (D.dim / D.conv_g) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.kb, (size_t)DFLASH_PEND_MAX * kvd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.vb, (size_t)DFLASH_PEND_MAX * kvd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.kblk, (size_t)T * kvd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.vblk, (size_t)T * kvd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.logits, (size_t)T * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.gate, (size_t)T * D.rank * sizeof(float)));
    const int in_max = D.n_taps * D.dim;
    CUDA_CHECK(cudaMalloc(&D.xq, (size_t)DFLASH_PEND_MAX * (in_max / QK8_1) * sizeof(block_q8_1_c)));
    CUDA_CHECK(cudaMalloc(&D.d_tv, (size_t)T * DF_TOPK * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.d_ti, (size_t)T * DF_TOPK * 128 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&D.d_pos, 128 * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&D.pin_tv, (size_t)T * DF_TOPK * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&D.pin_ti, (size_t)T * DF_TOPK * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&D.pin_gate, (size_t)T * D.rank * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&D.pin_pos, 128 * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&D.h_emb, (size_t)T * D.dim * sizeof(float)));
    D.active = 1;
    dflash_reset();
    fprintf(stderr, "[DFLASH] drafter %s: %d layers, block %d, taps %d..%d, %.1f MiB VRAM\n",
            path, D.n_layer, D.block, D.taps[0], D.taps[D.n_taps - 1], bytes / 1048576.0);
    return 1;
}

extern "C" int dflash_active(void) { return D.active; }
extern "C" void dflash_free(void) { D.active = 0; }
extern "C" int dflash_tap_index(uint32_t layer) {
    if (!D.active) return -1;
    for (int k = 0; k < D.n_taps; ++k) if ((uint32_t)D.taps[k] == layer) return k;
    return -1;
}
extern "C" float *dflash_tap_dst(int slot, int k) {
    return D.feat + ((size_t)slot * D.n_taps + k) * D.dim;
}
extern "C" int dflash_pending(void) { return D.n_pend; }
static void df_inject(void);
extern "C" void dflash_commit(int n, int pos0) {
    for (int j = 0; j < n && D.n_pend < DFLASH_PEND_MAX; ++j) D.pend_pos[D.n_pend++] = pos0 + j;
    if (D.n_pend >= DFLASH_PEND_MAX - 2) df_inject();
}
extern "C" float *dflash_stage_dst(int c, int k) {
    return D.stage + ((size_t)c * D.n_taps + k) * D.dim;
}
extern "C" void dflash_commit_stage(int keep, int pos0) {
    if (!D.active) return;
    const size_t row = (size_t)D.n_taps * D.dim * sizeof(float);
    for (int c = 0; c < keep && c < 2; ++c) {
        CUDA_CHECK(cudaMemcpyAsync(D.feat + (size_t)D.n_pend * D.n_taps * D.dim, D.stage + (size_t)c * D.n_taps * D.dim,
                                   row, cudaMemcpyDeviceToDevice, 0));
        dflash_commit(1, pos0 + c);
    }
}
extern "C" void dflash_reset(void) {
    if (!D.active) return;
    for (int i = 0; i < D.window; ++i) D.ring_pos[i] = -1;
    D.ctx_hi = 0;
    D.n_pend = 0;
}
extern "C" void dflash_stats(double *ms, unsigned long long *n) { *ms = D.ms; *n = D.n; }

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------
// RMSNorm of T rows of length n (one block per row).
__global__ void df_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    const float *xr = x + (size_t)blockIdx.x * n;
    float *yr = y + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[32];
    s = warp_reduce_sum_f(s);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
        t = warp_reduce_sum_f(t);
        if (threadIdx.x == 0) sh[0] = rsqrtf(t / n + eps);
    }
    __syncthreads();
    const float r = sh[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

// Per-head RMSNorm + NEOX RoPE. x: [T][nh][hd]; positions from pos[t].
__global__ void df_headnorm_rope(float *x, const float *w, const int *pos, int nh, int hd, float eps, float theta, int rope) {
    const int t = blockIdx.y, h = blockIdx.x;
    float *v = x + ((size_t)t * nh + h) * hd;
    __shared__ float sv[256];
    __shared__ float sr;
    const int i = threadIdx.x;   // hd threads
    float a = v[i];
    float s = warp_reduce_sum_f(a * a);
    __shared__ float sw[8];
    if ((i & 31) == 0) sw[i >> 5] = s;
    __syncthreads();
    if (i == 0) { float tt = 0; for (int k = 0; k < (hd >> 5); ++k) tt += sw[k]; sr = rsqrtf(tt / hd + eps); }
    __syncthreads();
    a = a * sr * w[i];
    sv[i] = a;
    __syncthreads();
    if (rope) {
        const int half = hd / 2;
        const int j = i < half ? i : i - half;
        const float freq = powf(theta, -2.0f * (float)j / (float)hd);
        const float ang = (float)pos[t] * freq;
        float c, sn; sincosf(ang, &sn, &c);
        const float x0 = sv[j], x1 = sv[j + half];
        a = i < half ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    v[i] = a;
}

// Dynamic depthwise 2-tap conv inside the block (left zero padding):
// y[t][d] = sum_tap (coef[t][g][tap][side] + base[side][tap][d]) * x[t - tap][d].
__global__ void df_conv(const float *x, const float *dyn, const float *base, float *y,
                        int T, int dim, int ng, int gs, int side) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= dim) return;
    const int g = d / gs;
    const float *co = dyn + (size_t)t * (ng * 2 * 2);
    float acc = 0.0f;
    for (int tap = 0; tap < 2 && tap <= t; ++tap) {
        const float wv = co[g + ng * (tap + 2 * side)] + base[d + (size_t)dim * (tap + 2 * side)];
        acc += wv * x[(size_t)(t - tap) * dim + d];
    }
    y[(size_t)t * dim + d] = acc;
}

__global__ void df_silu_mul(float *g, const float *u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float a = g[i]; g[i] = (a / (1.0f + expf(-a))) * u[i]; }
}
__global__ void df_add(float *y, const float *a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

// Store injected K/V rows (f32 [m][kvd]) into the f16 ring at slot pos % window.
__global__ void df_kv_store(__half *kc, __half *vc, const float *k, const float *v, const int *pos, int kvd, int window) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvd) return;
    const size_t slot = (size_t)(pos[t] % window) * kvd;
    kc[slot + i] = __float2half(k[(size_t)t * kvd + i]);
    vc[slot + i] = __float2half(v[(size_t)t * kvd + i]);
}

// Non-causal attention of the block's queries over [context ring slots (valid list) + block keys].
// grid (n_head, T), block hd threads. ctx_slots: device list of ring slots to attend (n_ctx).
__global__ void df_attn(const float *q, const __half *kc, const __half *vc, const float *kblk, const float *vblk,
                        const int *ctx_slots, int n_ctx, int T, int nh, int nkv, int hd, float scale, float *out) {
    extern __shared__ float sc[];      // n_ctx + T scores
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int kvh = h / (nh / nkv), kvd = nkv * hd;
    __shared__ float sq[256];
    sq[i] = q[((size_t)t * nh + h) * hd + i];
    __syncthreads();
    const int L = n_ctx + T;
    for (int k = i; k < L; k += blockDim.x) {
        float s = 0.0f;
        if (k < n_ctx) {
            const __half *kr = kc + (size_t)ctx_slots[k] * kvd + kvh * hd;
            for (int d = 0; d < hd; ++d) s += sq[d] * __half2float(kr[d]);
        } else {
            const float *kr = kblk + (size_t)(k - n_ctx) * kvd + kvh * hd;
            for (int d = 0; d < hd; ++d) s += sq[d] * kr[d];
        }
        sc[k] = s * scale;
    }
    __syncthreads();
    __shared__ float red[8];
    float m = -1e30f;
    for (int k = i; k < L; k += blockDim.x) m = fmaxf(m, sc[k]);
    m = warp_reduce_max_f(m);
    if ((i & 31) == 0) red[i >> 5] = m;
    __syncthreads();
    if (i == 0) { float mm = red[0]; for (int w = 1; w < (int)(blockDim.x >> 5); ++w) mm = fmaxf(mm, red[w]); red[0] = mm; }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0.0f;
    for (int k = i; k < L; k += blockDim.x) { const float e = expf(sc[k] - m); sc[k] = e; s += e; }
    s = warp_reduce_sum_f(s);
    if ((i & 31) == 0) red[i >> 5] = s;
    __syncthreads();
    if (i == 0) { float ss = 0; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) ss += red[w]; red[0] = 1.0f / ss; }
    __syncthreads();
    const float inv = red[0];
    float acc = 0.0f;
    for (int k = 0; k < L; ++k) {
        const float vv = k < n_ctx ? __half2float(vc[(size_t)ctx_slots[k] * kvd + kvh * hd + i])
                                   : vblk[(size_t)(k - n_ctx) * kvd + kvh * hd + i];
        acc += sc[k] * vv;
    }
    out[((size_t)t * nh + h) * hd + i] = acc * inv;
}

// Top-16 per row, pass 1: each block reduces a chunk of the row to its 16 largest.
__global__ void df_topk_p1(const float *x, int n, int chunk, float *ov, int *oi) {
    const int row = blockIdx.y, b = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    const int beg = b * chunk, end = min(n, beg + chunk);
    __shared__ float sv[1024]; __shared__ int si[1024];
    float bv = -1e30f; int bi = -1;
    // each thread keeps the max of its strided elements; 16 rounds of block argmax with removal
    // would need the full list, so stage the chunk in shared memory (chunk <= 1024).
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) { sv[i - beg] = xr[i]; si[i - beg] = i; }
    for (int i = end - beg + threadIdx.x; i < chunk; i += blockDim.x) { sv[i] = -1e30f; si[i] = -1; }
    __syncthreads();
    __shared__ float rv[32]; __shared__ int rp[32];
    for (int r = 0; r < DF_TOPK; ++r) {
        bv = -1e30f; int bp = -1;
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) if (sv[i] > bv) { bv = sv[i]; bp = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffff, bv, o);
            const int p2 = __shfl_xor_sync(0xffffffff, bp, o);
            if (v2 > bv || (v2 == bv && p2 < bp && p2 >= 0)) { bv = v2; bp = p2; }
        }
        if ((threadIdx.x & 31) == 0) { rv[threadIdx.x >> 5] = bv; rp[threadIdx.x >> 5] = bp; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = -1e30f; int pb = -1;
            for (int w = 0; w < (int)(blockDim.x >> 5); ++w) if (rv[w] > best) { best = rv[w]; pb = rp[w]; }
            const size_t o = ((size_t)row * gridDim.x + b) * DF_TOPK + r;
            ov[o] = best; oi[o] = pb >= 0 ? si[pb] : -1;
            if (pb >= 0) sv[pb] = -1e30f;
        }
        __syncthreads();
        (void)bi;
    }
}

// ---------------------------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------------------------
static void df_mmv(const void *w, const float *x, float *y, int in, int out, int T) {
    const int nb = in / QK8_1;
    for (int t = 0; t < T; ++t) launch_quantize_q8_1(x + (size_t)t * in, D.xq + (size_t)t * nb, in);
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int c = T - t0 < 8 ? T - t0 : 8;
        launch_mmvq_q8_0_dense_ncols((const block_q8_0*)w, D.xq + (size_t)t0 * nb, y + (size_t)t0 * out, out, in / QK8_0, c);
    }
}

static void df_inject(void) {
    if (D.n_pend == 0) return;
    const int dim = D.dim, kvd = D.n_kv * D.hd, in = D.n_taps * dim;
    for (int b0 = 0; b0 < D.n_pend; b0 += 8) {
        const int m = D.n_pend - b0 < 8 ? D.n_pend - b0 : 8;
        for (int j = 0; j < m; ++j) D.pin_pos[j] = D.pend_pos[b0 + j];
        CUDA_CHECK(cudaMemcpyAsync(D.d_pos, D.pin_pos, (size_t)m * sizeof(int), cudaMemcpyHostToDevice, 0));
        df_mmv(D.fc, D.feat + (size_t)b0 * in, D.t1, in, dim, m);
        df_rmsnorm<<<m, 256>>>(D.t1, D.enc_norm, D.xn, dim, D.eps);
        for (int l = 0; l < D.n_layer; ++l) {
            df_mmv(D.L[l].k, D.xn, D.kb, dim, kvd, m);
            df_mmv(D.L[l].v, D.xn, D.vb, dim, kvd, m);
            df_headnorm_rope<<<dim3(D.n_kv, m), D.hd>>>(D.kb, D.L[l].k_norm, D.d_pos, D.n_kv, D.hd, D.eps, D.theta, 1);
            df_kv_store<<<dim3((kvd + 255) / 256, m), 256>>>(D.kc + (size_t)l * D.window * kvd, D.vc + (size_t)l * D.window * kvd,
                                                           D.kb, D.vb, D.d_pos, kvd, D.window);
        }
        CUDA_CHECK(cudaStreamSynchronize(0));   // d_pos / pin_pos reuse
        for (int j = 0; j < m; ++j) {
            const int p = D.pend_pos[b0 + j];
            D.ring_pos[p % D.window] = p;
            if (p + 1 > D.ctx_hi) D.ctx_hi = p + 1;
        }
    }
    CUDA_CHECK_LAST();
    D.n_pend = 0;
}

static inline void deq_row_q8_0(const uint8_t *base, int32_t row, int n, float *out) {
    const block_q8_0 *b = (const block_q8_0*)base + (size_t)row * (n / QK8_0);
    for (int i = 0; i < n / QK8_0; ++i) {
        const float d = __half2float(*(const __half*)&b[i].d);
        for (int j = 0; j < QK8_0; ++j) out[i * QK8_0 + j] = d * (float)b[i].qs[j];
    }
}

static int *g_ctx_slots_d = NULL;
static int *g_ctx_slots_h = NULL;

extern "C" int dflash_draft(int32_t anchor, int pos, int k, int32_t *out) {
    if (!D.active || k <= 0) return 0;
    struct timespec ta, tb; clock_gettime(CLOCK_MONOTONIC, &ta);
    if (k > D.block - 1) k = D.block - 1;
    df_inject();
    const int T = k + 1, dim = D.dim, kvd = D.n_kv * D.hd, qd = D.n_head * D.hd;
    const int V = (int)D.tm->vocab_size, ng = dim / D.conv_g;

    // context slots within the window of the anchor position
    if (!g_ctx_slots_d) {
        CUDA_CHECK(cudaMalloc(&g_ctx_slots_d, (size_t)D.window * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&g_ctx_slots_h, (size_t)D.window * sizeof(int)));
    }
    int n_ctx = 0;
    const int lo = pos - D.window + 1 > 0 ? pos - D.window + 1 : 0;
    for (int p = lo; p < pos && p < D.ctx_hi; ++p) {
        const int s = p % D.window;
        if (D.ring_pos[s] == p) g_ctx_slots_h[n_ctx++] = s;
    }
    CUDA_CHECK(cudaMemcpyAsync(g_ctx_slots_d, g_ctx_slots_h, (size_t)n_ctx * sizeof(int), cudaMemcpyHostToDevice, 0));

    // block embeddings: anchor then masks (target token_embd, host)
    const tensor_t *te = &D.tm->token_embd;
    const size_t rb = ggml_row_size((ggml_type_t)te->info.type, dim);
    dequantize_row((ggml_type_t)te->info.type, (const uint8_t*)te->data + (size_t)anchor * rb, D.h_emb, dim);
    dequantize_row((ggml_type_t)te->info.type, (const uint8_t*)te->data + (size_t)D.mask_id * rb, D.h_emb + dim, dim);
    for (int t = 2; t < T; ++t) memcpy(D.h_emb + (size_t)t * dim, D.h_emb + dim, (size_t)dim * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(D.x, D.h_emb, (size_t)T * dim * sizeof(float), cudaMemcpyHostToDevice, 0));
    for (int t = 0; t < T; ++t) D.pin_pos[64 + t] = pos + t;
    CUDA_CHECK(cudaMemcpyAsync(D.d_pos + 64, D.pin_pos + 64, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, 0));
    const int *bpos = D.d_pos + 64;

    const int n = T * dim;
    for (int l = 0; l < D.n_layer; ++l) {
        const df_layer_t *L = &D.L[l];
        df_rmsnorm<<<T, 256>>>(D.x, L->attn_norm, D.xn, dim, D.eps);
        df_mmv(L->aconv, D.xn, D.dyn, dim, 2 * 2 * ng, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.xn, D.dyn, L->aconv_base, D.t1, T, dim, ng, D.conv_g, 0);
        df_mmv(L->q, D.t1, D.qb, dim, qd, T);
        df_mmv(L->k, D.t1, D.kblk, dim, kvd, T);
        df_mmv(L->v, D.t1, D.vblk, dim, kvd, T);
        df_headnorm_rope<<<dim3(D.n_head, T), D.hd>>>(D.qb, L->q_norm, bpos, D.n_head, D.hd, D.eps, D.theta, 1);
        df_headnorm_rope<<<dim3(D.n_kv, T), D.hd>>>(D.kblk, L->k_norm, bpos, D.n_kv, D.hd, D.eps, D.theta, 1);
        df_attn<<<dim3(D.n_head, T), D.hd, (size_t)(n_ctx + T) * sizeof(float)>>>(
            D.qb, D.kc + (size_t)l * D.window * kvd, D.vc + (size_t)l * D.window * kvd, D.kblk, D.vblk,
            g_ctx_slots_d, n_ctx, T, D.n_head, D.n_kv, D.hd, 1.0f / sqrtf((float)D.hd), D.att);
        df_mmv(L->o, D.att, D.t2, qd, dim, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.t2, D.dyn, L->aconv_base, D.t1, T, dim, ng, D.conv_g, 1);
        df_add<<<(n + 255) / 256, 256>>>(D.x, D.t1, n);                       // x = ffn_inp
        df_rmsnorm<<<T, 256>>>(D.x, L->ffn_norm, D.xn, dim, D.eps);
        df_mmv(L->fconv, D.xn, D.dyn, dim, 2 * 2 * ng, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.xn, D.dyn, L->fconv_base, D.t1, T, dim, ng, D.conv_g, 0);
        df_mmv(L->gate, D.t1, D.ffg, dim, D.ffn, T);
        df_mmv(L->up, D.t1, D.ffu, dim, D.ffn, T);
        df_silu_mul<<<(T * D.ffn + 255) / 256, 256>>>(D.ffg, D.ffu, T * D.ffn);
        df_mmv(L->down, D.ffg, D.t2, D.ffn, dim, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.t2, D.dyn, L->fconv_base, D.t1, T, dim, ng, D.conv_g, 1);
        df_add<<<(n + 255) / 256, 256>>>(D.x, D.t1, n);
    }
    df_rmsnorm<<<T, 256>>>(D.x, D.out_norm, D.xn, dim, D.eps);           // t_embd
    // lm head (target Q6_K) on positions 1..k, and the selector gate
    const int nb = dim / QK8_1;
    for (int t = 1; t < T; ++t) launch_quantize_q8_1(D.xn + (size_t)t * dim, D.xq + (size_t)(t - 1) * nb, dim);
    launch_mmvq_q6_K_dense_ncols((const block_q6_K*)D.tm->output_weight.dev, D.xq, D.logits, V, dim / QK_K, k);
    launch_mmvq_q8_0_dense_ncols((const block_q8_0*)D.sel_hidden, D.xq, D.gate, D.rank, dim / QK8_0, k);
    const int chunk = 1024;
    const int nchunks = (V + chunk - 1) / chunk;
    df_topk_p1<<<dim3(nchunks, k), 256>>>(D.logits, V, chunk, D.d_tv, D.d_ti);
    CUDA_CHECK_LAST();
    std::vector<float> hv((size_t)k * nchunks * DF_TOPK);
    std::vector<int> hi((size_t)k * nchunks * DF_TOPK);
    CUDA_CHECK(cudaMemcpyAsync(hv.data(), D.d_tv, hv.size() * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(hi.data(), D.d_ti, hi.size() * sizeof(int), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(D.pin_gate, D.gate, (size_t)k * D.rank * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));

    // host: merge chunk top-16 -> top-16 per position, then the lattice walk
    const int R = D.rank, K = DF_TOPK;
    int cand[DF_BLOCK_MAX][DF_TOPK]; float un[DF_BLOCK_MAX][DF_TOPK];
    for (int t = 0; t < k; ++t) {
        float *v = hv.data() + (size_t)t * nchunks * K; int *ix = hi.data() + (size_t)t * nchunks * K;
        for (int r = 0; r < K; ++r) {
            int best = -1;
            for (int j = 0; j < nchunks * K; ++j) if (ix[j] >= 0 && (best < 0 || v[j] > v[best])) best = j;
            cand[t][r] = best >= 0 ? ix[best] : 0; un[t][r] = best >= 0 ? v[best] : -1e30f;
            if (best >= 0) ix[best] = -1;
        }
    }
    std::vector<float> succ((size_t)K * R), prev((size_t)K * R);
    int pred = 0, n_out = 0;
    int prev_ids[DF_TOPK]; int n_prev = 1; prev_ids[0] = anchor;
    for (int t = 0; t < k; ++t) {
        const float *g = D.pin_gate + (size_t)t * R;
        for (int j = 0; j < K; ++j) deq_row_q8_0(D.sel_next, cand[t][j], R, succ.data() + (size_t)j * R);
        // only the chosen predecessor row matters for the walk
        const int pid = n_prev == 1 ? prev_ids[0] : prev_ids[pred];
        deq_row_q8_0(D.sel_prev, pid, R, prev.data());
        int bj = 0; float bs = -1e30f;
        for (int j = 0; j < K; ++j) {
            const float *s = succ.data() + (size_t)j * R;
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) acc += s[r] * prev[r] * g[r];
            acc += un[t][j];
            if (acc > bs) { bs = acc; bj = j; }
        }
        pred = bj;
        out[n_out++] = cand[t][bj];
        for (int j = 0; j < K; ++j) prev_ids[j] = cand[t][j];
        n_prev = K;
    }
    clock_gettime(CLOCK_MONOTONIC, &tb);
    D.ms += 1e3 * ((tb.tv_sec - ta.tv_sec) + 1e-9 * (tb.tv_nsec - ta.tv_nsec));
    D.n++;
    return n_out;
}
