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
typedef struct {
    __half2 dm;
    uint8_t qs[16];
} block_df_q4;

static int df_use_q4(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TUFF_DFLASH_Q4"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

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
static const void *df_up_w(const char *name, size_t *bytes_acc) {
    if (!df_use_q4()) return df_up(name, bytes_acc);
    const gguf_tensor_info_t *t = df_t(name);
    const size_t nblk = t->size_bytes / sizeof(block_q8_0);
    const block_q8_0 *src = (const block_q8_0*)(D.g->tensor_data_blob + t->offset);
    const size_t q4_bytes = nblk * sizeof(block_df_q4);
    block_df_q4 *h = (block_df_q4*)malloc(q4_bytes);
    for (size_t b = 0; b < nblk; ++b) {
        const float d8 = __half2float(*(const __half*)&src[b].d);
        int qmin = 127, qmax = -128;
        for (int i = 0; i < 32; ++i) {
            int q = (int)src[b].qs[i];
            if (q < qmin) qmin = q;
            if (q > qmax) qmax = q;
        }
        const float range = (float)(qmax - qmin);
        const float d4 = d8 * (range / 15.0f);
        const float m4 = d8 * (float)qmin;
        h[b].dm = __floats2half2_rn(d4, m4);
        const float id = (range > 0.0f) ? (15.0f / range) : 0.0f;
        for (int j = 0; j < 16; ++j) {
            int q0 = (int)roundf((float)((int)src[b].qs[j] - qmin) * id);
            int q1 = (int)roundf((float)((int)src[b].qs[j + 16] - qmin) * id);
            if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
            h[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
    void *d;
    CUDA_CHECK(cudaMalloc(&d, q4_bytes));
    CUDA_CHECK(cudaMemcpy(d, h, q4_bytes, cudaMemcpyHostToDevice));
    free(h);
    *bytes_acc += q4_bytes;
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
#define UPW(field, suffix) snprintf(nm, sizeof nm, "blk.%d." suffix, l); L->field = (decltype(L->field))df_up_w(nm, &bytes)
#define UP(field, suffix)  snprintf(nm, sizeof nm, "blk.%d." suffix, l); L->field = (decltype(L->field))df_up(nm, &bytes)
        UPW(q, "attn_q.weight"); UPW(k, "attn_k.weight"); UPW(v, "attn_v.weight"); UPW(o, "attn_output.weight");
        UPW(gate, "ffn_gate.weight"); UPW(up, "ffn_up.weight"); UPW(down, "ffn_down.weight");
        UP(aconv, "attn_conv_proj.weight"); UP(fconv, "ffn_conv_proj.weight");
        UP(attn_norm, "attn_norm.weight"); UP(ffn_norm, "ffn_norm.weight");
        UP(q_norm, "attn_q_norm.weight"); UP(k_norm, "attn_k_norm.weight");
        UP(aconv_base, "attn_conv_base"); UP(fconv_base, "ffn_conv_base");
#undef UPW
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
    CUDA_CHECK(cudaMalloc(&D.stage, (size_t)4 * D.n_taps * D.dim * sizeof(float)));
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
    const int nchunks = (V + 1023) / 1024;
    CUDA_CHECK(cudaMalloc(&D.d_tv, (size_t)T * DF_TOPK * (nchunks + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.d_ti, (size_t)T * DF_TOPK * (nchunks + 1) * sizeof(int)));
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
    if (D.n_pend >= DFLASH_PEND_MAX - 4) df_inject();
}
extern "C" float *dflash_stage_dst(int c, int k) {
    return D.stage + ((size_t)c * D.n_taps + k) * D.dim;
}
extern "C" void dflash_commit_stage(int keep, int pos0) {
    if (!D.active) return;
    const size_t row = (size_t)D.n_taps * D.dim * sizeof(float);
    for (int c = 0; c < keep && c < 4; ++c) {
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

// Fused SwiGLU + Q8_1 quantization: g = silu(g) * u, then warp-quantize 32 elements into xq.
__global__ void df_silu_mul_quant(const float * __restrict__ g, const float * __restrict__ u,
                                  block_q8_1_c * __restrict__ xq, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = 0.0f;
    if (i < n) {
        const float a = g[i];
        xi = (a / (1.0f + expf(-a))) * u[i];
    }
    float amax = fabsf(xi), sum = xi;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum  += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    if (i >= n) return;
    xq[i / QK8_1].qs[i % QK8_1] = q;
    if ((i & 31) == 0) xq[i / QK8_1].ds = make_half2(__float2half(d), __float2half(sum));
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

// Non-causal GQA attention: one block per KV head per query position (grid: n_kv, T; block: 128 threads).
// Each block loads the KV head's context and block K/V once and computes all G = nh/nkv query heads.
// Reads n_ctx from device pointer *d_n_ctx so the kernel can live inside a fixed CUDA graph per T.
__global__ void df_attn_gqa(const float *q, const __half *kc, const __half *vc, const float *kblk, const float *vblk,
                            const int *ctx_slots, const int *d_n_ctx, int T, int nh, int nkv, int hd, float scale, float *out) {
    extern __shared__ float smem[];    // G * L scores, where G = nh / nkv <= 4
    const int kvh = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int G = nh / nkv, kvd = nkv * hd, h0 = kvh * G;
    const int n_ctx = *d_n_ctx;
    const int L = n_ctx + T;
    __shared__ float sq[4][128];
    for (int g = 0; g < G; ++g) sq[g][i] = q[((size_t)t * nh + h0 + g) * hd + i];
    __syncthreads();

    // 4 warps (32 lanes each): each warp processes one key k per step; 32 lanes reduce hd=128 (4 elements/lane).
    const int warp = i >> 5, lane = i & 31, nwarps = blockDim.x >> 5;
    for (int k = warp; k < L; k += nwarps) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        if (k < n_ctx) {
            const half2 *kr2 = reinterpret_cast<const half2*>(kc + (size_t)ctx_slots[k] * kvd + kvh * hd);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int d2 = lane + j * 32;
                const float2 kv = __half22float2(kr2[d2]);
                const int d = d2 * 2;
                s0 += sq[0][d] * kv.x + sq[0][d + 1] * kv.y;
                if (G > 1) s1 += sq[1][d] * kv.x + sq[1][d + 1] * kv.y;
                if (G > 2) s2 += sq[2][d] * kv.x + sq[2][d + 1] * kv.y;
                if (G > 3) s3 += sq[3][d] * kv.x + sq[3][d + 1] * kv.y;
            }
        } else {
            const float2 *kr2 = reinterpret_cast<const float2*>(kblk + (size_t)(k - n_ctx) * kvd + kvh * hd);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int d2 = lane + j * 32;
                const float2 kv = kr2[d2];
                const int d = d2 * 2;
                s0 += sq[0][d] * kv.x + sq[0][d + 1] * kv.y;
                if (G > 1) s1 += sq[1][d] * kv.x + sq[1][d + 1] * kv.y;
                if (G > 2) s2 += sq[2][d] * kv.x + sq[2][d + 1] * kv.y;
                if (G > 3) s3 += sq[3][d] * kv.x + sq[3][d + 1] * kv.y;
            }
        }
        s0 = warp_reduce_sum_f(s0);
        if (G > 1) s1 = warp_reduce_sum_f(s1);
        if (G > 2) s2 = warp_reduce_sum_f(s2);
        if (G > 3) s3 = warp_reduce_sum_f(s3);
        if (lane == 0) {
            smem[0 * L + k] = s0 * scale;
            if (G > 1) smem[1 * L + k] = s1 * scale;
            if (G > 2) smem[2 * L + k] = s2 * scale;
            if (G > 3) smem[3 * L + k] = s3 * scale;
        }
    }
    __syncthreads();

    // Softmax per query head g: warp g handles query head g (since G=4 <= nwarps=4).
    if (warp < G) {
        float *sc = smem + warp * L;
        float m = -1e30f;
        for (int k = lane; k < L; k += 32) m = fmaxf(m, sc[k]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        float s = 0.0f;
        for (int k = lane; k < L; k += 32) { const float e = expf(sc[k] - m); sc[k] = e; s += e; }
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        const float inv = 1.0f / s;
        for (int k = lane; k < L; k += 32) sc[k] *= inv;
    }
    __syncthreads();

    // Value accumulation: thread i (0..hd-1) reads V[k][i] once and accumulates for all G query heads.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k = 0; k < L; ++k) {
        const float vv = k < n_ctx ? __half2float(vc[(size_t)ctx_slots[k] * kvd + kvh * hd + i])
                                   : vblk[(size_t)(k - n_ctx) * kvd + kvh * hd + i];
        acc0 += smem[0 * L + k] * vv;
        if (G > 1) acc1 += smem[1 * L + k] * vv;
        if (G > 2) acc2 += smem[2 * L + k] * vv;
        if (G > 3) acc3 += smem[3 * L + k] * vv;
    }
    out[((size_t)t * nh + h0 + 0) * hd + i] = acc0;
    if (G > 1) out[((size_t)t * nh + h0 + 1) * hd + i] = acc1;
    if (G > 2) out[((size_t)t * nh + h0 + 2) * hd + i] = acc2;
    if (G > 3) out[((size_t)t * nh + h0 + 3) * hd + i] = acc3;
}

// Top-16 per row, pass 1: each block reduces a chunk of the row to its 16 largest.
__global__ void df_topk_p1(const float *x, int n, int chunk, float *ov, int *oi) {
    const int row = blockIdx.y, b = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    const int beg = b * chunk, end = min(n, beg + chunk);
    __shared__ float sv[1024]; __shared__ int si[1024];
    float bv = -1e30f;
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
    }
}

// Top-16 per row, pass 2: one block of 256 threads per row merges `nchunks * DF_TOPK` candidates to 16.
__global__ void df_topk_p2(const float *iv, const int *ii, int ncand, float *ov, int *oi) {
    const int row = blockIdx.x;
    const float *vr = iv + (size_t)row * ncand;
    const int   *ir = ii + (size_t)row * ncand;
    __shared__ float sv[4096]; __shared__ int si[4096];
    for (int i = threadIdx.x; i < ncand; i += blockDim.x) { sv[i] = vr[i]; si[i] = ir[i]; }
    __syncthreads();
    __shared__ float rv[8]; __shared__ int rp[8];
    for (int r = 0; r < DF_TOPK; ++r) {
        float bv = -1e30f; int bp = -1;
        for (int i = threadIdx.x; i < ncand; i += blockDim.x) {
            if (si[i] >= 0 && (sv[i] > bv || (sv[i] == bv && (bp < 0 || i < bp)))) { bv = sv[i]; bp = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffff, bv, o);
            const int p2 = __shfl_xor_sync(0xffffffff, bp, o);
            if (p2 >= 0 && (v2 > bv || (v2 == bv && (bp < 0 || p2 < bp)))) { bv = v2; bp = p2; }
        }
        if ((threadIdx.x & 31) == 0) { rv[threadIdx.x >> 5] = bv; rp[threadIdx.x >> 5] = bp; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = -1e30f; int pb = -1;
            for (int w = 0; w < (int)(blockDim.x >> 5); ++w) {
                if (rp[w] >= 0 && (rv[w] > best || (rv[w] == best && (pb < 0 || rp[w] < pb)))) { best = rv[w]; pb = rp[w]; }
            }
            const size_t o = (size_t)row * DF_TOPK + r;
            ov[o] = pb >= 0 ? best : -1e30f;
            oi[o] = pb >= 0 ? si[pb] : 0;
            if (pb >= 0) si[pb] = -1;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------------------------
template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_df_q4_dense_ncols_kernel(
        const block_df_q4 * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk) {
    constexpr int tpsb = 4;
    constexpr int blocks_per_iter = 8;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_df_q4 * __restrict__ rw = w + (size_t)row * nblk;
    const int iqs = threadIdx.x % tpsb;
    float tmp[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) tmp[c] = 0.0f;

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_df_q4 *bw = rw + kbx;
        const int v  = get_int_b2_c(bw->qs, iqs);
        const int vl = v & 0x0F0F0F0F;
        const int vh = (v >> 4) & 0x0F0F0F0F;
        const float2 dm = __half22float2(bw->dm);
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const block_q8_1_c *by = y + (size_t)c * nblk + kbx;
            const int u0 = get_int_b4_c(by->qs, iqs);
            const int u1 = get_int_b4_c(by->qs, iqs + 4);
            const int sumi = __dp4a(vl, u0, __dp4a(vh, u1, 0));
            tmp[c] += dm.x * __low2float(by->ds) * (float)sumi + (iqs == 0 ? dm.y * __high2float(by->ds) : 0.0f);
        }
    }
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float s = warp_reduce_sum_f(tmp[c]);
        if (threadIdx.x == 0) dst[(size_t)c * out_dim + row] = s;
    }
}

static void launch_mmvq_df_w(const void *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols) {
    if (!df_use_q4()) {
        launch_mmvq_q8_0_dense_ncols((const block_q8_0*)w, xq, dst, out_dim, nblk, ncols);
        return;
    }
    const block_df_q4 *wq = (const block_df_q4*)w;
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    switch (ncols) {
        case 1: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 2: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 3: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 3><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 4: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 5: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 5><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 6: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 6><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 7: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 7><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        case 8: mmvq_df_q4_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 8><<<grid, block>>>(wq, xq, dst, out_dim, nblk); break;
        default: break;
    }
    CUDA_CHECK_LAST();
}

static void df_mmv(const void *w, const float *x, float *y, int in, int out, int T) {
    const int nb = in / QK8_1;
    launch_quantize_q8_1(x, D.xq, T * in);
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int c = T - t0 < 8 ? T - t0 : 8;
        launch_mmvq_df_w(w, D.xq + (size_t)t0 * nb, y + (size_t)t0 * out, out, in / QK8_0, c);
    }
}
static void df_mmv_q8(const void *w, const float *x, float *y, int in, int out, int T) {
    const int nb = in / QK8_1;
    launch_quantize_q8_1(x, D.xq, T * in);
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int c = T - t0 < 8 ? T - t0 : 8;
        launch_mmvq_q8_0_dense_ncols((const block_q8_0*)w, D.xq + (size_t)t0 * nb, y + (size_t)t0 * out, out, in / QK8_0, c);
    }
}
static void df_mmv_reuse(const void *w, float *y, int in, int out, int T) {
    const int nb = in / QK8_1;
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int c = T - t0 < 8 ? T - t0 : 8;
        launch_mmvq_df_w(w, D.xq + (size_t)t0 * nb, y + (size_t)t0 * out, out, in / QK8_0, c);
    }
}

static void df_inject(void) {
    if (D.n_pend == 0) return;
    const int dim = D.dim, kvd = D.n_kv * D.hd, in = D.n_taps * dim;
    for (int b0 = 0; b0 < D.n_pend; b0 += 8) {
        const int m = D.n_pend - b0 < 8 ? D.n_pend - b0 : 8;
        for (int j = 0; j < m; ++j) D.pin_pos[j] = D.pend_pos[b0 + j];
        CUDA_CHECK(cudaMemcpyAsync(D.d_pos, D.pin_pos, (size_t)m * sizeof(int), cudaMemcpyHostToDevice, 0));
        df_mmv_q8(D.fc, D.feat + (size_t)b0 * in, D.t1, in, dim, m);
        df_rmsnorm<<<m, 256>>>(D.t1, D.enc_norm, D.xn, dim, D.eps);
        launch_quantize_q8_1(D.xn, D.xq, m * dim);
        for (int l = 0; l < D.n_layer; ++l) {
            df_mmv_reuse(D.L[l].k, D.kb, dim, kvd, m);
            df_mmv_reuse(D.L[l].v, D.vb, dim, kvd, m);
            df_headnorm_rope<<<dim3(D.n_kv, m), D.hd>>>(D.kb, D.L[l].k_norm, D.d_pos, D.n_kv, D.hd, D.eps, D.theta, 1);
            df_kv_store<<<dim3((kvd + 255) / 256, m), 256>>>(D.kc + (size_t)l * D.window * kvd, D.vc + (size_t)l * D.window * kvd,
                                                           D.kb, D.vb, D.d_pos, kvd, D.window);
        }
        if (b0 + 8 < D.n_pend) CUDA_CHECK(cudaStreamSynchronize(0));   // d_pos / pin_pos reuse across chunks
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
static cudaGraphExec_t g_df_exec[DF_BLOCK_MAX + 1] = {0};

static void df_graph_body(int k) {
    const int T = k + 1, dim = D.dim, kvd = D.n_kv * D.hd, qd = D.n_head * D.hd;
    const int V = (int)D.tm->vocab_size, ng = dim / D.conv_g;
    const int *bpos = D.d_pos + 64;
    const int *d_n_ctx = D.d_pos + 96;
    const int n = T * dim;
    const int G = D.n_head / D.n_kv;
    const size_t max_smem = (size_t)G * (D.window + T) * sizeof(float);
    for (int l = 0; l < D.n_layer; ++l) {
        const df_layer_t *L = &D.L[l];
        df_rmsnorm<<<T, 256>>>(D.x, L->attn_norm, D.xn, dim, D.eps);
        df_mmv_q8(L->aconv, D.xn, D.dyn, dim, 2 * 2 * ng, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.xn, D.dyn, L->aconv_base, D.t1, T, dim, ng, D.conv_g, 0);
        launch_quantize_q8_1(D.t1, D.xq, T * dim);
        df_mmv_reuse(L->q, D.qb, dim, qd, T);
        df_mmv_reuse(L->k, D.kblk, dim, kvd, T);
        df_mmv_reuse(L->v, D.vblk, dim, kvd, T);
        df_headnorm_rope<<<dim3(D.n_head, T), D.hd>>>(D.qb, L->q_norm, bpos, D.n_head, D.hd, D.eps, D.theta, 1);
        df_headnorm_rope<<<dim3(D.n_kv, T), D.hd>>>(D.kblk, L->k_norm, bpos, D.n_kv, D.hd, D.eps, D.theta, 1);
        df_attn_gqa<<<dim3(D.n_kv, T), D.hd, max_smem>>>(
            D.qb, D.kc + (size_t)l * D.window * kvd, D.vc + (size_t)l * D.window * kvd, D.kblk, D.vblk,
            g_ctx_slots_d, d_n_ctx, T, D.n_head, D.n_kv, D.hd, 1.0f / sqrtf((float)D.hd), D.att);
        df_mmv(L->o, D.att, D.t2, qd, dim, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.t2, D.dyn, L->aconv_base, D.t1, T, dim, ng, D.conv_g, 1);
        df_add<<<(n + 255) / 256, 256>>>(D.x, D.t1, n);                       // x = ffn_inp
        df_rmsnorm<<<T, 256>>>(D.x, L->ffn_norm, D.xn, dim, D.eps);
        df_mmv_q8(L->fconv, D.xn, D.dyn, dim, 2 * 2 * ng, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.xn, D.dyn, L->fconv_base, D.t1, T, dim, ng, D.conv_g, 0);
        launch_quantize_q8_1(D.t1, D.xq, T * dim);
        df_mmv_reuse(L->gate, D.ffg, dim, D.ffn, T);
        df_mmv_reuse(L->up, D.ffu, dim, D.ffn, T);
        df_silu_mul_quant<<<(T * D.ffn + 255) / 256, 256>>>(D.ffg, D.ffu, D.xq, T * D.ffn);
        df_mmv_reuse(L->down, D.t2, D.ffn, dim, T);
        df_conv<<<dim3((dim + 255) / 256, T), 256>>>(D.t2, D.dyn, L->fconv_base, D.t1, T, dim, ng, D.conv_g, 1);
        df_add<<<(n + 255) / 256, 256>>>(D.x, D.t1, n);
    }
    df_rmsnorm<<<T, 256>>>(D.x, D.out_norm, D.xn, dim, D.eps);           // t_embd
    launch_quantize_q8_1(D.xn + dim, D.xq, k * dim);
    launch_mmvq_q6_K_dense_ncols((const block_q6_K*)D.tm->output_weight.dev, D.xq, D.logits, V, dim / QK_K, k);
    launch_mmvq_q8_0_dense_ncols((const block_q8_0*)D.sel_hidden, D.xq, D.gate, D.rank, dim / QK8_0, k);
    const int chunk = 1024;
    const int nchunks = (V + chunk - 1) / chunk;
    df_topk_p1<<<dim3(nchunks, k), 256>>>(D.logits, V, chunk, D.d_tv, D.d_ti);
    float *d_tv2 = D.d_tv + (size_t)DF_BLOCK_MAX * nchunks * DF_TOPK;
    int   *d_ti2 = D.d_ti + (size_t)DF_BLOCK_MAX * nchunks * DF_TOPK;
    df_topk_p2<<<k, 256>>>(D.d_tv, D.d_ti, nchunks * DF_TOPK, d_tv2, d_ti2);
    CUDA_CHECK(cudaMemcpyAsync(D.pin_tv, d_tv2, (size_t)k * DF_TOPK * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(D.pin_ti, d_ti2, (size_t)k * DF_TOPK * sizeof(int), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(D.pin_gate, D.gate, (size_t)k * D.rank * sizeof(float), cudaMemcpyDeviceToHost, 0));
}

extern "C" int dflash_draft(int32_t anchor, int pos, int k, int32_t *out) {
    if (!D.active || k <= 0) return 0;
    CUDA_CHECK(cudaStreamSynchronize(0));
    struct timespec ta, tb;
    clock_gettime(CLOCK_MONOTONIC, &ta);
    if (k > D.block - 1) k = D.block - 1;
    df_inject();
    const int T = k + 1, dim = D.dim;

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
    D.pin_pos[96] = n_ctx;
    CUDA_CHECK(cudaMemcpyAsync(D.d_pos + 64, D.pin_pos + 64, 33 * sizeof(int), cudaMemcpyHostToDevice, 0));

    if (!g_df_exec[k]) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
        df_graph_body(k);
        CUDA_CHECK(cudaStreamEndCapture(0, &g));
        CUDA_CHECK(cudaGraphInstantiate(&g_df_exec[k], g, NULL, NULL, 0));
        CUDA_CHECK(cudaGraphDestroy(g));
    }
    CUDA_CHECK(cudaGraphLaunch(g_df_exec[k], 0));
    CUDA_CHECK(cudaStreamSynchronize(0));

    // host: the lattice walk over the GPU-merged top-16 per position
    const int R = D.rank, K = DF_TOPK;
    int cand[DF_BLOCK_MAX][DF_TOPK]; float un[DF_BLOCK_MAX][DF_TOPK];
    for (int t = 0; t < k; ++t) {
        for (int r = 0; r < K; ++r) {
            cand[t][r] = D.pin_ti[t * K + r];
            un[t][r]   = D.pin_tv[t * K + r];
        }
    }
    float succ[DF_TOPK * 256], prev[256];
    int pred = 0, n_out = 0;
    int prev_ids[DF_TOPK]; int n_prev = 1; prev_ids[0] = anchor;
    for (int t = 0; t < k; ++t) {
        const float *g = D.pin_gate + (size_t)t * R;
        for (int j = 0; j < K; ++j) deq_row_q8_0(D.sel_next, cand[t][j], R, succ + (size_t)j * R);
        // only the chosen predecessor row matters for the walk
        const int pid = n_prev == 1 ? prev_ids[0] : prev_ids[pred];
        deq_row_q8_0(D.sel_prev, pid, R, prev);
        int bj = 0; float bs = -1e30f;
        for (int j = 0; j < K; ++j) {
            const float *s = succ + (size_t)j * R;
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
