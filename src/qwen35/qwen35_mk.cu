// Qwen3.5 0.8B / 2B decode megakernel for sm_86 (RTX 3070 Laptop, 40 SMs).
//
// Ported from the sm_120 batch-1 megakernel in
// /home/nii/src/megakernels/qwen3.5-0.8b/csrc/megakernel.cu (MIT, Samuel Reeder).
// The port replaces the TMA bulk-copy ring (sm_90+) with an sm_86 cp.async ring whose
// completion is signalled through an mbarrier (cp.async.mbarrier.arrive.noinc), keeps the
// flag-in-data dataflow, and reads the GGUF's exact fp16 Q8_0 scales instead of the
// example's bf16-rounded ones.  Every norm weight is used as stored in the GGUF: llama.cpp's
// converter already applies the HF `1 + w` offset to every *.norm.weight except
// linear_attn.norm (ssm_norm), so the kernel must NOT add one again.
//
// One cooperative CTA per SM, 17 warps: 16 consumer warps + 1 producer warp.
// All shape differences (0.8B vs 2B) are template parameters; the host dispatches once.
#include <cooperative_groups.h>
#include "qwen35_kernel.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace q35 {

__constant__ q35_params_t P;
// Per-launch batch-path state, one struct so the host resets it (and reads the watchdog record) with a
// single copy each.
struct BState {
    int wd_b, wd_b_blk;   // watchdog kind / CTA of the first timed-out wait (sticky until reset)
    unsigned wd_b_ep;     // epoch / counter target the timed-out wait was expecting
    unsigned vready;      // batch path (sliced prologues): CTAs that have written their vecf slice
    unsigned vdone;       // batch path: monotonic count of CTA "finished reading vecf" arrivals
    unsigned pad[3];
};
__device__ BState g_bs = {0, -1, 0, 0, 0, {0, 0, 0}};
#define g_wd_b (q35::g_bs.wd_b)
#define g_wd_b_blk (q35::g_bs.wd_b_blk)
#define g_wd_b_ep (q35::g_bs.wd_b_ep)
#define g_vready (q35::g_bs.vready)
#define g_vdone (q35::g_bs.vdone)
__device__ int g_wd_b_op = -1;
__device__ unsigned long long g_ssl[32 * 40];   // sliced rmsnorm: per-CTA partial sum of squares [entry][cta]

// ------------------------------------------------------------------ shape
constexpr int GRID = 40;              // RTX 3070 Laptop SM count (one CTA per SM)
constexpr int N_CONSUMER_WARPS = 16;
constexpr int N_CONSUMERS = N_CONSUMER_WARPS * 32;
constexpr int N_THREADS = N_CONSUMERS + 32;
constexpr int N_LAYERS = Q35_MAX_LAYERS;
constexpr int VOCAB = Q35_VOCAB;
constexpr int LIN_H = Q35_LIN_H, LIN_DK = Q35_LIN_DK, LIN_DV = Q35_LIN_DV;
constexpr int LIN_QK = Q35_LIN_QK, LIN_V = Q35_LIN_V;
constexpr int CONV_DIM = Q35_CONV_DIM, CONV_K = Q35_CONV_K, CONV_RING = Q35_CONV_RING;
constexpr int GDN_IN_ROWS = Q35_GDN_IN_ROWS;
constexpr int GDN_Z_OFF = CONV_DIM;
constexpr int GDN_B_OFF = CONV_DIM + LIN_V;
constexpr int GDN_A_OFF = GDN_B_OFF + LIN_H;
constexpr int GDN_SLICE = 32;
constexpr int PROJ_MAX = Q35_PROJ_MAX;
constexpr int NQ = Q35_NQ, NKV = Q35_NKV, HD = Q35_HD, ROT = Q35_ROT;
constexpr int Q_PER_KV = NQ / NKV;
constexpr int ATT_K_OFF = NQ * 2 * HD;
constexpr int ATT_V_OFF = ATT_K_OFF + NKV * HD;
constexpr int ATT_IN_ROWS = Q35_ATT_IN_ROWS;
constexpr int MAX_SPLITS = Q35_MAX_SPLITS;
constexpr int PART_STRIDE = Q35_PART_STRIDE;
constexpr int Q8_GROUP = 32;
constexpr int STAGES = 4;
constexpr int CHUNK = 16384;
constexpr int GROUP_BYTES = 544;   // one 16-row x k32 GEMV fragment tile

__host__ __device__ constexpr int row_bytes(int K) { return K + K / Q8_GROUP * 2; }

template <int HID, int INTER>
struct K {
    static constexpr int VEC_FLOATS = (INTER > HID ? (INTER > 4096 ? INTER : 4096)
                                                  : (HID > 4096 ? HID : 4096)) + 64;
    static constexpr int XS = (HID + GRID - 1) / GRID;
    static constexpr int RED_FLOATS = 64 + XS + 8;
    // 0.8B leaves room for a fifth stage; 2B's wider activation scratch does not.
    static constexpr int NS = STAGES;
    static constexpr int SMEM_BYTES = NS * CHUNK + VEC_FLOATS * 4 + RED_FLOATS * 4 + 2 * NS * 8;
};

// ------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ unsigned long long globaltimer();
__device__ __forceinline__ void mbar_init(uint64_t *b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
// sm_86 supports mbarrier.test_wait.parity (mbarrier.try_wait is sm_90+ in CUDA 12.4).
__device__ __forceinline__ bool mbar_try_wait(uint64_t *b, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n .reg .pred p;\n"
                 " mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
    return ok;
}
__device__ __forceinline__ bool mbar_wait_b(uint64_t *b, uint32_t parity) {
    unsigned long long t0 = 0;
    while (!mbar_try_wait(b, parity)) {
        __nanosleep(20);
        unsigned long long now = globaltimer();
        if (!t0) t0 = now;
        else if (now - t0 > 100000000ull) {
            if (g_wd_b == 0) { g_wd_b = 1; g_wd_b_blk = blockIdx.x; }
            return false;
        }
    }
    return true;
}
__device__ __forceinline__ void mbar_wait(uint64_t *b, uint32_t parity) {
    while (!mbar_try_wait(b, parity)) __nanosleep(20);
}
__device__ __forceinline__ void cp_async16(void *dst, const void *src, uint64_t policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                 ::"r"(smem_u32(dst)), "l"(src), "l"(policy) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t *b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void *p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p) : "memory");
}
__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, %0;" ::"n"(N_CONSUMERS) : "memory");
}
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

// KV cache is f16 (llama.cpp's default): unpack the 8 halves exactly.
__device__ __forceinline__ void unpack8(uint4 q, float *f) {
    const __half *h = reinterpret_cast<const __half *>(&q);
#pragma unroll
    for (int i = 0; i < 8; i++) f[i] = __half2float(h[i]);
}

__device__ __forceinline__ unsigned long long pack_score(float s, int idx) {
    uint32_t b = __float_as_uint(s);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)b << 32) | (0xFFFFFFFFu - (uint32_t)idx);
}
__device__ __forceinline__ int unpack_index(unsigned long long k) { return (int)(0xFFFFFFFFu - (uint32_t)k); }
__device__ __forceinline__ uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return h;
}

__device__ __forceinline__ void block_slice(int units, int &u0, int &u1) {
    u0 = (int)((long long)units * blockIdx.x / gridDim.x);
    u1 = (int)((long long)units * (blockIdx.x + 1) / gridDim.x);
}

// ------------------------------------------------------------------ embedding row from Q8
// Repacked row: K bytes of q+128, then K/32 fp16 scales.
template <int K>
__device__ __forceinline__ float embed_at(const unsigned char *base, long long tok, int i) {
    const unsigned char *row = base + (size_t)tok * row_bytes(K);
    const int q = (int)row[i] - 128;
    const __half s = *reinterpret_cast<const __half *>(row + K + (i / Q8_GROUP) * 2);
    return (float)q * __half2float(s);
}

// ------------------------------------------------------------------ producer
template <int HID, int INTER>
__device__ __forceinline__ void layer_gemv(const q35_layer_ptrs_t &L, int op, const char *&W, int &rows, int &Kd, int &G) {
    G = 1;
    switch (op) {
        case 0: W = (const char *)L.w_in; rows = L.is_attn ? ATT_IN_ROWS : GDN_IN_ROWS; Kd = HID; break;
        case 1: W = (const char *)L.w_out; rows = HID; Kd = 2048; break;
        case 2: W = (const char *)L.w_gu; rows = 2 * INTER; Kd = HID; G = 2; break;
        default: W = (const char *)L.w_down; rows = HID; Kd = INTER; break;
    }
}

struct Cursor {
    int s, l, op, u, u1, ub, upc;
    const char *W;
    template <int HID, int INTER>
    __device__ void setup() {
        int rows, Kd, G;
        if (l < N_LAYERS) layer_gemv<HID, INTER>(P.layer[l], op, W, rows, Kd, G);
        else { W = (const char *)P.lm_head; rows = VOCAB; Kd = HID; G = 1; }
        block_slice(rows / G, u, u1);
        ub = G * row_bytes(Kd);
        upc = CHUNK / ub;
    }
    __device__ void next_op(int n_steps, int lm_from) {
        if (l < N_LAYERS) { if (++op == 4) { op = 0; l++; } }
        else { l = 0; op = 0; s++; }
        if (l == N_LAYERS && s < lm_from) { l = 0; op = 0; s++; }
        (void)n_steps;
    }
    __device__ void skip_empty(int n_steps, int lm_from) {
        while (s < n_steps && u >= u1) next_op(n_steps, lm_from);
    }
    template <int HID, int INTER>
    __device__ void init(int n_steps, int lm_from) { s = 0; l = 0; op = 0; setup<HID, INTER>(); skip_empty(n_steps, lm_from); }
    __device__ bool valid(int n_steps) const { return s < n_steps; }
    __device__ int bytes() const { return min(upc, u1 - u) * ub; }
    __device__ const char *ptr() const { return W + (size_t)u * ub; }
    __device__ void advance(int n_steps, int lm_from) { u += upc; skip_empty(n_steps, lm_from); }
};

// All 32 lanes issue cp.async 16 B copies.  After a lane's last copy for a slot it
// arrives on full[slot] without incrementing the pending count; the barrier was
// initialised with 32, so consumers see it complete exactly when all bytes landed.
template <int HID, int INTER, int NS>
__device__ void producer(const q35_args_t &A, char *ring, uint64_t *full, uint64_t *empty,
                         volatile int *stop, volatile int *issued) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    volatile int *cseq = (volatile int *)empty;
    const int lane = threadIdx.x & 31;

    // Explicit state machine over (step, layer, op).  op 0..3 are the layer's four GEMVs;
    // op 4 means "LM head of this step" (l == N_LAYERS).  The host tells us the first step
    // whose LM head is needed through A.lm_from.
    int s = 0, l = 0, op = 0;
    auto advance = [&]() {
        if (l < N_LAYERS) {
            if (++op == 4) { op = 0; l++; }
            if (l == N_LAYERS && s < A.lm_from) { l = 0; op = 0; s++; }
        } else {
            l = 0; op = 0; s++;
        }
    };
    auto setup = [&](const char *&W, int &u, int &u1, int &ub, int &upc) {
        int rows, Kd, G;
        if (l < N_LAYERS) layer_gemv<HID, INTER>(P.layer[l], op, W, rows, Kd, G);
        else { W = (const char *)P.lm_head; rows = VOCAB; Kd = HID; G = 1; }
        block_slice(rows / G, u, u1);
        ub = G * row_bytes(Kd);
        upc = CHUNK / ub;
    };

    const char *W = nullptr;
    int u = 0, u1 = 0, ub = 0, upc = 0;
    int c = 0;
    while (s < A.n_steps) {
        setup(W, u, u1, ub, upc);
        if (u < u1) {
            for (; u < u1; u += upc) {
                const int slot = c % NS;
                const int r = c / NS;
                if (r > 0) {
                    while (cseq[slot] < r * N_CONSUMER_WARPS) {
                        if (*stop || g_wd_b) { if (lane == 0) *issued = c; return; }
                        __nanosleep(32);
                    }
                }
                const int n = min(upc, u1 - u) * ub;
                char *dst = ring + slot * CHUNK;
                const char *src = W + (size_t)u * ub;
                for (int i = lane * 16; i < n; i += 32 * 16) cp_async16(dst + i, src + i, policy);
                cp_async_arrive_noinc(&full[slot]);
                c++;
            }
        }
        advance();
    }
    if (lane == 0) *issued = c;
}

// ------------------------------------------------------------------ GEMV
// Input vectors are swizzled so a lane's 16-B loads hit distinct bank groups.
__device__ __forceinline__ int swz(int i) { return i ^ ((i >> 3) & 12); }

template <int Kd>
__device__ __forceinline__ void dot_row_q8(const char *row, const float *vec, int lane, float *out) {
    static_assert(Kd % 128 == 0, "swizzled input vectors");
    const uint4 *w = (const uint4 *)row;
    const __half *scale = (const __half *)(row + Kd);
    constexpr int NV = Kd / 16;
    int o[4];
#pragma unroll
    for (int c = 0; c < 4; c++) o[c] = swz(16 * lane + 4 * c);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < (NV + 31) / 32; i++) {
        const int v = lane + 32 * i;
        if (NV % 32 == 0 || v < NV) {
            const uint4 q = w[v];
            float f[16];
            const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                constexpr float BIAS = 8388608.f + 128.f;
                f[4 * c + 0] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7440)) - BIAS;
                f[4 * c + 1] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7441)) - BIAS;
                f[4 * c + 2] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7442)) - BIAS;
                f[4 * c + 3] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7443)) - BIAS;
            }
            const float sc = __half2float(scale[v >> 1]);
            const float *x = vec + 512 * i;
            float p = f[0] * x[o[0] + 0];
            p = fmaf(f[1], x[o[0] + 1], p);
            p = fmaf(f[2], x[o[0] + 2], p);
            p = fmaf(f[3], x[o[0] + 3], p);
            p = fmaf(f[4], x[o[1] + 0], p);
            p = fmaf(f[5], x[o[1] + 1], p);
            p = fmaf(f[6], x[o[1] + 2], p);
            p = fmaf(f[7], x[o[1] + 3], p);
            p = fmaf(f[8], x[o[2] + 0], p);
            p = fmaf(f[9], x[o[2] + 1], p);
            p = fmaf(f[10], x[o[2] + 2], p);
            p = fmaf(f[11], x[o[2] + 3], p);
            p = fmaf(f[12], x[o[3] + 0], p);
            p = fmaf(f[13], x[o[3] + 1], p);
            p = fmaf(f[14], x[o[3] + 2], p);
            p = fmaf(f[15], x[o[3] + 3], p);
            a = fmaf(p, sc, a);
        }
    }
    out[0] = warp_sum(a);
}


struct Ring {
    char *buf;
    uint64_t *full;
    volatile int *cseq;
    int c;
};

template <int HID, int INTER, int Kd, int G, int NS, class Epi>
__device__ __forceinline__ void gemv(int rows, Ring &R, const float *vec, bool timed, Epi epi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int u0, u1;
    block_slice(rows / G, u0, u1);
    constexpr int RB = row_bytes(Kd);
    constexpr int UPC = CHUNK / (G * RB);
    for (int u = u0; u < u1; u += UPC, R.c++) {
        const int slot = R.c % NS;
        (void)timed;
        mbar_wait(&R.full[slot], (R.c / NS) & 1);
        const int n = min(UPC, u1 - u);
        const char *tile = R.buf + slot * CHUNK;
        for (int j = (warp - (u - u0) % N_CONSUMER_WARPS + N_CONSUMER_WARPS) % N_CONSUMER_WARPS; j < n;
             j += N_CONSUMER_WARPS) {
            float acc[G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                float o;
                dot_row_q8<Kd>(tile + (j * G + g) * RB, vec, lane, &o);
                acc[g] = o;
            }
            epi(u + j, acc);
        }
        __syncwarp();
        if (lane == 0) atomicAdd((int *)&R.cseq[slot], 1);
    }
}

// Partial dot over one SPLIT-th of a row.  K/SPLIT stays a multiple of 128, so the
// swizzled input layout still applies; the scale index is global (part*KP/32 + v/2).
template <int Kd, int SPLIT>
__device__ __forceinline__ float dot_part_q8(const char *row, const float *vec, int lane, int part) {
    constexpr int KP = Kd / SPLIT;
    static_assert(KP % 128 == 0, "split keeps 128-aligned parts");
    constexpr int NV = KP / 16;
    const uint4 *w = (const uint4 *)(row + (size_t)part * KP);
    const __half *scale = (const __half *)(row + Kd);
    int o[4];
#pragma unroll
    for (int c = 0; c < 4; c++) o[c] = swz(16 * lane + 4 * c);
    const int sbase = part * (KP / Q8_GROUP);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < (NV + 31) / 32; i++) {
        const int v = lane + 32 * i;
        if (NV % 32 == 0 || v < NV) {
            const uint4 q = w[v];
            float f[16];
            const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                constexpr float BIAS = 8388608.f + 128.f;
                f[4 * c + 0] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7440)) - BIAS;
                f[4 * c + 1] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7441)) - BIAS;
                f[4 * c + 2] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7442)) - BIAS;
                f[4 * c + 3] = __uint_as_float(__byte_perm(ww[c], 0x4B000000u, 0x7443)) - BIAS;
            }
            const float sc = __half2float(scale[sbase + (v >> 1)]);
            const float *x = vec + (size_t)part * KP + 512 * i;
            float p = f[0] * x[o[0] + 0];
            p = fmaf(f[1], x[o[0] + 1], p);
            p = fmaf(f[2], x[o[0] + 2], p);
            p = fmaf(f[3], x[o[0] + 3], p);
            p = fmaf(f[4], x[o[1] + 0], p);
            p = fmaf(f[5], x[o[1] + 1], p);
            p = fmaf(f[6], x[o[1] + 2], p);
            p = fmaf(f[7], x[o[1] + 3], p);
            p = fmaf(f[8], x[o[2] + 0], p);
            p = fmaf(f[9], x[o[2] + 1], p);
            p = fmaf(f[10], x[o[2] + 2], p);
            p = fmaf(f[11], x[o[2] + 3], p);
            p = fmaf(f[12], x[o[3] + 0], p);
            p = fmaf(f[13], x[o[3] + 1], p);
            p = fmaf(f[14], x[o[3] + 2], p);
            p = fmaf(f[15], x[o[3] + 3], p);
            a = fmaf(p, sc, a);
        }
    }
    return warp_sum(a);
}

template <int Kd>
__host__ __device__ constexpr int choose_split(int upc) {
    constexpr int lim = Kd / 128;
    int best = 1;
    if (lim % 2 == 0 && upc * 2 <= 16) best = 2;
    if (lim % 4 == 0 && upc * 4 <= 16) best = 4;
    if (lim % 8 == 0 && upc * 8 <= 16) best = 8;
    if (lim % 16 == 0 && upc * 16 <= 16) best = 16;
    return best;
}

// Split-K GEMV: SPLIT warps cooperate on each row/unit, partials meet in shared memory.
template <int HID, int INTER, int Kd, int G, int NS, int SPLIT, class Epi>
__device__ __forceinline__ void gemv_split(int rows, Ring &R, const float *vec, float *red, Epi epi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int u0, u1;
    block_slice(rows / G, u0, u1);
    constexpr int RB = row_bytes(Kd);
    constexpr int UPC = CHUNK / (G * RB);
    for (int u = u0; u < u1; u += UPC, R.c++) {
        const int slot = R.c % NS;
        if (!mbar_wait_b(&R.full[slot], (R.c / NS) & 1)) break;
        const int n = min(UPC, u1 - u);
        const char *tile = R.buf + slot * CHUNK;
        const int j = warp / SPLIT, part = warp % SPLIT;
        float pv[G];
#pragma unroll
        for (int g = 0; g < G; g++) pv[g] = 0.f;
        if (j < n) {
#pragma unroll
            for (int g = 0; g < G; g++)
                pv[g] = dot_part_q8<Kd, SPLIT>(tile + (j * G + g) * RB, vec, lane, part);
        }
        if (lane == 0) {
#pragma unroll
            for (int g = 0; g < G; g++) red[(j * G + g) * SPLIT + part] = pv[g];
        }
        consumer_sync();
        if (part == 0 && j < n) {
            float acc[G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                float a = 0.f;
#pragma unroll
                for (int p = 0; p < SPLIT; p++) a += red[(j * G + g) * SPLIT + p];
                acc[g] = a;
            }
            epi(u + j, acc);
        }
        consumer_sync();
        if (lane == 0) atomicAdd((int *)&R.cseq[slot], 1);
    }
}

template <int HID, int INTER, int Kd, int G, int NS, class Epi>
__device__ __forceinline__ void run_gemv(int rows, Ring &R, const float *vec, float *red, Epi epi) {
    constexpr int RB = row_bytes(Kd);
    constexpr int UPC = CHUNK / (G * RB);
    constexpr int SPLIT = choose_split<Kd>(UPC);
    if constexpr (SPLIT == 1) gemv<HID, INTER, Kd, G, NS>(rows, R, vec, false, epi);
    else gemv_split<HID, INTER, Kd, G, NS, SPLIT>(rows, R, vec, red, epi);
}

// ------------------------------------------------------------------ LL exchange
__device__ __forceinline__ unsigned long long ld_ll(const unsigned long long *p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_ll_raw(unsigned long long *p, uint32_t v, uint32_t e) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(((unsigned long long)e << 32) | v) : "memory");
}
__device__ __forceinline__ void st_ll(unsigned long long *p, float v, uint32_t e) { st_ll_raw(p, __float_as_uint(v), e); }
__device__ __forceinline__ uint32_t ll_wait_raw(unsigned long long v, const unsigned long long *p, uint32_t e) {
    unsigned long long t0 = 0;
    while ((uint32_t)(v >> 32) != e) {
        if (*(volatile int *)&g_wd_b) return (uint32_t)v;   // another wait already timed out: fail fast
        __nanosleep(32);
        v = ld_ll(p);
        unsigned long long now = globaltimer();
        if (!t0) t0 = now;
        else if (now - t0 > 100000000ull) {
            if (atomicCAS(&g_wd_b, 0, 2) == 0) { g_wd_b_blk = blockIdx.x; g_wd_b_ep = e; }
            return (uint32_t)v;
        }
    }
    return (uint32_t)v;
}
__device__ __forceinline__ float ll_wait(unsigned long long v, const unsigned long long *p, uint32_t e) {
    return __uint_as_float(ll_wait_raw(v, p, e));
}
__device__ __forceinline__ float ll_get(const unsigned long long *p, uint32_t e) { return ll_wait(ld_ll(p), p, e); }

// Batch path vecf hazard: P.vecf is one global buffer that every CTA rewrites in each op's prologue.
// A CTA may only start overwriting it once every CTA has finished READING the previous op's
// contents.  Each CTA bumps g_vdone after the last vecf read of a GEMV; the next prologue waits for
// GRID * (GEMVs so far) arrivals before its first store.  (Debug grid.sync build: not needed.)
#ifdef Q35_DBG_GRIDSYNC
__device__ __forceinline__ void vdone_arrive() {}
__device__ __forceinline__ void vdone_wait(unsigned) {}
#else
__device__ __forceinline__ void vdone_arrive() {
    __syncthreads();   // every thread's vecf loads of this GEMV are complete
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_vdone) : "memory");
}
__device__ __forceinline__ void vdone_wait(unsigned target) {
    if (target == 0) return;   // uniform across the block
    if (threadIdx.x == 0) {
        unsigned long long t0 = 0;
        for (;;) {
            unsigned v;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(&g_vdone) : "memory");
            if (v >= target) break;
            if (*(volatile int *)&g_wd_b) break;
            __nanosleep(32);
            const unsigned long long now = globaltimer();
            if (!t0) t0 = now;
            else if (now - t0 > 100000000ull) {
                if (atomicCAS(&g_wd_b, 0, 3) == 0) { g_wd_b_blk = blockIdx.x; g_wd_b_ep = target; }
                break;
            }
        }
    }
    __syncthreads();
}
#endif

// Sliced prologues (H3): each CTA converts only its 1/GRID slice of the activation into vecf, then all
// CTAs meet here before any GEMV reads vecf.  target = GRID * (sliced prologues so far, incl. this one).
__device__ __forceinline__ void vready_sync(unsigned target) {
    __syncthreads();   // this CTA's slice stores are issued by all threads
    if (threadIdx.x == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_vready) : "memory");
        unsigned long long t0 = 0;
        for (;;) {
            unsigned v;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(&g_vready) : "memory");
            if (v >= target) break;
            if (*(volatile int *)&g_wd_b) break;
            __nanosleep(32);
            const unsigned long long now = globaltimer();
            if (!t0) t0 = now;
            else if (now - t0 > 100000000ull) {
                if (atomicCAS(&g_wd_b, 0, 4) == 0) { g_wd_b_blk = blockIdx.x; g_wd_b_ep = target; }
                break;
            }
        }
    }
    __syncthreads();
}

// ------------------------------------------------------------------ prologues
template <int NV>
__device__ __forceinline__ void rms_scale(float *red, const float v[NV], float &r) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < NV; k++) ss = fmaf(v[k], v[k], ss);
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    consumer_sync();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < N_CONSUMER_WARPS; i++) tot += red[i];
    r = rsqrtf(tot / (float)(NV * 512) + 1e-6f);
}

template <int HID>
__device__ void prologue_rmsnorm(float *vec, float *red, uint32_t ex,
                                 const float *w, const unsigned long long *src) {
    constexpr int NV = HID / 512;
    const int t = threadIdx.x;
    float v[NV], r;
    unsigned long long a[NV];
#pragma unroll
    for (int k = 0; k < NV; k++) a[k] = ld_ll(src + t + 512 * k);
#pragma unroll
    for (int k = 0; k < NV; k++) v[k] = ll_wait(a[k], src + t + 512 * k, ex);
    rms_scale<NV>(red, v, r);
#pragma unroll
    for (int k = 0; k < NV; k++) vec[swz(t + 512 * k)] = v[k] * r * w[t + 512 * k];
    consumer_sync();
}

template <int HID>
__device__ void prologue_emb(float *vec, float *red, const unsigned char *embed, long long tok, const float *w) {
    constexpr int NV = HID / 512;
    const int t = threadIdx.x;
    float v[NV], r;
#pragma unroll
    for (int k = 0; k < NV; k++) v[k] = embed_at<HID>(embed, tok, t + 512 * k);
    rms_scale<NV>(red, v, r);
#pragma unroll
    for (int k = 0; k < NV; k++) vec[swz(t + 512 * k)] = v[k] * r * w[t + 512 * k];
    consumer_sync();
}

// GDN output: per-head RMSNorm gated by silu(z); one warp per head.
__device__ void prologue_gated_norm(float *vec, const q35_layer_ptrs_t &L, uint32_t e) {
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned long long ro[4], rz[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        ro[k] = ld_ll(P.corel + h * LIN_DV + lane + 32 * k);
        rz[k] = ld_ll(P.projl + GDN_Z_OFF + h * LIN_DV + lane + 32 * k);
    }
    float o[4], ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        o[k] = ll_wait(ro[k], P.corel + h * LIN_DV + lane + 32 * k, e - 1);
        ss = fmaf(o[k], o[k], ss);
    }
    const float r = rsqrtf(warp_sum(ss) / LIN_DV + 1e-6f);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int d = lane + 32 * k;
        const float z = ll_wait(rz[k], P.projl + GDN_Z_OFF + h * LIN_DV + d, e - 2);
        vec[swz(h * LIN_DV + d)] = o[k] * r * L.gdn_norm[d] * silu(z);
    }
    consumer_sync();
}

__device__ void prologue_attn_gate(float *vec, uint32_t e) {
    constexpr int PER = NQ * HD / N_CONSUMERS;
    unsigned long long ra[PER], rg[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int i = threadIdx.x + k * N_CONSUMERS, h = i / HD, d = i % HD;
        ra[k] = ld_ll(P.attnl + i);
        rg[k] = ld_ll(P.projl + h * 2 * HD + HD + d);
    }
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int i = threadIdx.x + k * N_CONSUMERS, h = i / HD, d = i % HD;
        vec[swz(i)] = ll_wait(ra[k], P.attnl + i, e - 1) *
                      sigmoid(ll_wait(rg[k], P.projl + h * 2 * HD + HD + d, e - 2));
    }
    consumer_sync();
}

template <int INTER>
__device__ void prologue_act(float *vec, uint32_t e) {
    constexpr int PER = INTER / N_CONSUMERS;
    unsigned long long r[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) r[k] = ld_ll(P.actl + threadIdx.x + k * N_CONSUMERS);
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int i = threadIdx.x + k * N_CONSUMERS;
        vec[swz(i)] = ll_wait(r[k], P.actl + i, e - 1);
    }
    consumer_sync();
}

// Conv history of one slot/layer: CONV_RING rows of CONV_DIM floats ([ring][channel]), so a step reads
// three rows and writes one, coalesced across channels.  (The old [channel][ring] layout touched one
// 32-byte sector per channel per step for 4 useful bytes: +19% DRAM traffic over the GDN state.)
__device__ __forceinline__ int hist_at(int r) { return r * CONV_DIM; }

// ------------------------------------------------------------------ GDN recurrence
// Task = (head, 32 value columns); state is [head][v][k] so a column is one 512-B row.
__device__ void gdn_op(const q35_layer_ptrs_t &L, int pos, uint32_t e, float *sm) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float *sqk = sm;
    float *sv = sm + 256;
    float *sc = sm + 288;
    constexpr int SLICES = LIN_DV / GDN_SLICE;
    constexpr int RM = CONV_RING - 1;
    for (int task = blockIdx.x; task < LIN_H * SLICES; task += gridDim.x) {
        const int h = task / SLICES, sl = task % SLICES;
        if (t < 256 + GDN_SLICE) {
            const int ch = t < 128 ? h * LIN_DK + t
                         : t < 256 ? LIN_QK + h * LIN_DK + (t - 128)
                                   : 2 * LIN_QK + h * LIN_DV + sl * GDN_SLICE + (t - 256);
            float *hist = L.conv_hist + ch;   // [ring][channel] layout, see hist_at
            float win[CONV_K];
            win[0] = __ldcg(hist + hist_at((pos - 3) & RM));
            win[1] = __ldcg(hist + hist_at((pos - 2) & RM));
            win[2] = __ldcg(hist + hist_at((pos - 1) & RM));
            const float4 wq = *(const float4 *)(L.conv_w + ch * CONV_K);
            const unsigned long long raw = ld_ll(P.projl + ch);
            win[3] = ll_wait(raw, P.projl + ch, e - 1);
            const float acc = fmaf(wq.w, win[3], fmaf(wq.z, win[2], fmaf(wq.y, win[1], wq.x * win[0])));
            if (t >= 256 || sl == 0) __stcg(hist + hist_at(pos & RM), win[3]);
            if (t < 256) sqk[t] = silu(acc); else sv[t - 256] = silu(acc);
        } else if (t < 256 + GDN_SLICE + 1) {
            const unsigned long long *pb = P.projl + GDN_B_OFF + h;
            const unsigned long long *pa = P.projl + GDN_A_OFF + h;
            const unsigned long long rb = ld_ll(pb), ra = ld_ll(pa);
            const float beta = sigmoid(ll_wait(rb, pb, e - 1));
            const float a = ll_wait(ra, pa, e - 1);
            sc[0] = beta;
            sc[1] = __expf(L.ssm_a[h] * softplus(a + L.dt_bias[h]));
        }
        consumer_sync();
        if (warp < 2) {
            const int qk = warp;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) { const float v = sqk[qk * 128 + lane + 32 * i]; ss = fmaf(v, v, ss); }
            ss = warp_sum(ss);
            if (lane == 0) sc[2 + qk] = rsqrtf(ss + 1e-6f) * (qk == 0 ? rsqrtf((float)LIN_DK) : 1.f);
        }
        consumer_sync();
        float s[2][4];
        float *S[2];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const size_t off = ((size_t)h * LIN_DV + sl * GDN_SLICE + warp + 16 * c) * LIN_DK;
            S[c] = L.state + off;
        }
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) s[c][i] = __ldcg(S[c] + lane + 32 * i);
        const float beta = sc[0], decay = sc[1], rq = sc[2], rk = sc[3];
        float kv[4], qv[4], km[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int k = lane + 32 * i;
            kv[i] = sqk[128 + k] * rk;
            qv[i] = sqk[k] * rq;
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
#pragma unroll
            for (int i = 0; i < 4; i++) { s[c][i] *= decay; km[c] = fmaf(s[c][i], kv[i], km[c]); }
            km[c] = warp_sum(km[c]);
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
            float *dst = S[c];
            const float delta = (sv[warp + 16 * c] - km[c]) * beta;
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s[c][i] = fmaf(kv[i], delta, s[c][i]);
                __stcg(dst + lane + 32 * i, s[c][i]);
                o = fmaf(s[c][i], qv[i], o);
            }
            o = warp_sum(o);
            if (lane == 0) st_ll(P.corel + h * LIN_DV + sl * GDN_SLICE + warp + 16 * c, o, e);
        }
        consumer_sync();
    }
}

// ------------------------------------------------------------------ attention
constexpr int KV_BATCH = 1;
__device__ __forceinline__ void attn_split(int n, int &splits, int &chunk) {
    splits = min(MAX_SPLITS, (n + 31) / 32);
    chunk = (n + splits - 1) / splits;
    splits = (n + chunk - 1) / chunk;
}
__device__ __forceinline__ bool attn_task(int task, int pos, int &g, int &sp, int &splits, int &chunk) {
    attn_split(pos + 1, splits, chunk);
    if (task < NKV * splits) { g = task / splits; sp = task % splits; return true; }
    return false;
}
__device__ __forceinline__ void prefetch_kv(const q35_layer_ptrs_t &L, int pos) {
    int g, sp, splits, chunk;
    if (!attn_task(blockIdx.x, pos, g, sp, splits, chunk)) return;
    const int t0 = sp * chunk, t1 = min(pos, t0 + chunk);
    if (t1 <= t0) return;
    const size_t off = ((size_t)g * P.max_seq + t0) * HD;
    prefetch_l2((const __half *)L.k_cache + off);
    prefetch_l2((const __half *)L.v_cache + off);
}

__device__ void attn_op(const q35_layer_ptrs_t &L, int pos, uint32_t e, float *sm, float *red) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float *sq = sm;
    float *sacc = sm + 1024;
    float *sml = sm + 2048;
    float *skv = sm + 2176;
    int *sflag = (int *)(red + 32);
    const size_t ms = (size_t)P.max_seq;
    int g, sp, splits, chunk;
    for (int task = blockIdx.x; attn_task(task, pos, g, sp, splits, chunk); task += gridDim.x) {
        const int n = pos + 1;
        const int t0 = sp * chunk, t1 = min(n, t0 + chunk);
        const int nb = pos, nn = max(0, t1 - nb);
        const __half *Kc = (const __half *)L.k_cache + (size_t)g * ms * HD + lane * 8;
        const __half *Vc = (const __half *)L.v_cache + (size_t)g * ms * HD + lane * 8;
        uint4 kq[KV_BATCH], vq[KV_BATCH];
#pragma unroll
        for (int b = 0; b < KV_BATCH; b++) {
            const int tt = t0 + warp + b * N_CONSUMER_WARPS;
            if (tt < min(t1, pos)) {
                kq[b] = __ldcg((const uint4 *)(Kc + (size_t)tt * HD));
                vq[b] = __ldcg((const uint4 *)(Vc + (size_t)tt * HD));
            }
        }
        if (warp < Q_PER_KV + nn) {
            const bool isq = warp < Q_PER_KV;
            const int i = warp - Q_PER_KV;
            const unsigned long long *src = isq
                ? P.projl + (g * Q_PER_KV + warp) * 2 * HD
                : P.projl + ATT_K_OFF + g * HD;
            const float *nw = isq ? L.q_norm : L.k_norm;
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
            float v[8], ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) { v[j] = ll_wait(raw[j], src + lane * 8 + j, e - 1); ss = fmaf(v[j], v[j], ss); }
            const float r = rsqrtf(warp_sum(ss) / HD + 1e-6f);
            float *dst = isq ? sq + warp * HD : skv + i * 2 * HD;
#pragma unroll
            for (int j = 0; j < 8; j++) dst[lane * 8 + j] = v[j] * r * nw[lane * 8 + j];
        } else if (warp >= Q_PER_KV + 1 && warp < Q_PER_KV + 1 + nn) {
            const int i = warp - Q_PER_KV - 1;
            const unsigned long long *src = P.projl + ATT_V_OFF + g * HD;
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
#pragma unroll
            for (int j = 0; j < 8; j++) skv[i * 2 * HD + HD + lane * 8 + j] = ll_wait(raw[j], src + lane * 8 + j, e - 1);
        }
        consumer_sync();
        if (t < (Q_PER_KV + nn) * 32) {
            const int hh = t >> 5, i = t & 31;
            float *b = hh < Q_PER_KV ? sq + hh * HD : skv + (hh - Q_PER_KV) * 2 * HD;
            const float c = P.rope_cos[pos * (ROT / 2) + i], s = P.rope_sin[pos * (ROT / 2) + i];
            const float x1 = b[i], x2 = b[i + 32];
            b[i] = fmaf(x1, c, -(x2 * s));
            b[i + 32] = fmaf(x2, c, x1 * s);
        }
        consumer_sync();
        if (nn > 0) {
            for (int i = 0; i < nn; i++) {
                const int d = t & (HD - 1);
                __half *cache = (t < HD) ? (__half *)L.k_cache : (__half *)L.v_cache;
                cache[((size_t)g * ms + nb + i) * HD + d] = __float2half(skv[i * 2 * HD + t]);
            }
            consumer_sync();
        }
        float m[Q_PER_KV], l[Q_PER_KV], acc[Q_PER_KV][8];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            m[h] = -1e30f; l[h] = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[h][j] = 0.f;
        }
        for (int base = t0 + warp, b0 = 0; base < t1; base += KV_BATCH * N_CONSUMER_WARPS, b0++) {
            if (b0 > 0) {
#pragma unroll
                for (int b = 0; b < KV_BATCH; b++) {
                    const int tt = base + b * N_CONSUMER_WARPS;
                    if (tt < t1) {
                        kq[b] = __ldcg((const uint4 *)(Kc + (size_t)tt * HD));
                        vq[b] = __ldcg((const uint4 *)(Vc + (size_t)tt * HD));
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < KV_BATCH; b++) {
                const int tt = base + b * N_CONSUMER_WARPS;
                if (tt >= t1) break;
                if (tt >= pos) {
                    kq[b] = __ldcg((const uint4 *)(Kc + (size_t)tt * HD));
                    vq[b] = __ldcg((const uint4 *)(Vc + (size_t)tt * HD));
                }
                float kf[8], vf[8], sc[Q_PER_KV];
                unpack8(kq[b], kf);
                unpack8(vq[b], vf);
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) {
                    const float4 q0 = *(const float4 *)(sq + h * HD + lane * 8);
                    const float4 q1 = *(const float4 *)(sq + h * HD + lane * 8 + 4);
                    float d = q0.x * kf[0];
                    d = fmaf(q0.y, kf[1], d); d = fmaf(q0.z, kf[2], d); d = fmaf(q0.w, kf[3], d);
                    d = fmaf(q1.x, kf[4], d); d = fmaf(q1.y, kf[5], d); d = fmaf(q1.z, kf[6], d); d = fmaf(q1.w, kf[7], d);
                    sc[h] = d;
                }
#pragma unroll
                for (int o = 16; o; o >>= 1)
#pragma unroll
                    for (int h = 0; h < Q_PER_KV; h++) sc[h] += __shfl_xor_sync(~0u, sc[h], o);
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) sc[h] *= 0.0625f;
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) {
                    const float mn = fmaxf(m[h], sc[h]);
                    const float corr = __expf(m[h] - mn), p = __expf(sc[h] - mn);
                    l[h] = fmaf(l[h], corr, p);
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[h][j] = fmaf(p, vf[j], acc[h][j] * corr);
                    m[h] = mn;
                }
            }
        }
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < Q_PER_KV; h++) { sml[(warp * Q_PER_KV + h) * 2] = m[h]; sml[(warp * Q_PER_KV + h) * 2 + 1] = l[h]; }
        consumer_sync();
        float sc[Q_PER_KV];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            float M = -1e30f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + h) * 2]);
            sc[h] = __expf(m[h] - M);
        }
        float *const pg = P.attn_part + ((size_t)g) * MAX_SPLITS * Q_PER_KV * PART_STRIDE;
        float *part = pg + (size_t)sp * Q_PER_KV * PART_STRIDE;
        if (t < Q_PER_KV) {
            float M = -1e30f, Ls = 0.f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + t) * 2]);
            for (int w = 0; w < N_CONSUMER_WARPS; w++)
                Ls = fmaf(sml[(w * Q_PER_KV + t) * 2 + 1], __expf(sml[(w * Q_PER_KV + t) * 2] - M), Ls);
            __stcg(part + t * PART_STRIDE + HD, M);
            __stcg(part + t * PART_STRIDE + HD + 1, Ls);
        }
        consumer_sync();
        float *wbuf = sm;
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
#pragma unroll
            for (int j = 0; j < 8; j++) wbuf[warp * HD + lane * 8 + j] = acc[h][j] * sc[h];
            consumer_sync();
            if (t < HD) {
                float a = 0.f;
                for (int w = 0; w < N_CONSUMER_WARPS; w++) a += wbuf[w * HD + t];
                __stcg(part + h * PART_STRIDE + t, a);
            }
            consumer_sync();
        }
        unsigned *cnt = P.attn_cnt + g;
        if (t == 0) {
            __threadfence();
            *sflag = atomicAdd(cnt, 1u) == (unsigned)(splits - 1);
        }
        consumer_sync();
        if (*sflag) {
            __threadfence();
            float *sw = sacc;
            for (int i = t; i < splits * Q_PER_KV; i += N_CONSUMERS) {
                sw[i] = __ldcg(pg + i * PART_STRIDE + HD);
                sw[MAX_SPLITS * Q_PER_KV + i] = __ldcg(pg + i * PART_STRIDE + HD + 1);
            }
            consumer_sync();
            if (t < Q_PER_KV) {
                float M = -1e30f, den = 0.f;
                for (int p = 0; p < splits; p++) M = fmaxf(M, sw[p * Q_PER_KV + t]);
                for (int p = 0; p < splits; p++)
                    den = fmaf(__expf(sw[p * Q_PER_KV + t] - M), sw[MAX_SPLITS * Q_PER_KV + p * Q_PER_KV + t], den);
                for (int p = 0; p < splits; p++) sw[p * Q_PER_KV + t] = __expf(sw[p * Q_PER_KV + t] - M) / den;
            }
            consumer_sync();
            for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) {
                const int h = i / HD, d = i % HD;
                float v[MAX_SPLITS];
#pragma unroll
                for (int p = 0; p < MAX_SPLITS; p++)
                    v[p] = p < splits ? __ldcg(pg + (p * Q_PER_KV + h) * PART_STRIDE + d) : 0.f;
                float num = 0.f;
#pragma unroll
                for (int p = 0; p < MAX_SPLITS; p++) num = fmaf(p < splits ? sw[p * Q_PER_KV + h] : 0.f, v[p], num);
                st_ll(P.attnl + (g * Q_PER_KV + h) * HD + d, num, e);
            }
            if (t == 0) atomicExch(cnt, 0u);
        }
        consumer_sync();
    }
}

__device__ unsigned long long reduce_lm(const unsigned long long *slots, int stride, uint32_t e, unsigned long long *sk) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    unsigned long long key = 0;
    if (t < gridDim.x) {
        const unsigned long long *p = slots + (size_t)t * stride;
        const unsigned long long a = ld_ll(p), b = ld_ll(p + 1);
        key = ((unsigned long long)ll_wait_raw(a, p, e) << 32) | ll_wait_raw(b, p + 1, e);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(~0u, key, o);
        key = other > key ? other : key;
    }
    if (lane == 0) sk[warp] = key;
    consumer_sync();
    key = 0;
#pragma unroll
    for (int i = 0; i < N_CONSUMER_WARPS; i++) key = sk[i] > key ? sk[i] : key;
    consumer_sync();
    return key;
}

// ------------------------------------------------------------------ kernel
template <int HID, int INTER>
__global__ void __launch_bounds__(N_THREADS, 1) decode_kernel(q35_args_t A) {
    constexpr int NS = K<HID, INTER>::NS;
    constexpr int VB = K<HID, INTER>::VEC_FLOATS;
    constexpr int REDF = K<HID, INTER>::RED_FLOATS;
    extern __shared__ __align__(1024) char smem[];
    char *ring = smem;
    float *vec = (float *)(smem + NS * CHUNK);
    float *red = (float *)(smem + NS * CHUNK + VB * 4);
    uint64_t *full = (uint64_t *)(red + REDF);
    uint64_t *empty = full + NS;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;

    volatile int *stop = (volatile int *)(red + 36);
    volatile int *issued = (volatile int *)(red + 37);
    if (t == 0) {
        for (int s = 0; s < NS; s++) { mbar_init(&full[s], 32); ((volatile int *)empty)[s] = 0; }
    }
    __syncthreads();
    if (t == 0) { *stop = 0; *issued = -1; }
    __syncthreads();

    if (warp == N_CONSUMER_WARPS) {
        producer<HID, INTER, NS>(A, ring, full, empty, stop, issued);
        return;
    }

    int xr0, xr1;
    block_slice(HID, xr0, xr1);
    float *xs = red + 48;
    unsigned long long *sk = (unsigned long long *)(vec + 2048);

    Ring R{ring, full, (volatile int *)empty, 0};
    uint32_t e = A.epoch_base;
    int op = 0;
    int gop = 0;
    auto op_end = [&]() {
        consumer_sync();
        if (A.prof && blockIdx.x == 0 && t == 0) A.prof[gop] = globaltimer();
        e++;
        op++;
        gop++;
    };

    auto layer = [&](const q35_layer_ptrs_t &L, int pos, long long tok, bool is_first) {
        if (L.is_attn && t == 0) prefetch_kv(L, pos);
        if (is_first) prologue_emb<HID>(vec, red, (const unsigned char *)P.embed, tok, L.in_norm);
        else prologue_rmsnorm<HID>(vec, red, e - 1, L.in_norm, P.xl);
        auto store = [&](int r, const float *a) {
            if (lane == 0) st_ll(P.projl + r, a[0], e);
        };
        if (L.is_attn) run_gemv<HID, INTER, HID, 1, NS>(ATT_IN_ROWS, R, vec, red, store);
        else run_gemv<HID, INTER, HID, 1, NS>(GDN_IN_ROWS, R, vec, red, store);
        op_end();
        if (L.is_attn) attn_op(L, pos, e, vec, red);
        else gdn_op(L, pos, e, vec);
        op_end();
        if (L.is_attn) prologue_attn_gate(vec, e);
        else prologue_gated_norm(vec, L, e);
        run_gemv<HID, INTER, 2048, 1, NS>(HID, R, vec, red, [&](int r, const float *a) {
            if (lane == 0) {
                float *x = xs + r - xr0;
                const float xn = (is_first ? embed_at<HID>((const unsigned char *)P.embed, tok, r) : *x) + a[0];
                *x = xn;
                st_ll(P.xl + r, xn, e);
            }
        });
        op_end();
        prologue_rmsnorm<HID>(vec, red, e - 1, L.post_norm, P.xl);
        run_gemv<HID, INTER, HID, 2, NS>(2 * INTER, R, vec, red, [&](int u, const float *a) {
            if (lane == 0) st_ll(P.actl + u, silu(a[0]) * a[1], e);
        });
        op_end();
        prologue_act<INTER>(vec, e);
        run_gemv<HID, INTER, INTER, 1, NS>(HID, R, vec, red, [&](int r, const float *a) {
            if (lane == 0) {
                float *x = xs + r - xr0;
                const float xn = *x + a[0];
                *x = xn;
                st_ll(P.xl + r, xn, e);
            }
        });
        op_end();
        if (A.prof_step > 0 && op >= A.prof_step) return;
    };

    auto lm_head = [&](int s) {
        prologue_rmsnorm<HID>(vec, red, e - 1, P.final_norm, P.xl);
        unsigned long long best = 0;
        run_gemv<HID, INTER, HID, 1, NS>(VOCAB, R, vec, red, [&](int r, const float *a) {
            float score = a[0];
            if (A.logits_out && lane == 0) A.logits_out[(size_t)s * VOCAB + r] = score;
            if (A.inv_temperature > 0.f) {
                const uint32_t h = hash3(A.seed, (uint32_t)s, (uint32_t)r);
                const float u = ((h >> 8) + 0.5f) * (1.f / 16777216.f);
                score = fmaf(score, A.inv_temperature, -__logf(-__logf(u)));
            }
            const unsigned long long k = pack_score(score, r);
            best = k > best ? k : best;
        });
        if (lane == 0) sk[warp] = best;
        consumer_sync();
        if (t < 1) {
            unsigned long long b = 0;
            for (int i = 0; i < N_CONSUMER_WARPS; i++) b = sk[i] > b ? sk[i] : b;
            st_ll_raw(P.lml + blockIdx.x * 2, (uint32_t)(b >> 32), e);
            st_ll_raw(P.lml + blockIdx.x * 2 + 1, (uint32_t)b, e);
        }
        op_end();
        if (A.prof_step > 0 && op >= A.prof_step) return;
    };

    auto is_stop = [&](int token) {
        bool r = false;
        for (int i = 0; i < A.n_stop; i++) r |= (token == A.stop_ids[i]);
        return r;
    };

    int steps_done = A.n_steps, last_lm = -1;
    for (int s = 0; s < A.n_steps; s++) {
        const int pos = A.start_pos + s;
        op = 0;
        if (A.step_ts && blockIdx.x == 0 && t == 0) A.step_ts[s] = globaltimer();
        int token;
        if (s < A.n_forced) token = A.forced[s];
        else {
            const unsigned long long key = reduce_lm(P.lml, 2, e - 1, sk);
            if (blockIdx.x == 0 && t == 0) A.tokens_out[s - 1] = key;
            token = unpack_index(key);
            if (is_stop(token)) { steps_done = s; break; }
        }
        for (int l = 0; l < N_LAYERS; l++) {
            layer(P.layer[l], pos, token, l == 0);
            if (A.resid_out && t < xr1 - xr0)
                A.resid_out[((size_t)s * N_LAYERS + l) * HID + xr0 + t] = xs[t];
        }
        __threadfence();
        if (s >= A.lm_from) { lm_head(s); last_lm = s; }
    }
    if (steps_done != A.n_steps) {  // early stop: let the producer's in-flight copies land
        if (t == 0) {
            *stop = 1;
            while (*issued < 0) { __nanosleep(32); }
            for (int c = R.c; c < *issued; c++) mbar_wait(&full[c % NS], (c / NS) & 1);
        }
        consumer_sync();
    }
    if (blockIdx.x == 0 && steps_done == A.n_steps && last_lm == A.n_steps - 1) {
        const unsigned long long key = reduce_lm(P.lml, 2, e - 1, sk);
        if (t == 0) A.tokens_out[A.n_steps - 1] = key;
    }
    if (A.prof && blockIdx.x == 0 && t == 0) A.prof[gop] = globaltimer();
    if (blockIdx.x == 0 && t == 0) {
        if (A.step_ts) A.step_ts[steps_done] = globaltimer();
        A.out_info[0] = steps_done;
        A.out_info[1] = (int)e;
        A.out_info[7] = gop;

    }
}


// ================================================================== batched decode
// Phase 2: BT independent slots, one position each per launch.  Every GEMV uses
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32: M = 16 weight rows, N = 8 entries per
// n-tile (BT/8 tiles, BT < 8 padded), K = 16.  Weights are Q8_0, repacked at load time into
// the fragment layout [group][kt][544] (see qwen35_engine.c).  One 32-bit load per lane
// gives the two int8 of one A register; int8 -> f16 is exact and the fp16 block scale is
// applied in fp32 after the two k16 MMAs, blocks in ascending k order.  That makes each
// (row, entry) a fixed arithmetic sequence independent of BT and of the other columns (G4).
//
// The activation is f16, produced in fragment-major layout by the prologues.  Two GEMV
// shapes are used: rows with at least 64 groups per CTA stream A fragments straight from
// global with a deep register pipeline (LM head, big projections); small projections use a
// CTA-wide cp.async ring over one group at a time so each CTA reads its weight slice as one
// long sequential stream.

constexpr int TC_CHUNKT = 32;
constexpr int TC_NS = 4;
constexpr int TC_RING_BYTES = TC_NS * TC_CHUNKT * 544;

// int8 x4 -> two f16x2 (bytes 0,1 -> lo; bytes 2,3 -> hi), exact: PRMT builds the f16 1024 + (q + 128)
// and one HSUB2 removes 1152.  Equal to the I2F conversion for all 65,536 byte pairs (I2F is quarter
// rate on sm_86).
__device__ __forceinline__ void cvt4(uint32_t p, uint32_t &lo, uint32_t &hi) {
    const uint32_t x = p ^ 0x80808080u;
    const uint32_t a = __byte_perm(x, 0x6464u, 0x5140), b = __byte_perm(x, 0x6464u, 0x5342);
    const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    __half2 ha = __hsub2(*reinterpret_cast<const __half2 *>(&a), bias);
    __half2 hb = __hsub2(*reinterpret_cast<const __half2 *>(&b), bias);
    lo = *reinterpret_cast<uint32_t *>(&ha); hi = *reinterpret_cast<uint32_t *>(&hb);
}
__device__ __forceinline__ void mma_m16n8k16(float d[4], uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// vecf is written inside this kernel (each CTA writes a slice) and read by the GEMVs of the same
// kernel, so ld.global.nc (__ldg) is not allowed; ld.global.cg is L2-coherent.
__device__ __forceinline__ uint32_t ld_vecf(const uint32_t *p) { return __ldcg(p); }
// fragment-major activation index (u32): [k/8][n-tile][lane], lane = (entry%8)*4 + (k%8)/2
__device__ __forceinline__ int vf_idx(int k, int e) {
    return ((k >> 3) * Q35_NTMAX + (e >> 3)) * 32 + ((e & 7) << 2) + ((k & 7) >> 1);
}
__device__ __forceinline__ void vf_put(uint32_t *vecf, int k, int e, float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    vecf[vf_idx(k, e)] = *reinterpret_cast<uint32_t *>(&h);
}

// runtime-size RMS scale (reduction order fixed: warp butterfly, then warps 0..15)
__device__ __forceinline__ float rms_scale_n(float *red, float ss, int ntotal) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < N_CONSUMER_WARPS; i++) tot += red[i];
    return rsqrtf(tot / (float)ntotal + 1e-6f);
}

template <int HID>
__device__ void prologue_rmsnorm_b(uint32_t *vecf, float *red, uint32_t ex,
                                   const float *w, const unsigned long long *xl, int nt, unsigned vt) {
    constexpr int NP = HID / 1024;
    const int t = threadIdx.x;
    for (int e = 0; e < nt; e++) {
        const unsigned long long *src = xl + (size_t)e * HID;
        float v[2 * NP];
        unsigned long long a[2 * NP];
#pragma unroll
        for (int i = 0; i < NP; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int k = 2 * t + 1024 * i + j;
                a[2 * i + j] = ld_ll(src + k);
            }
        }
#pragma unroll
        for (int i = 0; i < NP; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int k = 2 * t + 1024 * i + j;
                v[2 * i + j] = ll_wait(a[2 * i + j], src + k, ex);
            }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 2 * NP; i++) ss = fmaf(v[i], v[i], ss);
        const float r = rms_scale_n(red, ss, HID);
        if (e == 0) vdone_wait(vt);
#pragma unroll
        for (int i = 0; i < NP; i++) {
            const int k = 2 * t + 1024 * i;
            vf_put(vecf, k, e, v[2 * i] * r * w[k], v[2 * i + 1] * r * w[k + 1]);
        }
        __syncthreads();
    }
}

template <int HID>
__device__ void prologue_emb_b(uint32_t *vecf, float *red, const unsigned char *embed,
                               const int32_t *tokens, int nt, const float *w) {
    constexpr int NP = HID / 1024;
    const int t = threadIdx.x;
    for (int e = 0; e < nt; e++) {
        float v[2 * NP];
#pragma unroll
        for (int i = 0; i < NP; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int k = 2 * t + 1024 * i + j;
                v[2 * i + j] = embed_at<HID>(embed, tokens[e], k);
            }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 2 * NP; i++) ss = fmaf(v[i], v[i], ss);
        const float r = rms_scale_n(red, ss, HID);
#pragma unroll
        for (int i = 0; i < NP; i++) {
            const int k = 2 * t + 1024 * i;
            vf_put(vecf, k, e, v[2 * i] * r * w[k], v[2 * i + 1] * r * w[k + 1]);
        }
        __syncthreads();
    }
}

__device__ void prologue_gated_norm_b(uint32_t *vecf, const q35_layer_ptrs_t &L, uint32_t e,
                                      int nt, unsigned vt) {
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int t = 0; t < nt; t++) {
        unsigned long long ro[4], rz[4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = 2 * lane + 64 * i + j;
                ro[2 * i + j] = ld_ll(P.corel + (size_t)t * LIN_V + h * LIN_DV + d);
                rz[2 * i + j] = ld_ll(P.projl + (size_t)t * PROJ_MAX + GDN_Z_OFF + h * LIN_DV + d);
            }
        float o[4], z[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = 2 * lane + 64 * i + j;
                o[2 * i + j] = ll_wait(ro[2 * i + j], P.corel + (size_t)t * LIN_V + h * LIN_DV + d, e - 1);
                ss = fmaf(o[2 * i + j], o[2 * i + j], ss);
            }
        const float r = rsqrtf(warp_sum(ss) / LIN_DV + 1e-6f);
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = 2 * lane + 64 * i + j;
                z[2 * i + j] = ll_wait(rz[2 * i + j], P.projl + (size_t)t * PROJ_MAX + GDN_Z_OFF + h * LIN_DV + d, e - 2);
            }
        if (t == 0) vdone_wait(vt);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int d = 2 * lane + 64 * i;
            vf_put(vecf, h * LIN_DV + d, t,
                   o[2 * i] * r * L.gdn_norm[d] * silu(z[2 * i]),
                   o[2 * i + 1] * r * L.gdn_norm[d + 1] * silu(z[2 * i + 1]));
        }
    }
    __syncthreads();
}

__device__ void prologue_attn_gate_b(uint32_t *vecf, uint32_t e, int nt, unsigned vt) {
    constexpr int PAIRS = NQ * HD / 2 / N_CONSUMERS;   // 2 pairs per thread
    for (int t = 0; t < nt; t++) {
        unsigned long long ra[2 * PAIRS], rg[2 * PAIRS];
#pragma unroll
        for (int p = 0; p < PAIRS; p++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int k = 2 * (threadIdx.x + N_CONSUMERS * p) + j;
                const int h = k / HD, d = k % HD;
                ra[2 * p + j] = ld_ll(P.attnl + (size_t)t * NQ * HD + k);
                rg[2 * p + j] = ld_ll(P.projl + (size_t)t * PROJ_MAX + h * 2 * HD + HD + d);
            }
#pragma unroll
        for (int p = 0; p < PAIRS; p++) {
            const int k = 2 * (threadIdx.x + N_CONSUMERS * p);
            float v0 = 0.f, v1 = 0.f;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int kk = k + j, h = kk / HD, d = kk % HD;
                const float av = ll_wait(ra[2 * p + j], P.attnl + (size_t)t * NQ * HD + kk, e - 1);
                const float gv = ll_wait(rg[2 * p + j], P.projl + (size_t)t * PROJ_MAX + h * 2 * HD + HD + d, e - 2);
                if (j == 0) v0 = av * sigmoid(gv); else v1 = av * sigmoid(gv);
            }
            if (t == 0 && p == 0) vdone_wait(vt);   // first store: every CTA is done with the old vecf
            vf_put(vecf, k, t, v0, v1);
        }
    }
    __syncthreads();
}

template <int INTER>
__device__ void prologue_act_b(uint32_t *vecf, uint32_t e, int nt, unsigned vt) {
    for (int t = 0; t < nt; t++) {
        for (int pair = threadIdx.x; pair < INTER / 2; pair += N_CONSUMERS) {
            const int k = 2 * pair;
            const unsigned long long a0 = ld_ll(P.actl + (size_t)t * INTER + k);
            const unsigned long long a1 = ld_ll(P.actl + (size_t)t * INTER + k + 1);
            const float v0 = ll_wait(a0, P.actl + (size_t)t * INTER + k, e - 1);
            const float v1 = ll_wait(a1, P.actl + (size_t)t * INTER + k + 1, e - 1);
            if (t == 0 && pair == (int)threadIdx.x) vdone_wait(vt);   // first store (every thread has >= 1 pair)
            vf_put(vecf, k, t, v0, v1);
        }
    }
    __syncthreads();
}

// ---- H3: sliced prologues.  Slice boundaries depend only on the row length and GRID, never on the
// batch size, so every reduction below has the same order for any BT / entry position (G4).
__device__ __forceinline__ void slice_of(int n, int &i0, int &cnt) {
    i0 = (int)((long long)n * blockIdx.x / gridDim.x);
    cnt = (int)((long long)n * (blockIdx.x + 1) / gridDim.x) - i0;
}

// RMSNorm, sliced.  The reduction is *defined* exactly as the unsliced block reduction of
// prologue_rmsnorm_b (per-thread FMA chain over the thread's 2*NP elements, warp butterfly, sum of the
// 16 warp partials in order 0..15), so the sliced and unsliced prologues are bitwise identical for any
// BT.  The 16 warp-partials of an entry are computed by CTAs 0..15 (CTA w plays "warp w": 128 elements
// per entry), exchanged through g_ssl, and every CTA then scales only its own 1/GRID slice of the row.
template <int HID, class Ld, class Fin>
__device__ void norm_slice_b(uint32_t *vecf, const float *w, int nt, uint32_t etag, unsigned vt,
                             unsigned rt, Ld ld, Fin fin) {
    constexpr int NP = HID / 1024;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int p0, np;
    slice_of(HID / 2, p0, np);                 // put-slice: pairs [p0, p0+np), np <= 26
    float v[2][2];
    unsigned long long raw[2][2], rawc[2][2 * NP];
    const bool chunk = blockIdx.x < N_CONSUMER_WARPS;   // this CTA produces warp-partial number blockIdx.x
#pragma unroll
    for (int ee = 0; ee < 2; ee++) {
        const int t = warp + 16 * ee;
        if (t < nt) {
            if (lane < np) {
                raw[ee][0] = ld(t, 2 * (p0 + lane));
                raw[ee][1] = ld(t, 2 * (p0 + lane) + 1);
            }
            if (chunk)
#pragma unroll
                for (int i = 0; i < 2 * NP; i++) rawc[ee][i] = ld(t, 2 * (32 * blockIdx.x + lane) + 1024 * (i >> 1) + (i & 1));
        }
    }
#pragma unroll
    for (int ee = 0; ee < 2; ee++) {
        const int t = warp + 16 * ee;
        if (t < nt && lane < np) {
            v[ee][0] = fin(raw[ee][0], t, 2 * (p0 + lane));
            v[ee][1] = fin(raw[ee][1], t, 2 * (p0 + lane) + 1);
        }
        if (t < nt && chunk) {
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 2 * NP; i++) {
                const float x = fin(rawc[ee][i], t, 2 * (32 * blockIdx.x + lane) + 1024 * (i >> 1) + (i & 1));
                ss = fmaf(x, x, ss);
            }
            ss = warp_sum(ss);
            if (lane == 0) st_ll(g_ssl + t * 40 + blockIdx.x, ss, etag);
        }
    }
    float rr[2];
#pragma unroll
    for (int ee = 0; ee < 2; ee++) {
        const int t = warp + 16 * ee;
        rr[ee] = 0.f;
        if (t < nt) {
            float pv = 0.f;
            if (lane < N_CONSUMER_WARPS) {
                const unsigned long long a0 = ld_ll(g_ssl + t * 40 + lane);
                pv = ll_wait(a0, g_ssl + t * 40 + lane, etag);
            }
            float tot = 0.f;   // sequential over the 16 warp partials, as rms_scale_n does
#pragma unroll
            for (int i = 0; i < N_CONSUMER_WARPS; i++) tot += __shfl_sync(0xffffffffu, pv, i);
            rr[ee] = rsqrtf(tot / (float)HID + 1e-6f);
        }
    }
    vdone_wait(vt);
#pragma unroll
    for (int ee = 0; ee < 2; ee++) {
        const int t = warp + 16 * ee;
        if (t < nt && lane < np) {
            const int k = 2 * (p0 + lane);
            vf_put(vecf, k, t, v[ee][0] * rr[ee] * w[k], v[ee][1] * rr[ee] * w[k + 1]);
        }
    }
    vready_sync(rt);
}

template <int HID>
__device__ void prologue_rmsnorm_s(uint32_t *vecf, uint32_t ex, uint32_t etag, const float *w,
                                   const unsigned long long *xl, int nt, unsigned vt, unsigned rt) {
    norm_slice_b<HID>(vecf, w, nt, etag, vt, rt,
        [&](int e, int k) { return ld_ll(xl + (size_t)e * HID + k); },
        [&](unsigned long long a, int e, int k) { return ll_wait(a, xl + (size_t)e * HID + k, ex); });
}

// Gated norm: unit = (entry, head); the units are split over the CTAs, one warp per unit.
__device__ void prologue_gated_norm_s(uint32_t *vecf, const q35_layer_ptrs_t &L, uint32_t e, int nt,
                                      unsigned vt, unsigned rt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int u0, nu;
    slice_of(nt * 16, u0, nu);          // nu <= 13 < 16 warps
    const int u = u0 + warp;
    const bool act = warp < nu;
    const int t = u >> 4, h = u & 15;
    float res[4] = {0.f, 0.f, 0.f, 0.f};
    if (act) {
        unsigned long long ro[4], rz[4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = 2 * lane + 64 * i + j;
                ro[2 * i + j] = ld_ll(P.corel + (size_t)t * LIN_V + h * LIN_DV + d);
                rz[2 * i + j] = ld_ll(P.projl + (size_t)t * PROJ_MAX + GDN_Z_OFF + h * LIN_DV + d);
            }
        float o[4], z[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = 2 * lane + 64 * i + j;
                o[2 * i + j] = ll_wait(ro[2 * i + j], P.corel + (size_t)t * LIN_V + h * LIN_DV + d, e - 1);
                ss = fmaf(o[2 * i + j], o[2 * i + j], ss);
            }
        const float r = rsqrtf(warp_sum(ss) / LIN_DV + 1e-6f);
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = 2 * lane + 64 * i + j;
                z[2 * i + j] = ll_wait(rz[2 * i + j], P.projl + (size_t)t * PROJ_MAX + GDN_Z_OFF + h * LIN_DV + d, e - 2);
            }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int d = 2 * lane + 64 * i;
            res[2 * i] = o[2 * i] * r * L.gdn_norm[d] * silu(z[2 * i]);
            res[2 * i + 1] = o[2 * i + 1] * r * L.gdn_norm[d + 1] * silu(z[2 * i + 1]);
        }
    }
    vdone_wait(vt);
    if (act)
#pragma unroll
        for (int i = 0; i < 2; i++) vf_put(vecf, h * LIN_DV + 2 * lane + 64 * i, t, res[2 * i], res[2 * i + 1]);
    vready_sync(rt);
}

// Attention output gate: pure elementwise, (entry, pair) items split by pair slice.
template <int BT>
__device__ void prologue_attn_gate_s(uint32_t *vecf, uint32_t e, int nt, unsigned vt, unsigned rt) {
    constexpr int NPR = NQ * HD / 2;                         // 1024 pairs per row
    constexpr int NPMAX = (NPR + 39) / 40;                   // 26
    constexpr int ITM = (BT * NPMAX + N_CONSUMERS - 1) / N_CONSUMERS;
    int p0, np;
    slice_of(NPR, p0, np);
    const int total = nt * np;
    unsigned long long ra[ITM][2], rg[ITM][2];
#pragma unroll
    for (int it = 0; it < ITM; it++) {
        const int idx = threadIdx.x + it * N_CONSUMERS;
        if (idx < total) {
            const int t = idx / np, pp = idx - t * np;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int k = 2 * (p0 + pp) + j, h = k / HD, d = k % HD;
                ra[it][j] = ld_ll(P.attnl + (size_t)t * NQ * HD + k);
                rg[it][j] = ld_ll(P.projl + (size_t)t * PROJ_MAX + h * 2 * HD + HD + d);
            }
        }
    }
    float vv[ITM][2];
#pragma unroll
    for (int it = 0; it < ITM; it++) {
        const int idx = threadIdx.x + it * N_CONSUMERS;
        if (idx < total) {
            const int t = idx / np, pp = idx - t * np;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int k = 2 * (p0 + pp) + j, h = k / HD, d = k % HD;
                const float av = ll_wait(ra[it][j], P.attnl + (size_t)t * NQ * HD + k, e - 1);
                const float gv = ll_wait(rg[it][j], P.projl + (size_t)t * PROJ_MAX + h * 2 * HD + HD + d, e - 2);
                vv[it][j] = av * sigmoid(gv);
            }
        }
    }
    vdone_wait(vt);
#pragma unroll
    for (int it = 0; it < ITM; it++) {
        const int idx = threadIdx.x + it * N_CONSUMERS;
        if (idx < total) {
            const int t = idx / np, pp = idx - t * np;
            vf_put(vecf, 2 * (p0 + pp), t, vv[it][0], vv[it][1]);
        }
    }
    vready_sync(rt);
}

template <int INTER, int BT>
__device__ void prologue_act_s(uint32_t *vecf, uint32_t e, int nt, unsigned vt, unsigned rt) {
    constexpr int NPR = INTER / 2;
    constexpr int NPMAX = (NPR + 39) / 40;
    constexpr int ITM = (BT * NPMAX + N_CONSUMERS - 1) / N_CONSUMERS;
    int p0, np;
    slice_of(NPR, p0, np);
    const int total = nt * np;
    unsigned long long a[ITM][2];
#pragma unroll
    for (int it = 0; it < ITM; it++) {
        const int idx = threadIdx.x + it * N_CONSUMERS;
        if (idx < total) {
            const int t = idx / np, pp = idx - t * np;
            a[it][0] = ld_ll(P.actl + (size_t)t * INTER + 2 * (p0 + pp));
            a[it][1] = ld_ll(P.actl + (size_t)t * INTER + 2 * (p0 + pp) + 1);
        }
    }
    float v[ITM][2];
#pragma unroll
    for (int it = 0; it < ITM; it++) {
        const int idx = threadIdx.x + it * N_CONSUMERS;
        if (idx < total) {
            const int t = idx / np, pp = idx - t * np;
            v[it][0] = ll_wait(a[it][0], P.actl + (size_t)t * INTER + 2 * (p0 + pp), e - 1);
            v[it][1] = ll_wait(a[it][1], P.actl + (size_t)t * INTER + 2 * (p0 + pp) + 1, e - 1);
        }
    }
    vdone_wait(vt);
#pragma unroll
    for (int it = 0; it < ITM; it++) {
        const int idx = threadIdx.x + it * N_CONSUMERS;
        if (idx < total) {
            const int t = idx / np, pp = idx - t * np;
            vf_put(vecf, 2 * (p0 + pp), t, v[it][0], v[it][1]);
        }
    }
    vready_sync(rt);
}

// ------------------------------------------------------------------ core ops (per entry)
// Task = (entry, head, 32 value columns); state is [head][v][k] so a column is one 512-B row.
template <int BT>
__device__ void gdn_op_b(const q35_layer_ptrs_t &L, int nt, uint32_t e, float *sm,
                         const int *positions, const int *slots) {
    constexpr int CPW = 2;   // value columns per warp per task (32-column slices; 8 = one task per head was not faster)
    constexpr int COLS = N_CONSUMER_WARPS * CPW;   // value columns per task
    constexpr int SLICES = LIN_DV / COLS;
    static_assert(SLICES * COLS == LIN_DV && 256 + COLS < N_CONSUMERS, "gdn task shape");
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float *sqk = sm;
    float *sv = sm + 256;
    float *sc = sm + 256 + COLS;
    constexpr int RM = CONV_RING - 1;
    for (int task = blockIdx.x; task < nt * LIN_H * SLICES; task += gridDim.x) {
        const int en = task / (LIN_H * SLICES);
        const int sub = task % (LIN_H * SLICES);
        const int h = sub / SLICES, sl = sub % SLICES;
        const int pos = positions[en];
        const int sid = slots[en];
        float *st = L.state + (size_t)sid * L.state_stride;
        float *hist0 = L.conv_hist + (size_t)sid * L.conv_stride;
        const unsigned long long *projl = P.projl + (size_t)en * PROJ_MAX;
        unsigned long long *corel = P.corel + (size_t)en * LIN_V;
        if (t < 256 + COLS) {
            const int ch = t < 128 ? h * LIN_DK + t
                         : t < 256 ? LIN_QK + h * LIN_DK + (t - 128)
                                   : 2 * LIN_QK + h * LIN_DV + sl * COLS + (t - 256);
            float *hist = hist0 + ch;   // [ring][channel] layout, see hist_at
            float win[CONV_K];
            win[0] = __ldcg(hist + hist_at((pos - 3) & RM));
            win[1] = __ldcg(hist + hist_at((pos - 2) & RM));
            win[2] = __ldcg(hist + hist_at((pos - 1) & RM));
            const float4 wq = *(const float4 *)(L.conv_w + ch * CONV_K);
            const unsigned long long raw = ld_ll(projl + ch);
            win[3] = ll_wait(raw, projl + ch, e - 1);
            const float acc = fmaf(wq.w, win[3], fmaf(wq.z, win[2], fmaf(wq.y, win[1], wq.x * win[0])));
            if (t >= 256 || sl == 0) __stcg(hist + hist_at(pos & RM), win[3]);
            if (t < 256) sqk[t] = silu(acc); else sv[t - 256] = silu(acc);
        } else if (t < 256 + COLS + 1) {
            const unsigned long long *pb = projl + GDN_B_OFF + h;
            const unsigned long long *pa = projl + GDN_A_OFF + h;
            const unsigned long long rb = ld_ll(pb), ra = ld_ll(pa);
            const float beta = sigmoid(ll_wait(rb, pb, e - 1));
            const float a = ll_wait(ra, pa, e - 1);
            sc[0] = beta;
            sc[1] = __expf(L.ssm_a[h] * softplus(a + L.dt_bias[h]));
        }
        consumer_sync();
        if (warp < 2) {
            const int qk = warp;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) { const float v = sqk[qk * 128 + lane + 32 * i]; ss = fmaf(v, v, ss); }
            ss = warp_sum(ss);
            if (lane == 0) sc[2 + qk] = rsqrtf(ss + 1e-6f) * (qk == 0 ? rsqrtf((float)LIN_DK) : 1.f);
        }
        consumer_sync();
        // column c of this warp = warp + 16 * c (same per-column arithmetic for every CPW)
        float s[CPW][4];
        float *S[CPW];
#pragma unroll
        for (int c = 0; c < CPW; c++) {
            const size_t off = ((size_t)h * LIN_DV + sl * COLS + warp + 16 * c) * LIN_DK;
            S[c] = st + off;
        }
#pragma unroll
        for (int c = 0; c < CPW; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) s[c][i] = __ldcg(S[c] + lane + 32 * i);
        const float beta = sc[0], decay = sc[1], rq = sc[2], rk = sc[3];
        float kv[4], qv[4], km[CPW];
#pragma unroll
        for (int c = 0; c < CPW; c++) km[c] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int k = lane + 32 * i;
            kv[i] = sqk[128 + k] * rk;
            qv[i] = sqk[k] * rq;
        }
#pragma unroll
        for (int c = 0; c < CPW; c++) {
#pragma unroll
            for (int i = 0; i < 4; i++) { s[c][i] *= decay; km[c] = fmaf(s[c][i], kv[i], km[c]); }
            km[c] = warp_sum(km[c]);
        }
#pragma unroll
        for (int c = 0; c < CPW; c++) {
            float *dst = S[c];
            const float delta = (sv[warp + 16 * c] - km[c]) * beta;
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s[c][i] = fmaf(kv[i], delta, s[c][i]);
                __stcg(dst + lane + 32 * i, s[c][i]);
                o = fmaf(s[c][i], qv[i], o);
            }
            o = warp_sum(o);
            if (lane == 0) st_ll(corel + h * LIN_DV + sl * COLS + warp + 16 * c, o, e);
        }
        consumer_sync();
    }
}

// Batch path split rule: a function of the position only (never of BT), so batch invariance (G4)
// holds.  Chunks of at least ATT_B_CHUNK positions: a batch has enough (entry, kv-head) tasks
// to fill the grid, so many tiny splits only add per-task latency and merge passes.
constexpr int ATT_B_CHUNK = 128;
__device__ __forceinline__ void attn_split_b(int n, int &splits, int &chunk) {
    splits = min(MAX_SPLITS, (n + ATT_B_CHUNK - 1) / ATT_B_CHUNK);
    chunk = (n + splits - 1) / splits;
    splits = (n + chunk - 1) / chunk;
}
__device__ __forceinline__ bool attn_task_b(int task, int nt, const int *positions, int &en, int &g, int &sp, int &splits, int &chunk) {
    for (int t = 0; t < nt; t++) {
        int splits_t, chunk_t;
        attn_split_b(positions[t] + 1, splits_t, chunk_t);
        if (task < NKV * splits_t) { en = t; g = task / splits_t; sp = task % splits_t; splits = splits_t; chunk = chunk_t; return true; }
        task -= NKV * splits_t;
    }
    return false;
}

template <int BT>
__device__ void attn_op_b(const q35_layer_ptrs_t &L, int nt, uint32_t e, float *sm, float *red,
                          const int *positions, const int *slots) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float *sq = sm;
    float *sacc = sm + 1024;
    float *sml = sm + 2048;
    float *skv = sm + 2176;
    int *sflag = (int *)(red + 32);
    const size_t ms = (size_t)P.max_seq;
    // positions per warp with loads in flight together; a warp still visits t0+warp+16k in increasing k,
    // so the online-softmax order (and the result) does not depend on KVB
    constexpr int KVB = 1;   // 2 and 4 raise register pressure and slow every op (B=32: -3% / -14%)
    int en, g, sp, splits, chunk;
    for (int task = blockIdx.x; attn_task_b(task, nt, positions, en, g, sp, splits, chunk); task += gridDim.x) {
        const int pos = positions[en];
        const int sid = slots[en];
        const int n = pos + 1;
        const int t0 = sp * chunk, t1 = min(n, t0 + chunk);
        const int nb = pos, nn = max(0, t1 - nb);
        const __half *Kc = (const __half *)L.k_cache + (size_t)sid * L.kv_stride + (size_t)g * ms * HD + lane * 8;
        const __half *Vc = (const __half *)L.v_cache + (size_t)sid * L.kv_stride + (size_t)g * ms * HD + lane * 8;
        const unsigned long long *projl = P.projl + (size_t)en * PROJ_MAX;
        uint4 kq[KVB], vq[KVB];
#pragma unroll
        for (int b = 0; b < KVB; b++) {
            const int tt = t0 + warp + b * N_CONSUMER_WARPS;
            if (tt < min(t1, pos)) {
                kq[b] = __ldcg((const uint4 *)(Kc + (size_t)tt * HD));
                vq[b] = __ldcg((const uint4 *)(Vc + (size_t)tt * HD));
            }
        }
        if (warp < Q_PER_KV + nn) {
            const bool isq = warp < Q_PER_KV;
            const int i = warp - Q_PER_KV;
            const unsigned long long *src = isq
                ? projl + (g * Q_PER_KV + warp) * 2 * HD
                : projl + ATT_K_OFF + g * HD;
            const float *nw = isq ? L.q_norm : L.k_norm;
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
            float v[8], ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) { v[j] = ll_wait(raw[j], src + lane * 8 + j, e - 1); ss = fmaf(v[j], v[j], ss); }
            const float r = rsqrtf(warp_sum(ss) / HD + 1e-6f);
            float *dst = isq ? sq + warp * HD : skv + i * 2 * HD;
#pragma unroll
            for (int j = 0; j < 8; j++) dst[lane * 8 + j] = v[j] * r * nw[lane * 8 + j];
        } else if (warp >= Q_PER_KV + 1 && warp < Q_PER_KV + 1 + nn) {
            const int i = warp - Q_PER_KV - 1;
            const unsigned long long *src = projl + ATT_V_OFF + g * HD;
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
#pragma unroll
            for (int j = 0; j < 8; j++) skv[i * 2 * HD + HD + lane * 8 + j] = ll_wait(raw[j], src + lane * 8 + j, e - 1);
        }
        consumer_sync();
        if (t < (Q_PER_KV + nn) * 32) {
            const int hh = t >> 5, i = t & 31;
            float *b = hh < Q_PER_KV ? sq + hh * HD : skv + (hh - Q_PER_KV) * 2 * HD;
            const float c = P.rope_cos[pos * (ROT / 2) + i], s = P.rope_sin[pos * (ROT / 2) + i];
            const float x1 = b[i], x2 = b[i + 32];
            b[i] = fmaf(x1, c, -(x2 * s));
            b[i + 32] = fmaf(x2, c, x1 * s);
        }
        consumer_sync();
        if (nn > 0) {
            for (int i = 0; i < nn; i++) {
                const int d = t & (HD - 1);
                __half *cache = (t < HD) ? (__half *)L.k_cache : (__half *)L.v_cache;
                cache[(size_t)sid * L.kv_stride + ((size_t)g * ms + nb + i) * HD + d] = __float2half(skv[i * 2 * HD + t]);
            }
            consumer_sync();
        }
        float m[Q_PER_KV], l[Q_PER_KV], acc[Q_PER_KV][8];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            m[h] = -1e30f; l[h] = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[h][j] = 0.f;
        }
        for (int base = t0 + warp, b0 = 0; base < t1; base += KVB * N_CONSUMER_WARPS, b0++) {
            if (b0 > 0) {
#pragma unroll
                for (int b = 0; b < KVB; b++) {
                    const int tt = base + b * N_CONSUMER_WARPS;
                    if (tt < t1) {
                        kq[b] = __ldcg((const uint4 *)(Kc + (size_t)tt * HD));
                        vq[b] = __ldcg((const uint4 *)(Vc + (size_t)tt * HD));
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < KVB; b++) {
                const int tt = base + b * N_CONSUMER_WARPS;
                if (tt >= t1) break;
                if (tt >= pos) {
                    kq[b] = __ldcg((const uint4 *)(Kc + (size_t)tt * HD));
                    vq[b] = __ldcg((const uint4 *)(Vc + (size_t)tt * HD));
                }
                float kf[8], vf[8], sc[Q_PER_KV];
                unpack8(kq[b], kf);
                unpack8(vq[b], vf);
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) {
                    const float4 q0 = *(const float4 *)(sq + h * HD + lane * 8);
                    const float4 q1 = *(const float4 *)(sq + h * HD + lane * 8 + 4);
                    float d = q0.x * kf[0];
                    d = fmaf(q0.y, kf[1], d); d = fmaf(q0.z, kf[2], d); d = fmaf(q0.w, kf[3], d);
                    d = fmaf(q1.x, kf[4], d); d = fmaf(q1.y, kf[5], d); d = fmaf(q1.z, kf[6], d); d = fmaf(q1.w, kf[7], d);
                    sc[h] = d;
                }
#pragma unroll
                for (int o = 16; o; o >>= 1)
#pragma unroll
                    for (int h = 0; h < Q_PER_KV; h++) sc[h] += __shfl_xor_sync(~0u, sc[h], o);
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) sc[h] *= 0.0625f;
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) {
                    const float mn = fmaxf(m[h], sc[h]);
                    const float corr = __expf(m[h] - mn), p = __expf(sc[h] - mn);
                    l[h] = fmaf(l[h], corr, p);
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[h][j] = fmaf(p, vf[j], acc[h][j] * corr);
                    m[h] = mn;
                }
            }
        }
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < Q_PER_KV; h++) { sml[(warp * Q_PER_KV + h) * 2] = m[h]; sml[(warp * Q_PER_KV + h) * 2 + 1] = l[h]; }
        consumer_sync();
        float sc[Q_PER_KV];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            float M = -1e30f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + h) * 2]);
            sc[h] = __expf(m[h] - M);
        }
        float *const pg = P.attn_part + ((size_t)en * NKV + g) * MAX_SPLITS * Q_PER_KV * PART_STRIDE;
        float *part = pg + (size_t)sp * Q_PER_KV * PART_STRIDE;
        if (t < Q_PER_KV) {
            float M = -1e30f, Ls = 0.f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + t) * 2]);
            for (int w = 0; w < N_CONSUMER_WARPS; w++)
                Ls = fmaf(sml[(w * Q_PER_KV + t) * 2 + 1], __expf(sml[(w * Q_PER_KV + t) * 2] - M), Ls);
            __stcg(part + t * PART_STRIDE + HD, M);
            __stcg(part + t * PART_STRIDE + HD + 1, Ls);
        }
        consumer_sync();
        float *wbuf = sm;
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
#pragma unroll
            for (int j = 0; j < 8; j++) wbuf[warp * HD + lane * 8 + j] = acc[h][j] * sc[h];
            consumer_sync();
            if (t < HD) {
                float a = 0.f;
                for (int w = 0; w < N_CONSUMER_WARPS; w++) a += wbuf[w * HD + t];
                __stcg(part + h * PART_STRIDE + t, a);
            }
            consumer_sync();
        }
        unsigned *cnt = P.attn_cnt + en * NKV + g;
        if (t == 0) {
            __threadfence();
            *sflag = atomicAdd(cnt, 1u) == (unsigned)(splits - 1);
        }
        consumer_sync();
        if (*sflag) {
            __threadfence();
            float *sw = sacc;
            for (int i = t; i < splits * Q_PER_KV; i += N_CONSUMERS) {
                sw[i] = __ldcg(pg + i * PART_STRIDE + HD);
                sw[MAX_SPLITS * Q_PER_KV + i] = __ldcg(pg + i * PART_STRIDE + HD + 1);
            }
            consumer_sync();
            if (t < Q_PER_KV) {
                float M = -1e30f, den = 0.f;
                for (int p = 0; p < splits; p++) M = fmaxf(M, sw[p * Q_PER_KV + t]);
                for (int p = 0; p < splits; p++)
                    den = fmaf(__expf(sw[p * Q_PER_KV + t] - M), sw[MAX_SPLITS * Q_PER_KV + p * Q_PER_KV + t], den);
                for (int p = 0; p < splits; p++) sw[p * Q_PER_KV + t] = __expf(sw[p * Q_PER_KV + t] - M) / den;
            }
            consumer_sync();
            for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) {
                const int h = i / HD, d = i % HD;
                float v[MAX_SPLITS];
#pragma unroll
                for (int p = 0; p < MAX_SPLITS; p++)
                    v[p] = p < splits ? __ldcg(pg + (p * Q_PER_KV + h) * PART_STRIDE + d) : 0.f;
                float num = 0.f;
#pragma unroll
                for (int p = 0; p < MAX_SPLITS; p++) num = fmaf(p < splits ? sw[p * Q_PER_KV + h] : 0.f, v[p], num);
                st_ll(P.attnl + (size_t)en * NQ * HD + (g * Q_PER_KV + h) * HD + d, num, e);
            }
            if (t == 0) atomicExch(cnt, 0u);
        }
        consumer_sync();
    }
}

// ---------------- tensor-core GEMV ----------------
// The epilogue is called per (group, n-tile) with v = {r0c0, r0c1, r1c0, r1c1} where
// r0 = lane/4, r1 = r0+8, columns e = n*8 + (lane&3)*2 and e+1.

template <int BT, int GPW, int DEPTH, class Epi>
__device__ void gemv_direct_f(const void *Wv, const uint32_t *vecf, int n_groups, int n_kt, Epi epi) {
    constexpr int NT = (BT + 7) / 8;
    const uint8_t *W = (const uint8_t *)Wv;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g0 = (int)((long long)n_groups * blockIdx.x / gridDim.x);
    const int g1 = (int)((long long)n_groups * (blockIdx.x + 1) / gridDim.x);
    const int r0 = lane >> 2;

    for (int gb = g0; gb < g1; gb += GPW * N_CONSUMER_WARPS) {
        int g[GPW];
        bool act[GPW];
#pragma unroll
        for (int k = 0; k < GPW; k++) {
            g[k] = gb + warp + k * N_CONSUMER_WARPS;
            act[k] = g[k] < g1;
        }
        uint32_t p[DEPTH][GPW][4];
        float sc[DEPTH][GPW][2];
        auto load = [&](int u, int d) {
            const int t = u;
#pragma unroll
            for (int k = 0; k < GPW; k++) {
                if (!act[k]) continue;
                const uint8_t *gt = W + ((size_t)g[k] * n_kt + t) * GROUP_BYTES;
                p[d][k][0] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((0 * 32 + lane) * 4)));
                p[d][k][1] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((1 * 32 + lane) * 4)));
                p[d][k][2] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((2 * 32 + lane) * 4)));
                p[d][k][3] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((3 * 32 + lane) * 4)));
                sc[d][k][0] = __half2float(*reinterpret_cast<const __half *>(gt + 512 + 2 * r0));
                sc[d][k][1] = __half2float(*reinterpret_cast<const __half *>(gt + 512 + 2 * (r0 + 8)));
            }
        };
        for (int d = 0; d < DEPTH; d++) if (d < n_kt) load(d, d);

        float acc[GPW][NT][4];
#pragma unroll
        for (int k = 0; k < GPW; k++)
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[k][n][j] = 0.f;

        for (int u0 = 0; u0 < n_kt; u0 += DEPTH) {
#pragma unroll
            for (int d = 0; d < DEPTH; d++) {
                const int u = u0 + d;
                if (u >= n_kt) break;
                uint32_t b[NT][2][2];
#pragma unroll
                for (int n = 0; n < NT; n++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        b[n][h][0] = ld_vecf(vecf + (size_t)((u * 4 + 2 * h + 0) * Q35_NTMAX + n) * 32 + lane);
                        b[n][h][1] = ld_vecf(vecf + (size_t)((u * 4 + 2 * h + 1) * Q35_NTMAX + n) * 32 + lane);
                    }
#pragma unroll
                for (int k = 0; k < GPW; k++) {
                    if (!act[k]) continue;
                    float tmp[NT][4];
#pragma unroll
                    for (int n = 0; n < NT; n++)
#pragma unroll
                        for (int j = 0; j < 4; j++) tmp[n][j] = 0.f;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        uint32_t a0, a1, a2, a3;
                        cvt4(p[d][k][h * 2 + 0], a0, a1);
                        cvt4(p[d][k][h * 2 + 1], a2, a3);
#pragma unroll
                        for (int n = 0; n < NT; n++)
                            mma_m16n8k16(tmp[n], a0, a1, a2, a3, b[n][h][0], b[n][h][1]);
                    }
#pragma unroll
                    for (int n = 0; n < NT; n++) {
                        acc[k][n][0] = fmaf(tmp[n][0], sc[d][k][0], acc[k][n][0]);
                        acc[k][n][1] = fmaf(tmp[n][1], sc[d][k][0], acc[k][n][1]);
                        acc[k][n][2] = fmaf(tmp[n][2], sc[d][k][1], acc[k][n][2]);
                        acc[k][n][3] = fmaf(tmp[n][3], sc[d][k][1], acc[k][n][3]);
                    }
                }
                if (u + DEPTH < n_kt) load(u + DEPTH, d);
            }
        }
#pragma unroll
        for (int k = 0; k < GPW; k++) {
            if (!act[k]) continue;
#pragma unroll
            for (int n = 0; n < NT; n++) {
                float v[4] = {acc[k][n][0], acc[k][n][1], acc[k][n][2], acc[k][n][3]};
                epi(g[k], n, v);
            }
        }
    }
}

template <int BT, class Epi>
__device__ void gemv_cta_f(const void *Wv, const uint32_t *vecf, int n_groups, int n_kt,
                           char *ring, float *red, Epi epi) {
    constexpr int NT = (BT + 7) / 8;
    constexpr int STAGE = TC_CHUNKT * GROUP_BYTES;
    const uint8_t *W = (const uint8_t *)Wv;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g0 = (int)((long long)n_groups * blockIdx.x / gridDim.x);
    const int g1 = (int)((long long)n_groups * (blockIdx.x + 1) / gridDim.x);
    const int r0 = lane >> 2;
    const int nch = (n_kt + TC_CHUNKT - 1) / TC_CHUNKT;
    const int total = (g1 - g0) * nch;
    if (total == 0) return;
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    auto chunk_g = [&](int u) { return g0 + u / nch; };
    auto chunk_c = [&](int u) { return u % nch; };
    auto issue = [&](int u, int slot) {
        const int g = chunk_g(u), c = chunk_c(u);
        const int t0 = c * TC_CHUNKT;
        const int nt = n_kt - t0 < TC_CHUNKT ? n_kt - t0 : TC_CHUNKT;
        const uint8_t *src = W + ((size_t)g * n_kt + t0) * GROUP_BYTES;
        char *dst = ring + slot * STAGE;
        const int nb = nt * GROUP_BYTES;
        for (int i = threadIdx.x * 16; i < nb; i += N_CONSUMERS * 16) cp_async16(dst + i, src + i, policy);
        cp_commit();
    };
    for (int u = 0; u < TC_NS && u < total; u++) issue(u, u);

    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[n][j] = 0.f;
    uint32_t bb[2][NT][2][2];

    for (int u = 0; u < total; u++) {
        const int c = chunk_c(u);
        const int t0 = c * TC_CHUNKT;
        const int nt = n_kt - t0 < TC_CHUNKT ? n_kt - t0 : TC_CHUNKT;
        const char *slot = ring + (u % TC_NS) * STAGE;
        int ntl = 0;
        // E1: with one chunk per group (K <= 1024) every chunk uses the same k-tiles per warp, so the
        // activation fragments are loaded once per op instead of once per group.
        // X3: the activation loads do not depend on the ring, so they are issued before the ring wait and
        // the barrier; their L2 latency overlaps the wait instead of following it.
        auto load_bb = [&]() {
            if (nch > 1 || u == 0)
            for (int j = warp; j < nt; j += N_CONSUMER_WARPS) {
                const int t = t0 + j;
#pragma unroll
                for (int n = 0; n < NT; n++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        bb[ntl][n][h][0] = ld_vecf(vecf + (size_t)((t * 4 + 2 * h + 0) * Q35_NTMAX + n) * 32 + lane);
                        bb[ntl][n][h][1] = ld_vecf(vecf + (size_t)((t * 4 + 2 * h + 1) * Q35_NTMAX + n) * 32 + lane);
                    }
                ntl++;
            }
        };
        load_bb();
        if (u + TC_NS <= total) cp_wait<TC_NS - 1>();
        else cp_wait<0>();
        __syncthreads();
        ntl = 0;
        for (int j = warp; j < nt; j += N_CONSUMER_WARPS) {
            const char *gt = slot + j * GROUP_BYTES;
            const __half *s = reinterpret_cast<const __half *>(gt + 512);
            const float s0 = __half2float(s[r0]), s1 = __half2float(s[r0 + 8]);
            float tmp[NT][4];
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int q = 0; q < 4; q++) tmp[n][q] = 0.f;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t p0 = *reinterpret_cast<const uint32_t *>(gt + ((h * 2 + 0) * 32 + lane) * 4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t *>(gt + ((h * 2 + 1) * 32 + lane) * 4);
                uint32_t a0, a1, a2, a3;
                cvt4(p0, a0, a1);
                cvt4(p1, a2, a3);
#pragma unroll
                for (int n = 0; n < NT; n++)
                    mma_m16n8k16(tmp[n], a0, a1, a2, a3, bb[ntl][n][h][0], bb[ntl][n][h][1]);
            }
#pragma unroll
            for (int n = 0; n < NT; n++) {
                acc[n][0] = fmaf(tmp[n][0], s0, acc[n][0]);
                acc[n][1] = fmaf(tmp[n][1], s0, acc[n][1]);
                acc[n][2] = fmaf(tmp[n][2], s1, acc[n][2]);
                acc[n][3] = fmaf(tmp[n][3], s1, acc[n][3]);
            }
            ntl++;
        }
        __syncthreads();
        // E2: reduce all NT n-tiles of a group at once (tiles 0..1 in `red`, tiles 2..3 in the ring slot
        // just consumed), warp n sums tile n.  Same order (warps 0..15) as the serial version, so the
        // result is bitwise identical; 2 barriers per group instead of 2*NT.
        if (c == nch - 1) {
            const int g = chunk_g(u);
            float *slotf = reinterpret_cast<float *>(ring + (u % TC_NS) * STAGE);
#pragma unroll
            for (int n = 0; n < NT; n++) {
                float *rb = (n < 2 ? red + n * 2048 : slotf + (n - 2) * 2048) + (warp * 32 + lane) * 4;
                *reinterpret_cast<float4 *>(rb) = make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
            }
            __syncthreads();
            if (warp < NT) {
                const float *rbase = warp < 2 ? red + warp * 2048 : slotf + (warp - 2) * 2048;
                float fin[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int q = 0; q < N_CONSUMER_WARPS; q++) {
                    const float4 r4 = *reinterpret_cast<const float4 *>(rbase + (q * 32 + lane) * 4);
                    fin[0] += r4.x; fin[1] += r4.y; fin[2] += r4.z; fin[3] += r4.w;
                }
                epi(g, warp, fin);
            }
            __syncthreads();
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[n][j] = 0.f;
        }
        if (u + TC_NS < total) issue(u + TC_NS, u % TC_NS);
    }
}

// H-ks: k-split GEMV without a CTA ring.  Warp w owns k-tiles t = w + 16*i (i < KPW = n_kt/16) of every
// row group of this CTA -- exactly the tiles and the order it had in gemv_cta_f -- and streams their A
// fragments straight from global with a D-deep register pipeline that runs across group boundaries, so
// there is no barrier per 32-tile chunk.  The cross-warp reduction (same 0..15 order, warp n sums n-tile n)
// is double-buffered in the (now unused) ring smem: one barrier per group.  Bitwise identical to gemv_cta_f.
constexpr int KS_MIN_BT = 16;   // smallest BT that uses the k-split path
constexpr int KS_D = 2;         // register pipeline depth (3 and 8 were slower: spills)
template <int BT, int D, bool HOIST, class Epi>
__device__ void gemv_ks_f(const void *Wv, const uint32_t *vecf, int n_groups, int n_kt, char *rbuf, Epi epi) {
    constexpr int NT = (BT + 7) / 8;
    const uint8_t *W = (const uint8_t *)Wv;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g0 = (int)((long long)n_groups * blockIdx.x / gridDim.x);
    const int g1 = (int)((long long)n_groups * (blockIdx.x + 1) / gridDim.x);
    const int r0 = lane >> 2;
    const int KPW = n_kt / N_CONSUMER_WARPS;
    const int U = (g1 - g0) * KPW;
    if (U <= 0) return;
    auto ldb = [&](uint32_t (&b)[NT][2][2], int t) {
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                b[n][h][0] = ld_vecf(vecf + (size_t)((t * 4 + 2 * h + 0) * Q35_NTMAX + n) * 32 + lane);
                b[n][h][1] = ld_vecf(vecf + (size_t)((t * 4 + 2 * h + 1) * Q35_NTMAX + n) * 32 + lane);
            }
    };
    uint32_t bh[HOIST ? 2 : 1][NT][2][2];
    if constexpr (HOIST) {
        ldb(bh[0], warp);
        if (KPW > 1) ldb(bh[1], warp + N_CONSUMER_WARPS);
    }
    uint32_t pa[D][4];
    float ps[D][2];
    uint32_t pb[HOIST ? 1 : D][NT][2][2];
    auto load = [&](int u, int d) {
        const int gi = u / KPW, i = u - gi * KPW;
        const int t = warp + N_CONSUMER_WARPS * i;
        const uint8_t *gt = W + ((size_t)(g0 + gi) * n_kt + t) * GROUP_BYTES;
#pragma unroll
        for (int q = 0; q < 4; q++) pa[d][q] = __ldcs(reinterpret_cast<const uint32_t *>(gt + (q * 32 + lane) * 4));
        ps[d][0] = __half2float(*reinterpret_cast<const __half *>(gt + 512 + 2 * r0));
        ps[d][1] = __half2float(*reinterpret_cast<const __half *>(gt + 512 + 2 * (r0 + 8)));
        if constexpr (!HOIST) ldb(pb[d], t);
    };
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[n][j] = 0.f;
    auto mma_unit = [&](const uint32_t (&b)[NT][2][2], int d) {
        float tmp[NT][4];
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) tmp[n][q] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            uint32_t a0, a1, a2, a3;
            cvt4(pa[d][h * 2 + 0], a0, a1);
            cvt4(pa[d][h * 2 + 1], a2, a3);
#pragma unroll
            for (int n = 0; n < NT; n++) mma_m16n8k16(tmp[n], a0, a1, a2, a3, b[n][h][0], b[n][h][1]);
        }
#pragma unroll
        for (int n = 0; n < NT; n++) {
            acc[n][0] = fmaf(tmp[n][0], ps[d][0], acc[n][0]);
            acc[n][1] = fmaf(tmp[n][1], ps[d][0], acc[n][1]);
            acc[n][2] = fmaf(tmp[n][2], ps[d][1], acc[n][2]);
            acc[n][3] = fmaf(tmp[n][3], ps[d][1], acc[n][3]);
        }
    };
#pragma unroll
    for (int d = 0; d < D; d++) if (d < U) load(d, d);
    for (int u0 = 0; u0 < U; u0 += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            const int u = u0 + d;
            if (u >= U) break;
            const int gi = u / KPW, i = u - gi * KPW;
            if constexpr (HOIST) {
                if (i == 0) mma_unit(bh[0], d); else mma_unit(bh[HOIST ? 1 : 0], d);
            } else {
                mma_unit(pb[HOIST ? 0 : d], d);
            }
            if (u + D < U) load(u + D, d);
            if (i == KPW - 1) {
                float *buf = reinterpret_cast<float *>(rbuf) + (gi & 1) * (2048 * NT);
#pragma unroll
                for (int n = 0; n < NT; n++)
                    *reinterpret_cast<float4 *>(buf + n * 2048 + (warp * 32 + lane) * 4) =
                        make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
                __syncthreads();
                if (warp < NT) {
                    const float *rb = buf + warp * 2048;
                    float fin[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int q = 0; q < N_CONSUMER_WARPS; q++) {
                        const float4 r4 = *reinterpret_cast<const float4 *>(rb + (q * 32 + lane) * 4);
                        fin[0] += r4.x; fin[1] += r4.y; fin[2] += r4.z; fin[3] += r4.w;
                    }
                    epi(g0 + gi, warp, fin);
                }
#pragma unroll
                for (int n = 0; n < NT; n++)
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[n][j] = 0.f;
            }
        }
    }
    __syncthreads();   // the last group's reducing warps are done with rbuf before the caller moves on
}

// Dispatch once per op: large row counts use the direct-load path, small ones the CTA ring.
template <int BT, class Epi>
__device__ void gemv_frag(const void *W, const uint32_t *vecf, int n_groups, int n_kt,
                          char *ring, float *red, Epi epi) {
    if (n_groups >= 64 * GRID) {
        if constexpr (BT <= 16) gemv_direct_f<BT, 4, 2>(W, vecf, n_groups, n_kt, epi);
        else gemv_direct_f<BT, 2, 2>(W, vecf, n_groups, n_kt, epi);   // depth 4 spills more (LM head +6%)
    } else {
        // Both paths compute bitwise-identical results, so the choice may depend on BT (G4 holds).
        // Ring: more weight bytes in flight (wins at BT <= 8, -4..5%); k-split: no per-chunk barrier
        // (wins at BT >= 16: B=32 2B +14%, 0.8B +6%).  Deeper k-split pipelines spill.
        if constexpr (BT >= KS_MIN_BT) {
            if (n_kt <= 2 * N_CONSUMER_WARPS) gemv_ks_f<BT, KS_D, true>(W, vecf, n_groups, n_kt, ring, epi);
            else gemv_ks_f<BT, KS_D, false>(W, vecf, n_groups, n_kt, ring, epi);
        } else {
            gemv_cta_f<BT>(W, vecf, n_groups, n_kt, ring, red, epi);
        }
    }
    vdone_arrive();
}

// H5: warm L2 with the first Q35_PF_BYTES of this CTA's weight slice of the upcoming GEMV while the
// CTA is still waiting for that GEMV's inputs (the serial prologue leaves DRAM idle for ~10 us).
#ifndef Q35_PF_BYTES
#define Q35_PF_BYTES 49152   // 48 KB per CTA (1.9 MB of the 4 MB L2); 32/72/80/128 KB and "after the previous GEMV" were slower
#endif
__device__ __forceinline__ void w_prefetch(const void *Wv, int n_groups, int n_kt) {
#if Q35_PF_BYTES > 0
    const int g0 = (int)((long long)n_groups * blockIdx.x / gridDim.x);
    const int g1 = (int)((long long)n_groups * (blockIdx.x + 1) / gridDim.x);
    const char *p = (const char *)Wv + (size_t)g0 * n_kt * GROUP_BYTES;
    const size_t bytes = (size_t)(g1 - g0) * n_kt * GROUP_BYTES;
    const size_t n = bytes < (size_t)Q35_PF_BYTES ? bytes : (size_t)Q35_PF_BYTES;
    for (size_t o = (size_t)threadIdx.x * 128; o < n; o += (size_t)N_CONSUMERS * 128) prefetch_l2(p + o);
#else
    (void)Wv; (void)n_groups; (void)n_kt;
#endif
}

// scatter {r0c0,r0c1,r1c0,r1c1} of one n-tile into an epoch-tagged [entry][row] buffer
__device__ __forceinline__ void epi_rows(unsigned long long *base, int stride, int g,
                                         int n, const float v[4], int nt, uint32_t e, int lane) {
    const int r0 = g * 16 + (lane >> 2), r1 = r0 + 8;
    const int ee = n * 8 + (lane & 3) * 2;
    if (ee < nt) {
        st_ll(base + (size_t)ee * stride + r0, v[0], e);
        st_ll(base + (size_t)ee * stride + r1, v[2], e);
    }
    if (ee + 1 < nt) {
        st_ll(base + (size_t)(ee + 1) * stride + r0, v[1], e);
        st_ll(base + (size_t)(ee + 1) * stride + r1, v[3], e);
    }
}

template <int HID, int INTER, int BT>
__global__ void __launch_bounds__(N_CONSUMERS, 1) decode_batch_kernel(q35_args_t A) {
    constexpr int XS = (HID + GRID - 1) / GRID + 16;
    extern __shared__ __align__(1024) char smem[];
    char *ring = smem;                          // TC_NS * TC_CHUNKT * 544
    float *xs = (float *)(smem + TC_RING_BYTES);
    constexpr int XSB = ((BT * XS * 4 + 15) & ~15);
    float *red = (float *)(smem + TC_RING_BYTES + XSB);   // >= 4352 floats, 16-byte aligned
    const int t = threadIdx.x, lane = t & 31;
    const int nt = A.n_entries;
#ifdef Q35_DBG_STALL
    // watchdog test hook: with args.lm_from == 1 (engine env Q35_DBG_STALL_LAUNCH=N) CTA 5 exits at once,
    // so every other CTA times out waiting for its outputs
    if (A.lm_from == 1 && blockIdx.x == 5) return;
#endif
    const int gv0 = (int)((long long)(HID / 16) * blockIdx.x / gridDim.x);
    const int xr0 = 16 * gv0;
    __syncthreads();
    int op = 0;
    uint32_t e = A.epoch_base;
#ifdef Q35_DBG_GRIDSYNC
    // debug only: a grid-wide barrier at every op boundary (bisects cross-CTA races)
    auto op_end = [&]() { __threadfence(); cooperative_groups::this_grid().sync(); e++; op++;
                          if (A.prof && blockIdx.x == 0 && t == 0 && op < 255) A.prof[op] = globaltimer(); };
#else
    auto op_end = [&]() { __syncthreads(); e++; op++;
                          if (A.prof && blockIdx.x == 0 && t == 0 && op < 255) A.prof[op] = globaltimer(); };
#endif
    if (A.prof && blockIdx.x == 0 && t == 0) A.prof[0] = globaltimer();   // Q35_PROF: CTA 0 stamps every op end
    // ... and, in prof[128 + op], the end of each GEMV op's prologue (so prologue and GEMV+epilogue separate)
    auto pro_end = [&]() { if (A.prof && blockIdx.x == 0 && t == 0 && op <= 120) A.prof[128 + op] = globaltimer(); };
    uint32_t *vecf = (uint32_t *)P.vecf;
    unsigned rt = 0;   // g_vready value once every CTA has written its slice of the sliced prologues so far
    constexpr bool SLICED = (BT >= 2);   // BT=1: the exchange costs more than it saves; it uses the unsliced prologues
    unsigned vt = 0;   // g_vdone value that proves every CTA finished the previous GEMV's vecf reads
    auto gemv_b = [&](auto &&...args) { gemv_frag<BT>(args...); vt += GRID; };

    auto layer = [&](const q35_layer_ptrs_t &L, int l, bool first) {
        w_prefetch(L.w_in_f, (L.is_attn ? ATT_IN_ROWS : GDN_IN_ROWS) / 16, HID / 32);
        if (first) prologue_emb_b<HID>(vecf, red, (const unsigned char *)P.embed, A.forced, nt, L.in_norm);
        else if constexpr (SLICED) { prologue_rmsnorm_s<HID>(vecf, e - 1, e, L.in_norm, P.xl, nt, vt, rt + GRID); rt += GRID; }
        else prologue_rmsnorm_b<HID>(vecf, red, e - 1, L.in_norm, P.xl, nt, vt);
        pro_end();
        gemv_b(L.w_in_f, vecf, (L.is_attn ? ATT_IN_ROWS : GDN_IN_ROWS) / 16, HID / 32, ring, red,
                      [&](int g, int n, const float v[4]) {
                          epi_rows(P.projl, PROJ_MAX, g, n, v, nt, e, lane);
                      });
        op_end();
        // sm uses red[0..4095] (attention wbuf), so the merge flag lives past it
        if (L.is_attn) attn_op_b<BT>(L, nt, e, red, red + 4064, A.positions, A.slot_ids);
        else gdn_op_b<BT>(L, nt, e, red, A.positions, A.slot_ids);
        op_end();
        if (A.prof_step > 0 && op >= A.prof_step) return;
        w_prefetch(L.w_out_f, HID / 16, 2048 / 32);
        if constexpr (SLICED) {
            if (L.is_attn) prologue_attn_gate_s<BT>(vecf, e, nt, vt, rt + GRID);
            else prologue_gated_norm_s(vecf, L, e, nt, vt, rt + GRID);
            rt += GRID;
        } else {
            if (L.is_attn) prologue_attn_gate_b(vecf, e, nt, vt);
            else prologue_gated_norm_b(vecf, L, e, nt, vt);
        }
        pro_end();
        gemv_b(L.w_out_f, vecf, HID / 16, 2048 / 32, ring, red,
                      [&](int g, int n, const float v[4]) {
                          const int r0 = g * 16 + (lane >> 2), r1 = r0 + 8;
                          const int ee = n * 8 + (lane & 3) * 2;
                          const float vv[4] = {v[0], v[1], v[2], v[3]};
                          const int rr[4] = {r0, r0, r1, r1};
                          const int ee2[4] = {ee, ee + 1, ee, ee + 1};
                          for (int j = 0; j < 4; j++) {
                              if (ee2[j] >= nt) continue;
                              float x = first ? 0.f : xs[ee2[j] * XS + rr[j] - xr0];
                              if (first) x = embed_at<HID>((const unsigned char *)P.embed, A.forced[ee2[j]], rr[j]);
                              const float xn = x + vv[j];
                              xs[ee2[j] * XS + rr[j] - xr0] = xn;
                              st_ll(P.xl + (size_t)ee2[j] * HID + rr[j], xn, e);
                          }
                      });
        op_end();
        if (A.prof_step > 0 && op >= A.prof_step) return;
        w_prefetch(L.w_gu_f, INTER / 8, HID / 32);
        if constexpr (SLICED) { prologue_rmsnorm_s<HID>(vecf, e - 1, e, L.post_norm, P.xl, nt, vt, rt + GRID); rt += GRID; }
        else prologue_rmsnorm_b<HID>(vecf, red, e - 1, L.post_norm, P.xl, nt, vt);
        pro_end();
        gemv_b(L.w_gu_f, vecf, INTER / 8, HID / 32, ring, red,
                      [&](int g, int n, const float v[4]) {
                          const int u = g * 8 + (lane >> 2);
                          const int ee = n * 8 + (lane & 3) * 2;
                          if (ee < nt) st_ll(P.actl + (size_t)ee * INTER + u, silu(v[0]) * v[2], e);
                          if (ee + 1 < nt) st_ll(P.actl + (size_t)(ee + 1) * INTER + u, silu(v[1]) * v[3], e);
                      });
        op_end();
        if (A.prof_step > 0 && op >= A.prof_step) return;
        w_prefetch(L.w_down_f, HID / 16, INTER / 32);
        if constexpr (SLICED) { prologue_act_s<INTER, BT>(vecf, e, nt, vt, rt + GRID); rt += GRID; }
        else prologue_act_b<INTER>(vecf, e, nt, vt);
        pro_end();
        gemv_b(L.w_down_f, vecf, HID / 16, INTER / 32, ring, red,
                      [&](int g, int n, const float v[4]) {
                          const int r0 = g * 16 + (lane >> 2), r1 = r0 + 8;
                          const int ee = n * 8 + (lane & 3) * 2;
                          const float vv[4] = {v[0], v[1], v[2], v[3]};
                          const int rr[4] = {r0, r0, r1, r1};
                          const int ee2[4] = {ee, ee + 1, ee, ee + 1};
                          for (int j = 0; j < 4; j++) {
                              if (ee2[j] >= nt) continue;
                              const float xn = xs[ee2[j] * XS + rr[j] - xr0] + vv[j];
                              xs[ee2[j] * XS + rr[j] - xr0] = xn;
                              st_ll(P.xl + (size_t)ee2[j] * HID + rr[j], xn, e);
                          }
                      });
        op_end();
        if (A.prof_step > 0 && op >= A.prof_step) return;
    };

    for (int l = 0; l < N_LAYERS; l++) {
        layer(P.layer[l], l, l == 0);
        if (A.prof_step > 0) {
            if (threadIdx.x == 0) { A.out_info[0] = 1; A.out_info[1] = (int)e; if (g_wd_b) A.out_info[2] = g_wd_b; }
            return;
        }
    }

    // LM head: one weight pass, per-entry argmax (deterministic atomic max: pack_score is a
    // total order on (score, token)) and optional full logits.
    w_prefetch(P.embed_f, VOCAB / 16, HID / 32);
    if constexpr (SLICED) { prologue_rmsnorm_s<HID>(vecf, e - 1, e, P.final_norm, P.xl, nt, vt, rt + GRID); rt += GRID; }
    else prologue_rmsnorm_b<HID>(vecf, red, e - 1, P.final_norm, P.xl, nt, vt);
    unsigned long long lbest[BT][2];
#pragma unroll
    for (int n = 0; n < BT; n++) { lbest[n][0] = 0; lbest[n][1] = 0; }
    pro_end();
    gemv_b(P.embed_f, vecf, VOCAB / 16, HID / 32, ring, red,
                  [&](int g, int n, const float v[4]) {
                      const int r0 = g * 16 + (lane >> 2), r1 = r0 + 8;
                      const int ee = n * 8 + (lane & 3) * 2;
                      if (ee < nt) {
                          if (A.logits_out) { A.logits_out[(size_t)ee * VOCAB + r0] = v[0]; A.logits_out[(size_t)ee * VOCAB + r1] = v[2]; }
                          unsigned long long k = pack_score(v[0], r0);
                          lbest[n][0] = k > lbest[n][0] ? k : lbest[n][0];
                          k = pack_score(v[2], r1);
                          lbest[n][0] = k > lbest[n][0] ? k : lbest[n][0];
                      }
                      if (ee + 1 < nt) {
                          if (A.logits_out) { A.logits_out[(size_t)(ee + 1) * VOCAB + r0] = v[1]; A.logits_out[(size_t)(ee + 1) * VOCAB + r1] = v[3]; }
                          unsigned long long k = pack_score(v[1], r0);
                          lbest[n][1] = k > lbest[n][1] ? k : lbest[n][1];
                          k = pack_score(v[3], r1);
                          lbest[n][1] = k > lbest[n][1] ? k : lbest[n][1];
                      }
                  });
    __syncthreads();
    // merge the per-lane maxima into one per entry (atomic max is order independent)
    unsigned long long *sbest = (unsigned long long *)red;
    if (t < BT) sbest[t] = 0;
    __syncthreads();
#pragma unroll
    for (int n = 0; n < BT; n++) {
        const int ee = n * 8 + (lane & 3) * 2;
        if (ee < nt && lbest[n][0]) atomicMax(&sbest[ee], lbest[n][0]);
        if (ee + 1 < nt && lbest[n][1]) atomicMax(&sbest[ee + 1], lbest[n][1]);
    }
    __syncthreads();
    if (t < nt) {
        st_ll_raw(P.lml + (blockIdx.x * BT + t) * 2, (uint32_t)(sbest[t] >> 32), e);
        st_ll_raw(P.lml + (blockIdx.x * BT + t) * 2 + 1, (uint32_t)sbest[t], e);
    }
    op_end();
    if (t < nt) {
        unsigned long long key = 0;
        for (int b = 0; b < gridDim.x; b++) {
            const unsigned long long *p = P.lml + (b * BT + t) * 2;
            const unsigned long long a = ld_ll(p), c2 = ld_ll(p + 1);
            const unsigned long long kk = ((unsigned long long)ll_wait_raw(a, p, e - 1) << 32) |
                                          ll_wait_raw(c2, p + 1, e - 1);
            key = kk > key ? kk : key;
        }
        A.tokens_out[t] = key;
    }
    if (blockIdx.x == 0 && t == 0) {
        A.out_info[0] = 1;
        A.out_info[1] = (int)e;
        A.out_info[2] = g_wd_b;   // watchdog: an ll_wait timed out and returned stale data
        A.out_info[3] = 0;
        A.out_info[4] = g_wd_b_blk;
        A.out_info[5] = (int)g_wd_b_ep;
    }
}

// ------------------------------------------------------------------ host API
int smem_bytes_for(int hid) {
    return hid == 1024 ? K<1024, 3584>::SMEM_BYTES : K<2048, 6144>::SMEM_BYTES;
}

}  // namespace q35

extern "C" int q35_mk_smem_bytes(int hid) { return q35::smem_bytes_for(hid); }

extern "C" int q35_mk_set_params(const q35_params_t *p) {
    return (int)cudaMemcpyToSymbol(q35::P, p, sizeof(*p));
}

extern "C" int q35_mk_launch(const q35_args_t *a, int grid, void *stream) {
    static bool attr = false;
    const void *kernels[2] = {(const void *)q35::decode_kernel<1024, 3584>,
                              (const void *)q35::decode_kernel<2048, 6144>};
    const int smem[2] = {q35::K<1024, 3584>::SMEM_BYTES, q35::K<2048, 6144>::SMEM_BYTES};
    // P.hid is set by the engine; use the smem of the instantiation the caller asks for.
    const int which = (a->hid == 2048) ? 1 : 0;
    if (!attr) {
        for (int i = 0; i < 2; i++) {
            cudaError_t e = cudaFuncSetAttribute(kernels[i], cudaFuncAttributeMaxDynamicSharedMemorySize, smem[i]);
            if (e != cudaSuccess) return (int)e;
        }
        attr = true;
    }
    q35_args_t args = *a;
    void *kargs[] = {&args};
    return (int)cudaLaunchCooperativeKernel(kernels[which], dim3(grid), dim3(q35::N_THREADS), kargs,
                                            smem[which], (cudaStream_t)stream);
}

extern "C" int q35_mk_launch_batch(const q35_args_t *a, int bt, int grid, void *stream) {
    static bool attr[2][6] = {{false, false, false, false, false, false},
                              {false, false, false, false, false, false}};
    const void *kern[2][6] = {
        {(const void *)q35::decode_batch_kernel<1024, 3584, 1>,
         (const void *)q35::decode_batch_kernel<1024, 3584, 2>,
         (const void *)q35::decode_batch_kernel<1024, 3584, 4>,
         (const void *)q35::decode_batch_kernel<1024, 3584, 8>,
         (const void *)q35::decode_batch_kernel<1024, 3584, 16>,
         (const void *)q35::decode_batch_kernel<1024, 3584, 32>},
        {(const void *)q35::decode_batch_kernel<2048, 6144, 1>,
         (const void *)q35::decode_batch_kernel<2048, 6144, 2>,
         (const void *)q35::decode_batch_kernel<2048, 6144, 4>,
         (const void *)q35::decode_batch_kernel<2048, 6144, 8>,
         (const void *)q35::decode_batch_kernel<2048, 6144, 16>,
         (const void *)q35::decode_batch_kernel<2048, 6144, 32>},
    };
    const int which = (a->hid == 2048) ? 1 : 0;
    int ti;
    switch (bt) {
        case 1: ti = 0; break;
        case 2: ti = 1; break;
        case 4: ti = 2; break;
        case 8: ti = 3; break;
        case 16: ti = 4; break;
        case 32: ti = 5; break;
        default: return -100;
    }
    const int HID = (which == 0) ? 1024 : 2048;
    const int XS = (HID + q35::GRID - 1) / q35::GRID + 16;
    const int smem = q35::TC_RING_BYTES + ((bt * XS * 4 + 15) & ~15) + 4352 * 4;
    if (!attr[which][ti]) {
        cudaError_t e = cudaFuncSetAttribute(kern[which][ti], cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return (int)e;
        attr[which][ti] = true;
    }
    {
        // per-launch reset: the vecf arrival counters and the watchdog record (one copy)
        static const q35::BState init = {0, -1, 0, 0, 0, {0, 0, 0}};
        cudaError_t e = cudaMemcpyToSymbolAsync(q35::g_bs, &init, sizeof(init), 0, cudaMemcpyHostToDevice, (cudaStream_t)stream);
        if (e != cudaSuccess) return (int)e;
    }
    q35_args_t args = *a;
    void *kargs[] = {&args};
    return (int)cudaLaunchCooperativeKernel(kern[which][ti], dim3(grid), dim3(q35::N_CONSUMERS), kargs,
                                            smem, (cudaStream_t)stream);
}

// Watchdog record of the last launch: returns the kind (0 none, 1 mbarrier, 2 ll_wait, 3 vecf-arrival
// wait) and, if requested, the CTA and the epoch / counter target that timed out.
extern "C" int q35_mk_watchdog_fired(int *blk, unsigned *ep) {
    q35::BState b;
    if (cudaMemcpyFromSymbol(&b, q35::g_bs, sizeof(b)) != cudaSuccess) return -1;
    if (blk) *blk = b.wd_b_blk;
    if (ep) *ep = b.wd_b_ep;
    return b.wd_b;
}
