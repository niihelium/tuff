#include "cpu_ops.h"
#include "tuff_exact.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <immintrin.h>

static inline float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000U;
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = 0xE0U << 23;
    union { uint32_t i; float f; } u_norm, u_denorm;
    u_norm.i = (two_w >> 4) + exp_offset;
    const float normalized_value = u_norm.f * 0x1.0p-112f;
    const uint32_t magic_mask = 126U << 23;
    u_denorm.i = (two_w >> 17) | magic_mask;
    const float denormalized_value = u_denorm.f - 0.5f;
    const uint32_t denormalized_cutoff = 1U << 27;
    return (two_w < denormalized_cutoff) ?
        (sign ? -denormalized_value : denormalized_value) :
        (sign ? -normalized_value : normalized_value);
#endif
}

static inline float bf16_to_fp32(uint16_t u) {
    union { uint32_t i; float f; } val;
    val.i = (uint32_t)u << 16;
    return val.f;
}

static inline uint16_t fp32_to_fp16(float f) {
#if defined(__F16C__)
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    // Software round-to-nearest-even fp32->fp16 (fallback for non-F16C builds).
    union { uint32_t u; float f; } v; v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp  = (int32_t)((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp >= 31)     return (uint16_t)(sign | 0x7c00);            // inf/nan
    if (exp <= -10)    return (uint16_t)(sign);                     // subnormal->0
    if (exp <= 0) {                                                 // subnormal
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t m = mant >> shift;
        m += (mant >> (shift - 1)) & 1u;                            // round
        return (uint16_t)(sign | m);
    }
    uint32_t m = mant >> 13;
    uint32_t lsb = m & 1u;
    uint32_t rest = mant & 0x1fffu;
    uint32_t round = (rest > 0x1000u) || (rest == 0x1000u && lsb);
    m = (m + round) >> 0;
    if (m >= 0x400u) { m = 0; exp++; }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
#endif
}

// H22: f32 -> Q8_0 weight quantization, matching llama.cpp quantize_row_q8_0 (round to nearest,
// per-32-element block scale stored as fp16). Used to re-quantize the F32 router (ffn_gate_inp)
// at load time when TUFF_ROUTER_Q8 is set.
void quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t k) {
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i * QK8_0 + j];
            amax = fmaxf(amax, fabsf(v));
        }
        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i * QK8_0 + j] * id;
            y[i].qs[j] = (int8_t)roundf(x0);
        }
    }
}

void dequantize_q8_0(const block_q8_0 *x, float *y, int64_t k) {
    assert(k % QK8_0 == 0);
    int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i * QK8_0 + j] = x[i].qs[j] * d;
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_q4_K(const block_q4_K *x, float *y, int64_t k) {
    assert(k % QK_K == 0);
    int nb = (int)(k / QK_K);

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = fp16_to_fp32(x[i].d);
        float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;

            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_q6_K(const block_q6_K *x, float *y, int64_t k) {
    assert(k % QK_K == 0);
    int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row(ggml_type_t type, const void *src, float *dst, int64_t k) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, k * sizeof(float));
            break;
        case GGML_TYPE_F16: {
            const uint16_t *u16 = (const uint16_t*)src;
            for (int64_t i = 0; i < k; ++i) dst[i] = fp16_to_fp32(u16[i]);
            break;
        }
        case GGML_TYPE_BF16: {
            const uint16_t *u16 = (const uint16_t*)src;
            for (int64_t i = 0; i < k; ++i) dst[i] = bf16_to_fp32(u16[i]);
            break;
        }
        case GGML_TYPE_Q8_0:
            dequantize_q8_0((const block_q8_0*)src, dst, k);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_q4_K((const block_q4_K*)src, dst, k);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_q6_K((const block_q6_K*)src, dst, k);
            break;
        default:
            fprintf(stderr, "[FATAL ERROR] Unsupported GGML type %d in dequantize_row!\n", type);
            exit(EXIT_FAILURE);
    }
}

// Fused Q4_K mat-vec: dequantize + dot-product block-wise WITHOUT materializing the full
// fp32 row. Produces bit-identical results to the current dequantize_row(GGML_TYPE_Q4_K)
// followed by a separate dot loop, because each element's fp32 value is evaluated with the
// exact same operations (d*sc, min*m, nibble extraction) as dequantize_q4_K, and the dot is
// accumulated in the same increasing-index order (c = 0..in_dim-1) as the reference loop.
static void mat_vec_q4_K_p(float *dst, const block_q4_K *x, const float *src, int64_t n_rows, int64_t in_dim, int parallel) {
    int64_t nb = in_dim / QK_K;

    #pragma omp parallel for schedule(static) if(parallel)
    for (int64_t r = 0; r < n_rows; ++r) {
        const block_q4_K *row = x + r * nb;
        const float *srow = src;
        // Serial scalar sequential accumulation (increasing index), matching the single-thread
        // per-row CUDA kernel bit-for-bit (same mul->fma contraction via -mfma / --fmad). No
        // volatile: GCC does not reassociate float adds without -ffast-math, so the accumulator
        // stays a register-held scalar sum in the exact CPU order -- bit-identical, but far faster
        // than forcing a memory round-trip per element. See gpu.cu mul_mat_vec_q4_K_kernel.
        float sum = 0.0f;

        for (int64_t b = 0; b < nb; ++b) {
            const block_q4_K *blk = row + b;
            const uint8_t *q = blk->qs;
            float d   = fp16_to_fp32(blk->d);
            float min = fp16_to_fp32(blk->dmin);
            int is = 0;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                float d1 = d * sc; float m1 = min * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                float d2 = d * sc; float m2 = min * m;

                for (int l = 0; l < 32; ++l) sum += (d1 * (q[l] & 0xF) - m1) * srow[l];
                for (int l = 0; l < 32; ++l) sum += (d2 * (q[l] >> 4)  - m2) * srow[32 + l];
                q += 32; srow += 64; is += 2;
            }
        }
        dst[r] = sum;
    }
}

static void mat_vec_q6_K_scalar_p(float *dst, const block_q6_K *x, const float *src, int64_t n_rows, int64_t in_dim, int parallel) {
    int64_t nb = in_dim / QK_K;

    #pragma omp parallel for schedule(static) if(parallel)
    for (int64_t r = 0; r < n_rows; ++r) {
        const block_q6_K *row = x + r * nb;
        const float *srow = src;
        // Scalar sequential accumulation with NO fma contraction (product materialized in t, then
        // added), matching the CPU dequant+dot reference (vmulps then vaddss) and the CUDA kernel
        // bit-for-bit. sum is register-held (no volatile) for speed; GCC does not reassociate float
        // adds without -ffast-math. See gpu.cu mul_mat_vec_q6_K_scalar_kernel.
        float sum = 0.0f;

        for (int64_t b = 0; b < nb; ++b) {
            const block_q6_K *blk = row + b;
            const float d = fp16_to_fp32(blk->d);

            for (int c = 0; c < QK_K; ++c) {
                const int chunk = c / 128;
                const int p128  = c % 128;
                const int l     = p128 % 32;
                const int slot  = p128 / 32;
                const uint8_t ql = blk->ql[chunk * 64 + l + ((slot & 1) ? 32 : 0)];
                const uint8_t qh = blk->qh[chunk * 32 + l];
                const uint8_t qn = ((slot < 2) ? (ql & 0xF) : (ql >> 4)) |
                                   (((qh >> (slot * 2)) & 3) << 4);
                const int8_t q = (int8_t)(qn - 32);
                const int8_t s = blk->scales[chunk * 8 + (l / 16) + slot * 2];
                const float yv = d * (float)s * (float)q;
                // Separate multiply then add forces materialization (no fma contraction into sum),
                // matching the no-FMA vectorized reference (vmulps+vaddss) and the CUDA kernel.
                const float t = yv * srow[c];
                sum += t;
            }
            srow += QK_K;
        }
        dst[r] = sum;
    }
}

void mat_vec_q6_K_scalar(float *dst, const block_q6_K *x, const float *src, int64_t n_rows, int64_t in_dim) {
    mat_vec_q6_K_scalar_p(dst, x, src, n_rows, in_dim, 1);
}

void mat_vec_cpu(float *dst, const tensor_t *weight, const float *src) {
    if (!weight || !weight->data) {
        fprintf(stderr, "[FATAL ERROR] Null weight data in mat_vec_cpu!\n");
        exit(EXIT_FAILURE);
    }
    int64_t in_dim  = weight->info.dims[0];
    int64_t out_dim = weight->info.dims[1];
    ggml_type_t type = (ggml_type_t)weight->info.type;

    // Routed MoE experts (the hot path) are Q4_K: fuse dequant+accumulate so we never
    // materialize the full fp32 row. Other types keep the generic dequant-then-dot path.
    if (type == GGML_TYPE_Q4_K) {
        mat_vec_q4_K_p(dst, (const block_q4_K*)weight->data, src, out_dim, in_dim, 1);
        return;
    }

    size_t row_bytes = ggml_row_size(type, in_dim);
    const uint8_t *base_ptr = (const uint8_t*)weight->data;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < out_dim; ++r) {
        const void *row_data = base_ptr + r * row_bytes;
        float row_f32[4096];
        float *buf_ptr = row_f32;
        if (in_dim > 4096) {
            buf_ptr = (float*)malloc(in_dim * sizeof(float));
        }

        dequantize_row(type, row_data, buf_ptr, in_dim);

        float sum = 0.0f;
        for (int64_t c = 0; c < in_dim; ++c) {
            sum += buf_ptr[c] * src[c];
        }
        dst[r] = sum;

        if (in_dim > 4096) free(buf_ptr);
    }
}

// ============================================================================
// P2 -- AVX2 int8 CPU expert kernels
//
// Ported from llama.cpp @ 84e908c, MIT License (Copyright (c) 2023-2024 The ggml authors):
//   ggml/src/ggml-quants.c:2768             quantize_row_q8_K_ref
//   ggml/src/ggml-impl.h                    nearest_int
//   ggml/src/ggml-cpu/arch/x86/quants.c:43  hsum_float_8
//   ggml/src/ggml-cpu/arch/x86/quants.c:527 get_scale_shuffle_k4
//   ggml/src/ggml-cpu/arch/x86/quants.c:540 get_scale_shuffle
//   ggml/src/ggml-cpu/arch/x86/quants.c:2038 ggml_vec_dot_q4_K_q8_K (AVX2 branch)
//   ggml/src/ggml-cpu/arch/x86/quants.c:2426 ggml_vec_dot_q6_K_q8_K (AVX2 branch)
// Adapted to tuff's block structs (identical byte layout) and fp16 helper.
//
// These are NOT bit-exact against the scalar fp32 kernels above: the activation is
// quantized to 8 bits per 256-element super-block first. parity.md sec.3 retires
// bit-exactness precisely to unblock this. TUFF_EXACT=1 restores the scalar path.
// ============================================================================

// TUFF_EXACT=1 -> force the scalar fp32 kernels. Uses P0's shared helper
// (include/tuff_exact.h), which caches the env lookup in a static.
#define cpu_ops_exact() tuff_exact()

size_t q8_K_row_bytes(int64_t k) {
    assert(k % QK_K == 0);
    return (size_t)(k / QK_K) * sizeof(block_q8_K);
}

// ggml-impl.h nearest_int()
static inline int cpu_nearest_int(float fval) {
    if (isnan(fval)) return 0;
    if (fval > 127.0f) return 127;
    if (fval < -127.0f) return -127;
    return (int)lrintf(fval);
}

// ggml-quants.c quantize_row_q8_K_ref()
void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float max  = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }
        // -127 (not -128) as in llama.cpp: keeps the AVX implementations simple.
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = cpu_nearest_int(iscale * x[j]);
            y[i].qs[j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1 / iscale;
        x += QK_K;
    }
}

#if defined(__AVX2__)

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// horizontally add 8 floats  (x86/quants.c:43)
static inline float cpu_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// x86/quants.c:527 -- broadcast the i-th int16 scale pair across a 256-bit lane
static inline __m256i cpu_get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

// x86/quants.c:540 -- broadcast the i-th int8 scale across 8 bytes
static inline __m128i cpu_get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

// x86/quants.c:2038 ggml_vec_dot_q4_K_q8_K, AVX2 branch. n = number of elements.
static float cpu_vec_dot_q4_K_q8_K(int64_t n, const block_q4_K *x, const block_q8_K *y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    const int64_t nb = n / QK_K;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc   = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float d    =  y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        // sum_j mins[j] * (sum of the 32 q8 quants of sub-block j), folded in at the end
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s    = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                              _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod   = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, cpu_get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, cpu_get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l    = _mm256_and_si256(q4bits, m4);
            const __m256i q4h    = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    return cpu_hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

// Fused dual-weight dot product: loads q8x activation ONCE into AVX2 registers while computing
// both gate and up projections simultaneously. Bit-identical to calling cpu_vec_dot_q4_K_q8_K twice.
static inline void cpu_vec_dot_q4_K_q8_K_dual_weights(int64_t n,
                                                      const block_q4_K *gx,
                                                      const block_q4_K *ux,
                                                      const block_q8_K *y,
                                                      float *g_out, float *u_out) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    const int64_t nb = n / QK_K;
    uint32_t g_utmp[4], u_utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 g_acc   = _mm256_setzero_ps();
    __m128 g_acc_m = _mm_setzero_ps();
    __m256 u_acc   = _mm256_setzero_ps();
    __m128 u_acc_m = _mm_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float g_d    =  y[i].d * fp16_to_fp32(gx[i].d);
        const float g_dmin = -y[i].d * fp16_to_fp32(gx[i].dmin);
        const float u_d    =  y[i].d * fp16_to_fp32(ux[i].d);
        const float u_dmin = -y[i].d * fp16_to_fp32(ux[i].dmin);

        memcpy(g_utmp, gx[i].scales, 12);
        g_utmp[3] = ((g_utmp[2] >> 4) & kmask2) | (((g_utmp[1] >> 6) & kmask3) << 4);
        const uint32_t g_uaux = g_utmp[1] & kmask1;
        g_utmp[1] = (g_utmp[2] & kmask2) | (((g_utmp[0] >> 6) & kmask3) << 4);
        g_utmp[2] = g_uaux;
        g_utmp[0] &= kmask1;

        memcpy(u_utmp, ux[i].scales, 12);
        u_utmp[3] = ((u_utmp[2] >> 4) & kmask2) | (((u_utmp[1] >> 6) & kmask3) << 4);
        const uint32_t u_uaux = u_utmp[1] & kmask1;
        u_utmp[1] = (u_utmp[2] & kmask2) | (((u_utmp[0] >> 6) & kmask3) << 4);
        u_utmp[2] = u_uaux;
        u_utmp[0] &= kmask1;

        const uint8_t *g_q4 = gx[i].qs;
        const uint8_t *u_q4 = ux[i].qs;
        const int8_t  *q8   = y[i].qs;

        const __m256i g_mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int)g_utmp[3], (int)g_utmp[2], (int)g_utmp[1], (int)g_utmp[0]));
        const __m256i u_mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int)u_utmp[3], (int)u_utmp[2], (int)u_utmp[1], (int)u_utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s    = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                              _mm256_extracti128_si256(q8sums, 1));
        const __m128i g_prod   = _mm_madd_epi16(_mm256_extracti128_si256(g_mins_and_scales, 1), q8s);
        g_acc_m = _mm_fmadd_ps(_mm_set1_ps(g_dmin), _mm_cvtepi32_ps(g_prod), g_acc_m);

        const __m128i u_prod   = _mm_madd_epi16(_mm256_extracti128_si256(u_mins_and_scales, 1), q8s);
        u_acc_m = _mm_fmadd_ps(_mm_set1_ps(u_dmin), _mm_cvtepi32_ps(u_prod), u_acc_m);

        const __m128i g_sc128  = _mm256_extracti128_si256(g_mins_and_scales, 0);
        const __m256i g_scales = MM256_SET_M128I(g_sc128, g_sc128);

        const __m128i u_sc128  = _mm256_extracti128_si256(u_mins_and_scales, 0);
        const __m256i u_scales = MM256_SET_M128I(u_sc128, u_sc128);

        __m256i g_sumi = _mm256_setzero_si256();
        __m256i u_sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i sh0 = cpu_get_scale_shuffle_k4(2 * j + 0);
            const __m256i sh1 = cpu_get_scale_shuffle_k4(2 * j + 1);

            const __m256i g_scale_l = _mm256_shuffle_epi8(g_scales, sh0);
            const __m256i g_scale_h = _mm256_shuffle_epi8(g_scales, sh1);

            const __m256i u_scale_l = _mm256_shuffle_epi8(u_scales, sh0);
            const __m256i u_scale_h = _mm256_shuffle_epi8(u_scales, sh1);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i g_q4bits = _mm256_loadu_si256((const __m256i*)g_q4); g_q4 += 32;
            const __m256i g_q4l    = _mm256_and_si256(g_q4bits, m4);
            const __m256i g_q4h    = _mm256_and_si256(_mm256_srli_epi16(g_q4bits, 4), m4);

            __m256i g_p16l = _mm256_maddubs_epi16(g_q4l, q8l);
            g_p16l = _mm256_madd_epi16(g_scale_l, g_p16l);
            __m256i g_p16h = _mm256_maddubs_epi16(g_q4h, q8h);
            g_p16h = _mm256_madd_epi16(g_scale_h, g_p16h);
            g_sumi = _mm256_add_epi32(g_sumi, _mm256_add_epi32(g_p16l, g_p16h));

            const __m256i u_q4bits = _mm256_loadu_si256((const __m256i*)u_q4); u_q4 += 32;
            const __m256i u_q4l    = _mm256_and_si256(u_q4bits, m4);
            const __m256i u_q4h    = _mm256_and_si256(_mm256_srli_epi16(u_q4bits, 4), m4);

            __m256i u_p16l = _mm256_maddubs_epi16(u_q4l, q8l);
            u_p16l = _mm256_madd_epi16(u_scale_l, u_p16l);
            __m256i u_p16h = _mm256_maddubs_epi16(u_q4h, q8h);
            u_p16h = _mm256_madd_epi16(u_scale_h, u_p16h);
            u_sumi = _mm256_add_epi32(u_sumi, _mm256_add_epi32(u_p16l, u_p16h));
        }

        g_acc = _mm256_fmadd_ps(_mm256_set1_ps(g_d), _mm256_cvtepi32_ps(g_sumi), g_acc);
        u_acc = _mm256_fmadd_ps(_mm256_set1_ps(u_d), _mm256_cvtepi32_ps(u_sumi), u_acc);
    }

    g_acc_m = _mm_add_ps(g_acc_m, _mm_movehl_ps(g_acc_m, g_acc_m));
    g_acc_m = _mm_add_ss(g_acc_m, _mm_movehdup_ps(g_acc_m));

    u_acc_m = _mm_add_ps(u_acc_m, _mm_movehl_ps(u_acc_m, u_acc_m));
    u_acc_m = _mm_add_ss(u_acc_m, _mm_movehdup_ps(u_acc_m));

    *g_out = cpu_hsum_float_8(g_acc) + _mm_cvtss_f32(g_acc_m);
    *u_out = cpu_hsum_float_8(u_acc) + _mm_cvtss_f32(u_acc_m);
}

// x86/quants.c:2426 ggml_vec_dot_q6_K_q8_K, AVX2 branch.
static float cpu_vec_dot_q6_K_q8_K(int64_t n, const block_q6_K *x, const block_q8_K *y) {
    const int64_t nb = n / QK_K;

    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);

        const uint8_t *q4 = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;

        // The Q6_K quants are stored biased by +32; undo it once per block via the q8 bsums
        // instead of subtracting 32 from every quant (keeps the maddubs operands unsigned).
        const __m256i q8sums   = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales   = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, cpu_get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, cpu_get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, cpu_get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, cpu_get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc  = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return cpu_hsum_float_8(acc);
}

static void mat_vec_q4_K_q8K(float *dst, const block_q4_K *x, const block_q8_K *y,
                             int64_t n_rows, int64_t in_dim, int parallel) {
    const int64_t nb = in_dim / QK_K;
    int64_t r = 0;
    for (; r + 1 < n_rows; r += 2) {
        dst[r + 0] = cpu_vec_dot_q4_K_q8_K(in_dim, x + (r + 0) * nb, y);
        dst[r + 1] = cpu_vec_dot_q4_K_q8_K(in_dim, x + (r + 1) * nb, y);
    }
    for (; r < n_rows; ++r) {
        dst[r] = cpu_vec_dot_q4_K_q8_K(in_dim, x + r * nb, y);
    }
}

static void mat_vec_q6_K_q8K(float *dst, const block_q6_K *x, const block_q8_K *y,
                             int64_t n_rows, int64_t in_dim, int parallel) {
    const int64_t nb = in_dim / QK_K;
    int64_t r = 0;
    for (; r + 1 < n_rows; r += 2) {
        _mm_prefetch((const char*)(x + (r + 2) * nb), _MM_HINT_T0);
        _mm_prefetch((const char*)(x + (r + 3) * nb), _MM_HINT_T0);
        dst[r + 0] = cpu_vec_dot_q6_K_q8_K(in_dim, x + (r + 0) * nb, y);
        dst[r + 1] = cpu_vec_dot_q6_K_q8_K(in_dim, x + (r + 1) * nb, y);
    }
    for (; r < n_rows; ++r) {
        dst[r] = cpu_vec_dot_q6_K_q8_K(in_dim, x + r * nb, y);
    }
}

#endif // __AVX2__

void expert_mat_vec_cpu(float *dst, const void *w, ggml_type_t type,
                        int64_t in_dim, int64_t out_dim,
                        const float *src, const block_q8_K *q8src, int parallel) {
    if (!w || (!src && !q8src)) {
        fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu: null weight/activation!\n");
        exit(EXIT_FAILURE);
    }

#if defined(__AVX2__)
    const int fast = !cpu_ops_exact() && (in_dim % QK_K == 0) &&
                     (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K);
    if (fast) {
        const block_q8_K *y = q8src;
        block_q8_K *tmp = NULL;
        if (!y) {
            tmp = (block_q8_K*)malloc(q8_K_row_bytes(in_dim));
            if (!tmp) {
                fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu: out of memory!\n");
                exit(EXIT_FAILURE);
            }
            quantize_row_q8_K(src, tmp, in_dim);
            y = tmp;
        }
        if (type == GGML_TYPE_Q4_K)
            mat_vec_q4_K_q8K(dst, (const block_q4_K*)w, y, out_dim, in_dim, parallel);
        else
            mat_vec_q6_K_q8K(dst, (const block_q6_K*)w, y, out_dim, in_dim, parallel);
        free(tmp);
        return;
    }
#endif

    // Exact scalar fallback (TUFF_EXACT=1, non-AVX2 build, or an unsupported type).
    if (!src) {
        fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu: exact path needs the fp32 activation!\n");
        exit(EXIT_FAILURE);
    }
    if (type == GGML_TYPE_Q4_K) {
        mat_vec_q4_K_p(dst, (const block_q4_K*)w, src, out_dim, in_dim, parallel);
    } else if (type == GGML_TYPE_Q6_K) {
        mat_vec_q6_K_scalar_p(dst, (const block_q6_K*)w, src, out_dim, in_dim, parallel);
    } else {
        const size_t row_bytes = ggml_row_size(type, in_dim);
        const uint8_t *base_ptr = (const uint8_t*)w;
        #pragma omp parallel for schedule(static) if(parallel)
        for (int64_t r = 0; r < out_dim; ++r) {
            float *buf = (float*)malloc((size_t)in_dim * sizeof(float));
            dequantize_row(type, base_ptr + (size_t)r * row_bytes, buf, in_dim);
            float sum = 0.0f;
            for (int64_t c = 0; c < in_dim; ++c) sum += buf[c] * src[c];
            dst[r] = sum;
            free(buf);
        }
    }
}

void cpu_expert_gateup_swiglu(float *act, const void *gw, const void *uw,
                              const float *src, const block_q8_K *q8x,
                              int64_t in_dim, int64_t ffn_dim) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && (in_dim % QK_K == 0)) {
        const block_q4_K *gw_q4 = (const block_q4_K*)gw;
        const block_q4_K *uw_q4 = (const block_q4_K*)uw;
        const int64_t nb = in_dim / QK_K;

        const block_q8_K *y = q8x;
        block_q8_K *tmp = NULL;
        if (!y) {
            tmp = (block_q8_K*)malloc(q8_K_row_bytes(in_dim));
            quantize_row_q8_K(src, tmp, in_dim);
            y = tmp;
        }

        for (int64_t r = 0; r < ffn_dim; ++r) {
            _mm_prefetch((const char*)(gw_q4 + (r + 2) * nb), _MM_HINT_T0);
            _mm_prefetch((const char*)(uw_q4 + (r + 2) * nb), _MM_HINT_T0);
            float g = cpu_vec_dot_q4_K_q8_K(in_dim, gw_q4 + r * nb, y);
            float u = cpu_vec_dot_q4_K_q8_K(in_dim, uw_q4 + r * nb, y);
            act[r] = (g / (1.0f + expf(-g))) * u;
        }

        if (tmp) free(tmp);
        return;
    }
#endif

    // Fallback: separate gate, up and swiglu
    float *gate = (float*)malloc((size_t)ffn_dim * sizeof(float));
    float *up   = (float*)malloc((size_t)ffn_dim * sizeof(float));
    expert_mat_vec_cpu(gate, gw, GGML_TYPE_Q4_K, in_dim, ffn_dim, src, q8x, 0);
    expert_mat_vec_cpu(up,   uw, GGML_TYPE_Q4_K, in_dim, ffn_dim, src, q8x, 0);
    swiglu(act, gate, up, (int)ffn_dim);
    free(gate);
    free(up);
}

// H10: rows [r0, r1) of the same mat-vec, for splitting one expert across worker threads.
// Row r reads the same weights and the same q8_K activation and accumulates in the same order
// as it does in the full-range call, so the output is bit-identical.
void expert_mat_vec_cpu_rows(float *dst, const void *w, ggml_type_t type,
                             int64_t in_dim, int64_t out_dim,
                             const block_q8_K *q8src, int64_t r0, int64_t r1) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && q8src && (in_dim % QK_K == 0) &&
        (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K)) {
        if (r0 < 0 || r1 > out_dim || r0 > r1) {
            fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu_rows: bad range [%lld,%lld) of %lld\n",
                    (long long)r0, (long long)r1, (long long)out_dim);
            exit(EXIT_FAILURE);
        }
        const int64_t nb = in_dim / QK_K;
        if (type == GGML_TYPE_Q4_K)
            mat_vec_q4_K_q8K(dst + r0, (const block_q4_K*)w + r0 * nb, q8src, r1 - r0, in_dim, 0);
        else
            mat_vec_q6_K_q8K(dst + r0, (const block_q6_K*)w + r0 * nb, q8src, r1 - r0, in_dim, 0);
        return;
    }
#endif
    fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu_rows: chunked path requires AVX2, a q8_K "
                    "activation, in_dim %% 256 == 0 and Q4_K/Q6_K weights\n");
    exit(EXIT_FAILURE);
}

// H10: rows [r0, r1) of the fused gate/up + SwiGLU. Same argument: one output row is one pair of
// independent dot products plus a pointwise SwiGLU, so a row range is bit-identical.
void cpu_expert_gateup_swiglu_rows(float *act, const void *gw, const void *uw,
                                   const block_q8_K *q8x,
                                   int64_t in_dim, int64_t ffn_dim, int64_t r0, int64_t r1) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && q8x && (in_dim % QK_K == 0)) {
        if (r0 < 0 || r1 > ffn_dim || r0 > r1) {
            fprintf(stderr, "[FATAL ERROR] cpu_expert_gateup_swiglu_rows: bad range [%lld,%lld) of %lld\n",
                    (long long)r0, (long long)r1, (long long)ffn_dim);
            exit(EXIT_FAILURE);
        }
        const block_q4_K *gw_q4 = (const block_q4_K*)gw;
        const block_q4_K *uw_q4 = (const block_q4_K*)uw;
        const int64_t nb = in_dim / QK_K;
        static int s_cpu_dual = -1;
        if (s_cpu_dual < 0) {
            const char *env = getenv("TUFF_CPU_DUAL");
            s_cpu_dual = env ? atoi(env) : 1;
        }
        for (int64_t r = r0; r < r1; ++r) {
            _mm_prefetch((const char*)(gw_q4 + (r + 2) * nb), _MM_HINT_T0);
            _mm_prefetch((const char*)(uw_q4 + (r + 2) * nb), _MM_HINT_T0);
            float g, u;
            if (s_cpu_dual) {
                cpu_vec_dot_q4_K_q8_K_dual_weights(in_dim, gw_q4 + r * nb, uw_q4 + r * nb, q8x, &g, &u);
            } else {
                g = cpu_vec_dot_q4_K_q8_K(in_dim, gw_q4 + r * nb, q8x);
                u = cpu_vec_dot_q4_K_q8_K(in_dim, uw_q4 + r * nb, q8x);
            }
            act[r] = (g / (1.0f + expf(-g))) * u;
        }
        return;
    }
#endif
    fprintf(stderr, "[FATAL ERROR] cpu_expert_gateup_swiglu_rows: chunked path requires AVX2, a "
                    "q8_K activation and in_dim %% 256 == 0\n");
    exit(EXIT_FAILURE);
}

void expert_mat_vec_cpu_t(float *dst, const tensor_t *weight,
                          const float *src, const block_q8_K *q8src, int parallel) {
    if (!weight || !weight->data) {
        fprintf(stderr, "[FATAL ERROR] Null weight data in expert_mat_vec_cpu_t!\n");
        exit(EXIT_FAILURE);
    }
    expert_mat_vec_cpu(dst, weight->data, (ggml_type_t)weight->info.type,
                       weight->info.dims[0], weight->info.dims[1], src, q8src, parallel);
}

// H15 dual (shared-miss) kernels. Each output element is an independent dot product, so
// computing both columns from one row loop over the weights is bit-identical per column to
// the single-column calls; the shared weight row is touched once instead of twice.
void cpu_expert_gateup_swiglu_dual(float *act0, float *act1, const void *gw, const void *uw,
                                   const block_q8_K *q8x0, const block_q8_K *q8x1,
                                   int64_t in_dim, int64_t ffn_dim) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && (in_dim % QK_K == 0) && q8x0 && q8x1) {
        const block_q4_K *gw_q4 = (const block_q4_K*)gw;
        const block_q4_K *uw_q4 = (const block_q4_K*)uw;
        const int64_t nb = in_dim / QK_K;
        for (int64_t r = 0; r < ffn_dim; ++r) {
            _mm_prefetch((const char*)(gw_q4 + (r + 2) * nb), _MM_HINT_T0);
            _mm_prefetch((const char*)(uw_q4 + (r + 2) * nb), _MM_HINT_T0);
            float g0 = cpu_vec_dot_q4_K_q8_K(in_dim, gw_q4 + r * nb, q8x0);
            float u0 = cpu_vec_dot_q4_K_q8_K(in_dim, uw_q4 + r * nb, q8x0);
            act0[r] = (g0 / (1.0f + expf(-g0))) * u0;
            float g1 = cpu_vec_dot_q4_K_q8_K(in_dim, gw_q4 + r * nb, q8x1);
            float u1 = cpu_vec_dot_q4_K_q8_K(in_dim, uw_q4 + r * nb, q8x1);
            act1[r] = (g1 / (1.0f + expf(-g1))) * u1;
        }
        return;
    }
#endif
    // Fallback (exact / non-AVX2): two single-column passes, correct but no batching.
    cpu_expert_gateup_swiglu(act0, gw, uw, NULL, q8x0, in_dim, ffn_dim);
    cpu_expert_gateup_swiglu(act1, gw, uw, NULL, q8x1, in_dim, ffn_dim);
}

// Row-range form of the dual gate/up, for splitting a shared miss across workers.
void cpu_expert_gateup_swiglu_dual_rows(float *act0, float *act1, const void *gw, const void *uw,
                                        const block_q8_K *q8x0, const block_q8_K *q8x1,
                                        int64_t in_dim, int64_t ffn_dim, int64_t r0, int64_t r1) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && (in_dim % QK_K == 0) && q8x0 && q8x1) {
        if (r0 < 0 || r1 > ffn_dim || r0 > r1) {
            fprintf(stderr, "[FATAL ERROR] cpu_expert_gateup_swiglu_dual_rows: bad range [%lld,%lld) of %lld\n",
                    (long long)r0, (long long)r1, (long long)ffn_dim);
            exit(EXIT_FAILURE);
        }
        const block_q4_K *gw_q4 = (const block_q4_K*)gw;
        const block_q4_K *uw_q4 = (const block_q4_K*)uw;
        const int64_t nb = in_dim / QK_K;
        for (int64_t r = r0; r < r1; ++r) {
            _mm_prefetch((const char*)(gw_q4 + (r + 2) * nb), _MM_HINT_T0);
            _mm_prefetch((const char*)(uw_q4 + (r + 2) * nb), _MM_HINT_T0);
            float g0 = cpu_vec_dot_q4_K_q8_K(in_dim, gw_q4 + r * nb, q8x0);
            float u0 = cpu_vec_dot_q4_K_q8_K(in_dim, uw_q4 + r * nb, q8x0);
            act0[r] = (g0 / (1.0f + expf(-g0))) * u0;
            float g1 = cpu_vec_dot_q4_K_q8_K(in_dim, gw_q4 + r * nb, q8x1);
            float u1 = cpu_vec_dot_q4_K_q8_K(in_dim, uw_q4 + r * nb, q8x1);
            act1[r] = (g1 / (1.0f + expf(-g1))) * u1;
        }
        return;
    }
#endif
    fprintf(stderr, "[FATAL ERROR] cpu_expert_gateup_swiglu_dual_rows: chunked dual path requires "
                    "AVX2, a q8_K activation and in_dim %% 256 == 0\n");
    exit(EXIT_FAILURE);
}

void expert_mat_vec_cpu_dual(float *dst0, float *dst1, const void *w, ggml_type_t type,
                             int64_t in_dim, int64_t out_dim,
                             const block_q8_K *q8x0, const block_q8_K *q8x1) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && (in_dim % QK_K == 0) && q8x0 && q8x1 &&
        (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K)) {
        const int64_t nb = in_dim / QK_K;
        if (type == GGML_TYPE_Q4_K) {
            const block_q4_K *x = (const block_q4_K*)w;
            for (int64_t r = 0; r < out_dim; ++r) {
                dst0[r] = cpu_vec_dot_q4_K_q8_K(in_dim, x + r * nb, q8x0);
                dst1[r] = cpu_vec_dot_q4_K_q8_K(in_dim, x + r * nb, q8x1);
            }
        } else {
            const block_q6_K *x = (const block_q6_K*)w;
            for (int64_t r = 0; r < out_dim; ++r) {
                _mm_prefetch((const char*)(x + (r + 2) * nb), _MM_HINT_T0);
                dst0[r] = cpu_vec_dot_q6_K_q8_K(in_dim, x + r * nb, q8x0);
                dst1[r] = cpu_vec_dot_q6_K_q8_K(in_dim, x + r * nb, q8x1);
            }
        }
        return;
    }
#endif
    expert_mat_vec_cpu(dst0, w, type, in_dim, out_dim, NULL, q8x0, 0);
    expert_mat_vec_cpu(dst1, w, type, in_dim, out_dim, NULL, q8x1, 0);
}

// H10-style row range of the dual down, for splitting a shared miss across workers.
void expert_mat_vec_cpu_dual_rows(float *dst0, float *dst1, const void *w, ggml_type_t type,
                                  int64_t in_dim, int64_t out_dim,
                                  const block_q8_K *q8x0, const block_q8_K *q8x1,
                                  int64_t r0, int64_t r1) {
#if defined(__AVX2__)
    if (!cpu_ops_exact() && q8x0 && q8x1 && (in_dim % QK_K == 0) &&
        (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K)) {
        if (r0 < 0 || r1 > out_dim || r0 > r1) {
            fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu_dual_rows: bad range [%lld,%lld) of %lld\n",
                    (long long)r0, (long long)r1, (long long)out_dim);
            exit(EXIT_FAILURE);
        }
        const int64_t nb = in_dim / QK_K;
        if (type == GGML_TYPE_Q4_K) {
            const block_q4_K *x = (const block_q4_K*)w;
            for (int64_t r = r0; r < r1; ++r) {
                dst0[r] = cpu_vec_dot_q4_K_q8_K(in_dim, x + r * nb, q8x0);
                dst1[r] = cpu_vec_dot_q4_K_q8_K(in_dim, x + r * nb, q8x1);
            }
        } else {
            const block_q6_K *x = (const block_q6_K*)w;
            for (int64_t r = r0; r < r1; ++r) {
                _mm_prefetch((const char*)(x + (r + 2) * nb), _MM_HINT_T0);
                dst0[r] = cpu_vec_dot_q6_K_q8_K(in_dim, x + r * nb, q8x0);
                dst1[r] = cpu_vec_dot_q6_K_q8_K(in_dim, x + r * nb, q8x1);
            }
        }
        return;
    }
#endif
    fprintf(stderr, "[FATAL ERROR] expert_mat_vec_cpu_dual_rows: chunked dual path requires "
                    "AVX2, a q8_K activation, in_dim %% 256 == 0 and Q4_K/Q6_K weights\n");
    exit(EXIT_FAILURE);
}

void rms_norm(float *dst, const float *src, const float *weight, int size, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += src[i] * src[i];
    }
    float scale = 1.0f / sqrtf(sum / (float)size + eps);
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i] * scale * weight[i];
    }
}

void swiglu(float *dst, const float *gate, const float *up, int size) {
    for (int i = 0; i < size; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        dst[i] = silu_g * up[i];
    }
}

void silu(float *dst, const float *src, int size) {
    for (int i = 0; i < size; ++i) {
        float x = src[i];
        dst[i] = x / (1.0f + expf(-x));
    }
}

void sigmoid(float *dst, const float *src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

void softplus(float *dst, const float *src, int size) {
    for (int i = 0; i < size; ++i) {
        float x = src[i];
        dst[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
    }
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

void l2_norm(float *x, int size, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += x[i] * x[i];
    }
    float inv_norm = 1.0f / sqrtf(sum + eps);
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_norm;
    }
}

// Port of ggml_mrope_cache_init() / rotate_pairs() from
// llama.cpp/ggml/src/ggml-cpu/ops.cpp, specialised to:
//   - mode == GGML_ROPE_TYPE_IMROPE (is_imrope=true, indep_sects=false)
//   - ext_factor == 0, freq_scale == 1 (no yarn scaling; this model has none)
//   - a single token (pos_t == pos_h == pos_w == pos, pos_e == 0), which is exactly
//     what llama-graph.cpp does for text-only input under M-RoPE.
void apply_mrope(float *q, float *k, int n_heads_q, int n_heads_k, int head_dim, int n_rot, int pos,
                const uint32_t sections[4], float freq_base) {
    if (n_rot <= 0 || n_rot > head_dim) {
        fprintf(stderr, "[FATAL ERROR] apply_mrope: invalid n_rot=%d for head_dim=%d\n", n_rot, head_dim);
        exit(EXIT_FAILURE);
    }
    int half_rot = n_rot / 2;
    int sect_dims = (int)(sections[0] + sections[1] + sections[2] + sections[3]);
    if (sect_dims <= 0 || sect_dims > half_rot) {
        fprintf(stderr, "[FATAL ERROR] apply_mrope: invalid rope.dimension_sections sum=%d for n_rot/2=%d\n",
                sect_dims, half_rot);
        exit(EXIT_FAILURE);
    }

    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);

    // theta_t = theta_h = theta_w = pos (text-only mrope); theta_e unused since
    // sections[3] is 0 for this architecture, but kept for fidelity to llama.cpp.
    float theta_t = (float)pos;
    float theta_h = (float)pos;
    float theta_w = (float)pos;
    float theta_e = 0.0f;

    float *cos_cache = (float*)malloc(half_rot * sizeof(float));
    float *sin_cache = (float*)malloc(half_rot * sizeof(float));

    for (int p = 0; p < half_rot; ++p) {
        int sector = p % sect_dims;

        float theta;
        // GGML_ROPE_TYPE_IMROPE selection (ggml_mrope_cache_init, is_imrope branch):
        // round-robins pairs across t/h/w in that order, bounded by each section's count.
        if (sector % 3 == 1 && sector < 3 * (int)sections[1]) {
            theta = theta_h;
        } else if (sector % 3 == 2 && sector < 3 * (int)sections[2]) {
            theta = theta_w;
        } else if (sector % 3 == 0 && sector < 3 * (int)sections[0]) {
            theta = theta_t;
        } else {
            theta = theta_e;
        }

        cos_cache[p] = cosf(theta);
        sin_cache[p] = sinf(theta);

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    // NEOX-style pairing (rotate_pairs<T>(n_dims, n_dims/2, ...)): pair (p, p+half_rot).
    // Dims [n_rot, head_dim) are left untouched (ggml's "fill the remain channels").
    for (int h = 0; h < n_heads_q; ++h) {
        float *qh = q + h * head_dim;
        for (int p = 0; p < half_rot; ++p) {
            float c = cos_cache[p], s = sin_cache[p];
            float x0 = qh[p];
            float x1 = qh[p + half_rot];
            qh[p]            = x0 * c - x1 * s;
            qh[p + half_rot] = x0 * s + x1 * c;
        }
    }
    for (int h = 0; h < n_heads_k; ++h) {
        float *kh = k + h * head_dim;
        for (int p = 0; p < half_rot; ++p) {
            float c = cos_cache[p], s = sin_cache[p];
            float x0 = kh[p];
            float x1 = kh[p + half_rot];
            kh[p]            = x0 * c - x1 * s;
            kh[p + half_rot] = x0 * s + x1 * c;
        }
    }

    free(cos_cache);
    free(sin_cache);
}
