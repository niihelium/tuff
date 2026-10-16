#include "lfm25_model.h"
#include "lfm25_engine.h"
#include "cuda_check.h"
#include "cpu_ops.h"
#include "ggml_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Constants and Data Structures
// -----------------------------------------------------------------------------
#define QK8_1 32
#define QI8_1 8                  // 32 int8 quants = 8 ints
#define QI4_K 32                 // QK_K/(4*QR4_K)
#define QR4_K 2
#define QI6_K 32                 // QK_K/(4*QR6_K)
#define QR6_K 2
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1

#ifndef LFM_NORM_BLOCKS
#define LFM_NORM_BLOCKS 8
#endif
#ifndef LFM_CONV_WARPS
#define LFM_CONV_WARPS 24
#endif
#ifndef MMVQ_ROWS_PER_BLOCK
#define MMVQ_ROWS_PER_BLOCK 4
#endif

typedef struct {
    __half2 ds;
    int8_t  qs[QK8_1];
} block_q8_1_c;
GGML_STATIC_ASSERT(sizeof(block_q8_1_c) == 36, "wrong q8_1 block size/padding");

static __device__ __forceinline__ float h2f(uint16_t h) {
    __half raw = *reinterpret_cast<__half *>(&h);
    return __half2float(raw);
}

static __device__ __forceinline__ int get_int_b2_c(const void *x, int i32) {
    const uint16_t *x16 = (const uint16_t *)x; // q6_K blocks are 2-byte aligned
    return (int)((uint32_t)x16[2*i32 + 0] | ((uint32_t)x16[2*i32 + 1] << 16));
}

static __device__ __forceinline__ int get_int_b4_c(const void *x, int i32) {
    return ((const int *)x)[i32];
}

static __device__ __forceinline__ float warp_reduce_sum_f(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    return x;
}

// -----------------------------------------------------------------------------
// Dot Product Primitive: Q4_K x Q8_1
// -----------------------------------------------------------------------------
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_c(
        const block_q4_K * __restrict__ bq4_K, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int   v[2];
    int   u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int *q4 = (const int *)(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t *scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1_c *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int *q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0));
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    return h2f(bq4_K->d)*sumf_d - h2f(bq4_K->dmin)*sumf_m;
}

// -----------------------------------------------------------------------------
// Dot Product Primitive: Q6_K x Q8_1
// -----------------------------------------------------------------------------
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_c(
        const block_q6_K * __restrict__ bq6_K, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    const int bq8_offset   = 2*QR6_K*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4)*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2*((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2_c(bq6_K->ql, iqs);
    const int vh = get_int_b2_c(bq6_K->qh, (QI6_K/4)*(iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t *scales = bq6_K->scales + scale_offset;

    int   u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4_c(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return h2f(bq6_K->d) * sumf;
}

// -----------------------------------------------------------------------------
// CUDA Kernels: Matrix-Vector Multiply (MMVQ)
// -----------------------------------------------------------------------------
template <int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_mmvq_q4_K_dense_kernel(
        const block_q4_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI4_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q4_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
#pragma unroll 4
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q4_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        if (accum) dst[row] += tmp;
        else       dst[row]  = tmp;
    }
}

template <int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_mmvq_q6_K_dense_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
#pragma unroll 16
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        if (accum) dst[row] += tmp;
        else       dst[row]  = tmp;
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Fused RMSNorm + Quantize to Q8_1
// -----------------------------------------------------------------------------
__global__ void lfm_fused_rms_norm_quant_kernel(
    const float * __restrict__ x,
    const float * __restrict__ w,
    float * __restrict__ dst_norm_x,
    block_q8_1_c * __restrict__ y,
    int in_dim,
    float eps) {
    const int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        float xi = x[i];
        sum_sq += xi * xi;
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask, 32);
    }
    __shared__ float s_warp_sum[8];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    if (lane_id == 0) s_warp_sum[warp_id] = sum_sq;
    __syncthreads();

    __shared__ float s_scale;
    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) total_sq += s_warp_sum[k];
        s_scale = rsqrtf(total_sq / (float)in_dim + eps);
    }
    __syncthreads();

    const float scale = s_scale;
    const int total_q_blocks = in_dim / QK8_1;
    // Multi-block: every block recomputes the identical sum above; each handles only its own share of Q8_1 blocks.
    for (int b = blockIdx.x * 8 + warp_id; b < total_q_blocks; b += gridDim.x * 8) {
        const int idx = b * QK8_1 + lane_id;
        float norm_val = x[idx] * scale * (w ? w[idx] : 1.0f);
        if (dst_norm_x) dst_norm_x[idx] = norm_val;
        if (y) {
            float amax = fabsf(norm_val);
            float sum = norm_val;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
                sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
            }
            const float d = amax / 127.0f;
            const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(norm_val / d);
            y[b].qs[lane_id] = q;
            if (lane_id == 0) {
                y[b].ds = make_half2(__float2half(d), __float2half(sum));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Standalone Q8_1 Vector Quantization
// -----------------------------------------------------------------------------
__global__ void lfm_quantize_q8_1_kernel(const float * __restrict__ x, block_q8_1_c * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float xi = i < n ? x[i] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    if (i >= n) return;
    y[i / QK8_1].qs[i % QK8_1] = q;
    if (i % QK8_1 == 0) y[i / QK8_1].ds = make_half2(__float2half(d), __float2half(sum));
}

// -----------------------------------------------------------------------------
// CUDA Kernel: ShortConv1D Decode Step (Fused u = b*x, 3-tap conv, state update, gating, Q8_1 quant)
// -----------------------------------------------------------------------------
__global__ void lfm_shortconv_step_kernel(
    const float * __restrict__ in_proj,
    float * __restrict__ conv_state,
    const float * __restrict__ conv_w,
    float * __restrict__ conv_out,
    block_q8_1_c * __restrict__ conv_out_q8,
    int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float b_val = in_proj[i];
    float c_val = in_proj[d + i];
    float x_val = in_proj[2 * d + i];

    float u = b_val * x_val;

    float s0 = conv_state[i];
    float s1 = conv_state[d + i];

    float k0 = conv_w[i * 3 + 0];
    float k1 = conv_w[i * 3 + 1];
    float k2 = conv_w[i * 3 + 2];

    float y_conv = s0 * k0 + s1 * k1 + u * k2;

    // Rolling state update: S0 <- S1, S1 <- u
    conv_state[i] = s1;
    conv_state[d + i] = u;

    float g = c_val * y_conv;
    conv_out[i] = g;

    // Fused quantization of g to q8_1
    if (conv_out_q8) {
        const int lane_id = threadIdx.x & 31;
        const int b = i >> 5;
        float amax = fabsf(g);
        float sum  = g;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float scale = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(g / scale);
        conv_out_q8[b].qs[lane_id] = q;
        if (lane_id == 0) {
            conv_out_q8[b].ds = make_half2(__float2half(scale), __float2half(sum));
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Fused in_proj (Q4_K MMVQ) + ShortConv1D step + gating + Q8_1 quant.
// A block owns 32 channels; computes the 96 in_proj rows [b | c | x] it needs with the exact per-row MMVQ math
// (bit-identical to lfm_mmvq_q4_K_dense_kernel), keeps them in shared memory, then runs the conv step unchanged.
// -----------------------------------------------------------------------------
template <int NW>
__launch_bounds__(NW * 32, 1)
__global__ void lfm_inproj_shortconv_kernel(
    const block_q4_K * __restrict__ w,
    const block_q8_1_c * __restrict__ y,
    float * __restrict__ conv_state,
    const float * __restrict__ conv_w,
    float * __restrict__ conv_out,
    block_q8_1_c * __restrict__ conv_out_q8,
    int d, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI4_K;

    __shared__ float s_proj[96];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int i0 = blockIdx.x * 32;
    const int kqs = vdr * (lane % tpsb);

    for (int r = warp; r < 96; r += NW) {
        const int row = (r >> 5) * d + i0 + (r & 31);
        const block_q4_K * __restrict__ rw = w + (size_t)row * nblk;
        float tmp = 0.0f;
        for (int kbx = lane / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            tmp += vec_dot_q4_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
        }
        tmp = warp_reduce_sum_f(tmp);
        if (lane == 0) s_proj[r] = tmp;
    }
    __syncthreads();
    if (warp != 0) return;

    const int i = i0 + lane;
    float b_val = s_proj[lane];
    float c_val = s_proj[32 + lane];
    float x_val = s_proj[64 + lane];

    float u = b_val * x_val;

    float s0 = conv_state[i];
    float s1 = conv_state[d + i];

    float k0 = conv_w[i * 3 + 0];
    float k1 = conv_w[i * 3 + 1];
    float k2 = conv_w[i * 3 + 2];

    float y_conv = s0 * k0 + s1 * k1 + u * k2;

    conv_state[i] = s1;
    conv_state[d + i] = u;

    float g = c_val * y_conv;
    conv_out[i] = g;

    if (conv_out_q8) {
        const int b = i >> 5;
        float amax = fabsf(g);
        float sum  = g;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float scale = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(g / scale);
        conv_out_q8[b].qs[lane] = q;
        if (lane == 0) {
            conv_out_q8[b].ds = make_half2(__float2half(scale), __float2half(sum));
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: QK Per-Head RMSNorm + NeoX RoPE + K Cache Write
// -----------------------------------------------------------------------------
__global__ void lfm_fused_qk_norm_neox_rope_kernel(
    float * __restrict__ q,
    float * __restrict__ k,
    const float * __restrict__ q_norm_w,
    const float * __restrict__ k_norm_w,
    float * __restrict__ k_cache,
    int n_head, int n_head_kv, int head_dim,
    const int * __restrict__ d_pos, int max_seq, float freq_base, float eps, int layer_idx) {
    const int pos = *d_pos;
    const int head_idx = blockIdx.x; // 0..39: 0..31 is Q, 32..39 is K
    const int j = threadIdx.x;        // 0..63

    const bool is_k = (head_idx >= n_head);
    const int local_head = is_k ? (head_idx - n_head) : head_idx;
    float *data_ptr = is_k ? (k + local_head * head_dim) : (q + local_head * head_dim);
    const float *w_norm = is_k ? k_norm_w : q_norm_w;

    float val = data_ptr[j];
    float val_sq = val * val;

    // 64-thread reduction (2 warps)
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val_sq += __shfl_xor_sync(0xffffffff, val_sq, mask, 32);
    }
    __shared__ float s_sum[2];
    if ((j & 31) == 0) s_sum[j >> 5] = val_sq;
    __syncthreads();

    float total_sq = s_sum[0] + s_sum[1];
    float inv_rms = rsqrtf(total_sq / (float)head_dim + eps);

    float norm_val = val * inv_rms * w_norm[j];

    // NeoX RoPE (exchange halves [0..31] and [32..63])
    __shared__ float s_norm[64];
    s_norm[j] = norm_val;
    __syncthreads();

    const int j_half = j % 32;
    float theta = powf(freq_base, -2.0f * (float)j_half / (float)head_dim);
    float alpha = (float)pos * theta;
    float cos_v = cosf(alpha);
    float sin_v = sinf(alpha);

    float rot_val;
    if (j < 32) {
        float x0 = s_norm[j];
        float x1 = s_norm[j + 32];
        rot_val = x0 * cos_v - x1 * sin_v;
    } else {
        float x0 = s_norm[j - 32];
        float x1 = s_norm[j];
        rot_val = x0 * sin_v + x1 * cos_v;
    }

    data_ptr[j] = rot_val;

    // If K head, also write to KV cache
    if (is_k) {
        size_t cache_idx = (size_t)layer_idx * (size_t)(n_head_kv * max_seq * head_dim) +
                           (size_t)local_head * (size_t)(max_seq * head_dim) +
                           (size_t)pos * (size_t)head_dim + (size_t)j;
        k_cache[cache_idx] = rot_val;
    }
}

__global__ void lfm_v_cache_write_kernel(
    const float * __restrict__ v,
    float * __restrict__ v_cache,
    int n_head_kv, int head_dim, const int * __restrict__ d_pos, int max_seq, int layer_idx) {
    const int pos = *d_pos;
    const int k = blockIdx.x; // 0..7
    const int j = threadIdx.x; // 0..63
    if (k >= n_head_kv || j >= head_dim) return;

    float val = v[k * head_dim + j];
    size_t cache_idx = (size_t)layer_idx * (size_t)(n_head_kv * max_seq * head_dim) +
                       (size_t)k * (size_t)(max_seq * head_dim) +
                       (size_t)pos * (size_t)head_dim + (size_t)j;
    v_cache[cache_idx] = val;
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Flash-Decoding Causal GQA Attention + Q8_1 Quantization
// -----------------------------------------------------------------------------
__global__ void lfm_flash_decoding_kernel(
    const float * __restrict__ q,
    const float * __restrict__ k_cache,
    const float * __restrict__ v_cache,
    float * __restrict__ out,
    block_q8_1_c * __restrict__ out_q8,
    int n_head, int n_head_kv, int head_dim, const int * __restrict__ d_pos, int max_seq, int layer_idx) {
    const int pos = *d_pos;
    const int h = blockIdx.x; // query head: 0..31
    const int j = threadIdx.x; // 0..63
    if (h >= n_head || j >= head_dim) return;

    const int k_head = h / (n_head / n_head_kv); // GQA: e.g. h / 4 for 32/8
    const float q_val = q[h * head_dim + j];
    const float scale = 1.0f / sqrtf((float)head_dim);

    const size_t head_cache_base = (size_t)layer_idx * (size_t)(n_head_kv * max_seq * head_dim) +
                                   (size_t)k_head * (size_t)(max_seq * head_dim);

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_j = 0.0f;

    __shared__ float s_dot[2];

    for (int t = 0; t <= pos; ++t) {
        size_t slot_idx = head_cache_base + (size_t)t * (size_t)head_dim + (size_t)j;
        float k_val = k_cache[slot_idx];
        float dot = q_val * k_val;

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, mask, 32);
        }
        if ((j & 31) == 0) s_dot[j >> 5] = dot;
        __syncthreads();

        float total_dot = s_dot[0] + s_dot[1];
        float score = total_dot * scale;

        float m_curr = fmaxf(m_prev, score);
        float alpha = expf(m_prev - m_curr);
        float beta = expf(score - m_curr);
        l_prev = l_prev * alpha + beta;

        float v_val = v_cache[slot_idx];
        o_j = o_j * alpha + beta * v_val;
        m_prev = m_curr;
        __syncthreads();
    }

    float out_val = (l_prev > 0.0f) ? (o_j / l_prev) : 0.0f;
    out[h * head_dim + j] = out_val;

    // Quantize out_val to q8_1
    if (out_q8) {
        const int lane_id = threadIdx.x & 31;
        const int b = (h * head_dim + j) >> 5;
        float amax = fabsf(out_val);
        float sum = out_val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float d = amax / 127.0f;
        const int8_t q8_q = (amax == 0.0f) ? 0 : (int8_t)roundf(out_val / d);
        out_q8[b].qs[lane_id] = q8_q;
        if (lane_id == 0) {
            out_q8[b].ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Fused Gate+Up MMVQ Q4_K with Epilogue SwiGLU
// -----------------------------------------------------------------------------
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_fused_swiglu_mmvq_q4_K_kernel(
    const block_q4_K * __restrict__ wg,
    const block_q4_K * __restrict__ wu,
    const block_q8_1_c * __restrict__ y,
    float * __restrict__ d_act,
    int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI4_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q4_K * __restrict__ rwg = wg + (size_t)row * nblk;
    const block_q4_K * __restrict__ rwu = wu + (size_t)row * nblk;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tg = 0.0f;
    float tu = 0.0f;

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c *yb = y + kbx * (QK_K / QK8_1);
        tg += vec_dot_q4_K_q8_1_c(rwg + kbx, yb, kqs);
        tu += vec_dot_q4_K_q8_1_c(rwu + kbx, yb, kqs);
    }

    tg = warp_reduce_sum_f(tg);
    tu = warp_reduce_sum_f(tu);

    if (threadIdx.x == 0) {
        float silu_g = tg / (1.0f + expf(-tg));
        d_act[row] = silu_g * tu;
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: LM Head MMVQ Q6_K + Argmax Reduction
// -----------------------------------------------------------------------------
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_mmvq_q6_K_head_argmax_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        float * __restrict__ block_max,
        int * __restrict__ block_idx,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    const bool valid = (row < out_dim);

    const block_q6_K * __restrict__ rw = valid ? (w + (size_t)row * nblk) : w;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    if (valid) {
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
        }
    }
    tmp = warp_reduce_sum_f(tmp);

    __shared__ float s_max[rows_per_block];
    __shared__ int   s_idx[rows_per_block];

    if (threadIdx.x == 0) {
        if (valid && dst) dst[row] = tmp;
        s_max[threadIdx.y] = valid ? tmp : -1e30f;
        s_idx[threadIdx.y] = valid ? row : 0;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float m = s_max[0];
        int idx = s_idx[0];
        for (int r = 1; r < rows_per_block; ++r) {
            if (s_max[r] > m) {
                m = s_max[r];
                idx = s_idx[r];
            }
        }
        block_max[blockIdx.x] = m;
        block_idx[blockIdx.x] = idx;
    }
}

__global__ void lfm_argmax_reduction_kernel(
    const float * __restrict__ block_max,
    const int * __restrict__ block_idx,
    int * __restrict__ best_token,
    int n_blocks) {
    float max_val = -1e30f;
    int best_idx = 0;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        float val = block_max[i];
        if (val > max_val) {
            max_val = val;
            best_idx = block_idx[i];
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, max_val, mask, 32);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, mask, 32);
        if (other_max > max_val) {
            max_val = other_max;
            best_idx = other_idx;
        }
    }
    __shared__ float s_max[8];
    __shared__ int s_idx[8];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (lane_id == 0) {
        s_max[warp_id] = max_val;
        s_idx[warp_id] = best_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float gm = s_max[0];
        int gi = s_idx[0];
        for (int w = 1; w < 8; ++w) {
            if (s_max[w] > gm) {
                gm = s_max[w];
                gi = s_idx[w];
            }
        }
        *best_token = gi;
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Q6_K token-embedding row dequant (token id read from device memory)
// Mirrors dequantize_q6_K (cpu_ops.c) element-for-element: y = (d * sc) * q, two separate fp32 multiplies.
// One block per 256-element super-block, one thread per element.
// -----------------------------------------------------------------------------
__global__ void lfm_embed_q6_K_kernel(const uint8_t * __restrict__ base, const int * __restrict__ d_tokpos,
                                      size_t row_bytes, float * __restrict__ y) {
    const block_q6_K *x = (const block_q6_K *)(base + (size_t)d_tokpos[0] * row_bytes) + blockIdx.x;
    const int e = threadIdx.x;       // 0..255
    const int n = e >> 7;            // which 128-half
    const int r = e & 127;
    const int l = r & 31;
    const int g = r >> 5;            // 0..3
    const uint8_t *ql = x->ql + n * 64;
    const uint8_t *qh = x->qh + n * 32;
    const int8_t  *sc = x->scales + n * 8;
    const uint8_t lo = (g & 1) ? ql[l + 32] : ql[l];
    const int nib = (g < 2) ? (lo & 0xF) : (lo >> 4);
    const int q = (int)(int8_t)(nib | (((qh[l] >> (2 * g)) & 3) << 4)) - 32;
    const float d = h2f(x->d);
    y[blockIdx.x * QK_K + e] = __fmul_rn(__fmul_rn(d, (float)sc[(l >> 4) + 2 * g]), (float)(int8_t)q);
}

// =============================================================================
// Batched (multi-token) variants for speculative-decoding verification (Task 5.1).
//
// Contract: for every (row, column b) the arithmetic is exactly the single-token kernel's - the same lane-level
// accumulation `tmp += dot(block kbx, y_b, kqs)` over the same kbx in the same order, then the same
// warp_reduce_sum_f, then the same store - so column b is bit-identical to a single-token launch on y_b.
// What changes is only that each weight block is loaded once per lane and applied to all NB activation vectors.
// The weight-only part of each dot product is split out (wpart) from the activation part; the arithmetic
// expressions are copied verbatim from vec_dot_q4_K_q8_1_c / vec_dot_q6_K_q8_1_c.
// =============================================================================
struct q4k_wpart {
    int     v0i[QR4_K];     // low/high-nibble ints of the two 32-bit weight words, per sub-block i
    int     v1i[QR4_K];
    uint8_t sc[QR4_K];
    uint8_t m[QR4_K];
    float   d, dmin;
    int     bq8_offset;
};

static __device__ __forceinline__ void q4k_load_w(const block_q4_K * __restrict__ bq4_K, int iqs, q4k_wpart &w) {
    int v[2];
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int *q4 = (const int *)(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t *scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m  = sc + 2;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        w.v0i[i] = (v[0] >> (4*i)) & 0x0F0F0F0F;
        w.v1i[i] = (v[1] >> (4*i)) & 0x0F0F0F0F;
        w.sc[i] = sc[i];
        w.m[i]  = m[i];
    }
    w.d = h2f(bq4_K->d);
    w.dmin = h2f(bq4_K->dmin);
    w.bq8_offset = bq8_offset;
}

static __device__ __forceinline__ float q4k_dot_y(const q4k_wpart &w, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int   u[2*QR4_K];
    float d8[QR4_K];
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1_c *bq8i = bq8_1 + w.bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int *q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int dot1 = __dp4a(w.v1i[i], u[2*i+1], __dp4a(w.v0i[i], u[2*i+0], 0));
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * w.sc[i]);
        sumf_m += d8[i] * (dot2 * w.m[i]);
    }
    return w.d*sumf_d - w.dmin*sumf_m;
}

struct q6k_wpart {
    int   vi[QR6_K];
    int   sc[QR6_K];
    float d;
    int   bq8_offset;
};

static __device__ __forceinline__ void q6k_load_w(const block_q6_K * __restrict__ bq6_K, int iqs, q6k_wpart &w) {
    const int bq8_offset   = 2*QR6_K*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4)*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2*((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2_c(bq6_K->ql, iqs);
    const int vh = get_int_b2_c(bq6_K->qh, (QI6_K/4)*(iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t *scales = bq6_K->scales + scale_offset;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        w.sc[i] = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        w.vi[i] = __vsubss4((vil | vih), 0x20202020);
    }
    w.d = h2f(bq6_K->d);
    w.bq8_offset = bq8_offset;
}

static __device__ __forceinline__ float q6k_dot_y(const q6k_wpart &w, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int   u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4_c(bq8_1[w.bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[w.bq8_offset + 2*i].ds);
    }
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        sumf += d8[i] * (__dp4a(w.vi[i], u[i], 0) * w.sc[i]);
    }
    return w.d * sumf;
}

// y: NB activation vectors, vector b at y + b*y_stride (in q8_1 blocks); dst: column b at dst + b*out_dim.
template <int NB, int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_mmvq_q4_K_batch_kernel(
        const block_q4_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk, int y_stride) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI4_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q4_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) tmp[b] = 0.0f;
#pragma unroll 4
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        q4k_wpart wp;
        q4k_load_w(rw + kbx, kqs, wp);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            tmp[b] += q4k_dot_y(wp, y + (size_t)b * y_stride + kbx * (QK_K / QK8_1), kqs);
        }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const float t = warp_reduce_sum_f(tmp[b]);
        if (threadIdx.x == 0) {
            if (accum) dst[(size_t)b * out_dim + row] += t;
            else       dst[(size_t)b * out_dim + row]  = t;
        }
    }
}

template <int NB, int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_mmvq_q6_K_batch_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk, int y_stride) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) tmp[b] = 0.0f;
#pragma unroll 16
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        q6k_wpart wp;
        q6k_load_w(rw + kbx, kqs, wp);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            tmp[b] += q6k_dot_y(wp, y + (size_t)b * y_stride + kbx * (QK_K / QK8_1), kqs);
        }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const float t = warp_reduce_sum_f(tmp[b]);
        if (threadIdx.x == 0) {
            if (accum) dst[(size_t)b * out_dim + row] += t;
            else       dst[(size_t)b * out_dim + row]  = t;
        }
    }
}

// Fused gate+up with SwiGLU epilogue for NB columns; d_act[b*out_dim + row].
template <int NB, int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_fused_swiglu_mmvq_q4_K_batch_kernel(
    const block_q4_K * __restrict__ wg,
    const block_q4_K * __restrict__ wu,
    const block_q8_1_c * __restrict__ y,
    float * __restrict__ d_act,
    int out_dim, int nblk, int y_stride) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI4_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q4_K * __restrict__ rwg = wg + (size_t)row * nblk;
    const block_q4_K * __restrict__ rwu = wu + (size_t)row * nblk;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tg[NB], tu[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) { tg[b] = 0.0f; tu[b] = 0.0f; }

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        q4k_wpart wpg, wpu;
        q4k_load_w(rwg + kbx, kqs, wpg);
        q4k_load_w(rwu + kbx, kqs, wpu);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const block_q8_1_c *yb = y + (size_t)b * y_stride + kbx * (QK_K / QK8_1);
            tg[b] += q4k_dot_y(wpg, yb, kqs);
            tu[b] += q4k_dot_y(wpu, yb, kqs);
        }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const float g = warp_reduce_sum_f(tg[b]);
        const float u = warp_reduce_sum_f(tu[b]);
        if (threadIdx.x == 0) {
            float silu_g = g / (1.0f + expf(-g));
            d_act[(size_t)b * out_dim + row] = silu_g * u;
        }
    }
}

// LM head Q6_K for NB columns: logits[b*out_dim + row], per-column block argmax in block_max/idx[b*gridDim.x + blockIdx.x].
template <int NB, int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_mmvq_q6_K_head_argmax_batch_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        float * __restrict__ block_max,
        int * __restrict__ block_idx,
        int out_dim, int nblk, int y_stride) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    const bool valid = (row < out_dim);

    const block_q6_K * __restrict__ rw = valid ? (w + (size_t)row * nblk) : w;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) tmp[b] = 0.0f;
    if (valid) {
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            q6k_wpart wp;
            q6k_load_w(rw + kbx, kqs, wp);
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                tmp[b] += q6k_dot_y(wp, y + (size_t)b * y_stride + kbx * (QK_K / QK8_1), kqs);
            }
        }
    }
    __shared__ float s_max[NB][rows_per_block];
    __shared__ int   s_idx[NB][rows_per_block];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const float t = warp_reduce_sum_f(tmp[b]);
        if (threadIdx.x == 0) {
            if (valid && dst) dst[(size_t)b * out_dim + row] = t;
            s_max[b][threadIdx.y] = valid ? t : -1e30f;
            s_idx[b][threadIdx.y] = valid ? row : 0;
        }
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < NB) {
        const int b = threadIdx.x;
        float m = s_max[b][0];
        int idx = s_idx[b][0];
        for (int r = 1; r < rows_per_block; ++r) {
            if (s_max[b][r] > m) {
                m = s_max[b][r];
                idx = s_idx[b][r];
            }
        }
        block_max[(size_t)b * gridDim.x + blockIdx.x] = m;
        block_idx[(size_t)b * gridDim.x + blockIdx.x] = idx;
    }
}

// ShortConv recurrence for n consecutive tokens, strictly sequential per channel (each thread owns one channel and
// walks the tokens in order) with the exact per-token math of lfm_shortconv_step_kernel. The state after token j is
// written to snap[j] (rollback source); the live state is left at the state after the last token.
// in_proj: [n][3d]; conv_out_q8: [n][d/32] blocks; snap: [n][2d].
__global__ void lfm_shortconv_seq_kernel(
    const float * __restrict__ in_proj,
    float * __restrict__ conv_state,
    float * __restrict__ snap,
    const float * __restrict__ conv_w,
    block_q8_1_c * __restrict__ conv_out_q8,
    int d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float s0 = conv_state[i];
    float s1 = conv_state[d + i];

    float k0 = conv_w[i * 3 + 0];
    float k1 = conv_w[i * 3 + 1];
    float k2 = conv_w[i * 3 + 2];

    const int lane_id = threadIdx.x & 31;
    for (int j = 0; j < n; ++j) {
        const float *ip = in_proj + (size_t)j * 3 * d;
        float b_val = ip[i];
        float c_val = ip[d + i];
        float x_val = ip[2 * d + i];

        float u = b_val * x_val;
        float y_conv = s0 * k0 + s1 * k1 + u * k2;

        // Rolling state update: S0 <- S1, S1 <- u
        s0 = s1;
        s1 = u;
        snap[(size_t)j * 2 * d + i] = s0;
        snap[(size_t)j * 2 * d + d + i] = s1;

        float g = c_val * y_conv;

        const int b = i >> 5;
        float amax = fabsf(g);
        float sum  = g;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float scale = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(g / scale);
        block_q8_1_c *o = conv_out_q8 + (size_t)j * (d / QK8_1);
        o[b].qs[lane_id] = q;
        if (lane_id == 0) {
            o[b].ds = make_half2(__float2half(scale), __float2half(sum));
        }
    }
    conv_state[i] = s0;
    conv_state[d + i] = s1;
}

// -----------------------------------------------------------------------------
// Host Dispatch Routines
// -----------------------------------------------------------------------------

// Stream every forward-path kernel/memcpy is enqueued on (0 = default stream, whichever the build selects).
// The engine points this at its capture stream while recording / running the CUDA graph.
static cudaStream_t g_lfm_stream = 0;
extern "C" void lfm25_gpu_set_stream(cudaStream_t s) { g_lfm_stream = s; }

void lfm25_gpu_embed(const int *d_tokpos, float *d_x, const lfm25_tensor_t *token_embd, int dim) {
    if (token_embd->info.type != GGML_TYPE_Q6_K) {
        fprintf(stderr, "[FATAL] lfm25_gpu_embed: token_embd type %d unsupported (only Q6_K)\n", (int)token_embd->info.type);
        exit(EXIT_FAILURE);
    }
    if (dim % QK_K != 0) { fprintf(stderr, "[FATAL] lfm25_gpu_embed: dim %d not a multiple of %d\n", dim, QK_K); exit(EXIT_FAILURE); }
    lfm_embed_q6_K_kernel<<<dim / QK_K, QK_K, 0, g_lfm_stream>>>((const uint8_t*)token_embd->dev, d_tokpos,
                                                              (size_t)ggml_row_size(GGML_TYPE_Q6_K, dim), d_x);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_fused_rms_norm_quant(const float *d_src, const float *d_weight, float *d_dst_norm, void *d_dst_q8, int dim, float eps) {
    int nb = (dim / QK8_1 + 7) / 8;
    if (nb > LFM_NORM_BLOCKS) nb = LFM_NORM_BLOCKS;
    if (nb < 1) nb = 1;
    lfm_fused_rms_norm_quant_kernel<<<nb, 256, 0, g_lfm_stream>>>(d_src, d_weight, d_dst_norm, (block_q8_1_c*)d_dst_q8, dim, eps);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_mmvq_q4_K(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum) {
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);

    if (accum) {
        lfm_mmvq_q4_K_dense_kernel<MMVQ_ROWS_PER_BLOCK, true><<<grid, block, 0, g_lfm_stream>>>(
            (const block_q4_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk);
    } else {
        lfm_mmvq_q4_K_dense_kernel<MMVQ_ROWS_PER_BLOCK, false><<<grid, block, 0, g_lfm_stream>>>(
            (const block_q4_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk);
    }
    CUDA_CHECK_LAST();
}

void lfm25_gpu_mmvq_q6_K(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum) {
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);

    if (accum) {
        lfm_mmvq_q6_K_dense_kernel<MMVQ_ROWS_PER_BLOCK, true><<<grid, block, 0, g_lfm_stream>>>(
            (const block_q6_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk);
    } else {
        lfm_mmvq_q6_K_dense_kernel<MMVQ_ROWS_PER_BLOCK, false><<<grid, block, 0, g_lfm_stream>>>(
            (const block_q6_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk);
    }
    CUDA_CHECK_LAST();
}

void lfm25_gpu_shortconv_step(const float *d_in_proj, float *d_conv_state, const float *d_conv_w, float *d_conv_out, void *d_conv_out_q8, int d) {
    const int threads = 256;
    const int blocks = (d + threads - 1) / threads;
    lfm_shortconv_step_kernel<<<blocks, threads, 0, g_lfm_stream>>>(
        d_in_proj, d_conv_state, d_conv_w, d_conv_out, (block_q8_1_c*)d_conv_out_q8, d);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_inproj_shortconv(const lfm25_tensor_t *in_proj, const void *d_xq8, float *d_conv_state, const float *d_conv_w, float *d_conv_out, void *d_conv_out_q8, int d) {
    if (in_proj->info.type != GGML_TYPE_Q4_K) {
        fprintf(stderr, "[FATAL] lfm25_gpu_inproj_shortconv: in_proj type %d unsupported (only Q4_K)\n", (int)in_proj->info.type);
        exit(EXIT_FAILURE);
    }
    if (d % 32 != 0 || d % QK_K != 0) { fprintf(stderr, "[FATAL] lfm25_gpu_inproj_shortconv: d=%d unsupported\n", d); exit(EXIT_FAILURE); }
    dim3 block(32u, (unsigned)LFM_CONV_WARPS, 1u);
    lfm_inproj_shortconv_kernel<LFM_CONV_WARPS><<<d / 32, block, 0, g_lfm_stream>>>(
        (const block_q4_K*)in_proj->dev, (const block_q8_1_c*)d_xq8, d_conv_state, d_conv_w, d_conv_out,
        (block_q8_1_c*)d_conv_out_q8, d, d / QK_K);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_qk_norm_rope(float *d_q, float *d_k, const float *d_q_norm, const float *d_k_norm, float *d_k_cache, int n_head, int n_head_kv, int head_dim, const int *d_pos, int max_seq, float freq_base, float eps, int layer_idx) {
    const int total_heads = n_head + n_head_kv; // 32 + 8 = 40
    lfm_fused_qk_norm_neox_rope_kernel<<<total_heads, head_dim, 0, g_lfm_stream>>>(
        d_q, d_k, d_q_norm, d_k_norm, d_k_cache, n_head, n_head_kv, head_dim, d_pos, max_seq, freq_base, eps, layer_idx);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_v_cache_write(const float *d_v, float *d_v_cache, int n_head_kv, int head_dim, const int *d_pos, int max_seq, int layer_idx) {
    lfm_v_cache_write_kernel<<<n_head_kv, head_dim, 0, g_lfm_stream>>>(d_v, d_v_cache, n_head_kv, head_dim, d_pos, max_seq, layer_idx);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_attention(const float *d_q, const float *d_k_cache, const float *d_v_cache, float *d_out, void *d_out_q8, int n_head, int n_head_kv, int head_dim, const int *d_pos, int max_seq, int layer_idx) {
    lfm_flash_decoding_kernel<<<n_head, head_dim, 0, g_lfm_stream>>>(
        d_q, d_k_cache, d_v_cache, d_out, (block_q8_1_c*)d_out_q8, n_head, n_head_kv, head_dim, d_pos, max_seq, layer_idx);
    CUDA_CHECK_LAST();
}

void lfm25_gpu_fused_swiglu(const void *dev_gate, const void *dev_up, const void *d_xq8, float *d_act, void *d_act_q8, int in_dim, int ffn_dim) {
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (ffn_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);

    lfm_fused_swiglu_mmvq_q4_K_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block, 0, g_lfm_stream>>>(
        (const block_q4_K*)dev_gate, (const block_q4_K*)dev_up, (const block_q8_1_c*)d_xq8, d_act, ffn_dim, nblk);
    CUDA_CHECK_LAST();

    if (d_act_q8) {
        lfm_quantize_q8_1_kernel<<<(ffn_dim + 255) / 256, 256, 0, g_lfm_stream>>>(d_act, (block_q8_1_c*)d_act_q8, ffn_dim);
        CUDA_CHECK_LAST();
    }
}

void lfm25_gpu_output_norm_head_argmax(
    const float *d_res_x, const float *d_norm_w, const void *dev_output_w,
    void *d_xq8, float *d_logits,
    float *d_block_max, int *d_block_idx, int *d_best_token,
    int dim, int vocab_size, float eps) {
    // 1. RMSNorm on residual x -> quantize to q8_1
    lfm25_gpu_fused_rms_norm_quant(d_res_x, d_norm_w, NULL, d_xq8, dim, eps);

    // 2. LM Head MMVQ Q6_K
    const int nblk = dim / QK_K;
    const int grid_blocks = (vocab_size + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);

    lfm_mmvq_q6_K_head_argmax_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block, 0, g_lfm_stream>>>(
        (const block_q6_K*)dev_output_w, (const block_q8_1_c*)d_xq8, d_logits, d_block_max, d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    // 3. Second reduction kernel for global argmax
    lfm_argmax_reduction_kernel<<<1, 256, 0, g_lfm_stream>>>(d_block_max, d_block_idx, d_best_token, grid_blocks);
    CUDA_CHECK_LAST();
}

// -----------------------------------------------------------------------------
// Batched host launchers (Task 5.1). nb = number of columns, 1..LFM25_MAX_BATCH. Activation vectors / outputs are
// stored token-major: vector b of an `in_dim` input starts at q8 block b*(in_dim/32); column b of the output at b*out_dim.
// -----------------------------------------------------------------------------
#define LFM_DISPATCH_NB(CALL)                                                                          \
    switch (nb) {                                                                                      \
        case 1: CALL(1); break;  case 2: CALL(2); break;  case 3: CALL(3); break;  case 4: CALL(4); break; \
        case 5: CALL(5); break;  case 6: CALL(6); break;  case 7: CALL(7); break;  case 8: CALL(8); break; \
        default: fprintf(stderr, "[FATAL] %s: nb=%d out of range 1..%d\n", __func__, nb, LFM25_MAX_BATCH); abort(); \
    }

void lfm25_gpu_mmvq_q4_K_batch(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum, int nb) {
    const int nblk = in_dim / QK_K;
    const int ystride = in_dim / QK8_1;
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
#define LFM_L(NB) do { \
        if (accum) lfm_mmvq_q4_K_batch_kernel<NB, MMVQ_ROWS_PER_BLOCK, true><<<grid, block, 0, g_lfm_stream>>>( \
            (const block_q4_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk, ystride); \
        else lfm_mmvq_q4_K_batch_kernel<NB, MMVQ_ROWS_PER_BLOCK, false><<<grid, block, 0, g_lfm_stream>>>( \
            (const block_q4_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk, ystride); } while (0)
    LFM_DISPATCH_NB(LFM_L)
#undef LFM_L
    CUDA_CHECK_LAST();
}

void lfm25_gpu_mmvq_q6_K_batch(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum, int nb) {
    const int nblk = in_dim / QK_K;
    const int ystride = in_dim / QK8_1;
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
#define LFM_L(NB) do { \
        if (accum) lfm_mmvq_q6_K_batch_kernel<NB, MMVQ_ROWS_PER_BLOCK, true><<<grid, block, 0, g_lfm_stream>>>( \
            (const block_q6_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk, ystride); \
        else lfm_mmvq_q6_K_batch_kernel<NB, MMVQ_ROWS_PER_BLOCK, false><<<grid, block, 0, g_lfm_stream>>>( \
            (const block_q6_K*)dev_w, (const block_q8_1_c*)d_xq8, d_dst, out_dim, nblk, ystride); } while (0)
    LFM_DISPATCH_NB(LFM_L)
#undef LFM_L
    CUDA_CHECK_LAST();
}

void lfm25_gpu_fused_swiglu_batch(const void *dev_gate, const void *dev_up, const void *d_xq8, float *d_act, void *d_act_q8, int in_dim, int ffn_dim, int nb) {
    const int nblk = in_dim / QK_K;
    const int ystride = in_dim / QK8_1;
    dim3 grid((unsigned)((ffn_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
#define LFM_L(NB) \
    lfm_fused_swiglu_mmvq_q4_K_batch_kernel<NB, MMVQ_ROWS_PER_BLOCK><<<grid, block, 0, g_lfm_stream>>>( \
        (const block_q4_K*)dev_gate, (const block_q4_K*)dev_up, (const block_q8_1_c*)d_xq8, d_act, ffn_dim, nblk, ystride)
    LFM_DISPATCH_NB(LFM_L)
#undef LFM_L
    CUDA_CHECK_LAST();

    if (d_act_q8) {
        // ffn_dim is a multiple of 32, so a quantize over the whole [nb][ffn_dim] array yields the per-token blocks.
        const int total = nb * ffn_dim;
        lfm_quantize_q8_1_kernel<<<(total + 255) / 256, 256, 0, g_lfm_stream>>>(d_act, (block_q8_1_c*)d_act_q8, total);
        CUDA_CHECK_LAST();
    }
}

// in_proj: [nb][3d] (token-major); conv_out_q8: [nb][d/32] blocks; snap: [nb][2d]; conv_state: live [2d].
void lfm25_gpu_shortconv_seq(const float *d_in_proj, float *d_conv_state, float *d_snap, const float *d_conv_w, void *d_conv_out_q8, int d, int nb) {
    if (d % 32 != 0) { fprintf(stderr, "[FATAL] lfm25_gpu_shortconv_seq: d=%d unsupported\n", d); exit(EXIT_FAILURE); }
    const int threads = 256;
    const int blocks = (d + threads - 1) / threads;
    lfm_shortconv_seq_kernel<<<blocks, threads, 0, g_lfm_stream>>>(
        d_in_proj, d_conv_state, d_snap, d_conv_w, (block_q8_1_c*)d_conv_out_q8, d, nb);
    CUDA_CHECK_LAST();
}

// Rollback: copy `n_layers` state slabs [2d floats each] from snap[l][slot] (layer stride snap_layer_stride floats)
// into the live state [l][2d] with one 2D copy.
void lfm25_gpu_conv_state_restore(float *d_conv_state, const float *d_snap, int n_layers, int slot, int d, size_t snap_layer_stride) {
    CUDA_CHECK(cudaMemcpy2DAsync(d_conv_state, (size_t)2 * d * sizeof(float),
                                 d_snap + (size_t)slot * 2 * d, snap_layer_stride * sizeof(float),
                                 (size_t)2 * d * sizeof(float), (size_t)n_layers, cudaMemcpyDeviceToDevice, g_lfm_stream));
}

// Final RMSNorm (per token) + batched LM head + per-column argmax.
//   d_res_x [nb][dim], d_xq8 [nb][dim/32] blocks, d_logits [nb][vocab], d_block_max/idx [nb][ceil(vocab/rows)], d_best_token [nb].
void lfm25_gpu_output_norm_head_argmax_batch(
    const float *d_res_x, const float *d_norm_w, const void *dev_output_w,
    void *d_xq8, float *d_logits,
    float *d_block_max, int *d_block_idx, int *d_best_token,
    int dim, int vocab_size, float eps, int nb) {
    for (int j = 0; j < nb; ++j) {
        lfm25_gpu_fused_rms_norm_quant(d_res_x + (size_t)j * dim, d_norm_w, NULL,
                                       (char*)d_xq8 + (size_t)j * (dim / QK8_1) * sizeof(block_q8_1_c), dim, eps);
    }
    const int nblk = dim / QK_K;
    const int ystride = dim / QK8_1;
    const int grid_blocks = (vocab_size + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
#define LFM_L(NB) \
    lfm_mmvq_q6_K_head_argmax_batch_kernel<NB, MMVQ_ROWS_PER_BLOCK><<<grid, block, 0, g_lfm_stream>>>( \
        (const block_q6_K*)dev_output_w, (const block_q8_1_c*)d_xq8, d_logits, d_block_max, d_block_idx, vocab_size, nblk, ystride)
    LFM_DISPATCH_NB(LFM_L)
#undef LFM_L
    CUDA_CHECK_LAST();
    for (int j = 0; j < nb; ++j) {
        lfm_argmax_reduction_kernel<<<1, 256, 0, g_lfm_stream>>>(d_block_max + (size_t)j * grid_blocks,
                                                               d_block_idx + (size_t)j * grid_blocks, d_best_token + j, grid_blocks);
        CUDA_CHECK_LAST();
    }
}

// -----------------------------------------------------------------------------
// GPU Preparation and Teardown
// -----------------------------------------------------------------------------
static void upload_tensor(lfm25_tensor_t *t) {
    if (!t->data || t->info.size_bytes == 0) return;
    CUDA_CHECK(cudaMalloc(&t->dev, t->info.size_bytes));
    CUDA_CHECK(cudaMemcpy(t->dev, t->data, t->info.size_bytes, cudaMemcpyHostToDevice));
}

static void free_tensor(lfm25_tensor_t *t) {
    if (t->dev) {
        cudaFree(t->dev);
        t->dev = NULL;
    }
}

int lfm25_gpu_prepare(lfm25_model_t *model, int *max_context_len, int fill_ram) {
    (void)fill_ram;
    if (!model) return -1;
    if (max_context_len && *max_context_len <= 0) {
        *max_context_len = model->context_length > 0 ? (int)model->context_length : 4096;
    }

    fprintf(stderr, "[LFM25 GPU] Uploading model weights to device VRAM...\n");

    // 1. Upload trunk tensors
    upload_tensor(&model->token_embd);
    upload_tensor(&model->token_embd_norm);
    if (model->output_weight.data != model->token_embd.data) {
        upload_tensor(&model->output_weight);
    } else {
        model->output_weight.dev = model->token_embd.dev;
    }

    // 2. Upload layers
    for (uint32_t l = 0; l < model->block_count; ++l) {
        lfm25_layer_t *L = &model->layers[l];

        // Norms
        upload_tensor(&L->attn_norm);
        upload_tensor(&L->ffn_norm);

        // Operator
        if (L->op_type == LFM25_OP_CONV) {
            upload_tensor(&L->shortconv_in_proj);
            upload_tensor(&L->shortconv_conv);
            upload_tensor(&L->shortconv_out_proj);
        } else {
            upload_tensor(&L->attn_q);
            upload_tensor(&L->attn_k);
            upload_tensor(&L->attn_v);
            upload_tensor(&L->attn_output);
            upload_tensor(&L->attn_q_norm);
            upload_tensor(&L->attn_k_norm);
        }

        // FFN
        if (L->ffn_type == LFM25_FFN_DENSE) {
            upload_tensor(&L->ffn_gate);
            upload_tensor(&L->ffn_up);
            upload_tensor(&L->ffn_down);
        } else {
            // For MoE layers, upload router and expert tensors if VRAM permits
            upload_tensor(&L->ffn_gate_inp);
            upload_tensor(&L->exp_probs_b);
            // In 8B (100% VRAM resident), upload all experts directly
            if (!model->is_moe || (model->expert_count <= 32 && !(getenv("LFM25_FORCE_EXPERT_CACHE") && getenv("LFM25_FORCE_EXPERT_CACHE")[0] == '1'))) {
                upload_tensor(&L->ffn_gate_exps);
                upload_tensor(&L->ffn_up_exps);
                upload_tensor(&L->ffn_down_exps);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "[LFM25 GPU] Weight upload complete.\n");
    return 0;
}

void lfm25_gpu_shutdown(lfm25_model_t *model) {
    if (!model) return;
    fprintf(stderr, "[LFM25 GPU] Freeing device weights...\n");

    // Tied LM head: output_weight.dev aliases token_embd.dev. Decide before freeing token_embd,
    // which NULLs its own pointer and would make the alias look like a separate allocation.
    if (model->output_weight.dev != model->token_embd.dev) {
        free_tensor(&model->output_weight);
    }
    model->output_weight.dev = NULL;
    free_tensor(&model->token_embd);
    free_tensor(&model->token_embd_norm);

    for (uint32_t l = 0; l < model->block_count; ++l) {
        lfm25_layer_t *L = &model->layers[l];
        free_tensor(&L->attn_norm);
        free_tensor(&L->ffn_norm);

        if (L->op_type == LFM25_OP_CONV) {
            free_tensor(&L->shortconv_in_proj);
            free_tensor(&L->shortconv_conv);
            free_tensor(&L->shortconv_out_proj);
        } else {
            free_tensor(&L->attn_q);
            free_tensor(&L->attn_k);
            free_tensor(&L->attn_v);
            free_tensor(&L->attn_output);
            free_tensor(&L->attn_q_norm);
            free_tensor(&L->attn_k_norm);
        }

        if (L->ffn_type == LFM25_FFN_DENSE) {
            free_tensor(&L->ffn_gate);
            free_tensor(&L->ffn_up);
            free_tensor(&L->ffn_down);
        } else {
            free_tensor(&L->ffn_gate_inp);
            free_tensor(&L->exp_probs_b);
            free_tensor(&L->ffn_gate_exps);
            free_tensor(&L->ffn_up_exps);
            free_tensor(&L->ffn_down_exps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// MoE Router: sigmoid gating, bias-for-selection-only, top-4 of 32 or 64, normalized weights
// -----------------------------------------------------------------------------
#define LFM_ROUTER_TOPK 4

// One block, 32 warps. Warp w computes logits w and (for 64 experts) w+32 in FP32; warp 0 then holds
// experts lane and lane+32 and does sigmoid, bias add, 4 rounds of warp arg-max and normalization.
// p = sigmoid(logit) is unbiased; the bias only steers selection, weights use p alone.
// Ties -> lower expert index.
template <int NE>
__global__ void lfm_router_top4_kernel(
        const float * __restrict__ x, const float * __restrict__ w_gate_inp,
        const float * __restrict__ bias, int *topk_ids, float *topk_w, float *logits_out, int dim) {
    constexpr int NC = NE / 32;   // candidates per lane
    __shared__ float s_logit[NE];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int e = warp + 32 * c;
        const float *wrow = w_gate_inp + (size_t)e * dim;
        float acc = 0.0f;
        for (int i = lane; i < dim; i += 32) acc = fmaf(wrow[i], x[i], acc);
        acc = warp_reduce_sum_f(acc);
        if (lane == 0) s_logit[e] = acc;
    }
    __syncthreads();

    if (warp != 0) return;

    float p[NC], sel[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const float logit = s_logit[lane + 32 * c];
        if (logits_out) logits_out[lane + 32 * c] = logit;
        p[c] = 1.0f / (1.0f + expf(-logit));
        sel[c] = p[c] + bias[lane + 32 * c];
    }

    int   my_id = -1;
    float my_p  = 0.0f;
    float psum  = 0.0f;
#pragma unroll
    for (int k = 0; k < LFM_ROUTER_TOPK; k++) {
        float bv = sel[0];
        int   bi = lane;
#pragma unroll
        for (int c = 1; c < NC; c++) {
            if (sel[c] > bv) { bv = sel[c]; bi = lane + 32 * c; }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, mask, 32);
            int   oi = __shfl_xor_sync(0xffffffff, bi, mask, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float wp = __shfl_sync(0xffffffff, p[0], bi & 31, 32);
#pragma unroll
        for (int c = 1; c < NC; c++) {
            const float wc = __shfl_sync(0xffffffff, p[c], bi & 31, 32);
            if ((bi >> 5) == c) wp = wc;
        }
        if (lane == k) { my_id = bi; my_p = wp; }
        psum += wp;
#pragma unroll
        for (int c = 0; c < NC; c++) if (bi == lane + 32 * c) sel[c] = -INFINITY;   // mark as taken
    }
    const float denom = fmaxf(psum, 6.103515625e-5f);
    if (lane < LFM_ROUTER_TOPK) {
        topk_ids[lane] = my_id;
        topk_w[lane]   = my_p / denom;
    }
}

void lfm25_gpu_router_top4(const float *d_x, const float *d_w_gate_inp, const float *d_bias,
                           int *d_topk_ids, float *d_topk_w, float *d_logits,
                           int dim, int n_expert) {
    if (n_expert == 32) {
        lfm_router_top4_kernel<32><<<1, 1024, 0, g_lfm_stream>>>(d_x, d_w_gate_inp, d_bias, d_topk_ids, d_topk_w, d_logits, dim);
    } else if (n_expert == 64) {
        lfm_router_top4_kernel<64><<<1, 1024, 0, g_lfm_stream>>>(d_x, d_w_gate_inp, d_bias, d_topk_ids, d_topk_w, d_logits, dim);
    } else {
        fprintf(stderr, "lfm25_gpu_router_top4: n_expert=%d unsupported (only 32 or 64)\n", n_expert);
        abort();
    }
    CUDA_CHECK_LAST();
}

// -----------------------------------------------------------------------------
// MoE expert FFN: gate/up (Q4_K) + SwiGLU, down (Q4_K or Q6_K), weighted ordered sum
// -----------------------------------------------------------------------------
void lfm25_gpu_quantize_q8_1(const float *d_x, void *d_q8, int n) {
    lfm_quantize_q8_1_kernel<<<(n + 255) / 256, 256, 0, g_lfm_stream>>>(d_x, (block_q8_1_c*)d_q8, n);
    CUDA_CHECK_LAST();
}

// grid (ceil(ffn_dim/rows), 4): blockIdx.y = k selects expert topk_ids[k] (read on device).
// Expert e starts at e * gate_stride / up_stride bytes. Writes act[k * ffn_dim + row].
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_moe_gate_up_swiglu_q4_K_kernel(
        const block_q4_K * __restrict__ wg,
        const block_q4_K * __restrict__ wu,
        const block_q8_1_c * __restrict__ y,
        const int * __restrict__ topk_ids,
        float * __restrict__ d_act,
        int ffn_dim, int nblk, size_t gate_stride, size_t up_stride) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI4_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= ffn_dim) return;
    const int k = blockIdx.y;
    const size_t roff = (size_t)row * nblk;
    const size_t eid = (size_t)topk_ids[k];
    const block_q4_K * __restrict__ rwg = (const block_q4_K*)((const char*)wg + eid * gate_stride) + roff;
    const block_q4_K * __restrict__ rwu = (const block_q4_K*)((const char*)wu + eid * up_stride) + roff;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tg = 0.0f;
    float tu = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c *yb = y + kbx * (QK_K / QK8_1);
        tg += vec_dot_q4_K_q8_1_c(rwg + kbx, yb, kqs);
        tu += vec_dot_q4_K_q8_1_c(rwu + kbx, yb, kqs);
    }
    tg = warp_reduce_sum_f(tg);
    tu = warp_reduce_sum_f(tu);

    if (threadIdx.x == 0) {
        float silu_g = tg / (1.0f + expf(-tg));
        d_act[(size_t)k * ffn_dim + row] = silu_g * tu;
    }
}

template <typename block_t> struct lfm_moe_down_traits;
template <> struct lfm_moe_down_traits<block_q4_K> {
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    static constexpr int qi  = QI4_K;
    static __device__ __forceinline__ float dot(const block_q4_K *b, const block_q8_1_c *y, int iqs) {
        return vec_dot_q4_K_q8_1_c(b, y, iqs);
    }
};
template <> struct lfm_moe_down_traits<block_q6_K> {
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    static constexpr int qi  = QI6_K;
    static __device__ __forceinline__ float dot(const block_q6_K *b, const block_q8_1_c *y, int iqs) {
        return vec_dot_q6_K_q8_1_c(b, y, iqs);
    }
};

// Each row computes y_k = down_e[row] . Q8_1(act_k) for k=0..3, then
// out = ((w0*y0 + w1*y1) + w2*y2) + w3*y3, stored or added to dst[row].
// nblk may be odd (7 for ffn_dim=1792); the strided loop handles any count.
template <typename block_t, int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void lfm_moe_down_kernel(
        const block_t * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        const int * __restrict__ topk_ids,
        const float * __restrict__ topk_w,
        float * __restrict__ dst,
        int out_dim, int nblk, size_t down_stride) {
    constexpr int vdr = lfm_moe_down_traits<block_t>::vdr;
    constexpr int tpsb = lfm_moe_down_traits<block_t>::qi / vdr;
    constexpr int blocks_per_iter = vdr * 32 / lfm_moe_down_traits<block_t>::qi;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float yk[LFM_ROUTER_TOPK];
#pragma unroll
    for (int k = 0; k < LFM_ROUTER_TOPK; k++) {
        const block_t * __restrict__ rw = (const block_t*)((const char*)w + (size_t)topk_ids[k] * down_stride) + (size_t)row * nblk;
        const block_q8_1_c *yk8 = y + (size_t)k * nblk * (QK_K / QK8_1);
        float tmp = 0.0f;
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            tmp += lfm_moe_down_traits<block_t>::dot(rw + kbx, yk8 + kbx * (QK_K / QK8_1), kqs);
        }
        yk[k] = warp_reduce_sum_f(tmp);
    }
    if (threadIdx.x == 0) {
        float o = topk_w[0] * yk[0];
        o = o + topk_w[1] * yk[1];
        o = o + topk_w[2] * yk[2];
        o = o + topk_w[3] * yk[3];
        if (accum) dst[row] += o;
        else       dst[row]  = o;
    }
}

void lfm25_gpu_moe_ffn(const void *d_xq8, const int *d_topk_ids, const float *d_topk_w,
                       const void *dev_gate_exps, const void *dev_up_exps,
                       const void *dev_down_exps, int down_type,
                       float *d_act, void *d_act_q8,
                       float *d_dst, bool accum, int dim, int ffn_dim,
                       size_t gate_stride, size_t up_stride, size_t down_stride) {
    if (down_type != GGML_TYPE_Q4_K && down_type != GGML_TYPE_Q6_K) {
        fprintf(stderr, "lfm25_gpu_moe_ffn: unsupported down_type=%d (only Q4_K/Q6_K)\n", down_type);
        abort();
    }
    if (dim % QK_K != 0 || ffn_dim % QK_K != 0) {
        fprintf(stderr, "lfm25_gpu_moe_ffn: dim=%d / ffn_dim=%d must be multiples of %d\n", dim, ffn_dim, QK_K);
        abort();
    }
    const int nblk_in = dim / QK_K;
    const int nblk_down = ffn_dim / QK_K;

    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    dim3 grid_gu((unsigned)((ffn_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), (unsigned)LFM_ROUTER_TOPK, 1u);
    lfm_moe_gate_up_swiglu_q4_K_kernel<MMVQ_ROWS_PER_BLOCK><<<grid_gu, block, 0, g_lfm_stream>>>(
        (const block_q4_K*)dev_gate_exps, (const block_q4_K*)dev_up_exps, (const block_q8_1_c*)d_xq8,
        d_topk_ids, d_act, ffn_dim, nblk_in, gate_stride, up_stride);
    CUDA_CHECK_LAST();

    lfm25_gpu_quantize_q8_1(d_act, d_act_q8, LFM_ROUTER_TOPK * ffn_dim);

    dim3 grid_dn((unsigned)((dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    const block_q8_1_c *yq = (const block_q8_1_c*)d_act_q8;
#define LFM_MOE_DOWN_LAUNCH(BT, ACC) \
    lfm_moe_down_kernel<BT, MMVQ_ROWS_PER_BLOCK, ACC><<<grid_dn, block, 0, g_lfm_stream>>>( \
        (const BT*)dev_down_exps, yq, d_topk_ids, d_topk_w, d_dst, dim, nblk_down, down_stride)
    if (down_type == GGML_TYPE_Q4_K) {
        if (accum) LFM_MOE_DOWN_LAUNCH(block_q4_K, true);
        else       LFM_MOE_DOWN_LAUNCH(block_q4_K, false);
    } else {
        if (accum) LFM_MOE_DOWN_LAUNCH(block_q6_K, true);
        else       LFM_MOE_DOWN_LAUNCH(block_q6_K, false);
    }
#undef LFM_MOE_DOWN_LAUNCH
    CUDA_CHECK_LAST();
}
