#include "state.h"

__global__ void mul_mat_vec_q8_0_kernel(const block_q8_0 * __restrict__ w,
                                        const float * __restrict__ x,
                                        float * __restrict__ y,
                                        int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nblk = in_dim / QK8_0;
    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

    float acc = 0.0f;
    for (int c = tid; c < in_dim; c += blockDim.x) {
        const block_q8_0 &b = rw[c / QK8_0];
        const float d = h2f(b.d);
        acc += (float)b.qs[c % QK8_0] * d * x[c];
    }

    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// Autoregressive gated delta-net per-head state update, offloaded from engine.c's CPU loop.
// Each (head, row) is fully independent given qh/kh/vh and the head's S slice, so the whole
// pass maps to one thread per (head, row). Per row this replicates the exact CPU math and
// accumulation order (j = 0..head_k_dim-1) of build_delta_net_autoregressive():
// Vectorized with float4, shared memory cached q/k, and fused loops for peak compute/bandwidth efficiency.

__global__ void mul_mat_vec_f32_kernel(const float * __restrict__ w,
                                       const float * __restrict__ x,
                                       float * __restrict__ y,
                                       int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * __restrict__ rw = w + (size_t)row * in_dim;

    float acc = 0.0f;
    for (int c = tid; c < in_dim; c += blockDim.x) {
        acc += rw[c] * x[c];
    }

    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// Q4_K mat-vec, matching mat_vec_q4_K() exactly (strict scalar sequential accumulation).
// One thread per output row iterates c in increasing order, replicating the CPU's volatile
// scalar sum bit-for-bit (both compile `sum += (dv*qn - mv)*x[c]` with fmad enabled, so the
// same mul->fma contractions occur). The Q4_K super-block layout (d/dmin fp16, 12-byte
// sub-block scales, both nibbles across 4 chunks of 64) is decoded via get_scale_min_k4().
__global__ void mul_mat_vec_q4_K_kernel(const block_q4_K * __restrict__ w,
                                        const float * __restrict__ x,
                                        float * __restrict__ y,
                                        int in_dim) {
    const int row = blockIdx.x;
    const int nblk = in_dim / QK_K;
    const block_q4_K * __restrict__ rw = w + (size_t)row * nblk;

    // Single thread per row, strict scalar sequential accumulation in the exact CPU order
    // (super-block -> 64-element chunk -> low nibble 32 -> high nibble 32). Iterated
    // incrementally (no per-element div/mod) so indexing is cheap while remaining bit-identical
    // to mat_vec_q4_K() and the previous explicit-div/mod kernel.
    float sum = 0.0f;
    int cidx = 0;
    for (int b = 0; b < nblk; ++b) {
        const block_q4_K &blk = rw[b];
        const float d   = h2f(blk.d);
        const float mn  = h2f(blk.dmin);
        const uint8_t *q = blk.qs;
        int is = 0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t sc, m;
            // sub-block 0 (low nibble)
            if (is < 4) { sc = blk.scales[is] & 63; m = blk.scales[is + 4] & 63; }
            else        { sc = (blk.scales[is + 4] & 0xF) | ((blk.scales[is - 4] >> 6) << 4);
                           m  = (blk.scales[is + 4] >> 4)  | ((blk.scales[is]     >> 6) << 4); }
            const float d1 = d * (float)sc;
            const float m1 = mn * (float)m;
            // sub-block 1 (high nibble)
            if (is + 1 < 4) { sc = blk.scales[is + 1] & 63; m = blk.scales[is + 5] & 63; }
            else            { sc = (blk.scales[is + 5] & 0xF) | ((blk.scales[is - 3] >> 6) << 4);
                               m  = (blk.scales[is + 5] >> 4)  | ((blk.scales[is + 1] >> 6) << 4); }
            const float d2 = d * (float)sc;
            const float m2 = mn * (float)m;

            for (int l = 0; l < 32; ++l) {
                // The CPU mat_vec_q4_K() reference accumulates `sum += t*x` WITHOUT FMA contraction
                // (GCC keeps a separate mul+add here, rounding the product), so match it with an
                // explicit __fmul_rn + __fadd_rn instead of __fmaf_rn to stay bit-exact.
                const float t = __fmaf_rn(d1, (float)(q[l] & 0xF), -m1);
                sum = __fadd_rn(sum, __fmul_rn(t, x[cidx]));
                cidx++;
            }
            for (int l = 0; l < 32; ++l) {
                const float t = __fmaf_rn(d2, (float)(q[l] >> 4), -m2);
                sum = __fadd_rn(sum, __fmul_rn(t, x[cidx]));
                cidx++;
            }
            q += 32;
            is += 2;
        }
    }
    y[row] = sum;
}

// Q6_K mat-vec. Reproduces dequantize_q6_K() indexing exactly (including the sub-block
// scale array layout) so the output projection matches the CPU oracle token-for-token.
__global__ void mul_mat_vec_q6_K_kernel(const block_q6_K * __restrict__ w,
                                        const float * __restrict__ x,
                                        float * __restrict__ y,
                                        int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nblk = in_dim / QK_K;
    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;

    float acc = 0.0f;
    for (int c = tid; c < in_dim; c += blockDim.x) {
        const block_q6_K &b = rw[c / QK_K];
        const int p = c % QK_K;
        const int chunk  = p / 128;
        const int p128   = p % 128;
        const int l      = p128 % 32;
        const int slot   = p128 / 32;
        const float d    = h2f(b.d);
        const uint8_t ql_byte = b.ql[chunk * 64 + l + ((slot & 1) ? 32 : 0)];
        const uint8_t qh_byte = b.qh[chunk * 32 + l];
        const int qn = ((slot < 2) ? (ql_byte & 0xF) : (ql_byte >> 4)) |
                       (((qh_byte >> (slot * 2)) & 3) << 4);
        const int8_t q = (int8_t)(qn - 32);
        const int8_t s = b.scales[chunk * 8 + (l / 16) + slot * 2];
        acc += (float)d * (float)s * (float)q * x[c];
    }

    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// ---------------------------------------------------------------------------
// Warp-per-row Fast Mat-Vec Kernels (Warp Shuffles, Register-Only Accumulation)
// ---------------------------------------------------------------------------

__global__ void mul_mat_vec_q8_0_fast_kernel(const block_q8_0 * __restrict__ w,
                                             const float * __restrict__ x,
                                             float * __restrict__ y,
                                             int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_x[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_x[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const int nblk = in_dim / QK8_0;
        const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            const block_q8_0 &b0 = rw[b + 0];
            const block_q8_0 &b1 = rw[b + 1];
            const block_q8_0 &b2 = rw[b + 2];
            const block_q8_0 &b3 = rw[b + 3];

            float d0 = (lane_id == 0) ? h2f(b0.d) : 0.0f;
            float d1 = (lane_id == 0) ? h2f(b1.d) : 0.0f;
            float d2 = (lane_id == 0) ? h2f(b2.d) : 0.0f;
            float d3 = (lane_id == 0) ? h2f(b3.d) : 0.0f;

            d0 = __shfl_sync(0xffffffff, d0, 0);
            d1 = __shfl_sync(0xffffffff, d1, 0);
            d2 = __shfl_sync(0xffffffff, d2, 0);
            d3 = __shfl_sync(0xffffffff, d3, 0);

            sum += ((float)b0.qs[lane_id] * s_x[(b + 0) * QK8_0 + lane_id]) * d0;
            sum += ((float)b1.qs[lane_id] * s_x[(b + 1) * QK8_0 + lane_id]) * d1;
            sum += ((float)b2.qs[lane_id] * s_x[(b + 2) * QK8_0 + lane_id]) * d2;
            sum += ((float)b3.qs[lane_id] * s_x[(b + 3) * QK8_0 + lane_id]) * d3;
        }
        for (; b < nblk; ++b) {
            const block_q8_0 &blk = rw[b];
            float d = (lane_id == 0) ? h2f(blk.d) : 0.0f;
            d = __shfl_sync(0xffffffff, d, 0);
            sum += ((float)blk.qs[lane_id] * s_x[b * QK8_0 + lane_id]) * d;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__global__ void mul_mat_vec_f32_fast_kernel(const float * __restrict__ w,
                                            const float * __restrict__ x,
                                            float * __restrict__ y,
                                            int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xf[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xf[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const float * __restrict__ rw = w + (size_t)row * in_dim;
        float sum = 0.0f;
        for (int c = lane_id; c < in_dim; c += 32) {
            sum += rw[c] * s_xf[c];
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__global__ void mul_mat_vec_q6_K_fast_kernel(const block_q6_K * __restrict__ w,
                                             const float * __restrict__ x,
                                             float * __restrict__ y,
                                             int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xq6[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xq6[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const int nblk = in_dim >> 8;
        const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;

        float acc = 0.0f;
        const int l = lane_id & 31;
        const int l_div_16 = l >> 4;

        for (int c = lane_id; c < in_dim; c += 32) {
            const block_q6_K &b = rw[c >> 8];
            const int p = c & 255;
            const int chunk  = p >> 7;
            const int slot   = (p & 127) >> 5;
            const float d    = h2f(b.d);
            const uint8_t ql_byte = b.ql[(chunk << 6) + l + ((slot & 1) ? 32 : 0)];
            const uint8_t qh_byte = b.qh[(chunk << 5) + l];
            const int qn = ((slot < 2) ? (ql_byte & 0xF) : (ql_byte >> 4)) |
                           (((qh_byte >> (slot * 2)) & 3) << 4);
            const int8_t q = (int8_t)(qn - 32);
            const int8_t s = b.scales[(chunk << 3) + l_div_16 + (slot << 1)];
            acc += d * (float)s * (float)q * s_xq6[c];
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, mask);
        }

        if (lane_id == 0) {
            y[row] = acc;
        }
    }
}

// H9 (defined with the q8_1 machinery further down): __dp4a mmvq form of the dense Q8_0 GEMV.
void gpu_mmvq_q8_0_dense(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim, int accum);
void gpu_mmvq_q8_0_dense_dual(const block_q8_0 *w0, float *y0, int out0,
                              const block_q8_0 *w1, float *y1, int out1,
                              const float *x, int in_dim);

void launch_mat_vec_q8_0(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim) {
    if (!tuff_exact()) { gpu_mmvq_q8_0_dense(w, x, y, in_dim, out_dim, 0); return; }
    dim3 block(32, 8);
    dim3 grid((out_dim + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_q8_0_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
    CUDA_CHECK_LAST();
}

__global__ void mul_mat_vec_q8_0_accum_fast_kernel(const block_q8_0 * __restrict__ w,
                                                   const float * __restrict__ x,
                                                   float * __restrict__ y,
                                                   int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xa[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xa[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const int nblk = in_dim / QK8_0;
        const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            const block_q8_0 &b0 = rw[b + 0];
            const block_q8_0 &b1 = rw[b + 1];
            const block_q8_0 &b2 = rw[b + 2];
            const block_q8_0 &b3 = rw[b + 3];

            float d0 = (lane_id == 0) ? h2f(b0.d) : 0.0f;
            float d1 = (lane_id == 0) ? h2f(b1.d) : 0.0f;
            float d2 = (lane_id == 0) ? h2f(b2.d) : 0.0f;
            float d3 = (lane_id == 0) ? h2f(b3.d) : 0.0f;

            d0 = __shfl_sync(0xffffffff, d0, 0);
            d1 = __shfl_sync(0xffffffff, d1, 0);
            d2 = __shfl_sync(0xffffffff, d2, 0);
            d3 = __shfl_sync(0xffffffff, d3, 0);

            sum += ((float)b0.qs[lane_id] * s_xa[(b + 0) * QK8_0 + lane_id]) * d0;
            sum += ((float)b1.qs[lane_id] * s_xa[(b + 1) * QK8_0 + lane_id]) * d1;
            sum += ((float)b2.qs[lane_id] * s_xa[(b + 2) * QK8_0 + lane_id]) * d2;
            sum += ((float)b3.qs[lane_id] * s_xa[(b + 3) * QK8_0 + lane_id]) * d3;
        }
        for (; b < nblk; ++b) {
            const block_q8_0 &blk = rw[b];
            float d = (lane_id == 0) ? h2f(blk.d) : 0.0f;
            d = __shfl_sync(0xffffffff, d, 0);
            sum += ((float)blk.qs[lane_id] * s_xa[b * QK8_0 + lane_id]) * d;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] += sum;
        }
    }
}

void launch_mat_vec_q8_0_accum(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim) {
    if (!tuff_exact()) { gpu_mmvq_q8_0_dense(w, x, y, in_dim, out_dim, 1); return; }
    dim3 block(32, 8);
    dim3 grid((out_dim + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_q8_0_accum_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
    CUDA_CHECK_LAST();
}

__global__ void mul_mat_vec_q8_0_dual_kernel(const block_q8_0 * __restrict__ w0, float * __restrict__ y0, int out0,
                                             const block_q8_0 * __restrict__ w1, float * __restrict__ y1, int out1,
                                             const float * __restrict__ x, int in_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int global_row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_x[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_x[i] = x[i];
    }
    __syncthreads();

    if (global_row < out0 + out1) {
        const bool is_first = (global_row < out0);
        const int row = is_first ? global_row : (global_row - out0);
        const block_q8_0 * __restrict__ w = is_first ? w0 : w1;
        float * __restrict__ y = is_first ? y0 : y1;

        const int nblk = in_dim / QK8_0;
        const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            const block_q8_0 &b0 = rw[b + 0];
            const block_q8_0 &b1 = rw[b + 1];
            const block_q8_0 &b2 = rw[b + 2];
            const block_q8_0 &b3 = rw[b + 3];

            float d0 = (lane_id == 0) ? h2f(b0.d) : 0.0f;
            float d1 = (lane_id == 0) ? h2f(b1.d) : 0.0f;
            float d2 = (lane_id == 0) ? h2f(b2.d) : 0.0f;
            float d3 = (lane_id == 0) ? h2f(b3.d) : 0.0f;

            d0 = __shfl_sync(0xffffffff, d0, 0);
            d1 = __shfl_sync(0xffffffff, d1, 0);
            d2 = __shfl_sync(0xffffffff, d2, 0);
            d3 = __shfl_sync(0xffffffff, d3, 0);

            sum += ((float)b0.qs[lane_id] * s_x[(b + 0) * QK8_0 + lane_id]) * d0;
            sum += ((float)b1.qs[lane_id] * s_x[(b + 1) * QK8_0 + lane_id]) * d1;
            sum += ((float)b2.qs[lane_id] * s_x[(b + 2) * QK8_0 + lane_id]) * d2;
            sum += ((float)b3.qs[lane_id] * s_x[(b + 3) * QK8_0 + lane_id]) * d3;
        }
        for (; b < nblk; ++b) {
            const block_q8_0 &blk = rw[b];
            float d = (lane_id == 0) ? h2f(blk.d) : 0.0f;
            d = __shfl_sync(0xffffffff, d, 0);
            sum += ((float)blk.qs[lane_id] * s_x[b * QK8_0 + lane_id]) * d;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

// TUFF_MMVQ_DUAL=0 keeps the f32-activation kernel for the paired projections (delta
// qkv+gate, shared expert gate+up) — the ones whose output feeds the recurrent state, where
// activation-quantization error compounds multiplicatively through decay. Diagnostic for the
// N = 60 -> 56 trade measured in iteration 39. Cached once and shared (H11 needs to know whether
// the shared expert produced a q8_1 of norm_x the routed experts can reuse).
extern "C" int mmvq_dual_enabled(void) {
    static int mmvq_dual = -1;
    if (mmvq_dual < 0) { const char *e = getenv("TUFF_MMVQ_DUAL"); mmvq_dual = (e && e[0] == '0') ? 0 : 1; }
    return mmvq_dual;
}

void launch_mat_vec_q8_0_dual(const block_q8_0 *w0, float *y0, int out0,
                                    const block_q8_0 *w1, float *y1, int out1,
                                    const float *x, int in_dim) {
    if (!tuff_exact() && mmvq_dual_enabled()) { gpu_mmvq_q8_0_dense_dual(w0, y0, out0, w1, y1, out1, x, in_dim); return; }
    dim3 block(32, 8);
    dim3 grid((out0 + out1 + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_q8_0_dual_kernel<<<grid, block, smem>>>(w0, y0, out0, w1, y1, out1, x, in_dim);
    CUDA_CHECK_LAST();
}

__global__ void mul_mat_vec_f32_dual_kernel(const float * __restrict__ w0, float * __restrict__ y0, int out0,
                                            const float * __restrict__ w1, float * __restrict__ y1, int out1,
                                            const float * __restrict__ x, int in_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int global_row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xf[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xf[i] = x[i];
    }
    __syncthreads();

    if (global_row < out0 + out1) {
        const bool is_first = (global_row < out0);
        const int row = is_first ? global_row : (global_row - out0);
        const float * __restrict__ w = is_first ? w0 : w1;
        float * __restrict__ y = is_first ? y0 : y1;

        const float * __restrict__ rw = w + (size_t)row * in_dim;
        float sum = 0.0f;
        for (int c = lane_id; c < in_dim; c += 32) {
            sum += rw[c] * s_xf[c];
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

void launch_mat_vec_f32_dual(const float *w0, float *y0, int out0,
                                   const float *w1, float *y1, int out1,
                                   const float *x, int in_dim) {
    dim3 block(32, 8);
    dim3 grid((out0 + out1 + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_f32_dual_kernel<<<grid, block, smem>>>(w0, y0, out0, w1, y1, out1, x, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mat_vec_f32(const float *w, const float *x, float *y, int in_dim, int out_dim) {
    dim3 block(32, 8);
    dim3 grid((out_dim + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_f32_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
    CUDA_CHECK_LAST();
}


void gpu_mat_vec(float *dst, const tensor_t *w, const float *src) {
    const int in_dim  = (int)w->info.dims[0];
    const int out_dim = (int)w->info.dims[1];

    if (!w->dev) { // not uploaded -> CPU path (experts, token_embd, output, norms)
        mat_vec_cpu(dst, w, src);
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(g_dx, src, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    switch ((int)w->info.type) {
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>(static_cast<const block_q8_0 *>(w->dev), g_dx, g_dy, in_dim);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_kernel<<<out_dim, 256>>>(static_cast<const block_q6_K *>(w->dev), g_dx, g_dy, in_dim);
            break;
        case GGML_TYPE_F32:
            mul_mat_vec_f32_kernel<<<out_dim, 256>>>(static_cast<const float *>(w->dev), g_dx, g_dy, in_dim);
            break;
        default:
            fprintf(stderr, "[FATAL CUDA ERROR] gpu_mat_vec: unsupported type %d (in_dim=%d out_dim=%d)\n",
                    (int)w->info.type, in_dim, out_dim);
            exit(EXIT_FAILURE);
    }
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(dst, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-9: run the four delta-net projections (attn_qkv Q8_0 2048->8192, attn_gate Q8_0
// 2048->4096, ssm_beta F32 2048->32, ssm_alpha F32 2048->32) with a single upload of the shared
// input norm_x and back-to-back launches, then download all four results. Each output element is
// still produced by the exact same kernel / accumulation order as gpu_mat_vec, so the result is
// bit-identical; we only remove the 3 redundant norm_x uploads and their serialized syncs.

__global__ void mul_mat_vec_q6_K_scalar_kernel(const block_q6_K * __restrict__ w,
                                               const float * __restrict__ x,
                                               float * __restrict__ y,
                                               int in_dim) {
    const int row = blockIdx.x;
    const int nblk = in_dim / QK_K;
    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;

    float sum = 0.0f;
    for (int c = 0; c < in_dim; ++c) {
        const block_q6_K &b = rw[c / QK_K];
        const int chunk = (c % QK_K) / 128;
        const int p128  = (c % QK_K) % 128;
        const int l     = p128 % 32;
        const int slot  = p128 / 32;
        const uint8_t ql = b.ql[chunk * 64 + l + ((slot & 1) ? 32 : 0)];
        const uint8_t qh = b.qh[chunk * 32 + l];
        const uint8_t qn = ((slot < 2) ? (ql & 0xF) : (ql >> 4)) |
                           (((qh >> (slot * 2)) & 3) << 4);
        const int8_t q = (int8_t)(qn - 32);
        const int8_t s = b.scales[chunk * 8 + (l / 16) + slot * 2];
        const float yv = h2f(b.d) * (float)s * (float)q;
        // __fmul_rn + __fadd_rn force NO fma, matching the CPU dequant+dot reference exactly.
        sum = __fadd_rn(sum, __fmul_rn(yv, x[c]));
    }
    y[row] = sum;
}

// Q6_K expert mat-vec (the routed down experts are Q6_K). Reuses the existing Q6_K kernel
// (row-parallel, 256 threads), which already matches the CPU oracle for the Q6_K output
// projection. in_dim x out_dim block_q6_K matrix in dev_w.

__global__ void quantize_q8_1_kernel(const float * __restrict__ x, block_q8_1_c * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // No early return: every lane must reach the shuffles below, so out-of-range lanes
    // contribute a neutral 0 and simply skip the store.
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


// QK8_0 == QK8_1 == 32, so weight block k pairs with activation block k.
template <int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_warprow_kernel(
        const block_q8_0 * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;                 // 4 lanes cover one block
    constexpr int blocks_per_iter = vdr * 32 / QI8_0; // 8 blocks per warp iteration

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q8_0_q8_1_c(rw + kbx, y + kbx, kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        if (accum) dst[row] += tmp;
        else       dst[row]  = tmp;
    }
}

// H7 stage 1: the same kernel over NCOLS activation vectors, reading each weight byte once.
// This is the whole premise of speculative decoding on this engine — if the second token is not
// close to free here, the trunk cannot verify γ+1 tokens for the price of one weight stream and
// H7 is dead before the expensive parts. Measured by --bench-dense-batch.
template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_ncols_kernel(
        const block_q8_0 * __restrict__ w,
        const block_q8_1_c * __restrict__ y,   // ncols activation rows, nblk blocks each
        float * __restrict__ dst,              // ncols output rows, out_dim each
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) tmp[c] = 0.0f;

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_0 *bw = rw + kbx;      // one weight block, reused by every column
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            tmp[c] += vec_dot_q8_0_q8_1_c(bw, y + (size_t)c * nblk + kbx, kqs);
        }
    }
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float s = warp_reduce_sum_f(tmp[c]);
        if (threadIdx.x == 0) dst[(size_t)c * out_dim + row] = s;
    }
}

block_q8_1_c *dense_quant_buf_ensure(int in_dim) {
    (void)in_dim;
    if (!g_d_dense_x_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_dense_x_q8, (size_t)(8192 / QK8_1) * sizeof(block_q8_1_c)));
    }
    return (block_q8_1_c *)g_d_dense_x_q8;
}

block_q8_1_c *norm_x_q8_ensure(int in_dim) {
    (void)in_dim;
    if (!g_d_norm_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_norm_q8, (size_t)(8192 / QK8_1) * sizeof(block_q8_1_c)));
    }
    return (block_q8_1_c *)g_d_norm_q8;
}

extern "C" void *gpu_get_d_dense_x_q8(void) {
    return (void*)dense_quant_buf_ensure(2048);
}

extern "C" void *gpu_get_d_norm_q8(void) {
    return (void*)norm_x_q8_ensure(2048);
}

extern "C" void gpu_set_norm_q8_valid(int v) {
    g_norm_q8_valid = v;
}

block_q8_1_c *dense_quantize_x(const float *x, int in_dim) {
    if (!g_d_dense_x_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_dense_x_q8, (size_t)(8192 / QK8_1) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq = (block_q8_1_c *)g_d_dense_x_q8;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(x, xq, in_dim);
    CUDA_CHECK_LAST();
    return xq;
}

// H11: quantize x into a caller-provided q8_1 buffer (used for the shared-expert norm_x, which
// the routed experts then reuse instead of re-quantizing). Same kernel, same layout as
// dense_quantize_x, only the destination differs.
block_q8_1_c *dense_quantize_x_to(const float *x, int in_dim, void *dst) {
    block_q8_1_c *xq = (block_q8_1_c *)dst;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(x, xq, in_dim);
    CUDA_CHECK_LAST();
    return xq;
}

void mmvq_q8_0_rows(const block_q8_0 *w, const block_q8_1_c *xq, float *y,
                           int in_dim, int out_dim, bool accum) {
    const int nblk = in_dim / QK8_0;
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    if (accum) mmvq_q8_0_dense_warprow_kernel<MMVQ_ROWS_PER_BLOCK, true ><<<grid, block>>>(w, xq, y, out_dim, nblk);
    else       mmvq_q8_0_dense_warprow_kernel<MMVQ_ROWS_PER_BLOCK, false><<<grid, block>>>(w, xq, y, out_dim, nblk);
    CUDA_CHECK_LAST();
}

void gpu_mmvq_q8_0_dense(const block_q8_0 *w, const float *x, float *y,
                         int in_dim, int out_dim, int accum) {
    mmvq_q8_0_rows(w, dense_quantize_x(x, in_dim), y, in_dim, out_dim, accum != 0);
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_dual_warprow_kernel(
        const block_q8_0 * __restrict__ w0,
        float * __restrict__ y0,
        int out0,
        const block_q8_0 * __restrict__ w1,
        float * __restrict__ y1,
        int out1,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    if (global_row >= out0 + out1) return;

    const bool is_w0 = global_row < out0;
    const int row = is_w0 ? global_row : (global_row - out0);
    const block_q8_0 * __restrict__ rw = is_w0 ? (w0 + (size_t)row * nblk) : (w1 + (size_t)row * nblk);
    float * __restrict__ dst = is_w0 ? y0 : y1;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q8_0_q8_1_c(rw + kbx, xq + kbx, kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 4)
__global__ void mmvq_q8_0_dense_delta_quad_warprow_kernel(
        const block_q8_0 * __restrict__ w_qkv,
        float * __restrict__ y_qkv,
        int out_qkv,
        const block_q8_0 * __restrict__ w_z,
        float * __restrict__ y_z,
        int out_z,
        const float * __restrict__ w_beta,
        float * __restrict__ y_beta,
        int out_beta,
        const float * __restrict__ w_alpha,
        float * __restrict__ y_alpha,
        int out_alpha,
        const block_q8_1_c * __restrict__ xq,
        const float * __restrict__ x_f32,
        int nblk,
        int in_dim) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    const int total_out = out_qkv + out_z + out_beta + out_alpha;
    if (global_row >= total_out) return;

    if (global_row < out_qkv + out_z) {
        const bool is_qkv = (global_row < out_qkv);
        const int row = is_qkv ? global_row : (global_row - out_qkv);
        const block_q8_0 * __restrict__ rw = is_qkv ? (w_qkv + (size_t)row * nblk) : (w_z + (size_t)row * nblk);
        float * __restrict__ dst = is_qkv ? y_qkv : y_z;

        const int kqs = vdr * (threadIdx.x % tpsb);
        float tmp = 0.0f;
        #pragma unroll 4
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            tmp += vec_dot_q8_0_q8_1_c(rw + kbx, xq + kbx, kqs);
        }
        tmp = warp_reduce_sum_f(tmp);
        if (threadIdx.x == 0) dst[row] = tmp;
    } else {
        const int rem = global_row - (out_qkv + out_z);
        const bool is_beta = (rem < out_beta);
        const int row = is_beta ? rem : (rem - out_beta);
        const float * __restrict__ rw = is_beta ? (w_beta + (size_t)row * in_dim) : (w_alpha + (size_t)row * in_dim);
        float * __restrict__ dst = is_beta ? y_beta : y_alpha;

        const float4 * __restrict__ rw4 = reinterpret_cast<const float4*>(rw);
        const float4 * __restrict__ x4  = reinterpret_cast<const float4*>(x_f32);
        const int n_f4 = in_dim / 4;

        float tmp = 0.0f;
        #pragma unroll 4
        for (int i = threadIdx.x; i < n_f4; i += 32) {
            float4 w_val = rw4[i];
            float4 x_val = x4[i];
            tmp += w_val.x * x_val.x + w_val.y * x_val.y + w_val.z * x_val.z + w_val.w * x_val.w;
        }
        tmp = warp_reduce_sum_f(tmp);
        if (threadIdx.x == 0) dst[row] = tmp;
    }
}

int tuff_h49(void) {
    const char *e = getenv("TUFF_H49");
    return (e && e[0] == '1') ? 1 : 0;
}

int tuff_h50(void) {
    const char *e = getenv("TUFF_H50");
    return (e && e[0] == '0') ? 0 : 1;
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_triple_warprow_kernel(
        const block_q8_0 * __restrict__ w0,
        float * __restrict__ y0,
        int out0,
        const block_q8_0 * __restrict__ w1,
        float * __restrict__ y1,
        int out1,
        const block_q8_0 * __restrict__ w2,
        float * __restrict__ y2,
        int out2,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    if (global_row >= out0 + out1 + out2) return;

    const block_q8_0 * __restrict__ rw;
    float * __restrict__ dst;
    int row;
    if (global_row < out0) {
        row = global_row;
        rw = w0 + (size_t)row * nblk;
        dst = y0;
    } else if (global_row < out0 + out1) {
        row = global_row - out0;
        rw = w1 + (size_t)row * nblk;
        dst = y1;
    } else {
        row = global_row - (out0 + out1);
        rw = w2 + (size_t)row * nblk;
        dst = y2;
    }

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q8_0_q8_1_c(rw + kbx, xq + kbx, kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

// Shared expert fused gate+up+swiglu kernel: calculates both projections and SwiGLU in one pass
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_shexp_swiglu_kernel(
        const block_q8_0 * __restrict__ w_gate,
        const block_q8_0 * __restrict__ w_up,
        float * __restrict__ dst_act,
        int sh_ffn_dim,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= sh_ffn_dim) return;

    const block_q8_0 * __restrict__ rw_g = w_gate + (size_t)row * nblk;
    const block_q8_0 * __restrict__ rw_u = w_up   + (size_t)row * nblk;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp_g = 0.0f;
    float tmp_u = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c *bx = xq + kbx;
        tmp_g += vec_dot_q8_0_q8_1_c(rw_g + kbx, bx, kqs);
        tmp_u += vec_dot_q8_0_q8_1_c(rw_u + kbx, bx, kqs);
    }
    tmp_g = warp_reduce_sum_f(tmp_g);
    tmp_u = warp_reduce_sum_f(tmp_u);
    if (threadIdx.x == 0) {
        float silu_g = tmp_g / (1.0f + expf(-tmp_g));
        dst_act[row] = silu_g * tmp_u;
    }
}

// Shared expert fused gate+up+swiglu+quantize kernel: calculates projections, SwiGLU, and Q8_1 quantization directly in registers (H50)
__launch_bounds__(1024, 1)
__global__ void mmvq_q8_0_dense_shexp_swiglu_quant_kernel(
        const block_q8_0 * __restrict__ w_gate,
        const block_q8_0 * __restrict__ w_up,
        block_q8_1_c * __restrict__ dst_q8,
        int sh_ffn_dim,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int row = block_idx * 32 + warp_id;

    __shared__ float s_act[32];

    if (row < sh_ffn_dim) {
        const block_q8_0 * __restrict__ rw_g = w_gate + (size_t)row * nblk;
        const block_q8_0 * __restrict__ rw_u = w_up   + (size_t)row * nblk;

        const int kqs = vdr * (lane_id % tpsb);
        float tmp_g = 0.0f;
        float tmp_u = 0.0f;
        for (int kbx = lane_id / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            const block_q8_1_c *bx = xq + kbx;
            tmp_g += vec_dot_q8_0_q8_1_c(rw_g + kbx, bx, kqs);
            tmp_u += vec_dot_q8_0_q8_1_c(rw_u + kbx, bx, kqs);
        }
        tmp_g = warp_reduce_sum_f(tmp_g);
        tmp_u = warp_reduce_sum_f(tmp_u);
        if (lane_id == 0) {
            float silu_g = tmp_g / (1.0f + expf(-tmp_g));
            s_act[warp_id] = silu_g * tmp_u;
        }
    } else {
        if (lane_id == 0) s_act[warp_id] = 0.0f;
    }
    __syncthreads();

    if (warp_id == 0) {
        float act_val = s_act[lane_id];
        float amax = fabsf(act_val);
        float sum = act_val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float d = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(act_val / d);
        dst_q8[block_idx].qs[lane_id] = q;
        if (lane_id == 0) {
            dst_q8[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// Two matrices sharing one activation: quantize once, single unified grid launch.
void gpu_mmvq_q8_0_dense_dual(const block_q8_0 *w0, float *y0, int out0,
                              const block_q8_0 *w1, float *y1, int out1,
                              const float *x, int in_dim) {
    const block_q8_1_c *xq = dense_quantize_x(x, in_dim);
    const int nblk = in_dim / QK8_0;
    const int total_out = out0 + out1;
    dim3 grid((unsigned)((total_out + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_dual_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block>>>(w0, y0, out0, w1, y1, out1, xq, nblk);
    CUDA_CHECK_LAST();
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q6_K_dense_warprow_kernel(
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
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) dst[row] = tmp;
}


// H7 stage 2a: the output head over `ncols` activation columns. 398 MB is the single largest
// read in the step, and both a draft and a verify need logits for more than one position; read
// once, score every column. Same argument as the Q8_0 form — see --bench-dense-batch.
template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q6_K_dense_ncols_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,   // ncols activation rows, nblk*(QK_K/QK8_1) blocks each
        float * __restrict__ dst,              // ncols rows of out_dim
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;
    constexpr int y_per_blk = QK_K / QK8_1;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) tmp[c] = 0.0f;

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q6_K *bw = rw + kbx;
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            tmp[c] += vec_dot_q6_K_q8_1_c(bw, y + (size_t)c * nblk * y_per_blk + kbx * y_per_blk, kqs);
        }
    }
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float s = warp_reduce_sum_f(tmp[c]);
        if (threadIdx.x == 0) dst[(size_t)c * out_dim + row] = s;
    }
}

void launch_mat_vec_q6_K(const block_q6_K *w, const float *x, float *y, int in_dim, int out_dim) {
    if (tuff_exact()) {
        dim3 block(32, 16);
        dim3 grid((out_dim + 15) / 16);
        size_t smem = (size_t)in_dim * sizeof(float);
        mul_mat_vec_q6_K_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
        CUDA_CHECK_LAST();
        return;
    }

    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / 32) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(x, xq8, in_dim);
    CUDA_CHECK_LAST();

    const int nblk = in_dim / QK_K;
    dim3 grid_dim((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block_dim(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q6_K_dense_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid_dim, block_dim>>>(w, xq8, y, out_dim, nblk);
    CUDA_CHECK_LAST();
}


static void shared_proj_launch(const tensor_t *w, const float *xin, float *yout, int in_dim) {
    const int out_dim = (int)w->info.dims[1];
    switch ((int)w->info.type) {
        case GGML_TYPE_Q8_0: mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>((const block_q8_0*)w->dev, xin, yout, in_dim); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q6_K_kernel<<<out_dim, 256>>>((const block_q6_K*)w->dev, xin, yout, in_dim); break;
        case GGML_TYPE_F32:  mul_mat_vec_f32_kernel <<<out_dim, 256>>>((const float*)w->dev, xin, yout, in_dim); break;
        default:
            fprintf(stderr, "[FATAL CUDA ERROR] shared_proj_launch: unsupported type %d\n", (int)w->info.type);
            exit(EXIT_FAILURE);
    }
}

void gpu_shared_expert_projections(float *sh_gate, const tensor_t *gate_t,
                                   float *sh_up,   const tensor_t *up_t,
                                   float *sh_ginp, const tensor_t *ginp_t,
                                   const float *norm_x, int in_dim) {
    const int sh_ffn = (int)gate_t->info.dims[1];
    if (!g_shg)  CUDA_CHECK(cudaMalloc(&g_shg,  (size_t)sh_ffn * sizeof(float)));
    if (!g_shu)  CUDA_CHECK(cudaMalloc(&g_shu,  (size_t)sh_ffn * sizeof(float)));
    if (!g_shgi) CUDA_CHECK(cudaMalloc(&g_shgi, (size_t)sh_ffn * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(g_dx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    shared_proj_launch(gate_t, g_dx, g_shg, in_dim);
    shared_proj_launch(up_t,   g_dx, g_shu, in_dim);
    shared_proj_launch(ginp_t, g_dx, g_shgi, in_dim);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(sh_gate, g_shg,  (size_t)sh_ffn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sh_up,   g_shu,  (size_t)sh_ffn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sh_ginp, g_shgi, sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-13: bit-exact full-attention for Qwen3.5/3.6, offloaded from engine.c's scalar CPU
// per-head loop. Replicates the CPU math and fp accumulation order EXACTLY so the oracle stays
// 32/32 bit-identical:
//   per query head h: kv_h = h / gqa_ratio
//     for t in [0,ctx): dotp = sum_{d=0..head_dim-1} qh[d]*kt[d];  scores[t] = dotp*scale
//     softmax(scores)   -- CPU order: max over indices 0..ctx (from x[0]), then exp(x-max)
//                         accumulating sum in index order, then multiply by 1/sum
//     concat_out[h][d] = sum_{t=0..ctx-1} scores[t] * vt[t][d]    (per d, t increasing)
// One block per query head (head_dim threads). Stage A writes this token's k/v rows into the
// device cache (each block writes its own kv_h group redundantly with identical bytes, so the
// write is visible to that block's own reads below); Stage B computes scores (one t per thread,
// each an independent sequential dotp); Stage C thread 0 runs the sequential softmax; Stage D
// each d thread accumulates weighted V over t in order. Low parallelism is fine: attention is not
// the compute bottleneck; the goal is to move it off the CPU critical path. The gate sigmoid
// multiply stays on the host (elementwise, order-independent, avoids a device-expf divergence).

extern "C" void gpu_bench_dense_batch(const tensor_t *w, int in_dim, int out_dim, int reps) {
    if (!w || !w->dev) { fprintf(stderr, "[BENCH] tensor has no device copy\n"); return; }
    const int nblk = in_dim / QK8_0;
    const size_t wbytes = (size_t)out_dim * nblk * sizeof(block_q8_0);

    float *d_x = NULL; void *d_xq = NULL; float *d_y = NULL;
    CUDA_CHECK(cudaMalloc(&d_x,  (size_t)4 * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xq, (size_t)4 * nblk * sizeof(block_q8_1_c)));
    CUDA_CHECK(cudaMalloc(&d_y,  (size_t)4 * out_dim * sizeof(float)));

    float *h_x = (float*)malloc((size_t)4 * in_dim * sizeof(float));
    for (int i = 0; i < 4 * in_dim; ++i) h_x[i] = 0.02f * ((i % 97) - 48);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)4 * in_dim * sizeof(float), cudaMemcpyHostToDevice));
    for (int c = 0; c < 4; ++c) {
        quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(d_x + (size_t)c * in_dim,
                                                            (block_q8_1_c*)d_xq + (size_t)c * nblk, in_dim);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("[BENCH] dense Q8_0 projection %d x %d, %.1f MiB of weights, %d reps\n",
           in_dim, out_dim, wbytes / 1048576.0, reps);
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    double t1 = 0.0;
    for (int ncols = 1; ncols <= 4; ncols *= 2) {
        for (int warm = 0; warm < 3; ++warm) {
            if (ncols == 1) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 2) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 4) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
        }
        CUDA_CHECK(cudaStreamSynchronize(0));
        const double t0 = (double)clock() / CLOCKS_PER_SEC;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        for (int r = 0; r < reps; ++r) {
            if (ncols == 1) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 2) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 4) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
        }
        CUDA_CHECK(cudaStreamSynchronize(0));
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        (void)t0;
        const double dt = (ts1.tv_sec - ts0.tv_sec) + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
        const double us = dt / reps * 1e6;
        if (ncols == 1) t1 = us;
        printf("  ncols=%d  %8.2f us/launch  %7.1f GB/s of weights  |  per column %7.2f us  (%.2fx of ncols=1)\n",
               ncols, us, wbytes / (us * 1e-6) / 1e9, us / ncols, us / t1);
    }
    free(h_x);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_xq)); CUDA_CHECK(cudaFree(d_y));
}

// H7 stage 2a measurement: the same question for the Q6_K output head, the single largest read
// in the step (398 MB). A draft and a verify both need logits for more than one position.
extern "C" void gpu_bench_head_batch(const tensor_t *w, int in_dim, int out_dim, int reps) {
    if (!w || !w->dev) { fprintf(stderr, "[BENCH] output head has no device copy\n"); return; }
    const int nblk = in_dim / QK_K;                 // q6_K super-blocks per row
    const int yblk = nblk * (QK_K / QK8_1);         // q8_1 blocks per activation column
    const size_t wbytes = (size_t)out_dim * nblk * sizeof(block_q6_K);

    float *d_x = NULL; void *d_xq = NULL; float *d_y = NULL;
    CUDA_CHECK(cudaMalloc(&d_x,  (size_t)4 * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xq, (size_t)4 * yblk * sizeof(block_q8_1_c)));
    CUDA_CHECK(cudaMalloc(&d_y,  (size_t)4 * out_dim * sizeof(float)));

    float *h_x = (float*)malloc((size_t)4 * in_dim * sizeof(float));
    for (int i = 0; i < 4 * in_dim; ++i) h_x[i] = 0.02f * ((i % 89) - 44);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)4 * in_dim * sizeof(float), cudaMemcpyHostToDevice));
    for (int c = 0; c < 4; ++c) {
        quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(d_x + (size_t)c * in_dim,
                                                            (block_q8_1_c*)d_xq + (size_t)c * yblk, in_dim);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("[BENCH] Q6_K output head %d x %d, %.1f MiB of weights, %d reps\n",
           in_dim, out_dim, wbytes / 1048576.0, reps);
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    double t1 = 0.0;
    for (int ncols = 1; ncols <= 4; ncols *= 2) {
        for (int pass = 0; pass < 2; ++pass) {          // pass 0 = warmup, pass 1 = timed
            const int n = pass ? reps : 2;
            struct timespec ts0, ts1;
            CUDA_CHECK(cudaStreamSynchronize(0));
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            for (int r = 0; r < n; ++r) {
                if (ncols == 1) mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>((const block_q6_K*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
                if (ncols == 2) mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>((const block_q6_K*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
                if (ncols == 4) mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>((const block_q6_K*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            if (!pass) continue;
            const double dt = (ts1.tv_sec - ts0.tv_sec) + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
            const double us = dt / n * 1e6;
            if (ncols == 1) t1 = us;
            printf("  ncols=%d  %8.2f us/launch  %7.1f GB/s of weights  |  per column %7.2f us  (%.2fx of ncols=1)\n",
                   ncols, us, wbytes / (us * 1e-6) / 1e9, us / ncols, us / t1);
        }
    }
    free(h_x);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_xq)); CUDA_CHECK(cudaFree(d_y));
}

// H12: the D2H half of gpu_router_logits_launch, for layers whose router compute is already
// inside a captured graph. The copies stay outside the graph because their destinations are host
// buffers the decode thread reads immediately after the sync.

void launch_quantize_q8_1(const float *x, block_q8_1_c *y, int n, cudaStream_t stream) {
    quantize_q8_1_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
    CUDA_CHECK_LAST();
}

void launch_mat_vec_q4_K(const block_q4_K *w, const float *x, float *y, int in_dim, int out_dim, cudaStream_t stream) {
    mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, stream>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mat_vec_q6_K_scalar(const block_q6_K *w, const float *x, float *y, int in_dim, int out_dim, cudaStream_t stream) {
    mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1, 0, stream>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mul_mat_vec_q8_0_std(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim) {
    mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mul_mat_vec_f32_std(const float *w, const float *x, float *y, int in_dim, int out_dim) {
    mul_mat_vec_f32_kernel<<<out_dim, 256>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mmvq_shexp_swiglu_quant(
    const block_q8_0 *w_gate, const block_q8_0 *w_up,
    block_q8_1_c *dst_q8, int sh_ffn_dim,
    const block_q8_1_c *xq, int nblk,
    cudaStream_t stream) {
    dim3 grid((unsigned)((sh_ffn_dim + 31) / 32), 1u, 1u);
    dim3 block(32u, 32u, 1u);
    mmvq_q8_0_dense_shexp_swiglu_quant_kernel<<<grid, block, 0, stream>>>(
        w_gate, w_up, dst_q8, sh_ffn_dim, xq, nblk);
    CUDA_CHECK_LAST();
}

void launch_mmvq_q8_0_dense_ncols(const block_q8_0 *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols, cudaStream_t stream) {
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    switch (ncols) {
        case 1: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 2: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 3: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 3><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 4: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 5: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 5><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 6: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 6><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 7: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 7><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 8: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 8><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        default: break;
    }
    CUDA_CHECK_LAST();
}

void launch_mmvq_q6_K_dense_ncols(const block_q6_K *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols, cudaStream_t stream) {
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    switch (ncols) {
        case 1: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 2: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 3: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 3><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 4: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 5: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 5><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 6: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 6><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 7: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 7><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 8: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 8><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        default: break;
    }
    CUDA_CHECK_LAST();
}

void launch_mmvq_delta_quad(const block_q8_0 *qkv_w, float *g_dpqkv, int out_qkv,
                           const block_q8_0 *z_w, float *g_dpz, int out_z,
                           const float *beta_w, float *g_dpb, int out_beta,
                           const float *alpha_w, float *g_dpa, int out_alpha,
                           const block_q8_1_c *xq, const float *d_norm_x, int nblk, int in_dim) {
    const int total_out = out_qkv + out_z + out_beta + out_alpha;
    dim3 grid((unsigned)((total_out + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_delta_quad_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block>>>(
        qkv_w, g_dpqkv, out_qkv,
        z_w, g_dpz, out_z,
        beta_w, g_dpb, out_beta,
        alpha_w, g_dpa, out_alpha,
        xq, d_norm_x, nblk, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mmvq_dense_triple(const block_q8_0 *q_dev, float *d_q_full, int q_full_dim,
                             const block_q8_0 *k_dev, float *d_k, int k_dim,
                             const block_q8_0 *v_dev, float *d_v, int v_dim,
                             const block_q8_1_c *xq, int nblk) {
    const int total_qkv = q_full_dim + k_dim + v_dim;
    dim3 qkv_grid((unsigned)((total_qkv + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 qkv_block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>(
        q_dev, d_q_full, q_full_dim,
        k_dev, d_k, k_dim,
        v_dev, d_v, v_dim,
        xq, nblk);
    CUDA_CHECK_LAST();
}
