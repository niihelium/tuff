#include "state.h"

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q6_K_dense_argmax_warprow_kernel(
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
        float b_max = s_max[0];
        int   b_idx = s_idx[0];
        #pragma unroll
        for (int r = 1; r < rows_per_block; ++r) {
            if (s_max[r] > b_max) {
                b_max = s_max[r];
                b_idx = s_idx[r];
            }
        }
        block_max[blockIdx.x] = b_max;
        block_idx[blockIdx.x] = b_idx;
    }
}


__global__ void argmax_block_kernel(const float * __restrict__ x, float * __restrict__ block_max,
                                    int * __restrict__ block_idx, int n) {
    __shared__ float s_max[256];
    __shared__ int   s_idx[256];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float cur_max = -1e30f;
    int   cur_idx = 0;
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        if (v > cur_max) {
            cur_max = v;
            cur_idx = i;
        }
    }
    s_max[tid] = cur_max;
    s_idx[tid] = cur_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = s_max[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}

__global__ void argmax_final_kernel(const float * __restrict__ block_max,
                                    const int * __restrict__ block_idx,
                                    int * __restrict__ out_best_token,
                                    int n_blocks) {
    __shared__ float s_max[256];
    __shared__ int   s_idx[256];
    const int tid = threadIdx.x;

    float cur_max = -1e30f;
    int   cur_idx = 0;
    for (int i = tid; i < n_blocks; i += 256) {
        float v = block_max[i];
        if (v > cur_max) {
            cur_max = v;
            cur_idx = block_idx[i];
        }
    }
    s_max[tid] = cur_max;
    s_idx[tid] = cur_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_best_token = s_idx[0];
    }
}

#define HEAD_ROWS_PER_BLOCK 8

int32_t gpu_output_argmax_dev(const tensor_t *output_weight, const float *d_norm_x, int in_dim, int vocab_size, float *host_logits_capture) {
    const block_q8_1_c *xq = dense_quantize_x(d_norm_x, in_dim);
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    float *dst_logits = host_logits_capture ? g_d_logits : NULL;

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq, dst_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits_capture && dst_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits_capture, dst_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token, grid_blocks);
    CUDA_CHECK_LAST();

    int32_t best_token = 0;
    CUDA_CHECK(cudaMemcpy(&best_token, g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return best_token;
}

// Fused Layer RMS-Norm and Q8_1 Quantization Kernel (H48/H49)
__global__ void fused_rms_norm_quantize_q8_1_kernel(
        const float * __restrict__ x,
        const float * __restrict__ w,
        float * __restrict__ dst_norm_x,
        block_q8_1_c * __restrict__ y,
        int in_dim,
        float eps) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < in_dim; i += num_threads) {
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
    for (int b = warp_id; b < total_q_blocks; b += 8) {
        const int idx = b * QK8_1 + lane_id;
        float norm_val = x[idx] * scale * w[idx];
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

extern "C" void gpu_fused_rms_norm_quant_dev(float *d_dst_norm_x, const float *d_src, const float *d_weight, void *d_dst_q8, int dim, float eps) {
    if (tuff_h49()) {
        fused_rms_norm_quantize_q8_1_kernel<<<1, 256>>>(d_src, d_weight, d_dst_norm_x, (block_q8_1_c *)d_dst_q8, dim, eps);
    } else {
        gpu_rms_norm_dev(d_dst_norm_x, d_src, d_weight, dim, eps);
        if (d_dst_q8) {
            launch_quantize_q8_1(d_dst_norm_x, (block_q8_1_c *)d_dst_q8, dim);
        }
    }
    CUDA_CHECK_LAST();
}

int32_t gpu_fused_output_norm_quant_argmax_dev(
        const tensor_t *output_weight,
        const float *d_res_x,
        const float *d_norm_w,
        int in_dim,
        int vocab_size,
        float eps,
        float *host_logits_capture) {
    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / QK8_1) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    gpu_fused_rms_norm_quant_dev(g_d_norm_x, d_res_x, d_norm_w, xq8, in_dim, eps);

    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    float *dst_logits = host_logits_capture ? g_d_logits : NULL;

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq8, dst_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits_capture && dst_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits_capture, dst_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token, grid_blocks);
    CUDA_CHECK_LAST();

    int32_t best_token = 0;
    CUDA_CHECK(cudaMemcpy(&best_token, g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return best_token;
}

void gpu_output_logits_dev(const tensor_t *output_weight, const float *d_norm_x, int in_dim, int vocab_size, float *host_logits) {
    if (!g_d_logits) {
        CUDA_CHECK(cudaMalloc(&g_d_logits, (size_t)vocab_size * sizeof(float)));
    }
    const block_q8_1_c *xq = dense_quantize_x(d_norm_x, in_dim);
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq, g_d_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits, g_d_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void gpu_fused_output_norm_quant_logits_dev(
        const tensor_t *output_weight,
        const float *d_res_x,
        const float *d_norm_w,
        int in_dim,
        int vocab_size,
        float eps,
        float *host_logits) {
    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / QK8_1) * sizeof(block_q8_1_c)));
    }
    if (!g_d_logits) {
        CUDA_CHECK(cudaMalloc(&g_d_logits, (size_t)vocab_size * sizeof(float)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    gpu_fused_rms_norm_quant_dev(g_d_norm_x, d_res_x, d_norm_w, xq8, in_dim, eps);

    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq8, g_d_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits, g_d_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// ===========================================================================
// H7 stage 2b: the two-column (verify) trunk.
//
// Speculative decoding verifies [x_{t+1}, draft] in one pass. Both inputs are
// known before the pass starts, so the trunk can be evaluated for both
// positions while the weight stream is read once — the property measured in
// iteration 42 (a second column costs 0.06% on the Q8_0 projections and 1.46x
// on the Q6_K head).
//
// The residual and normalized streams become [2][dim]; column c lives at
// base + c*dim. That layout is what makes this affordable: every entry point
// that is already correct for one token (the GDN recurrence, the attention
// layer, the routed experts, the MoE combine) runs per column with no kernel
// change at all, by passing base + c*dim. Only the places where the weight
// stream is genuinely shared get new launchers — the delta-layer input
// projections (~1.05 GiB/token, the largest shareable read in the step) and
// the Q6_K output head (398 MB).
//
// The single-column path below is deliberately untouched: it is what the
// oracle gate pins, and speculation must not be able to perturb it.
// ===========================================================================



void launch_argmax(const float *x, float *block_max, int *block_idx, int *best_token, int vocab_size, cudaStream_t stream) {
    argmax_block_kernel<<<256, 256, 0, stream>>>(x, block_max, block_idx, vocab_size);
    argmax_final_kernel<<<1, 256, 0, stream>>>(block_max, block_idx, best_token, 256);
    CUDA_CHECK_LAST();
}

// Lossless verify: the fused output head for one column, asynchronously. Same kernels as
// gpu_fused_output_norm_quant_argmax_dev; the token lands in pinned host memory (*h_tok) when the
// stream reaches it, so column 0's head can run while the host finishes column 1's last layer.
extern "C" void gpu_fused_output_argmax_async(const tensor_t *output_weight, const float *d_res_x, const float *d_norm_w,
                                              int in_dim, int vocab_size, float eps, int32_t *h_tok) {
    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / QK8_1) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    gpu_fused_rms_norm_quant_dev(g_d_norm_x, d_res_x, d_norm_w, xq8, in_dim, eps);
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);
    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq8, NULL, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();
    argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token, grid_blocks);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpyAsync(h_tok, g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost, 0));
}
extern "C" int32_t *gpu_pinned_tokens(void) {
    static int32_t *p = NULL;
    if (!p) CUDA_CHECK(cudaMallocHost(&p, 8 * sizeof(int32_t)));
    return p;
}
extern "C" void gpu_stream_sync(void) { CUDA_CHECK(cudaStreamSynchronize(0)); }
