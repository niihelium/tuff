#include "state.h"

void gpu_router_logits_launch(float *host_logits, float *host_norm_x, const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x, g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x, g_d_router_logits, in_dim, n_expert);
    }
    if (host_logits) {
        CUDA_CHECK(cudaMemcpyAsync(host_logits, g_d_router_logits, (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, 0));
}

extern "C" void gpu_router_logits_dev_only(const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x, g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x, g_d_router_logits, in_dim, n_expert);
    }
}

// H7: the router compute for column `col`. Column 1 lands in its own device buffer so BOTH
// columns' router work can sit inside one captured graph — with a single shared buffer the
// host had to drain column 0 before column 1 could be issued, which is exactly the launch
// serialization the graph exists to remove.
extern "C" void gpu_router_logits_dev_only_col(const tensor_t *ffn_gate_inp, const float *d_norm_x,
                                               int in_dim, int n_expert, int col) {
    if (col != 0 && !g_d_router_logits2)
        CUDA_CHECK(cudaMalloc(&g_d_router_logits2, (size_t)256 * sizeof(float)));
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x,
                            col ? g_d_router_logits2 : g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x,
                           col ? g_d_router_logits2 : g_d_router_logits, in_dim, n_expert);
    }
}

// Both columns' router logits and normalized activations, one burst of D2H after the graph.
extern "C" void gpu_router_logits_copy_out_n2(float *logits0, float *nx0, float *logits1, float *nx1,
                                              const float *d_norm_x2, int in_dim, int n_expert) {
    CUDA_CHECK(cudaMemcpyAsync(logits0, g_d_router_logits,  (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(nx0,     d_norm_x2,          (size_t)in_dim   * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(logits1, g_d_router_logits2, (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(nx1, d_norm_x2 + (size_t)in_dim, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// H7 stage 1 measurement: cost of a dense Q8_0 projection over 1, 2 and 4 activation columns.
// Weight bytes are read once per launch regardless of ncols, so if the card is bandwidth-bound
// the per-column cost should collapse. Reports wall time per launch and GB/s of weight bytes.

extern "C" void gpu_router_logits_copy_out(float *host_logits, float *host_norm_x,
                                           const float *d_norm_x, int in_dim, int n_expert) {
    if (host_logits) {
        CUDA_CHECK(cudaMemcpyAsync(host_logits, g_d_router_logits, (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, 0));
}

void gpu_router_logits_wait(void) {
    CUDA_CHECK(cudaStreamSynchronize(0));
}

// H6: wait only for the router-logits + norm_x D2H (marked by the event), not the
// whole stream. The shared expert, already ordered after the copies on stream 0, keeps
// running while the host does top-k and submits the CPU-miss work.
void gpu_router_logits_wait_event(void) {
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventSynchronize(g_router_d2h_event));
}

void gpu_router_logits_dev(float *host_logits, float *host_norm_x, const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    gpu_router_logits_launch(host_logits, host_norm_x, ffn_gate_inp, d_norm_x, in_dim, n_expert);
    gpu_router_logits_wait();
}


template <int experts_per_thread>
static __device__ void softmax_warp_inplace_f(float (&vals)[experts_per_thread], const int lane) {
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        max_val = fmaxf(max_val, vals[i]);
    }
    max_val = warp_reduce_max_f(max_val);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const float val = expf(vals[i] - max_val);
        vals[i] = val;
        sum += val;
    }
    sum = warp_reduce_sum_f(sum);
    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        vals[i] *= inv_sum;
    }
}

template <int n_experts, int n_expert_used>
__global__ void topk_moe_resident_kernel(
    const float * __restrict__ logits,
    float * __restrict__ weights,
    int * __restrict__ exslots,
    int layer_idx) {
    constexpr int experts_per_thread = (n_experts > 32) ? n_experts / 32 : 1;
    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += 32) {
        const int expert = i + threadIdx.x;
        wt[i / 32] = (expert < n_experts) ? logits[expert] : -INFINITY;
    }

    softmax_warp_inplace_f<experts_per_thread>(wt, threadIdx.x);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.0f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.0f;
    }

    for (int k = 0; k < n_expert_used; k++) {
        float max_val = wt[0];
        int max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * 32;
            if (expert < n_experts && wt[i] > max_val) {
                max_val = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val = __shfl_xor_sync(0xffffffff, max_val, mask, 32);
            const int expert = __shfl_xor_sync(0xffffffff, max_expert, mask, 32);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val = val;
                max_expert = expert;
            }
        }

        if ((max_expert & 31) == threadIdx.x) {
            wt[max_expert / 32] = -INFINITY;
            exslots[k] = layer_idx * 256 + max_expert;
            wt_sum += max_val;
        }

        if ((k & 31) == threadIdx.x) {
            output_weights[k / 32] = max_val;
        }
    }

    wt_sum = warp_reduce_sum_f(wt_sum);
    const float inv_sum = 1.0f / fmaxf(wt_sum, 1e-20f);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] *= inv_sum;
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * 32 + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i];
        }
    }
}

template <int n_experts, int n_expert_used>
__global__ void topk_moe_kernel(
    const float * __restrict__ logits,
    float * __restrict__ weights,
    int * __restrict__ indices) {
    constexpr int experts_per_thread = (n_experts > 32) ? n_experts / 32 : 1;
    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += 32) {
        const int expert = i + threadIdx.x;
        wt[i / 32] = (expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Step 1: Iterative argmax across all 256 raw logits
    float top_raw[n_expert_used];
    for (int k = 0; k < n_expert_used; k++) {
        float max_val = wt[0];
        int max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * 32;
            if (expert < n_experts && wt[i] > max_val) {
                max_val = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val = __shfl_xor_sync(0xffffffff, max_val, mask, 32);
            const int expert = __shfl_xor_sync(0xffffffff, max_expert, mask, 32);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val = val;
                max_expert = expert;
            }
        }

        if ((max_expert & 31) == threadIdx.x) {
            wt[max_expert / 32] = -INFINITY;
            indices[k] = max_expert;
        }

        top_raw[k] = max_val;
    }

    // Step 2: Exact softmax over the top-8 selected logits
    if (threadIdx.x < n_expert_used) {
        float max_l = top_raw[0];
#pragma unroll
        for (int k = 1; k < n_expert_used; k++) {
            max_l = fmaxf(max_l, top_raw[k]);
        }

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < n_expert_used; k++) {
            sum += expf(top_raw[k] - max_l);
        }

        const float inv_sum = 1.0f / sum;
        weights[threadIdx.x] = expf(top_raw[threadIdx.x] - max_l) * inv_sum;
    }
}

static cudaStream_t g_d2h_stream = NULL;
static cudaEvent_t  g_topk_done_event = NULL;

void gpu_router_topk_launch(int *host_top_indices, float *host_top_values, float *host_norm_x,
                            const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    if (!g_d2h_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_d2h_stream, cudaStreamNonBlocking));
    }
    if (!g_topk_done_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_topk_done_event, cudaEventDisableTiming));
    }
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x, g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x, g_d_router_logits, in_dim, n_expert);
    }
    topk_moe_kernel<256, 8><<<1, 32>>>(g_d_router_logits, g_d_top_values, g_d_top_indices);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaEventRecord(g_topk_done_event, 0));
    CUDA_CHECK(cudaStreamWaitEvent(g_d2h_stream, g_topk_done_event, 0));

    if (host_top_indices) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_indices, g_d_top_indices, 8 * sizeof(int), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_top_values) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_values, g_d_top_values, 8 * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, g_d2h_stream));
}

void gpu_router_topk_copy_out(int *host_top_indices, float *host_top_values, float *host_norm_x,
                              const float *d_norm_x, int in_dim) {
    if (!g_d2h_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_d2h_stream, cudaStreamNonBlocking));
    }
    if (!g_topk_done_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_topk_done_event, cudaEventDisableTiming));
    }
    topk_moe_kernel<256, 8><<<1, 32>>>(g_d_router_logits, g_d_top_values, g_d_top_indices);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaEventRecord(g_topk_done_event, 0));
    CUDA_CHECK(cudaStreamWaitEvent(g_d2h_stream, g_topk_done_event, 0));

    if (host_top_indices) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_indices, g_d_top_indices, 8 * sizeof(int), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_top_values) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_values, g_d_top_values, 8 * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, g_d2h_stream));
}



void launch_topk_moe_resident(const float *logits, float *weights, int *exslots, int layer_idx, cudaStream_t stream) {
    topk_moe_resident_kernel<256, 8><<<1, 32, 0, stream>>>(logits, weights, exslots, layer_idx);
    CUDA_CHECK_LAST();
}
