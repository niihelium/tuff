#include "state.h"

extern "C" float *gpu_get_d_res_batch(void)   { return g_d_res_batch; }
extern "C" float *gpu_get_d_norm_batch(void)  { return g_d_norm_batch; }
extern "C" float *gpu_get_d_shexp_batch(void) { return g_d_shexp_batch; }
extern "C" float *gpu_get_d_moe_buf_batch(void) { return g_d_moe_buf_batch; }

extern "C" void gpu_batch_prepare(int dim, int vocab_size, int max_batch) {
    if (g_batch_dim == dim && g_d_res_batch) return;
    g_batch_dim = dim;
    if (!g_d_res_batch)        CUDA_CHECK(cudaMalloc(&g_d_res_batch,        (size_t)max_batch * dim * sizeof(float)));
    if (!g_d_norm_batch)       CUDA_CHECK(cudaMalloc(&g_d_norm_batch,       (size_t)max_batch * dim * sizeof(float)));
    if (!g_d_nx_batch_q8)      CUDA_CHECK(cudaMalloc(&g_d_nx_batch_q8,      (size_t)max_batch * (dim / QK8_1) * sizeof(block_q8_1_c)));
    if (!g_d_logits_batch)     CUDA_CHECK(cudaMalloc(&g_d_logits_batch,     (size_t)max_batch * vocab_size * sizeof(float)));
    if (!g_d_best_token_batch) CUDA_CHECK(cudaMalloc(&g_d_best_token_batch, (size_t)max_batch * sizeof(int32_t)));
    if (!g_d_shexp_batch)      CUDA_CHECK(cudaMalloc(&g_d_shexp_batch,      (size_t)max_batch * dim * sizeof(float)));
    if (!g_d_moe_buf_batch)    CUDA_CHECK(cudaMalloc(&g_d_moe_buf_batch,    (size_t)max_batch * dim * sizeof(float)));

    const int out_qkv   = 8192;
    const int out_z     = 4096;
    const int out_beta  = 32;
    const int out_alpha = 32;
    if (!g_d_p_batch_qkv) CUDA_CHECK(cudaMalloc(&g_d_p_batch_qkv, (size_t)max_batch * out_qkv   * sizeof(float)));
    if (!g_d_p_batch_z)   CUDA_CHECK(cudaMalloc(&g_d_p_batch_z,   (size_t)max_batch * out_z     * sizeof(float)));
    if (!g_d_p_batch_b)   CUDA_CHECK(cudaMalloc(&g_d_p_batch_b,   (size_t)max_batch * out_beta  * sizeof(float)));
    if (!g_d_p_batch_a)   CUDA_CHECK(cudaMalloc(&g_d_p_batch_a,   (size_t)max_batch * out_alpha * sizeof(float)));

    for (int i = 0; i < max_batch - 1; ++i) {
        if (!g_dstate_snap_batch[i] && g_dstate_bytes > 0)
            CUDA_CHECK(cudaMallocHost(&g_dstate_snap_batch[i], g_dstate_bytes));
        if (!g_dconv_snap_batch[i] && g_dconv_bytes > 0)
            CUDA_CHECK(cudaMallocHost(&g_dconv_snap_batch[i], g_dconv_bytes));
    }
}

__global__ void accum_inplace_kernel(float * __restrict__ dst, const float * __restrict__ src, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] += src[i];
    }
}

extern "C" void gpu_upload_moe_buf(int c, const float *h_buf, int dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_moe_buf_batch + (size_t)c * dim, h_buf, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void gpu_residual_reset_batch(const float *h_x_all, int dim, int n) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_res_batch, h_x_all, (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void gpu_residual_accum(float *d_dst, const float *d_src, int count) {
    accum_inplace_kernel<<<(count + 255) / 256, 256>>>(d_dst, d_src, count);
    CUDA_CHECK_LAST();
}

extern "C" void gpu_rms_norm_dev_batch(float *d_dst, const float *d_src, const float *d_weight, int dim, float eps, int n) {
    for (int c = 0; c < n; ++c) {
        launch_rms_norm_dev(d_dst + (size_t)c * dim, d_src + (size_t)c * dim, d_weight, dim, eps);
    }
    CUDA_CHECK_LAST();
}

extern "C" void gpu_batch_quantize_norm_x(const float *d_src, int in_dim, int n) {
    const int nblk = in_dim / QK8_1;
    for (int c = 0; c < n; ++c) {
        launch_quantize_q8_1(
            d_src + (size_t)c * in_dim, (block_q8_1_c*)g_d_nx_batch_q8 + (size_t)c * nblk, in_dim);
    }
    CUDA_CHECK_LAST();
}

static void batch_dense_q8_0(const void *dev_w, float *d_dst, int in_dim, int out_dim, int n) {
    const int nblk = in_dim / QK8_0;
    launch_mmvq_q8_0_dense_ncols((const block_q8_0*)dev_w, (const block_q8_1_c*)g_d_nx_batch_q8, d_dst, out_dim, nblk, n);
}

static void batch_dense_q6_K(const void *dev_w, float *d_dst, int in_dim, int out_dim, int n) {
    const int nblk = in_dim / QK_K;
    launch_mmvq_q6_K_dense_ncols((const block_q6_K*)dev_w, (const block_q8_1_c*)g_d_nx_batch_q8, d_dst, out_dim, nblk, n);
}

extern "C" void gpu_delta_projections_batch(const tensor_t *attn_qkv, const tensor_t *attn_gate,
                                            const tensor_t *ssm_beta, const tensor_t *ssm_alpha,
                                            const float *d_norm_batch, int in_dim, int n) {
    const int out_qkv   = (int)attn_qkv->info.dims[1];
    const int out_z     = (int)attn_gate->info.dims[1];
    const int out_beta  = (int)ssm_beta->info.dims[1];
    const int out_alpha = (int)ssm_alpha->info.dims[1];

    gpu_batch_quantize_norm_x(d_norm_batch, in_dim, n);
    batch_dense_q8_0(attn_qkv->dev,  g_d_p_batch_qkv, in_dim, out_qkv, n);
    batch_dense_q8_0(attn_gate->dev, g_d_p_batch_z,   in_dim, out_z,   n);

    for (int c = 0; c < n; ++c) {
        launch_mat_vec_f32_dual((const float*)ssm_beta->dev,  g_d_p_batch_b + (size_t)c * out_beta,  out_beta,
                                (const float*)ssm_alpha->dev, g_d_p_batch_a + (size_t)c * out_alpha, out_alpha,
                                d_norm_batch + (size_t)c * in_dim, in_dim);
    }
}

extern "C" void gpu_delta_fused_gdn_dev_col_batch(int c, int layer_idx, const model_layer_t *L, float *d_res_x,
                                                  float eps, float scale_q,
                                                  int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                                  int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    gpu_delta_fused_gdn_dev_p(layer_idx, L, d_res_x, eps, scale_q,
                              g_d_p_batch_qkv + (size_t)c * qkv_dim,
                              g_d_p_batch_z   + (size_t)c * value_dim,
                              g_d_p_batch_b   + (size_t)c * n_v_heads,
                              g_d_p_batch_a   + (size_t)c * n_v_heads,
                              n_k_heads, n_v_heads, head_k_dim, head_v_dim,
                              key_dim, value_dim, qkv_dim, conv_kernel, dim);
}

extern "C" void gpu_delta_snapshot_layer_batch(int layer_idx, int col, int n_v_heads, int head_v_dim,
                                               int head_k_dim, int conv_kernel, int qkv_dim) {
    if (!g_dstate_all || g_dstate_bytes == 0 || col < 0 || col >= 7) return;
    if (!g_dstate_snap_batch[col]) CUDA_CHECK(cudaMallocHost(&g_dstate_snap_batch[col], g_dstate_bytes));
    if (!g_dconv_snap_batch[col] && g_dconv_bytes > 0) CUDA_CHECK(cudaMallocHost(&g_dconv_snap_batch[col], g_dconv_bytes));

    const size_t state_floats = (size_t)n_v_heads * head_v_dim * head_k_dim;
    CUDA_CHECK(cudaMemcpyAsync(g_dstate_snap_batch[col] + (size_t)layer_idx * state_floats,
                               g_dstate_all              + (size_t)layer_idx * state_floats,
                               state_floats * sizeof(float), cudaMemcpyDeviceToHost));

    if (g_dconv_all && g_dconv_snap_batch[col]) {
        const size_t conv_floats = (size_t)conv_kernel * qkv_dim;
        CUDA_CHECK(cudaMemcpyAsync(g_dconv_snap_batch[col] + (size_t)layer_idx * conv_floats,
                                   g_dconv_all              + (size_t)layer_idx * conv_floats,
                                   conv_floats * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

extern "C" void gpu_delta_rollback_batch(int n_keep) {
    int snap_idx = n_keep - 1;
    if (snap_idx >= 0 && snap_idx < 7 && g_dstate_snap_batch[snap_idx]) {
        CUDA_CHECK(cudaMemcpyAsync(g_dstate_all, g_dstate_snap_batch[snap_idx], g_dstate_bytes, cudaMemcpyHostToDevice));
        if (g_dconv_all && g_dconv_snap_batch[snap_idx] && g_dconv_bytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(g_dconv_all, g_dconv_snap_batch[snap_idx], g_dconv_bytes, cudaMemcpyHostToDevice));
        }
    }
}

extern "C" void gpu_output_argmax_batch(const tensor_t *output_weight, const float *d_norm_batch,
                                        int in_dim, int vocab_size, int n, int32_t *out_tokens,
                                        float *host_logits_capture) {
    gpu_batch_quantize_norm_x(d_norm_batch, in_dim, n);
    batch_dense_q6_K(output_weight->dev, g_d_logits_batch, in_dim, vocab_size, n);

    for (int c = 0; c < n; ++c) {
        launch_argmax(g_d_logits_batch + (size_t)c * vocab_size, g_d_block_max, g_d_block_idx, g_d_best_token_batch + c, vocab_size);
    }
    CUDA_CHECK(cudaMemcpy(out_tokens, g_d_best_token_batch, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_logits_capture) {
        CUDA_CHECK(cudaMemcpy(host_logits_capture, g_d_logits_batch, (size_t)n * vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

