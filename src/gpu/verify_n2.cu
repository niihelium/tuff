#include "state.h"

extern "C" void gpu_n2_prepare(int dim, int vocab_size) {
    if (g_n2_dim == dim && g_d_res_x2) return;
    g_n2_dim = dim;
    if (!g_d_res_x2)  CUDA_CHECK(cudaMalloc(&g_d_res_x2,  (size_t)2 * dim * sizeof(float)));
    if (!g_d_norm_x2) CUDA_CHECK(cudaMalloc(&g_d_norm_x2, (size_t)2 * dim * sizeof(float)));
    // The q8_1 staging buffer is sized by the widest activation the ncols kernels
    // see, which is the trunk width (dim), not any projection's output width.
    if (!g_d_nx2_q8)  CUDA_CHECK(cudaMalloc(&g_d_nx2_q8,  (size_t)2 * (dim / QK8_1) * sizeof(block_q8_1_c)));
    if (!g_d_logits2) CUDA_CHECK(cudaMalloc(&g_d_logits2, (size_t)2 * vocab_size * sizeof(float)));
}

// Second shared-expert output buffer, so both columns of a layer can be issued
// before the single router sync instead of one on each side of it.
extern "C" const float *gpu_get_shexp2_dev_ptr(void) {
    if (!g_d_shexp_out2 && g_n2_dim > 0)
        CUDA_CHECK(cudaMalloc(&g_d_shexp_out2, (size_t)g_n2_dim * sizeof(float)));
    return g_d_shexp_out2;
}

extern "C" float *gpu_get_d_res_x2(void)  { return g_d_res_x2; }
extern "C" float *gpu_get_d_norm_x2(void) { return g_d_norm_x2; }

extern "C" void gpu_residual_reset_n2(const float *h_x0, const float *h_x1, int dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_res_x2,              h_x0, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_d_res_x2 + (size_t)dim, h_x1, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void gpu_rms_norm_dev_n2(float *d_dst2, const float *d_src2, const float *d_weight, int dim, float eps) {
    launch_rms_norm_dev(d_dst2, d_src2, d_weight, dim, eps);
    launch_rms_norm_dev(d_dst2 + (size_t)dim, d_src2 + (size_t)dim, d_weight, dim, eps);
    CUDA_CHECK_LAST();
}

// Quantize both normalized columns to q8_1 once; every ncols kernel below reads
// this staging buffer. Q8_0 and Q6_K forms both consume in_dim/32 blocks per
// column (QK_K/QK8_1 = 8 sub-blocks per super-block), so one buffer serves both.
extern "C" void gpu_n2_quantize_norm_x(const float *d_src2, int in_dim) {
    const int nblk = in_dim / QK8_1;
    launch_quantize_q8_1(
        d_src2, (block_q8_1_c*)g_d_nx2_q8, in_dim);
    launch_quantize_q8_1(
        d_src2 + (size_t)in_dim, (block_q8_1_c*)g_d_nx2_q8 + nblk, in_dim);
    CUDA_CHECK_LAST();
}

// One weight read, two output columns. dst is [2][out_dim].
static void n2_dense_q8_0(const void *dev_w, float *d_dst2, int in_dim, int out_dim) {
    const int nblk = in_dim / QK8_0;
    launch_mmvq_q8_0_dense_ncols((const block_q8_0*)dev_w, (const block_q8_1_c*)g_d_nx2_q8, d_dst2, out_dim, nblk, 2);
}

// The delta-layer input projections for both columns. attn_qkv (2048x8192) and
// attn_gate are the two big Q8_0 reads and are shared; ssm_beta/ssm_alpha are
// F32 and a few hundred KB, so they just run twice.
extern "C" void gpu_delta_projections_n2(const tensor_t *attn_qkv, const tensor_t *attn_gate,
                                         const tensor_t *ssm_beta, const tensor_t *ssm_alpha,
                                         const float *d_norm_x2, int in_dim) {
    const int out_qkv   = (int)attn_qkv->info.dims[1];
    const int out_z     = (int)attn_gate->info.dims[1];
    const int out_beta  = (int)ssm_beta->info.dims[1];
    const int out_alpha = (int)ssm_alpha->info.dims[1];

    if (!g_d_p2_qkv) CUDA_CHECK(cudaMalloc(&g_d_p2_qkv, (size_t)2 * out_qkv   * sizeof(float)));
    if (!g_d_p2_z)   CUDA_CHECK(cudaMalloc(&g_d_p2_z,   (size_t)2 * out_z     * sizeof(float)));
    if (!g_d_p2_b)   CUDA_CHECK(cudaMalloc(&g_d_p2_b,   (size_t)2 * out_beta  * sizeof(float)));
    if (!g_d_p2_a)   CUDA_CHECK(cudaMalloc(&g_d_p2_a,   (size_t)2 * out_alpha * sizeof(float)));

    gpu_n2_quantize_norm_x(d_norm_x2, in_dim);
    n2_dense_q8_0(attn_qkv->dev,  g_d_p2_qkv, in_dim, out_qkv);
    n2_dense_q8_0(attn_gate->dev, g_d_p2_z,   in_dim, out_z);

    launch_mat_vec_f32_dual((const float*)ssm_beta->dev,  g_d_p2_b, out_beta,
                            (const float*)ssm_alpha->dev, g_d_p2_a, out_alpha,
                            d_norm_x2, in_dim);
    launch_mat_vec_f32_dual((const float*)ssm_beta->dev,  g_d_p2_b + out_beta,  out_beta,
                            (const float*)ssm_alpha->dev, g_d_p2_a + out_alpha, out_alpha,
                            d_norm_x2 + (size_t)in_dim, in_dim);
}

// H15: the shared expert over BOTH columns in one weight read. The gate/up pair
// (Q8_0, dim->sh_ffn_dim), the SwiGLU, the down (Q8_0, sh_ffn_dim->dim) and the
// optional gate_inp scale all run over the two columns via the ncols=2 mmvq
// launcher, so each weight row is read once for both tokens instead of twice.
// Column c's output is bit-identical to gpu_shared_expert_dev on column c alone
// while the mmvq dual path is active: the ncols kernel performs the same per-thread
// accumulation and warp reduction over the same q8_1 blocks. When that path is
// disabled (TUFF_EXACT=1 or TUFF_MMVQ_DUAL=0) it falls back to two per-column
// calls, which the ncols kernels cannot reproduce bit-for-bit.
void gpu_shared_expert_dev_n2(float *d_shexp_out0, float *d_shexp_out1, const model_layer_t *layer,
                              const float *d_norm_x2, int in_dim, int sh_ffn_dim) {
    static int mmvq_dual = -1;
    if (mmvq_dual < 0) { const char *e = getenv("TUFF_MMVQ_DUAL"); mmvq_dual = (e && e[0] == '0') ? 0 : 1; }
    if (tuff_exact() || !mmvq_dual) {
        gpu_shared_expert_dev(d_shexp_out0, layer, d_norm_x2, in_dim, sh_ffn_dim);
        gpu_shared_expert_dev(d_shexp_out1, layer, d_norm_x2 + (size_t)in_dim, in_dim, sh_ffn_dim);
        return;
    }
    const int dim = in_dim;
    if (!g_shg2)      CUDA_CHECK(cudaMalloc(&g_shg2,      (size_t)2 * sh_ffn_dim * sizeof(float)));
    if (!g_shu2)      CUDA_CHECK(cudaMalloc(&g_shu2,      (size_t)2 * sh_ffn_dim * sizeof(float)));
    if (!g_shgi2)     CUDA_CHECK(cudaMalloc(&g_shgi2,     (size_t)2             * sizeof(float)));
    if (!g_d_shact2)  CUDA_CHECK(cudaMalloc(&g_d_shact2,  (size_t)2 * sh_ffn_dim * sizeof(float)));
    if (!g_d_shdown2) CUDA_CHECK(cudaMalloc(&g_d_shdown2, (size_t)2 * dim        * sizeof(float)));

    // gate/up share the normalized input; one q8_1 quantization of both columns.
    gpu_n2_quantize_norm_x(d_norm_x2, in_dim);
    n2_dense_q8_0(layer->ffn_gate_shexp.dev, g_shg2, in_dim, sh_ffn_dim);
    n2_dense_q8_0(layer->ffn_up_shexp.dev,   g_shu2, in_dim, sh_ffn_dim);
    launch_swiglu_ncols(g_d_shact2, g_shg2, g_shu2, 2 * sh_ffn_dim);
    CUDA_CHECK_LAST();

    // The down projection's activation is the SwiGLU output, sh_ffn_dim per column.
    // The q8_1 staging buffer is free (gate/up consumed it); nblk = sh_ffn_dim/QK8_1.
    {
        const int nblk = sh_ffn_dim / QK8_1;
        launch_quantize_q8_1(
            g_d_shact2, (block_q8_1_c*)g_d_nx2_q8, sh_ffn_dim);
        launch_quantize_q8_1(
            g_d_shact2 + sh_ffn_dim, (block_q8_1_c*)g_d_nx2_q8 + nblk, sh_ffn_dim);
        CUDA_CHECK_LAST();
    }
    n2_dense_q8_0(layer->ffn_down_shexp.dev, g_d_shdown2, sh_ffn_dim, dim);

    if (layer->ffn_gate_inp_shexp.dev) {
        launch_mat_vec_f32((const float*)layer->ffn_gate_inp_shexp.dev, d_norm_x2,          g_shgi2,     in_dim, 1);
        launch_mat_vec_f32((const float*)layer->ffn_gate_inp_shexp.dev, d_norm_x2 + in_dim, g_shgi2 + 1, in_dim, 1);
        launch_scale_shexp(d_shexp_out0, g_d_shdown2, g_shgi2, dim);
        launch_scale_shexp(d_shexp_out1, g_d_shdown2 + dim, g_shgi2 + 1, dim);
        CUDA_CHECK_LAST();
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_shexp_out0, g_d_shdown2,       (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_shexp_out1, g_d_shdown2 + dim, (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// H17: run the fused GDN step for column c straight off the batched projections. The old
// gpu_delta_select_column_n2 published column c into the single-column scratch g_dpqkv/g_dpz/
// g_dpb/g_dpa with four D2D copies per column per layer (240 copies per verify step), because
// delta_conv_kernel consumed g_dpqkv in place. Pointing the fused kernel directly at
// g_d_p2_qkv + c*out_qkv is safe: column c's projection is read exactly once (by the conv), so
// overwriting it with qkv_conv clobbers nothing the rest of the step still needs, and column 1's
// projection is untouched until its own pass. g_d_p2_* are allocated by gpu_delta_projections_n2.
extern "C" void gpu_delta_fused_gdn_dev_col(int c, int layer_idx, const model_layer_t *L, float *d_res_x,
                                            float eps, float scale_q,
                                            int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                            int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    if (!g_d_p2_qkv) CUDA_CHECK(cudaMalloc(&g_d_p2_qkv, (size_t)2 * qkv_dim   * sizeof(float)));
    if (!g_d_p2_z)   CUDA_CHECK(cudaMalloc(&g_d_p2_z,   (size_t)2 * value_dim * sizeof(float)));
    if (!g_d_p2_b)   CUDA_CHECK(cudaMalloc(&g_d_p2_b,   (size_t)2 * n_v_heads * sizeof(float)));
    if (!g_d_p2_a)   CUDA_CHECK(cudaMalloc(&g_d_p2_a,   (size_t)2 * n_v_heads * sizeof(float)));
    gpu_delta_fused_gdn_dev_p(layer_idx, L, d_res_x, eps, scale_q,
                              g_d_p2_qkv + (size_t)c * qkv_dim,
                              g_d_p2_z   + (size_t)c * value_dim,
                              g_d_p2_b   + (size_t)c * n_v_heads,
                              g_d_p2_a   + (size_t)c * n_v_heads,
                              n_k_heads, n_v_heads, head_k_dim, head_v_dim,
                              key_dim, value_dim, qkv_dim, conv_kernel, dim);
}

// --- Rollback of the recurrent state for a rejected draft ---
//
// The snapshot is taken BETWEEN the two columns of a layer, not before the
// pass. Before the pass the state has consumed tokens up to pos-1; a rejected
// draft needs the state as of pos, i.e. after column 0 but before column 1.
// Taking it per layer slice at that point costs the same 60 MB of D2D in total
// (~0.15 ms, ~1% of the step) and saves re-running the accepted token.
extern "C" void gpu_delta_snapshot_layer_n2(int layer_idx, int n_v_heads, int head_v_dim,
                                            int head_k_dim, int conv_kernel, int qkv_dim) {
    if (!g_dstate_all || g_dstate_bytes == 0) return;
    if (!g_dstate_snap) CUDA_CHECK(cudaMalloc(&g_dstate_snap, g_dstate_bytes));
    if (!g_dconv_snap && g_dconv_bytes > 0) CUDA_CHECK(cudaMalloc(&g_dconv_snap, g_dconv_bytes));

    const size_t state_floats = (size_t)n_v_heads * head_v_dim * head_k_dim;
    CUDA_CHECK(cudaMemcpyAsync(g_dstate_snap + (size_t)layer_idx * state_floats,
                               g_dstate_all  + (size_t)layer_idx * state_floats,
                               state_floats * sizeof(float), cudaMemcpyDeviceToDevice));

    if (g_dconv_all && g_dconv_snap) {
        const size_t conv_floats = (size_t)conv_kernel * qkv_dim;
        CUDA_CHECK(cudaMemcpyAsync(g_dconv_snap + (size_t)layer_idx * conv_floats,
                                   g_dconv_all  + (size_t)layer_idx * conv_floats,
                                   conv_floats * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

extern "C" void gpu_delta_rollback_n2(void) {
    if (g_dstate_all && g_dstate_snap && g_dstate_bytes > 0)
        CUDA_CHECK(cudaMemcpyAsync(g_dstate_all, g_dstate_snap, g_dstate_bytes, cudaMemcpyDeviceToDevice));
    if (g_dconv_all && g_dconv_snap && g_dconv_bytes > 0)
        CUDA_CHECK(cudaMemcpyAsync(g_dconv_all, g_dconv_snap, g_dconv_bytes, cudaMemcpyDeviceToDevice));
}

// The output head over both columns: 398 MB read once, two argmaxes out.
// Measured at 1.46x a single column (iteration 42 stage 2a), not 1.0x.
extern "C" void gpu_output_argmax_n2(const tensor_t *output_weight, const float *d_norm_x2,
                                     int in_dim, int vocab_size, int32_t *out_tokens2) {
    gpu_n2_quantize_norm_x(d_norm_x2, in_dim);

    const int nblk = in_dim / QK_K;
    dim3 grid((unsigned)((vocab_size + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    launch_mmvq_q6_K_dense_ncols((const block_q6_K*)output_weight->dev, (const block_q8_1_c*)g_d_nx2_q8, g_d_logits2, vocab_size, nblk, 2);

    for (int c = 0; c < 2; ++c) {
        launch_argmax(g_d_logits2 + (size_t)c * vocab_size, g_d_block_max, g_d_block_idx, g_d_best_token, vocab_size);
        CUDA_CHECK(cudaMemcpy(&out_tokens2[c], g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
}

// ===========================================================================
// Generalised N-Column Batch Verify Trunk (Task Q1.1, 1 <= N <= 8)
// ===========================================================================

