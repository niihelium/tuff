#include "state.h"

__global__ void concat_norms_kernel(float * __restrict__ concat,
                                   const float * __restrict__ enorm,
                                   const float * __restrict__ hnorm,
                                   int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        concat[i] = enorm[i];
        concat[dim + i] = hnorm[i];
    }
}

int32_t gpu_mtp_draft_dev(const model_t *m, float *d_res_x, int32_t token_id, int pos, int max_seq) {
    if (!m || m->nextn_predict_layers == 0 || m->block_count <= 40) return -1;
    const model_layer_t *l40 = &m->layers[40];
    if (!l40->nextn_eh_proj.dev) return -1;

    int dim = (int)m->embedding_length; // 2048
    float eps = m->rms_norm_eps;

    if (!g_d_mtp_concat) {
        CUDA_CHECK(cudaMalloc(&g_d_mtp_concat, 2 * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_mtp_x, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_mtp_norm_x, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_token_embd, dim * sizeof(float)));
    }

    // 1. Fetch token embedding for token_id onto device
    float host_embd[2048];
    const uint8_t *embd_base = (const uint8_t*)m->token_embd.data;
    size_t embd_row_bytes = ggml_row_size((ggml_type_t)m->token_embd.info.type, dim);
    dequantize_row((ggml_type_t)m->token_embd.info.type, embd_base + (size_t)token_id * embd_row_bytes, host_embd, dim);
    CUDA_CHECK(cudaMemcpy(g_d_token_embd, host_embd, dim * sizeof(float), cudaMemcpyHostToDevice));

    if (getenv("TUFF_DEBUG_MTP") && pos == 1) {
        fprintf(stderr, "[MTP DEBUG] l40->nextn_enorm.dev=%p, l40->nextn_hnorm.dev=%p, l40->nextn_eh_proj.dev=%p\n",
                l40->nextn_enorm.dev, l40->nextn_hnorm.dev, l40->nextn_eh_proj.dev);
        fprintf(stderr, "[MTP DEBUG] host_embd[0..3]=%.4f %.4f %.4f %.4f\n",
                host_embd[0], host_embd[1], host_embd[2], host_embd[3]);
    }

    if (getenv("TUFF_DEBUG_MTP") && pos == 1) {
        float r[4], nw[4];
        CUDA_CHECK(cudaMemcpy(r, d_res_x, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nw, l40->nextn_hnorm.dev, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[MTP DEBUG] d_res_x[0..3]: %.4f %.4f %.4f %.4f, nextn_hnorm[0..3]: %.4f %.4f %.4f %.4f\n",
                r[0], r[1], r[2], r[3], nw[0], nw[1], nw[2], nw[3]);
    }

    // 2. RMS-norm token embedding and hidden state
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_token_embd, (const float*)l40->nextn_enorm.dev, dim, eps);
    gpu_rms_norm_dev(g_d_mtp_x, d_res_x, (const float*)l40->nextn_hnorm.dev, dim, eps);

    // 3. Concat [enorm, hnorm] -> 4096
    concat_norms_kernel<<<16, 256>>>(g_d_mtp_concat, g_d_mtp_norm_x, g_d_mtp_x, dim);
    CUDA_CHECK_LAST();

    // 4. eh_proj: (4096 -> 2048) -> g_d_mtp_x
    launch_mat_vec_q8_0((const block_q8_0*)l40->nextn_eh_proj.dev, g_d_mtp_concat, g_d_mtp_x, 2 * dim, dim);

    // 5. Layer 40 Attention block
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_mtp_x, (const float*)l40->attn_norm.dev, dim, eps);
    gpu_attention_layer_dev(10, g_d_mtp_x, l40, g_d_mtp_norm_x,
                            (int)m->n_attn_layers, (int)m->head_count, (int)m->head_count_kv,
                            (int)m->key_length, pos, pos + 1, max_seq,
                            1.0f / sqrtf((float)m->key_length), eps, dim,
                            (int)m->rope_dim_count, m->rope_dim_sections, m->rope_freq_base);

    // 6. Layer 40 Post-attention norm + Shared expert
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_mtp_x, (const float*)l40->post_attention_norm.dev, dim, eps);
    gpu_shared_expert_dev(g_d_mtp_concat, l40, g_d_mtp_norm_x, dim, (int)m->expert_shared_ffn_len);
    gpu_residual_accum(g_d_mtp_x, g_d_mtp_concat, dim);
    CUDA_CHECK_LAST();

    // 7. Output Norm (shared_head_norm or output_norm)
    const float *head_norm_w = l40->nextn_shared_head_norm.dev ? (const float*)l40->nextn_shared_head_norm.dev : (const float*)m->output_norm.dev;
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_mtp_x, head_norm_w, dim, eps);

    // 8. Argmax against output_weight
    int32_t draft_token = gpu_output_argmax_dev(&m->output_weight, g_d_mtp_norm_x, dim, (int)m->vocab_size, NULL);
    return draft_token;
}


// ---------------------------------------------------------------------------
// Pinned host buffers for the per-layer decode round trips (declared in gpu.h).
// cudaMemcpyAsync on pageable memory blocks the calling thread while the driver stages the
// bytes; on pinned memory the DMA engine reads host memory directly. The decode loop performs
// ~3 such copies per layer (norm_x D2H, router logits D2H, CPU-miss accumulator H2D).
