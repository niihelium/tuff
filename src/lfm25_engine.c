#include "lfm25_engine.h"
#include "lfm25_model.h"
#include "tuff_vtable.h"
#include "tuff_exact.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

lfm25_engine_t* lfm25_engine_init(lfm25_model_t *model, int max_context_len) {
    if (!model) return NULL;
    lfm25_engine_t *eng = (lfm25_engine_t*)calloc(1, sizeof(lfm25_engine_t));
    if (!eng) return NULL;

    eng->model = model;
    eng->max_context_len = max_context_len > 0 ? max_context_len : 4096;

    uint32_t d = model->embedding_length;
    uint32_t vocab = model->vocab_size;
    uint32_t ffn_dim = model->is_moe ? model->expert_ffn_dim : 8192;
    uint32_t n_head = model->head_count;
    uint32_t n_head_kv = model->head_count_kv;
    uint32_t head_dim = model->head_dim;

    // Allocate Conv1D rolling state buffer: [n_conv_layers * conv_state_len * d]
    size_t conv_state_bytes = (size_t)model->n_conv_layers * model->conv_state_len * d * sizeof(float);
    if (conv_state_bytes > 0) {
        CUDA_CHECK(cudaMalloc((void**)&eng->d_conv_state, conv_state_bytes));
        CUDA_CHECK(cudaMemset(eng->d_conv_state, 0, conv_state_bytes));
    }

    // Allocate KV Cache for Attention layers: [n_attn_layers * head_count_kv * max_context_len * head_dim]
    size_t kv_cache_bytes = (size_t)model->n_attn_layers * n_head_kv * eng->max_context_len * head_dim * sizeof(float);
    if (kv_cache_bytes > 0) {
        CUDA_CHECK(cudaMalloc((void**)&eng->d_k_cache, kv_cache_bytes));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_v_cache, kv_cache_bytes));
        CUDA_CHECK(cudaMemset(eng->d_k_cache, 0, kv_cache_bytes));
        CUDA_CHECK(cudaMemset(eng->d_v_cache, 0, kv_cache_bytes));
    }

    // Allocate scratch device buffers
    CUDA_CHECK(cudaMalloc((void**)&eng->d_x, (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_norm_x, (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_in_proj, (size_t)3 * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_conv_out, (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_op_out, (size_t)d * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&eng->d_q, (size_t)n_head * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_k, (size_t)n_head_kv * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_v, (size_t)n_head_kv * head_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_gate, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_up, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_act, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_down, (size_t)d * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&eng->d_logits, (size_t)vocab * sizeof(float)));

    // Quantized activation scratch buffers: block_q8_1 is 36 bytes per 32 elements
    size_t q8_x_bytes = (size_t)(d / 32) * 36;
    size_t q8_ffn_bytes = (size_t)(ffn_dim / 32) * 36;
    CUDA_CHECK(cudaMalloc(&eng->d_q8_x, q8_x_bytes));
    CUDA_CHECK(cudaMalloc(&eng->d_q8_op_out, q8_x_bytes));
    CUDA_CHECK(cudaMalloc(&eng->d_q8_ffn_act, q8_ffn_bytes));

    // Device argmax buffers
    size_t grid_blocks = (vocab + 4 - 1) / 4;
    CUDA_CHECK(cudaMalloc((void**)&eng->d_block_max, grid_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_block_idx, grid_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_best_token, sizeof(int)));

    // Host buffers
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_logits, (size_t)vocab * sizeof(float)));
    eng->h_embd = (float*)malloc((size_t)d * sizeof(float));

    return eng;
}

void lfm25_engine_free(lfm25_engine_t *eng) {
    if (!eng) return;

    if (eng->d_conv_state) cudaFree(eng->d_conv_state);
    if (eng->d_k_cache)    cudaFree(eng->d_k_cache);
    if (eng->d_v_cache)    cudaFree(eng->d_v_cache);

    if (eng->d_x)          cudaFree(eng->d_x);
    if (eng->d_norm_x)     cudaFree(eng->d_norm_x);
    if (eng->d_in_proj)    cudaFree(eng->d_in_proj);
    if (eng->d_conv_out)   cudaFree(eng->d_conv_out);
    if (eng->d_op_out)     cudaFree(eng->d_op_out);
    if (eng->d_q)          cudaFree(eng->d_q);
    if (eng->d_k)          cudaFree(eng->d_k);
    if (eng->d_v)          cudaFree(eng->d_v);
    if (eng->d_ffn_gate)   cudaFree(eng->d_ffn_gate);
    if (eng->d_ffn_up)     cudaFree(eng->d_ffn_up);
    if (eng->d_ffn_act)    cudaFree(eng->d_ffn_act);
    if (eng->d_ffn_down)   cudaFree(eng->d_ffn_down);
    if (eng->d_logits)     cudaFree(eng->d_logits);

    if (eng->d_q8_x)       cudaFree(eng->d_q8_x);
    if (eng->d_q8_op_out)  cudaFree(eng->d_q8_op_out);
    if (eng->d_q8_ffn_act) cudaFree(eng->d_q8_ffn_act);

    if (eng->d_block_max)  cudaFree(eng->d_block_max);
    if (eng->d_block_idx)  cudaFree(eng->d_block_idx);
    if (eng->d_best_token) cudaFree(eng->d_best_token);

    if (eng->h_logits)     cudaFreeHost(eng->h_logits);
    if (eng->h_embd)       free(eng->h_embd);

    free(eng);
}

void lfm25_engine_set_sampler(lfm25_engine_t *eng, const sampler_params_t *params) {
    if (eng && params) {
        eng->sampler = *params;
    }
}

static inline void lfm25_gpu_mat_vec(const lfm25_tensor_t *t, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum) {
    if (t->info.type == GGML_TYPE_Q6_K) {
        lfm25_gpu_mmvq_q6_K(t->dev, d_xq8, d_dst, in_dim, out_dim, accum);
    } else {
        lfm25_gpu_mmvq_q4_K(t->dev, d_xq8, d_dst, in_dim, out_dim, accum);
    }
}

int32_t lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos) {
    if (!eng || !eng->model) return -1;
    lfm25_model_t *m = eng->model;
    int dim = (int)m->embedding_length;
    float eps = m->rms_norm_eps;
    int ffn_dim = (int)(m->is_moe ? m->expert_ffn_dim : 8192);

    // 1. Fetch token embedding for input token
    lfm25_gpu_embed(token, eng->d_x, eng->h_embd, &m->token_embd, dim);

    int conv_idx = 0;
    int attn_idx = 0;

    for (uint32_t l = 0; l < m->block_count; ++l) {
        lfm25_layer_t *L = &m->layers[l];

        // --- OPERATOR SUB-BLOCK ---
        // RMSNorm on residual stream -> quantize to q8_1
        lfm25_gpu_fused_rms_norm_quant(eng->d_x, (const float*)L->attn_norm.dev, eng->d_norm_x, eng->d_q8_x, dim, eps);

        if (L->op_type == LFM25_OP_CONV) {
            // ShortConv1D
            // in_proj: 2048 -> 6144 (Q4_K)
            lfm25_gpu_mmvq_q4_K(L->shortconv_in_proj.dev, eng->d_q8_x, eng->d_in_proj, dim, 3 * dim, false);

            // Fused conv modulation, rolling dot, state update, gating & quantize to q8_1
            float *conv_state_ptr = eng->d_conv_state + (size_t)conv_idx * (2 * dim);
            lfm25_gpu_shortconv_step(eng->d_in_proj, conv_state_ptr, (const float*)L->shortconv_conv.dev,
                                     eng->d_conv_out, eng->d_q8_op_out, dim);

            // out_proj: 2048 -> 2048 (Q4_K) with accum into residual d_x
            lfm25_gpu_mmvq_q4_K(L->shortconv_out_proj.dev, eng->d_q8_op_out, eng->d_x, dim, dim, true);

            conv_idx++;
        } else {
            // GQA Attention
            // Q projection: 2048 -> 2048 (Q4_K)
            lfm25_gpu_mmvq_q4_K(L->attn_q.dev, eng->d_q8_x, eng->d_q, dim, (int)m->head_count * (int)m->head_dim, false);
            // K projection: 2048 -> 512 (Q4_K)
            lfm25_gpu_mmvq_q4_K(L->attn_k.dev, eng->d_q8_x, eng->d_k, dim, (int)m->head_count_kv * (int)m->head_dim, false);
            // V projection: 2048 -> 512 (Q4_K or Q6_K depending on layer)
            lfm25_gpu_mat_vec(&L->attn_v, eng->d_q8_x, eng->d_v, dim, (int)m->head_count_kv * (int)m->head_dim, false);

            // QK Norm + NeoX RoPE + K Cache Write
            lfm25_gpu_qk_norm_rope(eng->d_q, eng->d_k, (const float*)L->attn_q_norm.dev, (const float*)L->attn_k_norm.dev,
                                   eng->d_k_cache, (int)m->head_count, (int)m->head_count_kv, (int)m->head_dim,
                                   pos, eng->max_context_len, m->rope_freq_base, eps, attn_idx);

            // V Cache Write
            lfm25_gpu_v_cache_write(eng->d_v, eng->d_v_cache, (int)m->head_count_kv, (int)m->head_dim, pos, eng->max_context_len, attn_idx);

            // Flash-Decoding Attention + quantize to q8_1
            lfm25_gpu_attention(eng->d_q, eng->d_k_cache, eng->d_v_cache, eng->d_op_out, eng->d_q8_op_out,
                                (int)m->head_count, (int)m->head_count_kv, (int)m->head_dim, pos, eng->max_context_len, attn_idx);

            // out_proj: 2048 -> 2048 (Q4_K) with accum into residual d_x
            lfm25_gpu_mmvq_q4_K(L->attn_output.dev, eng->d_q8_op_out, eng->d_x, dim, dim, true);

            attn_idx++;
        }

        // --- FFN SUB-BLOCK ---
        // RMSNorm on post-operator residual -> quantize to q8_1
        lfm25_gpu_fused_rms_norm_quant(eng->d_x, (const float*)L->ffn_norm.dev, eng->d_norm_x, eng->d_q8_x, dim, eps);

        if (L->ffn_type == LFM25_FFN_DENSE) {
            // Fused SwiGLU: computes gate (Q4_K) and up (Q4_K), applies silu(gate)*up, quantizes to q8_1
            lfm25_gpu_fused_swiglu(L->ffn_gate.dev, L->ffn_up.dev, eng->d_q8_x, eng->d_ffn_act, eng->d_q8_ffn_act, dim, ffn_dim);

            // down proj: 8192 -> 2048 (Q4_K or Q6_K depending on layer) with accum into residual d_x
            lfm25_gpu_mat_vec(&L->ffn_down, eng->d_q8_ffn_act, eng->d_x, ffn_dim, dim, true);
        }
    }

    // Output Head + argmax
    int32_t next_tok = lfm25_gpu_output_norm_head_argmax(
        eng->d_x, (const float*)m->token_embd_norm.dev, m->output_weight.dev,
        eng->d_q8_x, eng->d_logits, eng->h_logits,
        eng->d_block_max, eng->d_block_idx, eng->d_best_token,
        dim, (int)m->vocab_size, eps);

    if (tuff_logit_capture_is_enabled()) {
        tuff_logit_capture(eng->h_logits, m->vocab_size);
    }

    return next_tok;
}

// -----------------------------------------------------------------------------
// VTable Adapter Functions
// -----------------------------------------------------------------------------

static void* vtable_lfm25_model_load(const char *path) {
    return lfm25_model_load(path);
}

static void vtable_lfm25_model_free(void *model) {
    lfm25_model_free((lfm25_model_t*)model);
}

static void vtable_lfm25_model_print_inspection(const void *model) {
    lfm25_model_print_inspection((const lfm25_model_t*)model);
}

static gguf_context_t* vtable_lfm25_model_get_gguf(void *model) {
    return ((lfm25_model_t*)model)->gguf;
}

static int32_t vtable_lfm25_model_get_bos(void *model) {
    (void)model;
    return 1; // Default BOS
}

static int32_t vtable_lfm25_model_get_eos(void *model) {
    (void)model;
    return 2; // Default EOS
}

static uint32_t vtable_lfm25_model_get_vocab_size(void *model) {
    return ((lfm25_model_t*)model)->vocab_size;
}

static int vtable_lfm25_gpu_prepare(void *model, int *max_context_len, int fill_ram) {
    return lfm25_gpu_prepare((lfm25_model_t*)model, max_context_len, fill_ram);
}

static void vtable_lfm25_gpu_shutdown(void *model) {
    lfm25_gpu_shutdown((lfm25_model_t*)model);
}

static void* vtable_lfm25_engine_init(void *model, int max_context_len) {
    return lfm25_engine_init((lfm25_model_t*)model, max_context_len);
}

static void vtable_lfm25_engine_free(void *engine) {
    lfm25_engine_free((lfm25_engine_t*)engine);
}

static void vtable_lfm25_engine_set_sampler(void *engine, const sampler_params_t *params) {
    lfm25_engine_set_sampler((lfm25_engine_t*)engine, params);
}

static int32_t vtable_lfm25_engine_forward_token(void *engine, int32_t token, int pos) {
    return lfm25_engine_forward_token((lfm25_engine_t*)engine, token, pos);
}

const engine_vtable_t lfm25_engine_vtable = {
    .arch                   = TUFF_ARCH_LFM2,
    .name                   = "lfm25",
    .model_load             = vtable_lfm25_model_load,
    .model_free             = vtable_lfm25_model_free,
    .model_print_inspection = vtable_lfm25_model_print_inspection,
    .model_get_gguf         = vtable_lfm25_model_get_gguf,
    .model_get_bos          = vtable_lfm25_model_get_bos,
    .model_get_eos          = vtable_lfm25_model_get_eos,
    .model_get_vocab_size   = vtable_lfm25_model_get_vocab_size,
    .gpu_prepare            = vtable_lfm25_gpu_prepare,
    .gpu_shutdown           = vtable_lfm25_gpu_shutdown,
    .engine_init            = vtable_lfm25_engine_init,
    .engine_free            = vtable_lfm25_engine_free,
    .engine_set_sampler     = vtable_lfm25_engine_set_sampler,
    .engine_forward_token   = vtable_lfm25_engine_forward_token,
};
