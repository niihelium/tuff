#include "lfm25_engine.h"
#include "lfm25_model.h"
#include "tuff_vtable.h"
#include "tuff_exact.h"
#include "cuda_check.h"
#include "ggml_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

lfm25_engine_t* lfm25_engine_init(lfm25_model_t *model, int max_context_len) {
    if (!model) return NULL;
    // Router / expert kernels support 32 or 64 experts, top-4. Models with >32 experts (LFM2.5-24B-A2B) use the
    // VRAM expert cache (src/lfm25_expert_cache.c) because their experts do not fit in VRAM.
    if (model->is_moe && ((model->expert_count != 32 && model->expert_count != 64) || model->expert_used_count != 4)) {
        fprintf(stderr, "[LFM25] ERROR: MoE with %u experts / top-%u is not supported (only 32 or 64 / top-4)\n",
                model->expert_count, model->expert_used_count);
        return NULL;
    }
    lfm25_engine_t *eng = (lfm25_engine_t*)calloc(1, sizeof(lfm25_engine_t));
    if (!eng) return NULL;

    eng->model = model;
    eng->max_context_len = max_context_len > 0 ? max_context_len : 4096;

    uint32_t d = model->embedding_length;
    uint32_t vocab = model->vocab_size;
    uint32_t ffn_dim = model->ffn_dim;
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

    if (model->is_moe) {
        size_t moe_n = (size_t)model->expert_used_count * model->expert_ffn_dim;
        CUDA_CHECK(cudaMalloc((void**)&eng->d_topk_ids, 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_topk_w, 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_moe_act, moe_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng->d_moe_act_q8, (moe_n / 32) * 36));
    }

    // Device argmax buffers
    size_t grid_blocks = (vocab + 4 - 1) / 4;
    CUDA_CHECK(cudaMalloc((void**)&eng->d_block_max, grid_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_block_idx, grid_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_best_token, sizeof(int)));

    // Host buffers
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_logits, (size_t)vocab * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_tokpos, 2 * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_best, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_tokpos, 2 * sizeof(int)));

    // CUDA graph: only when there is no host round-trip inside the forward pass (no expert cache).
    {
        const char *g = getenv("LFM25_CUDA_GRAPH");
        int want = !(g && g[0] == '0');
        eng->graph_enabled = want && !lfm25_expert_cache_wanted(model);
        if (eng->graph_enabled) CUDA_CHECK(cudaStreamCreateWithFlags(&eng->stream, cudaStreamNonBlocking));
    }

    // Expert cache: created last so its default size can take all VRAM left after trunk + KV + scratch.
    if (lfm25_expert_cache_wanted(model)) {
        CUDA_CHECK(cudaMalloc((void**)&eng->d_slot_ids, 4 * sizeof(int)));
        eng->expert_cache = lfm25_expert_cache_create(model);
        if (!eng->expert_cache) { lfm25_engine_free(eng); return NULL; }
    }

    return eng;
}

void lfm25_engine_free(lfm25_engine_t *eng) {
    if (!eng) return;

    lfm25_expert_cache_free(eng->expert_cache);
    if (eng->d_slot_ids)   cudaFree(eng->d_slot_ids);
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

    if (eng->d_topk_ids)   cudaFree(eng->d_topk_ids);
    if (eng->d_topk_w)     cudaFree(eng->d_topk_w);
    if (eng->d_moe_act)    cudaFree(eng->d_moe_act);
    if (eng->d_moe_act_q8) cudaFree(eng->d_moe_act_q8);

    if (eng->d_block_max)  cudaFree(eng->d_block_max);
    if (eng->d_block_idx)  cudaFree(eng->d_block_idx);
    if (eng->d_best_token) cudaFree(eng->d_best_token);

    if (eng->h_logits)     cudaFreeHost(eng->h_logits);
    if (eng->h_tokpos)     cudaFreeHost(eng->h_tokpos);
    if (eng->h_best)       cudaFreeHost(eng->h_best);
    if (eng->d_tokpos)     cudaFree(eng->d_tokpos);
    if (eng->graph_exec)   cudaGraphExecDestroy(eng->graph_exec);
    if (eng->graph)        cudaGraphDestroy(eng->graph);
    if (eng->stream)       cudaStreamDestroy(eng->stream);

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

// Enqueue one decode step on the stream selected via lfm25_gpu_set_stream(). Reads (token,pos) from eng->d_tokpos,
// so nothing in here depends on per-token host values; the sequence is identical for every token and capturable.
static void lfm25_enqueue_step(lfm25_engine_t *eng, cudaStream_t stream) {
    lfm25_model_t *m = eng->model;
    int dim = (int)m->embedding_length;
    float eps = m->rms_norm_eps;
    int ffn_dim = (int)m->ffn_dim;

    const int *d_pos = eng->d_tokpos + 1;

    // 0. (token,pos) host -> device, then 1. dequantize the token embedding row on the GPU
    CUDA_CHECK(cudaMemcpyAsync(eng->d_tokpos, eng->h_tokpos, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
    lfm25_gpu_embed(eng->d_tokpos, eng->d_x, &m->token_embd, dim);

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
                                   d_pos, eng->max_context_len, m->rope_freq_base, eps, attn_idx);

            // V Cache Write
            lfm25_gpu_v_cache_write(eng->d_v, eng->d_v_cache, (int)m->head_count_kv, (int)m->head_dim, d_pos, eng->max_context_len, attn_idx);

            // Flash-Decoding Attention + quantize to q8_1
            lfm25_gpu_attention(eng->d_q, eng->d_k_cache, eng->d_v_cache, eng->d_op_out, eng->d_q8_op_out,
                                (int)m->head_count, (int)m->head_count_kv, (int)m->head_dim, d_pos, eng->max_context_len, attn_idx);

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

            // down proj: ffn_dim -> 2048 (Q4_K or Q6_K depending on layer) with accum into residual d_x
            lfm25_gpu_mat_vec(&L->ffn_down, eng->d_q8_ffn_act, eng->d_x, ffn_dim, dim, true);
        } else {
            // Sparse MoE: sigmoid top-4 router on the normed FP32 input, then in-VRAM expert FFN accumulated into d_x
            lfm25_gpu_router_top4(eng->d_norm_x, (const float*)L->ffn_gate_inp.dev, (const float*)L->exp_probs_b.dev,
                                  eng->d_topk_ids, eng->d_topk_w, NULL, dim, (int)m->expert_count);
            const size_t gs0 = (size_t)m->expert_ffn_dim * ggml_row_size(L->ffn_gate_exps.info.type, dim);
            const size_t us0 = (size_t)m->expert_ffn_dim * ggml_row_size(L->ffn_up_exps.info.type, dim);
            const size_t ds0 = (size_t)dim * ggml_row_size(L->ffn_down_exps.info.type, m->expert_ffn_dim);
            if (eng->expert_cache) {
                // Cache path: ids -> host (sync, baseline), look up / fill slots, run the same kernels on the slot pools.
                lfm25_expert_cache_t *ec = eng->expert_cache;
                int h_ids[4], h_slots[4];
                CUDA_CHECK(cudaMemcpy(h_ids, eng->d_topk_ids, sizeof(h_ids), cudaMemcpyDeviceToHost));
                lfm25_expert_cache_lookup(ec, (int)l, h_ids, 4, h_slots);
                CUDA_CHECK(cudaMemcpyAsync(eng->d_slot_ids, h_slots, sizeof(h_slots), cudaMemcpyHostToDevice, stream));
                // h_slots is a stack array: the pageable async copy is staged before returning, but sync to be safe.
                CUDA_CHECK(cudaStreamSynchronize(stream));
                lfm25_gpu_moe_ffn(eng->d_q8_x, eng->d_slot_ids, eng->d_topk_w, lfm25_expert_cache_pool_gate(ec),
                                  lfm25_expert_cache_pool_up(ec), lfm25_expert_cache_pool_down(ec),
                                  (int)L->ffn_down_exps.info.type, eng->d_moe_act, eng->d_moe_act_q8,
                                  eng->d_x, true, dim, (int)m->expert_ffn_dim,
                                  lfm25_expert_cache_gate_slot_bytes(ec), lfm25_expert_cache_up_slot_bytes(ec),
                                  lfm25_expert_cache_down_slot_bytes(ec));
            } else {
                lfm25_gpu_moe_ffn(eng->d_q8_x, eng->d_topk_ids, eng->d_topk_w, L->ffn_gate_exps.dev, L->ffn_up_exps.dev,
                                  L->ffn_down_exps.dev, (int)L->ffn_down_exps.info.type, eng->d_moe_act, eng->d_moe_act_q8,
                                  eng->d_x, true, dim, (int)m->expert_ffn_dim, gs0, us0, ds0);
            }
        }
    }

    // Output Head + argmax; logits stay on the device unless capture asks for them (see forward_token)
    lfm25_gpu_output_norm_head_argmax(
        eng->d_x, (const float*)m->token_embd_norm.dev, m->output_weight.dev,
        eng->d_q8_x, eng->d_logits,
        eng->d_block_max, eng->d_block_idx, eng->d_best_token,
        dim, (int)m->vocab_size, eps);
    CUDA_CHECK(cudaMemcpyAsync(eng->h_best, eng->d_best_token, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

int32_t lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos) {
    if (!eng || !eng->model) return -1;
    eng->h_tokpos[0] = token;
    eng->h_tokpos[1] = pos;

    if (eng->graph_enabled) {
        if (!eng->graph_exec) {
            // First token: record the step into a graph (nothing executes during capture), then replay below.
            lfm25_gpu_set_stream(eng->stream);
            CUDA_CHECK(cudaStreamBeginCapture(eng->stream, cudaStreamCaptureModeThreadLocal));
            lfm25_enqueue_step(eng, eng->stream);
            CUDA_CHECK(cudaStreamEndCapture(eng->stream, &eng->graph));
            CUDA_CHECK(cudaGraphInstantiate(&eng->graph_exec, eng->graph, 0));
        }
        CUDA_CHECK(cudaGraphLaunch(eng->graph_exec, eng->stream));
        CUDA_CHECK(cudaStreamSynchronize(eng->stream));
        if (tuff_logit_capture_is_enabled()) {
            CUDA_CHECK(cudaMemcpy(eng->h_logits, eng->d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        }
    } else {
        lfm25_gpu_set_stream(0);
        lfm25_enqueue_step(eng, 0);
        CUDA_CHECK(cudaStreamSynchronize(0));
        if (tuff_logit_capture_is_enabled()) {
            CUDA_CHECK(cudaMemcpy(eng->h_logits, eng->d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    if (tuff_logit_capture_is_enabled()) {
        tuff_logit_capture(eng->h_logits, eng->model->vocab_size);
    }
    return *eng->h_best;
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
    const lfm25_model_t *m = (const lfm25_model_t*)model;
    return (int32_t)tuff_gguf_get_kv_u32(m->gguf, "tokenizer.ggml.bos_token_id", 1);
}

static int32_t vtable_lfm25_model_get_eos(void *model) {
    const lfm25_model_t *m = (const lfm25_model_t*)model;
    return (int32_t)tuff_gguf_get_kv_u32(m->gguf, "tokenizer.ggml.eos_token_id", 2);
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
