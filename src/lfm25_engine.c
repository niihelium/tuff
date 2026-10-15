#include "lfm25_engine.h"
#include "lfm25_model.h"
#include "tuff_vtable.h"
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

    // Allocate Conv1D rolling state buffer: [n_conv_layers * conv_state_len * d]
    size_t conv_state_bytes = (size_t)model->n_conv_layers * model->conv_state_len * d * sizeof(float);
    if (conv_state_bytes > 0) {
        CUDA_CHECK(cudaMalloc((void**)&eng->d_conv_state, conv_state_bytes));
        CUDA_CHECK(cudaMemset(eng->d_conv_state, 0, conv_state_bytes));
    }

    // Allocate KV Cache for Attention layers: [n_attn_layers * head_count_kv * max_context_len * head_dim]
    size_t kv_cache_bytes = (size_t)model->n_attn_layers * model->head_count_kv * eng->max_context_len * model->head_dim * sizeof(float);
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

    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_gate, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_up, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_act, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_down, (size_t)d * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&eng->d_logits, (size_t)vocab * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_logits, (size_t)vocab * sizeof(float)));

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
    if (eng->d_ffn_gate)   cudaFree(eng->d_ffn_gate);
    if (eng->d_ffn_up)     cudaFree(eng->d_ffn_up);
    if (eng->d_ffn_act)    cudaFree(eng->d_ffn_act);
    if (eng->d_ffn_down)   cudaFree(eng->d_ffn_down);
    if (eng->d_logits)     cudaFree(eng->d_logits);

    if (eng->h_logits)     cudaFreeHost(eng->h_logits);

    free(eng);
}

void lfm25_engine_set_sampler(lfm25_engine_t *eng, const sampler_params_t *params) {
    if (eng && params) {
        eng->sampler = *params;
    }
}

int32_t lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos) {
    (void)token;
    (void)pos;
    if (!eng || !eng->model) return -1;
    // Core kernel execution will be assembled in Task 1.4
    return 0;
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
