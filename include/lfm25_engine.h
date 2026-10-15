#ifndef LFM25_ENGINE_H
#define LFM25_ENGINE_H

#include "lfm25_model.h"
#include "sampler.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    lfm25_model_t *model;
    int max_context_len;

    // Device state buffers
    float *d_conv_state;     // [n_conv_layers * 2 * embedding_length]
    float *d_k_cache;        // [n_attn_layers * head_count_kv * max_context_len * head_dim]
    float *d_v_cache;        // [n_attn_layers * head_count_kv * max_context_len * head_dim]

    // Device scratch buffers
    float *d_x;              // (embedding_length)
    float *d_norm_x;         // (embedding_length)
    float *d_in_proj;        // (3 * embedding_length)
    float *d_conv_out;       // (embedding_length)
    float *d_op_out;         // (embedding_length)
    float *d_ffn_gate;       // (ffn_dim)
    float *d_ffn_up;         // (ffn_dim)
    float *d_ffn_act;        // (ffn_dim)
    float *d_ffn_down;       // (embedding_length)
    float *d_logits;         // (vocab_size)

    // Host buffers
    float *h_logits;         // (vocab_size)

    // Sampler
    sampler_params_t sampler;
} lfm25_engine_t;

// GPU lifecycle
int  lfm25_gpu_prepare(lfm25_model_t *model, int *max_context_len, int fill_ram);
void lfm25_gpu_shutdown(lfm25_model_t *model);

// Engine lifecycle
lfm25_engine_t* lfm25_engine_init(lfm25_model_t *model, int max_context_len);
void            lfm25_engine_free(lfm25_engine_t *eng);
void            lfm25_engine_set_sampler(lfm25_engine_t *eng, const sampler_params_t *params);
int32_t         lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos);

#ifdef __cplusplus
}
#endif

#endif // LFM25_ENGINE_H
