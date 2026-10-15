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
    float *d_q;              // (head_count * head_dim)
    float *d_k;              // (head_count_kv * head_dim)
    float *d_v;              // (head_count_kv * head_dim)
    float *d_ffn_gate;       // (ffn_dim)
    float *d_ffn_up;         // (ffn_dim)
    float *d_ffn_act;        // (ffn_dim)
    float *d_ffn_down;       // (embedding_length)
    float *d_logits;         // (vocab_size)

    // Quantized activation device scratch buffers
    void  *d_q8_x;           // block_q8_1_c for d_norm_x (embedding_length / 32)
    void  *d_q8_op_out;      // block_q8_1_c for conv_out or attn_out
    void  *d_q8_ffn_act;     // block_q8_1_c for ffn_act (ffn_dim / 32)

    // MoE scratch (allocated only when model->is_moe)
    int   *d_topk_ids;       // (4) selected expert ids
    float *d_topk_w;         // (4) normalized routing weights
    float *d_moe_act;        // (expert_used_count * expert_ffn_dim)
    void  *d_moe_act_q8;     // block_q8_1_c for d_moe_act (expert_used_count * expert_ffn_dim / 32)

    // Device argmax scratch buffers
    float *d_block_max;      // (vocab_size / 4)
    int   *d_block_idx;      // (vocab_size / 4)
    int   *d_best_token;     // (1 int)

    // Host buffers
    float *h_logits;         // (vocab_size)
    float *h_embd;           // (embedding_length)

    // Sampler
    sampler_params_t sampler;
} lfm25_engine_t;

// GPU lifecycle
int  lfm25_gpu_prepare(lfm25_model_t *model, int *max_context_len, int fill_ram);
void lfm25_gpu_shutdown(lfm25_model_t *model);

// GPU forward kernels
void lfm25_gpu_embed(int32_t token, float *d_x, float *h_embd, const lfm25_tensor_t *token_embd, int dim);
void lfm25_gpu_fused_rms_norm_quant(const float *d_src, const float *d_weight, float *d_dst_norm, void *d_dst_q8, int dim, float eps);
void lfm25_gpu_mmvq_q4_K(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum);
void lfm25_gpu_mmvq_q6_K(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum);
void lfm25_gpu_shortconv_step(const float *d_in_proj, float *d_conv_state, const float *d_conv_w, float *d_conv_out, void *d_conv_out_q8, int d);
void lfm25_gpu_qk_norm_rope(float *d_q, float *d_k, const float *d_q_norm, const float *d_k_norm, float *d_k_cache, int n_head, int n_head_kv, int head_dim, int pos, int max_seq, float freq_base, float eps, int layer_idx);
void lfm25_gpu_v_cache_write(const float *d_v, float *d_v_cache, int n_head_kv, int head_dim, int pos, int max_seq, int layer_idx);
void lfm25_gpu_attention(const float *d_q, const float *d_k_cache, const float *d_v_cache, float *d_out, void *d_out_q8, int n_head, int n_head_kv, int head_dim, int pos, int max_seq, int layer_idx);
void lfm25_gpu_fused_swiglu(const void *dev_gate, const void *dev_up, const void *d_xq8, float *d_act, void *d_act_q8, int in_dim, int ffn_dim);
int32_t lfm25_gpu_output_norm_head_argmax(const float *d_res_x, const float *d_norm_w, const void *dev_output_w, void *d_xq8, float *d_logits, float *h_logits, float *d_block_max, int *d_block_idx, int *d_best_token, int dim, int vocab_size, float eps);
void lfm25_gpu_router_top4(const float *d_x, const float *d_w_gate_inp, const float *d_bias, int *d_topk_ids, float *d_topk_w, float *d_logits /* nullable */, int dim, int n_expert);
void lfm25_gpu_quantize_q8_1(const float *d_x, void *d_q8, int n);
void lfm25_gpu_moe_ffn(const void *d_xq8, const int *d_topk_ids, const float *d_topk_w, const void *dev_gate_exps, const void *dev_up_exps, const void *dev_down_exps, int down_type, float *d_act /* [4*ffn_dim] */, void *d_act_q8 /* [4*ffn_dim/32] block_q8_1 */, float *d_dst, bool accum, int dim, int ffn_dim);

// Engine lifecycle
lfm25_engine_t* lfm25_engine_init(lfm25_model_t *model, int max_context_len);
void            lfm25_engine_free(lfm25_engine_t *eng);
void            lfm25_engine_set_sampler(lfm25_engine_t *eng, const sampler_params_t *params);
int32_t         lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos);

#ifdef __cplusplus
}
#endif

#endif // LFM25_ENGINE_H
