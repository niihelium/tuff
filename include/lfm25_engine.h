#ifndef LFM25_ENGINE_H
#define LFM25_ENGINE_H

#include "lfm25_model.h"
#include "sampler.h"
#include "lfm25_expert_cache.h"
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

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
    int   *d_slot_ids;       // (4) expert-cache slot indices (cache path only)
    lfm25_expert_cache_t *expert_cache; // non-NULL on the cache path
    int   *h_moe_ids;        // pinned (8): [0..3] real top-4 ids of this layer, [4..7] predicted ids of the next MoE layer
    int   *h_slot_ids;       // pinned (4): slot indices staged for the async H2D into d_slot_ids
    int   *d_pred_ids;       // (4) predicted next-layer ids (prefetch only; never feeds the math)
    float *d_pred_w;         // (4) their (unused) weights
    int    prefetch_n;       // how many of the 4 predicted experts (in router selection order) are prefetched
    int    prefetch;         // 1 = predict + prefetch the next MoE layer's experts (LFM25_PREFETCH=1; default off)

    // Device argmax scratch buffers
    float *d_block_max;      // (vocab_size / 4)
    int   *d_block_idx;      // (vocab_size / 4)
    int   *d_best_token;     // (1 int)

    // Host buffers
    float *h_logits;         // (vocab_size), pinned; filled only when logit capture is enabled
    int   *h_tokpos;         // pinned [token, pos] -> d_tokpos at the start of every step
    int   *h_best;           // pinned argmax result
    int   *d_tokpos;         // device [token, pos] read by the embed / RoPE / KV-write / attention kernels

    // Whole-trunk CUDA graph (dense + fully resident MoE; not the expert-cache path)
    cudaStream_t stream;     // explicit stream captured into the graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    int graph_enabled;       // eligible and not disabled by LFM25_CUDA_GRAPH=0

    // Batched forward (speculative-decoding verification, Task 5.1). Token-major [LFM25_MAX_BATCH][...] scratch.
    float *d_bx;             // [B * d] residual streams
    float *d_bnorm;          // [B * d] normed input (router)
    float *d_bin_proj;       // [B * 3d]
    float *d_bq, *d_bk, *d_bv;           // [B * n_head*head_dim], [B * n_head_kv*head_dim] x2
    float *d_bop_out;        // [B * d] attention output (float copy)
    float *d_bffn_act;       // [B * ffn_dim]
    void  *d_bq8_x, *d_bq8_op, *d_bq8_ffn;   // block_q8_1 per token
    float *d_blogits;        // [B * vocab]
    float *d_bblock_max;     // [B * ceil(vocab/4)]
    int   *d_bblock_idx;     // [B * ceil(vocab/4)]
    int   *d_bbest;          // [B]
    int   *d_btokpos;        // [(B+1) * 2] (token,pos) pairs; one spare pair so pos pointers never run off the end
    int   *h_btokpos;        // pinned copy of d_btokpos
    int   *h_bbest;          // pinned [B]
    float *d_conv_snap;      // [n_conv_layers][B][2d] conv state after each token of the last batch (rollback source)
    int    batch_n;          // tokens in the last forward_batch (0 = no rollback possible)

    // Sampler
    sampler_params_t sampler;
} lfm25_engine_t;

// GPU lifecycle
int  lfm25_gpu_prepare(lfm25_model_t *model, int *max_context_len, int fill_ram);
void lfm25_gpu_shutdown(lfm25_model_t *model);

// GPU forward kernels
void lfm25_gpu_set_stream(cudaStream_t s);  // stream for all forward-path kernels (0 = default stream)
void lfm25_gpu_embed(const int *d_tokpos /* [token,pos] on device */, float *d_x, const lfm25_tensor_t *token_embd, int dim);
void lfm25_gpu_fused_rms_norm_quant(const float *d_src, const float *d_weight, float *d_dst_norm, void *d_dst_q8, int dim, float eps);
void lfm25_gpu_mmvq_q4_K(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum);
void lfm25_gpu_mmvq_q6_K(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum);
void lfm25_gpu_shortconv_step(const float *d_in_proj, float *d_conv_state, const float *d_conv_w, float *d_conv_out, void *d_conv_out_q8, int d);
void lfm25_gpu_inproj_shortconv(const lfm25_tensor_t *in_proj, const void *d_xq8, float *d_conv_state, const float *d_conv_w, float *d_conv_out, void *d_conv_out_q8, int d);
void lfm25_gpu_qk_norm_rope(float *d_q, float *d_k, const float *d_q_norm, const float *d_k_norm, float *d_k_cache, int n_head, int n_head_kv, int head_dim, const int *d_pos, int max_seq, float freq_base, float eps, int layer_idx);
void lfm25_gpu_v_cache_write(const float *d_v, float *d_v_cache, int n_head_kv, int head_dim, const int *d_pos, int max_seq, int layer_idx);
void lfm25_gpu_attention(const float *d_q, const float *d_k_cache, const float *d_v_cache, float *d_out, void *d_out_q8, int n_head, int n_head_kv, int head_dim, const int *d_pos, int max_seq, int layer_idx);
void lfm25_gpu_fused_swiglu(const void *dev_gate, const void *dev_up, const void *d_xq8, float *d_act, void *d_act_q8, int in_dim, int ffn_dim);
void lfm25_gpu_output_norm_head_argmax(const float *d_res_x, const float *d_norm_w, const void *dev_output_w, void *d_xq8, float *d_logits, float *d_block_max, int *d_block_idx, int *d_best_token, int dim, int vocab_size, float eps);
void lfm25_gpu_router_top4(const float *d_x, const float *d_w_gate_inp, const float *d_bias, int *d_topk_ids, float *d_topk_w, float *d_logits /* nullable */, int dim, int n_expert);
void lfm25_gpu_quantize_q8_1(const float *d_x, void *d_q8, int n);
// Batched (multi-token) launchers for speculative-decoding verification; column b is bit-identical to the single-token kernel.
#define LFM25_MAX_BATCH 8
void lfm25_gpu_mmvq_q4_K_batch(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum, int nb);
void lfm25_gpu_mmvq_q6_K_batch(const void *dev_w, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum, int nb);
void lfm25_gpu_fused_swiglu_batch(const void *dev_gate, const void *dev_up, const void *d_xq8, float *d_act, void *d_act_q8, int in_dim, int ffn_dim, int nb);
void lfm25_gpu_shortconv_seq(const float *d_in_proj, float *d_conv_state, float *d_snap, const float *d_conv_w, void *d_conv_out_q8, int d, int nb);
void lfm25_gpu_conv_state_restore(float *d_conv_state, const float *d_snap, int n_layers, int slot, int d, size_t snap_layer_stride);
void lfm25_gpu_output_norm_head_argmax_batch(const float *d_res_x, const float *d_norm_w, const void *dev_output_w, void *d_xq8, float *d_logits, float *d_block_max, int *d_block_idx, int *d_best_token, int dim, int vocab_size, float eps, int nb);
void lfm25_gpu_moe_ffn(const void *d_xq8, const int *d_topk_ids, const float *d_topk_w, const void *dev_gate_exps, const void *dev_up_exps, const void *dev_down_exps, int down_type, float *d_act /* [4*ffn_dim] */, void *d_act_q8 /* [4*ffn_dim/32] block_q8_1 */, float *d_dst, bool accum, int dim, int ffn_dim, size_t gate_stride, size_t up_stride, size_t down_stride /* bytes per expert / slot */);

// Engine lifecycle
lfm25_engine_t* lfm25_engine_init(lfm25_model_t *model, int max_context_len);
void            lfm25_engine_free(lfm25_engine_t *eng);
void            lfm25_engine_set_sampler(lfm25_engine_t *eng, const sampler_params_t *params);
int32_t         lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos);
// Process tokens[0..n) at positions pos0..pos0+n-1 exactly as n consecutive forward_token calls would (bit-identical
// logits) and return every position's greedy argmax (+ host logits [n*vocab] if logits_out != NULL). 1 <= n <= LFM25_MAX_BATCH.
// Returns 0 on success, -1 on invalid arguments. The engine state afterwards is "all n tokens processed".
int             lfm25_engine_forward_batch(lfm25_engine_t *eng, const int32_t *tokens, int n, int pos0, int32_t *argmax_out, float *logits_out);
// After a forward_batch of n tokens: make the state equal to having processed only the first n_keep (1 <= n_keep <= n).
// Conv states come from per-position snapshots; KV entries beyond are overwritten later. Returns 0 / -1.
int             lfm25_engine_rollback(lfm25_engine_t *eng, int n_keep);

#ifdef __cplusplus
}
#endif

#endif // LFM25_ENGINE_H
