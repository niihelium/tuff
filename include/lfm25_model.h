#ifndef LFM25_MODEL_H
#define LFM25_MODEL_H

#include "gguf_parser.h"
#include "ggml_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LFM25_OP_CONV,
    LFM25_OP_ATTN
} lfm25_op_type_t;

typedef enum {
    LFM25_FFN_DENSE,
    LFM25_FFN_MOE
} lfm25_ffn_type_t;

typedef struct {
    const void *data;      // host pointer into mmap GGUF blob
    void *dev;             // device copy for GPU residency
    gguf_tensor_info_t info;
} lfm25_tensor_t;

typedef struct {
    lfm25_op_type_t  op_type;   // LFM25_OP_CONV or LFM25_OP_ATTN
    lfm25_ffn_type_t ffn_type;  // LFM25_FFN_DENSE or LFM25_FFN_MOE
    uint32_t         layer_idx;

    // Operator and FFN Norms
    lfm25_tensor_t attn_norm;
    lfm25_tensor_t ffn_norm;

    // Short-Convolution specific tensors (if op_type == LFM25_OP_CONV)
    lfm25_tensor_t shortconv_in_proj;   // (d_model, 3 * d_model) Q4_K
    lfm25_tensor_t shortconv_conv;      // (3, d_model) F32
    lfm25_tensor_t shortconv_out_proj;  // (d_model, d_model) Q4_K

    // GQA Attention specific tensors (if op_type == LFM25_OP_ATTN)
    lfm25_tensor_t attn_q;              // (d_model, n_heads * d_head) Q4_K
    lfm25_tensor_t attn_k;              // (d_model, n_kv_heads * d_head) Q4_K
    lfm25_tensor_t attn_v;              // (d_model, n_kv_heads * d_head) Q6_K
    lfm25_tensor_t attn_output;         // (n_heads * d_head, d_model) Q4_K
    lfm25_tensor_t attn_q_norm;         // (d_head,) F32
    lfm25_tensor_t attn_k_norm;         // (d_head,) F32

    // Dense SwiGLU FFN specific tensors (if ffn_type == LFM25_FFN_DENSE)
    lfm25_tensor_t ffn_gate;            // (d_model, ffn_dim) Q4_K
    lfm25_tensor_t ffn_up;              // (d_model, ffn_dim) Q4_K
    lfm25_tensor_t ffn_down;            // (ffn_dim, d_model) Q6_K

    // MoE specific tensors (if ffn_type == LFM25_FFN_MOE)
    lfm25_tensor_t ffn_gate_inp;        // (d_model, expert_count) F32 router
    lfm25_tensor_t exp_probs_b;         // (expert_count,) F32 router bias
    lfm25_tensor_t ffn_gate_exps;       // (d_model, expert_ffn_dim, expert_count) Q4_K
    lfm25_tensor_t ffn_up_exps;         // (d_model, expert_ffn_dim, expert_count) Q4_K
    lfm25_tensor_t ffn_down_exps;       // (expert_ffn_dim, d_model, expert_count) Q6_K
} lfm25_layer_t;

typedef struct {
    // Hyperparameters
    uint32_t block_count;              // 16 (1.2B), 30 (2.6B), 24 (8B), 40 (24B)
    uint32_t embedding_length;         // 2048
    uint32_t context_length;           // 128000 or 131072
    uint32_t head_count;               // 32
    uint32_t head_count_kv;            // 8 (max KV heads)
    uint32_t head_dim;                 // 64
    uint32_t vocab_size;               // 65536 or 128000
    float    rms_norm_eps;             // 1e-5
    float    rope_freq_base;           // 1e7
    
    // Short-Convolution params
    uint32_t conv_kernel_size;         // K = 3
    uint32_t conv_state_len;           // K - 1 = 2 past tokens

    // MoE params
    bool     is_moe;                   // true for 8B and 24B
    uint32_t expert_count;             // 32 (8B), 64 (24B)
    uint32_t expert_used_count;        // 4
    uint32_t expert_ffn_dim;           // 1792 (8B), 1536 (24B)
    uint32_t leading_dense_blocks;     // 2 (layers 0..1)

    // Layer type tallies
    uint32_t n_conv_layers;
    uint32_t n_attn_layers;
    uint32_t n_dense_ffn_layers;
    uint32_t n_moe_layers;

    // Trunk Tensors
    lfm25_tensor_t token_embd;         // (2048, vocab_size)
    lfm25_tensor_t token_embd_norm;    // (2048,) Final Output RMSNorm (Liquid AI quirk)
    lfm25_tensor_t output_weight;      // (2048, vocab_size) LM Head

    // Layer array
    lfm25_layer_t *layers;

    gguf_context_t *gguf;
} lfm25_model_t;

lfm25_model_t* lfm25_model_load(const char *gguf_filepath);
void           lfm25_model_print_inspection(const lfm25_model_t *model);
void           lfm25_model_free(lfm25_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // LFM25_MODEL_H
