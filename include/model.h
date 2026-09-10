#ifndef MODEL_H
#define MODEL_H

#include "gguf_parser.h"
#include "ggml_types.h"
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    LAYER_TYPE_ATTENTION,
    LAYER_TYPE_DELTA_NET
} layer_type_t;

typedef struct {
    const void *data;      // host pointer into the mmap'd GGUF blob
    void *dev;             // device copy for dense weights (M2+); NULL = CPU only
    void *dev_q8;          // H22: optional Q8_0 re-quantized device copy (TUFF_ROUTER_Q8)
    gguf_tensor_info_t info;
} tensor_t;

typedef struct {
    layer_type_t type; // LAYER_TYPE_ATTENTION or LAYER_TYPE_DELTA_NET
    uint32_t layer_idx;

    tensor_t attn_norm;
    tensor_t post_attention_norm;

    // MoE Tensors
    tensor_t ffn_gate_inp;       // (2048, 256) F32
    tensor_t ffn_gate_exps;      // (2048, 512, 256) Q4_K
    tensor_t ffn_up_exps;        // (2048, 512, 256) Q4_K
    tensor_t ffn_down_exps;      // (512, 2048, 256) Q4_K
    tensor_t ffn_gate_shexp;     // (2048, 512) Q8_0
    tensor_t ffn_up_shexp;       // (2048, 512) Q8_0
    tensor_t ffn_down_shexp;     // (512, 2048) Q8_0
    tensor_t ffn_gate_inp_shexp; // (2048,) F32

    // Attention layer specific (11 layers)
    tensor_t attn_q;             // Q8_0
    tensor_t attn_k;             // Q8_0
    tensor_t attn_v;             // Q8_0
    tensor_t attn_output;        // Q8_0
    tensor_t attn_q_norm;        // F32
    tensor_t attn_k_norm;        // F32

    // Delta-net layer specific (30 layers)
    tensor_t attn_qkv;           // Q8_0
    tensor_t attn_gate;          // Q8_0
    tensor_t ssm_out;            // Q8_0
    tensor_t ssm_conv1d;         // F32
    tensor_t ssm_alpha;          // F32
    tensor_t ssm_beta;           // F32
    tensor_t ssm_a;              // F32
    tensor_t ssm_dt_bias;        // F32
    tensor_t ssm_norm;           // F32

    // NextN / MTP prediction head specific (layer 40)
    tensor_t nextn_eh_proj;          // Q8_0 (4096, 2048)
    tensor_t nextn_hnorm;            // F32 (2048)
    tensor_t nextn_enorm;            // F32 (2048)
    tensor_t nextn_shared_head_norm; // F32 (2048)
} model_layer_t;

typedef struct {
    // Hyperparameters
    uint32_t block_count;              // 41
    uint32_t embedding_length;         // 2048
    uint32_t context_length;           // 262144
    uint32_t head_count;               // 16
    uint32_t head_count_kv;            // 2
    uint32_t key_length;               // 256
    uint32_t value_length;             // 256
    float    rope_freq_base;           // 1e7
    uint32_t rope_dim_count;           // 64
    uint32_t rope_dim_sections[4];     // [11, 11, 10, 0]
    uint32_t full_attention_interval; // 4
    uint32_t expert_count;             // 256
    uint32_t expert_used_count;        // 8
    uint32_t expert_ffn_length;        // 512
    uint32_t expert_shared_ffn_len;    // 512
    uint32_t ssm_conv_kernel;          // 4
    uint32_t ssm_state_size;           // 128
    uint32_t ssm_group_count;          // 16
    uint32_t ssm_time_step_rank;       // 32
    uint32_t ssm_inner_size;           // 4096
    uint32_t vocab_size;               // 248320
    uint32_t general_alignment;        // 32
    float    rms_norm_eps;             // qwen35moe.attention.layer_norm_rms_epsilon

    // Derived gated delta-net dims (computed from the ssm.* hparams above,
    // see PLAN.md sec.1 "Derived SSM dims" and llama.cpp qwen35moe.cpp load_block_trunk)
    uint32_t ssm_head_k_dim;           // = ssm_state_size            (128)
    uint32_t ssm_head_v_dim;           // = ssm_state_size            (128)
    uint32_t ssm_n_k_heads;            // = ssm_group_count           (16)
    uint32_t ssm_n_v_heads;            // = ssm_time_step_rank        (32)
    uint32_t ssm_key_dim;              // = head_k_dim * n_k_heads    (2048)
    uint32_t ssm_value_dim;            // = head_v_dim * n_v_heads = ssm_inner_size (4096)
    uint32_t ssm_qkv_dim;              // = 2*key_dim + value_dim     (8192)
    uint32_t ssm_conv_dim;             // same as ssm_qkv_dim, conv1d channel count
    uint32_t nextn_predict_layers;     // MTP head block count; excluded from the main pass

    // Layer type counts, derived by scanning the loaded layers (never hardcoded).
    uint32_t n_attn_layers;
    uint32_t n_delta_layers;

    // Tensors
    tensor_t token_embd;
    tensor_t output_weight;
    tensor_t output_norm;

    model_layer_t *layers;

    gguf_context_t *gguf;
} model_t;

model_t* model_load(const char *gguf_filepath);
void model_print_inspection(const model_t *model);
void model_free(model_t *model);

#endif // MODEL_H
