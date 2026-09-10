#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool bind_tensor(const gguf_context_t *gguf, tensor_t *t, const char *name, bool required) {
    const gguf_tensor_info_t *info = tuff_gguf_get_tensor(gguf, name);
    if (!info) {
        if (required) {
            fprintf(stderr, "[FATAL ERROR] Required tensor '%s' missing from GGUF file!\n", name);
            exit(EXIT_FAILURE);
        }
        memset(t, 0, sizeof(tensor_t));
        return false;
    }
    t->info = *info;
    t->data = gguf->tensor_data_blob + info->offset;
    return true;
}

model_t* model_load(const char *gguf_filepath) {
    gguf_context_t *gguf = tuff_gguf_parse_file(gguf_filepath);
    if (!gguf) return NULL;

    model_t *m = (model_t*)calloc(1, sizeof(model_t));
    m->gguf = gguf;

    // Load Hyperparameters from Metadata KV
    m->block_count              = tuff_gguf_get_kv_u32(gguf, "qwen35moe.block_count", 41);
    m->embedding_length         = tuff_gguf_get_kv_u32(gguf, "qwen35moe.embedding_length", 2048);
    m->context_length           = tuff_gguf_get_kv_u32(gguf, "qwen35moe.context_length", 262144);
    m->head_count               = tuff_gguf_get_kv_u32(gguf, "qwen35moe.attention.head_count", 16);
    m->head_count_kv            = tuff_gguf_get_kv_u32(gguf, "qwen35moe.attention.head_count_kv", 2);
    m->key_length               = tuff_gguf_get_kv_u32(gguf, "qwen35moe.attention.key_length", 256);
    m->value_length             = tuff_gguf_get_kv_u32(gguf, "qwen35moe.attention.value_length", 256);
    m->rope_freq_base           = tuff_gguf_get_kv_f32(gguf, "qwen35moe.rope.freq_base", 1e7f);
    m->rope_dim_count           = tuff_gguf_get_kv_u32(gguf, "qwen35moe.rope.dimension_count", 64);
    m->full_attention_interval  = tuff_gguf_get_kv_u32(gguf, "qwen35moe.full_attention_interval", 4);
    m->expert_count             = tuff_gguf_get_kv_u32(gguf, "qwen35moe.expert_count", 256);
    m->expert_used_count        = tuff_gguf_get_kv_u32(gguf, "qwen35moe.expert_used_count", 8);
    m->expert_ffn_length        = tuff_gguf_get_kv_u32(gguf, "qwen35moe.expert_feed_forward_length", 512);
    m->expert_shared_ffn_len    = tuff_gguf_get_kv_u32(gguf, "qwen35moe.expert_shared_feed_forward_length", 512);
    m->ssm_conv_kernel          = tuff_gguf_get_kv_u32(gguf, "qwen35moe.ssm.conv_kernel", 4);
    m->ssm_state_size           = tuff_gguf_get_kv_u32(gguf, "qwen35moe.ssm.state_size", 128);
    m->ssm_group_count          = tuff_gguf_get_kv_u32(gguf, "qwen35moe.ssm.group_count", 16);
    m->ssm_time_step_rank       = tuff_gguf_get_kv_u32(gguf, "qwen35moe.ssm.time_step_rank", 32);
    m->ssm_inner_size           = tuff_gguf_get_kv_u32(gguf, "qwen35moe.ssm.inner_size", 4096);
    m->general_alignment        = gguf->alignment;
    m->rms_norm_eps             = tuff_gguf_get_kv_f32(gguf, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);

    // Derived gated delta-net dims (see llama.cpp src/models/qwen35moe.cpp load_block_trunk):
    //   head_k_dim = head_v_dim = ssm.state_size
    //   n_k_heads  = ssm.group_count,  n_v_heads = ssm.time_step_rank
    //   key_dim    = head_k_dim * n_k_heads
    //   value_dim  = head_v_dim * n_v_heads  (== ssm.inner_size)
    //   qkv_dim    = 2*key_dim + value_dim   (conv1d channel count)
    m->ssm_head_k_dim = m->ssm_state_size;
    m->ssm_head_v_dim = m->ssm_state_size;
    m->ssm_n_k_heads  = m->ssm_group_count;
    m->ssm_n_v_heads  = m->ssm_time_step_rank;
    m->ssm_key_dim    = m->ssm_head_k_dim * m->ssm_n_k_heads;
    m->ssm_value_dim  = m->ssm_head_v_dim * m->ssm_n_v_heads;
    m->ssm_qkv_dim    = 2 * m->ssm_key_dim + m->ssm_value_dim;
    m->ssm_conv_dim   = m->ssm_qkv_dim;
    m->nextn_predict_layers = tuff_gguf_get_kv_u32(gguf, "qwen35moe.nextn_predict_layers", 0);

    // Load rope.dimension_sections if present
    const gguf_kv_t *rope_sec = tuff_gguf_get_kv(gguf, "qwen35moe.rope.dimension_sections");
    if (rope_sec && rope_sec->type == GGUF_TYPE_ARRAY && rope_sec->val.arr.len >= 4 && rope_sec->val.arr.data) {
        const uint32_t *arr = (const uint32_t*)rope_sec->val.arr.data;
        m->rope_dim_sections[0] = arr[0];
        m->rope_dim_sections[1] = arr[1];
        m->rope_dim_sections[2] = arr[2];
        m->rope_dim_sections[3] = arr[3];
    } else {
        m->rope_dim_sections[0] = 11;
        m->rope_dim_sections[1] = 11;
        m->rope_dim_sections[2] = 10;
        m->rope_dim_sections[3] = 0;
    }

    // Bind Global Tensors
    bind_tensor(gguf, &m->token_embd, "token_embd.weight", true);
    bind_tensor(gguf, &m->output_norm, "output_norm.weight", true);
    bind_tensor(gguf, &m->output_weight, "output.weight", true);

    m->vocab_size = (uint32_t)m->token_embd.info.dims[1];

    // Allocate & Bind Layers
    m->layers = (model_layer_t*)calloc(m->block_count, sizeof(model_layer_t));
    for (uint32_t i = 0; i < m->block_count; ++i) {
        model_layer_t *l = &m->layers[i];
        l->layer_idx = i;

        char name[128];
        // Norms
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        bind_tensor(gguf, &l->attn_norm, name, true);

        snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", i);
        bind_tensor(gguf, &l->post_attention_norm, name, true);

        // MoE Tensors
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", i);
        bind_tensor(gguf, &l->ffn_gate_inp, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", i);
        bind_tensor(gguf, &l->ffn_gate_exps, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", i);
        bind_tensor(gguf, &l->ffn_up_exps, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", i);
        bind_tensor(gguf, &l->ffn_down_exps, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", i);
        bind_tensor(gguf, &l->ffn_gate_shexp, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", i);
        bind_tensor(gguf, &l->ffn_up_shexp, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", i);
        bind_tensor(gguf, &l->ffn_down_shexp, name, true);

        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp_shexp.weight", i);
        bind_tensor(gguf, &l->ffn_gate_inp_shexp, name, true);

        // Determine Layer Type (Attention vs Delta-Net)
        // Check for presence of blk.N.attn_q.weight vs blk.N.attn_qkv.weight
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
        if (tuff_gguf_get_tensor(gguf, name) != NULL) {
            l->type = LAYER_TYPE_ATTENTION;
            bind_tensor(gguf, &l->attn_q, name, true);

            snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
            bind_tensor(gguf, &l->attn_k, name, true);

            snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
            bind_tensor(gguf, &l->attn_v, name, true);

            snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
            bind_tensor(gguf, &l->attn_output, name, true);

            snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", i);
            bind_tensor(gguf, &l->attn_q_norm, name, true);

            snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", i);
            bind_tensor(gguf, &l->attn_k_norm, name, true);
        } else {
            l->type = LAYER_TYPE_DELTA_NET;
            snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", i);
            bind_tensor(gguf, &l->attn_qkv, name, true);

            snprintf(name, sizeof(name), "blk.%u.attn_gate.weight", i);
            bind_tensor(gguf, &l->attn_gate, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_out.weight", i);
            bind_tensor(gguf, &l->ssm_out, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", i);
            bind_tensor(gguf, &l->ssm_conv1d, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_alpha.weight", i);
            bind_tensor(gguf, &l->ssm_alpha, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_beta.weight", i);
            bind_tensor(gguf, &l->ssm_beta, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_a", i);
            bind_tensor(gguf, &l->ssm_a, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", i);
            bind_tensor(gguf, &l->ssm_dt_bias, name, true);

            snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", i);
            bind_tensor(gguf, &l->ssm_norm, name, true);
        }

        // Bind optional NextN / MTP prediction head tensors
        snprintf(name, sizeof(name), "blk.%u.nextn.eh_proj.weight", i);
        if (!bind_tensor(gguf, &l->nextn_eh_proj, name, false)) {
            snprintf(name, sizeof(name), "blk.%u.nextn_eh_proj.weight", i);
            bind_tensor(gguf, &l->nextn_eh_proj, name, false);
        }
        snprintf(name, sizeof(name), "blk.%u.nextn.hnorm.weight", i);
        if (!bind_tensor(gguf, &l->nextn_hnorm, name, false)) {
            snprintf(name, sizeof(name), "blk.%u.nextn_hnorm.weight", i);
            bind_tensor(gguf, &l->nextn_hnorm, name, false);
        }
        snprintf(name, sizeof(name), "blk.%u.nextn.enorm.weight", i);
        if (!bind_tensor(gguf, &l->nextn_enorm, name, false)) {
            snprintf(name, sizeof(name), "blk.%u.nextn_enorm.weight", i);
            bind_tensor(gguf, &l->nextn_enorm, name, false);
        }
        snprintf(name, sizeof(name), "blk.%u.nextn.shared_head_norm.weight", i);
        bind_tensor(gguf, &l->nextn_shared_head_norm, name, false);
    }

    m->n_attn_layers = 0;
    m->n_delta_layers = 0;
    for (uint32_t i = 0; i < m->block_count; ++i) {
        if (m->layers[i].type == LAYER_TYPE_ATTENTION) m->n_attn_layers++;
        else m->n_delta_layers++;
    }

    return m;
}

void model_print_inspection(const model_t *m) {
    if (!m) return;

    uint32_t n_attn = 0;
    uint32_t n_delta = 0;
    size_t total_model_bytes = 0;
    size_t dense_bytes = 0;
    size_t expert_bytes = 0;

    for (uint64_t i = 0; i < m->gguf->n_tensors; ++i) {
        size_t b = m->gguf->tensors[i].size_bytes;
        total_model_bytes += b;
        if (strstr(m->gguf->tensors[i].name, "ffn_gate_exps") ||
            strstr(m->gguf->tensors[i].name, "ffn_up_exps") ||
            strstr(m->gguf->tensors[i].name, "ffn_down_exps")) {
            expert_bytes += b;
        } else if (strcmp(m->gguf->tensors[i].name, "token_embd.weight") != 0) {
            dense_bytes += b;
        }
    }

    for (uint32_t i = 0; i < m->block_count; ++i) {
        if (m->layers[i].type == LAYER_TYPE_ATTENTION) n_attn++;
        else n_delta++;
    }

    size_t single_expert_bytes = m->layers[0].ffn_gate_exps.info.size_bytes / m->expert_count +
                                m->layers[0].ffn_up_exps.info.size_bytes / m->expert_count +
                                m->layers[0].ffn_down_exps.info.size_bytes / m->expert_count;

    printf("\n========================================================================\n");
    printf("                  TUFF MODEL INSPECTION REPORT                    \n");
    printf("========================================================================\n");
    printf(" Architecture: qwen35moe\n");
    printf(" Total Layers (block_count): %u  [Attention: %u | Delta-Net: %u]\n",
           m->block_count, n_attn, n_delta);
    printf(" Embedding Length:           %u\n", m->embedding_length);
    printf(" Context Length:             %u\n", m->context_length);
    printf(" Vocabulary Size:            %u\n", m->vocab_size);
    printf(" Attention Heads (Q/KV):     %u / %u (Key Dim: %u, Value Dim: %u)\n",
           m->head_count, m->head_count_kv, m->key_length, m->value_length);
    printf(" RoPE Freq Base:             %.1fe7 (Dim Count: %u, mRoPE: [%u,%u,%u,%u])\n",
           m->rope_freq_base / 1e7f, m->rope_dim_count,
           m->rope_dim_sections[0], m->rope_dim_sections[1],
           m->rope_dim_sections[2], m->rope_dim_sections[3]);
    printf(" Full Attention Interval:    %u\n", m->full_attention_interval);
    printf(" MoE Experts (Total/Active): %u / %u (FFN Dim: %u, Shared FFN Dim: %u)\n",
           m->expert_count, m->expert_used_count, m->expert_ffn_length, m->expert_shared_ffn_len);
    printf(" Gated Delta-Net Dims:       Conv Kernel: %u | State: %u | Groups: %u | Rank: %u | Inner: %u\n",
           m->ssm_conv_kernel, m->ssm_state_size, m->ssm_group_count, m->ssm_time_step_rank, m->ssm_inner_size);
    printf("------------------------------------------------------------------------\n");
    printf(" MEMORY BREAKDOWN:\n");
    printf(" Total GGUF File Size:       %.2f GiB (%.2f MB)\n",
           m->gguf->file_size / (1024.0 * 1024.0 * 1024.0), m->gguf->file_size / (1024.0 * 1024.0));
    printf(" Total Tensor Data Weight:   %.2f GiB\n", total_model_bytes / (1024.0 * 1024.0 * 1024.0));
    printf(" Total Expert Weights:       %.2f GiB (%.2f MB / expert, %u total experts)\n",
           expert_bytes / (1024.0 * 1024.0 * 1024.0),
           single_expert_bytes / (1024.0 * 1024.0),
           m->block_count * m->expert_count);
    printf(" Dense GPU Weights:          %.2f GiB (excluding token_embd & experts)\n",
           dense_bytes / (1024.0 * 1024.0 * 1024.0));
    printf(" Active Experts / Token:     %.2f MB (41 layers x %u active x %.2f MB)\n",
           (m->block_count * m->expert_used_count * single_expert_bytes) / (1024.0 * 1024.0),
           m->expert_used_count, single_expert_bytes / (1024.0 * 1024.0));
    if (m->nextn_predict_layers > 0 && m->block_count > 40) {
        printf(" All Tensors in Layer 40 (NextN / MTP):\n");
        for (uint64_t i = 0; i < m->gguf->n_tensors; ++i) {
            if (strstr(m->gguf->tensors[i].name, "blk.40.")) {
                printf("   %s (type=%d, size=%zu)\n",
                       m->gguf->tensors[i].name,
                       m->gguf->tensors[i].type,
                       m->gguf->tensors[i].size_bytes);
            }
        }
    }
    printf("========================================================================\n");
    for (uint32_t i = 0; i < m->block_count; ++i) {
        const model_layer_t *l = &m->layers[i];
        if (l->ffn_down_exps.info.type != l->ffn_gate_exps.info.type)
            printf(" TYPE blk.%u: gate_exps=%d up_exps=%d down_exps=%d\n",
                   i, (int)l->ffn_gate_exps.info.type, (int)l->ffn_up_exps.info.type,
                   (int)l->ffn_down_exps.info.type);
    }
    printf("========================================================================\n\n");
    fflush(stdout);
}

void model_free(model_t *m) {
    if (!m) return;
    if (m->layers) free(m->layers);
    if (m->gguf) tuff_gguf_free(m->gguf);
    free(m);
}
