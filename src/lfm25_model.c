#include "lfm25_model.h"
#include "tuff_vtable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

tuff_arch_t tuff_detect_arch(const char *gguf_filepath) {
    gguf_context_t *gguf = tuff_gguf_parse_file(gguf_filepath);
    if (!gguf) return TUFF_ARCH_UNKNOWN;

    const gguf_kv_t *kv = tuff_gguf_get_kv(gguf, "general.architecture");
    tuff_arch_t arch = TUFF_ARCH_UNKNOWN;
    if (kv && kv->type == GGUF_TYPE_STRING) {
        if (strcmp(kv->val.str, "qwen35moe") == 0) {
            arch = TUFF_ARCH_QWEN35MOE;
        } else if (strcmp(kv->val.str, "lfm2") == 0) {
            arch = TUFF_ARCH_LFM2;
        } else if (strcmp(kv->val.str, "lfm2moe") == 0) {
            arch = TUFF_ARCH_LFM2MOE;
        }
    }
    tuff_gguf_free(gguf);
    return arch;
}

const char* tuff_arch_to_string(tuff_arch_t arch) {
    switch (arch) {
        case TUFF_ARCH_QWEN35MOE: return "qwen35moe";
        case TUFF_ARCH_LFM2:      return "lfm2";
        case TUFF_ARCH_LFM2MOE:   return "lfm2moe";
        default:                  return "unknown";
    }
}

static bool bind_tensor(const gguf_context_t *gguf, lfm25_tensor_t *t, const char *name, bool required) {
    const gguf_tensor_info_t *info = tuff_gguf_get_tensor(gguf, name);
    if (!info) {
        if (required) {
            fprintf(stderr, "[LFM25] FATAL: Required tensor '%s' missing from GGUF file!\n", name);
            exit(EXIT_FAILURE);
        }
        memset(t, 0, sizeof(lfm25_tensor_t));
        return false;
    }
    t->info = *info;
    t->data = gguf->tensor_data_blob + info->offset;
    t->dev = NULL;
    return true;
}

lfm25_model_t* lfm25_model_load(const char *gguf_filepath) {
    gguf_context_t *gguf = tuff_gguf_parse_file(gguf_filepath);
    if (!gguf) return NULL;

    const gguf_kv_t *arch_kv = tuff_gguf_get_kv(gguf, "general.architecture");
    if (!arch_kv || arch_kv->type != GGUF_TYPE_STRING) {
        fprintf(stderr, "[LFM25] Missing general.architecture metadata in GGUF\n");
        tuff_gguf_free(gguf);
        return NULL;
    }

    bool is_moe = (strcmp(arch_kv->val.str, "lfm2moe") == 0);
    const char *pfx = is_moe ? "lfm2moe." : "lfm2.";

    lfm25_model_t *m = (lfm25_model_t*)calloc(1, sizeof(lfm25_model_t));
    if (!m) {
        tuff_gguf_free(gguf);
        return NULL;
    }
    m->gguf = gguf;
    m->is_moe = is_moe;

    // Load hyperparameters
    char key[128];
    snprintf(key, sizeof(key), "%sblock_count", pfx);
    m->block_count = tuff_gguf_get_kv_u32(gguf, key, 16);

    snprintf(key, sizeof(key), "%sembedding_length", pfx);
    m->embedding_length = tuff_gguf_get_kv_u32(gguf, key, 2048);

    snprintf(key, sizeof(key), "%scontext_length", pfx);
    m->context_length = tuff_gguf_get_kv_u32(gguf, key, 128000);

    snprintf(key, sizeof(key), "%sattention.head_count", pfx);
    m->head_count = tuff_gguf_get_kv_u32(gguf, key, 32);

    snprintf(key, sizeof(key), "%sattention.head_count_kv", pfx);
    // head_count_kv can be u32 or array; default to 8
    const gguf_kv_t *kv_heads_kv = tuff_gguf_get_kv(gguf, key);
    if (kv_heads_kv && kv_heads_kv->type == GGUF_TYPE_UINT32) {
        m->head_count_kv = kv_heads_kv->val.u32;
    } else {
        m->head_count_kv = 8;
    }

    m->head_dim = m->head_count ? (m->embedding_length / m->head_count) : 64;

    snprintf(key, sizeof(key), "%sattention.layer_norm_rms_epsilon", pfx);
    m->rms_norm_eps = tuff_gguf_get_kv_f32(gguf, key, 1e-5f);

    snprintf(key, sizeof(key), "%sshortconv.l_cache", pfx);
    m->conv_kernel_size = tuff_gguf_get_kv_u32(gguf, key, 3);
    m->conv_state_len = m->conv_kernel_size > 1 ? (m->conv_kernel_size - 1) : 2;

    snprintf(key, sizeof(key), "%sfeed_forward_length", pfx);
    m->ffn_dim = tuff_gguf_get_kv_u32(gguf, key, 8192);

    snprintf(key, sizeof(key), "%srope.freq_base", pfx);
    m->rope_freq_base = tuff_gguf_get_kv_f32(gguf, key, 1e7f);

    if (is_moe) {
        snprintf(key, sizeof(key), "%sexpert_count", pfx);
        m->expert_count = tuff_gguf_get_kv_u32(gguf, key, 32);

        snprintf(key, sizeof(key), "%sexpert_used_count", pfx);
        m->expert_used_count = tuff_gguf_get_kv_u32(gguf, key, 4);

        snprintf(key, sizeof(key), "%sexpert_feed_forward_length", pfx);
        m->expert_ffn_dim = tuff_gguf_get_kv_u32(gguf, key, 1792);

        snprintf(key, sizeof(key), "%sleading_dense_block_count", pfx);
        m->leading_dense_blocks = tuff_gguf_get_kv_u32(gguf, key, 2);
    }

    // Bind trunk tensors
    bind_tensor(gguf, &m->token_embd, "token_embd.weight", true);
    m->vocab_size = (uint32_t)m->token_embd.info.dims[1];

    // Note: Liquid AI GGUF naming quirk keys the final output RMSNorm as token_embd_norm.weight
    if (!bind_tensor(gguf, &m->token_embd_norm, "token_embd_norm.weight", false)) {
        bind_tensor(gguf, &m->token_embd_norm, "output_norm.weight", true);
    }

    // LM Head output projection
    if (!bind_tensor(gguf, &m->output_weight, "output.weight", false)) {
        m->output_weight = m->token_embd; // weight tied
    }

    // Allocate layers
    m->layers = (lfm25_layer_t*)calloc(m->block_count, sizeof(lfm25_layer_t));
    if (!m->layers) {
        tuff_gguf_free(gguf);
        free(m);
        return NULL;
    }

    char tname[128];
    for (uint32_t l = 0; l < m->block_count; ++l) {
        lfm25_layer_t *L = &m->layers[l];
        L->layer_idx = l;

        // Norms
        snprintf(tname, sizeof(tname), "blk.%u.attn_norm.weight", l);
        bind_tensor(gguf, &L->attn_norm, tname, true);

        snprintf(tname, sizeof(tname), "blk.%u.ffn_norm.weight", l);
        bind_tensor(gguf, &L->ffn_norm, tname, true);

        // Operator: Check if layer has ShortConv
        snprintf(tname, sizeof(tname), "blk.%u.shortconv.in_proj.weight", l);
        if (bind_tensor(gguf, &L->shortconv_in_proj, tname, false)) {
            L->op_type = LFM25_OP_CONV;
            m->n_conv_layers++;

            snprintf(tname, sizeof(tname), "blk.%u.shortconv.conv.weight", l);
            bind_tensor(gguf, &L->shortconv_conv, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.shortconv.out_proj.weight", l);
            bind_tensor(gguf, &L->shortconv_out_proj, tname, true);
        } else {
            // Must be Attention layer
            L->op_type = LFM25_OP_ATTN;
            m->n_attn_layers++;

            snprintf(tname, sizeof(tname), "blk.%u.attn_q.weight", l);
            bind_tensor(gguf, &L->attn_q, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.attn_k.weight", l);
            bind_tensor(gguf, &L->attn_k, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.attn_v.weight", l);
            bind_tensor(gguf, &L->attn_v, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.attn_output.weight", l);
            bind_tensor(gguf, &L->attn_output, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.attn_q_norm.weight", l);
            bind_tensor(gguf, &L->attn_q_norm, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.attn_k_norm.weight", l);
            bind_tensor(gguf, &L->attn_k_norm, tname, true);
        }

        // FFN: Check if layer has MoE
        snprintf(tname, sizeof(tname), "blk.%u.ffn_gate_exps.weight", l);
        if (bind_tensor(gguf, &L->ffn_gate_exps, tname, false)) {
            L->ffn_type = LFM25_FFN_MOE;
            m->n_moe_layers++;

            snprintf(tname, sizeof(tname), "blk.%u.ffn_gate_inp.weight", l);
            bind_tensor(gguf, &L->ffn_gate_inp, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.exp_probs_b.bias", l);
            bind_tensor(gguf, &L->exp_probs_b, tname, false);

            snprintf(tname, sizeof(tname), "blk.%u.ffn_up_exps.weight", l);
            bind_tensor(gguf, &L->ffn_up_exps, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.ffn_down_exps.weight", l);
            bind_tensor(gguf, &L->ffn_down_exps, tname, true);
        } else {
            L->ffn_type = LFM25_FFN_DENSE;
            m->n_dense_ffn_layers++;

            snprintf(tname, sizeof(tname), "blk.%u.ffn_gate.weight", l);
            bind_tensor(gguf, &L->ffn_gate, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.ffn_up.weight", l);
            bind_tensor(gguf, &L->ffn_up, tname, true);

            snprintf(tname, sizeof(tname), "blk.%u.ffn_down.weight", l);
            bind_tensor(gguf, &L->ffn_down, tname, true);
        }
    }

    // Cross-check the dense FFN width against the first dense layer's ffn_gate dims (tensor wins).
    for (uint32_t l = 0; l < m->block_count; ++l) {
        const lfm25_layer_t *L = &m->layers[l];
        if (L->ffn_type != LFM25_FFN_DENSE) continue;
        uint32_t td = (uint32_t)L->ffn_gate.info.dims[1];
        if (td != m->ffn_dim) {
            fprintf(stderr, "[LFM25] WARNING: feed_forward_length=%u but blk.%u.ffn_gate has width %u; using tensor dims\n",
                    m->ffn_dim, l, td);
            m->ffn_dim = td;
        }
        break;
    }

    return m;
}

void lfm25_model_print_inspection(const lfm25_model_t *m) {
    if (!m) return;
    printf("==================================================================\n");
    printf("                  LFM2.5 MODEL INSPECTION                        \n");
    printf("==================================================================\n");
    printf("  Architecture:         %s\n", m->is_moe ? "lfm2moe (Sparse MoE)" : "lfm2 (Dense Hybrid)");
    printf("  Total Layers:         %u blocks\n", m->block_count);
    printf("  Embedding Dimension:  %u\n", m->embedding_length);
    printf("  Context Length:       %u tokens\n", m->context_length);
    printf("  Vocabulary Size:      %u tokens\n", m->vocab_size);
    printf("  Attention Heads:      %u Q, %u KV (dim %u)\n", m->head_count, m->head_count_kv, m->head_dim);
    printf("  ShortConv Window:     K = %u (state: %u past tokens/channel)\n", m->conv_kernel_size, m->conv_state_len);
    printf("  Layer Composition:    %u Conv1D layers, %u GQA Attention layers\n", m->n_conv_layers, m->n_attn_layers);
    printf("  FFN Composition:      %u Dense FFN layers, %u MoE layers\n", m->n_dense_ffn_layers, m->n_moe_layers);
    if (m->is_moe) {
        printf("  MoE Configuration:    %u experts, top-%u routing, FFN dim %u\n",
               m->expert_count, m->expert_used_count, m->expert_ffn_dim);
        printf("  Leading Dense Layers: %u\n", m->leading_dense_blocks);
    }
    printf("==================================================================\n");
}

void lfm25_model_free(lfm25_model_t *m) {
    if (!m) return;
    if (m->layers) free(m->layers);
    if (m->gguf) tuff_gguf_free(m->gguf);
    free(m);
}
