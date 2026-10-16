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

    // head_count_kv is resolved after the layers are bound (it is a per-layer INT32 array in the LFM GGUFs).
    m->head_count_kv = 0;

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
            bind_tensor(gguf, &L->exp_probs_b, tname, true);   // the router always reads the selection bias

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

    // Resolve head_count_kv. The GGUFs store it as a per-layer array (0 for ShortConv layers, n_kv for attention
    // layers); a plain scalar is accepted too. All attention layers must agree (the KV cache / kernels use one value).
    {
        snprintf(key, sizeof(key), "%sattention.head_count_kv", pfx);
        const gguf_kv_t *kvh = tuff_gguf_get_kv(gguf, key);
        uint32_t nkv = 0;
        bool have = false;
        if (kvh && (kvh->type == GGUF_TYPE_UINT32 || kvh->type == GGUF_TYPE_INT32)) {
            int64_t v = kvh->type == GGUF_TYPE_UINT32 ? (int64_t)kvh->val.u32 : (int64_t)kvh->val.i32;
            if (v > 0) { nkv = (uint32_t)v; have = true; }
        } else if (kvh && kvh->type == GGUF_TYPE_ARRAY &&
                   (kvh->val.arr.item_type == GGUF_TYPE_INT32 || kvh->val.arr.item_type == GGUF_TYPE_UINT32) && kvh->val.arr.data) {
            if (kvh->val.arr.len < m->block_count) {
                fprintf(stderr, "[LFM25] ERROR: %s has %llu entries but the model has %u layers\n",
                        key, (unsigned long long)kvh->val.arr.len, m->block_count);
                lfm25_model_free(m);
                return NULL;
            }
            const int32_t *arr = (const int32_t*)kvh->val.arr.data;
            for (uint32_t l = 0; l < m->block_count; ++l) {
                if (m->layers[l].op_type != LFM25_OP_ATTN) continue;
                int64_t v = kvh->val.arr.item_type == GGUF_TYPE_UINT32 ? (int64_t)(uint32_t)arr[l] : (int64_t)arr[l];
                if (v <= 0) {
                    fprintf(stderr, "[LFM25] ERROR: %s[%u] = %lld but layer %u is an attention layer\n", key, l, (long long)v, l);
                    lfm25_model_free(m);
                    return NULL;
                }
                if (have && (uint32_t)v != nkv) {
                    fprintf(stderr, "[LFM25] ERROR: attention layers disagree on head_count_kv (%u vs %lld at layer %u); "
                                    "per-layer KV head counts are not supported\n", nkv, (long long)v, l);
                    lfm25_model_free(m);
                    return NULL;
                }
                nkv = (uint32_t)v;
                have = true;
            }
        }
        if (!have) {
            // Not in the metadata (or no attention layers): derive from attn_k and head_dim when possible.
            for (uint32_t l = 0; l < m->block_count; ++l) {
                if (m->layers[l].op_type != LFM25_OP_ATTN || m->head_dim == 0) continue;
                nkv = (uint32_t)(m->layers[l].attn_k.info.dims[1] / m->head_dim);
                have = true;
                break;
            }
        }
        m->head_count_kv = have ? nkv : 8;
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

// -----------------------------------------------------------------------------
// Validation of everything the kernels assume (types, dims, shape constants). Run before any device allocation.
// -----------------------------------------------------------------------------
static const char* type_name(uint32_t t) {
    const ggml_type_traits_t *tr = ggml_get_type_traits((ggml_type_t)t);
    return tr ? tr->type_name : "unsupported";
}

#define VFAIL(...) do { fprintf(stderr, "[LFM25] ERROR: unsupported model: " __VA_ARGS__); fprintf(stderr, "\n"); return -1; } while (0)

// ok_a/ok_b: allowed ggml types (use -1 for "none"); dims: expected ne0..ne(n-1), 0 = unchecked.
static int check_tensor(const lfm25_tensor_t *t, const char *name, int ok_a, int ok_b,
                        int n_dims, uint64_t d0, uint64_t d1, uint64_t d2) {
    const gguf_tensor_info_t *i = &t->info;
    if ((int)i->type != ok_a && (int)i->type != ok_b) {
        fprintf(stderr, "[LFM25] ERROR: unsupported model: tensor '%s' has type %s (%u); this engine supports only %s%s%s%s.\n",
                name, type_name(i->type), i->type, ok_a >= 0 ? type_name((uint32_t)ok_a) : "", ok_b >= 0 ? " or " : "",
                ok_b >= 0 ? type_name((uint32_t)ok_b) : "", " here (Q4_K_M-style quantization)");
        return -1;
    }
    const uint64_t want[3] = {d0, d1, d2};
    bool bad = (int)i->n_dims != n_dims;
    for (int k = 0; !bad && k < n_dims; ++k) if (i->dims[k] != want[k]) bad = true;
    if (bad) {
        fprintf(stderr, "[LFM25] ERROR: unsupported model: tensor '%s' has dims [", name);
        for (uint32_t k = 0; k < i->n_dims; ++k) fprintf(stderr, "%s%llu", k ? ", " : "", (unsigned long long)i->dims[k]);
        fprintf(stderr, "], expected [");
        for (int k = 0; k < n_dims; ++k) fprintf(stderr, "%s%llu", k ? ", " : "", (unsigned long long)want[k]);
        fprintf(stderr, "]\n");
        return -1;
    }
    return 0;
}

int lfm25_model_validate(const lfm25_model_t *m) {
    if (!m || !m->layers) return -1;
    const uint64_t d = m->embedding_length;
    const int Q4 = GGML_TYPE_Q4_K, Q6 = GGML_TYPE_Q6_K, F32 = GGML_TYPE_F32;
    if (d == 0 || d % QK_K != 0) VFAIL("embedding_length %llu is not a multiple of %d", (unsigned long long)d, QK_K);
    if (m->head_count == 0 || d % m->head_count != 0) VFAIL("embedding_length %llu not divisible by head_count %u", (unsigned long long)d, m->head_count);
    if (m->head_dim != 64) VFAIL("head_dim %u (the attention kernels support only 64)", m->head_dim);
    if ((uint64_t)m->head_count * m->head_dim != d) VFAIL("head_count %u x head_dim %u != embedding_length %llu", m->head_count, m->head_dim, (unsigned long long)d);
    if (m->n_attn_layers > 0) {
        if (m->head_count_kv == 0 || m->head_count % m->head_count_kv != 0)
            VFAIL("head_count %u is not a multiple of head_count_kv %u", m->head_count, m->head_count_kv);
    }
    if (m->conv_kernel_size != 3) VFAIL("shortconv l_cache %u (the conv kernels support only 3)", m->conv_kernel_size);
    if (m->block_count == 0) VFAIL("block_count 0");
    if (m->vocab_size == 0) VFAIL("vocab_size 0");
    if (m->is_moe) {
        if (m->expert_ffn_dim == 0 || m->expert_ffn_dim % QK_K != 0) VFAIL("expert_feed_forward_length %u is not a multiple of %d", m->expert_ffn_dim, QK_K);
    }

    const uint64_t V = m->vocab_size, qd = (uint64_t)m->head_count * m->head_dim, kd = (uint64_t)m->head_count_kv * m->head_dim;
    // token_embd is read by the Q6_K embed kernel, the LM head by the Q6_K head kernel (tied or separate).
    if (check_tensor(&m->token_embd, "token_embd.weight", Q6, -1, 2, d, V, 0)) return -1;
    if (check_tensor(&m->output_weight, "output.weight", Q6, -1, 2, d, V, 0)) return -1;
    if (check_tensor(&m->token_embd_norm, "token_embd_norm.weight", F32, -1, 1, d, 0, 0)) return -1;

    char n[96];
#define CK(T, FMT, A, B, ND, D0, D1, D2) do { snprintf(n, sizeof(n), "blk.%u." FMT, l); \
        if (check_tensor(&L->T, n, A, B, ND, D0, D1, D2)) return -1; } while (0)
    for (uint32_t l = 0; l < m->block_count; ++l) {
        const lfm25_layer_t *L = &m->layers[l];
        CK(attn_norm, "attn_norm.weight", F32, -1, 1, d, 0, 0);
        CK(ffn_norm, "ffn_norm.weight", F32, -1, 1, d, 0, 0);
        if (L->op_type == LFM25_OP_CONV) {
            CK(shortconv_in_proj, "shortconv.in_proj.weight", Q4, -1, 2, d, 3 * d, 0);
            CK(shortconv_conv, "shortconv.conv.weight", F32, -1, 2, m->conv_kernel_size, d, 0);
            CK(shortconv_out_proj, "shortconv.out_proj.weight", Q4, -1, 2, d, d, 0);
        } else {
            CK(attn_q, "attn_q.weight", Q4, -1, 2, d, qd, 0);
            CK(attn_k, "attn_k.weight", Q4, -1, 2, d, kd, 0);
            CK(attn_v, "attn_v.weight", Q4, Q6, 2, d, kd, 0);
            CK(attn_output, "attn_output.weight", Q4, -1, 2, qd, d, 0);
            CK(attn_q_norm, "attn_q_norm.weight", F32, -1, 1, m->head_dim, 0, 0);
            CK(attn_k_norm, "attn_k_norm.weight", F32, -1, 1, m->head_dim, 0, 0);
        }
        if (L->ffn_type == LFM25_FFN_DENSE) {
            if (m->ffn_dim == 0 || m->ffn_dim % QK_K != 0) VFAIL("feed_forward_length %u is not a multiple of %d", m->ffn_dim, QK_K);
            CK(ffn_gate, "ffn_gate.weight", Q4, -1, 2, d, m->ffn_dim, 0);
            CK(ffn_up, "ffn_up.weight", Q4, -1, 2, d, m->ffn_dim, 0);
            CK(ffn_down, "ffn_down.weight", Q4, Q6, 2, m->ffn_dim, d, 0);
        } else {
            const uint64_t E = m->expert_count, F = m->expert_ffn_dim;
            CK(ffn_gate_inp, "ffn_gate_inp.weight", F32, -1, 2, d, E, 0);
            CK(exp_probs_b, "exp_probs_b.bias", F32, -1, 1, E, 0, 0);
            CK(ffn_gate_exps, "ffn_gate_exps.weight", Q4, -1, 3, d, F, E);
            CK(ffn_up_exps, "ffn_up_exps.weight", Q4, -1, 3, d, F, E);
            CK(ffn_down_exps, "ffn_down_exps.weight", Q4, Q6, 3, F, d, E);
        }
    }
#undef CK
    return 0;
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
