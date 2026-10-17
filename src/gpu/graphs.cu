#include "state.h"

static void cuda_graph_update_exec(cudaGraphExec_t *instance, cudaGraph_t graph) {
#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(*instance, graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(*instance, graph, &errorNode, &result_info);
#endif
    if (stat == cudaErrorGraphExecUpdateFailure) {
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(*instance));
        *instance = NULL;
        CUDA_CHECK(cudaGraphInstantiate(instance, graph, NULL, NULL, 0));
    } else {
        CUDA_CHECK(stat);
    }
}

int gpu_has_delta_graphs(void) {
    return g_layer_graphs_ready;
}

void gpu_set_device_pos(int pos) {
    if (!g_d_pos) {
        CUDA_CHECK(cudaMalloc(&g_d_pos, sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpyAsync(g_d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, 0));
}

// 🔴 This function duplicates the forward token body for all 40 layers (30 Delta-Net + 10 Attention)
// to build persistent CUDA Graphs. Attention layers utilize device-resident position pointer g_d_pos.
void gpu_capture_delta_graphs(model_t *m) {
    if (!m) return;
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int sh_ffn_dim = (int)m->expert_shared_ffn_len;

    if (!g_graph_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_graph_stream, cudaStreamNonBlocking));
    }
    if (!g_d_pos) {
        CUDA_CHECK(cudaMalloc(&g_d_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(g_d_pos, 0, sizeof(int)));
    }

    uint32_t delta_layer_idx = 0;
    uint32_t attn_layer_idx = 0;

    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];

        if (layer->type == LAYER_TYPE_ATTENTION) {
            uint32_t head_dim  = m->key_length;
            uint32_t n_head    = m->head_count;
            uint32_t n_head_kv = m->head_count_kv;
            float scale = 1.0f / sqrtf((float)head_dim);
            uint32_t q_full_dim = (uint32_t)layer->attn_q.info.dims[1];
            uint32_t k_dim      = (uint32_t)layer->attn_k.info.dims[1];
            uint32_t v_dim      = (uint32_t)layer->attn_v.info.dims[1];
            kv_cache_ensure((int)m->n_attn_layers, (int)n_head_kv, g_max_seq, (int)head_dim);
            attn_scores_ensure(n_head, g_max_seq);

            // Dry-run allocation
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            const block_q8_1_c *xq = (const block_q8_1_c *)g_d_dense_x_q8;
            const int nblk = dim / QK8_0;
            const int total_qkv = (int)q_full_dim + (int)k_dim + (int)v_dim;
            dim3 qkv_grid((unsigned)((total_qkv + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
            dim3 qkv_block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
            launch_mmvq_dense_triple((const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim, (const block_q8_0*)layer->attn_k.dev, g_d_k, (int)k_dim, (const block_q8_0*)layer->attn_v.dev, g_d_v, (int)v_dim, xq, nblk);
            launch_fused_qk_split_norm_rope(g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head, g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv, head_dim, eps, (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1], m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, g_d_pos);
            block_q8_1_c *out_q8 = (block_q8_1_c*)dense_quant_buf_ensure(dim);
            launch_attention_kernel(
                g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, NULL, g_d_attn_scores,
                n_head, n_head_kv, g_max_seq, head_dim, (int)attn_layer_idx, 0, 0, scale, out_q8, g_d_pos);
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_output.dev, out_q8, g_d_res_x, n_head * head_dim, dim, true);
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamSynchronize(0));

            // Capture Attention Graph on stream 0
            if (g_layer_graphs[l]) {
                CUDA_CHECK(cudaGraphDestroy(g_layer_graphs[l]));
                g_layer_graphs[l] = NULL;
            }
            CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            launch_mmvq_dense_triple((const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim, (const block_q8_0*)layer->attn_k.dev, g_d_k, (int)k_dim, (const block_q8_0*)layer->attn_v.dev, g_d_v, (int)v_dim, xq, nblk);
            launch_fused_qk_split_norm_rope(g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head, g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv, head_dim, eps, (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1], m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, g_d_pos);
            launch_attention_kernel(
                g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, NULL, g_d_attn_scores,
                n_head, n_head_kv, g_max_seq, head_dim, (int)attn_layer_idx, 0, 0, scale, out_q8, g_d_pos);
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_output.dev, out_q8, g_d_res_x, n_head * head_dim, dim, true);
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs[l]));

            if (g_layer_execs[l]) {
                cuda_graph_update_exec(&g_layer_execs[l], g_layer_graphs[l]);
            } else {
                CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs[l], g_layer_graphs[l], NULL, NULL, 0));
            }
            attn_layer_idx++;
        } else {
            // Delta-Net Layer
            uint32_t key_dim     = m->ssm_key_dim;
            uint32_t value_dim   = m->ssm_value_dim;
            uint32_t qkv_dim     = m->ssm_qkv_dim;
            uint32_t n_k_heads   = m->ssm_n_k_heads;
            uint32_t n_v_heads   = m->ssm_n_v_heads;
            uint32_t head_k_dim  = m->ssm_head_k_dim;
            uint32_t head_v_dim  = m->ssm_head_v_dim;
            uint32_t conv_kernel = m->ssm_conv_kernel;
            float scale_q = 1.0f / sqrtf((float)head_k_dim);

            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dim, eps);
            }
            gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate, &layer->ssm_beta, &layer->ssm_alpha, g_d_norm_x, dim);
            gpu_delta_fused_gdn_dev((int)delta_layer_idx, layer, g_d_res_x, eps, scale_q, (int)n_k_heads, (int)n_v_heads, (int)head_k_dim, (int)head_v_dim, (int)key_dim, (int)value_dim, (int)qkv_dim, (int)conv_kernel, dim);
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, dim, eps);
            }
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamSynchronize(0));

            // Capture graph on stream 0
            if (g_layer_graphs[l]) {
                CUDA_CHECK(cudaGraphDestroy(g_layer_graphs[l]));
                g_layer_graphs[l] = NULL;
            }

            CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dim, eps);
            }
            gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate, &layer->ssm_beta, &layer->ssm_alpha, g_d_norm_x, dim);
            gpu_delta_fused_gdn_dev((int)delta_layer_idx, layer, g_d_res_x, eps, scale_q, (int)n_k_heads, (int)n_v_heads, (int)head_k_dim, (int)head_v_dim, (int)key_dim, (int)value_dim, (int)qkv_dim, (int)conv_kernel, dim);
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, dim, eps);
            }
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs[l]));

            if (g_layer_execs[l]) {
                cuda_graph_update_exec(&g_layer_execs[l], g_layer_graphs[l]);
            } else {
                CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs[l], g_layer_graphs[l], NULL, NULL, 0));
            }

            delta_layer_idx++;
        }
    }
    g_layer_graphs_ready = 1;
    gpu_reset_delta_state();
    gpu_reset_kv();
}



void gpu_launch_delta_layer_graph(uint32_t delta_layer_idx) {
    if (g_layer_graphs_ready && delta_layer_idx < 64 && g_layer_execs[delta_layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs[delta_layer_idx], 0));
        g_graph_launches++;
    }
}

void gpu_launch_layer_graph(uint32_t layer_idx) {
    if (g_layer_graphs_ready && layer_idx < 64 && g_layer_execs[layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs[layer_idx], 0));
        g_graph_launches++;
    }
}

unsigned long long gpu_graph_launch_count(void) { return g_graph_launches; }

// ---------------------------------------------------------------------------
// H7: the same capture for the two-column verify trunk.
//
// The verify path issues roughly 50 launches per delta layer (two rms_norms, the
// batched projections, two column publishes, two GDN runs, the state snapshot,
// two rms_norms, two router computes and two shared experts) against the single
// graph launch the one-token path costs. Over 30 delta layers that is ~1500
// launches per verify step; iteration 40 priced ~1000 launches at ~4.5 ms of host
// time, which is the same order as the gap to break-even. Collapsing them is
// ordinary plumbing, and it is the last item between H7 and a positive result.
// ---------------------------------------------------------------------------




extern "C" int gpu_has_delta_graphs_n2(void) { return g_layer_graphs2_ready; }

// One delta layer of the verify trunk, both columns. Used for the dry run (to force
// every lazy allocation) and then captured verbatim.
static void n2_delta_layer_body(model_t *m, model_layer_t *layer, uint32_t delta_layer_idx) {
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int sh_ffn_dim = (int)m->expert_shared_ffn_len;
    const float scale_q = 1.0f / sqrtf((float)m->ssm_head_k_dim);
    float *d_res2  = g_d_res_x2;
    float *d_norm2 = g_d_norm_x2;
    const float *shexp[2] = { g_d_shexp_out, gpu_get_shexp2_dev_ptr() };

    gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->attn_norm.dev, dim, eps);
    gpu_delta_projections_n2(&layer->attn_qkv, &layer->attn_gate,
                             &layer->ssm_beta, &layer->ssm_alpha, d_norm2, dim);
    for (int c = 0; c < 2; ++c) {
        // H17: no column publish — the fused GDN points straight at the batched projections.
        gpu_delta_fused_gdn_dev_col(c, (int)delta_layer_idx, layer, d_res2 + (size_t)c * dim, eps, scale_q,
                                    (int)m->ssm_n_k_heads, (int)m->ssm_n_v_heads,
                                    (int)m->ssm_head_k_dim, (int)m->ssm_head_v_dim,
                                    (int)m->ssm_key_dim, (int)m->ssm_value_dim,
                                    (int)m->ssm_qkv_dim, (int)m->ssm_conv_kernel, dim);
        if (c == 0) {
            gpu_delta_snapshot_layer_n2((int)delta_layer_idx, (int)m->ssm_n_v_heads,
                                        (int)m->ssm_head_v_dim, (int)m->ssm_head_k_dim,
                                        (int)m->ssm_conv_kernel, (int)m->ssm_qkv_dim);
        }
    }
    gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->post_attention_norm.dev, dim, eps);
    for (int c = 0; c < 2; ++c) {
        gpu_router_logits_dev_only_col(&layer->ffn_gate_inp, d_norm2 + (size_t)c * dim,
                                       dim, (int)m->expert_count, c);
    }
    // H15: one weight read for both columns; must match verify_moe_stage_n2's
    // live path exactly or the captured graph and the live path diverge.
    gpu_shared_expert_dev_n2((float*)shexp[0], (float*)shexp[1], layer, d_norm2, dim, sh_ffn_dim);
}

extern "C" void gpu_capture_delta_graphs_n2(model_t *m) {
    if (!m || g_layer_graphs2_ready) return;
    gpu_n2_prepare((int)m->embedding_length, (int)m->vocab_size);

    uint32_t delta_layer_idx = 0;
    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];
        if (layer->type == LAYER_TYPE_ATTENTION) continue;

        // Dry run: every lazy cudaMalloc on this path must happen before capture.
        n2_delta_layer_body(m, layer, delta_layer_idx);
        CUDA_CHECK(cudaStreamSynchronize(0));

        if (g_layer_graphs2[delta_layer_idx]) {
            CUDA_CHECK(cudaGraphDestroy(g_layer_graphs2[delta_layer_idx]));
            g_layer_graphs2[delta_layer_idx] = NULL;
        }
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
        n2_delta_layer_body(m, layer, delta_layer_idx);
        CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs2[delta_layer_idx]));

        if (g_layer_execs2[delta_layer_idx]) {
            cuda_graph_update_exec(&g_layer_execs2[delta_layer_idx], g_layer_graphs2[delta_layer_idx]);
        } else {
            CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs2[delta_layer_idx], g_layer_graphs2[delta_layer_idx], NULL, NULL, 0));
        }
        delta_layer_idx++;
    }
    g_layer_graphs2_ready = 1;
    // The dry runs advanced the recurrent state; put it back.
    gpu_reset_delta_state();
}

extern "C" void gpu_launch_delta_layer_graph_n2(uint32_t delta_layer_idx) {
    if (g_layer_graphs2_ready && delta_layer_idx < 64 && g_layer_execs2[delta_layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs2[delta_layer_idx], 0));
        g_graph_launches++;
    }
}


// ---------------------------------------------------------------------------
// Lossless verify (2026-10): column 1 of a two-token verify pass replays the single-token layer
// graphs (bit-identical arithmetic by construction), except that its delta layers must not write
// the recurrent/conv state - a rejected draft then needs no rollback at all. These are those
// delta-layer graphs (no-write GDN), plus one graph that replays the state updates of an
// accepted column 1 with the state-writing kernels.
// ---------------------------------------------------------------------------
extern "C" void gpu_verify_set_positions(int pos0, int n);
extern "C" float *gpu_get_d_res_verify(int dim);
extern "C" void gpu_delta_commit_nw_slot(int slot, int layer_idx, const model_layer_t *L, float eps, float scale_q,
                                         int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                         int key_dim, int value_dim, int qkv_dim, int conv_kernel);
static int *g_d_vpos;
static cudaGraph_t     g_layer_graphs_nw[64];
static cudaGraphExec_t g_layer_execs_nw[64];
static cudaGraph_t     g_commit_graph[2] = {NULL, NULL};
static cudaGraphExec_t g_commit_exec[2]  = {NULL, NULL};
static int g_verify_graphs_ready = 0;

static void nw_delta_layer_body(model_t *m, model_layer_t *layer, uint32_t delta_layer_idx) {
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const float scale_q = 1.0f / sqrtf((float)m->ssm_head_k_dim);
    if (!tuff_exact() && mmvq_dual_enabled()) {
        gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
    } else {
        gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dim, eps);
    }
    gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate, &layer->ssm_beta, &layer->ssm_alpha, g_d_norm_x, dim);
    gpu_delta_fused_gdn_dev_nw((int)delta_layer_idx, layer, g_d_res_x, eps, scale_q,
                               (int)m->ssm_n_k_heads, (int)m->ssm_n_v_heads,
                               (int)m->ssm_head_k_dim, (int)m->ssm_head_v_dim,
                               (int)m->ssm_key_dim, (int)m->ssm_value_dim,
                               (int)m->ssm_qkv_dim, (int)m->ssm_conv_kernel, dim);
    if (!tuff_exact() && mmvq_dual_enabled()) {
        gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
    } else {
        gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, dim, eps);
    }
    gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
}

static void commit_body(model_t *m, int slot) {
    const float eps = m->rms_norm_eps;
    const float scale_q = 1.0f / sqrtf((float)m->ssm_head_k_dim);
    uint32_t di = 0;
    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];
        if (layer->type == LAYER_TYPE_ATTENTION) continue;
        gpu_delta_commit_nw_slot(slot, (int)di, layer, eps, scale_q,
                                 (int)m->ssm_n_k_heads, (int)m->ssm_n_v_heads,
                                 (int)m->ssm_head_k_dim, (int)m->ssm_head_v_dim,
                                 (int)m->ssm_key_dim, (int)m->ssm_value_dim,
                                 (int)m->ssm_qkv_dim, (int)m->ssm_conv_kernel);
        di++;
    }
}

extern "C" void gpu_capture_verify_graphs(model_t *m) {
    if (!m || g_verify_graphs_ready || !g_layer_graphs_ready) return;
    const char *ls = getenv("TUFF_VERIFY_LOCKSTEP");
    const int need_nw_graphs = (ls && ls[0] == '0');
    uint32_t di = 0;
    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];
        if (layer->type == LAYER_TYPE_ATTENTION) continue;
        if (di == 0 || need_nw_graphs) {
            nw_delta_layer_body(m, layer, di);
            CUDA_CHECK(cudaStreamSynchronize(0));
        }
        if (need_nw_graphs) {
            CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
            nw_delta_layer_body(m, layer, di);
            CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs_nw[l]));
            CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs_nw[l], g_layer_graphs_nw[l], NULL, NULL, 0));
        }
        di++;
    }
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
        commit_body(m, slot);
        CUDA_CHECK(cudaStreamEndCapture(0, &g_commit_graph[slot]));
        CUDA_CHECK(cudaGraphInstantiate(&g_commit_exec[slot], g_commit_graph[slot], NULL, NULL, 0));
    }
    g_verify_graphs_ready = 1;
    gpu_reset_delta_state();
    gpu_reset_kv();
}

extern "C" int gpu_has_verify_graphs(void) { return g_verify_graphs_ready; }

// Layer l for verify column 1: the no-write delta graph, or the ordinary graph for attention.
extern "C" void gpu_launch_layer_graph_nw(uint32_t layer_idx) {
    if (g_verify_graphs_ready && layer_idx < 64 && g_layer_execs_nw[layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs_nw[layer_idx], 0));
        g_graph_launches++;
    } else {
        gpu_launch_layer_graph(layer_idx);
    }
}

extern "C" void gpu_launch_verify_commit_slot(int slot) {
    if (slot >= 0 && slot < 2 && g_commit_exec[slot]) CUDA_CHECK(cudaGraphLaunch(g_commit_exec[slot], 0));
}

extern "C" void gpu_launch_verify_commit(void) {
    gpu_launch_verify_commit_slot(0);
}

// Lossless verify: per-column positions live on the device, so selecting column c's position for
// the attention graphs is a D2D copy (no pageable H2D per layer).
static int *g_h_vpos = NULL;
extern "C" void gpu_verify_set_positions(int pos0, int n) {
    if (!g_d_vpos) CUDA_CHECK(cudaMalloc(&g_d_vpos, 8 * sizeof(int)));
    if (!g_h_vpos) CUDA_CHECK(cudaMallocHost(&g_h_vpos, 8 * sizeof(int)));
    if (!g_d_pos) { CUDA_CHECK(cudaMalloc(&g_d_pos, sizeof(int))); }
    // The previous pass's copies out of g_h_vpos are complete: its argmax synchronized.
    for (int c = 0; c < n && c < 8; ++c) g_h_vpos[c] = pos0 + c;
    CUDA_CHECK(cudaMemcpyAsync(g_d_vpos, g_h_vpos, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, 0));
}
extern "C" void gpu_verify_select_pos(int c) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_pos, g_d_vpos + c, sizeof(int), cudaMemcpyDeviceToDevice, 0));
}
extern "C" void gpu_memcpy_d2d(void *dst, const void *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0));
}
static float *g_d_res_v = NULL;
extern "C" float *gpu_get_d_res_verify(int dim) {
    if (!g_d_res_v) CUDA_CHECK(cudaMalloc(&g_d_res_v, (size_t)8 * dim * sizeof(float)));
    return g_d_res_v;
}
extern "C" void gpu_memcpy_h2d(void *dst, const void *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, 0));
}
extern "C" void *gpu_event_new(void) {
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    return (void*)ev;
}
extern "C" void gpu_event_record(void *ev) { CUDA_CHECK(cudaEventRecord((cudaEvent_t)ev, 0)); }
extern "C" void gpu_event_sync(void *ev)   { CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)ev)); }

// ---------------------------------------------------------------------------
// Verify column graphs: one graph per (column, layer) holding everything v2_trunk issues -
// residual in, position select, the layer body (the single-token graph's kernels; column 1's delta
// layers in the no-write variant), router logits + norm_x D2H, shared expert, the routed-expert
// q8 save, residual out. One launch instead of ~13 API calls per column-layer. The kernels and
// their arguments are the single-token path's, so the arithmetic is unchanged.
// ---------------------------------------------------------------------------
static cudaGraphExec_t g_vcol_exec[2][64];
static int g_vcol_ready = 0;
static float *g_h_vrl[2] = {NULL, NULL};
static float *g_h_vnx[2] = {NULL, NULL};

static void vcol_layer_body(model_t *m, uint32_t l, uint32_t attn_idx, uint32_t delta_idx, int c,
                            float *res_c, float *shexp_c) {
    model_layer_t *layer = &m->layers[l];
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int sh_ffn_dim = (int)m->expert_shared_ffn_len;
    CUDA_CHECK(cudaMemcpyAsync(g_d_res_x, res_c, (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(g_d_pos, g_d_vpos + c, sizeof(int), cudaMemcpyDeviceToDevice, 0));
    if (layer->type == LAYER_TYPE_ATTENTION) {
        uint32_t head_dim  = m->key_length;
        uint32_t n_head    = m->head_count;
        uint32_t n_head_kv = m->head_count_kv;
        float scale = 1.0f / sqrtf((float)head_dim);
        uint32_t q_full_dim = (uint32_t)layer->attn_q.info.dims[1];
        uint32_t k_dim      = (uint32_t)layer->attn_k.info.dims[1];
        uint32_t v_dim      = (uint32_t)layer->attn_v.info.dims[1];
        const block_q8_1_c *xq = (const block_q8_1_c *)g_d_dense_x_q8;
        const int nblk = dim / QK8_0;
        gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
        launch_mmvq_dense_triple((const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim, (const block_q8_0*)layer->attn_k.dev, g_d_k, (int)k_dim, (const block_q8_0*)layer->attn_v.dev, g_d_v, (int)v_dim, xq, nblk);
        launch_fused_qk_split_norm_rope(g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head, g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv, head_dim, eps, (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1], m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, g_d_pos);
        block_q8_1_c *out_q8 = (block_q8_1_c*)dense_quant_buf_ensure(dim);
        launch_attention_kernel(
            g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, NULL, g_d_attn_scores,
            n_head, n_head_kv, g_max_seq, head_dim, (int)attn_idx, 0, 0, scale, out_q8, g_d_pos);
        mmvq_q8_0_rows((const block_q8_0*)layer->attn_output.dev, out_q8, g_d_res_x, n_head * head_dim, dim, true);
        gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
        gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
    } else if (c == 0) {
        const float scale_q = 1.0f / sqrtf((float)m->ssm_head_k_dim);
        if (!tuff_exact() && mmvq_dual_enabled()) {
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
        } else {
            gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dim, eps);
        }
        gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate, &layer->ssm_beta, &layer->ssm_alpha, g_d_norm_x, dim);
        gpu_delta_fused_gdn_dev((int)delta_idx, layer, g_d_res_x, eps, scale_q, (int)m->ssm_n_k_heads, (int)m->ssm_n_v_heads, (int)m->ssm_head_k_dim, (int)m->ssm_head_v_dim, (int)m->ssm_key_dim, (int)m->ssm_value_dim, (int)m->ssm_qkv_dim, (int)m->ssm_conv_kernel, dim);
        if (!tuff_exact() && mmvq_dual_enabled()) {
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
        } else {
            gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, dim, eps);
        }
        gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
    } else {
        nw_delta_layer_body(m, layer, delta_idx);
    }
    // The graphed single-token path: router D2H, then the shared expert, then (host) MoE.
    CUDA_CHECK(cudaMemcpyAsync(g_h_vrl[c], g_d_router_logits, (size_t)m->expert_count * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(g_h_vnx[c], g_d_norm_x, (size_t)dim * sizeof(float), cudaMemcpyDeviceToHost, 0));
    gpu_shared_expert_dev(shexp_c, layer, g_d_norm_x, dim, sh_ffn_dim);
    gpu_expert_save_norm_q8_col(dim, c);
    CUDA_CHECK(cudaMemcpyAsync(res_c, g_d_res_x, (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice, 0));
}

extern "C" void gpu_capture_verify_col_graphs(model_t *m) {
    if (!m || g_vcol_ready || !g_verify_graphs_ready) return;
    const int dim = (int)m->embedding_length;
    gpu_n2_prepare(dim, (int)m->vocab_size);
    float *res = gpu_get_d_res_verify(dim);
    float *shexp[2] = { g_d_shexp_out, (float*)gpu_get_shexp2_dev_ptr() };
    gpu_verify_set_positions(0, 2);
    for (int c = 0; c < 2; ++c) {
        if (!g_h_vrl[c]) CUDA_CHECK(cudaMallocHost(&g_h_vrl[c], 256 * sizeof(float)));
        if (!g_h_vnx[c]) CUDA_CHECK(cudaMallocHost(&g_h_vnx[c], (size_t)dim * sizeof(float)));
    }
    const int dual = !tuff_exact() && mmvq_dual_enabled();
    uint32_t ai = 0, di = 0;
    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];
        if (l >= (uint32_t)g_n_gpu_expert_layers) {
            // Lazy allocations of the shared expert happen on its first (uncaptured) call.
            gpu_set_norm_q8_valid(dual);
            gpu_shared_expert_dev(shexp[0], layer, g_d_norm_x, dim, (int)m->expert_shared_ffn_len);
            CUDA_CHECK(cudaStreamSynchronize(0));
            for (int c = 0; c < 2; ++c) {
                cudaGraph_t g;
                gpu_set_norm_q8_valid(dual);
                CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
                vcol_layer_body(m, l, ai, di, c, res + (size_t)c * dim, shexp[c]);
                CUDA_CHECK(cudaStreamEndCapture(0, &g));
                CUDA_CHECK(cudaGraphInstantiate(&g_vcol_exec[c][l], g, NULL, NULL, 0));
                CUDA_CHECK(cudaGraphDestroy(g));
            }
        }
        if (layer->type == LAYER_TYPE_ATTENTION) ai++; else di++;
    }
    gpu_set_norm_q8_valid(0);
    g_vcol_ready = 1;
}

extern "C" int gpu_has_verify_col_graphs(void) { return g_vcol_ready; }
extern "C" void gpu_launch_verify_col(int c, uint32_t l) {
    CUDA_CHECK(cudaGraphLaunch(g_vcol_exec[c][l], 0));
    g_graph_launches++;
}
extern "C" float *gpu_verify_host_rl(int c) { return g_h_vrl[c]; }
extern "C" float *gpu_verify_host_nx(int c) { return g_h_vnx[c]; }
extern "C" int *gpu_verify_d_vpos(void) { return g_d_vpos; }
