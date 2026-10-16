#include "lfm25_engine.h"
#include "lfm25_model.h"
#include "lfm25_cpu_expert.h"
#include "tuff_vtable.h"
#include "tuff_exact.h"
#include "cuda_check.h"
#include "ggml_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef LFM25_CPU_MISS_DEFAULT
#define LFM25_CPU_MISS_DEFAULT 0
#endif
#ifndef LFM25_CPU_MISS_ADMIT_DEFAULT
#define LFM25_CPU_MISS_ADMIT_DEFAULT 4
#endif

static void lfm25_bench_batch(lfm25_engine_t *eng);

lfm25_engine_t* lfm25_engine_init(lfm25_model_t *model, int max_context_len) {
    if (!model) return NULL;
    if (lfm25_model_validate(model) != 0) return NULL;   // before any allocation
    // Router / expert kernels support 32 or 64 experts, top-4. Models with >32 experts (LFM2.5-24B-A2B) use the
    // VRAM expert cache (src/lfm25_expert_cache.c) because their experts do not fit in VRAM.
    if (model->is_moe && ((model->expert_count != 32 && model->expert_count != 64) || model->expert_used_count != 4)) {
        fprintf(stderr, "[LFM25] ERROR: MoE with %u experts / top-%u is not supported (only 32 or 64 / top-4)\n",
                model->expert_count, model->expert_used_count);
        return NULL;
    }
    lfm25_engine_t *eng = (lfm25_engine_t*)calloc(1, sizeof(lfm25_engine_t));
    if (!eng) return NULL;

    eng->model = model;
    eng->max_context_len = max_context_len > 0 ? max_context_len : 4096;

    uint32_t d = model->embedding_length;
    uint32_t vocab = model->vocab_size;
    uint32_t ffn_dim = model->ffn_dim;
    uint32_t n_head = model->head_count;
    uint32_t n_head_kv = model->head_count_kv;
    uint32_t head_dim = model->head_dim;

    // Allocate Conv1D rolling state buffer: [n_conv_layers * conv_state_len * d]
    size_t conv_state_bytes = (size_t)model->n_conv_layers * model->conv_state_len * d * sizeof(float);
    if (conv_state_bytes > 0) {
        CUDA_CHECK(cudaMalloc((void**)&eng->d_conv_state, conv_state_bytes));
        CUDA_CHECK(cudaMemset(eng->d_conv_state, 0, conv_state_bytes));
    }

    // Allocate KV Cache for Attention layers: [n_attn_layers * head_count_kv * max_context_len * head_dim]
    size_t kv_cache_bytes = (size_t)model->n_attn_layers * n_head_kv * eng->max_context_len * head_dim * sizeof(float);
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

    CUDA_CHECK(cudaMalloc((void**)&eng->d_q, (size_t)n_head * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_k, (size_t)n_head_kv * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_v, (size_t)n_head_kv * head_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_gate, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_up, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_act, (size_t)ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_ffn_down, (size_t)d * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&eng->d_logits, (size_t)vocab * sizeof(float)));

    // Quantized activation scratch buffers: block_q8_1 is 36 bytes per 32 elements
    size_t q8_x_bytes = (size_t)(d / 32) * 36;
    size_t q8_ffn_bytes = (size_t)(ffn_dim / 32) * 36;
    CUDA_CHECK(cudaMalloc(&eng->d_q8_x, q8_x_bytes));
    CUDA_CHECK(cudaMalloc(&eng->d_q8_op_out, q8_x_bytes));
    CUDA_CHECK(cudaMalloc(&eng->d_q8_ffn_act, q8_ffn_bytes));

    if (model->is_moe) {
        size_t moe_n = (size_t)model->expert_used_count * model->expert_ffn_dim;
        CUDA_CHECK(cudaMalloc((void**)&eng->d_topk_ids, 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_topk_w, 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_moe_act, moe_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng->d_moe_act_q8, (moe_n / 32) * 36));
    }

    // Device argmax buffers
    size_t grid_blocks = (vocab + 4 - 1) / 4;
    CUDA_CHECK(cudaMalloc((void**)&eng->d_block_max, grid_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_block_idx, grid_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_best_token, sizeof(int)));

    // Host buffers
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_logits, (size_t)vocab * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_tokpos, 2 * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&eng->h_best, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&eng->d_tokpos, 2 * sizeof(int)));

    // Batched-forward scratch (Task 5.1). Allocated before the expert cache so the cache's default size accounts for it.
    {
        const size_t B = LFM25_MAX_BATCH;
        const size_t qdim = (size_t)n_head * head_dim, kvdim = (size_t)n_head_kv * head_dim;
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bx, B * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bnorm, B * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bin_proj, B * 3 * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bq, B * qdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bk, B * kvdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bv, B * kvdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bop_out, B * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bffn_act, B * ffn_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng->d_bq8_x, B * q8_x_bytes));
        CUDA_CHECK(cudaMalloc(&eng->d_bq8_op, B * q8_x_bytes));
        CUDA_CHECK(cudaMalloc(&eng->d_bq8_ffn, B * q8_ffn_bytes));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_blogits, B * vocab * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bblock_max, B * grid_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bblock_idx, B * grid_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_bbest, B * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_btokpos, (B + 1) * 2 * sizeof(int)));
        CUDA_CHECK(cudaMemset(eng->d_btokpos, 0, (B + 1) * 2 * sizeof(int)));
        CUDA_CHECK(cudaMallocHost((void**)&eng->h_btokpos, (B + 1) * 2 * sizeof(int)));
        CUDA_CHECK(cudaMallocHost((void**)&eng->h_bbest, B * sizeof(int)));
        size_t snap_bytes = (size_t)model->n_conv_layers * B * 2 * d * sizeof(float);
        if (snap_bytes > 0) {
            CUDA_CHECK(cudaMalloc((void**)&eng->d_conv_snap, snap_bytes));
            CUDA_CHECK(cudaMemset(eng->d_conv_snap, 0, snap_bytes));
        }
    }

    // CUDA graph: only when there is no host round-trip inside the forward pass (no expert cache).
    {
        const char *g = getenv("LFM25_CUDA_GRAPH");
        int want = !(g && g[0] == '0');
        eng->graph_enabled = want && !lfm25_expert_cache_wanted(model);
        // One explicit engine-owned stream for every forward-path launch/copy (graphed or not): no code relies on the
        // implicit legacy <-> per-thread default-stream barriers, so the build flag --default-stream does not matter.
        CUDA_CHECK(cudaStreamCreateWithFlags(&eng->stream, cudaStreamNonBlocking));
    }

    // Expert cache: created last so its default size can take all VRAM left after trunk + KV + scratch.
    if (lfm25_expert_cache_wanted(model)) {
        CUDA_CHECK(cudaMalloc((void**)&eng->d_slot_ids, 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_pred_ids, 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&eng->d_pred_w, 4 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost((void**)&eng->h_moe_ids, 8 * sizeof(int)));
        CUDA_CHECK(cudaMallocHost((void**)&eng->h_slot_ids, 4 * sizeof(int)));
        const char *pf = getenv("LFM25_PREFETCH");
        eng->prefetch = pf && pf[0] == '1';   // default OFF: measured net loss on PCIe gen3 x8 (see devlog / Task 4.3)
        const char *pn = getenv("LFM25_PREFETCH_TOPN");
        eng->prefetch_n = pn && *pn ? atoi(pn) : 4;
        if (eng->prefetch_n < 1) eng->prefetch_n = 1;
        if (eng->prefetch_n > 4) eng->prefetch_n = 4;
        // CPU evaluation of cache misses (hybrid path). Allocated before the cache: the cache takes all remaining VRAM.
        {
            const char *cm = getenv("LFM25_CPU_MISS");
            int on = cm && *cm ? cm[0] == '1' : LFM25_CPU_MISS_DEFAULT;
            const char *ad = getenv("LFM25_CPU_MISS_ADMIT");
            eng->cpu_admit = ad && *ad ? atoi(ad) : LFM25_CPU_MISS_ADMIT_DEFAULT;
            if (eng->cpu_admit < 0) eng->cpu_admit = 0;
            if (on) {
                eng->cpu_expert = lfm25_cpu_expert_create(0, (int)model->embedding_length, (int)model->expert_ffn_dim);
                if (!eng->cpu_expert) { lfm25_engine_free(eng); return NULL; }
                fprintf(stderr, "[LFM25 CACHE] cpu-miss: ON (%d threads, admission budget %d in-flight upload batches)\n",
                        lfm25_cpu_expert_threads(eng->cpu_expert), eng->cpu_admit);
            } else {
                fprintf(stderr, "[LFM25 CACHE] cpu-miss: OFF\n");
            }
        }
        eng->expert_cache = lfm25_expert_cache_create(model);
        if (!eng->expert_cache) { lfm25_engine_free(eng); return NULL; }
    }

    // Everything above (memsets, async host copies) must be complete before the non-blocking engine stream starts.
    CUDA_CHECK(cudaDeviceSynchronize());
    { const char *bb = getenv("LFM25_BENCH_BATCH"); if (bb && bb[0] == '1') lfm25_bench_batch(eng); }
    return eng;
}

void lfm25_engine_free(lfm25_engine_t *eng) {
    if (!eng) return;
    if (eng->stream) { cudaStreamSynchronize(eng->stream); lfm25_gpu_set_stream(0); }

    lfm25_expert_cache_free(eng->expert_cache);
    lfm25_cpu_expert_free(eng->cpu_expert);
    if (eng->d_slot_ids)   cudaFree(eng->d_slot_ids);
    if (eng->d_pred_ids)   cudaFree(eng->d_pred_ids);
    if (eng->d_pred_w)     cudaFree(eng->d_pred_w);
    if (eng->h_moe_ids)    cudaFreeHost(eng->h_moe_ids);
    if (eng->h_slot_ids)   cudaFreeHost(eng->h_slot_ids);
    if (eng->d_conv_state) cudaFree(eng->d_conv_state);
    if (eng->d_k_cache)    cudaFree(eng->d_k_cache);
    if (eng->d_v_cache)    cudaFree(eng->d_v_cache);

    if (eng->d_x)          cudaFree(eng->d_x);
    if (eng->d_norm_x)     cudaFree(eng->d_norm_x);
    if (eng->d_in_proj)    cudaFree(eng->d_in_proj);
    if (eng->d_conv_out)   cudaFree(eng->d_conv_out);
    if (eng->d_op_out)     cudaFree(eng->d_op_out);
    if (eng->d_q)          cudaFree(eng->d_q);
    if (eng->d_k)          cudaFree(eng->d_k);
    if (eng->d_v)          cudaFree(eng->d_v);
    if (eng->d_ffn_gate)   cudaFree(eng->d_ffn_gate);
    if (eng->d_ffn_up)     cudaFree(eng->d_ffn_up);
    if (eng->d_ffn_act)    cudaFree(eng->d_ffn_act);
    if (eng->d_ffn_down)   cudaFree(eng->d_ffn_down);
    if (eng->d_logits)     cudaFree(eng->d_logits);

    if (eng->d_q8_x)       cudaFree(eng->d_q8_x);
    if (eng->d_q8_op_out)  cudaFree(eng->d_q8_op_out);
    if (eng->d_q8_ffn_act) cudaFree(eng->d_q8_ffn_act);

    if (eng->d_topk_ids)   cudaFree(eng->d_topk_ids);
    if (eng->d_topk_w)     cudaFree(eng->d_topk_w);
    if (eng->d_moe_act)    cudaFree(eng->d_moe_act);
    if (eng->d_moe_act_q8) cudaFree(eng->d_moe_act_q8);

    if (eng->d_block_max)  cudaFree(eng->d_block_max);
    if (eng->d_block_idx)  cudaFree(eng->d_block_idx);
    if (eng->d_best_token) cudaFree(eng->d_best_token);

    if (eng->h_logits)     cudaFreeHost(eng->h_logits);
    if (eng->h_tokpos)     cudaFreeHost(eng->h_tokpos);
    if (eng->h_best)       cudaFreeHost(eng->h_best);
    if (eng->d_tokpos)     cudaFree(eng->d_tokpos);
    if (eng->d_bx)         cudaFree(eng->d_bx);
    if (eng->d_bnorm)      cudaFree(eng->d_bnorm);
    if (eng->d_bin_proj)   cudaFree(eng->d_bin_proj);
    if (eng->d_bq)         cudaFree(eng->d_bq);
    if (eng->d_bk)         cudaFree(eng->d_bk);
    if (eng->d_bv)         cudaFree(eng->d_bv);
    if (eng->d_bop_out)    cudaFree(eng->d_bop_out);
    if (eng->d_bffn_act)   cudaFree(eng->d_bffn_act);
    if (eng->d_bq8_x)      cudaFree(eng->d_bq8_x);
    if (eng->d_bq8_op)     cudaFree(eng->d_bq8_op);
    if (eng->d_bq8_ffn)    cudaFree(eng->d_bq8_ffn);
    if (eng->d_blogits)    cudaFree(eng->d_blogits);
    if (eng->d_bblock_max) cudaFree(eng->d_bblock_max);
    if (eng->d_bblock_idx) cudaFree(eng->d_bblock_idx);
    if (eng->d_bbest)      cudaFree(eng->d_bbest);
    if (eng->d_btokpos)    cudaFree(eng->d_btokpos);
    if (eng->d_conv_snap)  cudaFree(eng->d_conv_snap);
    if (eng->h_btokpos)    cudaFreeHost(eng->h_btokpos);
    if (eng->h_bbest)      cudaFreeHost(eng->h_bbest);
    if (eng->graph_exec)   cudaGraphExecDestroy(eng->graph_exec);
    if (eng->graph)        cudaGraphDestroy(eng->graph);
    if (eng->stream)       cudaStreamDestroy(eng->stream);

    free(eng);
}

void lfm25_engine_set_sampler(lfm25_engine_t *eng, const sampler_params_t *params) {
    if (eng && params) {
        eng->sampler = *params;
    }
}

static inline void lfm25_gpu_mat_vec(const lfm25_tensor_t *t, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum) {
    if (t->info.type == GGML_TYPE_Q6_K) {
        lfm25_gpu_mmvq_q6_K(t->dev, d_xq8, d_dst, in_dim, out_dim, accum);
    } else {
        lfm25_gpu_mmvq_q4_K(t->dev, d_xq8, d_dst, in_dim, out_dim, accum);
    }
}

// Sparse MoE FFN for one token: sigmoid top-4 router on the normed FP32 input d_norm_x, then the expert FFN on d_q8_x
// accumulated into the residual d_x. Used by the single-token step and, per token, by the batched forward.
static void lfm25_enqueue_moe_ffn(lfm25_engine_t *eng, cudaStream_t stream, uint32_t l,
                                  const float *d_norm_x, const void *d_q8_x, float *d_x) {
    lfm25_model_t *m = eng->model;
    lfm25_layer_t *L = &m->layers[l];
    int dim = (int)m->embedding_length;
    // Sparse MoE: sigmoid top-4 router on the normed FP32 input, then in-VRAM expert FFN accumulated into d_x
    lfm25_gpu_router_top4(d_norm_x, (const float*)L->ffn_gate_inp.dev, (const float*)L->exp_probs_b.dev,
                          eng->d_topk_ids, eng->d_topk_w, NULL, dim, (int)m->expert_count);
    const size_t gs0 = (size_t)m->expert_ffn_dim * ggml_row_size(L->ffn_gate_exps.info.type, dim);
    const size_t us0 = (size_t)m->expert_ffn_dim * ggml_row_size(L->ffn_up_exps.info.type, dim);
    const size_t ds0 = (size_t)dim * ggml_row_size(L->ffn_down_exps.info.type, m->expert_ffn_dim);
    if (eng->expert_cache) {
        // Cache path. Optionally also run the NEXT MoE layer's router on this layer's normed activations to
        // predict its experts (selection only), fetch both id sets with one sync, then resolve slots: misses are
        // uploaded on the cache's copy stream and the FFN stream only waits for the slots it reads.
        lfm25_expert_cache_t *ec = eng->expert_cache;
        int pred_layer = -1;
        if (eng->prefetch && l + 1 < m->block_count && m->layers[l + 1].ffn_type == LFM25_FFN_MOE) {
            const lfm25_layer_t *N = &m->layers[l + 1];
            lfm25_gpu_router_top4(d_norm_x, (const float*)N->ffn_gate_inp.dev, (const float*)N->exp_probs_b.dev,
                                  eng->d_pred_ids, eng->d_pred_w, NULL, dim, (int)m->expert_count);
            pred_layer = (int)l + 1;
        }
        CUDA_CHECK(cudaMemcpyAsync(eng->h_moe_ids, eng->d_topk_ids, 4 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        if (pred_layer >= 0)
            CUDA_CHECK(cudaMemcpyAsync(eng->h_moe_ids + 4, eng->d_pred_ids, 4 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        // CPU-miss mode: the host needs the layer input in Q8_1 in case an expert has to be evaluated on the CPU.
        if (eng->cpu_expert)
            CUDA_CHECK(cudaMemcpyAsync(lfm25_cpu_expert_host_xq8(eng->cpu_expert), d_q8_x, (size_t)(dim / 32) * 36,
                                       cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // After this sync every kernel enqueued before this layer's router has finished; only this layer's FFN
        // (enqueued below) can read slots while the prefetch for the next layer is in flight.
        unsigned miss_mask = 0;
        if (eng->cpu_expert) miss_mask = lfm25_expert_cache_lookup_cpu(ec, (int)l, eng->h_moe_ids, 4, eng->h_slot_ids, stream);
        else lfm25_expert_cache_lookup(ec, (int)l, eng->h_moe_ids, 4, eng->h_slot_ids, stream);
        for (int i = 0; i < 4; ++i) if (eng->h_slot_ids[i] < 0) eng->h_slot_ids[i] = 0;   // CPU-evaluated: slot never dereferenced
        // h_slot_ids is pinned and rewritten only after the next layer's sync, i.e. after this copy completed.
        CUDA_CHECK(cudaMemcpyAsync(eng->d_slot_ids, eng->h_slot_ids, 4 * sizeof(int), cudaMemcpyHostToDevice, stream));
        if (miss_mask == 0) {
            lfm25_gpu_moe_ffn(d_q8_x, eng->d_slot_ids, eng->d_topk_w, lfm25_expert_cache_pool_gate(ec),
                              lfm25_expert_cache_pool_up(ec), lfm25_expert_cache_pool_down(ec),
                              (int)L->ffn_down_exps.info.type, eng->d_moe_act, eng->d_moe_act_q8,
                              d_x, true, dim, (int)m->expert_ffn_dim,
                              lfm25_expert_cache_gate_slot_bytes(ec), lfm25_expert_cache_up_slot_bytes(ec),
                              lfm25_expert_cache_down_slot_bytes(ec));
        } else {
            const void *hg[4] = {0}, *hu[4] = {0}, *hd[4] = {0};
            int miss_ids[4], nmiss = 0;
            for (int i = 0; i < 4; ++i) {
                if (!((miss_mask >> i) & 1u)) continue;
                hg[i] = lfm25_expert_cache_host_slab(ec, (int)l, 0, eng->h_moe_ids[i]);
                hu[i] = lfm25_expert_cache_host_slab(ec, (int)l, 1, eng->h_moe_ids[i]);
                hd[i] = lfm25_expert_cache_host_slab(ec, (int)l, 2, eng->h_moe_ids[i]);
                miss_ids[nmiss++] = eng->h_moe_ids[i];
            }
            lfm25_cpu_expert_moe_ffn(eng->cpu_expert, stream, d_q8_x, lfm25_cpu_expert_host_xq8(eng->cpu_expert),
                                     eng->d_slot_ids, eng->d_topk_w, lfm25_expert_cache_pool_gate(ec),
                                     lfm25_expert_cache_pool_up(ec), lfm25_expert_cache_pool_down(ec),
                                     (int)L->ffn_down_exps.info.type, eng->d_moe_act, eng->d_moe_act_q8,
                                     d_x, true, dim, (int)m->expert_ffn_dim,
                                     lfm25_expert_cache_gate_slot_bytes(ec), lfm25_expert_cache_up_slot_bytes(ec),
                                     lfm25_expert_cache_down_slot_bytes(ec), miss_mask, hg, hu, hd);
            lfm25_expert_cache_admit(ec, (int)l, miss_ids, nmiss, eng->cpu_admit);
        }
        // Enqueued after the FFN so host-side staging work overlaps the GPU compute.
        if (pred_layer >= 0) lfm25_expert_cache_prefetch(ec, pred_layer, eng->h_moe_ids + 4, eng->prefetch_n);
    } else {
        lfm25_gpu_moe_ffn(d_q8_x, eng->d_topk_ids, eng->d_topk_w, L->ffn_gate_exps.dev, L->ffn_up_exps.dev,
                          L->ffn_down_exps.dev, (int)L->ffn_down_exps.info.type, eng->d_moe_act, eng->d_moe_act_q8,
                          d_x, true, dim, (int)m->expert_ffn_dim, gs0, us0, ds0);
    }
}

// Enqueue one decode step on the stream selected via lfm25_gpu_set_stream(). Reads (token,pos) from eng->d_tokpos,
// so nothing in here depends on per-token host values; the sequence is identical for every token and capturable.
static void lfm25_enqueue_step(lfm25_engine_t *eng, cudaStream_t stream) {
    lfm25_model_t *m = eng->model;
    int dim = (int)m->embedding_length;
    float eps = m->rms_norm_eps;
    int ffn_dim = (int)m->ffn_dim;

    const int *d_pos = eng->d_tokpos + 1;

    // 0. (token,pos) host -> device, then 1. dequantize the token embedding row on the GPU
    CUDA_CHECK(cudaMemcpyAsync(eng->d_tokpos, eng->h_tokpos, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
    lfm25_gpu_embed(eng->d_tokpos, eng->d_x, &m->token_embd, dim);

    int conv_idx = 0;
    int attn_idx = 0;

    for (uint32_t l = 0; l < m->block_count; ++l) {
        lfm25_layer_t *L = &m->layers[l];

        // --- OPERATOR SUB-BLOCK ---
        // RMSNorm on residual stream -> quantize to q8_1
        lfm25_gpu_fused_rms_norm_quant(eng->d_x, (const float*)L->attn_norm.dev, eng->d_norm_x, eng->d_q8_x, dim, eps);

        if (L->op_type == LFM25_OP_CONV) {
            // ShortConv1D
            // Fused in_proj (Q4_K 2048 -> 6144) + conv modulation, rolling dot, state update, gating & q8_1 quant
            float *conv_state_ptr = eng->d_conv_state + (size_t)conv_idx * (2 * dim);
            lfm25_gpu_inproj_shortconv(&L->shortconv_in_proj, eng->d_q8_x, conv_state_ptr, (const float*)L->shortconv_conv.dev,
                                       eng->d_conv_out, eng->d_q8_op_out, dim);

            // out_proj: 2048 -> 2048 (Q4_K) with accum into residual d_x
            lfm25_gpu_mmvq_q4_K(L->shortconv_out_proj.dev, eng->d_q8_op_out, eng->d_x, dim, dim, true);

            conv_idx++;
        } else {
            // GQA Attention
            // Q projection: 2048 -> 2048 (Q4_K)
            lfm25_gpu_mmvq_q4_K(L->attn_q.dev, eng->d_q8_x, eng->d_q, dim, (int)m->head_count * (int)m->head_dim, false);
            // K projection: 2048 -> 512 (Q4_K)
            lfm25_gpu_mmvq_q4_K(L->attn_k.dev, eng->d_q8_x, eng->d_k, dim, (int)m->head_count_kv * (int)m->head_dim, false);
            // V projection: 2048 -> 512 (Q4_K or Q6_K depending on layer)
            lfm25_gpu_mat_vec(&L->attn_v, eng->d_q8_x, eng->d_v, dim, (int)m->head_count_kv * (int)m->head_dim, false);

            // QK Norm + NeoX RoPE + K Cache Write
            lfm25_gpu_qk_norm_rope(eng->d_q, eng->d_k, (const float*)L->attn_q_norm.dev, (const float*)L->attn_k_norm.dev,
                                   eng->d_k_cache, (int)m->head_count, (int)m->head_count_kv, (int)m->head_dim,
                                   d_pos, eng->max_context_len, m->rope_freq_base, eps, attn_idx);

            // V Cache Write
            lfm25_gpu_v_cache_write(eng->d_v, eng->d_v_cache, (int)m->head_count_kv, (int)m->head_dim, d_pos, eng->max_context_len, attn_idx);

            // Flash-Decoding Attention + quantize to q8_1
            lfm25_gpu_attention(eng->d_q, eng->d_k_cache, eng->d_v_cache, eng->d_op_out, eng->d_q8_op_out,
                                (int)m->head_count, (int)m->head_count_kv, (int)m->head_dim, d_pos, eng->max_context_len, attn_idx);

            // out_proj: 2048 -> 2048 (Q4_K) with accum into residual d_x
            lfm25_gpu_mmvq_q4_K(L->attn_output.dev, eng->d_q8_op_out, eng->d_x, dim, dim, true);

            attn_idx++;
        }

        // --- FFN SUB-BLOCK ---
        // RMSNorm on post-operator residual -> quantize to q8_1
        lfm25_gpu_fused_rms_norm_quant(eng->d_x, (const float*)L->ffn_norm.dev, eng->d_norm_x, eng->d_q8_x, dim, eps);

        if (L->ffn_type == LFM25_FFN_DENSE) {
            // Fused SwiGLU: computes gate (Q4_K) and up (Q4_K), applies silu(gate)*up, quantizes to q8_1
            lfm25_gpu_fused_swiglu(L->ffn_gate.dev, L->ffn_up.dev, eng->d_q8_x, eng->d_ffn_act, eng->d_q8_ffn_act, dim, ffn_dim);

            // down proj: ffn_dim -> 2048 (Q4_K or Q6_K depending on layer) with accum into residual d_x
            lfm25_gpu_mat_vec(&L->ffn_down, eng->d_q8_ffn_act, eng->d_x, ffn_dim, dim, true);
        } else {
            lfm25_enqueue_moe_ffn(eng, stream, l, eng->d_norm_x, eng->d_q8_x, eng->d_x);
        }
    }

    // Output Head + argmax; logits stay on the device unless capture asks for them (see forward_token)
    lfm25_gpu_output_norm_head_argmax(
        eng->d_x, (const float*)m->token_embd_norm.dev, m->output_weight.dev,
        eng->d_q8_x, eng->d_logits,
        eng->d_block_max, eng->d_block_idx, eng->d_best_token,
        dim, (int)m->vocab_size, eps);
    CUDA_CHECK(cudaMemcpyAsync(eng->h_best, eng->d_best_token, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

int32_t lfm25_engine_forward_token(lfm25_engine_t *eng, int32_t token, int pos) {
    if (!eng || !eng->model) return -1;
    if (pos < 0 || pos >= eng->max_context_len) {
        fprintf(stderr, "[LFM25] ERROR: position %d is outside the context window [0, %d) (raise -c)\n", pos, eng->max_context_len);
        return -1;
    }
    if (token < 0 || (uint32_t)token >= eng->model->vocab_size) {
        fprintf(stderr, "[LFM25] ERROR: token id %d is outside the vocabulary [0, %u)\n", token, eng->model->vocab_size);
        return -1;
    }
    eng->batch_n = 0;   // a single-token step invalidates the rollback snapshots
    eng->h_tokpos[0] = token;
    eng->h_tokpos[1] = pos;

    if (eng->graph_enabled) {
        if (!eng->graph_exec) {
            // First token: record the step into a graph (nothing executes during capture), then replay below.
            lfm25_gpu_set_stream(eng->stream);
            CUDA_CHECK(cudaStreamBeginCapture(eng->stream, cudaStreamCaptureModeThreadLocal));
            lfm25_enqueue_step(eng, eng->stream);
            CUDA_CHECK(cudaStreamEndCapture(eng->stream, &eng->graph));
            CUDA_CHECK(cudaGraphInstantiate(&eng->graph_exec, eng->graph, 0));
        }
        CUDA_CHECK(cudaGraphLaunch(eng->graph_exec, eng->stream));
        CUDA_CHECK(cudaStreamSynchronize(eng->stream));
        if (tuff_logit_capture_is_enabled()) {
            CUDA_CHECK(cudaMemcpy(eng->h_logits, eng->d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        }
    } else {
        lfm25_gpu_set_stream(eng->stream);
        lfm25_enqueue_step(eng, eng->stream);
        CUDA_CHECK(cudaStreamSynchronize(eng->stream));
        if (tuff_logit_capture_is_enabled()) {
            CUDA_CHECK(cudaMemcpy(eng->h_logits, eng->d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    if (tuff_logit_capture_is_enabled()) {
        tuff_logit_capture(eng->h_logits, eng->model->vocab_size);
    }
    return *eng->h_best;
}

// -----------------------------------------------------------------------------
// Batched forward + conv-state rollback (Task 5.1)
//
// Weight-streaming matvecs run once for all n tokens (each weight block read once, applied to n activation vectors,
// per-column arithmetic identical to the single-token kernels). Everything else is the single-token math, per token:
// RMSNorm+Q8_1, QK-norm+RoPE+KV write, attention (token j attends positions <= pos0+j only), MoE router + experts.
// ShortConv layers run the recurrence sequentially over the tokens and snapshot the state after every token.
// Not graphed (the single-token graph path is untouched).
// -----------------------------------------------------------------------------
#define LFM25_Q8_BLOCK_BYTES 36

static void lfm25_gpu_mat_vec_batch(const lfm25_tensor_t *t, const void *d_xq8, float *d_dst, int in_dim, int out_dim, bool accum, int nb) {
    if (t->info.type == GGML_TYPE_Q6_K) {
        lfm25_gpu_mmvq_q6_K_batch(t->dev, d_xq8, d_dst, in_dim, out_dim, accum, nb);
    } else {
        lfm25_gpu_mmvq_q4_K_batch(t->dev, d_xq8, d_dst, in_dim, out_dim, accum, nb);
    }
}

int lfm25_engine_forward_batch(lfm25_engine_t *eng, const int32_t *tokens, int n, int pos0, int32_t *argmax_out, float *logits_out) {
    if (!eng || !eng->model || !tokens || !argmax_out || n < 1 || n > LFM25_MAX_BATCH || pos0 < 0 ||
        pos0 + n > eng->max_context_len) {
        if (eng && eng->model && tokens && n >= 1 && n <= LFM25_MAX_BATCH)
            fprintf(stderr, "[LFM25] ERROR: batch positions [%d, %d) are outside the context window [0, %d) (raise -c)\n",
                    pos0, pos0 + n, eng->max_context_len);
        return -1;
    }
    for (int j = 0; j < n; ++j)
        if (tokens[j] < 0 || (uint32_t)tokens[j] >= eng->model->vocab_size) {
            fprintf(stderr, "[LFM25] ERROR: token id %d is outside the vocabulary [0, %u)\n", tokens[j], eng->model->vocab_size);
            return -1;
        }
    lfm25_model_t *m = eng->model;
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int ffn_dim = (int)m->ffn_dim;
    const int qdim = (int)m->head_count * (int)m->head_dim;
    const int kvdim = (int)m->head_count_kv * (int)m->head_dim;
    const int vocab = (int)m->vocab_size;
    const size_t q8x_tok = (size_t)(dim / 32) * LFM25_Q8_BLOCK_BYTES;
    cudaStream_t stream = eng->stream;
    lfm25_gpu_set_stream(stream);
    eng->batch_n = 0;

    for (int j = 0; j < n; ++j) {
        eng->h_btokpos[2 * j + 0] = tokens[j];
        eng->h_btokpos[2 * j + 1] = pos0 + j;
    }
    CUDA_CHECK(cudaMemcpyAsync(eng->d_btokpos, eng->h_btokpos, (size_t)2 * n * sizeof(int), cudaMemcpyHostToDevice, stream));
    const int *bpos = eng->d_btokpos + 1;   // position of token j is bpos[2*j]

    for (int j = 0; j < n; ++j)
        lfm25_gpu_embed(eng->d_btokpos + 2 * j, eng->d_bx + (size_t)j * dim, &m->token_embd, dim);

    int conv_idx = 0;
    int attn_idx = 0;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        lfm25_layer_t *L = &m->layers[l];

        for (int j = 0; j < n; ++j)
            lfm25_gpu_fused_rms_norm_quant(eng->d_bx + (size_t)j * dim, (const float*)L->attn_norm.dev, eng->d_bnorm + (size_t)j * dim,
                                           (char*)eng->d_bq8_x + j * q8x_tok, dim, eps);

        if (L->op_type == LFM25_OP_CONV) {
            float *conv_state_ptr = eng->d_conv_state + (size_t)conv_idx * (2 * dim);
            float *snap_ptr = eng->d_conv_snap + (size_t)conv_idx * LFM25_MAX_BATCH * (2 * dim);
            lfm25_gpu_mat_vec_batch(&L->shortconv_in_proj, eng->d_bq8_x, eng->d_bin_proj, dim, 3 * dim, false, n);
            lfm25_gpu_shortconv_seq(eng->d_bin_proj, conv_state_ptr, snap_ptr, (const float*)L->shortconv_conv.dev, eng->d_bq8_op, dim, n);
            lfm25_gpu_mmvq_q4_K_batch(L->shortconv_out_proj.dev, eng->d_bq8_op, eng->d_bx, dim, dim, true, n);
            conv_idx++;
        } else {
            lfm25_gpu_mmvq_q4_K_batch(L->attn_q.dev, eng->d_bq8_x, eng->d_bq, dim, qdim, false, n);
            lfm25_gpu_mmvq_q4_K_batch(L->attn_k.dev, eng->d_bq8_x, eng->d_bk, dim, kvdim, false, n);
            lfm25_gpu_mat_vec_batch(&L->attn_v, eng->d_bq8_x, eng->d_bv, dim, kvdim, false, n);

            // K/V of all n tokens are written to their own positions before any attention runs.
            for (int j = 0; j < n; ++j) {
                const int *rpos = bpos + 2 * j;
                lfm25_gpu_qk_norm_rope(eng->d_bq + (size_t)j * qdim, eng->d_bk + (size_t)j * kvdim,
                                       (const float*)L->attn_q_norm.dev, (const float*)L->attn_k_norm.dev,
                                       eng->d_k_cache, (int)m->head_count, (int)m->head_count_kv, (int)m->head_dim,
                                       rpos, eng->max_context_len, m->rope_freq_base, eps, attn_idx);
                lfm25_gpu_v_cache_write(eng->d_bv + (size_t)j * kvdim, eng->d_v_cache, (int)m->head_count_kv, (int)m->head_dim,
                                        rpos, eng->max_context_len, attn_idx);
            }
            // Token j attends positions <= pos0+j (the kernel's own causal bound), so it never sees token j+1's K/V.
            for (int j = 0; j < n; ++j) {
                const int *apos = bpos + 2 * j;
                lfm25_gpu_attention(eng->d_bq + (size_t)j * qdim, eng->d_k_cache, eng->d_v_cache, eng->d_bop_out + (size_t)j * dim,
                                    (char*)eng->d_bq8_op + j * q8x_tok, (int)m->head_count, (int)m->head_count_kv,
                                    (int)m->head_dim, apos, eng->max_context_len, attn_idx);
            }
            lfm25_gpu_mmvq_q4_K_batch(L->attn_output.dev, eng->d_bq8_op, eng->d_bx, dim, dim, true, n);
            attn_idx++;
        }

        for (int j = 0; j < n; ++j)
            lfm25_gpu_fused_rms_norm_quant(eng->d_bx + (size_t)j * dim, (const float*)L->ffn_norm.dev, eng->d_bnorm + (size_t)j * dim,
                                           (char*)eng->d_bq8_x + j * q8x_tok, dim, eps);

        if (L->ffn_type == LFM25_FFN_DENSE) {
            lfm25_gpu_fused_swiglu_batch(L->ffn_gate.dev, L->ffn_up.dev, eng->d_bq8_x, eng->d_bffn_act, eng->d_bq8_ffn, dim, ffn_dim, n);
            lfm25_gpu_mat_vec_batch(&L->ffn_down, eng->d_bq8_ffn, eng->d_bx, ffn_dim, dim, true, n);
        } else {
            // Router + experts per token with the single-token calls (cache path: per-layer lookup per token).
            for (int j = 0; j < n; ++j)
                lfm25_enqueue_moe_ffn(eng, stream, l, eng->d_bnorm + (size_t)j * dim, (char*)eng->d_bq8_x + j * q8x_tok,
                                      eng->d_bx + (size_t)j * dim);
        }
    }

    lfm25_gpu_output_norm_head_argmax_batch(eng->d_bx, (const float*)m->token_embd_norm.dev, m->output_weight.dev,
                                            eng->d_bq8_x, eng->d_blogits, eng->d_bblock_max, eng->d_bblock_idx, eng->d_bbest,
                                            dim, vocab, eps, n);
    CUDA_CHECK(cudaMemcpyAsync(eng->h_bbest, eng->d_bbest, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (logits_out) {
        CUDA_CHECK(cudaMemcpy(logits_out, eng->d_blogits, (size_t)n * vocab * sizeof(float), cudaMemcpyDeviceToHost));
    }
    for (int j = 0; j < n; ++j) argmax_out[j] = eng->h_bbest[j];
    eng->batch_n = n;
    return 0;
}

int lfm25_engine_rollback(lfm25_engine_t *eng, int n_keep) {
    if (!eng || !eng->model || n_keep < 1 || n_keep > eng->batch_n) return -1;
    // The live conv state already is "after all batch_n tokens" (the sequential kernel leaves it there); only a
    // shorter keep needs the snapshot after token n_keep-1 copied back (one 2D copy over all conv layers).
    if (n_keep < eng->batch_n && eng->model->n_conv_layers > 0) {
        cudaStream_t stream = eng->stream;   // same stream as the batch that produced the snapshots; ordered, then synced
        lfm25_gpu_set_stream(stream);
        const int dim = (int)eng->model->embedding_length;
        lfm25_gpu_conv_state_restore(eng->d_conv_state, eng->d_conv_snap, (int)eng->model->n_conv_layers, n_keep - 1, dim,
                                     (size_t)LFM25_MAX_BATCH * 2 * dim);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    eng->batch_n = n_keep;   // slots [0, n_keep) stay valid, so a further (shorter) rollback is still exact
    return 0;
}

static double lfm25_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

// LFM25_BENCH_BATCH=1: wall time of forward_batch(n) vs n x forward_token for n = 1,2,4,8 (LFM25_BENCH_REPS, default 20).
// Runs at engine init, then resets the conv state; the KV cache needs no reset (positions are overwritten).
static void lfm25_bench_batch(lfm25_engine_t *eng) {
    const char *rs = getenv("LFM25_BENCH_REPS");
    int reps = rs && *rs ? atoi(rs) : 20;
    if (reps < 1) reps = 1;
    const int ns[4] = {1, 2, 4, 8};
    const int base = 8, warm = 3;
    int32_t toks[LFM25_MAX_BATCH], am[LFM25_MAX_BATCH];
    uint32_t vocab = eng->model->vocab_size;
    fprintf(stderr, "[BENCH-BATCH] reps=%d (+%d warmup), ctx pos %d.., ungraphed batch vs %s single-token\n", reps, warm, base,
            eng->graph_enabled ? "graphed" : "ungraphed");
    for (int ni = 0; ni < 4; ++ni) {
        const int n = ns[ni];
        double t_batch = 0.0, t_single = 0.0;
        for (int r = -warm; r < reps; ++r) {
            for (int j = 0; j < n; ++j) toks[j] = (int32_t)(1000u + (uint32_t)((r + warm) * 7919 + j * 104729) % (vocab - 2000u));
            // Alternate which variant goes first so neither systematically benefits from the other's expert-cache warming.
            double tb = 0.0, ts = 0.0;
            for (int phase = 0; phase < 2; ++phase) {
                double t0 = lfm25_now_ms();
                if (((r & 1) == 0) == (phase == 0)) {
                    if (lfm25_engine_forward_batch(eng, toks, n, base, am, NULL) != 0) { fprintf(stderr, "[BENCH-BATCH] forward_batch failed\n"); return; }
                    tb = lfm25_now_ms() - t0;
                } else {
                    for (int j = 0; j < n; ++j) lfm25_engine_forward_token(eng, toks[j], base + j);
                    ts = lfm25_now_ms() - t0;
                }
            }
            if (r >= 0) { t_batch += tb; t_single += ts; }
        }
        t_batch /= reps; t_single /= reps;
        fprintf(stderr, "[BENCH-BATCH] n=%d  forward_batch %8.3f ms (%.3f ms/tok)   n x forward_token %8.3f ms (%.3f ms/tok)   batch/single %.2f\n",
                n, t_batch, t_batch / n, t_single, t_single / n, t_batch / t_single);
    }
    if (eng->d_conv_state)
        CUDA_CHECK(cudaMemset(eng->d_conv_state, 0, (size_t)eng->model->n_conv_layers * eng->model->conv_state_len * eng->model->embedding_length * sizeof(float)));
    eng->batch_n = 0;
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
    const lfm25_model_t *m = (const lfm25_model_t*)model;
    return (int32_t)tuff_gguf_get_kv_u32(m->gguf, "tokenizer.ggml.bos_token_id", 1);
}

static int32_t vtable_lfm25_model_get_eos(void *model) {
    const lfm25_model_t *m = (const lfm25_model_t*)model;
    return (int32_t)tuff_gguf_get_kv_u32(m->gguf, "tokenizer.ggml.eos_token_id", 2);
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

static int vtable_lfm25_engine_forward_batch(void *engine, const int32_t *tokens, int n, int pos0,
                                             int32_t *argmax_out, float *logits_out) {
    return lfm25_engine_forward_batch((lfm25_engine_t*)engine, tokens, n, pos0, argmax_out, logits_out);
}

static int vtable_lfm25_engine_rollback(void *engine, int n_keep) {
    return lfm25_engine_rollback((lfm25_engine_t*)engine, n_keep);
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
    .engine_forward_batch   = vtable_lfm25_engine_forward_batch,
    .engine_rollback        = vtable_lfm25_engine_rollback,
};
