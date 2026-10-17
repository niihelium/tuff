#include "state.h"

#define EXPERT_CACHE_DEFAULT_SLOTS 2500

// Iteration-12 static expert split count (layers whose full expert sets are resident in VRAM).
int g_n_gpu_expert_layers = 0;

typedef struct {
    uint32_t layer;
    int      expert;
    uint8_t  valid;
    uint64_t heat;
    uint64_t last_used;
} expert_slot_meta_t;

static int g_n_slots = 0;
static int g_total_experts = 0;             // block_count * expert_count (direct-map key space)
static expert_slot_meta_t *g_slots = NULL;  // host metadata
static int *g_expert_map = NULL;            // key(layer*256+expert) -> slot, or -1
static uint64_t g_tick = 0;
static uint64_t g_cache_hits = 0, g_cache_misses = 0;
static int g_cache_ready = 0;
static int g_cache_async = 0;
static cudaStream_t g_upload_stream = 0;

// H18c: TUFF_EXPERT_HIST=1 counts per-(layer,expert) routing accesses so the
// skew of the routing distribution can be measured before deciding whether a
// static pin of hot experts could beat LFRU's eviction churn. Counted for every
// lookup (hit or miss); host-only, indexed by layer*256+expert.
static uint64_t *g_expert_hist = NULL;
static int g_expert_hist_on = 0;

// Iteration-10: O(1) slot allocation. The LRU victim search used to be an O(n_slots=2200) scan on
// every miss (the dominant miss-fill latency); replace it with a doubly-linked LRU list (MRU at
// head, LRU at tail) plus a freelist for the initial fill, so alloc is O(1) in steady state.
static int *g_lru_next = NULL;   // [n_slots]
static int *g_lru_prev = NULL;   // [n_slots]
static int g_lru_head = -1;      // MRU
static int g_lru_tail = -1;      // LRU (eviction victim)
static int g_free_head = -1;     // freelist of never-used (valid==0) slots
static int g_valid_count = 0;

// Dedicated async-expert scratch + stream, so the pipelined gate/up launches overlap the CPU
// down of the previous expert without colliding with the default-stream g_dx/g_dy scratch.

static int g_cache_enabled(void) {
    return !getenv("TUFF_CPU_EXPERT") && g_cache_ready;
}

// P1: background cache-fill worker (defined after the LRU helpers below).
static void bgfill_start(void);
static void bgfill_stop(void);

void gpu_expert_cache_init_ex(model_t *m, int *p_max_seq, int fill_ram) {
    if (!m || g_pool) return;

    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)m->layers[0].ffn_gate_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn; // 512 rows * 1152 = 589824 (Q4_K)
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn; // same
    // Down is MIXED per layer: Q4_K (589824 B/expert) in most layers, Q6_K (860160 B/expert)
    // in a few deep layers. Size the slot's down region for the largest, so any layer fits and
    // all down (including the Q6_K deep layers) runs on the GPU.
    size_t down_bytes = 0;
    for (uint32_t i = 0; i < m->block_count; ++i) {
        const size_t b = ggml_row_size((ggml_type_t)m->layers[i].ffn_down_exps.info.type, ffn) * dim;
        if (b > down_bytes) down_bytes = b;
    }
    g_tensor_off[0] = 0;
    g_tensor_off[1] = gate_bytes;
    g_tensor_off[2] = gate_bytes + up_bytes;
    g_slot_bytes = gate_bytes + up_bytes + down_bytes;

    // Allocate scratch buffers first so free VRAM measurement is accurate
    if (!g_exgx)   CUDA_CHECK(cudaMalloc(&g_exgx,   (size_t)dim  * sizeof(float)));
    if (!g_exgy)   CUDA_CHECK(cudaMalloc(&g_exgy,   (size_t)ffn  * sizeof(float)));
    if (!g_exgy2)  CUDA_CHECK(cudaMalloc(&g_exgy2,  (size_t)ffn  * sizeof(float)));
    g_ex_stride_dx    = (size_t)EXPERT_BATCH_MAX * ffn;
    g_ex_stride_dbat  = (size_t)EXPERT_BATCH_MAX * dim;
    g_ex_stride_slots = (size_t)EXPERT_BATCH_MAX;
    g_ex_stride_gxq8  = (size_t)(dim / 32) * 36;
    g_ex_stride_dxq8  = (size_t)EXPERT_BATCH_MAX * (ffn / 32) * 36;
    if (!g_exdx)   CUDA_CHECK(cudaMalloc(&g_exdx,   BATCH_COLS_MAX * g_ex_stride_dx * sizeof(float)));
    if (!g_exdy)   CUDA_CHECK(cudaMalloc(&g_exdy,   (size_t)dim  * sizeof(float)));
    if (!g_exgbat) CUDA_CHECK(cudaMalloc(&g_exgbat, (size_t)EXPERT_BATCH_MAX * ffn * sizeof(float)));
    if (!g_exubbat) CUDA_CHECK(cudaMalloc(&g_exubbat, (size_t)EXPERT_BATCH_MAX * ffn * sizeof(float)));
    if (!g_exdbat) CUDA_CHECK(cudaMalloc(&g_exdbat, BATCH_COLS_MAX * g_ex_stride_dbat * sizeof(float)));
    if (!g_exslots) CUDA_CHECK(cudaMalloc(&g_exslots, BATCH_COLS_MAX * g_ex_stride_slots * sizeof(int)));
    if (!g_d_exdst) CUDA_CHECK(cudaMalloc(&g_d_exdst, BATCH_COLS_MAX * EXPERT_BATCH_MAX * sizeof(int)));
    if (!g_exgx_q8) CUDA_CHECK(cudaMalloc(&g_exgx_q8, BATCH_COLS_MAX * g_ex_stride_gxq8));
    if (!g_exdx_q8) CUDA_CHECK(cudaMalloc(&g_exdx_q8, BATCH_COLS_MAX * g_ex_stride_dxq8));
    if (!g_d_union_slots) CUDA_CHECK(cudaMalloc(&g_d_union_slots, 64 * sizeof(int)));
    if (!g_d_union_m)     CUDA_CHECK(cudaMalloc(&g_d_union_m,     64 * sizeof(int)));
    if (!g_d_union_cols)  CUDA_CHECK(cudaMalloc(&g_d_union_cols,  64 * 8 * sizeof(int)));
    if (!g_d_union_dst)   CUDA_CHECK(cudaMalloc(&g_d_union_dst,   64 * 8 * sizeof(int)));
    if (!g_expert_stream) CUDA_CHECK(cudaStreamCreate(&g_expert_stream));

    // H56: Ampere L2 Cache Persistence window (sm_86, 4 MB L2).
    // Pin 1 MB of L2 cache for hot inter-layer slots and activation scratch buffers.
    size_t l2_persisting_bytes = 0;
    cudaDeviceGetLimit(&l2_persisting_bytes, cudaLimitPersistingL2CacheSize);
    if (l2_persisting_bytes < 1024 * 1024) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1024 * 1024);
    }
    cudaStreamAttrValue l2_attr;
    memset(&l2_attr, 0, sizeof(l2_attr));
    l2_attr.accessPolicyWindow.base_ptr = (void*)g_exslots;
    l2_attr.accessPolicyWindow.num_bytes = 1024 * 1024;
    l2_attr.accessPolicyWindow.hitRatio = 1.0f;
    l2_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    l2_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(g_expert_stream, cudaStreamAttributeAccessPolicyWindow, &l2_attr);

    // Dynamic expert slot adaptation based on context size and available VRAM:
    const int n_attn = (int)m->n_attn_layers;
    const int n_head_kv = (int)m->head_count_kv;
    const int head_dim = m->key_length ? (int)m->key_length : ((int)m->embedding_length / (int)m->head_count);
    size_t kv_bytes_per_tok = 0;
    if (g_kv_quant_mode == KV_QUANT_Q8_0) {
        kv_bytes_per_tok = (size_t)2 * n_attn * n_head_kv * (head_dim / 32) * sizeof(block_q8_0);
    } else {
        kv_bytes_per_tok = (size_t)2 * n_attn * n_head_kv * head_dim * sizeof(float);
    }

    int ctx_len = p_max_seq ? *p_max_seq : 4096;
    if (ctx_len < 16) ctx_len = 4096;

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    int margin_mb = 70;
    const char *margin_env = getenv("TUFF_VRAM_MARGIN_MB");
    if (margin_env && margin_env[0] != '\0') margin_mb = atoi(margin_env);
    if (margin_mb < 16) margin_mb = 16;
    const size_t margin_bytes = (size_t)margin_mb * 1024 * 1024;

    const size_t usable_b = (free_b > margin_bytes) ? (free_b - margin_bytes) : 0;
    size_t kv_bytes = g_kv_cache_bytes;

    const char *s = getenv("TUFF_EXPERT_SLOTS");
    if (s && s[0] != '\0') {
        const int dynamic_slots = atoi(s);
        g_n_slots = (g_n_gpu_expert_layers > 0)
                  ? g_n_gpu_expert_layers * (int)m->expert_count + dynamic_slots
                  : dynamic_slots;
    } else {
        int auto_slots = (int)(usable_b / g_slot_bytes);
        // Clamp auto_slots to maximum total routed experts in model if more VRAM is available
        const int total_model_experts = (int)m->block_count * (int)m->expert_count;
        if (auto_slots > total_model_experts) auto_slots = total_model_experts;
        g_n_slots = (g_n_gpu_expert_layers > 0)
                  ? g_n_gpu_expert_layers * (int)m->expert_count + auto_slots
                  : auto_slots;
    }
    if (g_n_slots <= 0) g_n_slots = 1;

    // Fill RAM / VRAM inflation: if enabled, use any leftover VRAM (that cannot fit an expert)
    // to expand the context window to maximize VRAM utilization.
    if (fill_ram || getenv("TUFF_FILL_VRAM") || getenv("TUFF_FILL_RAM")) {
        size_t pool_bytes_used = (size_t)g_n_slots * g_slot_bytes;
        size_t total_allocated = pool_bytes_used + kv_bytes;
        if (usable_b > total_allocated && kv_bytes_per_tok > 0) {
            size_t leftover_b = usable_b - total_allocated;
            size_t extra_toks = leftover_b / kv_bytes_per_tok;
            if (extra_toks > 0) {
                int old_ctx = ctx_len;
                ctx_len += (int)extra_toks;
                if (p_max_seq) *p_max_seq = ctx_len;
                kv_bytes = (size_t)ctx_len * kv_bytes_per_tok;
                fprintf(stderr, "[VRAM-ADAPT] fill_ram: expanding context %d -> %d (+%d tokens, utilizing +%.2f MiB leftover VRAM)\n",
                        old_ctx, ctx_len, (int)extra_toks, (double)(extra_toks * kv_bytes_per_tok) / (1024.0 * 1024.0));
            }
        }
    }

    const size_t pool_bytes = (size_t)g_n_slots * g_slot_bytes;
    CUDA_CHECK(cudaMalloc((void**)&g_pool, pool_bytes));
    CUDA_CHECK(cudaMemset(g_pool, 0, pool_bytes));

    g_total_experts = (int)(m->block_count * m->expert_count);
    g_slots = (expert_slot_meta_t*)calloc((size_t)g_n_slots, sizeof(expert_slot_meta_t));
    g_expert_map = (int*)malloc((size_t)g_total_experts * sizeof(int));
    for (int i = 0; i < g_total_experts; ++i) g_expert_map[i] = -1;
    if (getenv("TUFF_EXPERT_HIST")) {
        g_expert_hist = (uint64_t*)calloc((size_t)g_total_experts, sizeof(uint64_t));
        g_expert_hist_on = g_expert_hist != NULL;
        fprintf(stderr, "[EXPERT-HIST] per-(layer,expert) routing histogram ON (%d keys)\n", g_total_experts);
    }

    // O(1) alloc: freelist of dynamic slots (valid==0); LRU list starts empty.
    g_lru_next = (int*)malloc((size_t)g_n_slots * sizeof(int));
    g_lru_prev = (int*)malloc((size_t)g_n_slots * sizeof(int));
    g_lru_head = g_lru_tail = -1;
    g_free_head = -1;
    g_valid_count = 0;
    const int n_res_slots = (g_n_gpu_expert_layers > 0) ? (g_n_gpu_expert_layers * (int)m->expert_count) : 0;
    for (int i = 0; i < g_n_slots; ++i) { g_lru_next[i] = -1; g_lru_prev[i] = -1; }
    for (int i = g_n_slots - 1; i >= n_res_slots; --i) { g_lru_next[i] = g_free_head; g_free_head = i; }

    g_cache_async = 1; // async fill uploads on a dedicated stream, synced before the batch launch
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_upload_stream, cudaStreamNonBlocking));

    g_cache_ready = 1;
    if (g_moe_policy) { moe_policy_free(g_moe_policy); g_moe_policy = NULL; }
    g_moe_policy = moe_policy_create((int)m->block_count, (int)m->expert_count);
    bgfill_start();
    fprintf(stderr, "[EXPERT-CACHE] Context: %d tokens (KV: %.1f MiB, %.1f KiB/tok) | Expert Slots: %d (Pool: %.1f MiB, %.2f MiB/slot) | Margin: %d MiB | Free VRAM: %.1f MiB\n",
            ctx_len, (double)kv_bytes / (1024.0 * 1024.0), (double)kv_bytes_per_tok / 1024.0,
            g_n_slots, (double)pool_bytes / (1024.0 * 1024.0), (double)g_slot_bytes / (1024.0 * 1024.0),
            margin_mb, (double)(free_b - pool_bytes - kv_bytes) / (1024.0 * 1024.0));
}

void gpu_expert_cache_init(model_t *m) {
    gpu_expert_cache_init_ex(m, NULL, 0);
}

static void lru_push_front(int slot);

static void gpu_expert_cache_fill_pinned(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;

    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)expert * up_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)expert * down_bytes;

    if (g_cache_async) {
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice));
    }
    expert_slot_meta_t *sl = &g_slots[slot];
    sl->valid = 1;
    sl->layer = layer;
    sl->expert = expert;
    sl->heat = 1;
    sl->last_used = 0;
    g_expert_map[layer * 256 + expert] = slot;
    g_valid_count++;
}

void gpu_expert_cache_prefill(model_t *m) {
    if (!g_cache_ready) return;
    if (g_n_gpu_expert_layers > 0) {
        const int n_layers = g_n_gpu_expert_layers < (int)m->block_count ? g_n_gpu_expert_layers : (int)m->block_count;
        int n_filled = 0;
        for (int l = 0; l < n_layers; ++l) {
            for (int e = 0; e < (int)m->expert_count; ++e) {
                int slot = l * (int)m->expert_count + e;
                gpu_expert_cache_fill_pinned(slot, m, (uint32_t)l, e);
                n_filled++;
            }
        }
        gpu_expert_cache_sync();
        fprintf(stderr, "[EXPERT-CACHE] pre-filled %d resident layers x %d experts = %d slots "
                        "(%.2f MiB resident expert VRAM)\n",
                n_layers, (int)m->expert_count, n_filled,
                (double)n_filled * g_slot_bytes / 1048576.0);
    } else if (g_n_slots > 0) {
        // Predictive startup prefill: distribute all available slots evenly across all layers
        const int per_layer = g_n_slots / (int)m->block_count;
        int slot = 0;
        for (int l = 0; l < (int)m->block_count && slot < g_n_slots; ++l) {
            for (int e = 0; e < per_layer && slot < g_n_slots; ++e) {
                gpu_expert_cache_fill_pinned(slot, m, (uint32_t)l, e);
                lru_push_front(slot);
                slot++;
            }
        }
        g_free_head = (slot < g_n_slots) ? slot : -1;
        gpu_expert_cache_sync();
        fprintf(stderr, "[EXPERT-CACHE] pre-filled %d dynamic slots across %d layers (%d per layer, %.2f MiB)\n",
                slot, (int)m->block_count, per_layer, (double)slot * g_slot_bytes / 1048576.0);
    }
}

// H18c: dump the routing histogram at teardown. Reports the overall Zipf-ness
// (fraction of accesses covered by the top-X% of distinct experts) and, per
// layer, what fraction of that layer's accesses its top-K experts cover.
static int hist_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) ? -1 : (x < y);
}

static void gpu_expert_hist_dump(void) {
    if (!g_expert_hist_on || !g_expert_hist) return;
    const int n_layer = g_total_experts / 256;
    const int n_expert = 256;
    uint64_t total = 0;
    int distinct = 0;
    for (int i = 0; i < g_total_experts; ++i) { total += g_expert_hist[i]; if (g_expert_hist[i]) distinct++; }
    if (total == 0) { fprintf(stderr, "[EXPERT-HIST] no accesses recorded\n"); return; }

    uint64_t *sorted = (uint64_t*)malloc((size_t)distinct * sizeof(uint64_t));
    int n = 0;
    for (int i = 0; i < g_total_experts; ++i) if (g_expert_hist[i]) sorted[n++] = g_expert_hist[i];
    qsort(sorted, (size_t)n, sizeof(uint64_t), hist_cmp);

    fprintf(stderr, "[EXPERT-HIST] accesses=%llu distinct_experts=%d (of %d)\n",
            (unsigned long long)total, distinct, g_total_experts);
    const int fracs[] = {1, 2, 5, 10, 25, 50};
    for (size_t f = 0; f < sizeof(fracs) / sizeof(fracs[0]); ++f) {
        int k = distinct * fracs[f] / 100;
        if (k <= 0) k = 1;
        uint64_t cov = 0;
        for (int i = 0; i < k && i < n; ++i) cov += sorted[i];
        fprintf(stderr, "[EXPERT-HIST]   top %2d%% of distinct experts (%d) cover %.1f%% of accesses\n",
                fracs[f], k, 100.0 * (double)cov / (double)total);
    }

    // Per-layer concentration: for the 6 most-accessed layers, the coverage of
    // their top-K experts (K = the active count 8, then 16/32/64).
    int *layer_perm = (int*)malloc((size_t)n_layer * sizeof(int));
    uint64_t *layer_total = (uint64_t*)calloc((size_t)n_layer, sizeof(uint64_t));
    for (int l = 0; l < n_layer; ++l) { layer_perm[l] = l;
        for (int e = 0; e < n_expert; ++e) layer_total[l] += g_expert_hist[l * n_expert + e]; }
    for (int i = 0; i < n_layer; ++i)
        for (int j = i + 1; j < n_layer; ++j)
            if (layer_total[layer_perm[j]] > layer_total[layer_perm[i]]) {
                int t = layer_perm[i]; layer_perm[i] = layer_perm[j]; layer_perm[j] = t;
            }
    fprintf(stderr, "[EXPERT-HIST] per-layer top-K coverage (of that layer's accesses):\n");
    for (int li = 0; li < n_layer && li < 6; ++li) {
        int l = layer_perm[li];
        uint64_t *es = (uint64_t*)malloc((size_t)n_expert * sizeof(uint64_t));
        for (int e = 0; e < n_expert; ++e) es[e] = g_expert_hist[l * n_expert + e];
        qsort(es, (size_t)n_expert, sizeof(uint64_t), hist_cmp);
        if (layer_total[l] == 0) { free(es); continue; }
        fprintf(stderr, "[EXPERT-HIST]   layer %2d: accesses=%llu  top8=%.1f%% top16=%.1f%% top32=%.1f%% top64=%.1f%%\n",
                l, (unsigned long long)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7]) / (double)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7] + es[8] + es[9] + es[10] + es[11] + es[12] + es[13] + es[14] + es[15]) / (double)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7] + es[8] + es[9] + es[10] + es[11] + es[12] + es[13] + es[14] + es[15] + es[16] + es[17] + es[18] + es[19] + es[20] + es[21] + es[22] + es[23] + es[24] + es[25] + es[26] + es[27] + es[28] + es[29] + es[30] + es[31]) / (double)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7] + es[8] + es[9] + es[10] + es[11] + es[12] + es[13] + es[14] + es[15] + es[16] + es[17] + es[18] + es[19] + es[20] + es[21] + es[22] + es[23] + es[24] + es[25] + es[26] + es[27] + es[28] + es[29] + es[30] + es[31] + es[32] + es[33] + es[34] + es[35] + es[36] + es[37] + es[38] + es[39] + es[40] + es[41] + es[42] + es[43] + es[44] + es[45] + es[46] + es[47] + es[48] + es[49] + es[50] + es[51] + es[52] + es[53] + es[54] + es[55] + es[56] + es[57] + es[58] + es[59] + es[60] + es[61] + es[62] + es[63]) / (double)layer_total[l]);
        free(es);
    }
    free(sorted);
    free(layer_perm);
    free(layer_total);
}

void gpu_expert_cache_free(void) {
    bgfill_stop();
    gpu_expert_hist_dump();
    if (g_pool)       { CUDA_CHECK(cudaFree(g_pool)); g_pool = NULL; }
    if (g_upload_stream) { CUDA_CHECK(cudaStreamDestroy(g_upload_stream)); g_upload_stream = 0; }
    if (g_exgx)      { CUDA_CHECK(cudaFree(g_exgx));   g_exgx = NULL; }
    if (g_exgy)      { CUDA_CHECK(cudaFree(g_exgy));   g_exgy = NULL; }
    if (g_exgy2)     { CUDA_CHECK(cudaFree(g_exgy2));  g_exgy2 = NULL; }
    if (g_exdx)      { CUDA_CHECK(cudaFree(g_exdx));   g_exdx = NULL; }
    if (g_exdy)      { CUDA_CHECK(cudaFree(g_exdy));   g_exdy = NULL; }
    if (g_exgbat)    { CUDA_CHECK(cudaFree(g_exgbat)); g_exgbat = NULL; }
    if (g_exubbat)   { CUDA_CHECK(cudaFree(g_exubbat)); g_exubbat = NULL; }
    if (g_exdbat)    { CUDA_CHECK(cudaFree(g_exdbat)); g_exdbat = NULL; }
    if (g_exslots)   { CUDA_CHECK(cudaFree(g_exslots)); g_exslots = NULL; }
    if (g_d_exdst)   { CUDA_CHECK(cudaFree(g_d_exdst)); g_d_exdst = NULL; }
    if (g_exgx_q8)   { CUDA_CHECK(cudaFree(g_exgx_q8)); g_exgx_q8 = NULL; }
    if (g_exdx_q8)   { CUDA_CHECK(cudaFree(g_exdx_q8)); g_exdx_q8 = NULL; }
    if (g_d_union_slots) { CUDA_CHECK(cudaFree(g_d_union_slots)); g_d_union_slots = NULL; }
    if (g_d_union_m)     { CUDA_CHECK(cudaFree(g_d_union_m));     g_d_union_m = NULL; }
    if (g_d_union_cols)  { CUDA_CHECK(cudaFree(g_d_union_cols));  g_d_union_cols = NULL; }
    if (g_d_union_dst)   { CUDA_CHECK(cudaFree(g_d_union_dst));   g_d_union_dst = NULL; }
    if (g_expert_stream) { CUDA_CHECK(cudaStreamDestroy(g_expert_stream)); g_expert_stream = 0; }
    free(g_slots); g_slots = NULL;
    free(g_expert_map); g_expert_map = NULL;
    free(g_lru_next); g_lru_next = NULL;
    free(g_lru_prev); g_lru_prev = NULL;
    g_lru_head = g_lru_tail = -1; g_free_head = -1; g_valid_count = 0;
    g_n_slots = 0; g_slot_bytes = 0; g_cache_ready = 0;
    free(g_expert_hist); g_expert_hist = NULL; g_expert_hist_on = 0;
    if (g_moe_policy) { moe_policy_free(g_moe_policy); g_moe_policy = NULL; }
}

// --- O(1) LRU + freelist bookkeeping (iteration-10) ---
// The LRU list holds all valid slots, MRU at g_lru_head, LRU at g_lru_tail. A slot that is valid
// is exactly one node in the LRU list. g_lru_next/prev[-1] are self-terminating sentinels.
static void lru_remove(int slot) {
    const int nxt = g_lru_next[slot], prv = g_lru_prev[slot];
    if (prv < 0) g_lru_head = nxt; else g_lru_next[prv] = nxt;
    if (nxt < 0) g_lru_tail = prv; else g_lru_prev[nxt] = prv;
    g_lru_next[slot] = g_lru_prev[slot] = -1;
}

static void lru_push_front(int slot) {
    g_lru_prev[slot] = -1;
    g_lru_next[slot] = g_lru_head;
    if (g_lru_head >= 0) g_lru_prev[g_lru_head] = slot;
    else g_lru_tail = slot;
    g_lru_head = slot;
}

static int freelist_pop(void) {
    const int s = g_free_head;
    if (s >= 0) g_free_head = g_lru_next[s]; // reuse next as free-list link
    return s;
}

static void expert_slot_claim(int slot, uint32_t layer, int expert) {
    expert_slot_meta_t *sl = &g_slots[slot];
    if (sl->valid) { g_expert_map[sl->layer * 256 + sl->expert] = -1; lru_remove(slot); }
    else g_valid_count++;
    sl->valid = 1;
    sl->layer = layer;
    sl->expert = expert;
    sl->heat = 1;
    sl->last_used = ++g_tick;
    g_expert_map[layer * 256 + expert] = slot;
    lru_push_front(slot);
}

// Look up (layer,expert). Returns slot index on HIT (resident, data on device), -1 on MISS.
// A HIT moves the slot to the MRU end of the LRU list (O(1)).
int gpu_expert_lookup(uint32_t layer, int expert) {
    if (!g_cache_ready) return -1;
    if (g_expert_hist_on && layer < (uint32_t)g_total_experts / 256)
        g_expert_hist[layer * 256 + expert]++;
    int slot = g_expert_map[layer * 256 + expert];
    if (slot >= 0) {
        g_slots[slot].heat++;
        g_slots[slot].last_used = ++g_tick;
        if (layer >= (uint32_t)g_n_gpu_expert_layers && g_lru_head != slot) {
            lru_remove(slot);
            lru_push_front(slot);
        }
        g_cache_hits++;
    } else {
        g_cache_misses++;
    }
    return slot;
}

// Device pointer for tensor `which` (0=gate,1=up,2=down) inside slot `slot`.
const void *gpu_expert_dev_ptr(int slot, int which) {
    return g_pool + (size_t)slot * g_slot_bytes + g_tensor_off[which];
}

void gpu_expert_cache_fill(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;

    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)expert * up_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)expert * down_bytes;

    if (g_cache_async) {
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice));
    }
    expert_slot_claim(slot, layer, expert);
}

// Split-fill for overlapping the miss upload behind the batch compute. gate+up are needed by the
// gate/up batch (Phase A); down is needed only by the down batch (Phase C). Filling gate+up first
// lets Phase A start while the larger down upload is deferred and overlapped with the gate/up
// compute + swiglu, shrinking the per-layer blocking fill. Both are async on the upload stream.
// fill_gateup claims the slot (so lookups hit); fill_down only uploads down into an already-claimed
// slot. Bit-identical: same bytes, same slots, just reordered async uploads.
void gpu_expert_cache_fill_gateup(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn;
    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)expert * up_bytes;
    CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
    expert_slot_claim(slot, layer, expert);
}

void gpu_expert_cache_fill_down(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;
    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)expert * down_bytes;
    CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
}

static int select_tail_victim_slot_score(float *out_score) {
    if (out_score) *out_score = 0.0f;
    if (!g_moe_policy || getenv("TUFF_NO_LES")) {
        return g_lru_tail;
    }
    int cands[64];
    int n_cands = 0;
    int cur = g_lru_tail;
    while (cur >= 0 && n_cands < 64) {
        if (g_slots[cur].valid) cands[n_cands++] = cur;
        cur = g_lru_prev[cur];
    }
    if (n_cands == 0) return g_lru_tail;
    int best_i = 0;
    float min_score = 1e30f;
    for (int i = 0; i < n_cands; ++i) {
        int s = cands[i];
        float sc = moe_policy_score(g_moe_policy, (int)g_slots[s].layer, g_slots[s].expert);
        if (sc < min_score) {
            min_score = sc;
            best_i = i;
            if (min_score <= 0.0f) break;
        }
    }
    if (out_score) *out_score = min_score;
    return cands[best_i];
}

static int select_tail_victim_slot(void) {
    return select_tail_victim_slot_score(NULL);
}

// Pure query: check if an expert is currently resident without modifying LRU or counters
int gpu_expert_is_resident(uint32_t layer, int expert) {
    if (!g_cache_ready) return 0;
    if ((size_t)layer * 256 + (size_t)expert >= (size_t)g_total_experts) return 0;
    return g_expert_map[layer * 256 + expert] >= 0;
}

// Reserve a slot for a new expert (miss): free slot first, else policy/LRU victim. Does NOT upload.
// O(1) via freelist + tail candidate selection (Task 6.2 LES probability-driven cache).
int gpu_expert_cache_alloc_slot(uint32_t layer, int expert) {
    (void)layer; (void)expert;
    int victim = freelist_pop();
    if (victim < 0) {
        victim = select_tail_victim_slot();
    }
    return victim;
}

void gpu_expert_cache_stats(unsigned long long *hits, unsigned long long *misses) {
    if (hits) *hits = g_cache_hits;
    if (misses) *misses = g_cache_misses;
}

int gpu_expert_cache_active(void) { return g_cache_enabled(); }

moe_policy_t* gpu_get_moe_policy(void) { return g_moe_policy; }

// Ensure any in-flight async expert uploads are visible to the default stream before the
// engine reads a resident slot (cheap when the upload stream is idle). No-op if not async.
void gpu_expert_cache_sync(void) {
    if (g_cache_async && g_upload_stream) CUDA_CHECK(cudaStreamSynchronize(g_upload_stream));
}

// ---------------------------------------------------------------------------
// P1 (parity.md sec.4): background cache fills.
//
// The old miss policy pushed 1.95 MiB of expert weights over PCIe *from the decode thread*,
// in the critical path, per missing expert. That cost is not only PCIe: the GGUF weights are
// pageable mmap'd memory, so cudaMemcpyAsync stages them through a driver bounce buffer and
// blocks the calling thread while it does. Measured 21.0 ms/token = 35% of the token.
//
// Here the decode thread never copies. It only *reserves* a slot (host bookkeeping, O(1)) and
// hands (layer, expert, slot) to a dedicated fill thread, which does the staging memcpy +
// H2D on g_upload_stream and records an event. The decode thread later polls that event with
// cudaEventQuery (non-blocking) and only then publishes the slot into the lookup map.
//
// Race freedom, by construction:
//   * A reserved slot is removed from BOTH the freelist and the LRU list and its old
//     (layer,expert) key is erased from g_expert_map. So while a fill is in flight the slot is
//     unreachable: no lookup can return it (no kernel can read it) and no later reservation can
//     pick it as a victim.
//   * The slot is published (expert_slot_claim) only after cudaEventQuery reports the copy
//     complete, so any kernel that can subsequently see the slot runs after the copy.
//   * All slot-table / LRU / map mutation stays on the decode thread. The fill thread touches
//     only device memory of slots the decode thread promised not to touch, plus its own stream
//     and events. No lock is needed on the slot table.
//   * Reservations are issued only after the layer's expert kernels have completed
//     (gpu_expert_moe_down / gpu_expert_batch_down_sync_download both sync g_expert_stream),
//     so no in-flight kernel is reading the pool when a victim is chosen.
// ---------------------------------------------------------------------------
// P1 queue depth. Must fit a full token's fills with headroom (2 tokens' worth: up to
// 40 layers x g_fill_budget 2 = 80, plus slack). A power of two since the code indexes with
// `% BGFILL_QUEUE`. The old 8 gated issuance: g_bg_commit only advanced at the once-per-token
// drain, so at most 8 fills could be issued per token (measured 8.00/token) while a token
// routed up to 80 misses -> ~80% of routed experts took the CPU path every token. Deep enough
// that the queue never gates issuance; the commit boundary is now the sole ordering control.
#define BGFILL_QUEUE 256

typedef struct {
    const model_t *m;
    uint32_t layer;
    int      expert;
    int      slot;
    unsigned long token;   // issuing token index, recorded by the decode thread at reserve time
    cudaEvent_t ev;
} bgfill_req_t;

static bgfill_req_t g_bg[BGFILL_QUEUE];
static unsigned long g_bg_prod = 0;    // decode thread only (published with release)
static unsigned long g_bg_sub  = 0;    // fill thread only  (published with release)
static unsigned long g_bg_commit = 0;  // decode thread only
static unsigned long g_bg_token = 0;   // decode thread only: token index of the fills being issued
static pthread_t g_bg_thread;
static pthread_mutex_t g_bg_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_bg_cv  = PTHREAD_COND_INITIALIZER;
static int g_bg_running = 0;
static int g_bg_stop = 0;
static unsigned long long g_bg_issued = 0, g_bg_committed = 0;

static uint8_t *g_pinned_stage = NULL;

static void bgfill_do(const bgfill_req_t *r) {
    const model_layer_t *L = &r->m->layers[r->layer];
    const int dim = (int)r->m->embedding_length;
    const int ffn = (int)r->m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = gate_bytes;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;
    char *base = g_pool + (size_t)r->slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)r->expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)r->expert * up_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)r->expert * down_bytes;

    if (g_pinned_stage) {
        const int idx = (int)(r - g_bg);
        uint8_t *stage = g_pinned_stage + (size_t)idx * g_slot_bytes;
        memcpy(stage + g_tensor_off[0], gate, gate_bytes);
        memcpy(stage + g_tensor_off[1], up,   up_bytes);
        memcpy(stage + g_tensor_off[2], down, down_bytes);
        CUDA_CHECK(cudaMemcpyAsync(base, stage, g_slot_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    }
    CUDA_CHECK(cudaEventRecord(r->ev, g_upload_stream));
}

static void *bgfill_thread_fn(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_bg_mtx);
        while (!g_bg_stop && g_bg_sub == __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE))
            pthread_cond_wait(&g_bg_cv, &g_bg_mtx);
        const int stop = g_bg_stop && g_bg_sub == __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE);
        pthread_mutex_unlock(&g_bg_mtx);
        if (stop) break;
        bgfill_do(&g_bg[g_bg_sub % BGFILL_QUEUE]);
        __atomic_store_n(&g_bg_sub, g_bg_sub + 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void bgfill_start(void) {
    if (g_bg_running) return;
    if (!g_pinned_stage && g_slot_bytes > 0) {
        CUDA_CHECK(cudaMallocHost((void**)&g_pinned_stage, (size_t)BGFILL_QUEUE * g_slot_bytes));
    }
    for (int i = 0; i < BGFILL_QUEUE; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_bg[i].ev, cudaEventDisableTiming));
    }
    g_bg_prod = g_bg_sub = g_bg_commit = g_bg_token = 0;
    g_bg_stop = 0;
    if (pthread_create(&g_bg_thread, NULL, bgfill_thread_fn, NULL) != 0) {
        fprintf(stderr, "[FATAL ERROR] gpu: failed to create the background expert-fill thread\n");
        exit(EXIT_FAILURE);
    }
    g_bg_running = 1;
}

static void bgfill_stop(void) {
    if (!g_bg_running) return;
    pthread_mutex_lock(&g_bg_mtx);
    g_bg_stop = 1;
    pthread_cond_broadcast(&g_bg_cv);
    pthread_mutex_unlock(&g_bg_mtx);
    pthread_join(g_bg_thread, NULL);
    // Drain: everything the fill thread submitted has been recorded on g_upload_stream.
    if (g_upload_stream) CUDA_CHECK(cudaStreamSynchronize(g_upload_stream));
    for (int i = 0; i < BGFILL_QUEUE; ++i) {
        if (g_bg[i].ev) { CUDA_CHECK(cudaEventDestroy(g_bg[i].ev)); g_bg[i].ev = 0; }
    }
    g_bg_running = 0;
}

// Reserve a victim slot and make it unreachable (see the race-freedom note above). Decode
// thread only. Returns -1 when no slot can be spared (every slot is reserved in flight).
static int expert_slot_reserve(void) {
    int slot = freelist_pop();
    if (slot < 0) {
        slot = select_tail_victim_slot();
        if (slot < 0) return -1;   // all slots reserved in flight
        lru_remove(slot);
    }
    expert_slot_meta_t *sl = &g_slots[slot];
    if (sl->valid) {
        g_expert_map[sl->layer * 256 + sl->expert] = -1;
        sl->valid = 0;
        g_valid_count--;
    }
    return slot;
}

int gpu_expert_cache_bgfill(const model_t *m, uint32_t layer, int expert) {
    if (!g_cache_ready || !g_bg_running) return 0;
    if (g_bg_prod - g_bg_commit >= BGFILL_QUEUE) return 0;   // queue full -> skip this fill
    const int slot = expert_slot_reserve();
    if (slot < 0) return 0;
    bgfill_req_t *r = &g_bg[g_bg_prod % BGFILL_QUEUE];
    r->m = m; r->layer = layer; r->expert = expert; r->slot = slot;
    r->token = g_bg_token;   // deterministic function of token history, not of copy latency
    pthread_mutex_lock(&g_bg_mtx);
    __atomic_store_n(&g_bg_prod, g_bg_prod + 1, __ATOMIC_RELEASE);
    pthread_cond_signal(&g_bg_cv);
    pthread_mutex_unlock(&g_bg_mtx);
    g_bg_issued++;
    return 1;
}

static int expert_is_resident_cb(void *ctx, int layer, int expert) {
    (void)ctx;
    return gpu_expert_is_resident((uint32_t)layer, expert);
}

static int bgfill_is_in_flight(uint32_t layer, int expert) {
    const unsigned long prod = __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE);
    for (unsigned long i = g_bg_commit; i < prod; ++i) {
        const bgfill_req_t *r = &g_bg[i % BGFILL_QUEUE];
        if (r->layer == layer && r->expert == expert) return 1;
    }
    return 0;
}

// Advanced LES: Long-Lead Asynchronous Inter-Token Expert Prefetching
// Schedules high-probability non-resident experts over PCIe on g_upload_stream
// during the inter-token / dense execution window.
int gpu_expert_cache_les_prefetch(const model_t *m) {
    if (!g_cache_ready || !g_bg_running || !g_moe_policy) return 0;
    // Opt-in: corpus A/B measured the inter-token prefetch at -2.3% (52.92 vs 54.12 tok/s), the
    // extra PCIe fills contend with the decode path's own traffic (same finding as iteration 37).
    static int enabled = -1;
    if (enabled < 0) { const char *e = getenv("TUFF_LES_PREFETCH"); enabled = (e && *e) ? atoi(e) != 0 : 0; }
    if (!enabled || getenv("TUFF_NO_LES")) return 0;

    // Do not queue prefetches if queue is half full
    if (g_bg_prod - g_bg_commit >= BGFILL_QUEUE / 2) return 0;

    const char *e_budget = getenv("TUFF_LES_PREFETCH_BUDGET");
    int budget = (e_budget && *e_budget) ? atoi(e_budget) : 2;
    if (budget <= 0) return 0;
    if (budget > 8) budget = 8;

    const char *e_thresh = getenv("TUFF_LES_THRESHOLD");
    float min_thresh = (e_thresh && *e_thresh) ? (float)atof(e_thresh) : 0.005f;

    int plan_layers[8];
    int plan_experts[8];
    int n_planned = moe_policy_prefetch_plan(g_moe_policy, budget, min_thresh,
                                             expert_is_resident_cb, NULL,
                                             plan_layers, plan_experts);
    if (n_planned <= 0) return 0;

    int issued = 0;
    for (int p = 0; p < n_planned; ++p) {
        int l = plan_layers[p];
        int e = plan_experts[p];
        if (bgfill_is_in_flight((uint32_t)l, e)) continue;

        float score_new = moe_policy_score(g_moe_policy, l, e);
        float score_victim = 0.0f;
        int victim_slot = -1;
        if (g_free_head < 0) {
            victim_slot = select_tail_victim_slot_score(&score_victim);
        }

        // Only evict if incoming score beats victim score by threshold
        if (victim_slot >= 0 && score_new <= score_victim + min_thresh) {
            continue;
        }

        if (gpu_expert_cache_bgfill(m, (uint32_t)l, e)) {
            issued++;
        }
    }
    return issued;
}

void gpu_expert_cache_bgfill_poll(void) {
    if (!g_bg_running) return;
    const unsigned long t = g_bg_token;
    const unsigned long prod = __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE);
    while (g_bg_commit < prod) {
        bgfill_req_t *r = &g_bg[g_bg_commit % BGFILL_QUEUE];
        if (r->token > t) break;
        while (g_bg_commit >= __atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE))
            sched_yield();
        CUDA_CHECK(cudaEventSynchronize(r->ev));
        expert_slot_claim(r->slot, r->layer, r->expert);
        g_bg_commit++;
        g_bg_committed++;
    }
    g_bg_token = t + 1;
}

void gpu_expert_cache_bgfill_poll_ready(void) {
    if (!g_bg_running) return;
    const unsigned long sub = __atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE);
    while (g_bg_commit < sub) {
        bgfill_req_t *r = &g_bg[g_bg_commit % BGFILL_QUEUE];
        cudaError_t st = cudaEventQuery(r->ev);
        if (st == cudaSuccess) {
            expert_slot_claim(r->slot, r->layer, r->expert);
            g_bg_commit++;
            g_bg_committed++;
        } else if (st == cudaErrorNotReady) {
            break;
        } else {
            CUDA_CHECK(st);
        }
    }
}

void gpu_expert_cache_bgfill_stats(unsigned long long *issued, unsigned long long *committed) {
    if (issued) *issued = g_bg_issued;
    if (committed) *committed = g_bg_committed;
}

// Debug: read back a slot tensor region (which: 0=gate 1=up 2=down) and count bytes differing
// from a host reference. Returns the mismatch count, or -1 on invalid slot.
int gpu_expert_debug_check_down(int slot, const void *host_ref, size_t nbytes) {
    if (!g_cache_ready || slot < 0) return -1;
    if (!g_cache_ready || slot < 0) return -1;
    unsigned char *buf = (unsigned char*)malloc(nbytes);
    CUDA_CHECK(cudaMemcpy(buf, gpu_expert_dev_ptr(slot, 2), nbytes, cudaMemcpyDeviceToHost));
    const unsigned char *ref = (const unsigned char*)host_ref;
    int ndiff = 0;
    for (size_t i = 0; i < nbytes; ++i) if (buf[i] != ref[i]) ndiff++;
    free(buf);
    return ndiff;
}

