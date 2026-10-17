#include "state.h"

#define EXPERT_CACHE_DEFAULT_SLOTS 2500

// Iteration-12 static expert split count (layers whose full expert sets are resident in VRAM).
int g_n_gpu_expert_layers = 0;

typedef struct {
    uint32_t layer;      // owning layer (fixed for the life of the pool)
    int      expert;     // resident expert while valid; the incoming expert while a fill is reserved
    uint8_t  valid;
    uint64_t heat;
    uint64_t last_used;
    uint64_t fill_token; // logical token that reserved the in-flight fill
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

// ---------------------------------------------------------------------------
// Per-layer partitioned cache (lossless speculation, 2026-10).
//
// Every slot is owned by one layer for the life of the pool, and every layer has its own
// LRU order, its own free list and its own queue of reserved-but-unclaimed fills. Victim
// selection (LES score over the layer's LRU tail) only ever looks at the layer's own slots.
// The point is order independence: the cache state a lookup at (token t, layer l) observes is
// a function of the operations at layer l alone, in token order. A verify pass that evaluates
// tokens t and t+1 layer by layer (both columns at layer l before either at l+1) therefore
// sees, at every layer, exactly the hit/miss sets a token-by-token decode would have seen, and
// the speculative output is bit-identical to plain decode. The old single global LRU coupled
// the layers (token t's fill at layer 30 could evict a layer-5 expert that token t+1 needs).
//
// Fill lifecycle, per slot:
//   reserve  (decode thread, logical): victim chosen, unmapped, removed from LRU; the slot is
//            appended to the layer's pending list with fill_token = current logical token.
//   push     (decode thread, physical): queued on the fill thread, which stages the weights
//            and issues the H2D after `ev_dep` (recorded on the compute stream at push time,
//            so the copy cannot overwrite a slot an in-flight expert kernel still reads).
//   claim    (lazily, at the first access to the layer by a later logical token): the slot is
//            mapped and pushed MRU, exactly where the old token-boundary commit put it.
//   ready    the copy's event. A kernel that reads a claimed-but-not-yet-ready slot makes the
//            compute stream wait on that event (GPU-side; the host never blocks on PCIe).
// ---------------------------------------------------------------------------
#define LAYER_PEND_MAX 64

typedef struct {
    int  cap;             // slots owned
    int *owned;           // [cap] slot ids owned by this layer
    int *order;           // [cap] valid slots, MRU first
    int  n_order;
    int *freel;           // [cap] owned slots holding nothing
    int  n_free;
    int  pend[LAYER_PEND_MAX];   // reserved fills not yet claimed, in reservation order
    int  n_pend;
    uint64_t begun_tok;   // logical token whose claims have been applied (+1; 0 = never)
} layer_cache_t;

static layer_cache_t *g_lc = NULL;
static int g_n_lc = 0;
static uint64_t g_ctok = 0;            // logical token of the current cache operations

// Physical fill state per slot, kept OUT of the logical snapshot (a pass may push column 0's
// fills after column 1's snapshot was taken).
static uint64_t *g_slot_ready = NULL;  // [n_slots] fill seq + 1 whose completion makes data valid; 0 = ready
static int g_prefill_per_layer = 0;

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

    // Per-layer partition. The trunk layers (the MTP head's layer gets nothing: it is only
    // looked up by --spec-mtp drafting, and a slot parked there was a slot lost to the trunk).
    g_n_lc = (int)m->block_count;
    g_lc = (layer_cache_t*)calloc((size_t)g_n_lc, sizeof(layer_cache_t));
    g_slot_ready = (uint64_t*)calloc((size_t)g_n_slots, sizeof(uint64_t));
    {
        const int n_trunk = (int)m->block_count - (int)m->nextn_predict_layers;
        const int n_res_layers = g_n_gpu_expert_layers < n_trunk ? g_n_gpu_expert_layers : n_trunk;
        const int n_res_slots = (g_n_gpu_expert_layers > 0) ? (g_n_gpu_expert_layers * (int)m->expert_count) : 0;
        int *owner = (int*)malloc((size_t)g_n_slots * sizeof(int));
        for (int i = 0; i < g_n_slots; ++i) owner[i] = -1;
        for (int i = 0; i < n_res_slots && i < g_n_slots; ++i) owner[i] = i / (int)m->expert_count;
        const int n_dyn_layers = n_trunk - n_res_layers;
        if (n_dyn_layers > 0) {
            const int n_dyn = g_n_slots - n_res_slots;
            // Prefill keeps the old even split over ALL block_count layers (so a frozen cache,
            // TUFF_FILL_BUDGET=0, is resident-identical to the global-LRU build: the golden gate
            // holds). The MTP layer's share and the remainder become free slots of the trunk
            // layers, round-robin.
            const int per = n_dyn / (int)m->block_count;
            g_prefill_per_layer = per;
            int s = n_res_slots;
            for (int l = n_res_layers; l < n_trunk; ++l)
                for (int e = 0; e < per && s < g_n_slots; ++e) owner[s++] = l;
            for (int l = n_res_layers; s < g_n_slots; l = (l + 1 < n_trunk) ? l + 1 : n_res_layers) owner[s++] = l;
        }
        for (int i = 0; i < g_n_slots; ++i) if (owner[i] >= 0) g_lc[owner[i]].cap++;
        for (int l = 0; l < g_n_lc; ++l) {
            layer_cache_t *lc = &g_lc[l];
            const int c = lc->cap > 0 ? lc->cap : 1;
            lc->owned = (int*)malloc((size_t)c * sizeof(int));
            lc->order = (int*)malloc((size_t)c * sizeof(int));
            lc->freel = (int*)malloc((size_t)c * sizeof(int));
            lc->cap = 0; lc->n_order = 0; lc->n_free = 0; lc->n_pend = 0; lc->begun_tok = 0;
        }
        for (int i = 0; i < g_n_slots; ++i) {
            if (owner[i] < 0) continue;
            layer_cache_t *lc = &g_lc[owner[i]];
            lc->owned[lc->cap++] = i;
            g_slots[i].layer = (uint32_t)owner[i];
        }
        // Free lists: dynamic slots only, lowest slot id popped first.
        for (int l = 0; l < g_n_lc; ++l) {
            layer_cache_t *lc = &g_lc[l];
            for (int k = lc->cap - 1; k >= 0; --k)
                if (lc->owned[k] >= n_res_slots) lc->freel[lc->n_free++] = lc->owned[k];
        }
        free(owner);
    }
    g_ctok = 0;

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
        // Predictive startup prefill: every layer's owned slots get experts 0..cap-1 (the
        // even split of the old global prefill, now over the trunk layers only).
        int n_filled = 0, n_layers = 0, per_min = 1 << 30;
        for (int l = 0; l < g_n_lc; ++l) {
            layer_cache_t *lc = &g_lc[l];
            if (lc->cap == 0) continue;
            n_layers++;
            if (lc->cap < per_min) per_min = lc->cap;
            int e = 0;
            while (lc->n_free > 0 && e < g_prefill_per_layer && e < (int)m->expert_count) {
                const int slot = lc->freel[--lc->n_free];
                gpu_expert_cache_fill_pinned(slot, m, (uint32_t)l, e++);
                lru_push_front(slot);
                n_filled++;
            }
        }
        gpu_expert_cache_sync();
        fprintf(stderr, "[EXPERT-CACHE] pre-filled %d dynamic slots across %d layers (>=%d per layer, per-layer LRU, %.2f MiB)\n",
                n_filled, n_layers, n_layers ? per_min : 0, (double)n_filled * g_slot_bytes / 1048576.0);
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
    for (int l = 0; l < g_n_lc; ++l) { free(g_lc[l].owned); free(g_lc[l].order); free(g_lc[l].freel); }
    free(g_lc); g_lc = NULL; g_n_lc = 0;
    free(g_slot_ready); g_slot_ready = NULL;

    g_n_slots = 0; g_slot_bytes = 0; g_cache_ready = 0;
    free(g_expert_hist); g_expert_hist = NULL; g_expert_hist_on = 0;
    if (g_moe_policy) { moe_policy_free(g_moe_policy); g_moe_policy = NULL; }
}

// --- Per-layer LRU + freelist bookkeeping ---
// order[] is MRU first. A layer owns ~48 slots, so the O(cap) move-to-front is a few dozen ints.
static void lru_remove(int slot) {
    layer_cache_t *lc = &g_lc[g_slots[slot].layer];
    for (int i = 0; i < lc->n_order; ++i) {
        if (lc->order[i] == slot) {
            memmove(lc->order + i, lc->order + i + 1, (size_t)(lc->n_order - i - 1) * sizeof(int));
            lc->n_order--;
            return;
        }
    }
}

static void lru_push_front(int slot) {
    layer_cache_t *lc = &g_lc[g_slots[slot].layer];
    memmove(lc->order + 1, lc->order, (size_t)lc->n_order * sizeof(int));
    lc->order[0] = slot;
    lc->n_order++;
}

static int freelist_pop(uint32_t layer) {
    layer_cache_t *lc = &g_lc[layer];
    return lc->n_free > 0 ? lc->freel[--lc->n_free] : -1;
}

static void expert_slot_claim(int slot, uint32_t layer, int expert) {
    expert_slot_meta_t *sl = &g_slots[slot];
    if (sl->valid) { g_expert_map[sl->layer * 256 + sl->expert] = -1; lru_remove(slot); }
    sl->valid = 1;
    sl->layer = layer;
    sl->expert = expert;
    sl->heat = 1;
    sl->last_used = ++g_tick;
    g_expert_map[layer * 256 + expert] = slot;
    lru_push_front(slot);
}

static unsigned long long g_bg_committed = 0;

// Apply the claims every earlier logical token owes this layer. Lazy, but the observable state is
// the old token-boundary commit's: claims land in reservation order, before the first access of a
// later token, and nothing else touches the layer in between.
// TUFF_FILL_LAG (default 0): a fill reserved at token t is claimed by token t + 1 + lag. With
// lag 0 it is the old next-token commit; with lag 1 a copy always has a whole token to land, and in
// a two-token verify column 1 never needs a fill column 0 issued in the same pass (that copy would
// otherwise queue behind every upload already on the copy engine, and stall the stream).
static uint64_t fill_lag(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TUFF_FILL_LAG"); v = (e && *e) ? atoi(e) : 0; if (v < 0) v = 0; }
    return (uint64_t)v;
}

static void layer_begin(uint32_t layer) {
    layer_cache_t *lc = &g_lc[layer];
    if (lc->begun_tok == g_ctok + 1) return;
    const uint64_t lag = fill_lag();
    int w = 0;
    for (int i = 0; i < lc->n_pend; ++i) {
        const int s = lc->pend[i];
        if (g_slots[s].fill_token + lag < g_ctok) {
            expert_slot_claim(s, layer, g_slots[s].expert);
            g_bg_committed++;
        } else {
            lc->pend[w++] = s;
        }
    }
    lc->n_pend = w;
    lc->begun_tok = g_ctok + 1;
}

void gpu_expert_cache_layer_begin(uint32_t layer) {
    if (!g_cache_ready || (int)layer >= g_n_lc) return;
    layer_begin(layer);
}

void gpu_expert_cache_set_token(uint64_t tok) {
    g_ctok = tok;
    if (g_moe_policy) moe_policy_set_step(g_moe_policy, (int)tok);
}

uint64_t gpu_expert_cache_get_token(void) { return g_ctok; }

void gpu_expert_cache_token_end(void) { gpu_expert_cache_set_token(g_ctok + 1); }

// Look up (layer,expert). Returns slot index on HIT (resident, data on device or arriving), -1 on
// MISS. A HIT moves the slot to the MRU end of its layer's LRU order.
int gpu_expert_lookup(uint32_t layer, int expert) {
    if (!g_cache_ready) return -1;
    layer_begin(layer);
    if (g_expert_hist_on && layer < (uint32_t)g_total_experts / 256)
        g_expert_hist[layer * 256 + expert]++;
    int slot = g_expert_map[layer * 256 + expert];
    if (slot >= 0) {
        g_slots[slot].heat++;
        g_slots[slot].last_used = ++g_tick;
        if (layer >= (uint32_t)g_n_gpu_expert_layers && g_lc[layer].order[0] != slot) {
            lru_remove(slot);
            lru_push_front(slot);
        }
        g_cache_hits++;
    } else {
        g_cache_misses++;
    }
    return slot;
}
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

// LES victim over the layer's own LRU tail (up to 64 candidates, least recent first; the first
// minimum wins, and a zero score ends the search).
static int select_tail_victim_slot_score(uint32_t layer, float *out_score) {
    if (out_score) *out_score = 0.0f;
    layer_cache_t *lc = &g_lc[layer];
    if (lc->n_order == 0) return -1;
    if (!g_moe_policy || getenv("TUFF_NO_LES")) {
        return lc->order[lc->n_order - 1];
    }
    int best = lc->order[lc->n_order - 1];
    float min_score = 1e30f;
    for (int i = lc->n_order - 1, n = 0; i >= 0 && n < 64; --i, ++n) {
        const int s = lc->order[i];
        float sc = moe_policy_score(g_moe_policy, (int)layer, g_slots[s].expert);
        if (sc < min_score) {
            min_score = sc;
            best = s;
            if (min_score <= 0.0f) break;
        }
    }
    if (out_score) *out_score = min_score;
    return best;
}

// Pure query: check if an expert is currently resident without modifying LRU or counters
int gpu_expert_is_resident(uint32_t layer, int expert) {
    if (!g_cache_ready) return 0;
    if ((size_t)layer * 256 + (size_t)expert >= (size_t)g_total_experts) return 0;
    return g_expert_map[layer * 256 + expert] >= 0;
}

// Reserve a slot for a new expert (miss): free slot first, else policy/LRU victim. Does NOT upload.
int gpu_expert_cache_alloc_slot(uint32_t layer, int expert) {
    (void)expert;
    if (!g_cache_ready || (int)layer >= g_n_lc) return -1;
    layer_begin(layer);
    int victim = freelist_pop(layer);
    if (victim < 0) victim = select_tail_victim_slot_score(layer, NULL);
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
// Ring depth. A request's stage buffer and events are reused only after the request at the same
// index (seq - BGFILL_QUEUE) has completed; the producer waits for that, so the queue never drops
// a fill (a drop would make residency depend on copy latency, not on token history).
#define BGFILL_QUEUE 256

typedef struct {
    const model_t *m;
    uint32_t layer;
    int      expert;
    int      slot;
    cudaEvent_t ev;       // copy complete (recorded on g_upload_stream by the fill thread)
    cudaEvent_t ev_dep;   // compute-stream point the copy must not overtake (recorded at push)
} bgfill_req_t;

static bgfill_req_t g_bg[BGFILL_QUEUE];
static unsigned long g_bg_prod = 0;    // decode thread only (published with release)
static unsigned long g_bg_sub  = 0;    // fill thread only  (published with release)
static unsigned long g_bg_done = 0;    // decode thread only: every seq < g_bg_done has completed
static pthread_t g_bg_thread;
static pthread_mutex_t g_bg_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_bg_cv  = PTHREAD_COND_INITIALIZER;
static int g_bg_running = 0;
static int g_bg_stop = 0;
static unsigned long long g_bg_issued = 0;

static uint8_t *g_pinned_stage = NULL;

// Reserved fills not yet handed to the fill thread, tagged with a group (0 = the token being
// decoded, 1.. = a speculative verify column whose fate is not yet known).
#define UNPUSHED_MAX 512
typedef struct { int slot; int group; int expert; } unpushed_t;   // expert: the reservation's, not the slot's
                                                                   // (a later column may re-reserve the slot)
static unpushed_t g_unpushed[UNPUSHED_MAX];
static int g_n_unpushed = 0;

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
        CUDA_CHECK(cudaStreamWaitEvent(g_upload_stream, r->ev_dep, 0));
        CUDA_CHECK(cudaMemcpyAsync(base, stage, g_slot_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaStreamWaitEvent(g_upload_stream, r->ev_dep, 0));
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
        CUDA_CHECK(cudaEventCreateWithFlags(&g_bg[i].ev_dep, cudaEventDisableTiming));
    }
    g_bg_prod = g_bg_sub = g_bg_done = 0;
    g_n_unpushed = 0;
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
        if (g_bg[i].ev_dep) { CUDA_CHECK(cudaEventDestroy(g_bg[i].ev_dep)); g_bg[i].ev_dep = 0; }
    }
    g_bg_running = 0;
}

static unsigned long long g_sw_n = 0, g_sw_gap = 0, g_sw_behind = 0;
static double g_bg_wait_s = 0.0;
static unsigned long long g_bg_wait_n = 0;
static void bg_wait_submitted(unsigned long seq) {
    if (__atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE) > seq) return;
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    while (__atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE) <= seq) sched_yield();
    clock_gettime(CLOCK_MONOTONIC, &b);
    g_bg_wait_s += (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec);
    g_bg_wait_n++;
}
void gpu_expert_cache_wait_stats(double *s, unsigned long long *n) { *s = g_bg_wait_s; *n = g_bg_wait_n; }

// Advance g_bg_done over completed requests (non-blocking).
static void bg_advance_done(void) {
    const unsigned long sub = __atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE);
    while (g_bg_done < sub) {
        cudaError_t st = cudaEventQuery(g_bg[g_bg_done % BGFILL_QUEUE].ev);
        if (st == cudaSuccess) { g_bg_done++; continue; }
        if (st != cudaErrorNotReady) CUDA_CHECK(st);
        break;
    }
}

// Hand one reserved slot to the fill thread.
static void bg_push(const model_t *m, int slot, int expert) {
    const unsigned long seq = g_bg_prod;
    if (seq >= BGFILL_QUEUE && g_bg_done <= seq - BGFILL_QUEUE) {
        bg_advance_done();
        if (g_bg_done <= seq - BGFILL_QUEUE) {
            const unsigned long old = seq - BGFILL_QUEUE;
            bg_wait_submitted(old);
            CUDA_CHECK(cudaEventSynchronize(g_bg[old % BGFILL_QUEUE].ev));
            g_bg_done = old + 1;
            bg_advance_done();
        }
    }
    bgfill_req_t *r = &g_bg[seq % BGFILL_QUEUE];
    r->m = m; r->layer = g_slots[slot].layer; r->expert = expert; r->slot = slot;
    CUDA_CHECK(cudaEventRecord(r->ev_dep, 0));
    g_slot_ready[slot] = (uint64_t)seq + 1;
    pthread_mutex_lock(&g_bg_mtx);
    __atomic_store_n(&g_bg_prod, seq + 1, __ATOMIC_RELEASE);
    pthread_cond_signal(&g_bg_cv);
    pthread_mutex_unlock(&g_bg_mtx);
    g_bg_issued++;
}

// Reserve a victim slot in `layer` and make it unreachable. Decode thread only.
static int expert_slot_reserve(uint32_t layer) {
    int slot = freelist_pop(layer);
    if (slot < 0) {
        slot = select_tail_victim_slot_score(layer, NULL);
        if (slot < 0) return -1;
        lru_remove(slot);
    }
    expert_slot_meta_t *sl = &g_slots[slot];
    if (sl->valid) {
        g_expert_map[sl->layer * 256 + sl->expert] = -1;
        sl->valid = 0;
    }
    return slot;
}

static const model_t *g_fill_model = NULL;
static unsigned long long g_les_rejects = 0;
static float les_admit_margin(void) {
    static float v = -2.0f;
    if (v < -1.5f) { const char *e = getenv("TUFF_LES_ADMIT"); v = (e && *e) ? (float)atof(e) : -1.0f; }
    return v;
}
unsigned long long gpu_expert_cache_les_rejects(void) { return g_les_rejects; }

// Logical half of a fill. group 0 is queued on the layer's pending list at once; group >= 1
// (a speculative verify column) is not, until gpu_expert_cache_fill_commit(group) accepts it.
// Returns 1 if a slot was reserved.
int gpu_expert_cache_fill_reserve(const model_t *m, uint32_t layer, int expert, int group) {
    if (!g_cache_ready || !g_bg_running || (int)layer >= g_n_lc) return 0;
    layer_begin(layer);
    layer_cache_t *lc = &g_lc[layer];
    if (lc->n_pend >= LAYER_PEND_MAX || g_n_unpushed >= UNPUSHED_MAX) return 0;
    // LES admission (TUFF_LES_ADMIT=<margin>, off when unset): an eviction is a 1.95 MiB PCIe copy,
    // so only make it when the incoming expert's LES score beats the victim's by the margin. Both
    // scores are this layer's own (decayed routing frequency), so the decision stays a function of
    // the layer's token history and the speculative verify remains lossless.
    if (les_admit_margin() >= 0.0f && g_moe_policy && lc->n_free == 0) {
        float sv = 0.0f;
        if (select_tail_victim_slot_score(layer, &sv) >= 0 &&
            moe_policy_score(g_moe_policy, (int)layer, expert) <= sv + les_admit_margin()) {
            g_les_rejects++;
            return 0;
        }
    }
    const int slot = expert_slot_reserve(layer);
    if (slot < 0) return 0;
    g_fill_model = m;
    g_slots[slot].expert = expert;
    g_slots[slot].fill_token = g_ctok;
    if (group == 0) lc->pend[lc->n_pend++] = slot;
    g_unpushed[g_n_unpushed].slot = slot;
    g_unpushed[g_n_unpushed].group = group;
    g_unpushed[g_n_unpushed].expert = expert;
    g_n_unpushed++;
    return 1;
}

// Physical half: hand every reserved slot of `group` to the fill thread, in reservation order.
// The copies are ordered after everything already issued on the compute stream.
void gpu_expert_cache_fill_push(int group) {
    int w = 0;
    for (int i = 0; i < g_n_unpushed; ++i) {
        if (g_unpushed[i].group == group) bg_push(g_fill_model, g_unpushed[i].slot, g_unpushed[i].expert);
        else g_unpushed[w++] = g_unpushed[i];
    }
    g_n_unpushed = w;
}

// A verify column was accepted: its reserved fills join their layers' pending lists (in order)
// and are pushed.
void gpu_expert_cache_fill_commit(int group) {
    for (int i = 0; i < g_n_unpushed; ++i) {
        if (g_unpushed[i].group != group) continue;
        const int s = g_unpushed[i].slot;
        layer_cache_t *lc = &g_lc[g_slots[s].layer];
        if (lc->n_pend < LAYER_PEND_MAX) lc->pend[lc->n_pend++] = s;
    }
    gpu_expert_cache_fill_push(group);
}

// A verify column was rejected: forget its reserved fills (the caller restores the layers'
// logical state, which puts the victims back; their device data was never touched).
void gpu_expert_cache_fill_drop(int group) {
    int w = 0;
    for (int i = 0; i < g_n_unpushed; ++i) {
        if (g_unpushed[i].group != group) g_unpushed[w++] = g_unpushed[i];
    }
    g_n_unpushed = w;
}

int gpu_expert_cache_bgfill(const model_t *m, uint32_t layer, int expert) {
    if (!gpu_expert_cache_fill_reserve(m, layer, expert, 0)) return 0;
    gpu_expert_cache_fill_push(0);
    return 1;
}

// Make the compute stream wait for any of `slots` whose last ISSUED fill has not been observed
// complete. Copies complete in issue order on the one upload stream, so waiting on the newest
// suffices. A slot whose replacement is only reserved (not yet issued) still holds - and is being
// read for - its previous contents; the replacement copy is ordered after the read by ev_dep.
void gpu_expert_cache_wait_slots(const int *slots, int n) {
    if (!g_cache_ready) return;
    uint64_t need = 0;
    for (int i = 0; i < n; ++i) {
        const int s = slots[i];
        if (s >= 0 && g_slot_ready[s] > need) need = g_slot_ready[s];
    }
    if (need == 0 || need <= g_bg_done) return;
    bg_advance_done();
    if (need <= g_bg_done) return;
    const unsigned long seq = (unsigned long)(need - 1);
    bg_wait_submitted(seq);
    g_sw_n++;
    g_sw_gap += (unsigned long long)(g_bg_prod - g_bg_done);
    g_sw_behind += (unsigned long long)(need - g_bg_done);
    CUDA_CHECK(cudaStreamWaitEvent(0, g_bg[seq % BGFILL_QUEUE].ev, 0));
}
void gpu_expert_cache_streamwait_stats(unsigned long long *n, unsigned long long *gap, unsigned long long *behind) {
    *n = g_sw_n; *gap = g_sw_gap; *behind = g_sw_behind;
}

// --- Logical snapshot of one layer (a speculative column's cache operations are undone if the
// column is rejected). Physical fill state (g_slot_ready / unpushed) is deliberately excluded.
struct expert_layer_snap {
    int n_order, n_free, n_pend;
    uint64_t begun_tok;
    int *order, *freel, *pend;
    int map_row[256];
    expert_slot_meta_t *meta;   // [cap], in owned[] order
    unsigned char *pol;         // policy row
    int cap;
};

expert_layer_snap_t *gpu_expert_cache_snap_alloc(void) {
    if (!g_cache_ready) return NULL;
    expert_layer_snap_t *sn = (expert_layer_snap_t*)calloc((size_t)g_n_lc, sizeof(expert_layer_snap_t));
    const size_t pb = g_moe_policy ? moe_policy_row_bytes(g_moe_policy) : 0;
    for (int l = 0; l < g_n_lc; ++l) {
        const int c = g_lc[l].cap > 0 ? g_lc[l].cap : 1;
        sn[l].cap = g_lc[l].cap;
        sn[l].order = (int*)malloc((size_t)c * sizeof(int));
        sn[l].freel = (int*)malloc((size_t)c * sizeof(int));
        sn[l].pend  = (int*)malloc(LAYER_PEND_MAX * sizeof(int));
        sn[l].meta  = (expert_slot_meta_t*)malloc((size_t)c * sizeof(expert_slot_meta_t));
        sn[l].pol   = pb ? (unsigned char*)malloc(pb) : NULL;
    }
    return sn;
}

void gpu_expert_cache_snap_layer(expert_layer_snap_t *sn, uint32_t layer) {
    if (!sn || (int)layer >= g_n_lc) return;
    layer_cache_t *lc = &g_lc[layer];
    expert_layer_snap_t *s = &sn[layer];
    s->n_order = lc->n_order; s->n_free = lc->n_free; s->n_pend = lc->n_pend; s->begun_tok = lc->begun_tok;
    memcpy(s->order, lc->order, (size_t)lc->n_order * sizeof(int));
    memcpy(s->freel, lc->freel, (size_t)lc->n_free * sizeof(int));
    memcpy(s->pend, lc->pend, (size_t)lc->n_pend * sizeof(int));
    memcpy(s->map_row, g_expert_map + (size_t)layer * 256, sizeof(s->map_row));
    for (int i = 0; i < lc->cap; ++i) s->meta[i] = g_slots[lc->owned[i]];
    if (s->pol) moe_policy_row_save(g_moe_policy, (int)layer, s->pol);
}

void gpu_expert_cache_restore_layer(const expert_layer_snap_t *sn, uint32_t layer) {
    if (!sn || (int)layer >= g_n_lc) return;
    layer_cache_t *lc = &g_lc[layer];
    const expert_layer_snap_t *s = &sn[layer];
    lc->n_order = s->n_order; lc->n_free = s->n_free; lc->n_pend = s->n_pend; lc->begun_tok = s->begun_tok;
    memcpy(lc->order, s->order, (size_t)s->n_order * sizeof(int));
    memcpy(lc->freel, s->freel, (size_t)s->n_free * sizeof(int));
    memcpy(lc->pend, s->pend, (size_t)s->n_pend * sizeof(int));
    memcpy(g_expert_map + (size_t)layer * 256, s->map_row, sizeof(s->map_row));
    for (int i = 0; i < lc->cap; ++i) g_slots[lc->owned[i]] = s->meta[i];
    if (s->pol) moe_policy_row_restore(g_moe_policy, (int)layer, s->pol);
}

void gpu_expert_cache_snap_free(expert_layer_snap_t *sn) {
    if (!sn) return;
    for (int l = 0; l < g_n_lc; ++l) {
        free(sn[l].order); free(sn[l].freel); free(sn[l].pend); free(sn[l].meta); free(sn[l].pol);
    }
    free(sn);
}

static int expert_is_resident_cb(void *ctx, int layer, int expert) {
    (void)ctx;
    return gpu_expert_is_resident((uint32_t)layer, expert);
}

static int bgfill_is_in_flight(uint32_t layer, int expert) {
    const layer_cache_t *lc = &g_lc[layer];
    for (int i = 0; i < lc->n_pend; ++i)
        if (g_slots[lc->pend[i]].expert == expert) return 1;
    return 0;
}

// Advanced LES: Long-Lead Asynchronous Inter-Token Expert Prefetching
// Schedules high-probability non-resident experts over PCIe on g_upload_stream
// during the inter-token / dense execution window.
int gpu_expert_cache_les_prefetch(const model_t *m) {
    if (!g_cache_ready || !g_bg_running || !g_moe_policy) return 0;
    // Opt-in: corpus A/B measured the inter-token prefetch at -2.3% (52.92 vs 54.12 tok/s), the
    // extra PCIe fills contend with the decode path's own traffic (same finding as iteration 37).
    // Not part of the speculative-verify trajectory: with spec on it makes the output depend on
    // where the verify steps fall.
    static int enabled = -1;
    if (enabled < 0) { const char *e = getenv("TUFF_LES_PREFETCH"); enabled = (e && *e) ? atoi(e) != 0 : 0; }
    if (!enabled || getenv("TUFF_NO_LES")) return 0;

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
        if (l >= g_n_lc || g_lc[l].cap == 0) continue;
        layer_begin((uint32_t)l);
        if (bgfill_is_in_flight((uint32_t)l, e)) continue;

        float score_new = moe_policy_score(g_moe_policy, l, e);
        float score_victim = 0.0f;
        int victim_slot = -1;
        if (g_lc[l].n_free == 0) {
            victim_slot = select_tail_victim_slot_score((uint32_t)l, &score_victim);
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

// Completion bookkeeping only (claims are lazy, per layer; readiness is GPU-side).
void gpu_expert_cache_bgfill_poll(void) {
    if (!g_bg_running) return;
    bg_advance_done();
}

void gpu_expert_cache_bgfill_poll_ready(void) {
    gpu_expert_cache_bgfill_poll();
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

