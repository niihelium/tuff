#include "lfm25_expert_cache.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct lfm25_expert_cache {
    lfm25_model_t *model;
    int n_layers, n_experts;
    int n_slots;
    size_t gate_slot, up_slot, down_slot;
    char *d_gate, *d_up, *d_down;

    int      *map;        // [n_layers * n_experts] -> slot or -1
    int      *slot_key;   // [n_slots] -> layer * n_experts + expert, or -1
    uint64_t *slot_use;   // [n_slots] last-use counter (LRU). TODO(roadmap): frequency-aware LFRU.
    uint64_t  tick;

    int pinned;           // 1 if mmap regions were registered
    void    **reg_ptr;
    int       n_reg;

    uint64_t hits, misses;
    uint64_t up_bytes;
    double   up_sec;
    int      timing;
    cudaEvent_t ev0, ev1;
};

static size_t env_mb(const char *name, size_t def_mb) {
    const char *e = getenv(name);
    if (!e || !*e) return def_mb;
    return (size_t)strtoull(e, NULL, 10);
}

int lfm25_expert_cache_wanted(const lfm25_model_t *m) {
    if (!m || !m->is_moe) return 0;
    const char *f = getenv("LFM25_FORCE_EXPERT_CACHE");
    return m->expert_count > 32 || (f && f[0] == '1');
}

typedef struct { uintptr_t lo, hi; } range_t;
static int range_cmp(const void *a, const void *b) {
    uintptr_t x = ((const range_t*)a)->lo, y = ((const range_t*)b)->lo;
    return x < y ? -1 : x > y;
}

// Register the page-aligned union of all expert tensor regions (read-only) for fast DMA.
static void try_register(lfm25_expert_cache_t *c) {
    lfm25_model_t *m = c->model;
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    range_t *r = (range_t*)malloc(sizeof(range_t) * (size_t)m->block_count * 3);
    int n = 0;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        if (m->layers[l].ffn_type != LFM25_FFN_MOE) continue;
        const lfm25_tensor_t *t[3] = { &m->layers[l].ffn_gate_exps, &m->layers[l].ffn_up_exps, &m->layers[l].ffn_down_exps };
        for (int i = 0; i < 3; ++i) {
            uintptr_t lo = (uintptr_t)t[i]->data, hi = lo + t[i]->info.size_bytes;
            r[n].lo = lo & ~(page - 1);
            r[n].hi = (hi + page - 1) & ~(page - 1);
            n++;
        }
    }
    qsort(r, (size_t)n, sizeof(range_t), range_cmp);
    int k = 0;
    for (int i = 1; i < n; ++i) {
        if (r[i].lo <= r[k].hi) { if (r[i].hi > r[k].hi) r[k].hi = r[i].hi; }
        else r[++k] = r[i];
    }
    n = k + 1;
    c->reg_ptr = (void**)calloc((size_t)n, sizeof(void*));
    int ok = 1;
    unsigned flags = cudaHostRegisterReadOnly;
    for (int i = 0; i < n; ++i) {
        cudaError_t e = cudaHostRegister((void*)r[i].lo, r[i].hi - r[i].lo, flags);
        if (e != cudaSuccess && i == 0 && flags == cudaHostRegisterReadOnly) {
            fprintf(stderr, "[LFM25 CACHE] cudaHostRegister(ReadOnly) failed (%s); retrying with default flags\n", cudaGetErrorString(e));
            (void)cudaGetLastError();
            flags = cudaHostRegisterDefault;
            e = cudaHostRegister((void*)r[i].lo, r[i].hi - r[i].lo, flags);
        }
        if (e != cudaSuccess) {
            fprintf(stderr, "[LFM25 CACHE] cudaHostRegister failed (%s); falling back to pageable copies\n", cudaGetErrorString(e));
            (void)cudaGetLastError();
            ok = 0;
            break;
        }
        c->reg_ptr[c->n_reg++] = (void*)r[i].lo;
    }
    if (!ok) {
        for (int i = 0; i < c->n_reg; ++i) cudaHostUnregister(c->reg_ptr[i]);
        c->n_reg = 0;
        c->pinned = 0;
    } else {
        c->pinned = 1;
    }
    free(r);
}

lfm25_expert_cache_t* lfm25_expert_cache_create(lfm25_model_t *m) {
    lfm25_expert_cache_t *c = (lfm25_expert_cache_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->model = m;
    c->n_layers = (int)m->block_count;
    c->n_experts = (int)m->expert_count;

    const int dim = (int)m->embedding_length;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        const lfm25_layer_t *L = &m->layers[l];
        if (L->ffn_type != LFM25_FFN_MOE) continue;
        size_t g = L->ffn_gate_exps.info.size_bytes / m->expert_count;
        size_t u = L->ffn_up_exps.info.size_bytes / m->expert_count;
        size_t d = L->ffn_down_exps.info.size_bytes / m->expert_count;
        if (g > c->gate_slot) c->gate_slot = g;
        if (u > c->up_slot) c->up_slot = u;
        if (d > c->down_slot) c->down_slot = d;
    }
    (void)dim;
    size_t slot_bytes = c->gate_slot + c->up_slot + c->down_slot;
    if (slot_bytes == 0) { fprintf(stderr, "[LFM25 CACHE] ERROR: no MoE layers\n"); free(c); return NULL; }

    size_t budget;
    const char *cap = getenv("LFM25_EXPERT_CACHE_MB");
    if (cap && *cap) {
        budget = env_mb("LFM25_EXPERT_CACHE_MB", 0) << 20;
    } else {
        size_t fr = 0, tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
        size_t reserve = env_mb("LFM25_VRAM_RESERVE_MB", 512) << 20;
        budget = fr > reserve ? fr - reserve : 0;
    }
    size_t max_slots = (size_t)c->n_layers * c->n_experts;
    size_t ns = budget / slot_bytes;
    if (ns > max_slots) ns = max_slots;
    if (ns < 8) {
        fprintf(stderr, "[LFM25 CACHE] ERROR: expert cache budget %zu MiB holds only %zu slots (need >= 8, %.2f MiB each); "
                "set LFM25_EXPERT_CACHE_MB or lower LFM25_VRAM_RESERVE_MB / context\n",
                budget >> 20, ns, slot_bytes / 1048576.0);
        free(c);
        return NULL;
    }
    c->n_slots = (int)ns;
    CUDA_CHECK(cudaMalloc((void**)&c->d_gate, ns * c->gate_slot));
    CUDA_CHECK(cudaMalloc((void**)&c->d_up, ns * c->up_slot));
    CUDA_CHECK(cudaMalloc((void**)&c->d_down, ns * c->down_slot));

    c->map = (int*)malloc(max_slots * sizeof(int));
    for (size_t i = 0; i < max_slots; ++i) c->map[i] = -1;
    c->slot_key = (int*)malloc(ns * sizeof(int));
    c->slot_use = (uint64_t*)calloc(ns, sizeof(uint64_t));
    for (size_t i = 0; i < ns; ++i) c->slot_key[i] = -1;

    const char *st = getenv("LFM25_CACHE_STATS");
    c->timing = st && st[0] == '1';
    if (c->timing) { CUDA_CHECK(cudaEventCreate(&c->ev0)); CUDA_CHECK(cudaEventCreate(&c->ev1)); }

    // cudaHostRegister on the read-only MAP_SHARED GGUF mapping fails on this driver (ReadOnly: "operation not
    // supported", default flags: "invalid argument") and every failed attempt shows up as an API error under
    // compute-sanitizer, so it is opt-in. TODO(Phase 4): pinned staging buffer (or a pinned private mapping)
    // for faster misses; pageable copies measure ~4-6 GB/s.
    const char *reg = getenv("LFM25_TRY_HOST_REGISTER");
    if (reg && reg[0] == '1') try_register(c);
    fprintf(stderr, "[LFM25 CACHE] %d slots (%.0f MiB: gate %zu + up %zu + down %zu B per slot), %zu of %zu expert slabs fit; "
            "host path: %s (%d region%s)\n",
            c->n_slots, (double)(ns * slot_bytes) / 1048576.0, c->gate_slot, c->up_slot, c->down_slot,
            ns, (size_t)m->n_moe_layers * m->expert_count,
            c->pinned ? "PINNED (cudaHostRegister ReadOnly)" : "PAGEABLE", c->n_reg, c->n_reg == 1 ? "" : "s");
    return c;
}

void lfm25_expert_cache_lookup(lfm25_expert_cache_t *c, int layer, const int *ids, int n, int *slots) {
    const lfm25_layer_t *L = &c->model->layers[layer];
    // Pass 1: resolve hits and protect them (stamp with the current tick) so misses can't evict them.
    c->tick++;
    int miss[8], nmiss = 0;
    for (int i = 0; i < n; ++i) {
        int s = c->map[layer * c->n_experts + ids[i]];
        if (s >= 0) { slots[i] = s; c->slot_use[s] = c->tick; c->hits++; }
        else { slots[i] = -1; miss[nmiss++] = i; }
    }
    if (nmiss == 0) return;

    if (c->timing) CUDA_CHECK(cudaEventRecord(c->ev0, 0));
    // Pass 2: fill misses; victim = least recently used slot (never one stamped this tick).
    for (int j = 0; j < nmiss; ++j) {
        int i = miss[j];
        int victim = -1;
        uint64_t best = UINT64_MAX;
        for (int s = 0; s < c->n_slots; ++s) {
            if (c->slot_use[s] == c->tick) continue;
            if (c->slot_use[s] < best) { best = c->slot_use[s]; victim = s; if (best == 0) break; }
        }
        if (victim < 0) { fprintf(stderr, "[LFM25 CACHE] FATAL: no evictable slot\n"); exit(EXIT_FAILURE); }
        if (c->slot_key[victim] >= 0) c->map[c->slot_key[victim]] = -1;
        int key = layer * c->n_experts + ids[i];
        c->slot_key[victim] = key;
        c->map[key] = victim;
        c->slot_use[victim] = c->tick;
        slots[i] = victim;
        c->misses++;

        size_t gb = L->ffn_gate_exps.info.size_bytes / c->n_experts;
        size_t ub = L->ffn_up_exps.info.size_bytes / c->n_experts;
        size_t db = L->ffn_down_exps.info.size_bytes / c->n_experts;
        CUDA_CHECK(cudaMemcpyAsync(c->d_gate + (size_t)victim * c->gate_slot,
                                   (const char*)L->ffn_gate_exps.data + (size_t)ids[i] * gb, gb, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(c->d_up + (size_t)victim * c->up_slot,
                                   (const char*)L->ffn_up_exps.data + (size_t)ids[i] * ub, ub, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(c->d_down + (size_t)victim * c->down_slot,
                                   (const char*)L->ffn_down_exps.data + (size_t)ids[i] * db, db, cudaMemcpyHostToDevice, 0));
        c->up_bytes += gb + ub + db;
    }
    if (c->timing) {
        CUDA_CHECK(cudaEventRecord(c->ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(c->ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, c->ev0, c->ev1));
        c->up_sec += ms * 1e-3;
    }
}

void* lfm25_expert_cache_pool_gate(const lfm25_expert_cache_t *c) { return c->d_gate; }
void* lfm25_expert_cache_pool_up(const lfm25_expert_cache_t *c) { return c->d_up; }
void* lfm25_expert_cache_pool_down(const lfm25_expert_cache_t *c) { return c->d_down; }
size_t lfm25_expert_cache_gate_slot_bytes(const lfm25_expert_cache_t *c) { return c->gate_slot; }
size_t lfm25_expert_cache_up_slot_bytes(const lfm25_expert_cache_t *c) { return c->up_slot; }
size_t lfm25_expert_cache_down_slot_bytes(const lfm25_expert_cache_t *c) { return c->down_slot; }

void lfm25_expert_cache_free(lfm25_expert_cache_t *c) {
    if (!c) return;
    uint64_t tot = c->hits + c->misses;
    fprintf(stderr, "[LFM25 CACHE] stats: %llu lookups, %llu hits, %llu misses, hit rate %.2f%% (%d slots, %s)\n",
            (unsigned long long)tot, (unsigned long long)c->hits, (unsigned long long)c->misses,
            tot ? 100.0 * (double)c->hits / (double)tot : 0.0, c->n_slots, c->pinned ? "pinned" : "pageable");
    if (c->timing && c->up_sec > 0)
        fprintf(stderr, "[LFM25 CACHE] uploaded %.1f MiB in %.3f s -> %.2f GB/s (%s)\n",
                (double)c->up_bytes / 1048576.0, c->up_sec, (double)c->up_bytes / c->up_sec / 1e9,
                c->pinned ? "pinned" : "pageable");
    cudaDeviceSynchronize();
    for (int i = 0; i < c->n_reg; ++i) cudaHostUnregister(c->reg_ptr[i]);
    free(c->reg_ptr);
    if (c->timing) { cudaEventDestroy(c->ev0); cudaEventDestroy(c->ev1); }
    cudaFree(c->d_gate); cudaFree(c->d_up); cudaFree(c->d_down);
    free(c->map); free(c->slot_key); free(c->slot_use);
    free(c);
}
