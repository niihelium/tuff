#include "lfm25_expert_cache.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Expert cache design (Task 4.3)
//  * Slots live in three device pools (gate/up/down) sharing one slot index.
//  * Misses are uploaded on a dedicated non-blocking COPY stream; the compute stream only waits (cudaStreamWaitEvent)
//    for the copies of the slots it is about to read. Each upload batch records one event from a small ring.
//  * Host source of a miss, best first:
//      PINNED STORE   all expert tensors copied once into one pinned (cudaHostRegister'd) buffer (process-wide, kept across the
//                     engine create/free cycles of --runs, freed at exit) -> DMA at pinned speed.
//      STAGING RING   small pinned buffers filled by host memcpy from the mmap and DMA'd on the copy stream, so the
//                     memcpy of one slab overlaps the DMA of the previous one.
//      PAGEABLE       cudaMemcpyAsync straight from the mmap (driver stages internally; the old baseline).
//  * Prefetch (lfm25_expert_cache_prefetch) fills slots for the PREDICTED experts of the next MoE layer. Safety rules:
//      - a prefetch victim is never a slot stamped (slot_prot) with the current tick, i.e. never a slot the
//        in-flight/queued MoE FFN of the current layer reads. Because the engine syncs on the router ids of layer L,
//        the FFN of layer L-1 has finished before layer L's lookup, so FFN(L) is the only compute that can still be
//        reading slots while copies for layer L+1 are being issued.
//      - every slot being filled carries slot_ev (the ring event of its batch); the consumer stream waits on it before
//        any kernel reads the slot, and a re-fill is ordered after the earlier fill by copy-stream order.
//      - the router result of the real layer is authoritative; a wrong prediction only warms the cache.

#define EV_RING 512
#define STAGE_BUFS 8

typedef struct {
    cudaEvent_t start, end;   // start only created in stats mode
    size_t bytes;
    int used;
} ev_ent_t;

typedef struct {
    char *ptr;
    cudaEvent_t done;
    int used;
} stage_t;

// ---- process-wide pinned expert store ---------------------------------------------------------------------------
typedef struct {
    const lfm25_model_t *model;
    char *base;
    size_t bytes;
    size_t *off;        // [n_layers * 3] byte offset of each (layer, gate/up/down) tensor in base
    double load_sec;
    int nthreads;
    int atexit_set;
} store_t;
static store_t g_store;

struct lfm25_expert_cache {
    lfm25_model_t *model;
    int n_layers, n_experts;
    int n_slots;
    size_t gate_slot, up_slot, down_slot;
    char *d_gate, *d_up, *d_down;

    int      *map;        // [n_layers * n_experts] -> slot or -1
    int      *slot_key;   // [n_slots] -> layer * n_experts + expert, or -1
    uint64_t *slot_use;   // [n_slots] tick of last real use / fill (recency)
    uint64_t *slot_prot;  // [n_slots] tick that protects the slot from eviction
    uint32_t *slot_cnt;   // [n_slots] decayed use count (LFRU)
    int      *slot_ev;    // [n_slots] ring index of the upload batch not yet waited on by the consumer, or -1
    uint8_t  *slot_pf;    // [n_slots] 1 = filled by prefetch and not yet used
    uint64_t  tick;

    int policy;           // 0 = LRU, 1 = LFRU
    int recent_pct;       // LFRU: candidates = slots whose recency is in the oldest recent_pct% of the recency range
    uint64_t decay_period, since_decay;

    cudaStream_t copy_stream;
    ev_ent_t ring[EV_RING];
    int ring_head;

    int host_mode;        // 0 pageable, 1 pinned store, 2 staging ring
    stage_t stage[STAGE_BUFS];
    size_t stage_bytes;
    int stage_head;

    // stats
    uint64_t hits, misses;
    uint64_t up_bytes, pf_bytes;
    double   up_sec;
    int      timing;
    uint64_t pf_slabs, pf_hits, pf_hidden, pf_wasted;
    uint64_t pred_total, pred_correct;
    int      pred_layer, pred_ids[8], pred_n;
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

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static size_t mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    }
    fclose(f);
    return kb * 1024;
}

static const lfm25_tensor_t* exps_tensor(const lfm25_layer_t *L, int t) {
    return t == 0 ? &L->ffn_gate_exps : t == 1 ? &L->ffn_up_exps : &L->ffn_down_exps;
}

// ---- pinned store build -----------------------------------------------------------------------------------------
typedef struct { char *dst; const char *src; size_t len; } copy_seg_t;
typedef struct { copy_seg_t *seg; size_t nseg; size_t chunk; atomic_size_t next; size_t ntask; } copy_job_t;

static void* copy_worker(void *arg) {
    copy_job_t *j = (copy_job_t*)arg;
    for (;;) {
        size_t t = atomic_fetch_add(&j->next, 1);
        if (t >= j->ntask) break;
        // task t = (segment index, chunk index) flattened by walking segments (nseg is small: <= 3 * layers)
        size_t rem = t;
        for (size_t s = 0; s < j->nseg; ++s) {
            size_t nc = (j->seg[s].len + j->chunk - 1) / j->chunk;
            if (rem < nc) {
                size_t o = rem * j->chunk, n = j->seg[s].len - o;
                if (n > j->chunk) n = j->chunk;
                memcpy(j->seg[s].dst + o, j->seg[s].src + o, n);
                break;
            }
            rem -= nc;
        }
    }
    return NULL;
}

static void store_release(void) {
    if (g_store.base) { cudaHostUnregister(g_store.base); munmap(g_store.base, g_store.bytes); }
    free(g_store.off);
    memset(&g_store, 0, sizeof(g_store));
}

// Returns 1 if the store for `m` is available (built now or earlier in this process).
static int store_acquire(const lfm25_model_t *m, int force) {
    if (g_store.base && g_store.model == m) return 1;
    if (g_store.base) store_release();

    size_t total = 0;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        if (m->layers[l].ffn_type != LFM25_FFN_MOE) continue;
        for (int t = 0; t < 3; ++t) total += (exps_tensor(&m->layers[l], t)->info.size_bytes + 4095) & ~(size_t)4095;
    }
    size_t avail = mem_available_bytes();
    const size_t margin = (size_t)8 << 30;
    if (!force && avail < total + margin) {
        fprintf(stderr, "[LFM25 CACHE] pinned store skipped: MemAvailable %.1f GiB < experts %.1f GiB + 8 GiB margin\n",
                (double)avail / 1073741824.0, (double)total / 1073741824.0);
        return 0;
    }
    double t0 = now_sec();
    // Anonymous private mapping + parallel first-touch (the copy threads take the page faults in parallel), then
    // cudaHostRegister to pin it. This is ~4x faster than a single cudaHostAlloc of the same size (8 s vs 2 s for
    // 12.9 GiB; cudaHostAlloc pins page by page on one thread) and gives the same pinned DMA speed.
    char *base = (char*)mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[LFM25 CACHE] pinned store: mmap(%.1f GiB) failed; using staging ring\n", (double)total / 1073741824.0);
        return 0;
    }
    double t1 = now_sec();

    size_t nseg = 0;
    copy_seg_t *seg = (copy_seg_t*)malloc(sizeof(copy_seg_t) * (size_t)m->block_count * 3);
    size_t *off = (size_t*)calloc((size_t)m->block_count * 3, sizeof(size_t));
    if (!seg || !off) { free(seg); free(off); munmap(base, total); return 0; }
    size_t pos = 0;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        if (m->layers[l].ffn_type != LFM25_FFN_MOE) continue;
        for (int t = 0; t < 3; ++t) {
            const lfm25_tensor_t *x = exps_tensor(&m->layers[l], t);
            off[l * 3 + t] = pos;
            seg[nseg].dst = base + pos; seg[nseg].src = (const char*)x->data; seg[nseg].len = x->info.size_bytes;
            nseg++;
            pos += (x->info.size_bytes + 4095) & ~(size_t)4095;
        }
    }
    copy_job_t job;
    job.seg = seg; job.nseg = nseg; job.chunk = (size_t)4 << 20;
    atomic_init(&job.next, 0);
    job.ntask = 0;
    for (size_t s = 0; s < nseg; ++s) job.ntask += (seg[s].len + job.chunk - 1) / job.chunk;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int nth = ncpu < 1 ? 1 : ncpu > 16 ? 16 : (int)ncpu;
    pthread_t th[16];
    int started = 0;
    for (int i = 1; i < nth; ++i) {
        if (pthread_create(&th[started], NULL, copy_worker, &job) == 0) started++;
    }
    copy_worker(&job);
    for (int i = 0; i < started; ++i) pthread_join(th[i], NULL);
    free(seg);
    double t2 = now_sec();
    cudaError_t re = cudaHostRegister(base, total, cudaHostRegisterDefault);
    if (re != cudaSuccess) {
        fprintf(stderr, "[LFM25 CACHE] pinned store: cudaHostRegister(%.1f GiB) failed (%s); using staging ring\n",
                (double)total / 1073741824.0, cudaGetErrorString(re));
        (void)cudaGetLastError();
        munmap(base, total); free(off);
        return 0;
    }

    g_store.model = m; g_store.base = base; g_store.bytes = total; g_store.off = off;
    g_store.nthreads = started + 1;
    g_store.load_sec = now_sec() - t0;
    if (!g_store.atexit_set) { atexit(store_release); g_store.atexit_set = 1; }
    fprintf(stderr, "[LFM25 CACHE] pinned expert store: %.2f GiB in %.2f s (mmap %.2f s + parallel copy %.2f s [%d threads, %.2f GB/s] + cudaHostRegister %.2f s)\n",
            (double)total / 1073741824.0, g_store.load_sec, t1 - t0, t2 - t1, g_store.nthreads,
            (double)total / (t2 - t1) / 1e9, now_sec() - t2);
    return 1;
}

// ---- event ring ------------------------------------------------------------------------------------------------
static void ring_drain_entry(lfm25_expert_cache_t *c, ev_ent_t *r) {
    if (!r->used) return;
    if (c->timing) {
        CUDA_CHECK(cudaEventSynchronize(r->end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, r->start, r->end));
        c->up_sec += ms * 1e-3;
    }
    r->used = 0;
}

static int batch_begin(lfm25_expert_cache_t *c) {
    int e = c->ring_head;
    c->ring_head = (c->ring_head + 1) % EV_RING;
    ev_ent_t *r = &c->ring[e];
    ring_drain_entry(c, r);
    r->used = 1;
    r->bytes = 0;
    if (c->timing) CUDA_CHECK(cudaEventRecord(r->start, c->copy_stream));
    return e;
}

static void batch_end(lfm25_expert_cache_t *c, int e) {
    CUDA_CHECK(cudaEventRecord(c->ring[e].end, c->copy_stream));
}

// ---- slot copy -------------------------------------------------------------------------------------------------
static void issue_copy(lfm25_expert_cache_t *c, char *dst, const char *src_mmap, size_t pinned_off, size_t bytes) {
    if (c->host_mode == 1) {
        CUDA_CHECK(cudaMemcpyAsync(dst, g_store.base + pinned_off, bytes, cudaMemcpyHostToDevice, c->copy_stream));
    } else if (c->host_mode == 2) {
        stage_t *s = &c->stage[c->stage_head];
        c->stage_head = (c->stage_head + 1) % STAGE_BUFS;
        if (s->used) CUDA_CHECK(cudaEventSynchronize(s->done));   // previous DMA out of this buffer finished
        memcpy(s->ptr, src_mmap, bytes);
        CUDA_CHECK(cudaMemcpyAsync(dst, s->ptr, bytes, cudaMemcpyHostToDevice, c->copy_stream));
        CUDA_CHECK(cudaEventRecord(s->done, c->copy_stream));
        s->used = 1;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dst, src_mmap, bytes, cudaMemcpyHostToDevice, c->copy_stream));
    }
}

// Pick an eviction victim among slots not protected by the current tick. Returns -1 if none.
static int pick_victim(lfm25_expert_cache_t *c) {
    const int n = c->n_slots;
    int victim = -1;
    if (c->policy == 0) {
        uint64_t best = UINT64_MAX;
        for (int s = 0; s < n; ++s) {
            if (c->slot_prot[s] == c->tick) continue;
            if (c->slot_use[s] < best) { best = c->slot_use[s]; victim = s; if (best == 0) break; }
        }
        return victim;
    }
    // LFRU: restrict to the least recently used half (by recency range midpoint), pick the lowest decayed use count
    // there (ties -> older). Empty slots (never filled) win immediately. O(n_slots).
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int s = 0; s < n; ++s) {
        if (c->slot_prot[s] == c->tick) continue;
        if (c->slot_key[s] < 0) return s;
        if (c->slot_use[s] < lo) lo = c->slot_use[s];
        if (c->slot_use[s] > hi) hi = c->slot_use[s];
    }
    if (lo == UINT64_MAX) return -1;
    const uint64_t mid = lo + (uint64_t)((double)(hi - lo) * (double)c->recent_pct / 100.0);
    uint32_t bc = UINT32_MAX;
    uint64_t bu = UINT64_MAX;
    for (int s = 0; s < n; ++s) {
        if (c->slot_prot[s] == c->tick || c->slot_use[s] > mid) continue;
        if (c->slot_cnt[s] < bc || (c->slot_cnt[s] == bc && c->slot_use[s] < bu)) {
            bc = c->slot_cnt[s]; bu = c->slot_use[s]; victim = s;
        }
    }
    return victim;
}

// Assign `key` a slot (evicting a victim) and enqueue its three uploads on the copy stream inside batch `ev`.
static int fill_slot(lfm25_expert_cache_t *c, int layer, int expert, int ev, int is_prefetch) {
    int victim = pick_victim(c);
    if (victim < 0) return -1;
    if (c->slot_key[victim] >= 0) {
        c->map[c->slot_key[victim]] = -1;
        if (c->slot_pf[victim]) c->pf_wasted++;
    }
    int key = layer * c->n_experts + expert;
    c->slot_key[victim] = key;
    c->map[key] = victim;
    // A prefetched slot gets the OLDEST recency (and count 0): if the prediction was wrong it is the first victim of the
    // next miss instead of polluting the cache; a real use promotes it (tick, count 1).
    c->slot_use[victim] = is_prefetch ? 0 : c->tick;
    c->slot_prot[victim] = c->tick;
    c->slot_cnt[victim] = is_prefetch ? 0 : 1;
    c->slot_pf[victim] = (uint8_t)is_prefetch;
    c->slot_ev[victim] = ev;

    const lfm25_layer_t *L = &c->model->layers[layer];
    size_t sz[3] = { L->ffn_gate_exps.info.size_bytes / (size_t)c->n_experts,
                     L->ffn_up_exps.info.size_bytes / (size_t)c->n_experts,
                     L->ffn_down_exps.info.size_bytes / (size_t)c->n_experts };
    char *dst[3] = { c->d_gate + (size_t)victim * c->gate_slot, c->d_up + (size_t)victim * c->up_slot,
                     c->d_down + (size_t)victim * c->down_slot };
    size_t total = 0;
    for (int t = 0; t < 3; ++t) {
        const lfm25_tensor_t *x = exps_tensor(L, t);
        size_t poff = c->host_mode == 1 ? g_store.off[layer * 3 + t] + (size_t)expert * sz[t] : 0;
        issue_copy(c, dst[t], (const char*)x->data + (size_t)expert * sz[t], poff, sz[t]);
        total += sz[t];
    }
    c->ring[ev].bytes += total;
    c->up_bytes += total;
    if (is_prefetch) { c->pf_bytes += total; c->pf_slabs++; }
    return victim;
}

lfm25_expert_cache_t* lfm25_expert_cache_create(lfm25_model_t *m) {
    // Checks first, allocations after (no leaks on the error paths).
    size_t gate_slot = 0, up_slot = 0, down_slot = 0;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        const lfm25_layer_t *L = &m->layers[l];
        if (L->ffn_type != LFM25_FFN_MOE) continue;
        size_t g = L->ffn_gate_exps.info.size_bytes / m->expert_count;
        size_t u = L->ffn_up_exps.info.size_bytes / m->expert_count;
        size_t d = L->ffn_down_exps.info.size_bytes / m->expert_count;
        if (g > gate_slot) gate_slot = g;
        if (u > up_slot) up_slot = u;
        if (d > down_slot) down_slot = d;
    }
    size_t slot_bytes = gate_slot + up_slot + down_slot;
    if (slot_bytes == 0) { fprintf(stderr, "[LFM25 CACHE] ERROR: no MoE layers\n"); return NULL; }

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
    // Slots are only ever keyed by MoE (layer, expert): cap by n_moe_layers (the leading dense layers have no experts).
    // The key -> slot map keeps its block_count x n_experts size because keys use the absolute layer index.
    size_t max_slots = (size_t)m->n_moe_layers * m->expert_count;
    const size_t map_size = (size_t)m->block_count * m->expert_count;
    size_t ns = budget / slot_bytes;
    if (ns > max_slots) ns = max_slots;
    if (ns < 8) {
        fprintf(stderr, "[LFM25 CACHE] ERROR: expert cache budget %zu MiB holds only %zu slots (need >= 8, %.2f MiB each); "
                "set LFM25_EXPERT_CACHE_MB or lower LFM25_VRAM_RESERVE_MB / context\n",
                budget >> 20, ns, slot_bytes / 1048576.0);
        return NULL;
    }

    lfm25_expert_cache_t *c = (lfm25_expert_cache_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->model = m;
    c->n_layers = (int)m->block_count;
    c->n_experts = (int)m->expert_count;
    c->gate_slot = gate_slot; c->up_slot = up_slot; c->down_slot = down_slot;
    c->n_slots = (int)ns;
    c->pred_layer = -1;

    const char *st = getenv("LFM25_CACHE_STATS");
    c->timing = st && st[0] == '1';
    const char *pol = getenv("LFM25_CACHE_POLICY");
    c->policy = (pol && strcmp(pol, "lru") == 0) ? 0 : 1;
    c->decay_period = (uint64_t)env_mb("LFM25_CACHE_DECAY", 0);
    if (c->decay_period == 0) c->decay_period = 8 * ns;
    c->recent_pct = (int)env_mb("LFM25_CACHE_RECENT_PCT", 50);

    CUDA_CHECK(cudaMalloc((void**)&c->d_gate, ns * c->gate_slot));
    CUDA_CHECK(cudaMalloc((void**)&c->d_up, ns * c->up_slot));
    CUDA_CHECK(cudaMalloc((void**)&c->d_down, ns * c->down_slot));

    c->map = (int*)malloc(map_size * sizeof(int));
    for (size_t i = 0; i < map_size; ++i) c->map[i] = -1;
    c->slot_key = (int*)malloc(ns * sizeof(int));
    c->slot_ev = (int*)malloc(ns * sizeof(int));
    c->slot_use = (uint64_t*)calloc(ns, sizeof(uint64_t));
    c->slot_prot = (uint64_t*)calloc(ns, sizeof(uint64_t));
    c->slot_cnt = (uint32_t*)calloc(ns, sizeof(uint32_t));
    c->slot_pf = (uint8_t*)calloc(ns, sizeof(uint8_t));
    for (size_t i = 0; i < ns; ++i) { c->slot_key[i] = -1; c->slot_ev[i] = -1; }

    CUDA_CHECK(cudaStreamCreateWithFlags(&c->copy_stream, cudaStreamNonBlocking));
    for (int i = 0; i < EV_RING; ++i) {
        if (c->timing) CUDA_CHECK(cudaEventCreate(&c->ring[i].start));
        CUDA_CHECK(cudaEventCreateWithFlags(&c->ring[i].end, c->timing ? cudaEventDefault : cudaEventDisableTiming));
    }

    // Host source selection.
    const char *pe = getenv("LFM25_PINNED_EXPERTS");
    int pin_off = pe && pe[0] == '0';
    int pin_force = pe && pe[0] == '1';
    c->host_mode = 0;
    if (!pin_off && store_acquire(m, pin_force)) c->host_mode = 1;
    if (c->host_mode == 0) {
        // staging ring: STAGE_BUFS pinned buffers, each big enough for the largest single slab
        size_t mx = c->gate_slot > c->up_slot ? c->gate_slot : c->up_slot;
        if (c->down_slot > mx) mx = c->down_slot;
        c->stage_bytes = (mx + 4095) & ~(size_t)4095;
        int ok = 1;
        for (int i = 0; i < STAGE_BUFS && ok; ++i) {
            if (cudaHostAlloc((void**)&c->stage[i].ptr, c->stage_bytes, cudaHostAllocDefault) != cudaSuccess) {
                (void)cudaGetLastError(); c->stage[i].ptr = NULL; ok = 0; break;
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&c->stage[i].done, cudaEventDisableTiming));
        }
        if (ok) c->host_mode = 2;
        else {
            for (int i = 0; i < STAGE_BUFS; ++i) {
                if (c->stage[i].ptr) { cudaEventDestroy(c->stage[i].done); cudaFreeHost(c->stage[i].ptr); c->stage[i].ptr = NULL; }
            }
        }
    }
    static const char *mode_name[3] = { "PAGEABLE (direct from mmap)", "PINNED STORE (anonymous mapping + cudaHostRegister)", "PINNED STAGING RING (memcpy + DMA overlap)" };
    fprintf(stderr, "[LFM25 CACHE] %d slots (%.0f MiB: gate %zu + up %zu + down %zu B per slot), %zu of %zu expert slabs fit; "
            "host path: %s; policy: %s\n",
            c->n_slots, (double)(ns * slot_bytes) / 1048576.0, c->gate_slot, c->up_slot, c->down_slot,
            ns, (size_t)m->n_moe_layers * m->expert_count, mode_name[c->host_mode],
            c->policy ? "LFRU" : "LRU");
    return c;
}

// Make `consumer` wait for every pending upload of the slots in `slots`.
static void wait_slots(lfm25_expert_cache_t *c, const int *slots, int n, cudaStream_t consumer) {
    int last = -1;
    for (int i = 0; i < n; ++i) {
        int s = slots[i];
        int ev = c->slot_ev[s];
        if (ev < 0) continue;
        if (ev != last) CUDA_CHECK(cudaStreamWaitEvent(consumer, c->ring[ev].end, 0));
        last = ev;
        c->slot_ev[s] = -1;
    }
}

void lfm25_expert_cache_lookup(lfm25_expert_cache_t *c, int layer, const int *ids, int n, int *slots, cudaStream_t consumer) {
    // Prediction accuracy bookkeeping (prediction never influences anything below).
    if (c->pred_layer == layer) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < c->pred_n; ++j)
                if (ids[i] == c->pred_ids[j]) { c->pred_correct++; break; }
        c->pred_total += (uint64_t)n;
        c->pred_layer = -1;
    }

    // Pass 1: resolve hits and protect them (stamp with the new tick) so misses can't evict them.
    c->tick++;
    int miss[8], nmiss = 0;
    for (int i = 0; i < n; ++i) {
        int s = c->map[layer * c->n_experts + ids[i]];
        if (s >= 0) {
            slots[i] = s; c->slot_use[s] = c->tick; c->slot_prot[s] = c->tick;
            if (c->slot_cnt[s] < 0xFFFFFFFFu) c->slot_cnt[s]++;
            c->hits++;
            if (c->slot_pf[s]) {
                c->slot_pf[s] = 0; c->pf_hits++;
                // stats mode only: the copy already finished when the real router result arrived = fully hidden
                if (c->timing && (c->slot_ev[s] < 0 || cudaEventQuery(c->ring[c->slot_ev[s]].end) == cudaSuccess)) c->pf_hidden++;
            }
        } else { slots[i] = -1; miss[nmiss++] = i; }
    }
    c->since_decay += (uint64_t)n;
    if (c->policy == 1 && c->since_decay >= c->decay_period) {
        c->since_decay = 0;
        for (int s = 0; s < c->n_slots; ++s) c->slot_cnt[s] >>= 1;
    }

    if (nmiss > 0) {
        // Pass 2: fill misses (victim policy; never a slot stamped this tick).
        int ev = batch_begin(c);
        for (int j = 0; j < nmiss; ++j) {
            int i = miss[j];
            int s = fill_slot(c, layer, ids[i], ev, 0);
            if (s < 0) { fprintf(stderr, "[LFM25 CACHE] FATAL: no evictable slot\n"); exit(EXIT_FAILURE); }
            slots[i] = s;
            c->misses++;
        }
        batch_end(c, ev);
    }
    wait_slots(c, slots, n, consumer);
}

void lfm25_expert_cache_prefetch(lfm25_expert_cache_t *c, int layer, const int *ids, int n) {
    // Same tick as the lookup that just ran for the previous layer: its slots are protected.
    int issued = 0, ev = -1;
    for (int i = 0; i < n; ++i) {
        int s = c->map[layer * c->n_experts + ids[i]];
        if (s >= 0) { c->slot_prot[s] = c->tick; continue; }     // already resident (or being filled): just protect it
        if (!issued) { ev = batch_begin(c); issued = 1; }
        if (fill_slot(c, layer, ids[i], ev, 1) < 0) break;       // nothing evictable: skip the rest
    }
    if (issued) batch_end(c, ev);
    c->pred_layer = layer;
    c->pred_n = n > 8 ? 8 : n;
    for (int i = 0; i < c->pred_n; ++i) c->pred_ids[i] = ids[i];
}

void* lfm25_expert_cache_pool_gate(const lfm25_expert_cache_t *c) { return c->d_gate; }
void* lfm25_expert_cache_pool_up(const lfm25_expert_cache_t *c) { return c->d_up; }
void* lfm25_expert_cache_pool_down(const lfm25_expert_cache_t *c) { return c->d_down; }
size_t lfm25_expert_cache_gate_slot_bytes(const lfm25_expert_cache_t *c) { return c->gate_slot; }
size_t lfm25_expert_cache_up_slot_bytes(const lfm25_expert_cache_t *c) { return c->up_slot; }
size_t lfm25_expert_cache_down_slot_bytes(const lfm25_expert_cache_t *c) { return c->down_slot; }

void lfm25_expert_cache_free(lfm25_expert_cache_t *c) {
    if (!c) return;
    cudaDeviceSynchronize();
    for (int i = 0; i < EV_RING; ++i) ring_drain_entry(c, &c->ring[i]);
    static const char *mode_short[3] = { "pageable", "pinned store", "staging ring" };
    uint64_t tot = c->hits + c->misses;
    fprintf(stderr, "[LFM25 CACHE] stats: %llu lookups, %llu hits, %llu misses, hit rate %.2f%% (%d slots, %s, %s)\n",
            (unsigned long long)tot, (unsigned long long)c->hits, (unsigned long long)c->misses,
            tot ? 100.0 * (double)c->hits / (double)tot : 0.0, c->n_slots, mode_short[c->host_mode],
            c->policy ? "LFRU" : "LRU");
    if (c->timing) {
        if (c->up_sec > 0)
            fprintf(stderr, "[LFM25 CACHE] uploaded %.1f MiB (%.1f MiB prefetched) in %.3f s of copy-stream time -> %.2f GB/s (%s)\n",
                    (double)c->up_bytes / 1048576.0, (double)c->pf_bytes / 1048576.0, c->up_sec,
                    (double)c->up_bytes / c->up_sec / 1e9, mode_short[c->host_mode]);
        if (c->pf_slabs || c->pred_total)
            fprintf(stderr, "[LFM25 CACHE] prefetch: %llu slots issued, prediction accuracy %.1f%% (%llu/%llu real experts predicted), "
                    "%llu prefetched slots used (%llu already complete when needed = fully hidden), %llu evicted unused\n",
                    (unsigned long long)c->pf_slabs,
                    c->pred_total ? 100.0 * (double)c->pred_correct / (double)c->pred_total : 0.0,
                    (unsigned long long)c->pred_correct, (unsigned long long)c->pred_total,
                    (unsigned long long)c->pf_hits, (unsigned long long)c->pf_hidden, (unsigned long long)c->pf_wasted);
    }
    for (int i = 0; i < EV_RING; ++i) {
        if (c->timing) cudaEventDestroy(c->ring[i].start);
        cudaEventDestroy(c->ring[i].end);
    }
    for (int i = 0; i < STAGE_BUFS; ++i) {
        if (c->stage[i].ptr) { cudaEventDestroy(c->stage[i].done); cudaFreeHost(c->stage[i].ptr); }
    }
    cudaStreamDestroy(c->copy_stream);
    cudaFree(c->d_gate); cudaFree(c->d_up); cudaFree(c->d_down);
    free(c->map); free(c->slot_key); free(c->slot_use); free(c->slot_prot); free(c->slot_cnt); free(c->slot_ev); free(c->slot_pf);
    free(c);
}
