#define _GNU_SOURCE
#include "engine.h"
#include "gpu.h"
#include "tuff_exact.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <immintrin.h>

// Iteration-12 static expert split count (set by main.c from --n-gpu-expert-layers).
extern int g_n_gpu_expert_layers;

// --spec-mtp: evaluate the blk.40 MTP draft head after each token (declared in engine.h).
int g_spec_mtp = 1;

// H17/H29: --spec-draft=ngram|mtp|ngram+mtp; --spec-draft-len sets candidate depth (1..3).
int g_spec_draft     = SPEC_DRAFT_NGRAM;
int g_spec_draft_len = 2;
int g_spec_ngram_n   = 3;
int g_spec_ngram_min = 2;

// H6: cached TUFF_H6 flag. Read once; 1 enables the event-scoped router-logits wait
// (gpu_router_logits_wait_event) instead of a full cudaStreamSynchronize(0).
static int tuff_h6(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TUFF_H6"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

// H31: cached TUFF_BGFILL_OVERLAP flag. 1 moves the background-fill commit poll from
// the token boundary to layer 0 of the NEXT token: after its attention GPU work has been
// issued but before the first expert-cache lookup (the slot table must be frozen before
// gpu_expert_lookup reads it). The committed set per token is identical to the serial
// poll — only the host timeline of the commit changes, and no lookup can observe it.
static int tuff_bgfill_overlap(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TUFF_BGFILL_OVERLAP"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

// H25: cached TUFF_GPU_ROUTER flag. 1 enables GPU-native Top-8 reduction + softmax.
static int tuff_gpu_router(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TUFF_GPU_ROUTER"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

// ===========================================================================
// P1 (parity.md sec.2.1 / sec.4): MoE miss worker pool.
//
// The router selects expert_used_count experts per layer. Those resident in the LFRU VRAM
// cache (hits) run on the GPU exactly as before. The misses used to be *uploaded over PCIe
// from this thread* and then run on the GPU -- 189 MB/token at ~9 GB/s = 21.0 ms/token, 35%
// of the whole token, all of it on the critical path. They now run on the CPU straight out of
// the mmap'd GGUF (P2's AVX2 int8 kernels, 36-41 GB/s) on a dedicated worker pool, while this
// thread keeps issuing GPU work. The pool is joined before the k-order combine, so the
// accumulation order into moe_buf is bit-for-bit the order it always had.
//
// Iteration 12 tried this and regressed to ~8 tok/s because it called mat_vec_cpu() from a
// worker thread; that function opens its own `#pragma omp parallel for`, so the worker's OMP
// region and the main thread's oversubscribed all 8 cores. This pool calls
// expert_mat_vec_cpu(..., parallel=0), which contains no OMP region at all, and owns its own
// explicitly-pinned threads.
// ===========================================================================
#define MOE_MAX_WORKERS 15

typedef struct {
    pthread_t   th;
    int         id;
    float      *gate;    // [ffn_dim]
    float      *up;      // [ffn_dim]
    float      *act;     // [ffn_dim]
    block_q8_K *q8act;   // q8_K(act), the down projection's activation
} moe_worker_t;

#define MOE_MAX_MISS 16

typedef struct {
    const model_layer_t *L;
    const float        *x;      // norm_x (fp32), shared by every expert of the layer
    const block_q8_K   *q8x;    // q8_K(norm_x), quantized ONCE per layer and reused
    // H7: a verify pass submits both columns' misses as ONE batch, and the two
    // tokens have different activations, so the activation is per entry. The
    // single-token path sets every entry to x/q8x, which is what it always did.
    const float        *xk[MOE_MAX_MISS];
    const block_q8_K   *q8xk[MOE_MAX_MISS];
    // H15: a miss selected by BOTH columns (a shared miss) is one entry whose
    // weights are read once for both activations; xk2/q8xk2/down_out2 are then
    // non-NULL and carry the second column's activation and output slot. NULL for
    // single-column entries, so the legacy paths behave exactly as before.
    const float        *xk2[MOE_MAX_MISS];
    const block_q8_K   *q8xk2[MOE_MAX_MISS];
    float              *down_out2[MOE_MAX_MISS];
    int      dim, ffn;
    ggml_type_t gtype, dtype;
    size_t   gate_bytes, up_bytes, down_bytes;
    int      experts[MOE_MAX_MISS];
    float   *down_out[MOE_MAX_MISS];  // down_arr + k*dim, i.e. the expert's slot in k order
    int      n;
    // H10: with the split path each expert's intermediate activation is shared by the threads
    // computing its row chunks, so it lives here rather than in per-worker scratch.
    float      *act[MOE_MAX_MISS];    // [ffn_dim]
    block_q8_K *q8act[MOE_MAX_MISS];
    // H15: the shared misses' second activation (column 1), allocated alongside act/q8act.
    float      *act2[MOE_MAX_MISS];
    block_q8_K *q8act2[MOE_MAX_MISS];
    int      chunks;            // row chunks per expert per phase; 0 = legacy whole-expert tasks
} moe_batch_t;

// Background LFRU fills per layer per token (TUFF_FILL_BUDGET, default 2). 0 freezes the
// cache contents; every extra fill is PCIe traffic on the background thread, never a stall here.
unsigned long long g_ovl_shared = 0, g_ovl_total = 0;
static int g_fill_budget = -1;
static int g_fill_stride = 1;   // H14: issue fills only on every k-th layer

static moe_worker_t g_moe_w[MOE_MAX_WORKERS];
static moe_worker_t g_moe_main;         // scratch for the decode thread's own share
static int  g_moe_nw = -1;              // -1 = uninitialised, 0 = pool disabled
static moe_batch_t g_moe_batch;
static volatile unsigned long g_moe_gen = 0;
static volatile int g_moe_next_in_q = 0;
static volatile int g_moe_done_in_q = 0;
static volatile int g_moe_next = 0;   // legacy whole-expert claim, and H10 phase-A claim
static volatile int g_moe_done = 0;
// H10: the two extra phases of the split path — q8_K(act) per expert, then the down chunks.
static volatile int g_moe_next_q = 0;
static volatile int g_moe_done_q = 0;
static volatile int g_moe_next_b = 0;
static volatile int g_moe_done_b = 0;
static int g_moe_split = -1;          // TUFF_MOE_SPLIT: -1 auto, 0 legacy, >0 forced chunks
static int g_moe_shutdown = 0;
static pthread_mutex_t g_moe_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_moe_cv  = PTHREAD_COND_INITIALIZER;

// One expert, start to finish, on one thread: gate, up, SwiGLU, down. Every mat-vec runs with
// parallel=0 -- no nested OpenMP region, which is the iteration-12 defect. A shared miss
// (xk2[t] != NULL) reads the weights once and evaluates both columns' activations against them.
static void moe_run_expert(moe_worker_t *w, int t) {
    const moe_batch_t *b = &g_moe_batch;
    const int e = b->experts[t];
    const uint8_t *gw = (const uint8_t*)b->L->ffn_gate_exps.data + (size_t)e * b->gate_bytes;
    const uint8_t *uw = (const uint8_t*)b->L->ffn_up_exps.data   + (size_t)e * b->up_bytes;
    const uint8_t *dw = (const uint8_t*)b->L->ffn_down_exps.data + (size_t)e * b->down_bytes;

    if (b->xk2[t]) {
        // Shared miss: one pass over the weights for both columns.
        cpu_expert_gateup_swiglu_dual(w->act, b->act2[t], gw, uw, b->q8xk[t], b->q8xk2[t], b->dim, b->ffn);
        quantize_row_q8_K(w->act, w->q8act, b->ffn);
        quantize_row_q8_K(b->act2[t], b->q8act2[t], b->ffn);
        expert_mat_vec_cpu_dual(b->down_out[t], b->down_out2[t], dw, b->dtype, b->ffn, b->dim,
                                w->q8act, b->q8act2[t]);
    } else {
        cpu_expert_gateup_swiglu(w->act, gw, uw, b->xk[t], b->q8xk[t], b->dim, b->ffn);
        quantize_row_q8_K(w->act, w->q8act, b->ffn);
        expert_mat_vec_cpu(b->down_out[t], dw, b->dtype, b->ffn, b->dim, w->act, w->q8act, 0);
    }
}

// H10: one row chunk of one expert. Task t encodes (expert j, chunk c) as j*chunks + c, and the
// row range is computed from the chunk index so it partitions the rows exactly — every row is
// computed once, by the same arithmetic as the whole-expert call, hence bit-identically.
// H15: a shared miss computes the same weight rows against both columns' activations.
static void moe_run_chunk_gateup(int t) {
    const moe_batch_t *b = &g_moe_batch;
    const int C = b->chunks, j = t / C, c = t % C;
    const int64_t r0 = (int64_t)b->ffn * c / C, r1 = (int64_t)b->ffn * (c + 1) / C;
    const int e = b->experts[j];
    const uint8_t *gw = (const uint8_t*)b->L->ffn_gate_exps.data + (size_t)e * b->gate_bytes;
    const uint8_t *uw = (const uint8_t*)b->L->ffn_up_exps.data   + (size_t)e * b->up_bytes;
    if (b->xk2[j])
        cpu_expert_gateup_swiglu_dual_rows(b->act[j], b->act2[j], gw, uw, b->q8xk[j], b->q8xk2[j],
                                           b->dim, b->ffn, r0, r1);
    else
        cpu_expert_gateup_swiglu_rows(b->act[j], gw, uw, b->q8xk[j], b->dim, b->ffn, r0, r1);
}

static void moe_run_chunk_down(int t) {
    const moe_batch_t *b = &g_moe_batch;
    const int C = b->chunks, j = t / C, c = t % C;
    const int64_t r0 = (int64_t)b->dim * c / C, r1 = (int64_t)b->dim * (c + 1) / C;
    const int e = b->experts[j];
    const uint8_t *dw = (const uint8_t*)b->L->ffn_down_exps.data + (size_t)e * b->down_bytes;
    if (b->xk2[j])
        expert_mat_vec_cpu_dual_rows(b->down_out[j], b->down_out2[j], dw, b->dtype, b->ffn, b->dim,
                                     b->q8act[j], b->q8act2[j], r0, r1);
    else
        expert_mat_vec_cpu_rows(b->down_out[j], dw, b->dtype, b->ffn, b->dim, b->q8act[j], r0, r1);
}

// Everything a participant (worker thread or the decode thread) does for one published batch.
// `mygen` is the generation the participant entered on: every claim loop re-checks it, because
// once the batch is complete the decode thread may publish the next one, and a straggler that
// claimed a phase-B task from the *new* batch would run it before that batch's phase A.
static void moe_participate(moe_worker_t *w, unsigned long mygen) {
    const int n = g_moe_batch.n;
    const int C = g_moe_batch.chunks;

    // Phase 0: Parallel input activation quantization (8 blocks of QK_K = 256 elements)
    const int nb_in = g_moe_batch.dim / QK_K;
    for (;;) {
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        const int b = __atomic_fetch_add(&g_moe_next_in_q, 1, __ATOMIC_ACQ_REL);
        if (b >= nb_in) break;
        quantize_row_q8_K(g_moe_batch.x + (size_t)b * QK_K, (block_q8_K*)g_moe_batch.q8x + b, QK_K);
        if (g_moe_batch.xk2[0]) {
            quantize_row_q8_K(g_moe_batch.xk2[0] + (size_t)b * QK_K, (block_q8_K*)g_moe_batch.q8xk2[0] + b, QK_K);
        }
        __atomic_fetch_add(&g_moe_done_in_q, 1, __ATOMIC_ACQ_REL);
    }
    while (__atomic_load_n(&g_moe_done_in_q, __ATOMIC_ACQUIRE) < nb_in) {
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        _mm_pause();
    }

    if (C == 0) {                       // legacy: one whole expert per task
        for (;;) {
            if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
            const int t = __atomic_fetch_add(&g_moe_next, 1, __ATOMIC_ACQ_REL);
            if (t >= n) break;
            moe_run_expert(w, t);
            __atomic_fetch_add(&g_moe_done, 1, __ATOMIC_ACQ_REL);
        }
        return;
    }

    const int na = n * C;
    for (;;) {                          // phase A: gate/up/SwiGLU row chunks
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        const int t = __atomic_fetch_add(&g_moe_next, 1, __ATOMIC_ACQ_REL);
        if (t >= na) break;
        moe_run_chunk_gateup(t);
        __atomic_fetch_add(&g_moe_done, 1, __ATOMIC_ACQ_REL);
    }
    while (__atomic_load_n(&g_moe_done, __ATOMIC_ACQUIRE) < na) {
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        _mm_pause();
    }
    for (;;) {                          // the down projection's activation, one task per expert
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        const int j = __atomic_fetch_add(&g_moe_next_q, 1, __ATOMIC_ACQ_REL);
        if (j >= n) break;
        quantize_row_q8_K(g_moe_batch.act[j], g_moe_batch.q8act[j], g_moe_batch.ffn);
        if (g_moe_batch.xk2[j])
            quantize_row_q8_K(g_moe_batch.act2[j], g_moe_batch.q8act2[j], g_moe_batch.ffn);
        __atomic_fetch_add(&g_moe_done_q, 1, __ATOMIC_ACQ_REL);
    }
    while (__atomic_load_n(&g_moe_done_q, __ATOMIC_ACQUIRE) < n) {
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        _mm_pause();
    }
    for (;;) {                          // phase B: down row chunks
        if (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) != mygen) return;
        const int t = __atomic_fetch_add(&g_moe_next_b, 1, __ATOMIC_ACQ_REL);
        if (t >= na) break;
        moe_run_chunk_down(t);
        __atomic_fetch_add(&g_moe_done_b, 1, __ATOMIC_ACQ_REL);
    }
}

static void *moe_worker_fn(void *arg) {
    moe_worker_t *w = (moe_worker_t*)arg;
    unsigned long mygen = 0;
    for (;;) {
        while (__atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE) == mygen) {
            _mm_pause();
            if (__atomic_load_n(&g_moe_shutdown, __ATOMIC_ACQUIRE)) return NULL;
        }
        mygen = __atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&g_moe_shutdown, __ATOMIC_ACQUIRE)) break;
        moe_participate(w, mygen);
    }
    return NULL;
}

// Physical cores first (this CPU pairs SMT siblings adjacently: core i == cpus 2i, 2i+1), then
// the sibling threads. The decode thread is left on cpu 0/1 unpinned.
static const int k_moe_cpu_order[MOE_MAX_WORKERS] = {
    2, 4, 6, 8, 10, 12, 14, 3, 5, 7, 9, 11, 13, 15, 1
};

static void moe_worker_alloc(moe_worker_t *w, int ffn) {
    w->gate  = (float*)malloc((size_t)ffn * sizeof(float));
    w->up    = (float*)malloc((size_t)ffn * sizeof(float));
    w->act   = (float*)malloc((size_t)ffn * sizeof(float));
    w->q8act = (block_q8_K*)malloc(q8_K_row_bytes(ffn));
    if (!w->gate || !w->up || !w->act || !w->q8act) {
        fprintf(stderr, "[FATAL ERROR] moe worker: scratch allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

static void moe_worker_free(moe_worker_t *w) {
    free(w->gate); free(w->up); free(w->act); free(w->q8act);
    w->gate = w->up = w->act = NULL; w->q8act = NULL;
}

static void moe_pool_init(const model_t *m) {
    if (g_moe_nw >= 0) return;
    const char *s = getenv("TUFF_MOE_WORKERS");
    int nw = s ? atoi(s) : 7;
    if (nw < 0) nw = 0;
    if (nw > MOE_MAX_WORKERS) nw = MOE_MAX_WORKERS;

    // Default 1, not 2: iteration 37 swept this. Above 1 the background fills buy hit rate and
    // *lose* throughput, because they contend for the same PCIe the decode path needs —
    // budget 1 / 2 / 4 measure 42.24 / 35.07 / 35.93 tok/s at 71.2 / 75.3 / 78.4% hit rate.
    const char *fb = getenv("TUFF_FILL_BUDGET");
    g_fill_budget = fb ? atoi(fb) : 1;
    if (g_fill_budget < 0) g_fill_budget = 0;
    const char *fs = getenv("TUFF_FILL_STRIDE");
    g_fill_stride = fs ? atoi(fs) : 1;
    if (g_fill_stride < 1) g_fill_stride = 1;

    const int ffn = (int)m->expert_ffn_length;
    moe_worker_alloc(&g_moe_main, ffn);
    g_moe_shutdown = 0;
    g_moe_batch.n = 0;

    // H10: the split path shares one `act` per expert across the threads computing its chunks,
    // so these belong to the batch, not to a worker. TUFF_MOE_SPLIT=0 restores the legacy
    // one-expert-per-thread path (the A/B baseline); >0 forces a chunk count.
    const char *sp = getenv("TUFF_MOE_SPLIT");
    g_moe_split = sp ? atoi(sp) : -1;
    for (int j = 0; j < MOE_MAX_MISS; ++j) {
        g_moe_batch.act[j]   = (float*)malloc((size_t)ffn * sizeof(float));
        g_moe_batch.q8act[j] = (block_q8_K*)malloc(q8_K_row_bytes(ffn));
        g_moe_batch.act2[j]   = (float*)malloc((size_t)ffn * sizeof(float));
        g_moe_batch.q8act2[j] = (block_q8_K*)malloc(q8_K_row_bytes(ffn));
        if (!g_moe_batch.act[j] || !g_moe_batch.q8act[j] ||
            !g_moe_batch.act2[j] || !g_moe_batch.q8act2[j]) {
            fprintf(stderr, "[FATAL ERROR] moe pool: split scratch allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < nw; ++i) {
        moe_worker_t *w = &g_moe_w[i];
        w->id = i;
        moe_worker_alloc(w, ffn);
        if (pthread_create(&w->th, NULL, moe_worker_fn, w) != 0) {
            fprintf(stderr, "[FATAL ERROR] moe pool: pthread_create failed\n");
            exit(EXIT_FAILURE);
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(k_moe_cpu_order[i % MOE_MAX_WORKERS], &set);
        pthread_setaffinity_np(w->th, sizeof(set), &set);
    }
    g_moe_nw = nw;
    if (getenv("TUFF_PROFILE"))
        fprintf(stderr, "[MOE-POOL] %d pinned worker thread(s) for CPU-computed expert misses\n", nw);
}

static void moe_pool_shutdown(void) {
    if (g_moe_nw < 0) return;
    __atomic_store_n(&g_moe_shutdown, 1, __ATOMIC_RELEASE);
    __atomic_fetch_add(&g_moe_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < g_moe_nw; ++i) { pthread_join(g_moe_w[i].th, NULL); moe_worker_free(&g_moe_w[i]); }
    moe_worker_free(&g_moe_main);
    for (int j = 0; j < MOE_MAX_MISS; ++j) {
        free(g_moe_batch.act[j]);   g_moe_batch.act[j] = NULL;
        free(g_moe_batch.q8act[j]); g_moe_batch.q8act[j] = NULL;
        free(g_moe_batch.act2[j]);  g_moe_batch.act2[j] = NULL;
        free(g_moe_batch.q8act2[j]); g_moe_batch.q8act2[j] = NULL;
    }
    g_moe_nw = -1;
}

// H10: how many row chunks each expert is split into. The pool holds g_moe_nw workers plus the
// decode thread, so with n misses the whole machine is busy at ceil(T/n) chunks; the caps keep a
// chunk from falling below ~64 gate/up rows or ~128 down rows, where the two spin barriers would
// start to cost more than the split saves. 1 chunk means the split has nothing to give, so the
// legacy whole-expert path runs instead and pays no barriers at all.
static int moe_chunks_for(int n_miss) {
    if (g_moe_split == 0 || n_miss <= 0) return 0;
    if (g_moe_split > 0) return g_moe_split;
    const int threads = g_moe_nw + 1;
    int c = (threads + n_miss - 1) / n_miss;
    const int cap_a = g_moe_batch.ffn / 64;
    const int cap_b = g_moe_batch.dim / 128;
    if (c > cap_a) c = cap_a;
    if (c > cap_b) c = cap_b;
    if (c < 1) c = 1;
    return c == 1 ? 0 : c;
}

// Publish g_moe_batch to the pool with zero mutex locks.
static void moe_pool_submit(void) {
    g_moe_batch.chunks = moe_chunks_for(g_moe_batch.n);
    __atomic_store_n(&g_moe_next_in_q, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_done_in_q, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_next,   0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_done,   0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_next_q, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_done_q, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_next_b, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_moe_done_b, 0, __ATOMIC_RELAXED);
    if (g_moe_nw == 0) return;
    __atomic_fetch_add(&g_moe_gen, 1, __ATOMIC_RELEASE);
}

// Join. The decode thread claims any task still unstarted rather than idling on it.
static void moe_pool_wait(void) {
    const int n = g_moe_batch.n;
    const int C = g_moe_batch.chunks;
    moe_participate(&g_moe_main, __atomic_load_n(&g_moe_gen, __ATOMIC_ACQUIRE));
    if (C == 0) {
        while (__atomic_load_n(&g_moe_done, __ATOMIC_ACQUIRE) < n) _mm_pause();
    } else {
        while (__atomic_load_n(&g_moe_done_b, __ATOMIC_ACQUIRE) < n * C) _mm_pause();
    }
}

// ---------------------------------------------------------------------------
// M2: route a mat-vec to the GPU when the tensor has a device copy (dense weights);
// everything else (experts, token_embd, output) stays on the CPU path.
static void mat_vec(float *dst, const tensor_t *w, const float *src) {
    if (w->dev) gpu_mat_vec(dst, w, src);
    else mat_vec_cpu(dst, w, src);
}

// Debug dump of a float tensor to $TUFF_DUMP_DIR/<name>_<il>.bin as raw f32 (for
// comparing against llama.cpp's LLAMA_DUMP_DIR graph-node dumps). Only dumps during the
// first forward pass (pos == 0) so later forwards don't overwrite the reference.
static int g_dump_active = 0;

// P4a: TUFF_CPU_DELTA / TUFF_CPU_ATTN are read inside the per-layer, per-token loop
// (~90 environ scans/token from the delta branch alone). getenv() is a linear scan of environ,
// so cache each in a static after the first read -- the same pattern as tuff_exact().
static int g_cpu_delta = -1;
static int g_cpu_attn  = -1;
static int cpu_delta_flag(void) {
    int v = g_cpu_delta;
    if (v < 0) { v = getenv("TUFF_CPU_DELTA") ? 1 : 0; g_cpu_delta = v; }
    return v;
}
static int cpu_attn_flag(void) {
    int v = g_cpu_attn;
    if (v < 0) { v = getenv("TUFF_CPU_ATTN") ? 1 : 0; g_cpu_attn = v; }
    return v;
}
// Same reason: TUFF_DEBUG_LOGITS was scanned once per token and TUFF_DEBUG_MTP once per
// attention layer per token, on the hot path.
static int g_dbg_logits = -1;
static int g_dbg_mtp    = -1;
static int dbg_logits_flag(void) {
    int v = g_dbg_logits;
    if (v < 0) { v = getenv("TUFF_DEBUG_LOGITS") ? 1 : 0; g_dbg_logits = v; }
    return v;
}
static int dbg_mtp_flag(void) {
    int v = g_dbg_mtp;
    if (v < 0) { v = getenv("TUFF_DEBUG_MTP") ? 1 : 0; g_dbg_mtp = v; }
    return v;
}
static void dump_f32(const char *name, int il, const float *data, size_t n) {
    if (!g_dump_active) return;
    const char *dir = getenv("TUFF_DUMP_DIR");
    if (!dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s-%d.bin", dir, name, il);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fprintf(fp, "ne0=%zu\n", n);
        fwrite(data, sizeof(float), n, fp);
        fclose(fp);
    }
}

static void dump_f32_noil(const char *name, const float *data, size_t n) {
    if (!g_dump_active) return;
    const char *dir = getenv("TUFF_DUMP_DIR");
    if (!dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fprintf(fp, "ne0=%zu\n", n);
        fwrite(data, sizeof(float), n, fp);
        fclose(fp);
    }
}

engine_t* engine_init(model_t *model, int max_seq_len) {
    engine_t *eng = (engine_t*)calloc(1, sizeof(engine_t));
    eng->model = model;
    eng->max_seq_len = max_seq_len > 0 ? max_seq_len : 4096;

    // All sizes below are derived from GGUF-loaded hyperparameters (PLAN.md sec.4 rule 4) —
    // no hardcoded 30/11/32/128/8192/etc.
    size_t delta_floats = (size_t)model->n_delta_layers *
                           model->ssm_n_v_heads * model->ssm_head_v_dim * model->ssm_head_k_dim;
    eng->delta_state = (float*)calloc(delta_floats > 0 ? delta_floats : 1, sizeof(float));

    size_t conv_floats = (size_t)model->n_delta_layers * model->ssm_conv_kernel * model->ssm_conv_dim;
    eng->conv_state = (float*)calloc(conv_floats > 0 ? conv_floats : 1, sizeof(float));

    size_t kv_floats = (size_t)model->n_attn_layers * model->head_count_kv *
                        (size_t)eng->max_seq_len * model->key_length;
    eng->k_cache = (float*)calloc(kv_floats > 0 ? kv_floats : 1, sizeof(float));
    eng->v_cache = (float*)calloc(kv_floats > 0 ? kv_floats : 1, sizeof(float));

    if (!eng->delta_state || !eng->conv_state || !eng->k_cache || !eng->v_cache) {
        fprintf(stderr, "[FATAL ERROR] engine_init: state allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // H17: the n-gram draft source keeps the accepted token sequence itself. Allocated lazily
    // so the default mtp path pays nothing.
    if (g_spec_draft != SPEC_DRAFT_MTP) {
        eng->hist_cap = max_seq_len > 0 ? max_seq_len : 4096;
        if (eng->hist_cap < 1024) eng->hist_cap = 1024;
        if (eng->hist_cap > 65536) eng->hist_cap = 65536;
        eng->hist_tokens = (int32_t*)malloc((size_t)eng->hist_cap * sizeof(int32_t));
        if (!eng->hist_tokens) {
            fprintf(stderr, "[FATAL ERROR] engine_init: ngram history allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }

    uint32_t dim = model->embedding_length;
    uint32_t n_expert = model->expert_count;
    uint32_t n_expert_used = model->expert_used_count;
    uint32_t ffn_dim = model->expert_ffn_length;
    uint32_t sh_ffn_dim = model->expert_shared_ffn_len;

    eng->x             = (float*)gpu_host_alloc(dim * sizeof(float));
    eng->norm_x        = (float*)gpu_host_alloc(dim * sizeof(float));
    eng->moe_buf       = (float*)gpu_host_alloc(dim * sizeof(float));
    eng->router_logits = (float*)gpu_host_alloc(n_expert * sizeof(float));
    eng->top_indices   = (int*)malloc(n_expert_used * sizeof(int));
    eng->top_values    = (float*)malloc(n_expert_used * sizeof(float));
    eng->gate_exp      = (float*)malloc(ffn_dim * sizeof(float));
    eng->up_exp        = (float*)malloc(ffn_dim * sizeof(float));
    eng->act_exp       = (float*)malloc(ffn_dim * sizeof(float));
    eng->down_exp      = (float*)malloc(dim * sizeof(float));
    eng->sh_gate_exp   = (float*)malloc(sh_ffn_dim * sizeof(float));
    eng->exp_slots     = (int*)malloc(n_expert_used * sizeof(int));
    eng->gate_arr      = (float*)malloc(n_expert_used * ffn_dim * sizeof(float));
    eng->up_arr        = (float*)malloc(n_expert_used * ffn_dim * sizeof(float));
    eng->act_arr       = (float*)malloc(n_expert_used * ffn_dim * sizeof(float));
    eng->down_arr      = (float*)malloc(n_expert_used * dim * sizeof(float));
    eng->down_hit      = (float*)malloc(n_expert_used * dim * sizeof(float));
    eng->sh_up_exp     = (float*)malloc(sh_ffn_dim * sizeof(float));
    eng->q8_normx      = (block_q8_K*)malloc(q8_K_row_bytes(dim));
    eng->norm_x2        = (float*)gpu_host_alloc(dim * sizeof(float));
    eng->moe_buf2       = (float*)gpu_host_alloc(dim * sizeof(float));
    eng->router_logits2 = (float*)gpu_host_alloc(n_expert * sizeof(float));
    eng->down_arr2      = (float*)malloc(n_expert_used * dim * sizeof(float));
    eng->q8_normx2      = (block_q8_K*)malloc(q8_K_row_bytes(dim));
    eng->sh_act_exp    = (float*)malloc(sh_ffn_dim * sizeof(float));
    eng->sh_down_exp   = (float*)malloc(dim * sizeof(float));

    // Keep the persistent device delta-net state in sync with this freshly-zeroed engine.
    gpu_reset_delta_state();
    // Keep the persistent device KV cache in sync with this freshly-zeroed host engine.
    gpu_reset_kv();

    // P1: the CPU miss workers are process-global (threads created once, reused across every
    // engine instance and every token); moe_pool_init is idempotent.
    if (!tuff_exact()) {
        moe_pool_init(model);
        // H12 (iteration 41): capture in every configuration. Until now this was gated on
        // g_n_gpu_expert_layers > 0 — the configuration nobody benchmarks — which is the whole
        // reason iteration 38 concluded the graphs "buy nothing". The delta span is static (the
        // recurrent and conv states are device-resident and updated in place; no pos, no KV
        // offset), so one capture at init is replayed unchanged for every token.
        // TUFF_GRAPHS=0 disables it, for the order-balanced A/B.
        const char *g = getenv("TUFF_GRAPHS");
        if (!(g && g[0] == '0')) {
            gpu_capture_delta_graphs(model);
            // H7: the verify trunk needs its own graphs, or it issues ~50 launches per
            // delta layer against the one-token path's single launch.
            if (g_spec_mtp) gpu_capture_delta_graphs_n2(model);
        }
    }

    sampler_params_t def_samp = sampler_params_default();
    sampler_init(&eng->sampler, &def_samp, (int32_t)model->vocab_size);
    eng->host_logits = NULL;

    return eng;
}

void engine_set_sampler(engine_t *eng, const sampler_params_t *params) {
    if (!eng) return;
    sampler_free(&eng->sampler);
    sampler_init(&eng->sampler, params, (int32_t)eng->model->vocab_size);
}

// ---------------------------------------------------------------------------
// H17: the n-gram / prompt-lookup draft source.
//
// The draft for the position after `next` is a pure function of the accepted
// sequence: take the last g_spec_ngram_n tokens (ending with `next`, which the
// trunk has already committed to), find the most recent occurrence of that
// n-token suffix at an earlier position, and return the token that followed it.
// No model evaluation at all. Returns -1 ("no draft") when there is no match,
// which engine_spec_step handles by doing a plain single-token forward.
//
// The accepted sequence is kept in eng->hist_tokens, appended in exactly the
// same order the trunk consumes tokens (engine_forward_token appends its input;
// engine_spec_step appends the consumed token and, on acceptance, the accepted
// draft). It is only maintained while an n-gram source is active.
static void engine_hist_append(engine_t *eng, int32_t tok) {
    if (!eng->hist_tokens) return;
    if (eng->hist_len >= eng->hist_cap) {
        int cap = eng->hist_cap > 0 ? eng->hist_cap : 1024;
        while (eng->hist_len >= cap) cap *= 2;
        if (cap > 65536) cap = 65536;          // cap: drop the oldest, never grow unbounded
        int32_t *nt = (int32_t*)realloc(eng->hist_tokens, (size_t)cap * sizeof(int32_t));
        if (!nt) return;
        eng->hist_tokens = nt;
        eng->hist_cap = cap;
    }
    if (eng->hist_len >= eng->hist_cap) {       // hard cap reached: drop the oldest
        memmove(eng->hist_tokens, eng->hist_tokens + 1, (size_t)(eng->hist_cap - 1) * sizeof(int32_t));
        eng->hist_tokens[eng->hist_cap - 1] = tok;
        return;
    }
    eng->hist_tokens[eng->hist_len++] = tok;
}

// seq = hist_tokens[0..H-1] + [next], L = H + 1. The suffix is the last n tokens of seq;
// the match must sit entirely before the suffix (i + n - 1 <= L - n - 1), so the draft is
// always a token the trunk has already verified. Tries n from n_max down to n_min.
static int engine_ngram_draft_multi(const engine_t *eng, int32_t next, int n_max, int n_min,
                                    int32_t *out_drafts, int max_drafts) {
    if (n_max < 1 || n_min < 1 || n_max < n_min || n_max > 8 || max_drafts < 1) return 0;
    const int32_t *hist = eng->hist_tokens;
    const int H = eng->hist_len;
    const int L = H + 1;
    if (!hist) return 0;

    for (int n = n_max; n >= n_min; --n) {
        if (L < 2 * n + 1) continue;
        int32_t sfx[8];
        for (int k = 0; k < n; ++k) sfx[k] = (k == n - 1) ? next : hist[H - n + 1 + k];
        for (int i = L - 2 * n - 1; i >= 0; --i) {
            int ok = 1;
            for (int k = 0; k < n; ++k) {
                const int32_t v = (i + k == H) ? next : hist[i + k];
                if (v != sfx[k]) { ok = 0; break; }
            }
            if (ok) {
                int count = 0;
                for (int d = 0; d < max_drafts; ++d) {
                    const int j = i + n + d;
                    if (j < H) {
                        out_drafts[count++] = hist[j];
                    } else if (j == H) {
                        out_drafts[count++] = next;
                    } else {
                        break;
                    }
                }
                return count;
            }
        }
    }
    return 0;
}

static int32_t engine_ngram_draft(const engine_t *eng, int32_t next, int n_max, int n_min) {
    int32_t d[1];
    int k = engine_ngram_draft_multi(eng, next, n_max, n_min, d, 1);
    return (k > 0) ? d[0] : -1;
}

#include "drafter.h"

void engine_set_drafter(engine_t *eng, struct drafter_s *d) {
    if (eng) eng->drafter = (drafter_t*)d;
}

// Compute the draft for the position after `last_token` and store it in
// eng->last_draft_tokens. `d_res_x` is only read by the MTP head. Dispatch on
// --spec-draft; ngram+mtp tries the free ngram match first and pays the head
// only on the steps where it is the sole option; external uses the background drafter.
static void engine_set_draft(engine_t *eng, int32_t last_token, int pos, const float *d_res_x) {
    model_t *m = eng->model;
    eng->last_draft_count = 0;
    eng->last_draft_token = -1;

    if (g_spec_draft == SPEC_DRAFT_EXTERNAL && eng->drafter) {
        drafter_submit_accepted(eng->drafter, last_token, pos);
        return;
    }
    if (!(g_spec_mtp && m->nextn_predict_layers > 0)) { return; }
    if (g_spec_draft == SPEC_DRAFT_NGRAM) {
        int k = engine_ngram_draft_multi(eng, last_token, g_spec_ngram_n, g_spec_ngram_min,
                                         eng->last_draft_tokens, g_spec_draft_len);
        eng->last_draft_count = k;
        eng->last_draft_token = (k > 0) ? eng->last_draft_tokens[0] : -1;
        return;
    }
    if (g_spec_draft == SPEC_DRAFT_NGRAM_MTP) {
        int k = engine_ngram_draft_multi(eng, last_token, g_spec_ngram_n, g_spec_ngram_min,
                                         eng->last_draft_tokens, g_spec_draft_len);
        if (k > 0) {
            eng->last_draft_count = k;
            eng->last_draft_token = eng->last_draft_tokens[0];
            return;
        }
    }
    int32_t d = gpu_mtp_draft_dev(m, d_res_x, last_token, pos, eng->max_seq_len);
    eng->last_draft_tokens[0] = d;
    eng->last_draft_count = (d >= 0) ? 1 : 0;
    eng->last_draft_token = d;
}

int32_t engine_forward_token(engine_t *eng, int32_t token_id, int pos) {
    model_t *m = eng->model;
    uint32_t dim = m->embedding_length; // 2048
    uint32_t vocab_size = m->vocab_size;
    const float eps = m->rms_norm_eps;
    double t_token = omp_get_wtime();

    // Check for garbage input token ID
    if (token_id < 0 || token_id >= (int32_t)vocab_size) {
        fprintf(stderr, "[FATAL ERROR] Invalid/garbage token ID %d (vocab size: %u)!\n", token_id, vocab_size);
        exit(EXIT_FAILURE);
    }

    // H17: the accepted sequence for the n-gram draft. token_id occupies `pos` and is now
    // committed by the trunk; it is the last element the next draft matches against.
    if (eng->hist_tokens) engine_hist_append(eng, token_id);

    g_dump_active = (pos == 0);

    if (pos < 0 || pos >= eng->max_seq_len) {
        fprintf(stderr, "[FATAL ERROR] Position %d out of range (max_seq_len=%d)!\n", pos, eng->max_seq_len);
        exit(EXIT_FAILURE);
    }

    float *x             = eng->x;
    float *norm_x        = eng->norm_x;
    float *moe_buf       = eng->moe_buf;
    uint32_t n_expert     = m->expert_count;
    uint32_t n_expert_used = m->expert_used_count;
    uint32_t ffn_dim      = m->expert_ffn_length;      // 512
    uint32_t sh_ffn_dim   = m->expert_shared_ffn_len;  // 512

    float *router_logits = eng->router_logits;
    int   *top_indices   = eng->top_indices;
    float *top_values    = eng->top_values;
    float *gate_exp      = eng->gate_exp;
    float *up_exp        = eng->up_exp;
    float *act_exp       = eng->act_exp;
    float *down_exp      = eng->down_exp;
    float *sh_gate_exp   = eng->sh_gate_exp;
    int   *exp_slots     = eng->exp_slots;
    float *gate_arr      = eng->gate_arr;
    float *up_arr        = eng->up_arr;
    float *act_arr       = eng->act_arr;
    float *down_arr      = eng->down_arr;
    float *down_hit      = eng->down_hit;
    float *sh_up_exp     = eng->sh_up_exp;
    block_q8_K *q8_normx = eng->q8_normx;
    const int exact      = tuff_exact();
    float *sh_act_exp    = eng->sh_act_exp;
    float *sh_down_exp   = eng->sh_down_exp;

    if (pos == 0 && dbg_logits_flag()) {
        const float *onw = (const float*)m->output_norm.data;
        fprintf(stderr, "[DBG] output_norm[0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                onw[0], onw[1], onw[2], onw[3], onw[4], onw[5], onw[6], onw[7]);
    }

    // Dequantize input token embedding
    const uint8_t *embd_base = (const uint8_t*)m->token_embd.data;
    size_t embd_row_bytes = ggml_row_size((ggml_type_t)m->token_embd.info.type, dim);
    dequantize_row((ggml_type_t)m->token_embd.info.type, embd_base + token_id * embd_row_bytes, x, dim);
    dump_f32_noil("model.input_embed", x, dim);

    if (pos == 0 && dbg_logits_flag()) {
        fprintf(stderr, "[DBG] embd[token=%d] row_bytes=%zu [0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                token_id, embd_row_bytes, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        double esum = 0; for (int i = 0; i < (int)dim; ++i) esum += x[i];
        fprintf(stderr, "[DBG] embd sum = %.3f\n", esum);
    }

    // -----------------------------------------------------------------------
    // P4b: Fast Path (Device-Resident Residual Stream)
    // Runs when neither reference mode nor debug tensor dump is requested.
    // -----------------------------------------------------------------------
    if (!exact && !cpu_delta_flag() && !cpu_attn_flag() && !g_dump_active) {
        gpu_residual_reset(x, (int)dim);

        float *d_res_x = gpu_get_d_res_x();
        const float *d_norm_x = gpu_get_d_norm_x();
        const float *d_shexp_out = gpu_get_shexp_dev_ptr();

        uint32_t attn_layer_idx = 0;
        uint32_t delta_layer_idx = 0;
        int cache_on = gpu_expert_cache_active();

        gpu_set_device_pos(pos);

        for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
            model_layer_t *layer = &m->layers[l];

            if (gpu_has_delta_graphs()) {
                gpu_launch_layer_graph(l);
                if (!tuff_exact() && mmvq_dual_enabled()) {
                    gpu_set_norm_q8_valid(1);
                }
                if (l == 0 && tuff_bgfill_overlap()) {
                    gpu_expert_cache_bgfill_poll();
                }
                if (tuff_gpu_router()) {
                    gpu_router_topk_copy_out(top_indices, top_values, norm_x, d_norm_x, (int)dim);
                } else {
                    gpu_router_logits_copy_out(router_logits, norm_x, d_norm_x, (int)dim, (int)n_expert);
                }
                gpu_shared_expert_dev((float*)d_shexp_out, layer, d_norm_x, (int)dim, (int)sh_ffn_dim);
                double tw0 = omp_get_wtime();
                if (tuff_h6() || tuff_gpu_router()) gpu_router_logits_wait_event(); else gpu_router_logits_wait();
                eng->t_expert_sync += omp_get_wtime() - tw0;
                if (layer->type == LAYER_TYPE_ATTENTION) {
                    attn_layer_idx++;
                } else {
                    delta_layer_idx++;
                }
            } else if (layer->type == LAYER_TYPE_ATTENTION) {
                double ta0 = omp_get_wtime();
                uint32_t head_dim  = m->key_length;
                uint32_t n_head    = m->head_count;
                uint32_t n_head_kv = m->head_count_kv;
                int ctx_len = pos + 1;
                float scale = 1.0f / sqrtf((float)head_dim);

                if (!tuff_exact() && mmvq_dual_enabled()) {
                    gpu_fused_rms_norm_quant_dev((float*)d_norm_x, d_res_x, (const float*)layer->attn_norm.dev, gpu_get_d_dense_x_q8(), (int)dim, eps);
                } else {
                    gpu_rms_norm_dev((float*)d_norm_x, d_res_x, (const float*)layer->attn_norm.dev, (int)dim, eps);
                }
                gpu_attention_layer_dev((int)attn_layer_idx, d_res_x, layer, d_norm_x,
                                        (int)m->n_attn_layers, (int)n_head, (int)n_head_kv, (int)head_dim,
                                        pos, ctx_len, eng->max_seq_len, scale, eps, (int)dim,
                                        (int)m->rope_dim_count, m->rope_dim_sections, m->rope_freq_base);
                if (!tuff_exact() && mmvq_dual_enabled()) {
                    gpu_fused_rms_norm_quant_dev((float*)d_norm_x, d_res_x, (const float*)layer->post_attention_norm.dev, gpu_get_d_norm_q8(), (int)dim, eps);
                    gpu_set_norm_q8_valid(1);
                } else {
                    gpu_rms_norm_dev((float*)d_norm_x, d_res_x, (const float*)layer->post_attention_norm.dev, (int)dim, eps);
                }
                if (l == 0 && tuff_bgfill_overlap()) {
                    eng->t_attn += omp_get_wtime() - ta0;
                    gpu_expert_cache_bgfill_poll();
                    ta0 = omp_get_wtime();
                }
                if (l < (uint32_t)g_n_gpu_expert_layers) {
                    gpu_router_logits_dev_only(&layer->ffn_gate_inp, d_norm_x, (int)dim, (int)n_expert);
                    gpu_shared_expert_dev((float*)d_shexp_out, layer, d_norm_x, (int)dim, (int)sh_ffn_dim);
                } else {
                    if (tuff_gpu_router()) {
                        gpu_router_topk_launch(top_indices, top_values, norm_x,
                                               &layer->ffn_gate_inp, d_norm_x, (int)dim, (int)n_expert);
                    } else {
                        gpu_router_logits_launch(router_logits, norm_x,
                                                 &layer->ffn_gate_inp, d_norm_x, (int)dim, (int)n_expert);
                    }
                    gpu_shared_expert_dev((float*)d_shexp_out, layer, d_norm_x, (int)dim, (int)sh_ffn_dim);
                    double tw0 = omp_get_wtime();
                    if (tuff_h6() || tuff_gpu_router()) gpu_router_logits_wait_event(); else gpu_router_logits_wait();
                    eng->t_expert_sync += omp_get_wtime() - tw0;
                }
                eng->t_attn += omp_get_wtime() - ta0;
            } else {
                double td0 = omp_get_wtime();
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
                    gpu_fused_rms_norm_quant_dev((float*)d_norm_x, d_res_x, (const float*)layer->attn_norm.dev, gpu_get_d_dense_x_q8(), (int)dim, eps);
                } else {
                    gpu_rms_norm_dev((float*)d_norm_x, d_res_x, (const float*)layer->attn_norm.dev, (int)dim, eps);
                }
                gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate,
                                          &layer->ssm_beta, &layer->ssm_alpha,
                                          d_norm_x, (int)dim);
                gpu_delta_fused_gdn_dev((int)delta_layer_idx, layer, d_res_x, eps, scale_q,
                                        (int)n_k_heads, (int)n_v_heads,
                                        (int)head_k_dim, (int)head_v_dim,
                                        (int)key_dim, (int)value_dim, (int)qkv_dim, (int)conv_kernel, (int)dim);
                if (!tuff_exact() && mmvq_dual_enabled()) {
                    gpu_fused_rms_norm_quant_dev((float*)d_norm_x, d_res_x, (const float*)layer->post_attention_norm.dev, gpu_get_d_norm_q8(), (int)dim, eps);
                    gpu_set_norm_q8_valid(1);
                } else {
                    gpu_rms_norm_dev((float*)d_norm_x, d_res_x, (const float*)layer->post_attention_norm.dev, (int)dim, eps);
                }
                if (l == 0 && tuff_bgfill_overlap()) {
                    eng->t_delta += omp_get_wtime() - td0;
                    gpu_expert_cache_bgfill_poll();
                    td0 = omp_get_wtime();
                }
                if (l < (uint32_t)g_n_gpu_expert_layers) {
                    gpu_router_logits_dev_only(&layer->ffn_gate_inp, d_norm_x, (int)dim, (int)n_expert);
                    gpu_shared_expert_dev((float*)d_shexp_out, layer, d_norm_x, (int)dim, (int)sh_ffn_dim);
                } else {
                    if (tuff_gpu_router()) {
                        gpu_router_topk_launch(top_indices, top_values, norm_x,
                                               &layer->ffn_gate_inp, d_norm_x, (int)dim, (int)n_expert);
                    } else {
                        gpu_router_logits_launch(router_logits, norm_x, &layer->ffn_gate_inp, d_norm_x, (int)dim, (int)n_expert);
                    }
                    gpu_shared_expert_dev((float*)d_shexp_out, layer, d_norm_x, (int)dim, (int)sh_ffn_dim);
                    double tw0 = omp_get_wtime();
                    if (tuff_h6() || tuff_gpu_router()) gpu_router_logits_wait_event(); else gpu_router_logits_wait();
                    eng->t_expert_sync += omp_get_wtime() - tw0;
                }
                eng->t_delta += omp_get_wtime() - td0;
                delta_layer_idx++;
            }

            double te0 = omp_get_wtime();

            const ggml_type_t down_type = (ggml_type_t)layer->ffn_down_exps.info.type;

            if (l < (uint32_t)g_n_gpu_expert_layers) {
                gpu_expert_resident_layer_dev((int)l, down_type, d_norm_x, d_res_x, d_shexp_out, (int)dim, (int)ffn_dim, (int)n_expert);
                eng->t_expert += omp_get_wtime() - te0;
                continue;
            }

            // 4. Top-k selection + softmax
            double ttk0 = omp_get_wtime();
            if (!tuff_gpu_router()) {
                for (uint32_t k = 0; k < n_expert_used; ++k) {
                    top_indices[k] = (int)k;
                    top_values[k] = router_logits[k];
                }
                for (uint32_t i = n_expert_used; i < n_expert; ++i) {
                    float val = router_logits[i];
                    uint32_t min_pos = 0;
                    float min_val = top_values[0];
                    for (uint32_t k = 1; k < n_expert_used; ++k) {
                        if (top_values[k] < min_val) {
                            min_val = top_values[k];
                            min_pos = k;
                        }
                    }
                    if (val > min_val) {
                        top_values[min_pos] = val;
                        top_indices[min_pos] = (int)i;
                    }
                }
                softmax(top_values, (int)n_expert_used);
            }

            // 5. Check cache hits / misses
            int hit_slots[8], hit_pos[8];
            int n_batch = 0;
            int miss_k[16], n_miss = 0;
            for (uint32_t k = 0; k < n_expert_used; ++k) {
                int slot = cache_on ? gpu_expert_lookup(l, top_indices[k]) : -1;
                if (slot >= 0) {
                    hit_slots[n_batch] = slot;
                    hit_pos[n_batch] = (int)k;
                    n_batch++;
                } else {
                    miss_k[n_miss++] = (int)k;
                }
            }

            eng->t_miss += omp_get_wtime() - ttk0;

            // 6. If CPU misses exist, submit to worker pool (norm_x is already on host!)
            double tpub0 = omp_get_wtime();
            if (n_miss > 0) {
                g_moe_batch.L      = layer;
                g_moe_batch.x      = norm_x;
                g_moe_batch.q8x    = q8_normx;
                g_moe_batch.dim    = (int)dim;
                g_moe_batch.ffn    = (int)ffn_dim;
                g_moe_batch.gtype  = (ggml_type_t)layer->ffn_gate_exps.info.type;
                g_moe_batch.dtype  = down_type;
                g_moe_batch.gate_bytes = ggml_row_size(g_moe_batch.gtype, dim) * ffn_dim;
                g_moe_batch.up_bytes   = g_moe_batch.gate_bytes;
                g_moe_batch.down_bytes = ggml_row_size(down_type, ffn_dim) * dim;
                for (int j = 0; j < n_miss; ++j) {
                    g_moe_batch.experts[j]  = top_indices[miss_k[j]];
                    g_moe_batch.down_out[j] = down_arr + (size_t)miss_k[j] * dim;
                    g_moe_batch.xk[j]       = norm_x;
                    g_moe_batch.q8xk[j]     = q8_normx;
                    g_moe_batch.xk2[j]      = NULL;   // H15: single-token path has no second column
                    g_moe_batch.q8xk2[j]    = NULL;
                    g_moe_batch.down_out2[j]= NULL;
                }
                g_moe_batch.n = n_miss;
                moe_pool_submit();
            }
            eng->t_cpumoe_pub += omp_get_wtime() - tpub0;

            // 7. GPU Routed Hit Experts (Shared expert was already computed in Phase 1!)
            double tgu0 = omp_get_wtime();
            if (n_batch > 0) {
                gpu_expert_moe_gateup_swiglu_dev(hit_slots, n_batch, d_norm_x, (int)dim, (int)ffn_dim);
                gpu_expert_moe_down_dev(n_batch, down_type, (int)ffn_dim, (int)dim);
            }
            eng->t_gu += omp_get_wtime() - tgu0;

            // 9. Join CPU miss workers and sum miss contributions
            float *host_miss_ptr = NULL;
            if (n_miss > 0) {
                double tw0 = omp_get_wtime();
                moe_pool_wait();
                eng->t_cpumoe += omp_get_wtime() - tw0;
                memset(moe_buf, 0, dim * sizeof(float));
                for (int j = 0; j < n_miss; ++j) {
                    int k = miss_k[j];
                    float w = top_values[k];
                    const float *dd = down_arr + (size_t)k * dim;
                    __m256 vw = _mm256_set1_ps(w);
                    for (uint32_t i = 0; i < dim; i += 8) {
                        __m256 mb = _mm256_loadu_ps(moe_buf + i);
                        __m256 vd = _mm256_loadu_ps(dd + i);
                        mb = _mm256_fmadd_ps(vw, vd, mb);
                        _mm256_storeu_ps(moe_buf + i, mb);
                    }
                }
                host_miss_ptr = moe_buf;
            }

            // 10. Fused MoE Combine & Residual Add directly into d_res_x on GPU
            double tres0 = omp_get_wtime();
            gpu_moe_combine_accum(d_res_x, hit_pos, n_batch, top_values, d_shexp_out, host_miss_ptr, (int)dim);
            eng->t_resid += omp_get_wtime() - tres0;

            // 11. Queue background fills
            double tf0 = omp_get_wtime();
            if (cache_on && g_fill_budget > 0 && (l % (uint32_t)g_fill_stride) == 0) {
                for (int j = 0, filled = 0; j < n_miss && filled < g_fill_budget; ++j)
                    filled += gpu_expert_cache_bgfill(m, l, top_indices[miss_k[j]]);
            }
            eng->t_fill += omp_get_wtime() - tf0;
            eng->t_expert += omp_get_wtime() - te0;
        }

        // 12. Commit background fills at token boundary
        // H31: with TUFF_BGFILL_OVERLAP=1 the commit moved into layer 0 of the next
        // token, overlapping the fill-event wait with that layer's attention kernels.
        if (!tuff_bgfill_overlap()) gpu_expert_cache_bgfill_poll();

        // 13. Final output norm on GPU & 14. Output projection and GPU argmax / stochastic sample
        int32_t best_token = 0;
        if (eng->sampler.params.temp <= 0.0f) {
            // Zero-overhead greedy fast-path
            float *capture_ptr = NULL;
            float *captured_logits = NULL;
            if (tuff_logit_capture_is_enabled() || dbg_logits_flag()) {
                captured_logits = (float*)malloc((size_t)vocab_size * sizeof(float));
                capture_ptr = captured_logits;
            }
            if (!tuff_exact() && mmvq_dual_enabled()) {
                best_token = gpu_fused_output_norm_quant_argmax_dev(
                    &m->output_weight, d_res_x, (const float*)m->output_norm.dev, (int)dim, (int)vocab_size, eps, capture_ptr);
            } else {
                gpu_rms_norm_dev((float*)d_norm_x, d_res_x, (const float*)m->output_norm.dev, (int)dim, eps);
                best_token = gpu_output_argmax_dev(&m->output_weight, d_norm_x, (int)dim, (int)vocab_size, capture_ptr);
            }
            if (capture_ptr) {
                tuff_logit_capture(capture_ptr, vocab_size);
                free(captured_logits);
            }
            sampler_accept_token(&eng->sampler, best_token);
        } else {
            // Dedicated stochastic sampling path
            if (!eng->host_logits) {
                eng->host_logits = (float*)malloc((size_t)vocab_size * sizeof(float));
            }
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_output_norm_quant_logits_dev(
                    &m->output_weight, d_res_x, (const float*)m->output_norm.dev, (int)dim, (int)vocab_size, eps, eng->host_logits);
            } else {
                gpu_rms_norm_dev((float*)d_norm_x, d_res_x, (const float*)m->output_norm.dev, (int)dim, eps);
                gpu_output_logits_dev(&m->output_weight, d_norm_x, (int)dim, (int)vocab_size, eng->host_logits);
            }
            if (tuff_logit_capture_is_enabled() || dbg_logits_flag()) {
                tuff_logit_capture(eng->host_logits, vocab_size);
            }
            best_token = sampler_sample(&eng->sampler, eng->host_logits, (int32_t)vocab_size);
        }

        // 15. Native MTP speculative draft for token t+1.
        // Only when speculation is actually requested: the MTP head costs a full attention block,
        // a shared expert and a second 248320-row Q6_K output projection per token, which is
        // dead weight in ordinary decode. H17: the n-gram source is free (no model evaluation).
        engine_set_draft(eng, best_token, pos, d_res_x);

        eng->t_total += omp_get_wtime() - t_token;
        return best_token;
    }

    uint32_t attn_layer_idx = 0;
    uint32_t delta_layer_idx = 0;

    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];

        rms_norm(norm_x, x, (const float*)layer->attn_norm.data, dim, eps);
        dump_f32("attn_norm", (int)l, norm_x, dim);

        float *attn_out = (float*)calloc(dim, sizeof(float));

        if (layer->type == LAYER_TYPE_ATTENTION) {
            double ta0 = omp_get_wtime();
            // Qwen3.5/3.6 full-attention layer. attn_q is a FUSED [Q(head_dim) | gate(head_dim)]
            // projection per head (2*head_dim*n_head output columns) — see
            // llama.cpp/src/models/qwen35moe.cpp build_layer_attn(): Qcur_full is split into a
            // Qcur view and a gate view, and the attention context is multiplied by
            // sigmoid(gate) before the output projection. This is NOT plain GQA with
            // n_head = attn_q_out_dim/head_dim.
            uint32_t head_dim  = m->key_length;    // 256 (== value_length)
            uint32_t n_head    = m->head_count;    // 16
            uint32_t n_head_kv = m->head_count_kv; // 2

            uint32_t q_full_dim = (uint32_t)layer->attn_q.info.dims[1];
            uint32_t k_dim      = (uint32_t)layer->attn_k.info.dims[1];
            uint32_t v_dim      = (uint32_t)layer->attn_v.info.dims[1];

            if (q_full_dim != 2 * head_dim * n_head || k_dim != head_dim * n_head_kv || v_dim != head_dim * n_head_kv) {
                fprintf(stderr, "[FATAL ERROR] layer %u: unexpected attention tensor shapes "
                                 "(q=%u expected %u, k=%u v=%u expected %u)\n",
                        l, q_full_dim, 2 * head_dim * n_head, k_dim, v_dim, head_dim * n_head_kv);
                exit(EXIT_FAILURE);
            }

            float *q_full = (float*)malloc(q_full_dim * sizeof(float));
            float *k = (float*)malloc(k_dim * sizeof(float));
            float *v = (float*)malloc(v_dim * sizeof(float));

            mat_vec(q_full, &layer->attn_q, norm_x);
            mat_vec(k, &layer->attn_k, norm_x);
            mat_vec(v, &layer->attn_v, norm_x);

            float *q = (float*)malloc((size_t)n_head * head_dim * sizeof(float));
            float *gate = (float*)malloc((size_t)n_head * head_dim * sizeof(float));
            for (uint32_t h = 0; h < n_head; ++h) {
                memcpy(q + (size_t)h * head_dim, q_full + (size_t)h * 2 * head_dim, head_dim * sizeof(float));
                memcpy(gate + (size_t)h * head_dim, q_full + (size_t)h * 2 * head_dim + head_dim, head_dim * sizeof(float));
            }

            for (uint32_t h = 0; h < n_head; ++h) {
                rms_norm(q + (size_t)h * head_dim, q + (size_t)h * head_dim,
                          (const float*)layer->attn_q_norm.data, head_dim, eps);
            }
            for (uint32_t h = 0; h < n_head_kv; ++h) {
                rms_norm(k + (size_t)h * head_dim, k + (size_t)h * head_dim,
                          (const float*)layer->attn_k_norm.data, head_dim, eps);
            }

            apply_mrope(q, k, (int)n_head, (int)n_head_kv, (int)head_dim, (int)m->rope_dim_count,
                        pos, m->rope_dim_sections, m->rope_freq_base);

            for (uint32_t kv_h = 0; kv_h < n_head_kv; ++kv_h) {
                size_t off = ((size_t)attn_layer_idx * n_head_kv * eng->max_seq_len +
                              (size_t)kv_h * eng->max_seq_len + pos) * head_dim;
                memcpy(eng->k_cache + off, k + (size_t)kv_h * head_dim, head_dim * sizeof(float));
                memcpy(eng->v_cache + off, v + (size_t)kv_h * head_dim, head_dim * sizeof(float));
            }

            float *concat_out = (float*)malloc((size_t)n_head * head_dim * sizeof(float));
            uint32_t gqa_ratio = (n_head_kv > 0) ? (n_head / n_head_kv) : 1;
            int ctx_len = pos + 1;
            float scale = 1.0f / sqrtf((float)head_dim);

            if (cpu_attn_flag()) {
                // CPU reference path (A/B verification): scalar per-head loop, bit-identical to
                // the CUDA attention kernel below.
                // Sized by max_seq_len (not a fixed 512), since ctx_len can grow up to max_seq_len.
                float *scores = (float*)malloc((size_t)eng->max_seq_len * sizeof(float));

                for (uint32_t h = 0; h < n_head; ++h) {
                    uint32_t kv_h = h / gqa_ratio;
                    float *qh = q + (size_t)h * head_dim;

                    for (int t = 0; t < ctx_len; ++t) {
                        size_t k_pos_off = ((size_t)attn_layer_idx * n_head_kv * eng->max_seq_len +
                                            (size_t)kv_h * eng->max_seq_len + t) * head_dim;
                        const float *kt = eng->k_cache + k_pos_off;
                        float dotp = 0.0f;
                        for (uint32_t d = 0; d < head_dim; ++d) dotp += qh[d] * kt[d];
                        scores[t] = dotp * scale;
                    }

                    softmax(scores, ctx_len);

                    float *head_out = concat_out + (size_t)h * head_dim;
                    memset(head_out, 0, head_dim * sizeof(float));
                    for (int t = 0; t < ctx_len; ++t) {
                        size_t v_pos_off = ((size_t)attn_layer_idx * n_head_kv * eng->max_seq_len +
                                            (size_t)kv_h * eng->max_seq_len + t) * head_dim;
                        const float *vt = eng->v_cache + v_pos_off;
                        float weight = scores[t];
                        for (uint32_t d = 0; d < head_dim; ++d) head_out[d] += weight * vt[d];
                    }
                }
                free(scores);
            } else {
                // GPU attention: device-resident KV cache + bit-exact scoring/softmax/weighted-V.
                gpu_attention((int)attn_layer_idx, concat_out, q, k, v,
                              (int)m->n_attn_layers, (int)n_head, (int)n_head_kv, (int)head_dim,
                              (int)pos, ctx_len, eng->max_seq_len, scale);
            }

            // Gate the attention context with sigmoid(gate) before the output projection.
            for (uint32_t i = 0; i < n_head * head_dim; ++i) {
                concat_out[i] *= 1.0f / (1.0f + expf(-gate[i]));
            }

            mat_vec(attn_out, &layer->attn_output, concat_out);

            free(q_full); free(k); free(v); free(q); free(gate); free(concat_out);
            eng->t_attn += omp_get_wtime() - ta0;
            attn_layer_idx++;
        } else {
            double td0 = omp_get_wtime();
            // Gated Delta-Net (linear attention) layer. Spec: llama.cpp
            // src/models/delta-net-base.cpp build_delta_net_autoregressive(), wired up in
            // src/models/qwen35moe.cpp build_layer_attn_linear().
            uint32_t key_dim     = m->ssm_key_dim;     // 2048
            uint32_t value_dim   = m->ssm_value_dim;   // 4096
            uint32_t qkv_dim     = m->ssm_qkv_dim;     // 8192
            uint32_t n_k_heads   = m->ssm_n_k_heads;   // 16
            uint32_t n_v_heads   = m->ssm_n_v_heads;   // 32
            uint32_t head_k_dim  = m->ssm_head_k_dim;  // 128
            uint32_t head_v_dim  = m->ssm_head_v_dim;  // 128
            uint32_t conv_kernel = m->ssm_conv_kernel; // 4
            float scale_q = 1.0f / sqrtf((float)head_k_dim);

            if (cpu_delta_flag()) {
                // =================================================================
                // TUFF_CPU_DELTA=1 reference path, verbatim. All host buffers and the
                // per-layer host elementwise work are retained exactly as before.
                // =================================================================
                float *qkv_mixed = (float*)malloc(qkv_dim * sizeof(float));
                float *z         = (float*)malloc(value_dim * sizeof(float));
                float *beta_raw  = (float*)malloc(n_v_heads * sizeof(float));
                float *alpha_raw = (float*)malloc(n_v_heads * sizeof(float));
                float *beta      = (float*)malloc(n_v_heads * sizeof(float));
                float *gate      = (float*)malloc(n_v_heads * sizeof(float));

                double td_mat = omp_get_wtime();
                gpu_delta_projections(qkv_mixed, &layer->attn_qkv,
                                      z, &layer->attn_gate,
                                      beta_raw, &layer->ssm_beta,
                                      alpha_raw, &layer->ssm_alpha,
                                      norm_x, (int)dim, 0);
                eng->t_d_mat += omp_get_wtime() - td_mat;
                double td_conv = omp_get_wtime();

                sigmoid(beta, beta_raw, (int)n_v_heads);

                const float *dt_bias = (const float*)layer->ssm_dt_bias.data;
                // ssm_a, per llama.cpp build_layer_attn_linear comment, is stored pre-negated
                // and pre-exponentiated: ssm_a == -exp(A_log). gate = softplus(alpha+bias) * ssm_a,
                // and decay = exp(gate) is applied later — matches build_delta_net_autoregressive's
                // `g = ggml_exp(g); s = ggml_mul(s, g);`.
                const float *ssm_a = (const float*)layer->ssm_a.data;

                for (uint32_t h = 0; h < n_v_heads; ++h) {
                    float alpha_biased = alpha_raw[h] + dt_bias[h];
                    float softp = (alpha_biased > 20.0f) ? alpha_biased : logf(1.0f + expf(alpha_biased));
                    gate[h] = softp * ssm_a[h];
                }

                float *c_state = eng->conv_state + (size_t)delta_layer_idx * conv_kernel * qkv_dim;
                memmove(c_state, c_state + qkv_dim, (size_t)(conv_kernel - 1) * qkv_dim * sizeof(float));
                memcpy(c_state + (size_t)(conv_kernel - 1) * qkv_dim, qkv_mixed, qkv_dim * sizeof(float));

                float *qkv_conv = (float*)calloc(qkv_dim, sizeof(float));
                const float *conv_w = (const float*)layer->ssm_conv1d.data;

                for (uint32_t c = 0; c < qkv_dim; ++c) {
                    float sum = 0.0f;
                    for (uint32_t k_idx = 0; k_idx < conv_kernel; ++k_idx) {
                        sum += c_state[k_idx * qkv_dim + c] * conv_w[c * conv_kernel + k_idx];
                    }
                    qkv_conv[c] = sum / (1.0f + expf(-sum)); // SiLU
                }
                dump_f32("conv_output_silu", (int)l, qkv_conv, qkv_dim);

                const float *q_conv = qkv_conv;                  // (n_k_heads, head_k_dim)
                const float *k_conv = qkv_conv + key_dim;         // (n_k_heads, head_k_dim)
                const float *v_conv = qkv_conv + 2 * key_dim;     // (n_v_heads, head_v_dim)

                float *q_norm = (float*)malloc(key_dim * sizeof(float));
                float *k_norm = (float*)malloc(key_dim * sizeof(float));
                memcpy(q_norm, q_conv, key_dim * sizeof(float));
                memcpy(k_norm, k_conv, key_dim * sizeof(float));

                for (uint32_t h = 0; h < n_k_heads; ++h) {
                    l2_norm(q_norm + (size_t)h * head_k_dim, head_k_dim, eps);
                    l2_norm(k_norm + (size_t)h * head_k_dim, head_k_dim, eps);
                }

                float *qk_pre_norm = (float*)malloc(2 * key_dim * sizeof(float));
                memcpy(qk_pre_norm, q_norm, key_dim * sizeof(float));
                memcpy(qk_pre_norm + key_dim, k_norm, key_dim * sizeof(float));
                dump_f32("qk_pre_norm", (int)l, qk_pre_norm, 2 * key_dim);
                free(qk_pre_norm);
                dump_f32("q_norm", (int)l, q_norm, key_dim);
                dump_f32("k_norm", (int)l, k_norm, key_dim);
                dump_f32("v_conv", (int)l, v_conv, value_dim);
                dump_f32("beta_sigmoid", (int)l, beta, n_v_heads);
                dump_f32("gate", (int)l, gate, n_v_heads);
                dump_f32("z", (int)l, z, value_dim);

                float *dnet_out = (float*)malloc(value_dim * sizeof(float));
                float *dnet_S = eng->delta_state + (size_t)delta_layer_idx * n_v_heads * head_v_dim * head_k_dim;

                dump_f32("state_predelta", (int)l, dnet_S, (size_t)n_v_heads * head_v_dim * head_k_dim);

                // CPU reference path (A/B verification): bit-identical to the CUDA kernel
                // below. Each (head,row) is independent; decay/oh accumulate j in 0..head_k_dim-1.
                float *sk = (float*)malloc(head_v_dim * sizeof(float));
                float *d  = (float*)malloc(head_v_dim * sizeof(float));

                for (uint32_t h = 0; h < n_v_heads; ++h) {
                    // q_conv/k_conv only have n_k_heads groups; llama.cpp repeats them up to
                    // n_v_heads via ggml_repeat_4d, which TILES the source array
                    // (dst[t] = src[t % n_k_heads]) rather than repeat-interleaving it — see
                    // ggml_compute_forward_repeat_f32 in ggml/src/ggml-cpu/ops.cpp (nested loop:
                    // outer i1 in [0,nr1), inner k1 in [0,ne01), dst[i1*ne01+k1] = src[k1]).
                    uint32_t hk = h % n_k_heads;
                    const float *qh = q_norm + (size_t)hk * head_k_dim;
                    const float *kh = k_norm + (size_t)hk * head_k_dim;
                    const float *vh = v_conv + (size_t)h * head_v_dim;
                    float decay = expf(gate[h]);
                    float b_val = beta[h];

                    float *S = dnet_S + (size_t)h * head_v_dim * head_k_dim;

                    for (uint32_t i = 0; i < head_v_dim * head_k_dim; ++i) S[i] *= decay;

                    for (uint32_t i = 0; i < head_v_dim; ++i) {
                        float sum = 0.0f;
                        for (uint32_t j = 0; j < head_k_dim; ++j) sum += S[i * head_k_dim + j] * kh[j];
                        sk[i] = sum;
                    }

                    for (uint32_t i = 0; i < head_v_dim; ++i) d[i] = (vh[i] - sk[i]) * b_val;

                    for (uint32_t i = 0; i < head_v_dim; ++i) {
                        for (uint32_t j = 0; j < head_k_dim; ++j) {
                            S[i * head_k_dim + j] += d[i] * kh[j];
                        }
                    }

                    float *oh = dnet_out + (size_t)h * head_v_dim;
                    for (uint32_t i = 0; i < head_v_dim; ++i) {
                        float sum = 0.0f;
                        for (uint32_t j = 0; j < head_k_dim; ++j) sum += S[i * head_k_dim + j] * (qh[j] * scale_q);
                        oh[i] = sum;
                    }
                }
                free(sk); free(d);
                eng->t_d_conv += omp_get_wtime() - td_conv;
                double td_norm = omp_get_wtime();
                dump_f32("new_state", (int)l, dnet_S, (size_t)n_v_heads * head_v_dim * head_k_dim);
                dump_f32("dnet_out", (int)l, dnet_out, value_dim);

                const float *ssm_norm_w = (const float*)layer->ssm_norm.data;
                // CPU reference path for the fused out-stage: per-head rms_norm, silu(z) gating,
                // then the ssm_out mat-vec (identical math to the CUDA gpu_delta_post).
                float *gated_norm = (float*)malloc(value_dim * sizeof(float));
                for (uint32_t h = 0; h < n_v_heads; ++h) {
                    rms_norm(gated_norm + (size_t)h * head_v_dim, dnet_out + (size_t)h * head_v_dim,
                              ssm_norm_w, head_v_dim, eps);
                }
                for (uint32_t i = 0; i < value_dim; ++i) {
                    float zi = z[i];
                    float silu_z = zi / (1.0f + expf(-zi));
                    gated_norm[i] *= silu_z;
                }
                dump_f32("gated_norm", (int)l, gated_norm, value_dim);

                double td_out = omp_get_wtime();
                mat_vec(attn_out, &layer->ssm_out, gated_norm);
                eng->t_d_out += omp_get_wtime() - td_out;
                free(gated_norm);
                eng->t_d_norm += omp_get_wtime() - td_norm;
                dump_f32("attn_output", (int)l, attn_out, dim);

                free(qkv_mixed); free(z); free(beta_raw); free(alpha_raw);
                free(beta); free(gate); free(qkv_conv); free(q_norm);
                free(k_norm); free(dnet_out);
            } else {
                // =================================================================
                // P4a fast path: the whole delta layer runs on device end to end with a
                // single host round-trip (the final attn_out download in gpu_delta_post).
                // gpu_delta_pre consumes g_dpqkv/g_dpb/g_dpa kept resident by the projections
                // (keep_dev=1) and produces q_norm/k_norm/v_conv/decay/beta in device scratch;
                // gpu_delta_state_update_dev reads those directly, skipping the five H2D
                // uploads. No host qkv_mixed/z/beta_raw/alpha_raw/beta/gate/decay/qkv_conv/
                // q_norm/k_norm buffers are allocated on this path.
                // =================================================================
                double td_mat = omp_get_wtime();
                gpu_delta_projections(NULL, &layer->attn_qkv, NULL, &layer->attn_gate,
                                      NULL, &layer->ssm_beta, NULL, &layer->ssm_alpha,
                                      norm_x, (int)dim, 1);
                eng->t_d_mat += omp_get_wtime() - td_mat;
                double td_conv = omp_get_wtime();
                gpu_delta_pre((int)delta_layer_idx, layer, eps,
                              (int)n_k_heads, (int)n_v_heads,
                              (int)head_k_dim, (int)head_v_dim,
                              (int)key_dim, (int)value_dim, (int)qkv_dim, (int)conv_kernel);

                // On-demand dumps (only when TUFF_DUMP_DIR is set): download the device
                // tensors here rather than keeping unconditional host copies.
                if (g_dump_active) {
                    gpu_delta_dev_t d; gpu_delta_get_device(&d);
                    float *tmp = (float*)malloc(qkv_dim * sizeof(float));
                    float *qk  = (float*)malloc(2 * key_dim * sizeof(float));
                    gpu_copy_dev_f32(d.qkv_conv, tmp, qkv_dim);
                    dump_f32("conv_output_silu", (int)l, tmp, qkv_dim);
                    gpu_copy_dev_f32(d.q_norm, tmp, key_dim);
                    dump_f32("q_norm", (int)l, tmp, key_dim);
                    gpu_copy_dev_f32(d.k_norm, tmp, key_dim);
                    dump_f32("k_norm", (int)l, tmp, key_dim);
                    gpu_copy_dev_f32(d.v_conv, tmp, value_dim);
                    dump_f32("v_conv", (int)l, tmp, value_dim);
                    gpu_copy_dev_f32(d.beta, tmp, n_v_heads);
                    dump_f32("beta_sigmoid", (int)l, tmp, n_v_heads);
                    gpu_copy_dev_f32(d.gate, tmp, n_v_heads);
                    dump_f32("gate", (int)l, tmp, n_v_heads);
                    gpu_copy_dev_f32(d.z, tmp, value_dim);
                    dump_f32("z", (int)l, tmp, value_dim);
                    gpu_copy_dev_f32(d.q_norm, qk, key_dim);
                    gpu_copy_dev_f32(d.k_norm, qk + key_dim, key_dim);
                    dump_f32("qk_pre_norm", (int)l, qk, 2 * key_dim);
                    free(qk); free(tmp);
                    // state before the update
                    size_t sfl = (size_t)n_v_heads * head_v_dim * head_k_dim;
                    float *st = (float*)malloc(sfl * sizeof(float));
                    gpu_copy_dev_f32(d.state + (size_t)delta_layer_idx * sfl, st, sfl);
                    dump_f32("state_predelta", (int)l, st, sfl);
                    free(st);
                }
                eng->t_d_conv += omp_get_wtime() - td_conv;
                double td_state = omp_get_wtime();
                gpu_delta_state_update_dev((int)delta_layer_idx, NULL, scale_q,
                                           (int)n_v_heads, (int)n_k_heads,
                                           (int)head_v_dim, (int)head_k_dim);
                eng->t_d_state += omp_get_wtime() - td_state;
                double td_norm = omp_get_wtime();
                if (g_dump_active) {
                    gpu_delta_dev_t d; gpu_delta_get_device(&d);
                    size_t sfl = (size_t)n_v_heads * head_v_dim * head_k_dim;
                    float *st = (float*)malloc(sfl * sizeof(float));
                    gpu_copy_dev_f32(d.state + (size_t)delta_layer_idx * sfl, st, sfl);
                    dump_f32("new_state", (int)l, st, sfl);
                    free(st);
                }
                // Iteration-9 fused CUDA out-stage: rms_norm + silu(z) + ssm_out all on device;
                // the single host download of the delta layer (attn_out) happens here.
                double td_out = omp_get_wtime();
                gpu_delta_post(attn_out, &layer->ssm_out,
                               (const float*)layer->ssm_norm.dev, eps,
                               (int)n_v_heads, (int)head_v_dim, (int)value_dim, (int)dim);
                eng->t_d_out += omp_get_wtime() - td_out;
                eng->t_d_norm += omp_get_wtime() - td_norm;
                dump_f32("attn_output", (int)l, attn_out, dim);
            }
            eng->t_delta += omp_get_wtime() - td0;

            delta_layer_idx++;
        }

        for (uint32_t i = 0; i < dim; ++i) x[i] += attn_out[i];
        free(attn_out);

        rms_norm(norm_x, x, (const float*)layer->post_attention_norm.data, dim, eps);
        double te0 = omp_get_wtime();

        // MoE router: softmax(logits) over all experts, then top-k selection, then
        // renormalize the selected weights to sum to 1 (llama-graph.cpp build_moe_ffn,
        // gating_op=SOFTMAX, norm_w=true). Selecting top-k by raw logit first and then
        // softmaxing only those k values is algebraically identical (the common
        // sum-over-all-experts denominator cancels in the renormalization), so this
        // stays a single pass without materializing the full softmax.
        mat_vec(router_logits, &layer->ffn_gate_inp, norm_x);
        dump_f32("ffn_moe_logits", (int)l, router_logits, n_expert);

        double ttopk0 = omp_get_wtime();
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            top_indices[k] = (int)k;
            top_values[k] = router_logits[k];
        }

        for (uint32_t i = n_expert_used; i < n_expert; ++i) {
            float val = router_logits[i];
            uint32_t min_pos = 0;
            float min_val = top_values[0];
            for (uint32_t k = 1; k < n_expert_used; ++k) {
                if (top_values[k] < min_val) {
                    min_val = top_values[k];
                    min_pos = k;
                }
            }
            if (val > min_val) {
                top_values[min_pos] = val;
                top_indices[min_pos] = (int)i;
            }
        }

        softmax(top_values, (int)n_expert_used);
        dump_f32("ffn_moe_weights", (int)l, top_values, n_expert_used);
        {
            int *sel = (int*)malloc(n_expert_used * sizeof(int));
            memcpy(sel, top_indices, n_expert_used * sizeof(int));
            dump_f32_noil("ffn_moe_topk_sel", (const float*)sel, n_expert_used);
            free(sel);
        }

        memset(moe_buf, 0, dim * sizeof(float));

        // M6: per-expert LFRU cache (gate/up Q4_K; down Q4_K or Q6_K per layer; all bit-exact on
        // GPU). ALL experts run on the GPU. Any MISS just fills a slot first (a one-time upload
        // for future tokens) instead of computing on the CPU, then all gate/up + down run as
        // batched kernels: launch all gate/up, sync+download once; launch all down, sync+download
        // once — 2 stream syncs per layer. TUFF_CPU_EXPERT=1 falls back to the all-CPU path.

        int *slots = exp_slots;
        // Iteration-12 within-layer CPU/GPU overlap: the router picks 8 experts; LFRU-resident
        // (hit) experts run on the GPU (fused batch kernels), the miss experts run on a CPU worker
        // thread (fused mat_vec_cpu) -- concurrent within the layer's MoE stage. This removes the
        // per-token PCIe miss-fill upload (misses are computed on CPU, not streamed) and hides the
        // CPU expert work behind the GPU's latency-bound expert batch. Both paths are bit-exact, so
        // combining down_arr in k order is oracle-identical. TUFF_CPU_EXPERT=1 -> all CPU.
        const ggml_type_t down_type = (ggml_type_t)layer->ffn_down_exps.info.type;
        int cache_on = gpu_expert_cache_active();
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            slots[k] = cache_on ? gpu_expert_lookup(l, top_indices[k]) : -1;
        }
        eng->t_miss += omp_get_wtime() - ttopk0;

        int miss_k[16], n_miss = 0;
        for (uint32_t k = 0; k < n_expert_used; ++k)
            if (slots[k] < 0) miss_k[n_miss++] = (int)k;

        if (exact) {
            // Bit-exact reference path (TUFF_EXACT=1), unchanged: a miss is uploaded (async,
            // gate/up + down), its slot claimed, and it runs on the GPU batch THIS token. This is
            // the 21 ms/token PCIe stall P1 removes -- it is kept here precisely so the numeric
            // gate has an untouched reference to compare against.
            if (cache_on) {
                double tpref0 = omp_get_wtime();
                for (int j = 0; j < n_miss; ++j) {
                    const uint32_t k = (uint32_t)miss_k[j];
                    int slot2 = gpu_expert_cache_alloc_slot(l, top_indices[k]);
                    gpu_expert_cache_fill_gateup(slot2, m, l, top_indices[k]);
                    gpu_expert_cache_fill_down(slot2, m, l, top_indices[k]);
                    slots[k] = slot2;
                }
                n_miss = 0;
                eng->t_pref += omp_get_wtime() - tpref0;
            }
        } else if (n_miss > 0) {
            // P1 fast path: hand the misses to the CPU worker pool and keep going.
            double tcpu0 = omp_get_wtime();
            g_moe_batch.L      = layer;
            g_moe_batch.x      = norm_x;
            g_moe_batch.q8x    = q8_normx;
            g_moe_batch.dim    = (int)dim;
            g_moe_batch.ffn    = (int)ffn_dim;
            g_moe_batch.gtype  = (ggml_type_t)layer->ffn_gate_exps.info.type;
            g_moe_batch.dtype  = down_type;
            g_moe_batch.gate_bytes = ggml_row_size(g_moe_batch.gtype, dim) * ffn_dim;
            g_moe_batch.up_bytes   = g_moe_batch.gate_bytes;
            g_moe_batch.down_bytes = ggml_row_size(down_type, ffn_dim) * dim;
            for (int j = 0; j < n_miss; ++j) {
                g_moe_batch.experts[j]  = top_indices[miss_k[j]];
                g_moe_batch.down_out[j] = down_arr + (size_t)miss_k[j] * dim;
                g_moe_batch.xk[j]       = g_moe_batch.x;
                g_moe_batch.q8xk[j]     = g_moe_batch.q8x;
                g_moe_batch.xk2[j]      = NULL;   // H15: single-token path has no second column
                g_moe_batch.q8xk2[j]    = NULL;
                g_moe_batch.down_out2[j]= NULL;
            }
            g_moe_batch.n = n_miss;
            moe_pool_submit();
            eng->t_cpumoe_pub += omp_get_wtime() - tcpu0;
        }

        // Shared-expert gate/up/gate_inp all take norm_x as input -> batch into one upload/launch/
        // download (bit-identical, removes 2 redundant norm_x round-trips per layer).
        float sh_gate_logit;
        gpu_shared_expert_projections(sh_gate_exp, &layer->ffn_gate_shexp,
                                      sh_up_exp, &layer->ffn_up_shexp,
                                      &sh_gate_logit, &layer->ffn_gate_inp_shexp,
                                      norm_x, (int)dim);
        swiglu(sh_act_exp, sh_gate_exp, sh_up_exp, sh_ffn_dim);
        mat_vec(sh_down_exp, &layer->ffn_down_shexp, sh_act_exp);
        dump_f32("ffn_shexp", (int)l, sh_down_exp, dim);
        float sh_gate_val = 1.0f / (1.0f + expf(-sh_gate_logit));
        {
            float g = sh_gate_val;
            dump_f32_noil("sh_gate_val", &g, 1);
        }

        // GPU: batched routed experts for the LFRU-resident (hit) experts. Under TUFF_EXACT=1
        // the freshly-filled misses are in here too, so the upload stream must be synced first.
        // On the P1 fast path the decode thread issues no uploads at all: background fills target
        // reserved slots that are unreachable from g_expert_map until their event has completed,
        // so there is nothing to wait for here (see gpu.cu's bgfill race-freedom note).
        if (exact) gpu_expert_cache_sync();
        int n_batch = 0;
        int hit_slots[8], hit_pos[8]; // compacted resident slot list (max 8 experts)
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            if (slots[k] >= 0) { hit_slots[n_batch] = slots[k]; hit_pos[n_batch] = (int)k; n_batch++; }
        }
        if (n_batch > 0) {
            double tgu0 = omp_get_wtime();
            if (tuff_exact()) {
                // Bit-exact scalar path (TUFF_EXACT=1): gate/up on GPU, SwiGLU on the host,
                // act re-uploaded for the down projection.
                gpu_expert_batch_gateup_launch_fused(hit_slots, n_batch, norm_x, (int)dim, (int)ffn_dim);
                gpu_expert_batch_gateup_sync_download(gate_arr, up_arr, n_batch, (int)ffn_dim);
                for (int j = 0; j < n_batch; ++j) {
                    swiglu(act_arr + (size_t)j * ffn_dim, gate_arr + (size_t)j * ffn_dim,
                           up_arr + (size_t)j * ffn_dim, ffn_dim);
                }
                eng->t_gu += omp_get_wtime() - tgu0;
                double tdown0 = omp_get_wtime();
                gpu_expert_batch_down_launch_fused(n_batch, down_type, act_arr, (int)ffn_dim, (int)dim);
                gpu_expert_batch_down_sync_download(hit_slots, down_hit, n_batch, (int)dim);
                eng->t_down += omp_get_wtime() - tdown0;
            } else {
                // P3b: gate+up+SwiGLU fused into one mmvq launch; the activation stays on device
                // (already q8_1) and feeds the down projection directly.
                gpu_expert_moe_gateup_swiglu(hit_slots, n_batch, norm_x, (int)dim, (int)ffn_dim);
                eng->t_gu += omp_get_wtime() - tgu0;
                double tdown0 = omp_get_wtime();
                gpu_expert_moe_down(n_batch, down_type, (int)ffn_dim, (int)dim, down_hit);
                eng->t_down += omp_get_wtime() - tdown0;
            }
            // Scatter the compacted GPU downs into down_arr at their original k positions so the
            // combine below walks k in the exact CPU accumulation order.
            for (int j = 0; j < n_batch; ++j)
                memcpy(down_arr + (size_t)hit_pos[j] * dim, down_hit + (size_t)j * dim,
                       (size_t)dim * sizeof(float));
        }

        // P1: join the CPU miss workers. Their down vectors already sit at their k positions in
        // down_arr, so the combine below walks k in the unchanged accumulation order.
        if (n_miss > 0) {
            double tw0 = omp_get_wtime();
            moe_pool_wait();
            eng->t_cpumoe += omp_get_wtime() - tw0;
        }

        // P1: warm the cache in the background. Rate-limited to TUFF_FILL_BUDGET experts per
        // layer per token and never awaited; the decode thread only reserves a slot and queues
        // the request. Issued here, after gpu_expert_moe_down() has synced g_expert_stream, so no
        // kernel can still be reading the pool when a victim slot is chosen.
        if (cache_on && !exact && g_fill_budget > 0) {
            double tf0 = omp_get_wtime();
            for (int j = 0, filled = 0; j < n_miss && filled < g_fill_budget; ++j)
                filled += gpu_expert_cache_bgfill(m, l, top_indices[miss_k[j]]);
            eng->t_pref += omp_get_wtime() - tf0;
        }

        // Combine the routed contributions into moe_buf in k order (miss and hit positions alike),
        // then the shared-expert contribution -- preserving the exact fp accumulation order.
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            float weight = top_values[k];
            const float *dd = down_arr + (size_t)k * dim;
            for (uint32_t i = 0; i < dim; ++i) moe_buf[i] += weight * dd[i];
        }

        double tr0 = omp_get_wtime();
        for (uint32_t i = 0; i < dim; ++i) {
            moe_buf[i] += sh_gate_val * sh_down_exp[i];
        }

        if (g_dump_active) {
            float *routed = (float*)malloc(dim * sizeof(float));
            float *sh_gated = (float*)malloc(dim * sizeof(float));
            for (uint32_t i = 0; i < dim; ++i) {
                routed[i] = moe_buf[i] - sh_gate_val * sh_down_exp[i];
                sh_gated[i] = sh_gate_val * sh_down_exp[i];
            }
            dump_f32("ffn_moe_out", (int)l, routed, dim);
            dump_f32("ffn_shexp_gated", (int)l, sh_gated, dim);
            free(routed);
            free(sh_gated);
        }

        eng->t_expert += omp_get_wtime() - te0;

        for (uint32_t i = 0; i < dim; ++i) x[i] += moe_buf[i];
        eng->t_resid += omp_get_wtime() - tr0;

        dump_f32("post_moe", (int)l, x, dim);
    }

    // P1 determinism fix (PLAN.md §3.1): fills are still *issued* per layer (above, budget-limited
    // by g_fill_budget), but they are *committed* exactly once per token, here at the token
    // boundary, after the last layer's forward work and before the next token begins. Within a
    // token the slot table seen by gpu_expert_lookup is frozen, so whether a routed expert is
    // VRAM-resident (GPU mmvq vs CPU AVX2) is a deterministic function of the token history, not
    // of when the background fill thread happened to finish a copy. No-op when bgfill is inactive.
    gpu_expert_cache_bgfill_poll();

    rms_norm(norm_x, x, (const float*)m->output_norm.data, dim, eps);
    dump_f32_noil("h_nextn", norm_x, dim);

    float *logits = (float*)malloc(vocab_size * sizeof(float));
    mat_vec(logits, &m->output_weight, norm_x);
    // P0 tolerance gate: hand the final logits to the --dump-logits sink. No-op (one predicate)
    // unless main.c armed it, so normal decode pays nothing. See include/tuff_exact.h.
    tuff_logit_capture(logits, vocab_size);
    dump_f32_noil("result_output", logits, vocab_size);

    // Debug: dump top-10 logits for the first forward pass (TUFF_DEBUG_LOGITS=1)
    if (pos == 0 && dbg_logits_flag()) {
        int top[10];
        float tv[10];
        for (int i = 0; i < 10; ++i) { top[i] = i; tv[i] = logits[i]; }
        for (uint32_t i = 10; i < vocab_size; ++i) {
            int minp = 0; float minv = tv[0];
            for (int k = 1; k < 10; ++k) if (tv[k] < minv) { minv = tv[k]; minp = k; }
            if (logits[i] > minv) { tv[minp] = logits[i]; top[minp] = (int)i; }
        }
        fprintf(stderr, "[DBG] first-forward top-10 logits (token, logit):");
        for (int i = 0; i < 10; ++i) fprintf(stderr, " (%d, %.3f)", top[i], tv[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBG] first-forward logit[11]=%.3f logit[8536]=%.3f logit[9419]=%.3f\n",
                logits[11], logits[8536], logits[9419]);
    }

    // Validate logits for NaN/Inf (garbage output check)
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (isnan(logits[i]) || isinf(logits[i])) {
            int nn=0; for(uint32_t z=0;z<dim;++z){if(isnan(x[z])||isinf(x[z]))nn++;}
            fprintf(stderr, "[FATAL ERROR] Garbage output detected: logit[%u] is NaN/Inf! (pos=%d, x has %d NaN)\n", i, pos, nn);
            free(x); free(norm_x); free(moe_buf); free(logits);
            exit(EXIT_FAILURE);
        }
    }

    int32_t best_token = 0;
    if (eng->sampler.params.temp <= 0.0f) {
        float max_logit = -1e30f;
        for (uint32_t i = 0; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_token = (int32_t)i;
            }
        }
        sampler_accept_token(&eng->sampler, best_token);
    } else {
        best_token = sampler_sample(&eng->sampler, logits, (int32_t)vocab_size);
    }

    free(logits);
    eng->t_total += omp_get_wtime() - t_token;
    return best_token;
}

void engine_free(engine_t *eng) {
    if (!eng) return;
    if (getenv("TUFF_PROFILE")) {
        double sum = eng->t_attn + eng->t_delta + eng->t_expert;
        fprintf(stderr, "[PROFILE] per-token s: total=%.4f attn=%.4f (%.0f%%) delta=%.4f (%.0f%%) "
                        "expert=%.4f (%.0f%%) other=%.4f (%.0f%%)\n",
                eng->t_total, eng->t_attn, eng->t_attn / sum * 100.0,
                eng->t_delta, eng->t_delta / sum * 100.0,
                eng->t_expert, eng->t_expert / sum * 100.0,
                eng->t_total - sum, (eng->t_total - sum) / eng->t_total * 100.0);
        double tkw = eng->t_gu + eng->t_fill + eng->t_down + eng->t_miss + eng->t_resid +
                     eng->t_expert_sync + eng->t_pref + eng->t_cpumoe + eng->t_cpumoe_pub;
        fprintf(stderr, "[PROFILE] expert phases: gu=%.4f fill=%.4f down=%.4f topk/lookup=%.4f "
                        "resid/shared=%.4f cachesync=%.4f pref=%.4f cpumoe=%.4f (pub=%.4f) "
                        "(sum=%.4f of expert=%.4f)\n",
                eng->t_gu, eng->t_fill, eng->t_down, eng->t_miss, eng->t_resid, eng->t_expert_sync,
                eng->t_pref, eng->t_cpumoe, eng->t_cpumoe_pub, tkw, eng->t_expert);
        unsigned long long bgi = 0, bgc = 0;
        gpu_expert_cache_bgfill_stats(&bgi, &bgc);
        fprintf(stderr, "[PROFILE] bgfill: %llu issued, %llu committed\n", bgi, bgc);
        fprintf(stderr, "[PROFILE] delta-layer graph launches: %llu\n", gpu_graph_launch_count());
        fprintf(stderr, "[PROFILE] delta phases: mat=%.4f conv/l2=%.4f statekern=%.4f "
                        "rms/gate=%.4f out=%.4f (sum=%.4f of delta=%.4f)\n",
                eng->t_d_mat, eng->t_d_conv, eng->t_d_state, eng->t_d_norm, eng->t_d_out,
                eng->t_d_mat + eng->t_d_conv + eng->t_d_state + eng->t_d_norm + eng->t_d_out, eng->t_delta);
    }
    moe_pool_shutdown();
    if (eng->delta_state) free(eng->delta_state);
    if (eng->conv_state) free(eng->conv_state);
    if (eng->k_cache) free(eng->k_cache);
    if (eng->v_cache) free(eng->v_cache);

    if (eng->x) gpu_host_free(eng->x);
    if (eng->norm_x) gpu_host_free(eng->norm_x);
    if (eng->moe_buf) gpu_host_free(eng->moe_buf);
    if (eng->router_logits) gpu_host_free(eng->router_logits);
    if (eng->top_indices) free(eng->top_indices);
    if (eng->top_values) free(eng->top_values);
    if (eng->gate_exp) free(eng->gate_exp);
    if (eng->up_exp) free(eng->up_exp);
    if (eng->act_exp) free(eng->act_exp);
    if (eng->down_exp) free(eng->down_exp);
    if (eng->sh_gate_exp) free(eng->sh_gate_exp);
    if (eng->exp_slots) free(eng->exp_slots);
    if (eng->gate_arr) free(eng->gate_arr);
    if (eng->up_arr) free(eng->up_arr);
    if (eng->act_arr) free(eng->act_arr);
    if (eng->down_arr) free(eng->down_arr);
    if (eng->down_hit) free(eng->down_hit);
    if (eng->sh_up_exp) free(eng->sh_up_exp);
    if (eng->q8_normx) free(eng->q8_normx);
    if (eng->sh_act_exp) free(eng->sh_act_exp);
    if (eng->sh_down_exp) free(eng->sh_down_exp);
    if (eng->hist_tokens) free(eng->hist_tokens);
    if (eng->host_logits) free(eng->host_logits);
    sampler_free(&eng->sampler);

    free(eng);
}

int32_t engine_get_last_draft_token(const engine_t *eng) {
    return eng ? eng->last_draft_token : -1;
}

void engine_reset_context(engine_t *eng) {
    if (!eng || !eng->model) return;
    const model_t *model = eng->model;
    size_t delta_floats = (size_t)model->n_delta_layers *
                           model->ssm_n_v_heads * model->ssm_head_v_dim * model->ssm_head_k_dim;
    if (eng->delta_state && delta_floats > 0) memset(eng->delta_state, 0, delta_floats * sizeof(float));

    size_t conv_floats = (size_t)model->n_delta_layers * model->ssm_conv_kernel * model->ssm_conv_dim;
    if (eng->conv_state && conv_floats > 0) memset(eng->conv_state, 0, conv_floats * sizeof(float));

    size_t kv_floats = (size_t)model->n_attn_layers * model->head_count_kv *
                        (size_t)eng->max_seq_len * model->key_length;
    if (eng->k_cache && kv_floats > 0) memset(eng->k_cache, 0, kv_floats * sizeof(float));
    if (eng->v_cache && kv_floats > 0) memset(eng->v_cache, 0, kv_floats * sizeof(float));

    // H17: the n-gram draft's accepted sequence resets with the rest of the context.
    if (eng->hist_tokens) eng->hist_len = 0;

    gpu_reset_delta_state();
    gpu_reset_kv();
}

void engine_state_save_to(const engine_t *eng, int pos, const int32_t *tokens, engine_state_t *st) {
    if (!eng || !eng->model || !st) return;
    const model_t *model = eng->model;
    st->pos = pos;
    if (pos > 0 && tokens) {
        st->tokens = (int32_t*)realloc(st->tokens, (size_t)pos * sizeof(int32_t));
        memcpy(st->tokens, tokens, (size_t)pos * sizeof(int32_t));
    }

    size_t delta_bytes = (size_t)model->n_delta_layers *
                         model->ssm_n_v_heads * model->ssm_head_v_dim * model->ssm_head_k_dim * sizeof(float);
    size_t conv_bytes = (size_t)model->n_delta_layers * model->ssm_conv_kernel * model->ssm_conv_dim * sizeof(float);

    if (delta_bytes > 0 && eng->delta_state) {
        st->delta_state = (float*)realloc(st->delta_state, delta_bytes);
        memcpy(st->delta_state, eng->delta_state, delta_bytes);
        st->delta_state_bytes = delta_bytes;
    }
    if (conv_bytes > 0 && eng->conv_state) {
        st->conv_state = (float*)realloc(st->conv_state, conv_bytes);
        memcpy(st->conv_state, eng->conv_state, conv_bytes);
        st->conv_state_bytes = conv_bytes;
    }

    size_t gpu_ds_sz = 0, gpu_dc_sz = 0;
    gpu_get_recurrent_state_sizes(&gpu_ds_sz, &gpu_dc_sz);
    if (gpu_ds_sz > 0) {
        st->gpu_dstate = (float*)realloc(st->gpu_dstate, gpu_ds_sz);
        st->gpu_dstate_bytes = gpu_ds_sz;
    }
    if (gpu_dc_sz > 0) {
        st->gpu_dconv = (float*)realloc(st->gpu_dconv, gpu_dc_sz);
        st->gpu_dconv_bytes = gpu_dc_sz;
    }
    if (gpu_ds_sz > 0 || gpu_dc_sz > 0) {
        gpu_save_recurrent_state(st->gpu_dstate, st->gpu_dconv);
    }
}

void engine_state_restore_from(engine_t *eng, const engine_state_t *st) {
    if (!eng || !st) return;
    if (st->delta_state && eng->delta_state && st->delta_state_bytes > 0) {
        memcpy(eng->delta_state, st->delta_state, st->delta_state_bytes);
    }
    if (st->conv_state && eng->conv_state && st->conv_state_bytes > 0) {
        memcpy(eng->conv_state, st->conv_state, st->conv_state_bytes);
    }
    if (st->gpu_dstate || st->gpu_dconv) {
        gpu_restore_recurrent_state(st->gpu_dstate, st->gpu_dconv);
    }
}

void engine_state_free_members(engine_state_t *st) {
    if (!st) return;
    free(st->tokens); st->tokens = NULL;
    free(st->delta_state); st->delta_state = NULL; st->delta_state_bytes = 0;
    free(st->conv_state); st->conv_state = NULL; st->conv_state_bytes = 0;
    free(st->gpu_dstate); st->gpu_dstate = NULL; st->gpu_dstate_bytes = 0;
    free(st->gpu_dconv); st->gpu_dconv = NULL; st->gpu_dconv_bytes = 0;
    st->pos = 0;
}

engine_prefix_cache_t* engine_prefix_cache_create(int max_seq_len) {
    engine_prefix_cache_t *cache = (engine_prefix_cache_t*)calloc(1, sizeof(engine_prefix_cache_t));
    cache->cap_tokens = max_seq_len > 0 ? max_seq_len : 4096;
    cache->tokens = (int32_t*)malloc((size_t)cache->cap_tokens * sizeof(int32_t));
    cache->n_tokens = 0;
    cache->n_checkpoints = 0;
    return cache;
}

void engine_prefix_cache_free(engine_prefix_cache_t *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->n_checkpoints; ++i) {
        engine_state_free_members(&cache->checkpoints[i]);
    }
    free(cache->tokens);
    free(cache);
}

void engine_prefix_cache_reset(engine_prefix_cache_t *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->n_checkpoints; ++i) {
        engine_state_free_members(&cache->checkpoints[i]);
    }
    cache->n_checkpoints = 0;
    cache->n_tokens = 0;
}

void engine_prefix_cache_save_checkpoint(engine_prefix_cache_t *cache, const engine_t *eng) {
    if (!cache || !eng || cache->n_tokens <= 0) return;

    for (int i = 0; i < cache->n_checkpoints; ++i) {
        if (cache->checkpoints[i].pos == cache->n_tokens) {
            engine_state_save_to(eng, cache->n_tokens, cache->tokens, &cache->checkpoints[i]);
            return;
        }
    }

    int slot = cache->n_checkpoints;
    if (slot >= ENGINE_MAX_CHECKPOINTS) {
        engine_state_free_members(&cache->checkpoints[1]);
        for (int i = 1; i < ENGINE_MAX_CHECKPOINTS - 1; ++i) {
            cache->checkpoints[i] = cache->checkpoints[i + 1];
        }
        slot = ENGINE_MAX_CHECKPOINTS - 1;
        memset(&cache->checkpoints[slot], 0, sizeof(engine_state_t));
    } else {
        cache->n_checkpoints++;
    }
    engine_state_save_to(eng, cache->n_tokens, cache->tokens, &cache->checkpoints[slot]);
}

prefill_result_t engine_prefill_with_cache(engine_t *eng, engine_prefix_cache_t *cache,
                                          const int32_t *prompt_tokens, int n_prompt) {
    prefill_result_t res = {0};
    res.total_prompt_tokens = n_prompt;
    if (!eng || !cache || !prompt_tokens || n_prompt <= 0) return res;

    int match_len = 0;
    while (match_len < cache->n_tokens && match_len < n_prompt &&
           cache->tokens[match_len] == prompt_tokens[match_len]) {
        match_len++;
    }

    if (match_len < cache->n_tokens) {
        int best_idx = -1;
        int best_pos = 0;
        for (int i = 0; i < cache->n_checkpoints; ++i) {
            int c_pos = cache->checkpoints[i].pos;
            if (c_pos > 0 && c_pos <= match_len) {
                int ok = 1;
                for (int k = 0; k < c_pos; ++k) {
                    if (cache->checkpoints[i].tokens[k] != prompt_tokens[k]) {
                        ok = 0;
                        break;
                    }
                }
                if (ok && c_pos > best_pos) {
                    best_pos = c_pos;
                    best_idx = i;
                }
            }
        }

        if (best_idx >= 0) {
            engine_state_restore_from(eng, &cache->checkpoints[best_idx]);
            cache->n_tokens = cache->checkpoints[best_idx].pos;
            match_len = cache->n_tokens;
        } else {
            engine_reset_context(eng);
            cache->n_tokens = 0;
            match_len = 0;
        }
    }

    res.cached_tokens = match_len;
    res.new_tokens_prefilled = n_prompt - match_len;

    double t0 = omp_get_wtime();
    int pos = match_len;
    int32_t curr_token = (match_len > 0) ? cache->tokens[match_len - 1] : prompt_tokens[0];

    if (n_prompt > cache->cap_tokens) {
        cache->cap_tokens = n_prompt + 1024;
        cache->tokens = (int32_t*)realloc(cache->tokens, (size_t)cache->cap_tokens * sizeof(int32_t));
    }

    for (int i = match_len; i < n_prompt; ++i) {
        curr_token = engine_forward_token(eng, prompt_tokens[i], pos++);
        cache->tokens[i] = prompt_tokens[i];
    }
    cache->n_tokens = pos;
    double t1 = omp_get_wtime();

    res.prefill_time_s = t1 - t0;
    res.prefill_tok_per_sec = (res.new_tokens_prefilled > 0 && res.prefill_time_s > 0) ?
                              ((double)res.new_tokens_prefilled / res.prefill_time_s) : 0.0;
    res.first_generated_token = curr_token;

    return res;
}

// ===========================================================================
// H7 stage 2b/2c: real MTP speculative decoding.
//
// Until now --spec-mtp only *measured* the draft head: it computed a draft,
// compared it with the token the model had already chosen, and threw it away.
// That is a 6% tax for no gain, because accepting a draft requires proving it
// is what the model would have produced — and proving that requires evaluating
// the trunk at two positions in one pass.
//
// The verify pass below does that. Both inputs are known before it starts
// (the accepted token and the draft), so it is a genuine batch, not a
// sequential pair: the delta-layer input projections and the Q6_K output head
// read their weights once for both columns (iteration 42), while the parts
// that cannot be shared — the GDN recurrence, the per-token expert sets — run
// twice. The host tail and the ~40 per-layer syncs are paid once for two
// tokens instead of once each, which is where a large part of the win comes
// from.
// ===========================================================================

// The router + shared-expert + routed-expert stage for ONE column of the verify
// pass. This is the corresponding block of engine_forward_token verbatim,
// parameterized by which column's activation it reads and which residual it
// accumulates into. The expert sets genuinely differ between the two tokens, so
// this is the part of a verify step that costs 2x by construction.
// The router + shared-expert + routed-expert stage for BOTH columns of a layer.
//
// The structure is the whole point of H7: the two columns' router projections
// and shared experts are issued back to back and drained by ONE
// gpu_router_logits_wait(), and both columns' CPU-side expert misses go to the
// worker pool as ONE batch with one join. A verify step therefore pays 40
// syncs and 40 pool joins for two tokens instead of 80 and 80. A first cut
// that simply ran the single-column stage twice measured 90% of the step in
// this stage and came out slower than plain decode; the sync count was why.
//
// What genuinely costs 2x is the expert arithmetic itself: the two tokens
// route to different experts, so both sets have to be computed.
static void verify_moe_stage_n2(engine_t *eng, uint32_t l, model_layer_t *layer,
                                const float *d_norm2, float *d_res2, int cache_on) {
    model_t *m = eng->model;
    uint32_t dim            = m->embedding_length;
    uint32_t n_expert       = m->expert_count;
    uint32_t n_expert_used  = m->expert_used_count;
    uint32_t ffn_dim        = m->expert_ffn_length;
    uint32_t sh_ffn_dim     = m->expert_shared_ffn_len;

    const ggml_type_t down_type = (ggml_type_t)layer->ffn_down_exps.info.type;
    const float *shexp[2] = { gpu_get_shexp_dev_ptr(), gpu_get_shexp2_dev_ptr() };

    if (l < (uint32_t)g_n_gpu_expert_layers) {
        // Fully device-resident: no host round trip to batch in the first place.
        for (int c = 0; c < 2; ++c) {
            const float *d_nc = d_norm2 + (size_t)c * dim;
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, d_nc, (int)dim, (int)n_expert);
            gpu_shared_expert_dev((float*)shexp[c], layer, d_nc, (int)dim, (int)sh_ffn_dim);
            gpu_expert_resident_layer_dev((int)l, down_type, d_nc, d_res2 + (size_t)c * dim,
                                          shexp[c], (int)dim, (int)ffn_dim, (int)n_expert);
        }
        return;
    }

    float      *rl[2] = { eng->router_logits, eng->router_logits2 };
    float      *nx[2] = { eng->norm_x,        eng->norm_x2 };
    block_q8_K *q8[2] = { eng->q8_normx,      eng->q8_normx2 };
    float      *da[2] = { eng->down_arr,      eng->down_arr2 };

    // Phase 1: issue both columns, drain once. When the delta graph already carried
    // the router compute and the shared experts (H12's arrangement, two columns), the
    // host only has to fetch the results.
    if (!(layer->type != LAYER_TYPE_ATTENTION && gpu_has_delta_graphs_n2())) {
        for (int c = 0; c < 2; ++c) {
            const float *d_nc = d_norm2 + (size_t)c * dim;
            gpu_router_logits_dev_only_col(&layer->ffn_gate_inp, d_nc, (int)dim, (int)n_expert, c);
        }
        // H15: one weight read for both columns instead of one per column.
        gpu_shared_expert_dev_n2((float*)shexp[0], (float*)shexp[1], layer, d_norm2, (int)dim, (int)sh_ffn_dim);
    }
    gpu_router_logits_copy_out_n2(rl[0], nx[0], rl[1], nx[1], d_norm2, (int)dim, (int)n_expert);
    double tw0 = omp_get_wtime();
    gpu_router_logits_wait();
    eng->t_expert_sync += omp_get_wtime() - tw0;

    // Phase 2: top-k and cache lookup per column.
    int   top_idx[2][8];
    float top_val[2][8];
    int   hit_slots[2][8], hit_pos[2][8], n_batch[2];
    int   miss_k[2][8], n_miss[2];

    for (int c = 0; c < 2; ++c) {
        int   *ti = top_idx[c];
        float *tv = top_val[c];
        for (uint32_t k = 0; k < n_expert_used; ++k) { ti[k] = (int)k; tv[k] = rl[c][k]; }
        for (uint32_t i = n_expert_used; i < n_expert; ++i) {
            float val = rl[c][i];
            uint32_t min_pos = 0;
            float min_val = tv[0];
            for (uint32_t k = 1; k < n_expert_used; ++k) {
                if (tv[k] < min_val) { min_val = tv[k]; min_pos = k; }
            }
            if (val > min_val) { tv[min_pos] = val; ti[min_pos] = (int)i; }
        }
        softmax(tv, (int)n_expert_used);

        n_batch[c] = 0; n_miss[c] = 0;
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            int slot = cache_on ? gpu_expert_lookup(l, ti[k]) : -1;
            if (slot >= 0) {
                hit_slots[c][n_batch[c]] = slot;
                hit_pos[c][n_batch[c]] = (int)k;
                n_batch[c]++;
            } else {
                miss_k[c][n_miss[c]++] = (int)k;
            }
        }
    }

    // Instrumentation (TUFF_SPEC_OVERLAP=1): how much do the two tokens' expert
    // sets coincide? Shared experts could have their weights read once for both
    // activations by an ncols-style expert kernel, the same trick the dense trunk uses.
    if (getenv("TUFF_SPEC_OVERLAP")) {
        extern unsigned long long g_ovl_shared, g_ovl_total;
        for (uint32_t a = 0; a < n_expert_used; ++a)
            for (uint32_t b = 0; b < n_expert_used; ++b)
                if (top_idx[0][a] == top_idx[1][b]) { g_ovl_shared++; break; }
        g_ovl_total += n_expert_used;
    }

    // Phase 3: both columns' misses as one batch, one submit, one join.
    const int total_miss = n_miss[0] + n_miss[1];
    if (total_miss > 0) {
        double tpub0 = omp_get_wtime();
        g_moe_batch.L      = layer;
        g_moe_batch.x      = nx[0];
        g_moe_batch.q8x    = q8[0];
        g_moe_batch.dim    = (int)dim;
        g_moe_batch.ffn    = (int)ffn_dim;
        g_moe_batch.gtype  = (ggml_type_t)layer->ffn_gate_exps.info.type;
        g_moe_batch.dtype  = down_type;
        g_moe_batch.gate_bytes = ggml_row_size(g_moe_batch.gtype, dim) * ffn_dim;
        g_moe_batch.up_bytes   = g_moe_batch.gate_bytes;
        g_moe_batch.down_bytes = ggml_row_size(down_type, ffn_dim) * dim;
        int j = 0;
        // H15: a miss selected by BOTH columns (a shared miss) becomes ONE entry
        // whose weights are read once and evaluated against both columns'
        // activations (xk2/q8xk2/down_out2 carry column 1). Exclusive misses stay
        // single-column entries with xk2 == NULL, which is the old behaviour.
        for (int i = 0; i < n_miss[0]; ++i) {
            const int e0 = top_idx[0][miss_k[0][i]];
            int in1 = -1;
            for (int i1 = 0; i1 < n_miss[1]; ++i1)
                if (top_idx[1][miss_k[1][i1]] == e0) { in1 = i1; break; }
            g_moe_batch.experts[j]  = e0;
            g_moe_batch.xk[j]       = nx[0];
            g_moe_batch.q8xk[j]     = q8[0];
            g_moe_batch.down_out[j] = da[0] + (size_t)miss_k[0][i] * dim;
            if (in1 >= 0) {
                g_moe_batch.xk2[j]       = nx[1];
                g_moe_batch.q8xk2[j]     = q8[1];
                g_moe_batch.down_out2[j] = da[1] + (size_t)miss_k[1][in1] * dim;
            } else {
                g_moe_batch.xk2[j] = NULL; g_moe_batch.q8xk2[j] = NULL; g_moe_batch.down_out2[j] = NULL;
            }
            j++;
        }
        for (int i1 = 0; i1 < n_miss[1]; ++i1) {
            int consumed = 0;
            for (int i0 = 0; i0 < n_miss[0]; ++i0)
                if (top_idx[0][miss_k[0][i0]] == top_idx[1][miss_k[1][i1]]) { consumed = 1; break; }
            if (consumed) continue;
            g_moe_batch.experts[j]  = top_idx[1][miss_k[1][i1]];
            g_moe_batch.xk[j]       = nx[1];
            g_moe_batch.q8xk[j]     = q8[1];
            g_moe_batch.down_out[j] = da[1] + (size_t)miss_k[1][i1] * dim;
            g_moe_batch.xk2[j] = NULL; g_moe_batch.q8xk2[j] = NULL; g_moe_batch.down_out2[j] = NULL;
            j++;
        }
        g_moe_batch.n = j;
        moe_pool_submit();
        eng->t_cpumoe_pub += omp_get_wtime() - tpub0;
    }

    // Phase 4: BOTH columns' GPU expert work is issued before the pool join, so the
    // CPU misses of both tokens hide behind it exactly as they do in the single-token
    // stage. This is what the doubled expert scratch buys: with one shared down buffer
    // column 1 had to wait until after the join, and the measurement showed the CPU
    // work had simply stopped hiding (phase sum 22.6 ms under a 28.6 ms stage).
    //
    // H15: partition each column's hit list into SHARED (selected by both tokens)
    // and exclusive sets. A shared expert is evaluated against both activations with
    // its weights read ONCE by an ncols=2 kernel; the exclusive experts keep the
    // single-column path, with an explicit destination-index table so the compressed
    // slot lists still land every output at its original hit position. The per-column
    // combine below is untouched, so it sees exactly the values and order it always did.
    {
        int shared_slots[8], shared_dst0[8], shared_dst1[8], n_shared = 0;
        int only0_slots[8], only0_dst[8], n_only0 = 0;
        int only1_slots[8], only1_dst[8], n_only1 = 0;
        int hit_exp[2][8];
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < n_batch[c]; ++i)
                hit_exp[c][i] = (int)top_idx[c][hit_pos[c][i]];

        for (int i = 0; i < n_batch[0]; ++i) {
            int in1 = -1;
            for (int j = 0; j < n_batch[1]; ++j)
                if (hit_exp[1][j] == hit_exp[0][i]) { in1 = j; break; }
            if (in1 >= 0) {
                shared_slots[n_shared] = hit_slots[0][i];
                shared_dst0[n_shared]  = i;
                shared_dst1[n_shared]  = in1;
                n_shared++;
            } else {
                only0_slots[n_only0] = hit_slots[0][i];
                only0_dst[n_only0]   = i;
                n_only0++;
            }
        }
        for (int j = 0; j < n_batch[1]; ++j) {
            int in0 = -1;
            for (int i = 0; i < n_batch[0]; ++i)
                if (hit_exp[0][i] == hit_exp[1][j]) { in0 = i; break; }
            if (in0 < 0) {
                only1_slots[n_only1] = hit_slots[1][j];
                only1_dst[n_only1]   = j;
                n_only1++;
            }
        }

        if (n_shared > 0) {
            gpu_expert_moe_gateup_swiglu_dev_n2_shared(shared_slots, shared_dst0, shared_dst1, n_shared,
                                                       d_norm2, d_norm2 + (size_t)dim,
                                                       (int)dim, (int)ffn_dim);
        }
        if (n_only0 > 0) {
            gpu_expert_moe_gateup_swiglu_dev_col_dst(only0_slots, only0_dst, n_only0,
                                                     d_norm2, (int)dim, (int)ffn_dim, 0);
        }
        if (n_only1 > 0) {
            gpu_expert_moe_gateup_swiglu_dev_col_dst(only1_slots, only1_dst, n_only1,
                                                     d_norm2 + (size_t)dim, (int)dim, (int)ffn_dim, 1);
        }
        // The down kernels read the FULL hit set per column, so the activation
        // quantization must cover all n_batch[c] experts, not just the exclusive set.
        if (n_batch[0] > 0) gpu_expert_moe_act_quant_col(n_batch[0], (int)ffn_dim, 0);
        if (n_batch[1] > 0) gpu_expert_moe_act_quant_col(n_batch[1], (int)ffn_dim, 1);
        if (n_shared > 0) {
            gpu_expert_moe_down_dev_n2_shared(shared_slots, shared_dst0, shared_dst1, n_shared,
                                              down_type, (int)ffn_dim, (int)dim);
        }
        if (n_only0 > 0) {
            gpu_expert_moe_down_dev_col_dst(only0_slots, only0_dst, n_only0,
                                            down_type, (int)ffn_dim, (int)dim, 0);
        }
        if (n_only1 > 0) {
            gpu_expert_moe_down_dev_col_dst(only1_slots, only1_dst, n_only1,
                                            down_type, (int)ffn_dim, (int)dim, 1);
        }
    }

    if (total_miss > 0) {
        double tw1 = omp_get_wtime();
        moe_pool_wait();
        eng->t_cpumoe += omp_get_wtime() - tw1;
    }

    for (int c = 0; c < 2; ++c) {
        // Each column needs its own accumulator: the combine uploads it
        // asynchronously, so reusing one buffer would let column 1 overwrite bytes
        // column 0's copy has not read yet.
        float *mb = (c == 0) ? eng->moe_buf : eng->moe_buf2;
        float *host_miss_ptr = NULL;
        if (n_miss[c] > 0) {
            memset(mb, 0, dim * sizeof(float));
            for (int i = 0; i < n_miss[c]; ++i) {
                int k = miss_k[c][i];
                float w = top_val[c][k];
                const float *dd = da[c] + (size_t)k * dim;
                for (uint32_t q = 0; q < dim; ++q) mb[q] += w * dd[q];
            }
            host_miss_ptr = mb;
        }
        gpu_moe_combine_accum_col(d_res2 + (size_t)c * dim, hit_pos[c], n_batch[c],
                                  top_val[c], shexp[c], host_miss_ptr, (int)dim, c);
    }

    // One token's fill budget for the whole step, NOT one per column. Doubling it to
    // match the two tokens was measured and is worse (42.2 vs 47.9 tok/s): the extra
    // 1.95 MiB H2D fills push the latency-critical router D2H behind them, which is
    // H14's finding about the copy engine, and cachesync rose 1.21 -> 1.79 s.
    if (cache_on && g_fill_budget > 0 && (l % (uint32_t)g_fill_stride) == 0) {
        for (int c = 0, filled = 0; c < 2; ++c)
            for (int i = 0; i < n_miss[c] && filled < g_fill_budget; ++i)
                filled += gpu_expert_cache_bgfill(m, l, top_idx[c][miss_k[c][i]]);
    }
}

// Evaluate the trunk at two consecutive positions in one pass.
// tok0 is the already-accepted token at `pos`; tok1 is the draft at `pos+1`.
// out_next[0] is the model's own continuation of tok0 — the token that decides
// whether the draft was right. out_next[1] is the continuation of tok1 and is
// only meaningful when out_next[0] == tok1.
//
// On return the recurrent state, conv state and KV cache have advanced through
// BOTH positions. If the draft is rejected the caller must roll the delta/conv
// state back to the snapshot taken before the call; the KV cache needs no
// rollback because it is written before it is read at every position, so the
// stale entries at pos+1 are overwritten by the next step.
void engine_forward_verify(engine_t *eng, int32_t tok0, int32_t tok1, int pos, int32_t out_next[2]) {
    model_t *m = eng->model;
    uint32_t dim = m->embedding_length;
    uint32_t vocab_size = m->vocab_size;
    const float eps = m->rms_norm_eps;
    double t_token = omp_get_wtime();

    if (tok0 < 0 || tok0 >= (int32_t)vocab_size || tok1 < 0 || tok1 >= (int32_t)vocab_size) {
        fprintf(stderr, "[FATAL ERROR] Invalid token in verify pass (%d, %d, vocab %u)!\n",
                tok0, tok1, vocab_size);
        exit(EXIT_FAILURE);
    }
    if (pos < 0 || pos + 1 >= eng->max_seq_len) {
        fprintf(stderr, "[FATAL ERROR] Verify position %d out of range (max_seq_len=%d)!\n",
                pos, eng->max_seq_len);
        exit(EXIT_FAILURE);
    }

    gpu_n2_prepare((int)dim, (int)vocab_size);

    // Both embeddings, then one H2D per column.
    const uint8_t *embd_base = (const uint8_t*)m->token_embd.data;
    size_t embd_row_bytes = ggml_row_size((ggml_type_t)m->token_embd.info.type, dim);
    float *x0 = eng->x;
    float *x1 = eng->norm_x;   // borrowed until the first rms_norm consumes it
    dequantize_row((ggml_type_t)m->token_embd.info.type, embd_base + (size_t)tok0 * embd_row_bytes, x0, dim);
    dequantize_row((ggml_type_t)m->token_embd.info.type, embd_base + (size_t)tok1 * embd_row_bytes, x1, dim);
    gpu_residual_reset_n2(x0, x1, (int)dim);

    float *d_res2  = gpu_get_d_res_x2();
    float *d_norm2 = gpu_get_d_norm_x2();

    uint32_t attn_layer_idx = 0;
    uint32_t delta_layer_idx = 0;
    int cache_on = gpu_expert_cache_active();

    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];

        if (layer->type == LAYER_TYPE_ATTENTION) {
            double ta0 = omp_get_wtime();
            uint32_t head_dim  = m->key_length;
            uint32_t n_head    = m->head_count;
            uint32_t n_head_kv = m->head_count_kv;
            float scale = 1.0f / sqrtf((float)head_dim);

            gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->attn_norm.dev, (int)dim, eps);
            // Column 1 attends over column 0's key/value, so the two calls are
            // ordered on the default stream rather than batched.
            for (int c = 0; c < 2; ++c) {
                gpu_attention_layer_dev((int)attn_layer_idx, d_res2 + (size_t)c * dim, layer,
                                        d_norm2 + (size_t)c * dim,
                                        (int)m->n_attn_layers, (int)n_head, (int)n_head_kv, (int)head_dim,
                                        pos + c, pos + c + 1, eng->max_seq_len, scale, eps, (int)dim,
                                        (int)m->rope_dim_count, m->rope_dim_sections, m->rope_freq_base);
            }
            gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->post_attention_norm.dev, (int)dim, eps);
            eng->t_attn += omp_get_wtime() - ta0;
            attn_layer_idx++;
        } else {
            double td0 = omp_get_wtime();
            uint32_t key_dim     = m->ssm_key_dim;
            uint32_t value_dim   = m->ssm_value_dim;
            uint32_t qkv_dim     = m->ssm_qkv_dim;
            uint32_t n_k_heads   = m->ssm_n_k_heads;
            uint32_t n_v_heads   = m->ssm_n_v_heads;
            uint32_t head_k_dim  = m->ssm_head_k_dim;
            uint32_t head_v_dim  = m->ssm_head_v_dim;
            uint32_t conv_kernel = m->ssm_conv_kernel;
            float scale_q = 1.0f / sqrtf((float)head_k_dim);

            if (gpu_has_delta_graphs_n2()) {
                // ~50 launches for this layer collapse into one; the router D2H for
                // both columns is the only thing the host still issues, exactly as
                // H12 does for the single-token path.
                gpu_launch_delta_layer_graph_n2(delta_layer_idx);
                eng->t_delta += omp_get_wtime() - td0;
                delta_layer_idx++;
                goto verify_moe;
            }

            gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->attn_norm.dev, (int)dim, eps);

            // The one place the batch actually pays: attn_qkv (2048x8192) and
            // attn_gate are read once for both columns. 30 such layers per token.
            gpu_delta_projections_n2(&layer->attn_qkv, &layer->attn_gate,
                                     &layer->ssm_beta, &layer->ssm_alpha,
                                     d_norm2, (int)dim);

            // The recurrence is sequential by definition: S_{t+1} feeds S_{t+2}.
            for (int c = 0; c < 2; ++c) {
                // H17: no column publish — the fused GDN reads the batched projections in
                // place (see gpu_delta_fused_gdn_dev_col).
                gpu_delta_fused_gdn_dev_col(c, (int)delta_layer_idx, layer, d_res2 + (size_t)c * dim,
                                            eps, scale_q,
                                            (int)n_k_heads, (int)n_v_heads,
                                            (int)head_k_dim, (int)head_v_dim,
                                            (int)key_dim, (int)value_dim, (int)qkv_dim,
                                            (int)conv_kernel, (int)dim);
                // Between the columns the state is exactly "accepted token
                // consumed, draft not yet" — the point a rejection rewinds to.
                if (c == 0) {
                    gpu_delta_snapshot_layer_n2((int)delta_layer_idx, (int)n_v_heads,
                                                (int)head_v_dim, (int)head_k_dim,
                                                (int)conv_kernel, (int)qkv_dim);
                }
            }
            gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->post_attention_norm.dev, (int)dim, eps);
            eng->t_delta += omp_get_wtime() - td0;
            delta_layer_idx++;
        }

verify_moe:;
        double te0 = omp_get_wtime();
        verify_moe_stage_n2(eng, l, layer, d_norm2, d_res2, cache_on);
        eng->t_expert += omp_get_wtime() - te0;
    }

    gpu_expert_cache_bgfill_poll();

    static int ovl_step = 0;
    if (g_ovl_total && getenv("TUFF_SPEC_OVERLAP") && (++ovl_step % 40) == 0) {
        fprintf(stderr, "[SPEC-OVERLAP] %.1f%% of the two tokens' experts coincide (%llu/%llu)\n",
                100.0 * (double)g_ovl_shared / (double)g_ovl_total, g_ovl_shared, g_ovl_total);
    }

    gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)m->output_norm.dev, (int)dim, eps);

    // 398 MB read once for both positions (1.46x a single column, iteration 42).
    gpu_output_argmax_n2(&m->output_weight, d_norm2, (int)dim, (int)vocab_size, out_next);

    eng->t_total += omp_get_wtime() - t_token;
}

// One speculative step.
//
// Contract, matching what engine_forward_token already establishes: on entry
// `token` is the (not yet consumed) token at `pos`, the engine has consumed
// 0..pos-1, and eng->last_draft_token holds a draft for pos+1. On return
// eng->last_draft_token drafts the position after *next_token.
//
// Safety: the draft head is a heuristic and nothing depends on its quality
// beyond speed. Every token this function emits is the argmax of the real
// trunk over a verified prefix, so a bad draft costs a rollback, never a wrong
// token. That is the property iteration 36 lost by accepting the draft on its
// own say-so.
int engine_spec_step(engine_t *eng, int32_t token, int pos,
                     int32_t out_tokens[4], int32_t *next_token, spec_stats_t *st) {
    model_t *m = eng->model;
    int32_t d0 = (g_spec_draft == SPEC_DRAFT_EXTERNAL && eng->drafter) ?
                 drafter_poll_draft(eng->drafter) :
                 ((eng->last_draft_count > 0) ? eng->last_draft_tokens[0] : eng->last_draft_token);
    int32_t d1 = (eng->last_draft_count > 1) ? eng->last_draft_tokens[1] : -1;

    // No draft yet (first step), or no room for a second position: plain decode.
    if (d0 < 0 || d0 >= (int32_t)m->vocab_size || pos + 1 >= eng->max_seq_len) {
        int32_t next = engine_forward_token(eng, token, pos);
        out_tokens[0] = next;
        *next_token = next;
        if (st) { st->steps++; st->tokens_emitted++; }
        return 1;
    }

    int32_t out[2];
    engine_forward_verify(eng, token, d0, pos, out);
    if (st) { st->steps++; st->drafted++; }

    const int accepted0 = (out[0] == d0);
    float *d_res2 = gpu_get_d_res_x2();
    uint32_t dim = m->embedding_length;
    int n_emitted = 0;

    if (accepted0) {
        if (st) st->accepted++;
        if (eng->hist_tokens) {
            engine_hist_append(eng, token);
            engine_hist_append(eng, out[0]);
        }

        // Check if secondary draft token d1 is available, valid, and matched by out[1]
        if (d1 >= 0 && d1 < (int32_t)m->vocab_size && out[1] == d1 && pos + 2 < eng->max_seq_len) {
            if (st) { st->drafted++; st->accepted++; }
            if (eng->hist_tokens) {
                engine_hist_append(eng, out[1]);
            }
            // Both d0 (at pos+1) and d1 (at pos+2) stand!
            // Evaluate continuation of d1 at pos+2 with single forward pass:
            int32_t y2 = engine_forward_token(eng, d1, pos + 2);
            out_tokens[0] = out[0];
            out_tokens[1] = out[1];
            out_tokens[2] = y2;
            *next_token = y2;
            n_emitted = 3;
        } else {
            // d0 stands at pos+1, out[1] at pos+2 is the true next token
            out_tokens[0] = out[0];
            out_tokens[1] = out[1];
            *next_token = out[1];
            n_emitted = 2;
            engine_set_draft(eng, out[1], pos + 1, d_res2 + (size_t)dim);
        }
    } else {
        // The draft was wrong: keep only the verified token and rewind the
        // recurrent state to the snapshot taken between the two columns.
        gpu_delta_rollback_n2();
        if (eng->drafter) drafter_rollback(eng->drafter, pos + 1);
        out_tokens[0] = out[0];
        *next_token = out[0];
        n_emitted = 1;
        // H17: the accepted sequence now ends with token (pos); the draft below
        // predicts the token after out[0] (pos+1).
        if (eng->hist_tokens) engine_hist_append(eng, token);
        engine_set_draft(eng, out[0], pos, d_res2);
    }

    if (st) st->tokens_emitted += n_emitted;
    return n_emitted;
}
