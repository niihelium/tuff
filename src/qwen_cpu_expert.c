// Byte-identical CPU miss path for Qwen3.5/3.6 MoE architecture.
// See include/qwen_cpu_expert.h.
#include "qwen_cpu_expert.h"
#include "cuda_check.h"
#include "ggml_types.h"
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef lfm25_q8blk_t q8blk;

static inline float h2f_c(uint16_t h) { return _cvtsh_ss(h); }
static inline int32_t ld32(const void *p) { int32_t v; memcpy(&v, p, 4); return v; }

static inline float reduce_tail16(const float *s) {
    __m128 r0 = _mm_loadu_ps(s), r1 = _mm_loadu_ps(s + 4), r2 = _mm_loadu_ps(s + 8), r3 = _mm_loadu_ps(s + 12);
    __m128 t0 = _mm_add_ps(r0, r2), t1 = _mm_add_ps(r1, r3);      // mask 8
    __m128 u = _mm_add_ps(t0, t1);                                // mask 4
    __m128 v = _mm_add_ps(u, _mm_shuffle_ps(u, u, _MM_SHUFFLE(1, 0, 3, 2)));   // mask 2
    return _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));          // mask 1
}

static float row_dot_q4k_avx2(const block_q4_K *rw, const q8blk *y, int nblk) {
    __m128 acc[2][4];
    for (int h = 0; h < 2; ++h) for (int g = 0; g < 4; ++g) acc[h][g] = _mm_setzero_ps();
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m128 negz = _mm_set1_ps(-0.0f);
    for (int kbx = 0; kbx < nblk; ++kbx) {
        const block_q4_K *b = rw + kbx;
        const q8blk *yb = y + kbx * 8;
        const int h = kbx & 1;
        uint16_t scales[6];
        memcpy(scales, b->scales, 12);
        const __m128 dv = _mm_set1_ps(h2f_c(b->d)), dminv = _mm_set1_ps(h2f_c(b->dmin));
        for (int g = 0; g < 4; ++g) {
            uint16_t aux[2];
            if (g < 2) {
                aux[0] = scales[g + 0] & 0x3f3f;
                aux[1] = scales[g + 2] & 0x3f3f;
            } else {
                aux[0] = ((scales[g + 2] >> 0) & 0x0f0f) | ((scales[g - 2] & 0xc0c0) >> 2);
                aux[1] = ((scales[g + 2] >> 4) & 0x0f0f) | ((scales[g - 0] & 0xc0c0) >> 2);
            }
            const uint8_t *sc = (const uint8_t *)aux;
            const uint8_t *m = sc + 2;
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(b->qs + 32 * g));
            const __m256i lo = _mm256_and_si256(qs, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qs, 4), m0f);
            const __m256i y0 = _mm256_loadu_si256((const __m256i *)yb[2 * g].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i *)yb[2 * g + 1].qs);
            const __m256i p10 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), ones16);
            const __m256i p11 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), ones16);
            const __m256i p20 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, y0), ones16);
            const __m256i p21 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, y1), ones16);
            const __m128i dot1_0 = _mm_add_epi32(_mm256_castsi256_si128(p10), _mm256_extracti128_si256(p10, 1));
            const __m128i dot1_1 = _mm_add_epi32(_mm256_castsi256_si128(p11), _mm256_extracti128_si256(p11, 1));
            const __m128i dot2_0 = _mm_add_epi32(_mm256_castsi256_si128(p20), _mm256_extracti128_si256(p20, 1));
            const __m128i dot2_1 = _mm_add_epi32(_mm256_castsi256_si128(p21), _mm256_extracti128_si256(p21, 1));
            const __m128 d8_0 = _mm_set1_ps(h2f_c(yb[2 * g].d)), d8_1 = _mm_set1_ps(h2f_c(yb[2 * g + 1].d));
            __m128 sumf_d = _mm_fmadd_ps(d8_0, _mm_cvtepi32_ps(_mm_mullo_epi32(dot1_0, _mm_set1_epi32(sc[0]))), _mm_setzero_ps());
            sumf_d = _mm_fmadd_ps(d8_1, _mm_cvtepi32_ps(_mm_mullo_epi32(dot1_1, _mm_set1_epi32(sc[1]))), sumf_d);
            __m128 sumf_m = _mm_fmadd_ps(d8_0, _mm_cvtepi32_ps(_mm_mullo_epi32(dot2_0, _mm_set1_epi32(m[0]))), _mm_setzero_ps());
            sumf_m = _mm_fmadd_ps(d8_1, _mm_cvtepi32_ps(_mm_mullo_epi32(dot2_1, _mm_set1_epi32(m[1]))), sumf_m);
            const __m128 negprod = _mm_xor_ps(_mm_mul_ps(sumf_m, dminv), negz);
            const __m128 res = _mm_fmadd_ps(sumf_d, dv, negprod);
            acc[h][g] = _mm_add_ps(acc[h][g], res);
        }
    }
    float s[16];
    for (int g = 0; g < 4; ++g) _mm_storeu_ps(s + 4 * g, _mm_add_ps(acc[0][g], acc[1][g]));   // mask 16
    return reduce_tail16(s);
}

static float row_dot_q6k_avx2(const block_q6_K *rw, const q8blk *y, int nblk) {
    __m256 acc[4];
    for (int i = 0; i < 4; ++i) acc[i] = _mm256_setzero_ps();
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int kbx = 0; kbx < nblk; ++kbx) {
        const block_q6_K *b = rw + kbx;
        const q8blk *yb = y + kbx * 8;
        const __m256 dv = _mm256_set1_ps(h2f_c(b->d));
        for (int h = 0; h < 2; ++h) {
            const __m256i A = _mm256_loadu_si256((const __m256i *)(b->ql + 64 * h));
            const __m256i B = _mm256_loadu_si256((const __m256i *)(b->ql + 64 * h + 32));
            const __m256i H = _mm256_loadu_si256((const __m256i *)(b->qh + 32 * h));
            __m256 sumf[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
            for (int i = 0; i < 2; ++i) {
                const __m256i nA = i ? _mm256_and_si256(_mm256_srli_epi16(A, 4), m0f) : _mm256_and_si256(A, m0f);
                const __m256i nB = i ? _mm256_and_si256(_mm256_srli_epi16(B, 4), m0f) : _mm256_and_si256(B, m0f);
                const __m256i hA = _mm256_slli_epi16(_mm256_and_si256(i ? _mm256_srli_epi16(H, 4) : H, m03), 4);
                const __m256i hB = _mm256_slli_epi16(_mm256_and_si256(i ? _mm256_srli_epi16(H, 6) : _mm256_srli_epi16(H, 2), m03), 4);
                const __m256i vuA = _mm256_or_si256(nA, hA), vuB = _mm256_or_si256(nB, hB);
                for (int part = 0; part < 2; ++part) {
                    const q8blk *yq = yb + 4 * h + 2 * i + part;
                    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq->qs);
                    const __m256i vu = part ? vuB : vuA;
                    const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(vu, yv), ones16);
                    const __m256i s2 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, yv), ones16);
                    const __m256i dot = _mm256_sub_epi32(s1, _mm256_slli_epi32(s2, 5));
                    const int8_t *sp = b->scales + 8 * h + 4 * i + 2 * part;
                    const __m128i sc_lo = _mm_set1_epi32(sp[0]), sc_hi = _mm_set1_epi32(sp[1]);
                    const __m256i sc = _mm256_inserti128_si256(_mm256_castsi128_si256(sc_lo), sc_hi, 1);
                    const __m256 x = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot, sc));
                    sumf[part] = _mm256_fmadd_ps(_mm256_set1_ps(h2f_c(yq->d)), x, sumf[part]);
                }
            }
            for (int part = 0; part < 2; ++part)
                acc[2 * h + part] = _mm256_fmadd_ps(dv, sumf[part], acc[2 * h + part]);
        }
    }
    float s[16];
    _mm256_storeu_ps(s, _mm256_add_ps(acc[0], acc[2]));        // mask 16
    _mm256_storeu_ps(s + 8, _mm256_add_ps(acc[1], acc[3]));
    return reduce_tail16(s);
}

static inline float qwen_cpu_row_dot_avx2(int type, const void *row, const q8blk *y, int nblk) {
    return type == GGML_TYPE_Q4_K ? row_dot_q4k_avx2((const block_q4_K *)row, y, nblk)
                                  : row_dot_q6k_avx2((const block_q6_K *)row, y, nblk);
}

#define CHUNK_ROWS 8

typedef struct {
    int type;
    int nblk;
    int rows;
    size_t row_bytes;
    const char *w;
    const q8blk *y;
    float *dst;
    int first_task;
} qwen_cjob_t;

typedef struct {
    qwen_cjob_t jobs[16];
    int njobs;
} qwen_cdesc_t;

typedef struct {
    pthread_t *th;
    int nth;
    qwen_cdesc_t desc[2];
    int par;
    atomic_uint_fast64_t ctl;
    atomic_int done;
    atomic_int quit;
    atomic_int sleepers;
    atomic_uint wake;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    long spin_ns;
} qwen_cpool_t;

static void run_task(const qwen_cdesc_t *d, int t) {
    int j = d->njobs - 1;
    while (j > 0 && d->jobs[j].first_task > t) --j;
    const qwen_cjob_t *jb = &d->jobs[j];
    const int r0 = (t - jb->first_task) * CHUNK_ROWS;
    int r1 = r0 + CHUNK_ROWS;
    if (r1 > jb->rows) r1 = jb->rows;
    for (int r = r0; r < r1; ++r) {
        const void *row = jb->w + (size_t)r * jb->row_bytes;
        jb->dst[r] = qwen_cpu_row_dot_avx2(jb->type, row, jb->y, jb->nblk);
    }
}

static int drain(qwen_cpool_t *p) {
    int n = 0;
    for (;;) {
        uint64_t v = atomic_load_explicit(&p->ctl, memory_order_acquire);
        uint32_t idx = (uint32_t)(v & 0xffffffffu), nt = (uint32_t)((v >> 32) & 0xffffu);
        if (idx >= nt) return n;
        v = atomic_fetch_add_explicit(&p->ctl, 1, memory_order_acq_rel);
        idx = (uint32_t)(v & 0xffffffffu); nt = (uint32_t)((v >> 32) & 0xffffu);
        if (idx >= nt) continue;
        run_task(&p->desc[(v >> 48) & 1], (int)idx);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_acq_rel);
        ++n;
    }
}

static long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static void* pool_worker(void *arg) {
    qwen_cpool_t *p = (qwen_cpool_t *)arg;
    long idle_since = now_ns();
    unsigned spins = 0;
    for (;;) {
        if (atomic_load(&p->quit)) break;
        if (drain(p) > 0) { idle_since = now_ns(); spins = 0; continue; }
        _mm_pause();
        if ((++spins & 255u) != 0) continue;
        if (now_ns() - idle_since < p->spin_ns) continue;
        pthread_mutex_lock(&p->mu);
        atomic_fetch_add(&p->sleepers, 1);
        for (;;) {
            uint64_t v = atomic_load(&p->ctl);
            if (atomic_load(&p->quit) || (uint32_t)(v & 0xffffffffu) < (uint32_t)((v >> 32) & 0xffffu)) break;
            pthread_cond_wait(&p->cv, &p->mu);
        }
        atomic_fetch_sub(&p->sleepers, 1);
        pthread_mutex_unlock(&p->mu);
        idle_since = now_ns();
    }
    return NULL;
}

static qwen_cpool_t* pool_create(int nthreads) {
    qwen_cpool_t *p = (qwen_cpool_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    const char *e = getenv("TUFF_CPU_SPIN_US");
    p->spin_ns = (e && *e ? atol(e) : 2000) * 1000L;
    atomic_init(&p->ctl, 0); atomic_init(&p->done, 0); atomic_init(&p->quit, 0);
    atomic_init(&p->sleepers, 0); atomic_init(&p->wake, 0);
    p->nth = nthreads > 1 ? nthreads - 1 : 0;
    p->th = (pthread_t *)calloc((size_t)p->nth + 1, sizeof(pthread_t));
    int started = 0;
    for (int i = 0; i < p->nth; ++i)
        if (pthread_create(&p->th[started], NULL, pool_worker, p) == 0) started++;
    p->nth = started;
    return p;
}

static void pool_free(qwen_cpool_t *p) {
    if (!p) return;
    atomic_store(&p->quit, 1);
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nth; ++i) pthread_join(p->th[i], NULL);
    free(p->th);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p);
}

static void pool_run(qwen_cpool_t *p, const qwen_cjob_t *jobs, int njobs) {
    qwen_cdesc_t *d = &p->desc[p->par ^ 1];
    int nt = 0;
    for (int j = 0; j < njobs; ++j) {
        d->jobs[j] = jobs[j];
        d->jobs[j].first_task = nt;
        nt += (jobs[j].rows + CHUNK_ROWS - 1) / CHUNK_ROWS;
    }
    d->njobs = njobs;
    p->par ^= 1;
    atomic_store(&p->done, 0);
    atomic_store_explicit(&p->ctl, ((uint64_t)p->par << 48) | ((uint64_t)nt << 32), memory_order_seq_cst);
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    drain(p);
    while (atomic_load_explicit(&p->done, memory_order_acquire) < nt) _mm_pause();
}

struct qwen_cpu_expert {
    qwen_cpool_t *pool;
    int nthreads, dim, ffn;
    void *h_xq8, *h_act_q8;
    float *h_pre_gu, *h_pre_y;
    float *d_pre_gu, *d_pre_y;
    float *d_act;
    void  *d_act_q8;
};

// Declared in src/gpu.cu
void gpu_qwen_moe_swiglu_pre_gu(cudaStream_t stream, const float *d_pre_gu,
                                float *d_act, void *d_act_q8,
                                unsigned miss_mask, int ffn_dim);

qwen_cpu_expert_t* qwen_cpu_expert_create(int nthreads, int dim, int ffn_dim) {
    if (nthreads <= 0) {
        const char *e = getenv("TUFF_CPU_THREADS");
        nthreads = e && *e ? atoi(e) : 8;
    }
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;

    qwen_cpu_expert_t *ce = (qwen_cpu_expert_t *)calloc(1, sizeof(*ce));
    if (!ce) return NULL;
    ce->dim = dim;
    ce->ffn = ffn_dim;

    CUDA_CHECK(cudaMallocHost(&ce->h_xq8, (size_t)(dim / 32) * 36));
    CUDA_CHECK(cudaMallocHost(&ce->h_act_q8, (size_t)8 * (ffn_dim / 32) * 36));
    CUDA_CHECK(cudaMallocHost((void **)&ce->h_pre_gu, (size_t)8 * 2 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **)&ce->h_pre_y, (size_t)8 * dim * sizeof(float)));

    memset(ce->h_pre_gu, 0, (size_t)8 * 2 * ffn_dim * sizeof(float));
    memset(ce->h_pre_y, 0, (size_t)8 * dim * sizeof(float));

    CUDA_CHECK(cudaMalloc((void **)&ce->d_pre_gu, (size_t)8 * 2 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&ce->d_pre_y, (size_t)8 * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&ce->d_act, (size_t)8 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ce->d_act_q8, (size_t)8 * (ffn_dim / 32) * 36));
    CUDA_CHECK(cudaMemset(ce->d_pre_gu, 0, (size_t)8 * 2 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(ce->d_pre_y, 0, (size_t)8 * dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(ce->d_act, 0, (size_t)8 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(ce->d_act_q8, 0, (size_t)8 * (ffn_dim / 32) * 36));

    ce->pool = pool_create(nthreads);
    if (!ce->pool) { qwen_cpu_expert_free(ce); return NULL; }
    ce->nthreads = ce->pool->nth + 1;
    return ce;
}

void qwen_cpu_expert_free(qwen_cpu_expert_t *ce) {
    if (!ce) return;
    pool_free(ce->pool);
    cudaFreeHost(ce->h_xq8);
    cudaFreeHost(ce->h_act_q8);
    cudaFreeHost(ce->h_pre_gu);
    cudaFreeHost(ce->h_pre_y);
    cudaFree(ce->d_pre_gu);
    cudaFree(ce->d_pre_y);
    cudaFree(ce->d_act);
    cudaFree(ce->d_act_q8);
    free(ce);
}

int   qwen_cpu_expert_threads(const qwen_cpu_expert_t *ce) { return ce->nthreads; }
void* qwen_cpu_expert_host_xq8(qwen_cpu_expert_t *ce) { return ce->h_xq8; }
float* qwen_cpu_expert_host_pre_y(qwen_cpu_expert_t *ce) { return ce->h_pre_y; }
float* qwen_cpu_expert_d_pre_gu(qwen_cpu_expert_t *ce) { return ce->d_pre_gu; }
float* qwen_cpu_expert_d_pre_y(qwen_cpu_expert_t *ce) { return ce->d_pre_y; }


void qwen_cpu_expert_eval(qwen_cpu_expert_t *ce, cudaStream_t stream,
                          const void *d_xq8, int down_type,
                          int dim, int ffn_dim,
                          unsigned miss_mask,
                          const void *const hg[8], const void *const hu[8], const void *const hd[8]) {
    miss_mask &= 0xFFu;
    if (miss_mask == 0u) return;

    // 1. Download xq8 to host
    CUDA_CHECK(cudaMemcpyAsync(ce->h_xq8, d_xq8, (size_t)(dim / 32) * 36, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Gate/Up row dots on CPU
    const int nblk_in = dim / 256;
    const int nblk_dn = ffn_dim / 256;
    qwen_cjob_t jobs[16];
    int nj = 0;
    for (int k = 0; k < 8; ++k) {
        if (!((miss_mask >> k) & 1u)) continue;
        qwen_cjob_t g = { GGML_TYPE_Q4_K, nblk_in, ffn_dim, (size_t)nblk_in * sizeof(block_q4_K),
                          (const char *)hg[k], (const q8blk *)ce->h_xq8,
                          ce->h_pre_gu + (size_t)(2 * k + 0) * ffn_dim, 0 };
        qwen_cjob_t u = g;
        u.w = (const char *)hu[k];
        u.dst = ce->h_pre_gu + (size_t)(2 * k + 1) * ffn_dim;
        jobs[nj++] = g;
        jobs[nj++] = u;
    }
    pool_run(ce->pool, jobs, nj);

    // 3. Upload pre_gu to device and run GPU SwiGLU + Q8_1 quantize
    CUDA_CHECK(cudaMemcpyAsync(ce->d_pre_gu, ce->h_pre_gu, (size_t)8 * 2 * ffn_dim * sizeof(float), cudaMemcpyHostToDevice, stream));
    gpu_qwen_moe_swiglu_pre_gu(stream, ce->d_pre_gu, ce->d_act, ce->d_act_q8, miss_mask, ffn_dim);

    // 4. Download missed experts' act_q8 to host
    CUDA_CHECK(cudaMemcpyAsync(ce->h_act_q8, ce->d_act_q8, (size_t)8 * (ffn_dim / 32) * 36, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 5. Down row dots on CPU
    const size_t row_dn = (size_t)nblk_dn * (down_type == GGML_TYPE_Q6_K ? sizeof(block_q6_K) : sizeof(block_q4_K));
    nj = 0;
    for (int k = 0; k < 8; ++k) {
        if (!((miss_mask >> k) & 1u)) continue;
        qwen_cjob_t d = { down_type, nblk_dn, dim, row_dn, (const char *)hd[k],
                          (const q8blk *)ce->h_act_q8 + (size_t)k * (ffn_dim / 32),
                          ce->h_pre_y + (size_t)k * dim, 0 };
        jobs[nj++] = d;
    }
    pool_run(ce->pool, jobs, nj);

    // 6. Upload pre_y to device
    CUDA_CHECK(cudaMemcpyAsync(ce->d_pre_y, ce->h_pre_y, (size_t)8 * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
}
