// CPU evaluation of MoE experts (hybrid miss path). See include/lfm25_cpu_expert.h.
//
// Bit-exactness contract (checked by tests/test_lfm25_cpu_expert.cu against the GPU kernels, memcmp):
//   per lane, per block   Q4_K: integer dp4a sums dot1/dot2, then
//                           sumf_d = fma(d8[0], (float)(dot1_0*sc0), 0); sumf_d = fma(d8[1], (float)(dot1_1*sc1), sumf_d)
//                           sumf_m likewise with dot2/m;  result = fma(sumf_d, d, -(sumf_m*dmin))
//                         Q6_K: sumf = fma(d8[0], (float)(dp4a0*sc0), 0); sumf = fma(d8[1], (float)(dp4a1*sc1), sumf)
//   lane accumulation     Q4_K: tmp += result (plain FADD); Q6_K: tmp = fma(d, sumf, tmp) (nvcc contracts tmp += d*sumf),
//                         over increasing kbx starting from 0.0f
//   warp reduction        for mask in 16,8,4,2,1: x[l] = x[l] + x[l^mask]; lane 0 is the row value
// This file must be compiled with -ffp-contract=off (set in CMake) so the C compiler adds no contractions of its own.
#include "lfm25_cpu_expert.h"
#include "lfm25_engine.h"
#include "ggml_types.h"
#include "cuda_check.h"
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

// ---- scalar reference (literal transcription of the CUDA source) ----------------------------------------------------
static int dp4a_s(int a, int b) {
    int s = 0;
    for (int i = 0; i < 4; ++i) s += (int)(int8_t)((a >> (8 * i)) & 0xff) * (int)(int8_t)((b >> (8 * i)) & 0xff);
    return s;
}

static float dot_q4k_s(const block_q4_K *b, const q8blk *y, int iqs) {
    int v[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const uint8_t *q4 = b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    v[0] = ld32(q4);
    v[1] = ld32(q4 + 16);
    uint16_t scales[6];
    memcpy(scales, b->scales, 12);
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m = sc + 2;
    for (int i = 0; i < 2; ++i) {
        const q8blk *bq8i = y + bq8_offset + i;
        d8[i] = h2f_c(bq8i->d);
        const int8_t *q8 = bq8i->qs + 4 * ((iqs / 2) % 4);
        u[2 * i + 0] = ld32(q8);
        u[2 * i + 1] = ld32(q8 + 16);
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = dp4a_s(v1i, u[2 * i + 1]) + dp4a_s(v0i, u[2 * i + 0]);
        const int dot2 = dp4a_s(0x01010101, u[2 * i + 1]) + dp4a_s(0x01010101, u[2 * i + 0]);
        sumf_d = fmaf(d8[i], (float)(dot1 * (int)sc[i]), sumf_d);
        sumf_m = fmaf(d8[i], (float)(dot2 * (int)m[i]), sumf_m);
    }
    const float prod = sumf_m * h2f_c(b->dmin);
    return fmaf(sumf_d, h2f_c(b->d), -prod);
}

static float dot_q6k_s(const block_q6_K *b, const q8blk *y, int iqs) {
    const int bq8_offset = 2 * 2 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = ld32(b->ql + 4 * iqs);
    const int vh = ld32(b->qh + 4 * (8 * (iqs / 16) + iqs % 8)) >> vh_shift;
    const int8_t *scales = b->scales + scale_offset;
    int u[2];
    float d8[2];
    for (int i = 0; i < 2; ++i) {
        u[i] = ld32(y[bq8_offset + 2 * i].qs + 4 * (iqs % 8));
        d8[i] = h2f_c(y[bq8_offset + 2 * i].d);
    }
    float sumf = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = (int)(((uint32_t)(vh >> (4 * i))) << 4) & 0x30303030;
        const int vv = vil | vih;
        int vi = 0;
        for (int bb = 0; bb < 4; ++bb) {   // __vsubss4: per-byte signed saturating subtract of 0x20
            int t = (int)(int8_t)((vv >> (8 * bb)) & 0xff) - 32;
            if (t < -128) t = -128;
            if (t > 127) t = 127;
            vi |= (t & 0xff) << (8 * bb);
        }
        sumf = fmaf(d8[i], (float)(dp4a_s(vi, u[i]) * sc), sumf);
    }
    return sumf;   // the caller applies d: tmp = fma(d, sumf, tmp) (nvcc contracts `tmp += d*sumf` in the Q6_K kernels)
}

static float reduce32_s(const float *acc) {
    float x[32], t[32];
    memcpy(x, acc, sizeof(x));
    for (int mask = 16; mask > 0; mask >>= 1) {
        for (int l = 0; l < 32; ++l) t[l] = x[l] + x[l ^ mask];
        memcpy(x, t, sizeof(x));
    }
    return x[0];
}

float lfm25_cpu_row_dot_scalar(int type, const void *row, const q8blk *y, int nblk) {
    float acc[32];
    for (int l = 0; l < 32; ++l) {
        float tmp = 0.0f;
        if (type == GGML_TYPE_Q4_K) {
            const int kqs = 2 * (l % 16);
            for (int kbx = l / 16; kbx < nblk; kbx += 2) tmp += dot_q4k_s((const block_q4_K *)row + kbx, y + kbx * 8, kqs);
        } else {
            for (int kbx = 0; kbx < nblk; kbx += 1) {
                const block_q6_K *b6 = (const block_q6_K *)row + kbx;
                tmp = fmaf(h2f_c(b6->d), dot_q6k_s(b6, y + kbx * 8, l), tmp);
            }
        }
        acc[l] = tmp;
    }
    return reduce32_s(acc);
}

// ---- AVX2 -------------------------------------------------------------------------------------------------------------
// Final warp reduction from s[q] = x[q] + x[q+16] (q = 0..15): masks 8,4,2,1 for lane 0.
static inline float reduce_tail16(const float *s) {
    __m128 r0 = _mm_loadu_ps(s), r1 = _mm_loadu_ps(s + 4), r2 = _mm_loadu_ps(s + 8), r3 = _mm_loadu_ps(s + 12);
    __m128 t0 = _mm_add_ps(r0, r2), t1 = _mm_add_ps(r1, r3);      // mask 8
    __m128 u = _mm_add_ps(t0, t1);                                // mask 4
    __m128 v = _mm_add_ps(u, _mm_shuffle_ps(u, u, _MM_SHUFFLE(1, 0, 3, 2)));   // mask 2: v0=u0+u2, v1=u1+u3
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
    __m256 acc[4];   // lanes 0-7, 8-15, 16-23, 24-31
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
            __m256 sumf[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };   // [A (m 0..7), B (m 8..15)]
            for (int i = 0; i < 2; ++i) {
                const __m256i nA = i ? _mm256_and_si256(_mm256_srli_epi16(A, 4), m0f) : _mm256_and_si256(A, m0f);
                const __m256i nB = i ? _mm256_and_si256(_mm256_srli_epi16(B, 4), m0f) : _mm256_and_si256(B, m0f);
                const __m256i hA = _mm256_slli_epi16(_mm256_and_si256(i ? _mm256_srli_epi16(H, 4) : H, m03), 4);
                const __m256i hB = _mm256_slli_epi16(_mm256_and_si256(i ? _mm256_srli_epi16(H, 6) : _mm256_srli_epi16(H, 2), m03), 4);
                const __m256i vuA = _mm256_or_si256(nA, hA), vuB = _mm256_or_si256(nB, hB);   // 0..63 (value + 32)
                for (int part = 0; part < 2; ++part) {
                    const q8blk *yq = yb + 4 * h + 2 * i + part;
                    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq->qs);
                    const __m256i vu = part ? vuB : vuA;
                    const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(vu, yv), ones16);
                    const __m256i s2 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, yv), ones16);
                    const __m256i dot = _mm256_sub_epi32(s1, _mm256_slli_epi32(s2, 5));      // sum(vu*y) - 32*sum(y)
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

float lfm25_cpu_row_dot_avx2(int type, const void *row, const q8blk *y, int nblk) {
    return type == GGML_TYPE_Q4_K ? row_dot_q4k_avx2((const block_q4_K *)row, y, nblk)
                                  : row_dot_q6k_avx2((const block_q6_K *)row, y, nblk);
}

// ---- persistent thread pool ---------------------------------------------------------------------------------------------
#define CHUNK_ROWS 8

typedef struct {
    int type, nblk, rows;
    size_t row_bytes;
    const char *w;
    const q8blk *y;
    float *dst;
    int first_task;
} cjob_t;

typedef struct {
    cjob_t jobs[16];
    int njobs;
    int scalar;
} cdesc_t;

typedef struct {
    pthread_t *th;
    int nth;                    // background threads
    cdesc_t desc[2];
    int par;
    atomic_uint_fast64_t ctl;   // bits 0-31 next task, 32-47 ntasks, 48 descriptor parity
    atomic_int done;
    atomic_int quit;
    atomic_int sleepers;
    atomic_uint wake;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    long spin_ns;
} cpool_t;

static void run_task(const cdesc_t *d, int t) {
    int j = d->njobs - 1;
    while (j > 0 && d->jobs[j].first_task > t) --j;
    const cjob_t *jb = &d->jobs[j];
    const int r0 = (t - jb->first_task) * CHUNK_ROWS;
    int r1 = r0 + CHUNK_ROWS;
    if (r1 > jb->rows) r1 = jb->rows;
    for (int r = r0; r < r1; ++r) {
        const void *row = jb->w + (size_t)r * jb->row_bytes;
        jb->dst[r] = d->scalar ? lfm25_cpu_row_dot_scalar(jb->type, row, jb->y, jb->nblk)
                               : lfm25_cpu_row_dot_avx2(jb->type, row, jb->y, jb->nblk);
    }
}

// Claim and run tasks until none are left; returns the number run.
static int drain(cpool_t *p) {
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
    cpool_t *p = (cpool_t *)arg;
    long idle_since = now_ns();
    unsigned spins = 0;
    for (;;) {
        if (atomic_load(&p->quit)) break;
        if (drain(p) > 0) { idle_since = now_ns(); spins = 0; continue; }
        _mm_pause();
        if ((++spins & 255u) != 0) continue;
        if (now_ns() - idle_since < p->spin_ns) continue;
        // idle too long: sleep until a job is published
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

static cpool_t* pool_create(int nthreads) {
    cpool_t *p = (cpool_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    const char *e = getenv("LFM25_CPU_SPIN_US");
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

static void pool_free(cpool_t *p) {
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

// Run all rows of all jobs; returns when every row is written.
static void pool_run(cpool_t *p, const cjob_t *jobs, int njobs, int scalar) {
    cdesc_t *d = &p->desc[p->par ^ 1];
    int nt = 0;
    for (int j = 0; j < njobs; ++j) {
        d->jobs[j] = jobs[j];
        d->jobs[j].first_task = nt;
        nt += (jobs[j].rows + CHUNK_ROWS - 1) / CHUNK_ROWS;
    }
    d->njobs = njobs;
    d->scalar = scalar;
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

// ---- hybrid orchestration ---------------------------------------------------------------------------------------------------
struct lfm25_cpu_expert {
    cpool_t *pool;
    int nthreads, dim, ffn, scalar;
    void *h_xq8, *h_act_q8;
    float *h_pre_gu, *h_pre_y;
    float *d_pre_gu, *d_pre_y;
};

lfm25_cpu_expert_t* lfm25_cpu_expert_create(int nthreads, int dim, int ffn_dim) {
    if (nthreads <= 0) {
        const char *e = getenv("LFM25_CPU_THREADS");
        nthreads = e && *e ? atoi(e) : 12;
    }
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    lfm25_cpu_expert_t *ce = (lfm25_cpu_expert_t *)calloc(1, sizeof(*ce));
    if (!ce) return NULL;
    ce->dim = dim; ce->ffn = ffn_dim;
    CUDA_CHECK(cudaMallocHost(&ce->h_xq8, (size_t)(dim / 32) * 36));
    CUDA_CHECK(cudaMallocHost(&ce->h_act_q8, (size_t)4 * (ffn_dim / 32) * 36));
    CUDA_CHECK(cudaMallocHost((void **)&ce->h_pre_gu, (size_t)8 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **)&ce->h_pre_y, (size_t)4 * dim * sizeof(float)));
    memset(ce->h_pre_gu, 0, (size_t)8 * ffn_dim * sizeof(float));
    memset(ce->h_pre_y, 0, (size_t)4 * dim * sizeof(float));
    CUDA_CHECK(cudaMalloc((void **)&ce->d_pre_gu, (size_t)8 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&ce->d_pre_y, (size_t)4 * dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(ce->d_pre_gu, 0, (size_t)8 * ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(ce->d_pre_y, 0, (size_t)4 * dim * sizeof(float)));
    ce->pool = pool_create(nthreads);
    if (!ce->pool) { lfm25_cpu_expert_free(ce); return NULL; }
    ce->nthreads = ce->pool->nth + 1;
    return ce;
}

void lfm25_cpu_expert_free(lfm25_cpu_expert_t *ce) {
    if (!ce) return;
    pool_free(ce->pool);
    cudaFreeHost(ce->h_xq8); cudaFreeHost(ce->h_act_q8); cudaFreeHost(ce->h_pre_gu); cudaFreeHost(ce->h_pre_y);
    cudaFree(ce->d_pre_gu); cudaFree(ce->d_pre_y);
    free(ce);
}

int lfm25_cpu_expert_threads(const lfm25_cpu_expert_t *ce) { return ce->nthreads; }
void lfm25_cpu_expert_set_scalar(lfm25_cpu_expert_t *ce, int scalar) { ce->scalar = scalar; }
void* lfm25_cpu_expert_host_xq8(lfm25_cpu_expert_t *ce) { return ce->h_xq8; }

void lfm25_cpu_expert_moe_ffn(lfm25_cpu_expert_t *ce, cudaStream_t stream, const void *d_xq8, const void *h_xq8,
                              const int *d_slot_ids, const float *d_topk_w,
                              const void *d_gate, const void *d_up, const void *d_down, int down_type,
                              float *d_act, void *d_act_q8, float *d_dst, bool accum, int dim, int ffn_dim,
                              size_t gate_stride, size_t up_stride, size_t down_stride,
                              unsigned miss_mask, const void *const hg[4], const void *const hu[4], const void *const hd[4]) {
    const unsigned all = 0xFu;
    miss_mask &= all;
    // 1. gate/up of the resident experts start on the GPU now; they overlap the CPU work below.
    if (miss_mask != all)
        lfm25_gpu_moe_gate_up(d_xq8, d_slot_ids, d_gate, d_up, d_act, dim, ffn_dim, gate_stride, up_stride, miss_mask, 0u, NULL);
    // 2. CPU: raw gate/up row dots of the missed experts.
    const int nblk_in = dim / 256, nblk_dn = ffn_dim / 256;
    cjob_t jobs[8];
    int nj = 0;
    for (int k = 0; k < 4; ++k) {
        if (!((miss_mask >> k) & 1u)) continue;
        cjob_t g = { GGML_TYPE_Q4_K, nblk_in, ffn_dim, (size_t)nblk_in * sizeof(block_q4_K), (const char *)hg[k],
                     (const q8blk *)h_xq8, ce->h_pre_gu + (size_t)(k * 2 + 0) * ffn_dim, 0 };
        cjob_t u = g;
        u.w = (const char *)hu[k];
        u.dst = ce->h_pre_gu + (size_t)(k * 2 + 1) * ffn_dim;
        jobs[nj++] = g;
        jobs[nj++] = u;
    }
    pool_run(ce->pool, jobs, nj, ce->scalar);
    CUDA_CHECK(cudaMemcpyAsync(ce->d_pre_gu, ce->h_pre_gu, (size_t)8 * ffn_dim * sizeof(float), cudaMemcpyHostToDevice, stream));
    lfm25_gpu_moe_gate_up(d_xq8, d_slot_ids, d_gate, d_up, d_act, dim, ffn_dim, gate_stride, up_stride, all & ~miss_mask, miss_mask, ce->d_pre_gu);
    lfm25_gpu_quantize_q8_1(d_act, d_act_q8, 4 * ffn_dim);
    // 3. activations of the missed experts to the host, CPU down rows, back.
    CUDA_CHECK(cudaMemcpyAsync(ce->h_act_q8, d_act_q8, (size_t)4 * (ffn_dim / 32) * 36, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const size_t row_dn = (size_t)nblk_dn * (down_type == GGML_TYPE_Q6_K ? sizeof(block_q6_K) : sizeof(block_q4_K));
    nj = 0;
    for (int k = 0; k < 4; ++k) {
        if (!((miss_mask >> k) & 1u)) continue;
        cjob_t d = { down_type, nblk_dn, dim, row_dn, (const char *)hd[k],
                     (const q8blk *)ce->h_act_q8 + (size_t)k * (ffn_dim / 32), ce->h_pre_y + (size_t)k * dim, 0 };
        jobs[nj++] = d;
    }
    pool_run(ce->pool, jobs, nj, ce->scalar);
    CUDA_CHECK(cudaMemcpyAsync(ce->d_pre_y, ce->h_pre_y, (size_t)4 * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
    lfm25_gpu_moe_down(d_act_q8, d_slot_ids, d_topk_w, d_down, down_type, d_dst, accum, dim, ffn_dim, down_stride, miss_mask, ce->d_pre_y);
}
