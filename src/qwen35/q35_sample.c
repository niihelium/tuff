#include "q35_sample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void q35_sampling_defaults(q35_sampling *p) {
    memset(p, 0, sizeof(*p));
    p->temp = 0.80f;
    p->top_k = 40;
    p->top_p = 0.95f;
    p->min_p = 0.05f;
    p->repeat_penalty = 1.0f;
    p->freq_penalty = 0.0f;
    p->presence_penalty = 0.0f;
    p->penalty_last_n = 64;
    p->seed = 0;
    p->top_logprobs = 0;
}

static int penalties_enabled(const q35_sampling *p) {
    if (p->penalty_last_n <= 0) return 0;
    return p->repeat_penalty != 1.0f || p->freq_penalty != 0.0f || p->presence_penalty != 0.0f;
}

int q35_sampling_needs_logits(const q35_sampling *p) {
    if (p->logprobs) return 1;
    if (p->temp > 0.0f) return 1;
    return penalties_enabled(p);
}

uint64_t q35_rng_next(uint64_t *state) {
    // splitmix64: one uint64 of state, good enough for chi-square and reproducible.
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t q35_rng_seed(uint64_t user_seed, int slot, uint64_t counter) {
    uint64_t s = user_seed;
    if (s == 0) s = 0x853c49e6748fea9bull ^ counter;
    s ^= (uint64_t)(uint32_t)slot * 0x9E3779B97F4A7C15ull;
    if (s == 0) s = 1;
    return s;
}

// ---------------------------------------------------------------------------
// Penalties (llama_sampler_penalties: window of the last penalty_last_n tokens)
// ---------------------------------------------------------------------------

static int i32_cmp(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

static void apply_penalties(float *logits, int V, const q35_sampling *p,
                            const int32_t *hist, int n_hist) {
    if (!penalties_enabled(p)) return;
    int win = p->penalty_last_n;
    if (win > n_hist) win = n_hist;
    if (win <= 0) return;
    const int32_t *h = hist + (n_hist - win);

    if (win <= 256) {
        int32_t ids[256];
        int cnt[256];
        int n = 0;
        for (int i = 0; i < win; i++) {
            int32_t t = h[i];
            if (t < 0 || t >= V) continue;
            int j = 0;
            for (; j < n; j++) if (ids[j] == t) break;
            if (j < n) { cnt[j]++; continue; }
            ids[n] = t;
            cnt[n] = 1;
            n++;
        }
        for (int j = 0; j < n; j++) {
            float *l = &logits[ids[j]];
            if (*l <= 0.0f) *l *= p->repeat_penalty;
            else *l /= p->repeat_penalty;
            *l -= (float)cnt[j] * p->freq_penalty + (float)(cnt[j] > 0) * p->presence_penalty;
        }
        return;
    }
    // Large windows: sort the window and group identical tokens.
    int32_t *ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)win);
    int n = 0;
    for (int i = 0; i < win; i++) if (h[i] >= 0 && h[i] < V) ids[n++] = h[i];
    qsort(ids, (size_t)n, sizeof(int32_t), i32_cmp);
    for (int i = 0; i < n;) {
        int j = i + 1;
        while (j < n && ids[j] == ids[i]) j++;
        float *l = &logits[ids[i]];
        if (*l <= 0.0f) *l *= p->repeat_penalty;
        else *l /= p->repeat_penalty;
        *l -= (float)(j - i) * p->freq_penalty + p->presence_penalty;
        i = j;
    }
    free(ids);
}

// ---------------------------------------------------------------------------
// Candidate selection
// ---------------------------------------------------------------------------

typedef struct { float logit; int32_t id; } cand;

static int cand_cmp(const void *a, const void *b) {
    const cand *x = (const cand *)a, *y = (const cand *)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    return (x->id < y->id) ? -1 : (x->id > y->id);
}

// Bounded min-heap of size k (largest logits kept).
static void heap_sift_down(cand *h, int n, int i, int k) {
    (void)k;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, small = i;
        if (l < n && cand_cmp(&h[l], &h[small]) > 0) small = l;
        if (r < n && cand_cmp(&h[r], &h[small]) > 0) small = r;
        if (small == i) break;
        cand t = h[i]; h[i] = h[small]; h[small] = t;
        i = small;
    }
}

static int select_topk(const float *logits, int V, int k, cand *out) {
    if (k <= 0 || k > V) k = V;
    cand *heap = (cand *)malloc(sizeof(cand) * (size_t)k);
    int n = 0;
    for (int i = 0; i < V; i++) {
        cand c = { logits[i], i };
        if (n < k) {
            heap[n] = c;
            n++;
            // sift up (min-heap: parent must be <= child)
            int j = n - 1;
            while (j > 0) {
                int p = (j - 1) / 2;
                if (cand_cmp(&heap[p], &heap[j]) <= 0) break;
                cand t = heap[p]; heap[p] = heap[j]; heap[j] = t;
                j = p;
            }
        } else if (cand_cmp(&c, &heap[0]) < 0) {
            heap[0] = c;
            heap_sift_down(heap, n, 0, k);
        }
    }
    memcpy(out, heap, sizeof(cand) * (size_t)n);
    free(heap);
    qsort(out, (size_t)n, sizeof(cand), cand_cmp);
    return n;
}

// Exact slow path for top_k == 0: full-vocabulary ordering.
static int select_all(const float *logits, int V, cand *out) {
    for (int i = 0; i < V; i++) { out[i].logit = logits[i]; out[i].id = i; }
    qsort(out, (size_t)V, sizeof(cand), cand_cmp);
    return V;
}

// top_p over a sorted candidate set; returns the kept count.
static int apply_top_p(const q35_sampling *p, cand *c, int n) {
    if (p->top_p >= 1.0f || n <= 1) return n;
    float max_l = c[0].logit;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += exp((double)(c[i].logit - max_l));
    double acc = 0;
    for (int i = 0; i < n; i++) {
        acc += exp((double)(c[i].logit - max_l)) / sum;
        if (acc >= p->top_p) return i + 1;
    }
    return n;
}

static int apply_min_p(const q35_sampling *p, cand *c, int n) {
    if (p->min_p <= 0.0f || n <= 1) return n;
    float min_logit = c[0].logit + logf(p->min_p);
    int k = 1;
    while (k < n && c[k].logit >= min_logit) k++;
    return k;
}

// ---------------------------------------------------------------------------
// Chain
// ---------------------------------------------------------------------------

int32_t q35_sampling_greedy(float *logits, int V, const q35_sampling *p,
                            const int32_t *hist, int n_hist) {
    apply_penalties(logits, V, p, hist, n_hist);
    int best = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

int32_t q35_sampling_sample(float *logits, int V, const q35_sampling *p,
                            const int32_t *hist, int n_hist, uint64_t *rng) {
    apply_penalties(logits, V, p, hist, n_hist);

    if (p->temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    cand *c = (cand *)malloc(sizeof(cand) * (size_t)V);
    int n;
    if (p->top_k > 0) n = select_topk(logits, V, p->top_k, c);
    else n = select_all(logits, V, c);
    if (n <= 0) { free(c); return 0; }

    n = apply_top_p(p, c, n);
    n = apply_min_p(p, c, n);

    const float inv_t = 1.0f / p->temp;
    float max_l = c[0].logit * inv_t;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        c[i].logit *= inv_t;
        if (i > 0 && c[i].logit > max_l) max_l = c[i].logit;
    }
    for (int i = 0; i < n; i++) sum += exp((double)(c[i].logit - max_l));
    double u = (double)(q35_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0) * sum;
    double acc = 0;
    int32_t chosen = c[n - 1].id;
    for (int i = 0; i < n; i++) {
        acc += exp((double)(c[i].logit - max_l));
        if (u < acc) { chosen = c[i].id; break; }
    }
    free(c);
    return chosen;
}

// ---------------------------------------------------------------------------
// Logprobs
// ---------------------------------------------------------------------------

typedef struct { float logit; int32_t id; } lp_cand;

static int lp_cmp(const void *a, const void *b) {
    const lp_cand *x = (const lp_cand *)a, *y = (const lp_cand *)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    return (x->id < y->id) ? -1 : (x->id > y->id);
}

void q35_compute_logprobs(const float *logits, int V, int32_t chosen,
                          int32_t n_top, q35_logprobs *out) {
    memset(out, 0, sizeof(*out));
    if (n_top > 20) n_top = 20;
    float max_l = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > max_l) max_l = logits[i];
    double sum = 0;
    for (int i = 0; i < V; i++) sum += exp((double)(logits[i] - max_l));
    double logZ = log(sum) + (double)max_l;
    out->chosen_logprob = (float)((double)logits[chosen] - logZ);

    if (n_top <= 0) return;
    // partial selection of the n_top largest raw logits
    lp_cand *c = (lp_cand *)malloc(sizeof(lp_cand) * (size_t)n_top);
    int n = 0;
    for (int i = 0; i < V; i++) {
        lp_cand x = { logits[i], i };
        if (n < n_top) {
            c[n++] = x;
            int j = n - 1;
            while (j > 0) {
                int pa = (j - 1) / 2;
                if (lp_cmp(&c[pa], &c[j]) <= 0) break;
                lp_cand t = c[pa]; c[pa] = c[j]; c[j] = t;
                j = pa;
            }
        } else if (lp_cmp(&x, &c[0]) < 0) {
            c[0] = x;
            for (int j = 0;;) {
                int l = 2 * j + 1, r = l + 1, small = j;
                if (l < n && lp_cmp(&c[l], &c[small]) > 0) small = l;
                if (r < n && lp_cmp(&c[r], &c[small]) > 0) small = r;
                if (small == j) break;
                lp_cand t = c[j]; c[j] = c[small]; c[small] = t;
                j = small;
            }
        }
    }
    qsort(c, (size_t)n, sizeof(lp_cand), lp_cmp);
    out->n_top = n;
    for (int i = 0; i < n; i++) {
        out->top[i].id = c[i].id;
        out->top[i].logprob = (float)((double)c[i].logit - logZ);
    }
    free(c);
}
