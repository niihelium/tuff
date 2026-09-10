#include "sampler.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

sampler_params_t sampler_params_default(void) {
    sampler_params_t p;
    p.temp = 0.0f;           // Greedy by default
    p.top_p = 1.0f;          // Disabled
    p.top_k = 0;             // Disabled
    p.min_p = 0.0f;          // Disabled
    p.repeat_penalty = 1.0f; // Disabled
    p.repeat_last_n = 64;
    p.seed = 0;              // Auto
    return p;
}

static inline uint64_t xorshift128p(uint64_t s[2]) {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;
    s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return s[1] + s0;
}

static inline float rand_uniform(uint64_t s[2]) {
    return (float)(xorshift128p(s) >> 11) * (1.0f / 9007199254740992.0f);
}

static void init_rng(sampler_t *s, uint64_t seed) {
    if (seed == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        seed = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    s->rng_state[0] = seed ^ 0x8a5cd789635d2dffULL;
    s->rng_state[1] = (seed + 0x9e3779b97f4a7c15ULL) ^ 0x123456789abcdef0ULL;
    // Warm up
    for (int i = 0; i < 16; ++i) {
        xorshift128p(s->rng_state);
    }
}

void sampler_init(sampler_t *s, const sampler_params_t *params, int32_t vocab_size) {
    memset(s, 0, sizeof(*s));
    if (params) {
        s->params = *params;
    } else {
        s->params = sampler_params_default();
    }
    s->vocab_size = vocab_size;
    s->recent_cap = (s->params.repeat_last_n > 0) ? s->params.repeat_last_n : 64;
    if (s->recent_cap < 16) s->recent_cap = 16;
    s->recent_tokens = (int32_t*)calloc((size_t)s->recent_cap, sizeof(int32_t));
    s->candidates = (token_candidate_t*)malloc((size_t)vocab_size * sizeof(token_candidate_t));
    init_rng(s, s->params.seed);
}

void sampler_free(sampler_t *s) {
    if (s->recent_tokens) {
        free(s->recent_tokens);
        s->recent_tokens = NULL;
    }
    if (s->candidates) {
        free(s->candidates);
        s->candidates = NULL;
    }
}

void sampler_reset(sampler_t *s) {
    s->recent_head = 0;
    s->recent_count = 0;
    if (s->recent_tokens) {
        memset(s->recent_tokens, 0, (size_t)s->recent_cap * sizeof(int32_t));
    }
    init_rng(s, s->params.seed);
}

void sampler_accept_token(sampler_t *s, int32_t token_id) {
    if (!s->recent_tokens || s->recent_cap <= 0) return;
    s->recent_tokens[s->recent_head] = token_id;
    s->recent_head = (s->recent_head + 1) % s->recent_cap;
    if (s->recent_count < s->recent_cap) {
        s->recent_count++;
    }
}

static int compare_candidates_desc(const void *a, const void *b) {
    const token_candidate_t *ca = (const token_candidate_t*)a;
    const token_candidate_t *cb = (const token_candidate_t*)b;
    if (cb->logit > ca->logit) return 1;
    if (cb->logit < ca->logit) return -1;
    return 0;
}

int32_t sampler_sample(sampler_t *s, const float *logits, int32_t vocab_size) {
    if (!logits || vocab_size <= 0) return 0;

    // 1. Pure Greedy Fast-Path
    if (s->params.temp <= 0.0f) {
        float max_l = logits[0];
        int32_t best_id = 0;
        for (int32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = i;
            }
        }
        sampler_accept_token(s, best_id);
        return best_id;
    }

    // 2. Stochastic Sampling Path
    const float temp = s->params.temp;
    const float inv_temp = 1.0f / temp;
    const float repeat_penalty = s->params.repeat_penalty;
    const float min_p = s->params.min_p;
    const float top_p = s->params.top_p;
    const int32_t top_k = s->params.top_k;

    // A. Apply Repetition Penalty & Find Maximum Logit
    // We copy logits to working array
    token_candidate_t *cands = s->candidates;
    int32_t n_cands = 0;

    float max_logit = -1e30f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cands[i].id = i;
        cands[i].logit = logits[i];
    }

    if (repeat_penalty > 1.0f && s->recent_count > 0) {
        for (int32_t k = 0; k < s->recent_count; ++k) {
            int32_t past_tok = s->recent_tokens[k];
            if (past_tok >= 0 && past_tok < vocab_size) {
                float l = cands[past_tok].logit;
                cands[past_tok].logit = (l > 0.0f) ? (l / repeat_penalty) : (l * repeat_penalty);
            }
        }
    }

    for (int32_t i = 0; i < vocab_size; ++i) {
        if (cands[i].logit > max_logit) {
            max_logit = cands[i].logit;
        }
    }

    // B. Min-P Cutoff Filter (Pre-filtering)
    // For Min-P: P_i >= P_max * min_p  <=>  (L_i - L_max)/T >= ln(min_p)
    float min_scaled_logit = -1e30f;
    if (min_p > 0.0f && min_p < 1.0f) {
        min_scaled_logit = logf(min_p);
    }

    n_cands = 0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        float scaled = (cands[i].logit - max_logit) * inv_temp;
        if (scaled >= min_scaled_logit) {
            cands[n_cands].id = cands[i].id;
            cands[n_cands].logit = scaled; // store scaled logit
            n_cands++;
        }
    }

    if (n_cands == 0) {
        // Fallback: at least max candidate
        cands[0].id = 0;
        cands[0].logit = 0.0f;
        n_cands = 1;
    }

    // C. Sort candidates descending
    qsort(cands, (size_t)n_cands, sizeof(token_candidate_t), compare_candidates_desc);

    // D. Top-K Filter
    if (top_k > 0 && n_cands > top_k) {
        n_cands = top_k;
    }

    // E. Softmax Probabilities over remaining candidates
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < n_cands; ++i) {
        float ex = expf(cands[i].logit);
        cands[i].logit = ex; // reuse field for probability weight
        sum_exp += ex;
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
    for (int32_t i = 0; i < n_cands; ++i) {
        cands[i].logit *= inv_sum;
    }

    // F. Top-P (Nucleus) Truncation
    int32_t active_cands = n_cands;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum_p = 0.0f;
        for (int32_t i = 0; i < n_cands; ++i) {
            cum_p += cands[i].logit;
            if (cum_p >= top_p) {
                active_cands = i + 1;
                break;
            }
        }
        // Renormalize active candidates
        float sub_sum = 0.0f;
        for (int32_t i = 0; i < active_cands; ++i) sub_sum += cands[i].logit;
        if (sub_sum > 0.0f) {
            float inv_sub = 1.0f / sub_sum;
            for (int32_t i = 0; i < active_cands; ++i) cands[i].logit *= inv_sub;
        }
    }

    // G. PRNG Sample
    float r = rand_uniform(s->rng_state);
    float cum = 0.0f;
    int32_t chosen_id = cands[0].id;
    for (int32_t i = 0; i < active_cands; ++i) {
        cum += cands[i].logit;
        if (r <= cum || i == active_cands - 1) {
            chosen_id = cands[i].id;
            break;
        }
    }

    sampler_accept_token(s, chosen_id);
    return chosen_id;
}
