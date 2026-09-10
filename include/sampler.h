#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    temp;            // <= 0.0f = greedy argmax, > 0.0f = stochastic
    float    top_p;           // nucleus probability (0.0 < top_p <= 1.0, 1.0 = disabled)
    int32_t  top_k;           // top-k threshold (0 = disabled)
    float    min_p;           // min-p cutoff (0.0 = disabled, e.g. 0.05)
    float    repeat_penalty;  // 1.0 = disabled, > 1.0 = penalize repetition
    int32_t  repeat_last_n;   // window of past tokens for repeat penalty (default 64)
    uint64_t seed;            // RNG seed (0 = auto clock seed)
} sampler_params_t;

typedef struct {
    int32_t id;
    float   logit;
} token_candidate_t;

typedef struct {
    sampler_params_t params;
    uint64_t         rng_state[2];       // XorShift128+ state
    int32_t         *recent_tokens;      // Ring buffer of past tokens
    int32_t          recent_cap;
    int32_t          recent_head;
    int32_t          recent_count;
    token_candidate_t *candidates;       // Working memory for Top-K / Top-P
    int32_t          vocab_size;
} sampler_t;

sampler_params_t sampler_params_default(void);

void sampler_init(sampler_t *s, const sampler_params_t *params, int32_t vocab_size);
void sampler_free(sampler_t *s);
void sampler_reset(sampler_t *s);
void sampler_accept_token(sampler_t *s, int32_t token_id);

// Samples a token from raw logits.
// If s->params.temp <= 0.0f, performs pure greedy argmax.
int32_t sampler_sample(sampler_t *s, const float *logits, int32_t vocab_size);

#ifdef __cplusplus
}
#endif

#endif // SAMPLER_H
