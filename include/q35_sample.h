#ifndef Q35_SAMPLE_H
#define Q35_SAMPLE_H

// Host-side sampler and logprob computation for backend B.
//
// The GPU decode_batch returns the engine argmax and, when asked, the full fp32 logits
// row.  This module applies llama.cpp's sampler chain order on the host:
//
//   penalties -> top_k -> top_p -> min_p -> temperature -> dist/greedy
//
// (DRY, XTC, typical-p, top-n-sigma and mirostat are not implemented; requests using
// them are ignored with X-Tuf-Ignored.)  Logprobs are computed from the raw logits at
// temperature 1, exactly like llama-server's default post_sampling_probs=false path.

#include <stdint.h>
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    temp;              // <= 0 = greedy
    int32_t  top_k;             // 0 = disabled
    float    top_p;             // >= 1 = disabled
    float    min_p;             // <= 0 = disabled
    float    repeat_penalty;    // 1.0 = disabled
    float    freq_penalty;      // 0 = disabled
    float    presence_penalty;  // 0 = disabled
    int32_t  penalty_last_n;    // 0 = disabled
    uint64_t seed;              // 0 = time-based
    int      logprobs;          // request asked for logprobs
    int32_t  top_logprobs;      // n <= 20
} q35_sampling;

typedef struct {
    int32_t id;
    float   logprob;
} q35_lp_entry;

typedef struct {
    float        chosen_logprob;
    q35_lp_entry top[20];
    int          n_top;
} q35_logprobs;

// llama-server/common defaults (common_params_sampling): temp .8, top_k 40, top_p .95,
// min_p .05, penalties disabled, penalty_last_n 64.
void q35_sampling_defaults(q35_sampling *p);

// True when the sampler cannot use the engine's argmax alone.
int q35_sampling_needs_logits(const q35_sampling *p);

// Apply the full chain to `logits` (modified in place for penalties).  `hist` is the
// sequence so far (prompt + generated), oldest first.  `rng` must be non-NULL.
int32_t q35_sampling_sample(float *logits, int V, const q35_sampling *p,
                            const int32_t *hist, int n_hist, uint64_t *rng);

// Penalties + argmax (temperature 0 fast path when logits are already on the host).
int32_t q35_sampling_greedy(float *logits, int V, const q35_sampling *p,
                            const int32_t *hist, int n_hist);

// Raw-logit log-softmax at temperature 1 over the full vocabulary.
void q35_compute_logprobs(const float *logits, int V, int32_t chosen,
                          int32_t n_top, q35_logprobs *out);

// Deterministic per-request RNG seeding.
uint64_t q35_rng_seed(uint64_t user_seed, int slot, uint64_t counter);
uint64_t q35_rng_next(uint64_t *state);

#ifdef __cplusplus
}
#endif

#endif
