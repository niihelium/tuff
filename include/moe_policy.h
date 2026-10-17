#ifndef MOE_POLICY_H
#define MOE_POLICY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moe_policy moe_policy_t;

// Candidate callback for victim selection:
// Given candidate index i (0 <= i < n_candidates), returns 1 if valid and writes
// *out_layer and *out_expert, or 0 if invalid / protected.
typedef int (*moe_policy_candidate_fn)(void *ctx, int idx, int *out_layer, int *out_expert);

// Query if an expert is currently resident
typedef int (*moe_policy_resident_fn)(void *ctx, int layer, int expert);

// Create a model-agnostic MoE policy tracker
moe_policy_t* moe_policy_create(int n_layers, int n_experts);
void moe_policy_free(moe_policy_t *pol);

// Reset context counters for a new sequence / context
void moe_policy_reset(moe_policy_t *pol);

// Update decay: call at token boundaries
void moe_policy_step(moe_policy_t *pol);

// Observe authoritative routing decision for a layer
void moe_policy_observe(moe_policy_t *pol, int layer, const int *ids, int n);

// Observe lookahead prediction for a layer (e.g. from previous layer's normed state)
void moe_policy_observe_lookahead(moe_policy_t *pol, int layer, const int *ids, const float *probs, int n);

// Score of an expert in a layer (combining context statistics and lookahead)
float moe_policy_score(const moe_policy_t *pol, int layer, int expert);

// Pick victim with lowest score among candidates
// Returns candidate index idx, or -1 if none found
int moe_policy_select_victim(const moe_policy_t *pol, moe_policy_candidate_fn fn, void *ctx, int n_candidates);

// Generate prefetch plan of non-resident experts with highest scores beating threshold
// Returns number of planned prefetches (up to budget)
int moe_policy_prefetch_plan(const moe_policy_t *pol, int budget, float min_score,
                             moe_policy_resident_fn is_res_fn, void *ctx,
                             int *out_layers, int *out_experts);

#ifdef __cplusplus
}
#endif

#endif // MOE_POLICY_H
