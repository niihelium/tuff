#include "moe_policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct moe_policy {
    int n_layers;
    int n_experts;
    float decay;
    float alpha; // blend weight for lookahead vs context frequency

    // Context statistics per expert: n[layer * n_experts + expert]
    float *ctx_counts;
    float *layer_totals;

    // Lookahead probabilities per layer: lh[layer * n_experts + expert]
    float *lh_probs;
    uint8_t *lh_valid; // [n_layers] 1 if lookahead populated for current step

    // Logical token step. The decay is applied LAZILY per layer: layer_step[l] is the step the
    // layer's counters have been decayed up to, and every access first multiplies the row by
    // `decay` once per missed step. Per element that is exactly the sequence of multiplies the
    // eager per-step loop performed, so the counters are bit-identical, but a layer's state now
    // depends only on that layer's own observations and the step number. That independence is
    // what lets a two-column verify pass process token t and t+1 layer by layer and still see,
    // at every layer, the counters a token-by-token decode would have seen.
    int step_count;
    int *layer_step;   // [n_layers]
};

static void policy_sync_layer(moe_policy_t *p, int l) {
    if (p->decay > 0.0f && p->decay < 1.0f) {
        float *row = p->ctx_counts + (size_t)l * p->n_experts;
        while (p->layer_step[l] < p->step_count) {
            for (int e = 0; e < p->n_experts; ++e) row[e] *= p->decay;
            p->layer_totals[l] *= p->decay;
            p->layer_step[l]++;
        }
    }
    p->layer_step[l] = p->step_count;
}

moe_policy_t* moe_policy_create(int n_layers, int n_experts) {
    if (n_layers <= 0 || n_experts <= 0) return NULL;

    moe_policy_t *p = (moe_policy_t*)calloc(1, sizeof(moe_policy_t));
    if (!p) return NULL;

    p->n_layers = n_layers;
    p->n_experts = n_experts;

    const char *e_decay = getenv("TUFF_LES_DECAY");
    p->decay = (e_decay && *e_decay) ? (float)atof(e_decay) : 0.98f;

    const char *e_alpha = getenv("TUFF_LES_ALPHA");
    p->alpha = (e_alpha && *e_alpha) ? (float)atof(e_alpha) : 0.35f;

    size_t total_experts = (size_t)n_layers * n_experts;
    p->ctx_counts = (float*)calloc(total_experts, sizeof(float));
    p->layer_totals = (float*)calloc((size_t)n_layers, sizeof(float));
    p->lh_probs = (float*)calloc(total_experts, sizeof(float));
    p->lh_valid = (uint8_t*)calloc((size_t)n_layers, sizeof(uint8_t));
    p->layer_step = (int*)calloc((size_t)n_layers, sizeof(int));

    if (!p->ctx_counts || !p->layer_totals || !p->lh_probs || !p->lh_valid || !p->layer_step) {
        moe_policy_free(p);
        return NULL;
    }

    return p;
}

void moe_policy_free(moe_policy_t *pol) {
    if (!pol) return;
    free(pol->ctx_counts);
    free(pol->layer_totals);
    free(pol->lh_probs);
    free(pol->lh_valid);
    free(pol->layer_step);
    free(pol);
}

void moe_policy_reset(moe_policy_t *pol) {
    if (!pol) return;
    size_t total_experts = (size_t)pol->n_layers * pol->n_experts;
    memset(pol->ctx_counts, 0, total_experts * sizeof(float));
    memset(pol->layer_totals, 0, (size_t)pol->n_layers * sizeof(float));
    memset(pol->lh_probs, 0, total_experts * sizeof(float));
    memset(pol->lh_valid, 0, (size_t)pol->n_layers * sizeof(uint8_t));
    memset(pol->layer_step, 0, (size_t)pol->n_layers * sizeof(int));
    pol->step_count = 0;
}

void moe_policy_step(moe_policy_t *pol) {
    if (!pol) return;
    moe_policy_set_step(pol, pol->step_count + 1);
}

void moe_policy_set_step(moe_policy_t *pol, int step) {
    if (!pol) return;
    if (step != pol->step_count) {
        // Clear step-local lookahead flags
        memset(pol->lh_valid, 0, (size_t)pol->n_layers * sizeof(uint8_t));
    }
    pol->step_count = step;
}

int moe_policy_get_step(const moe_policy_t *pol) { return pol ? pol->step_count : 0; }

size_t moe_policy_row_bytes(const moe_policy_t *pol) {
    return pol ? (size_t)pol->n_experts * sizeof(float) + sizeof(float) + sizeof(int) : 0;
}

void moe_policy_row_save(const moe_policy_t *pol, int layer, void *buf) {
    if (!pol || layer < 0 || layer >= pol->n_layers) return;
    char *b = (char*)buf;
    memcpy(b, pol->ctx_counts + (size_t)layer * pol->n_experts, (size_t)pol->n_experts * sizeof(float));
    b += (size_t)pol->n_experts * sizeof(float);
    memcpy(b, &pol->layer_totals[layer], sizeof(float)); b += sizeof(float);
    memcpy(b, &pol->layer_step[layer], sizeof(int));
}

void moe_policy_row_restore(moe_policy_t *pol, int layer, const void *buf) {
    if (!pol || layer < 0 || layer >= pol->n_layers) return;
    const char *b = (const char*)buf;
    memcpy(pol->ctx_counts + (size_t)layer * pol->n_experts, b, (size_t)pol->n_experts * sizeof(float));
    b += (size_t)pol->n_experts * sizeof(float);
    memcpy(&pol->layer_totals[layer], b, sizeof(float)); b += sizeof(float);
    memcpy(&pol->layer_step[layer], b, sizeof(int));
}

void moe_policy_observe(moe_policy_t *pol, int layer, const int *ids, int n) {
    if (!pol || layer < 0 || layer >= pol->n_layers || !ids || n <= 0) return;

    policy_sync_layer(pol, layer);
    size_t base = (size_t)layer * pol->n_experts;
    for (int i = 0; i < n; ++i) {
        int e = ids[i];
        if (e >= 0 && e < pol->n_experts) {
            pol->ctx_counts[base + e] += 1.0f;
            pol->layer_totals[layer] += 1.0f;
        }
    }
}

void moe_policy_observe_lookahead(moe_policy_t *pol, int layer, const int *ids, const float *probs, int n) {
    if (!pol || layer < 0 || layer >= pol->n_layers || !ids || n <= 0) return;

    size_t base = (size_t)layer * pol->n_experts;
    memset(pol->lh_probs + base, 0, (size_t)pol->n_experts * sizeof(float));

    for (int i = 0; i < n; ++i) {
        int e = ids[i];
        if (e >= 0 && e < pol->n_experts) {
            pol->lh_probs[base + e] = probs ? probs[i] : (1.0f / (float)n);
        }
    }
    pol->lh_valid[layer] = 1;
}

float moe_policy_score(const moe_policy_t *pol, int layer, int expert) {
    if (!pol || layer < 0 || layer >= pol->n_layers || expert < 0 || expert >= pol->n_experts)
        return 0.0f;

    policy_sync_layer((moe_policy_t*)pol, layer);
    size_t idx = (size_t)layer * pol->n_experts + expert;
    float tot = pol->layer_totals[layer];
    float p_ctx = (tot > 1e-4f) ? (pol->ctx_counts[idx] / tot) : 0.0f;

    if (pol->lh_valid[layer]) {
        float p_lh = pol->lh_probs[idx];
        return (1.0f - pol->alpha) * p_ctx + pol->alpha * p_lh;
    }
    return p_ctx;
}

int moe_policy_select_victim(const moe_policy_t *pol, moe_policy_candidate_fn fn, void *ctx, int n_candidates) {
    if (!pol || !fn || n_candidates <= 0) return -1;

    int best_cand = -1;
    float lowest_score = 1e30f;

    for (int i = 0; i < n_candidates; ++i) {
        int l = -1, e = -1;
        if (!fn(ctx, i, &l, &e)) continue;

        float s = moe_policy_score(pol, l, e);
        if (s < lowest_score) {
            lowest_score = s;
            best_cand = i;
            if (lowest_score <= 0.0f) {
                // Cannot get lower than zero score
                break;
            }
        }
    }
    return best_cand;
}

int moe_policy_prefetch_plan(const moe_policy_t *pol, int budget, float min_score,
                             moe_policy_resident_fn is_res_fn, void *ctx,
                             int *out_layers, int *out_experts) {
    if (!pol || budget <= 0 || !out_layers || !out_experts || !is_res_fn) return 0;

    int planned = 0;
    // Greedily find top candidates across all layers that are non-resident
    for (int b = 0; b < budget; ++b) {
        float max_s = min_score;
        int best_l = -1, best_e = -1;

        for (int l = 0; l < pol->n_layers; ++l) {
            for (int e = 0; e < pol->n_experts; ++e) {
                // Skip if resident
                if (is_res_fn(ctx, l, e)) continue;

                // Check if already planned
                int already = 0;
                for (int p = 0; p < planned; ++p) {
                    if (out_layers[p] == l && out_experts[p] == e) {
                        already = 1;
                        break;
                    }
                }
                if (already) continue;

                float s = moe_policy_score(pol, l, e);
                if (s > max_s) {
                    max_s = s;
                    best_l = l;
                    best_e = e;
                }
            }
        }

        if (best_l >= 0 && best_e >= 0) {
            out_layers[planned] = best_l;
            out_experts[planned] = best_e;
            planned++;
        } else {
            break;
        }
    }

    return planned;
}
