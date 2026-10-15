#ifndef LFM25_EXPERT_CACHE_H
#define LFM25_EXPERT_CACHE_H

#include "lfm25_model.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// VRAM slot cache for MoE expert slabs (Task 3.1b). Experts stay in the host mmap; a slot holds one
// (layer, expert) gate + up + down slab in three parallel device pools sharing the same slot index.
typedef struct lfm25_expert_cache lfm25_expert_cache_t;

// True when the model must (or is forced to) use the cache: expert_count > 32 or LFM25_FORCE_EXPERT_CACHE=1.
int lfm25_expert_cache_wanted(const lfm25_model_t *model);

// Call after all other device allocations (trunk, KV, scratch). Returns NULL on failure (message printed).
lfm25_expert_cache_t* lfm25_expert_cache_create(lfm25_model_t *model);
// Prints stats (hit rate etc.) to stderr, releases pools and host registrations.
void lfm25_expert_cache_free(lfm25_expert_cache_t *c);

// Resolve the n_ids expert ids of `layer` to slot indices, uploading missing slabs host->device
// (asynchronously on the default stream). All hits of this call are protected from eviction.
void lfm25_expert_cache_lookup(lfm25_expert_cache_t *c, int layer, const int *ids, int n_ids, int *slots_out);

void* lfm25_expert_cache_pool_gate(const lfm25_expert_cache_t *c);
void* lfm25_expert_cache_pool_up(const lfm25_expert_cache_t *c);
void* lfm25_expert_cache_pool_down(const lfm25_expert_cache_t *c);
size_t lfm25_expert_cache_gate_slot_bytes(const lfm25_expert_cache_t *c);
size_t lfm25_expert_cache_up_slot_bytes(const lfm25_expert_cache_t *c);
size_t lfm25_expert_cache_down_slot_bytes(const lfm25_expert_cache_t *c);

#ifdef __cplusplus
}
#endif

#endif
