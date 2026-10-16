#ifndef LFM25_EXPERT_CACHE_H
#define LFM25_EXPERT_CACHE_H

#include "lfm25_model.h"
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// VRAM slot cache for MoE expert slabs (Task 3.1b). Experts stay in the host mmap; a slot holds one
// (layer, expert) gate + up + down slab in three parallel device pools sharing the same slot index.
typedef struct lfm25_expert_cache lfm25_expert_cache_t;

// True when the model must (or is forced to) use the cache: expert_count > 32 or LFM25_FORCE_EXPERT_CACHE=1.
int lfm25_expert_cache_wanted(const lfm25_model_t *model);

// Frees the process-wide pinned expert store (it is keyed by model pointer: call before the model is freed).
void lfm25_expert_store_release(void);

// Call after all other device allocations (trunk, KV, scratch). Returns NULL on failure (message printed).
lfm25_expert_cache_t* lfm25_expert_cache_create(lfm25_model_t *model);
// Prints stats (hit rate etc.) to stderr, releases pools and host registrations.
void lfm25_expert_cache_free(lfm25_expert_cache_t *c);

// Resolve the n_ids expert ids of `layer` to slot indices, uploading missing slabs host->device asynchronously on the
// cache's own copy stream. `consumer` (the stream that will run the MoE FFN) is made to wait for the uploads of every
// returned slot (cudaStreamWaitEvent), so the host does not block. All slots of this call are protected from eviction
// until the next lookup. Not thread-safe.
void lfm25_expert_cache_lookup(lfm25_expert_cache_t *c, int layer, const int *ids, int n_ids, int *slots_out, cudaStream_t consumer);

// Hybrid CPU-miss mode. Like lfm25_expert_cache_lookup, but a miss gets NO slot and NO upload: slots_out[i] = -1 and bit i
// of the returned mask is set; the caller evaluates those experts on the CPU from lfm25_expert_cache_host_slab. Hits
// behave exactly as in lookup (protected, consumer waits for their pending uploads).
unsigned lfm25_expert_cache_lookup_cpu(lfm25_expert_cache_t *c, int layer, const int *ids, int n_ids, int *slots_out, cudaStream_t consumer);
// Host address of expert `expert`'s slab of tensor 0/1/2 (gate/up/down) in `layer` (pinned store or the GGUF mmap).
const void* lfm25_expert_cache_host_slab(const lfm25_expert_cache_t *c, int layer, int tensor, int expert);
int lfm25_expert_cache_host_pinned(const lfm25_expert_cache_t *c);
// After this layer's FFN is enqueued: asynchronously upload the CPU-evaluated experts so later tokens hit. At most
// `max_inflight` admission batches may be incomplete (0 disables admission). Call after lookup_cpu of the same layer.
void lfm25_expert_cache_admit(lfm25_expert_cache_t *c, int layer, const int *experts, int n, int max_inflight);

// Warm the cache for the PREDICTED experts of `layer` (the next MoE layer). Call right after the lookup of the
// previous layer, once that layer's FFN is enqueued: its slots are protected, everything else may be evicted. Purely
// a cache-warming hint; the real lookup of `layer` stays authoritative and nothing numerical depends on it.
void lfm25_expert_cache_prefetch(lfm25_expert_cache_t *c, int layer, const int *ids, int n_ids);

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
