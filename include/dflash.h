// DFlash2 block drafter for Qwen3.6-35B-A3B (incoai/Qwen3.6-35B-A3B-DFlash2, GGUF converted with
// llama.cpp's convert_hf_to_gguf.py). Reference: llama.cpp src/models/dflash.cpp and
// common/speculative.cpp (common_speculative_impl_draft_dflash).
//
// The drafter only proposes tokens; the lossless verify decides what is emitted, so its numerics
// need not match any reference bit for bit (q8_1 activations, f16 KV).
//
// Context: for every token the target commits, the residual stream ENTERING target layers
// 1, 6, 11, 16, 22, 27, 32, 37 (8 x 2048 floats) is captured ("taps"), projected by fc + RMSNorm and
// injected as K/V into the drafter's sliding-window cache. A draft runs the block
// [anchor = next input token, mask x k] at positions pos..pos+k through 6 non-causal layers and the
// target's own output head, then walks the top-16 selector lattice.
#ifndef DFLASH_H
#define DFLASH_H

#include <stdint.h>
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DFLASH_MAX_TAPS 8
#define DFLASH_PEND_MAX 64

// Load and upload the drafter (call before the expert cache sizes itself from free VRAM).
int  dflash_load(const char *path, const model_t *target);
int  dflash_active(void);
void dflash_free(void);

// Target layer l is a tap: returns its tap index, or -1.
int  dflash_tap_index(uint32_t layer);
// Device row (2048 floats) that receives tap k of pending slot `slot` (slot = committed-but-not-
// injected token index, 0..DFLASH_PEND_MAX-1).
float *dflash_tap_dst(int slot, int k);
int  dflash_pending(void);
// The token whose taps were just written to slot dflash_pending() + j is committed at position pos.
void dflash_commit(int n, int pos0);
// Verify columns: their taps land in fixed stage rows (captured graphs); an accepted prefix of
// `keep` columns starting at position pos0 is committed after the pass.
float *dflash_stage_dst(int c, int k);
void dflash_commit_stage(int keep, int pos0);
// Forget everything (new sequence).
void dflash_reset(void);

// Draft up to k tokens following `anchor` (the next input token, at position `pos`). Injects the
// pending context first. Returns the number of drafts written to out.
int  dflash_draft(int32_t anchor, int pos, int k, int32_t *out);

void dflash_stats(double *ms_total, unsigned long long *n_drafts);

#ifdef __cplusplus
}
#endif
#endif
