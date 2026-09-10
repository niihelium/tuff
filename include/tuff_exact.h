#ifndef TUFF_EXACT_H
#define TUFF_EXACT_H

// P0 (parity.md sec.3/sec.4): the correctness contract moves from "bit-identical to
// Tuff's own scalar CPU path" to a tolerance gate against llama.cpp. The scalar path is
// retained so any divergence can be bisected against it.
//
//   TUFF_EXACT unset / 0  -> fast path allowed (default)
//   TUFF_EXACT=1          -> force the bit-exact reference kernels
//
// The environment is read exactly ONCE (cached in a static), so this is safe to call from
// inside per-layer / per-row hot loops; getenv() is a linear scan of environ and must never
// appear in one. Callable from C and from CUDA/C++.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 => use the bit-exact reference kernels, 0 => fast paths permitted.
int tuff_exact(void);

// ---------------------------------------------------------------------------
// Final-logit capture sink, used by main.c's --dump-logits mode.
//
// The engine calls tuff_logit_capture() once per forward pass with the final logit
// vector, immediately after the output projection and before the vector is freed. It is a
// no-op (one predicate) unless --dump-logits armed it, so it costs nothing in normal runs.
void tuff_logit_capture(const float *logits, uint32_t n);

// Arm/disarm the sink. Returns 0 on success, -1 on allocation failure.
int  tuff_logit_capture_enable(int on);
int  tuff_logit_capture_is_enabled(void);
// Most recently captured vector, or NULL if nothing has been captured since the last
// tuff_logit_capture_take(). *n_out receives the vector length.
const float* tuff_logit_capture_take(uint32_t *n_out);

#ifdef __cplusplus
}
#endif

#endif // TUFF_EXACT_H
