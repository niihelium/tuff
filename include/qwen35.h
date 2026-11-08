#ifndef QWEN35_H
#define QWEN35_H

// Qwen3.5 0.8B / 2B dense (qwen35) engine: GGUF load, Q8_0 repack, persistent decode.
#include <stdint.h>
#include "gguf_parser.h"
#include "qwen35_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q35_model q35_model;
typedef struct q35_engine q35_engine;

int         q35_model_is_arch(const char *path);
q35_model *q35_model_load(const char *path);
void       q35_model_free(q35_model *m);
gguf_context_t *q35_model_gguf(q35_model *m);
int        q35_model_vocab(const q35_model *m);
int        q35_model_bos(const q35_model *m);
int        q35_model_eos(const q35_model *m);

// Repack/upload all weights.  *max_context_len is left unchanged (the caller's -c wins).
int q35_prepare(q35_model *m, int *max_context_len);

// Create an engine with n_slots sequences, each with `ctx` positions of KV.
q35_engine *q35_engine_init(q35_model *m, int ctx, int n_slots);
void        q35_engine_free(q35_engine *e);
void        q35_engine_reset(q35_engine *e, int slot);

// One decode step for `slot`: feed `token` at `pos`, return argmax and optionally the
// full fp32 logit vector (host memory, Q35_VOCAB floats).
int q35_engine_step(q35_engine *e, int slot, int32_t token, int pos, int32_t *argmax, float *logits_host);

// Feed n tokens in one cooperative launch; logits (optional) are [n][VOCAB].
int q35_engine_steps(q35_engine *e, int slot, const int32_t *tokens, int n, int pos0,
                     int32_t *argmax, float *logits_host);

// Batched decode: n independent slots, one token each per launch.  `slots[i]`,
// `positions[i]`, `tokens[i]` describe entry i; each produces one argmax and (optionally)
// one full fp32 logit row.  n <= 32; the host picks the template tier.
int q35_engine_decode_batch(q35_engine *e, const int *slots, const int *positions,
                            const int32_t *tokens, int n, int32_t *argmax, float *logits_host);

#ifdef __cplusplus
}
#endif

#endif
