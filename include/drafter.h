#ifndef DRAFTER_H
#define DRAFTER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct drafter_s drafter_t;

// Initialize external drafter with GGUF model path and thread count
drafter_t* drafter_init(const char *model_path, int n_threads, int n_gpu_layers);

// Warmup / prime drafter with prompt tokens
void drafter_prefill(drafter_t *d, const int32_t *prompt_tokens, int n_prompt);

// Asynchronously request draft prediction following the accepted token
void drafter_submit_accepted(drafter_t *d, int32_t accepted_token, int pos);

// Poll / wait for draft token result
int32_t drafter_poll_draft(drafter_t *d);

// Rollback drafter state after verification rejection
void drafter_rollback(drafter_t *d, int valid_pos);

// Clean up drafter resources
void drafter_free(drafter_t *d);

#ifdef __cplusplus
}
#endif

#endif // DRAFTER_H
