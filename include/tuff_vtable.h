#ifndef TUFF_VTABLE_H
#define TUFF_VTABLE_H

#include <stdint.h>
#include <stdbool.h>
#include "sampler.h"
#include "gguf_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TUFF_ARCH_UNKNOWN = 0,
    TUFF_ARCH_QWEN35MOE,   // Qwen3.6 / Qwen3.5 hybrid DeltaNet + MoE
    TUFF_ARCH_LFM2,        // Liquid AI LFM2 / LFM2.5 Dense (ShortConv + GQA)
    TUFF_ARCH_LFM2MOE      // Liquid AI LFM2.5 MoE (ShortConv + GQA + Sparse MoE)
} tuff_arch_t;

// Architecture detection function (reads GGUF general.architecture once)
tuff_arch_t tuff_detect_arch(const char *gguf_filepath);
const char* tuff_arch_to_string(tuff_arch_t arch);

// Unified Engine Virtual Method Table
typedef struct engine_vtable {
    tuff_arch_t arch;
    const char *name;

    // Model lifecycle
    void*           (*model_load)(const char *path);
    void            (*model_free)(void *model);
    void            (*model_print_inspection)(const void *model);
    gguf_context_t* (*model_get_gguf)(void *model);
    int32_t         (*model_get_bos)(void *model);
    int32_t         (*model_get_eos)(void *model);
    uint32_t        (*model_get_vocab_size)(void *model);

    // GPU allocation and weight preparation
    int             (*gpu_prepare)(void *model, int *max_context_len, int fill_ram);
    void            (*gpu_shutdown)(void *model);

    // Engine lifecycle
    void*           (*engine_init)(void *model, int max_context_len);
    void            (*engine_free)(void *engine);
    void            (*engine_set_sampler)(void *engine, const sampler_params_t *params);

    // Single-token greedy / sampled decode step (the inner loop call)
    int32_t         (*engine_forward_token)(void *engine, int32_t token, int pos);
} engine_vtable_t;

// VTable registry getters
const engine_vtable_t* tuff_get_vtable(tuff_arch_t arch);

#ifdef __cplusplus
}
#endif

#endif // TUFF_VTABLE_H
