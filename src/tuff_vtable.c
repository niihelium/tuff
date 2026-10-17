#include "tuff_vtable.h"
#include "model.h"
#include "gpu.h"
#include "engine.h"
#include "lfm25.h"
#include <string.h>

// -----------------------------------------------------------------------------
// Qwen3.6 / Qwen3.5 VTable Adapters
// -----------------------------------------------------------------------------

static void* vtable_qwen_model_load(const char *path) {
    return model_load(path);
}

static void vtable_qwen_model_free(void *model) {
    model_free((model_t*)model);
}

static void vtable_qwen_model_print_inspection(const void *model) {
    model_print_inspection((const model_t*)model);
}

static gguf_context_t* vtable_qwen_model_get_gguf(void *model) {
    return ((model_t*)model)->gguf;
}

static int32_t vtable_qwen_model_get_bos(void *model) {
    (void)model;
    return 151644; // Qwen <|im_start|>
}

static int32_t vtable_qwen_model_get_eos(void *model) {
    (void)model;
    return 151645; // Qwen <|im_end|>
}

static uint32_t vtable_qwen_model_get_vocab_size(void *model) {
    return ((model_t*)model)->vocab_size;
}

static int vtable_qwen_gpu_prepare(void *model, int *max_context_len, int fill_ram) {
    gpu_prepare_ex((model_t*)model, max_context_len, fill_ram);
    return 0;
}

static void vtable_qwen_gpu_shutdown(void *model) {
    gpu_shutdown((model_t*)model);
}

static void* vtable_qwen_engine_init(void *model, int max_context_len) {
    return engine_init((model_t*)model, max_context_len);
}

static void vtable_qwen_engine_free(void *engine) {
    engine_free((engine_t*)engine);
}

static void vtable_qwen_engine_set_sampler(void *engine, const sampler_params_t *params) {
    engine_set_sampler((engine_t*)engine, params);
}

static int32_t vtable_qwen_engine_forward_token(void *engine, int32_t token, int pos) {
    return engine_forward_token((engine_t*)engine, token, pos);
}

static int vtable_qwen_engine_forward_batch(void *engine, const int32_t *tokens, int n, int pos0,
                                            int32_t *argmax_out, float *logits_out) {
    return engine_forward_batch((engine_t*)engine, tokens, n, pos0, argmax_out, logits_out);
}

static int vtable_qwen_engine_rollback(void *engine, int n_keep) {
    return engine_rollback((engine_t*)engine, n_keep);
}

static const engine_vtable_t qwen_engine_vtable = {
    .arch                   = TUFF_ARCH_QWEN35MOE,
    .name                   = "qwen35moe",
    .model_load             = vtable_qwen_model_load,
    .model_free             = vtable_qwen_model_free,
    .model_print_inspection = vtable_qwen_model_print_inspection,
    .model_get_gguf         = vtable_qwen_model_get_gguf,
    .model_get_bos          = vtable_qwen_model_get_bos,
    .model_get_eos          = vtable_qwen_model_get_eos,
    .model_get_vocab_size   = vtable_qwen_model_get_vocab_size,
    .gpu_prepare            = vtable_qwen_gpu_prepare,
    .gpu_shutdown           = vtable_qwen_gpu_shutdown,
    .engine_init            = vtable_qwen_engine_init,
    .engine_free            = vtable_qwen_engine_free,
    .engine_set_sampler     = vtable_qwen_engine_set_sampler,
    .engine_forward_token   = vtable_qwen_engine_forward_token,
    .engine_forward_batch   = vtable_qwen_engine_forward_batch,
    .engine_rollback        = vtable_qwen_engine_rollback,
};

const engine_vtable_t* tuff_get_vtable(tuff_arch_t arch) {
    switch (arch) {
        case TUFF_ARCH_QWEN35MOE:
            return &qwen_engine_vtable;
        case TUFF_ARCH_LFM2:
        case TUFF_ARCH_LFM2MOE:
            return &lfm25_engine_vtable;
        default:
            return NULL;
    }
}
