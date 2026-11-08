#include "tuff_vtable.h"
#include "model.h"
#include "gpu.h"
#include "engine.h"
#include "lfm25.h"
#include "qwen35.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------
// Qwen3.5 dense (qwen35) VTable Adapters
// -----------------------------------------------------------------------------
// main.c owns the logit-capture sink and is not linked into tuff-server, so resolve it
// weakly: the qwen35 engine works in both binaries, and --dump-logits only in tuff.
extern int  tuff_logit_capture_is_enabled(void) __attribute__((weak));
extern void tuff_logit_capture(const float *logits, uint32_t n) __attribute__((weak));

static void* q35_vt_load(const char *path) { return q35_model_load(path); }
static void q35_vt_free(void *m) { q35_model_free((q35_model*)m); }
static void q35_vt_inspect(const void *m) {
    fprintf(stderr, "[qwen35] dense hybrid DeltaNet + gated attention: vocab=%d\n",
            q35_model_vocab((const q35_model*)m));
}
static gguf_context_t* q35_vt_gguf(void *m) { return q35_model_gguf((q35_model*)m); }
static int32_t q35_vt_bos(void *m) { (void)m; return -1; }
static int32_t q35_vt_eos(void *m) { return q35_model_eos((const q35_model*)m); }
static uint32_t q35_vt_vocab(void *m) { return (uint32_t)q35_model_vocab((const q35_model*)m); }
static int q35_vt_prepare(void *m, int *ctx, int fill_ram) { (void)fill_ram; return q35_prepare((q35_model*)m, ctx); }
static void q35_vt_shutdown(void *m) { (void)m; }
static void* q35_vt_engine_init(void *m, int ctx) { return q35_engine_init((q35_model*)m, ctx, 1); }
static void q35_vt_engine_free(void *e) { q35_engine_free((q35_engine*)e); }
static void q35_vt_set_sampler(void *e, const sampler_params_t *p) { (void)e; (void)p; }
static int32_t q35_vt_forward(void *e, int32_t token, int pos) {
    static float *logits = NULL;
    int32_t am = -1;
    if (tuff_logit_capture_is_enabled && tuff_logit_capture_is_enabled()) {
        if (!logits) logits = (float*)malloc(sizeof(float) * Q35_VOCAB);
        if (q35_engine_step((q35_engine*)e, 0, token, pos, &am, logits) != 0) return -1;
        if (tuff_logit_capture) tuff_logit_capture(logits, Q35_VOCAB);
    } else if (q35_engine_step((q35_engine*)e, 0, token, pos, &am, NULL) != 0) {
        return -1;
    }
    return am;
}
static int q35_vt_forward_batch(void *e, const int32_t *tokens, int n, int pos0,
                                int32_t *argmax_out, float *logits_out) {
    return q35_engine_steps((q35_engine*)e, 0, tokens, n, pos0, argmax_out, logits_out);
}

static const engine_vtable_t q35_engine_vtable = {
    .arch                   = TUFF_ARCH_QWEN35,
    .name                   = "qwen35",
    .model_load             = q35_vt_load,
    .model_free             = q35_vt_free,
    .model_print_inspection = q35_vt_inspect,
    .model_get_gguf         = q35_vt_gguf,
    .model_get_bos          = q35_vt_bos,
    .model_get_eos          = q35_vt_eos,
    .model_get_vocab_size   = q35_vt_vocab,
    .gpu_prepare            = q35_vt_prepare,
    .gpu_shutdown           = q35_vt_shutdown,
    .engine_init            = q35_vt_engine_init,
    .engine_free            = q35_vt_engine_free,
    .engine_set_sampler     = q35_vt_set_sampler,
    .engine_forward_token   = q35_vt_forward,
    .engine_forward_batch   = q35_vt_forward_batch,
    .engine_rollback        = NULL,
};

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
        case TUFF_ARCH_QWEN35:
            return &q35_engine_vtable;
        case TUFF_ARCH_QWEN35MOE:
            return &qwen_engine_vtable;
        case TUFF_ARCH_LFM2:
        case TUFF_ARCH_LFM2MOE:
            return &lfm25_engine_vtable;
        default:
            return NULL;
    }
}
