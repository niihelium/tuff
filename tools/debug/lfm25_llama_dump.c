// lfm25_llama_dump: decode the BOS token once in llama.cpp (single-token batch, all layers on GPU) and dump
// EVERY graph tensor (F32/F16, contiguous) via cb_eval into <outdir>/<name>.{f32,f16}; the tensor list goes to stdout.
// Use it to localize a numeric divergence layer by layer against a tuff per-layer dump (see wiki/lfm25/devlog.md).
//
// build: gcc -O2 tools/debug/lfm25_llama_dump.c -I$LLAMA/include -I$LLAMA/ggml/include
//            -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o lfm25_llama_dump
// usage: lfm25_llama_dump <model.gguf> <outdir> [n_gpu_layers=99]
//
// Notes: names collide when llama.cpp reuses a label (e.g. "norm-N" for both the operator and FFN norm; the later
// one wins). llama.cpp labels the FFN-norm OUTPUT "model.layers.{}.ffn_out-N" (cb quirk in src/models/lfm2.cpp).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_dir;
static int g_on = 0;

static bool cb(struct ggml_tensor *t, bool ask, void *ud) {
    (void)ud;
    if (ask) return g_on;
    if (!g_on) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) return true;
    if (!ggml_is_contiguous(t)) return true;
    void *buf = malloc(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf, 0, ggml_nbytes(t));
    char p[512];
    snprintf(p, sizeof p, "%s/%s.%s", g_dir, t->name, t->type == GGML_TYPE_F16 ? "f16" : "f32");
    for (char *c = p + strlen(g_dir) + 1; *c; c++) if (*c == '/' || *c == ' ') *c = '_';
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(buf, 1, ggml_nbytes(t), f); fclose(f); }
    printf("%s %s %s [%lld %lld %lld]\n", t->name, ggml_op_desc(t), ggml_type_name(t->type),
           (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
    free(buf);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <outdir> [n_gpu_layers]\n", argv[0]); return 2; }
    g_dir = argv[2];
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 3 ? atoi(argv[3]) : 99;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.cb_eval = cb;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    llama_token t = llama_vocab_bos(llama_model_get_vocab(model));
    g_on = 1;
    if (llama_decode(ctx, llama_batch_get_one(&t, 1))) return 1;
    g_on = 0;
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
