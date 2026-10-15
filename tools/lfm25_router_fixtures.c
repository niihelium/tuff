// lfm25_router_fixtures: dump llama.cpp's MoE router tensors for tests/test_lfm25_router.cu.
//
// Greedy single-token decode of "Hello" on GPU (n_batch = n_ubatch = 1); for every step and every
// MoE layer writes the router input (ffn_norm output), logits, sigmoid probs, top-k ids and
// normalized weights, plus the expert-FFN activations, per-expert down outputs and combined MoE
// output, as raw files into <outdir>. tools/lfm25_pack_router_fixtures.py and
// tools/lfm25_pack_moe_fixtures.py pack them into tests/fixtures/lfm25_8b_{router,moe_ffn}_cases.bin.
//
// build: gcc -O2 tools/lfm25_router_fixtures.c -I$LLAMA/include -I$LLAMA/ggml/include
//            -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o lfm25_router_fixtures
// usage: lfm25_router_fixtures <8b-model.gguf> <outdir> <n_steps>     (fixtures use n_steps = 8)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_dir;
static int g_step = 0;

static int wanted(const char *n) {
    return !strncmp(n, "ffn_moe_logits-", 15) || !strncmp(n, "ffn_moe_probs-", 14) ||
           !strncmp(n, "ffn_moe_topk-", 13) || !strncmp(n, "ffn_moe_weights_norm-", 21) ||
           !strncmp(n, "model.layers.{}.ffn_out-", 24) ||
           // expert FFN (tests/fixtures/lfm25_8b_moe_ffn_cases.bin)
           !strncmp(n, "ffn_moe_swiglu-", 15) || !strncmp(n, "ffn_moe_down-", 13) || !strncmp(n, "ffn_moe_out-", 12);
}

static bool cb(struct ggml_tensor *t, bool ask, void *ud) {
    (void)ud;
    if (ask) return wanted(t->name);
    if (!wanted(t->name) || !ggml_is_contiguous(t)) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return true;
    void *buf = malloc(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf, 0, ggml_nbytes(t));
    char name[256]; snprintf(name, sizeof name, "%s", t->name);
    // "model.layers.{}.ffn_out-L" is the ffn_norm output (llama.cpp cb label quirk) = router input
    if (!strncmp(name, "model.layers.{}.ffn_out-", 24)) snprintf(name, sizeof name, "router_in-%s", t->name + 24);
    char p[512]; snprintf(p, sizeof p, "%s/%s-t%d.%s", g_dir, name, g_step, t->type == GGML_TYPE_I32 ? "i32" : "f32");
    FILE *f = fopen(p, "wb"); fwrite(buf, 1, ggml_nbytes(t), f); fclose(f);
    free(buf);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    g_dir = argv[2];
    int n = atoi(argv[3]);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.cb_eval = cb;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(vocab);
    llama_token toks[64];
    int np = llama_tokenize(vocab, "Hello", 5, toks, 64, true, false);
    llama_token cur = toks[0];
    for (int s = 0; s < n; s++) {
        g_step = s;
        cur = s < np ? toks[s] : cur;
        if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) return 1;
        const float *lg = llama_get_logits_ith(ctx, -1);
        int best = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        printf("step %d token %d -> %d\n", s, cur, best);
        cur = best;
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
