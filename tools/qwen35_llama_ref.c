// qwen35-llama-ref: llama.cpp oracle for the tuf qwen35 kernel.
//
// Feeds a token-id list one token at a time (n_batch = n_ubatch = 1, all layers on GPU),
// exactly like lfm25_llama_ref.c, and dumps the logit vector produced *after* each fed
// token.  This is the same recipe llama-server uses for a decode loop, and it makes the
// per-position comparison with tools/qwen35_ref.py and the megakernel exact.
//
// Outputs:
//   --logits-out  raw float32 [n_pos, vocab]
//   --margins-out "<pos> <top1> <top1-top2>" per position
//   --tokens-out  greedy token id per position (argmax of the matching logit row)
//
// Build: linked against the same libllama as tuff (LLAMA_CPP_DIR).
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr, "usage: qwen35-llama-ref --model <gguf> --tokens <ids.txt> --logits-out <f> "
                    "[--margins-out <f>] [--tokens-out <f>] [--ngl <n>]\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *tokens_path = NULL, *logits_out = NULL;
    const char *margins_out = NULL, *tokens_out = NULL;
    int ngl = 99, fa = -1, kv32 = 0, gen = 0;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { usage(); return 2; }
        if      (!strcmp(a, "--kv-f32"))      { kv32 = 1; continue; }
        else if (!strcmp(a, "--generate"))    { gen = atoi(v); ++i; continue; }
        else if (!strcmp(a, "--model"))       model_path = v;
        else if (!strcmp(a, "--tokens"))      tokens_path = v;
        else if (!strcmp(a, "--logits-out"))  logits_out = v;
        else if (!strcmp(a, "--margins-out")) margins_out = v;
        else if (!strcmp(a, "--tokens-out"))  tokens_out = v;
        else if (!strcmp(a, "--ngl"))         ngl = atoi(v);
        else if (!strcmp(a, "--fa"))          fa = atoi(v);

        else { usage(); return 2; }
        ++i;
    }
    if (!model_path || !tokens_path || !logits_out) { usage(); return 2; }

    int32_t *toks = NULL;
    int n_toks = 0, cap = 4096;
    toks = (int32_t *)malloc(sizeof(int32_t) * cap);
    FILE *ft = fopen(tokens_path, "r");
    if (!ft) { fprintf(stderr, "cannot open %s\n", tokens_path); return 1; }
    while (fscanf(ft, "%d", &toks[n_toks]) == 1) {
        if (++n_toks == cap) { cap *= 2; toks = (int32_t *)realloc(toks, sizeof(int32_t) * cap); }
    }
    fclose(ft);
    if (n_toks == 0) { fprintf(stderr, "empty token list\n"); return 1; }

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cp = llama_context_default_params();
    if (fa >= 0) cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (kv32) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    cp.n_ctx = (uint32_t)(n_toks + gen + 32);
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    FILE *fl = fopen(logits_out, "wb");
    FILE *fm = margins_out ? fopen(margins_out, "w") : NULL;
    FILE *fo = tokens_out ? fopen(tokens_out, "w") : NULL;
    if (!fl) { fprintf(stderr, "cannot open %s\n", logits_out); return 1; }

    for (int s = 0; s < n_toks; ++s) {
        if (llama_decode(ctx, llama_batch_get_one(&toks[s], 1))) {
            fprintf(stderr, "decode failed at %d\n", s);
            return 1;
        }
        const float *lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { fprintf(stderr, "no logits at %d\n", s); return 1; }
        int best = 0, second = 1;
        for (int i = 1; i < n_vocab; ++i) {
            if (lg[i] > lg[best]) { second = best; best = i; }
            else if (i != best && (second == best || lg[i] > lg[second])) second = i;
        }
        fwrite(lg, sizeof(float), (size_t)n_vocab, fl);
        if (fm) fprintf(fm, "%d %d %.6f\n", s, best, lg[best] - lg[second]);
        if (fo) fprintf(fo, "%d\n", best);
    }
    for (int g = 0; g < gen; g++) {
        const float *lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (lg[i] > lg[best]) best = i;
        if (fo) fprintf(fo, "%d\n", best);
        int32_t bt = best;
        if (llama_decode(ctx, llama_batch_get_one(&bt, 1))) { fprintf(stderr, "gen decode failed\n"); return 1; }
    }
    if (fo) { /* already written */ }
    fclose(fl);
    if (fm) fclose(fm);
    if (fo) fclose(fo);

    fprintf(stderr, "fed %d tokens (+%d greedy), vocab %d\n", n_toks, gen, n_vocab);
    free(toks);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
