// lfm25_llama_ref: reproducible llama.cpp oracle for the LFM2.5 gate (tests/verify_lfm25_oracle.sh).
//
// Runs llama.cpp (libllama, linked from the same tree tuff builds against) in the configuration
// that mirrors tuff's decode path: every token -- prompt included -- is decoded as its own
// single-token batch (n_batch = n_ubatch = 1), all layers on the GPU, greedy argmax over the raw
// logits. Batching the prompt or offloading fewer layers switches llama.cpp to different kernels
// and therefore a different (equally valid) token stream, which is why the old reference could not
// be regenerated: its recipe was never recorded. This tool is that recipe.
//
// Outputs:
//   --tokens-out  one generated token ID per line (the reference stream)
//   --margins-out one line per step: "<step> <top1> <top1-top2 logit margin>"
//   --logits-out  raw float32 [n x vocab] logits, step i = the vector that selected token i
//   stderr        "prompt tokens (N): ..." so the gate can check tuff tokenized identically
//
// --cpu-experts keeps the MoE expert tensors (ffn_{gate,up,down}_exps) in host memory and runs them
// on llama.cpp's CPU backend while everything else stays on the GPU -- the only way the 24B model
// fits an 8 GB card. CPU expert matmuls use different activation quantization than CUDA, so this is
// a different (equally valid) reference stream; the gate's FP64 stage judges both runtimes alike.
//
// With --teacher <file>, step i is fed token i of <file> instead of the argmax (logits/margins
// then describe llama.cpp's view of that fixed stream).
#include "llama.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "usage: lfm25-llama-ref --model <gguf> [--prompt <text>] [-n <steps>] [--ngl <layers>]\n"
        "                       [--teacher <tokens.txt>] [--tokens-out <f>] [--margins-out <f>]\n"
        "                       [--logits-out <f>] [--cpu-experts]\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = "Hello", *teacher = NULL;
    const char *tok_out = NULL, *margin_out = NULL, *logit_out = NULL;
    int n = 128, ngl = 99, cpu_experts = 0;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "--cpu-experts")) { cpu_experts = 1; continue; }
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { usage(); return 2; }
        if      (!strcmp(a, "--model"))       model_path = v;
        else if (!strcmp(a, "--prompt"))      prompt = v;
        else if (!strcmp(a, "-n"))            n = atoi(v);
        else if (!strcmp(a, "--ngl"))         ngl = atoi(v);
        else if (!strcmp(a, "--teacher"))     teacher = v;
        else if (!strcmp(a, "--tokens-out"))  tok_out = v;
        else if (!strcmp(a, "--margins-out")) margin_out = v;
        else if (!strcmp(a, "--logits-out"))  logit_out = v;
        else { usage(); return 2; }
        ++i;
    }
    if (!model_path || n <= 0) { usage(); return 2; }

    int32_t *ref = NULL;
    int n_ref = 0;
    if (teacher) {
        FILE *f = fopen(teacher, "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", teacher); return 1; }
        ref = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
        while (n_ref < n && fscanf(f, "%d", &ref[n_ref]) == 1) n_ref++;
        fclose(f);
        if (n_ref < n) n = n_ref;
    }

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    struct llama_model_tensor_buft_override ovr[2] = {{NULL, NULL}, {NULL, NULL}};
    if (cpu_experts) {
        ovr[0].pattern = "ffn_(up|down|gate)_exps";
        ovr[0].buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        mp.tensor_buft_overrides = ovr;
    }
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token prompt_tokens[1024];
    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), prompt_tokens, 1024, true, false);
    if (n_prompt <= 0) { fprintf(stderr, "tokenization failed\n"); return 1; }
    fprintf(stderr, "prompt tokens (%d):", n_prompt);
    for (int i = 0; i < n_prompt; ++i) fprintf(stderr, " %d", prompt_tokens[i]);
    fprintf(stderr, "\n");

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n_prompt + n + 16);
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    FILE *ft = tok_out ? fopen(tok_out, "w") : NULL;
    FILE *fm = margin_out ? fopen(margin_out, "w") : NULL;
    FILE *fl = logit_out ? fopen(logit_out, "wb") : NULL;

    for (int i = 0; i < n_prompt; ++i) {
        llama_token t = prompt_tokens[i];
        if (llama_decode(ctx, llama_batch_get_one(&t, 1))) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    for (int s = 0; s < n; ++s) {
        const float *lg = llama_get_logits_ith(ctx, -1);
        int best = 0, second = -1;
        for (int i = 1; i < n_vocab; ++i) {
            if (lg[i] > lg[best]) { second = best; best = i; }
            else if (second < 0 || lg[i] > lg[second]) second = i;
        }
        if (ft) fprintf(ft, "%d\n", best);
        if (fm) fprintf(fm, "%d %d %.6f\n", s, best, lg[best] - lg[second]);
        if (fl) fwrite(lg, sizeof(float), (size_t)n_vocab, fl);
        llama_token next = teacher ? ref[s] : best;
        if (llama_decode(ctx, llama_batch_get_one(&next, 1))) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    if (ft) fclose(ft);
    if (fm) fclose(fm);
    if (fl) fclose(fl);
    free(ref);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
