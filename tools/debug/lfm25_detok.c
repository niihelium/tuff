// lfm25_detok: print the text of token-id files (one id per line) with the model's own vocab (llama.cpp, vocab only).
// build: gcc -O2 tools/debug/lfm25_detok.c -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin -lllama
//            -Wl,-rpath,$LLAMA/build/bin -o lfm25_detok
// usage: lfm25_detok <model.gguf> <ids.txt>...
#include "llama.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <ids.txt>...\n", argv[0]); return 2; }
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = 1;
    struct llama_model *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    const struct llama_vocab *vo = llama_model_get_vocab(m);
    for (int a = 2; a < argc; a++) {
        FILE *f = fopen(argv[a], "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[a]); continue; }
        int t;
        char buf[256];
        printf("== %s\n", argv[a]);
        while (fscanf(f, "%d", &t) == 1) {
            int n = llama_token_to_piece(vo, t, buf, 255, 0, 1);
            buf[n > 0 ? n : 0] = 0;
            printf("%s", buf);
        }
        printf("\n");
        fclose(f);
    }
    llama_model_free(m);
    return 0;
}
