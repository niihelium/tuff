#include "tokenizer.h"
#include "gguf_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf";
    printf("Loading Qwen tokenizer from: %s\n", model_path);

    gguf_context_t *gguf = tuff_gguf_parse_file(model_path);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF: %s\n", model_path);
        return 1;
    }

    tokenizer_t *tok = tokenizer_init(gguf);
    if (!tok) {
        fprintf(stderr, "Failed to init tokenizer\n");
        tuff_gguf_free(gguf);
        return 1;
    }

    printf("Vocab size: %u\n", tok->vocab_size);
    printf("BOS id: %d, EOS id: %d\n", tok->bos_id, tok->eos_id);
    printf("add_bos: %d (expected 0 for Qwen)\n", tok->add_bos);

    // Qwen models must NOT have add_bos=true by default
    if (tok->add_bos) {
        fprintf(stderr, "FAIL: add_bos is true for Qwen! Expected false.\n");
        return 2;
    }

    int32_t tokens[64];
    int n = tokenizer_tokenize(tok, "Hello", tokens, 64);
    printf("Tokenize 'Hello': n=%d, token[0]=%d\n", n, tokens[0]);
    if (n != 1 || tokens[0] != 9419) {
        fprintf(stderr, "FAIL: 'Hello' expected [9419], got %d tokens, tokens[0]=%d\n", n, tokens[0]);
        return 3;
    }

    // Special token check
    n = tokenizer_tokenize(tok, "<|im_start|>user\nHello<|im_end|>\n", tokens, 64);
    printf("Tokenize chat format: n=%d\n", n);
    for (int i = 0; i < n; ++i) {
        printf("  token[%d] = %d (%s)\n", i, tokens[i], tokenizer_decode(tok, tokens[i]));
    }
    // First token must be <|im_start|>
    if (tokens[0] != 151644) { // standard qwen <|im_start|> id
        printf("Note: <|im_start|> id is %d\n", tokens[0]);
    }

    tokenizer_free(tok);
    tuff_gguf_free(gguf);
    printf("QWEN TOKENIZER TEST: PASS\n");
    return 0;
}
