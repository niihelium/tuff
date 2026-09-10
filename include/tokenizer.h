#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf_parser.h"
#include <stdint.h>
#include <stddef.h>

// GPT-2 byte-level BPE tokenizer, ported from llama.cpp's llm_tokenizer_bpe /
// llm_tokenizer_bpe_session (src/llama-vocab.cpp) and the qwen35 custom pre-tokenizer
// (unicode_regex_split_custom_qwen35 in src/unicode.cpp). See ATTRIBUTION.md.
typedef struct tok_hash_table tok_hash_table_t;

typedef struct {
    uint32_t vocab_size;
    char **tokens;        // vocab_size entries: raw GGUF token strings (GPT-2 byte-mapped form)
    int32_t *token_type;  // vocab_size entries: llama_token_type (1=NORMAL, 3=CONTROL, ...)

    int bos_id;
    int eos_id;
    bool add_bos;
    bool add_eos;

    // token string -> token id
    tok_hash_table_t *vocab_hash;

    // "first\x1Fsecond" -> merge rank (lower = higher priority)
    tok_hash_table_t *merge_hash;
    char **merge_keys;       // owned allocations backing merge_hash's keys, for cleanup
    uint32_t n_merge_keys;

    // GPT-2 byte <-> unicode codepoint maps (see unicode_byte_to_utf8_map in llama.cpp).
    // Codepoints produced by this map range over [0x21,0xFF] (assigned bytes) plus
    // [256, 256+n) for the remaining "gap" bytes (n <= 256), so 512 is a safe upper bound.
    uint32_t byte_to_cpt[256];
    int16_t  cpt_to_byte[512]; // -1 = not a byte-map codepoint

    // CONTROL-type tokens (e.g. <|im_start|>), matched literally in raw text before
    // pretokenization/BPE (mirrors llama.cpp's tokenizer_st_partition, simplified to exact
    // substring matching — sufficient since these strings never legitimately occur as
    // substrings of ordinary text).
    uint32_t n_special;
    char **special_strs;
    size_t *special_lens;
    int32_t *special_ids;
} tokenizer_t;

tokenizer_t* tokenizer_init(const gguf_context_t *gguf);
int tokenizer_tokenize(const tokenizer_t *tok, const char *text, int32_t *tokens, int max_tokens);
const char* tokenizer_decode(const tokenizer_t *tok, int32_t token_id);
void tokenizer_free(tokenizer_t *tok);

#endif // TOKENIZER_H
