// GPT-2 byte-level BPE tokenizer for the qwen35 vocab (tokenizer.ggml.model = "gpt2",
// tokenizer.ggml.pre = "qwen35").
//
// Ported from llama.cpp (MIT license, see ATTRIBUTION.md):
//   - src/llama-vocab.cpp        llm_tokenizer_bpe / llm_tokenizer_bpe_session (BPE merge loop,
//                                 GGUF metadata loading, bpe_ranks, byte-fallback)
//   - src/unicode.cpp            unicode_regex_split_custom_qwen35 (pretokenizer),
//                                 unicode_byte_to_utf8_map / unicode_utf8_to_byte_map
//                                 (GPT-2 byte<->unicode mapping), unicode_cpt_from_utf8 /
//                                 unicode_cpt_to_utf8 (UTF-8 codec)
//
// Known limitation: exact Unicode General Category tables (\p{L}, \p{N}, \p{M}, whitespace)
// are not ported from llama.cpp's generated unicode-data.cpp (tens of thousands of ranges).
// Classification here is an approximation covering ASCII exactly (matches llama.cpp bit-for-bit
// for pure ASCII input, which is what the M1 oracle prompt uses) plus the common non-ASCII
// letter/digit/mark/whitespace blocks. Exotic scripts may pretokenize slightly differently.

#define _POSIX_C_SOURCE 200809L // for strdup()

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------------------
// Simple open-addressing string -> int32 hash table (FNV-1a, linear probing, first-insert-wins)
// ---------------------------------------------------------------------------------------

typedef struct {
    const char *key;
    size_t key_len;
    int32_t value;
    int used;
} tok_hash_slot_t;

struct tok_hash_table {
    tok_hash_slot_t *slots;
    size_t cap; // power of two
    size_t count;
};

static uint64_t fnv1a(const char *data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (unsigned char)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static tok_hash_table_t* hash_create(size_t expected_n) {
    tok_hash_table_t *t = (tok_hash_table_t*)calloc(1, sizeof(tok_hash_table_t));
    size_t cap = 16;
    while (cap < expected_n * 2 + 16) cap <<= 1;
    t->cap = cap;
    t->slots = (tok_hash_slot_t*)calloc(cap, sizeof(tok_hash_slot_t));
    t->count = 0;
    return t;
}

// Key pointer must remain valid for the hash table's lifetime (caller owns it).
static void hash_put(tok_hash_table_t *t, const char *key, size_t key_len, int32_t value) {
    uint64_t h = fnv1a(key, key_len);
    size_t idx = h & (t->cap - 1);
    while (t->slots[idx].used) {
        if (t->slots[idx].key_len == key_len && memcmp(t->slots[idx].key, key, key_len) == 0) {
            return; // first occurrence wins (correct for BPE merge rank priority)
        }
        idx = (idx + 1) & (t->cap - 1);
    }
    t->slots[idx].key = key;
    t->slots[idx].key_len = key_len;
    t->slots[idx].value = value;
    t->slots[idx].used = 1;
    t->count++;
}

static int hash_get(const tok_hash_table_t *t, const char *key, size_t key_len, int32_t *out_value) {
    if (!t || t->cap == 0) return 0;
    uint64_t h = fnv1a(key, key_len);
    size_t idx = h & (t->cap - 1);
    size_t start = idx;
    while (t->slots[idx].used) {
        if (t->slots[idx].key_len == key_len && memcmp(t->slots[idx].key, key, key_len) == 0) {
            *out_value = t->slots[idx].value;
            return 1;
        }
        idx = (idx + 1) & (t->cap - 1);
        if (idx == start) break;
    }
    return 0;
}

static void hash_free(tok_hash_table_t *t) {
    if (!t) return;
    free(t->slots);
    free(t);
}

// ---------------------------------------------------------------------------------------
// UTF-8 helpers (port of unicode_len_utf8 / unicode_cpt_from_utf8 / unicode_cpt_to_utf8)
// ---------------------------------------------------------------------------------------

static size_t utf8_char_len(unsigned char c) {
    static const size_t lookup[16] = { 1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4 };
    return lookup[c >> 4];
}

static size_t utf8_encode_cpt(uint32_t cpt, char *out) {
    if (cpt <= 0x7F) {
        out[0] = (char)cpt;
        return 1;
    }
    if (cpt <= 0x7FF) {
        out[0] = (char)(0xC0 | ((cpt >> 6) & 0x1F));
        out[1] = (char)(0x80 | (cpt & 0x3F));
        return 2;
    }
    if (cpt <= 0xFFFF) {
        out[0] = (char)(0xE0 | ((cpt >> 12) & 0x0F));
        out[1] = (char)(0x80 | ((cpt >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cpt & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | ((cpt >> 18) & 0x07));
    out[1] = (char)(0x80 | ((cpt >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cpt >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cpt & 0x3F));
    return 4;
}

// Decodes `s[0..len)` into codepoints, also returning the byte offset of each codepoint
// (out_offsets has n+1 entries; out_offsets[n] == len) so callers can slice the original
// byte string by codepoint range without re-encoding.
static uint32_t* utf8_to_cpts_with_offsets(const char *s, size_t len, size_t *out_n, size_t **out_offsets) {
    uint32_t *cpts = (uint32_t*)malloc((len + 1) * sizeof(uint32_t));
    size_t *offs = (size_t*)malloc((len + 2) * sizeof(size_t));
    size_t n = 0, i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)s[i];
        size_t clen = utf8_char_len(c);
        if (i + clen > len) clen = 1;
        int valid = 1;
        for (size_t k = 1; k < clen; ++k) {
            if (((unsigned char)s[i + k] & 0xC0) != 0x80) { valid = 0; break; }
        }
        if (!valid) clen = 1;

        uint32_t cpt;
        if (clen == 1) {
            cpt = c;
        } else if (clen == 2) {
            cpt = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
        } else if (clen == 3) {
            cpt = ((uint32_t)(c & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6) | ((unsigned char)s[i + 2] & 0x3F);
        } else {
            cpt = ((uint32_t)(c & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12) |
                  (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
        }
        offs[n] = i;
        cpts[n] = cpt;
        n++;
        i += clen;
    }
    offs[n] = len;
    *out_n = n;
    *out_offsets = offs;
    return cpts;
}

// ---------------------------------------------------------------------------------------
// Approximate Unicode classification (see file header for the exactness caveat).
// Exact for ASCII; approximate for everything else via representative ranges.
// ---------------------------------------------------------------------------------------

static int cpt_is_whitespace(uint32_t c) {
    switch (c) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return 1;
        default: break;
    }
    if (c >= 0x2000 && c <= 0x200A) return 1;
    return 0;
}

static int cpt_is_number(uint32_t c) {
    if (c >= '0' && c <= '9') return 1;
    if (c >= 0x0660 && c <= 0x0669) return 1; // Arabic-Indic digits
    if (c >= 0x06F0 && c <= 0x06F9) return 1; // Extended Arabic-Indic digits
    if (c >= 0x0966 && c <= 0x096F) return 1; // Devanagari digits
    if (c >= 0xFF10 && c <= 0xFF19) return 1; // fullwidth digits
    return 0;
}

static int cpt_is_accent_mark(uint32_t c) {
    if (c >= 0x0300 && c <= 0x036F) return 1; // combining diacritical marks
    if (c >= 0x1AB0 && c <= 0x1AFF) return 1;
    if (c >= 0x1DC0 && c <= 0x1DFF) return 1;
    if (c >= 0x20D0 && c <= 0x20FF) return 1;
    if (c >= 0xFE20 && c <= 0xFE2F) return 1;
    return 0;
}

static int cpt_is_letter(uint32_t c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return 1;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return 1;
    if (c >= 0xC0 && c <= 0xD6) return 1;
    if (c >= 0xD8 && c <= 0xF6) return 1;
    if (c >= 0xF8 && c <= 0x2AF) return 1;    // Latin Extended-A/B, IPA extensions
    if (c >= 0x370 && c <= 0x3FF && c != 0x374 && c != 0x375 && c != 0x37E) return 1; // Greek
    if (c >= 0x400 && c <= 0x4FF) return 1;   // Cyrillic
    if (c >= 0x530 && c <= 0x58F) return 1;   // Armenian
    if (c >= 0x590 && c <= 0x5FF) return 1;   // Hebrew
    if (c >= 0x600 && c <= 0x6FF) return 1;   // Arabic
    if (c >= 0x900 && c <= 0x97F) return 1;   // Devanagari
    if (c >= 0x3040 && c <= 0x30FF) return 1; // Hiragana/Katakana
    if (c >= 0x3400 && c <= 0x9FFF) return 1; // CJK
    if (c >= 0xAC00 && c <= 0xD7A3) return 1; // Hangul syllables
    if (c >= 0xF900 && c <= 0xFAFF) return 1; // CJK compatibility
    if (c >= 0x20000 && c <= 0x2FA1F) return 1; // CJK ext B and beyond
    return 0;
}

static uint32_t cpt_tolower(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    return c;
}

// ---------------------------------------------------------------------------------------
// qwen35 pretokenizer split — direct port of unicode_regex_split_custom_qwen35()
// (llama.cpp/src/unicode.cpp). Regex being hand-implemented:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//   ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// ---------------------------------------------------------------------------------------

typedef struct { size_t start, len; } span_t;

#define CPT_OOR UINT32_MAX

static span_t* qwen35_split(const uint32_t *cpts, size_t n, size_t *out_n) {
    span_t *spans = (span_t*)malloc((n + 1) * sizeof(span_t));
    size_t n_spans = 0;
    size_t prev_end = 0;
    size_t pos = 0;

    while (pos < n) {
        uint32_t cpt = cpts[pos];

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t c1 = cpt_tolower(cpts[pos + 1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
                prev_end = pos;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t c2 = cpt_tolower(cpts[pos + 2]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
                    prev_end = pos;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cpt == '\r' || cpt == '\n' || cpt_is_number(cpt))) {
            uint32_t next_cpt = (pos + 1 < n) ? cpts[pos + 1] : CPT_OOR;
            int next_letter = (next_cpt != CPT_OOR) && cpt_is_letter(next_cpt);
            int next_accent = (next_cpt != CPT_OOR) && cpt_is_accent_mark(next_cpt);
            if (cpt_is_letter(cpt) || cpt_is_accent_mark(cpt) || next_letter || next_accent) {
                size_t p = pos + 1;
                while (p < n && (cpt_is_letter(cpts[p]) || cpt_is_accent_mark(cpts[p]))) p++;
                spans[n_spans].start = prev_end; spans[n_spans].len = p - prev_end; n_spans++;
                pos = p; prev_end = p;
                continue;
            }
        }

        // \p{N}
        if (cpt_is_number(cpt)) {
            pos += 1;
            spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
            prev_end = pos;
            continue;
        }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            uint32_t f2 = (cpt == ' ' && pos + 1 < n) ? cpts[pos + 1] : ((cpt == ' ') ? CPT_OOR : cpt);
            int f2_valid = (f2 != CPT_OOR);
            int f2_class = f2_valid && (cpt_is_whitespace(f2) || cpt_is_letter(f2) || cpt_is_accent_mark(f2) || cpt_is_number(f2));
            if (f2_valid && !f2_class) {
                size_t p = pos + (cpt == ' ' ? 1 : 0);
                uint32_t fc = (p < n) ? cpts[p] : CPT_OOR;
                int fc_valid = (fc != CPT_OOR);
                int fc_class = fc_valid && (cpt_is_whitespace(fc) || cpt_is_letter(fc) || cpt_is_accent_mark(fc) || cpt_is_number(fc));
                while (fc_valid && !fc_class) {
                    p++;
                    fc = (p < n) ? cpts[p] : CPT_OOR;
                    fc_valid = (fc != CPT_OOR);
                    fc_class = fc_valid && (cpt_is_whitespace(fc) || cpt_is_letter(fc) || cpt_is_accent_mark(fc) || cpt_is_number(fc));
                }
                uint32_t cpt2 = (p < n) ? cpts[p] : CPT_OOR;
                while (cpt2 == '\r' || cpt2 == '\n') {
                    p++;
                    cpt2 = (p < n) ? cpts[p] : CPT_OOR;
                }
                spans[n_spans].start = prev_end; spans[n_spans].len = p - prev_end; n_spans++;
                pos = p; prev_end = p;
                continue;
            }
        }

        // whitespace family: \s*[\r\n]+ | \s+(?!\S) | \s+
        {
            size_t num_ws = 0;
            size_t last_end_rn = 0;
            while (pos + num_ws < n && cpt_is_whitespace(cpts[pos + num_ws])) {
                uint32_t c2 = cpts[pos + num_ws];
                if (c2 == '\r' || c2 == '\n') last_end_rn = pos + num_ws + 1;
                num_ws++;
            }
            if (last_end_rn > 0) {
                pos = last_end_rn;
                spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
                prev_end = pos;
                continue;
            }
            if (num_ws > 1 && (pos + num_ws) < n) {
                pos += num_ws - 1;
                spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
                prev_end = pos;
                continue;
            }
            if (num_ws > 0) {
                pos += num_ws;
                spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
                prev_end = pos;
                continue;
            }
        }

        // no rule matched: consume one codepoint as its own token
        pos += 1;
        spans[n_spans].start = prev_end; spans[n_spans].len = pos - prev_end; n_spans++;
        prev_end = pos;
    }

    *out_n = n_spans;
    return spans;
}

#undef CPT_OOR

// ---------------------------------------------------------------------------------------
// BPE merge (port of llm_tokenizer_bpe_session::tokenize's per-word merge loop)
// ---------------------------------------------------------------------------------------

typedef struct { const char *text; int len; int prev, next; } bpe_sym_t;

static int32_t find_merge_rank(const tokenizer_t *tok, const char *a, size_t alen, const char *b, size_t blen) {
    char keybuf[1024];
    if (alen + blen + 1 >= sizeof(keybuf)) return -1;
    memcpy(keybuf, a, alen);
    keybuf[alen] = '\x1F';
    memcpy(keybuf + alen + 1, b, blen);
    int32_t rank;
    if (hash_get(tok->merge_hash, keybuf, alen + 1 + blen, &rank)) return rank;
    return -1;
}

// Merges `encoded_word` (already GPT-2 byte-mapped) and appends resulting vocab ids to `out`.
// Returns the number of ids written (capped at max_out).
static int bpe_merge_word(const tokenizer_t *tok, const char *encoded_word, size_t enc_len,
                           int32_t *out, int max_out) {
    if (enc_len == 0 || max_out <= 0) return 0;

    int cap = (int)enc_len;
    bpe_sym_t *syms = (bpe_sym_t*)malloc((size_t)cap * sizeof(bpe_sym_t));
    int n = 0;
    size_t off = 0;
    while (off < enc_len) {
        size_t clen = utf8_char_len((unsigned char)encoded_word[off]);
        if (off + clen > enc_len) clen = enc_len - off;
        syms[n].text = encoded_word + off;
        syms[n].len = (int)clen;
        syms[n].prev = n - 1;
        syms[n].next = (off + clen < enc_len) ? n + 1 : -1;
        n++;
        off += clen;
    }

    if (n > 1) {
        while (1) {
            int best_i = -1;
            int32_t best_rank = INT32_MAX;
            for (int i = 0; i != -1; i = syms[i].next) {
                int j = syms[i].next;
                if (j == -1) break;
                int32_t rank = find_merge_rank(tok, syms[i].text, (size_t)syms[i].len, syms[j].text, (size_t)syms[j].len);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank;
                    best_i = i;
                }
            }
            if (best_i == -1) break;
            int j = syms[best_i].next;
            syms[best_i].len = (int)((syms[j].text - syms[best_i].text) + syms[j].len);
            syms[best_i].next = syms[j].next;
            if (syms[j].next != -1) syms[syms[j].next].prev = best_i;
        }
    }

    int count = 0;
    for (int i = 0; i != -1 && count < max_out; i = syms[i].next) {
        int32_t id;
        if (hash_get(tok->vocab_hash, syms[i].text, (size_t)syms[i].len, &id)) {
            out[count++] = id;
            continue;
        }
        // Byte-level fallback: this symbol (a merged BPE piece) has no direct vocab entry —
        // should not normally happen for a self-consistent {vocab, merges} pair. Fall back to
        // its individual byte-mapped unicode characters, each of which is guaranteed to be a
        // base vocab token for a byte-level BPE vocabulary.
        size_t o2 = 0;
        while (o2 < (size_t)syms[i].len && count < max_out) {
            size_t clen = utf8_char_len((unsigned char)syms[i].text[o2]);
            if (o2 + clen > (size_t)syms[i].len) clen = (size_t)syms[i].len - o2;
            int32_t bid;
            if (!hash_get(tok->vocab_hash, syms[i].text + o2, clen, &bid)) {
                fprintf(stderr, "[FATAL ERROR] tokenizer: no vocab entry for a base byte-level "
                                "token — vocab/merges are inconsistent with the GGUF\n");
                exit(EXIT_FAILURE);
            }
            out[count++] = bid;
            o2 += clen;
        }
    }

    free(syms);
    return count;
}

static int tokenize_segment(const tokenizer_t *tok, const char *seg, size_t seg_len, int32_t *out, int max_out) {
    if (seg_len == 0 || max_out <= 0) return 0;

    size_t n_cpts;
    size_t *byte_off;
    uint32_t *cpts = utf8_to_cpts_with_offsets(seg, seg_len, &n_cpts, &byte_off);

    size_t n_spans;
    span_t *spans = qwen35_split(cpts, n_cpts, &n_spans);

    int count = 0;
    for (size_t s = 0; s < n_spans && count < max_out; ++s) {
        size_t b_start = byte_off[spans[s].start];
        size_t b_end = byte_off[spans[s].start + spans[s].len];
        size_t word_len = b_end - b_start;
        if (word_len == 0) continue;

        // GPT-2 byte-level encode: map every raw byte through byte_to_cpt and re-emit as UTF-8.
        char *encoded = (char*)malloc(word_len * 4 + 1);
        size_t enc_len = 0;
        for (size_t bi = 0; bi < word_len; ++bi) {
            unsigned char byte = (unsigned char)seg[b_start + bi];
            char buf4[4];
            size_t clen = utf8_encode_cpt(tok->byte_to_cpt[byte], buf4);
            memcpy(encoded + enc_len, buf4, clen);
            enc_len += clen;
        }

        count += bpe_merge_word(tok, encoded, enc_len, out + count, max_out - count);
        free(encoded);
    }

    free(spans);
    free(cpts);
    free(byte_off);
    return count;
}

// ---------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------

static void build_byte_maps(tokenizer_t *tok) {
    int assigned[256] = {0};
    for (int ch = 0x21; ch <= 0x7E; ++ch) { tok->byte_to_cpt[ch] = (uint32_t)ch; assigned[ch] = 1; }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) { tok->byte_to_cpt[ch] = (uint32_t)ch; assigned[ch] = 1; }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) { tok->byte_to_cpt[ch] = (uint32_t)ch; assigned[ch] = 1; }
    int n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (!assigned[ch]) {
            tok->byte_to_cpt[ch] = 256 + (uint32_t)n;
            n++;
        }
    }
    for (int i = 0; i < 512; ++i) tok->cpt_to_byte[i] = -1;
    for (int b = 0; b < 256; ++b) {
        uint32_t cpt = tok->byte_to_cpt[b];
        if (cpt < 512) tok->cpt_to_byte[cpt] = (int16_t)b;
    }
}

tokenizer_t* tokenizer_init(const gguf_context_t *gguf) {
    const gguf_kv_t *model_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.model");
    if (!model_kv || model_kv->type != GGUF_TYPE_STRING || strcmp(model_kv->val.str, "gpt2") != 0) {
        fprintf(stderr, "[FATAL ERROR] tokenizer.ggml.model is not \"gpt2\" (got %s) — "
                        "this tokenizer implementation only supports GPT-2 byte-level BPE\n",
                model_kv ? model_kv->val.str : "<missing>");
        exit(EXIT_FAILURE);
    }

    const gguf_kv_t *tokens_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.tokens");
    if (!tokens_kv || tokens_kv->type != GGUF_TYPE_ARRAY || tokens_kv->val.arr.item_type != GGUF_TYPE_STRING
        || tokens_kv->val.arr.len == 0 || !tokens_kv->val.arr.data) {
        fprintf(stderr, "[FATAL ERROR] tokenizer.ggml.tokens missing or empty in GGUF\n");
        exit(EXIT_FAILURE);
    }

    tokenizer_t *tok = (tokenizer_t*)calloc(1, sizeof(tokenizer_t));
    tok->vocab_size = (uint32_t)tokens_kv->val.arr.len;
    tok->tokens = (char**)calloc(tok->vocab_size, sizeof(char*));

    char **gguf_tokens = (char**)tokens_kv->val.arr.data;
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        tok->tokens[i] = strdup(gguf_tokens[i] ? gguf_tokens[i] : "");
    }

    tok->token_type = (int32_t*)calloc(tok->vocab_size, sizeof(int32_t));
    const gguf_kv_t *type_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.token_type");
    if (type_kv && type_kv->type == GGUF_TYPE_ARRAY && type_kv->val.arr.data &&
        type_kv->val.arr.len == tok->vocab_size) {
        const int32_t *types = (const int32_t*)type_kv->val.arr.data;
        memcpy(tok->token_type, types, tok->vocab_size * sizeof(int32_t));
    } else {
        for (uint32_t i = 0; i < tok->vocab_size; ++i) tok->token_type[i] = 1; // NORMAL
    }

    // Vocab string -> id hash map.
    tok->vocab_hash = hash_create(tok->vocab_size);
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        hash_put(tok->vocab_hash, tok->tokens[i], strlen(tok->tokens[i]), (int32_t)i);
    }

    // BPE merges -> rank hash map. Key = "first" + 0x1F + "second"; value = rank (lower = earlier
    // = higher merge priority), matching llama.cpp's bpe_ranks (index in the merges array).
    const gguf_kv_t *merges_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.merges");
    if (!merges_kv || merges_kv->type != GGUF_TYPE_ARRAY || merges_kv->val.arr.item_type != GGUF_TYPE_STRING
        || !merges_kv->val.arr.data) {
        fprintf(stderr, "[FATAL ERROR] tokenizer.ggml.merges missing from GGUF\n");
        exit(EXIT_FAILURE);
    }
    uint32_t n_merges = (uint32_t)merges_kv->val.arr.len;
    tok->merge_hash = hash_create(n_merges);
    tok->merge_keys = (char**)malloc((n_merges > 0 ? n_merges : 1) * sizeof(char*));
    tok->n_merge_keys = 0;
    char **gguf_merges = (char**)merges_kv->val.arr.data;
    for (uint32_t i = 0; i < n_merges; ++i) {
        const char *word = gguf_merges[i];
        if (!word) continue;
        size_t wlen = strlen(word);
        // llama.cpp: `word.find(' ', 1)` — search for the separator starting at index 1.
        size_t sp = (size_t)-1;
        for (size_t p = 1; p < wlen; ++p) {
            if (word[p] == ' ') { sp = p; break; }
        }
        if (sp == (size_t)-1) continue;
        size_t alen = sp, blen = wlen - sp - 1;
        char *key = (char*)malloc(alen + 1 + blen);
        memcpy(key, word, alen);
        key[alen] = '\x1F';
        memcpy(key + alen + 1, word + sp + 1, blen);
        hash_put(tok->merge_hash, key, alen + 1 + blen, (int32_t)i);
        tok->merge_keys[tok->n_merge_keys++] = key; // owned; freed in tokenizer_free
    }

    tok->bos_id = (int)tuff_gguf_get_kv_u32(gguf, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
    tok->eos_id = (int)tuff_gguf_get_kv_u32(gguf, "tokenizer.ggml.eos_token_id", (uint32_t)-1);
    if (tok->bos_id < 0 || tok->eos_id < 0) {
        fprintf(stderr, "[FATAL ERROR] tokenizer.ggml.bos_token_id / eos_token_id missing from GGUF\n");
        exit(EXIT_FAILURE);
    }

    const gguf_kv_t *add_bos_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.add_bos_token");
    if (add_bos_kv) {
        tok->add_bos = add_bos_kv->type == GGUF_TYPE_BOOL && add_bos_kv->val.b;
    } else {
        // Key absent: llama.cpp defaults add_bos = true for the llama3 pre-tokenizer group.
        static const char *llama3_group[] = { "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1",
                                              "pixtral", "midm-2.0", "lfm2", "jina-v5-nano" };
        const gguf_kv_t *pre_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.pre");
        tok->add_bos = false;
        if (pre_kv && pre_kv->type == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < sizeof(llama3_group) / sizeof(llama3_group[0]); i++)
                if (strcmp(pre_kv->val.str, llama3_group[i]) == 0) { tok->add_bos = true; break; }
        }
    }
    const gguf_kv_t *add_eos_kv = tuff_gguf_get_kv(gguf, "tokenizer.ggml.add_eos_token");
    tok->add_eos = add_eos_kv && add_eos_kv->type == GGUF_TYPE_BOOL && add_eos_kv->val.b;

    build_byte_maps(tok);

    // Collect CONTROL-type tokens (e.g. <|im_start|>) for literal pre-split matching.
    tok->n_special = 0;
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        if (tok->token_type[i] == 3 /* LLAMA_TOKEN_TYPE_CONTROL */) tok->n_special++;
    }
    tok->special_strs = (char**)malloc((tok->n_special ? tok->n_special : 1) * sizeof(char*));
    tok->special_lens = (size_t*)malloc((tok->n_special ? tok->n_special : 1) * sizeof(size_t));
    tok->special_ids  = (int32_t*)malloc((tok->n_special ? tok->n_special : 1) * sizeof(int32_t));
    uint32_t sp_i = 0;
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        if (tok->token_type[i] == 3) {
            tok->special_strs[sp_i] = tok->tokens[i];
            tok->special_lens[sp_i] = strlen(tok->tokens[i]);
            tok->special_ids[sp_i] = (int32_t)i;
            sp_i++;
        }
    }

    return tok;
}

int tokenizer_tokenize(const tokenizer_t *tok, const char *text, int32_t *out_tokens, int max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens <= 0) return 0;
    int count = 0;

    if (tok->add_bos && count < max_tokens) out_tokens[count++] = tok->bos_id;

    size_t text_len = strlen(text);
    size_t cur = 0;
    while (cur < text_len && count < max_tokens) {
        size_t match_pos = text_len, match_len = 0;
        int32_t match_id = -1;
        for (uint32_t i = 0; i < tok->n_special; ++i) {
            if (tok->special_lens[i] == 0) continue;
            const char *hay = text + cur;
            const char *p = strstr(hay, tok->special_strs[i]);
            if (p) {
                size_t pos = cur + (size_t)(p - hay);
                size_t slen = tok->special_lens[i];
                if (pos < match_pos || (pos == match_pos && slen > match_len)) {
                    match_pos = pos;
                    match_len = slen;
                    match_id = tok->special_ids[i];
                }
            }
        }

        if (match_id < 0) {
            count += tokenize_segment(tok, text + cur, text_len - cur, out_tokens + count, max_tokens - count);
            break;
        }
        if (match_pos > cur) {
            count += tokenize_segment(tok, text + cur, match_pos - cur, out_tokens + count, max_tokens - count);
        }
        if (count < max_tokens) out_tokens[count++] = match_id;
        cur = match_pos + match_len;
    }

    if (tok->add_eos && count < max_tokens) out_tokens[count++] = tok->eos_id;
    return count;
}

const char* tokenizer_decode(const tokenizer_t *tok, int32_t token_id) {
    static char buf[4096];
    if (!tok || token_id < 0 || token_id >= (int32_t)tok->vocab_size) {
        snprintf(buf, sizeof(buf), "[%d]", token_id);
        return buf;
    }

    const char *piece = tok->tokens[token_id];
    if (!piece) { buf[0] = '\0'; return buf; }

    int32_t ttype = tok->token_type ? tok->token_type[token_id] : 1;
    if (ttype == 3 /* CONTROL */) {
        snprintf(buf, sizeof(buf), "%s", piece);
        return buf;
    }

    size_t plen = strlen(piece);
    size_t n_cpts;
    size_t *offs;
    uint32_t *cpts = utf8_to_cpts_with_offsets(piece, plen, &n_cpts, &offs);

    size_t out_len = 0;
    for (size_t i = 0; i < n_cpts && out_len < sizeof(buf) - 1; ++i) {
        uint32_t cpt = cpts[i];
        int byte_val = (cpt < 512) ? tok->cpt_to_byte[cpt] : -1;
        if (byte_val >= 0) {
            buf[out_len++] = (char)byte_val;
        } else {
            size_t b0 = offs[i], b1 = offs[i + 1];
            size_t clen = b1 - b0;
            if (out_len + clen < sizeof(buf)) {
                memcpy(buf + out_len, piece + b0, clen);
                out_len += clen;
            }
        }
    }
    buf[out_len] = '\0';

    free(cpts);
    free(offs);
    return buf;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (uint32_t i = 0; i < tok->vocab_size; ++i) free(tok->tokens[i]);
        free(tok->tokens);
    }
    free(tok->token_type);
    free(tok->special_strs);
    free(tok->special_lens);
    free(tok->special_ids);
    hash_free(tok->vocab_hash);
    hash_free(tok->merge_hash);
    if (tok->merge_keys) {
        for (uint32_t i = 0; i < tok->n_merge_keys; ++i) free(tok->merge_keys[i]);
        free(tok->merge_keys);
    }
    free(tok);
}
