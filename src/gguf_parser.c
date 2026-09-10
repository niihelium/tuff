#include "gguf_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <assert.h>

static uint64_t parse_string(const uint8_t *data, size_t max_len, char *out, size_t out_max) {
    if (max_len < 8) return 0;
    uint64_t str_len = *(const uint64_t*)data;
    if (8 + str_len > max_len) {
        if (out && out_max > 0) out[0] = '\0';
        return 8;
    }
    if (out && out_max > 0) {
        size_t copy_len = str_len < (out_max - 1) ? str_len : (out_max - 1);
        memcpy(out, data + 8, copy_len);
        out[copy_len] = '\0';
    }
    return 8 + str_len;
}

// Like parse_string, but heap-allocates a buffer sized exactly to the string (no truncation).
// Used for GGUF string arrays such as tokenizer.ggml.tokens / .merges, where entries must not
// be silently truncated at an arbitrary fixed width.
static uint64_t parse_string_alloc(const uint8_t *data, size_t max_len, char **out) {
    if (max_len < 8) { *out = NULL; return 0; }
    uint64_t str_len = *(const uint64_t*)data;
    if (8 + str_len > max_len) {
        *out = (char*)calloc(1, 1);
        return 8;
    }
    char *s = (char*)malloc(str_len + 1);
    memcpy(s, data + 8, str_len);
    s[str_len] = '\0';
    *out = s;
    return 8 + str_len;
}

gguf_context_t* tuff_gguf_parse_file(const char *filepath) {
    if (!filepath) return NULL;

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[GGUF Parser Error] Failed to open file %s\n", filepath);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    void *addr = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "[GGUF Parser Error] Failed to mmap file %s\n", filepath);
        close(fd);
        return NULL;
    }

    const uint8_t *ptr = (const uint8_t*)addr;

    // Advise kernel of random/sequential read pattern
    madvise(addr, st.st_size, MADV_WILLNEED);

    const uint8_t *curr = (const uint8_t*)ptr;
    uint32_t magic = *(const uint32_t*)curr;
    curr += 4;

    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[GGUF Parser Error] Invalid magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        munmap(addr, st.st_size);
        close(fd);
        return NULL;
    }

    gguf_context_t *ctx = (gguf_context_t*)calloc(1, sizeof(gguf_context_t));
    ctx->fd = fd;
    ctx->file_size = st.st_size;
    ctx->mmap_addr = addr;
    ctx->alignment = 32; // Default GGUF alignment

    ctx->version = *(const uint32_t*)curr;
    curr += 4;
    ctx->n_tensors = *(const uint64_t*)curr;
    curr += 8;
    ctx->n_kv = *(const uint64_t*)curr;
    curr += 8;

    // Parse Key-Values
    ctx->kv_pairs = (gguf_kv_t*)calloc(ctx->n_kv, sizeof(gguf_kv_t));
    for (uint64_t i = 0; i < ctx->n_kv; ++i) {
        gguf_kv_t *kv = &ctx->kv_pairs[i];
        curr += parse_string(curr, st.st_size - (curr - ptr), kv->key, sizeof(kv->key));
        kv->type = *(const uint32_t*)curr;
        curr += 4;

        switch (kv->type) {
            case GGUF_TYPE_UINT8:   kv->val.u8 = *curr; curr += 1; break;
            case GGUF_TYPE_INT8:    kv->val.i8 = *(const int8_t*)curr; curr += 1; break;
            case GGUF_TYPE_UINT16:  kv->val.u16 = *(const uint16_t*)curr; curr += 2; break;
            case GGUF_TYPE_INT16:   kv->val.i16 = *(const int16_t*)curr; curr += 2; break;
            case GGUF_TYPE_UINT32:  kv->val.u32 = *(const uint32_t*)curr; curr += 4; break;
            case GGUF_TYPE_INT32:   kv->val.i32 = *(const int32_t*)curr; curr += 4; break;
            case GGUF_TYPE_FLOAT32: kv->val.f32 = *(const float*)curr; curr += 4; break;
            case GGUF_TYPE_UINT64:  kv->val.u64 = *(const uint64_t*)curr; curr += 8; break;
            case GGUF_TYPE_INT64:   kv->val.i64 = *(const int64_t*)curr; curr += 8; break;
            case GGUF_TYPE_FLOAT64: kv->val.f64 = *(const double*)curr; curr += 8; break;
            case GGUF_TYPE_BOOL:    kv->val.b = (*curr != 0); curr += 1; break;
            case GGUF_TYPE_STRING:  curr += parse_string(curr, st.st_size - (curr - ptr), kv->val.str, sizeof(kv->val.str)); break;
            case GGUF_TYPE_ARRAY: {
                uint32_t arr_type = *(const uint32_t*)curr; curr += 4;
                uint64_t arr_len = *(const uint64_t*)curr; curr += 8;
                kv->val.arr.item_type = arr_type;
                kv->val.arr.len = arr_len;
                if (arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_UINT32) {
                    kv->val.arr.data = malloc(arr_len * 4);
                    memcpy(kv->val.arr.data, curr, arr_len * 4);
                    curr += arr_len * 4;
                } else if (arr_type == GGUF_TYPE_STRING) {
                    // Store the full string array (needed for tokenizer.ggml.tokens / .merges) —
                    // previously these were parsed and discarded into a throwaway buffer, which
                    // silently dropped the entire vocabulary and merge list.
                    char **strs = (arr_len > 0) ? (char**)malloc(arr_len * sizeof(char*)) : NULL;
                    for (uint64_t a = 0; a < arr_len; ++a) {
                        char *s = NULL;
                        curr += parse_string_alloc(curr, st.st_size - (curr - ptr), &s);
                        strs[a] = s;
                    }
                    kv->val.arr.data = strs;
                }
                break;
            }
            default:
                fprintf(stderr, "[GGUF Parser Warning] Unknown metadata type %u for key '%s'\n", kv->type, kv->key);
                curr += 4;
                break;
        }

        if (strcmp(kv->key, "general.alignment") == 0 && kv->type == GGUF_TYPE_UINT32) {
            ctx->alignment = kv->val.u32;
        }
    }

    // Parse Tensors Descriptors
    ctx->tensors = (gguf_tensor_info_t*)calloc(ctx->n_tensors, sizeof(gguf_tensor_info_t));
    for (uint64_t i = 0; i < ctx->n_tensors; ++i) {
        curr += parse_string(curr, st.st_size - (curr - ptr), ctx->tensors[i].name, sizeof(ctx->tensors[i].name));
        ctx->tensors[i].n_dims = *(const uint32_t*)curr;
        curr += 4;

        // GGML convention: unused trailing dims are 1, not 0 (so nelements and
        // any dims[d>=n_dims] reads behave like a size-1 axis, matching ggml_tensor::ne[]).
        ctx->tensors[i].dims[0] = 1;
        ctx->tensors[i].dims[1] = 1;
        ctx->tensors[i].dims[2] = 1;
        ctx->tensors[i].dims[3] = 1;

        ctx->tensors[i].nelements = 1;
        for (uint32_t d = 0; d < ctx->tensors[i].n_dims; ++d) {
            ctx->tensors[i].dims[d] = *(const uint64_t*)curr;
            ctx->tensors[i].nelements *= ctx->tensors[i].dims[d];
            curr += 8;
        }

        ctx->tensors[i].type = *(const uint32_t*)curr;
        curr += 4;
        ctx->tensors[i].offset = *(const uint64_t*)curr;
        curr += 8;

        // Compute exact byte size
        ctx->tensors[i].size_bytes = ggml_row_size(ctx->tensors[i].type, ctx->tensors[i].nelements);
    }

    // Align the data blob pointer
    uintptr_t current_offset = (uintptr_t)(curr - (const uint8_t*)ptr);
    uintptr_t alignment = ctx->alignment ? ctx->alignment : 32;
    uintptr_t padding = (alignment - (current_offset % alignment)) % alignment;
    ctx->tensor_data_blob = curr + padding;

    return ctx;
}

// Iteration 11: warm every mmap'd byte of model weights into RAM before timed decode begins.
// Linux lazily faults mmap pages on first access. During token generation, missed-expert reads
// stall on disk I/O / page faults unless the pages are already hot in the page cache.
// Touching one byte per 4096-byte page forces the kernel to populate the page table.
void tuff_gguf_warmup_memory(gguf_context_t *ctx) {
    if (!ctx || !ctx->mmap_addr || ctx->file_size == 0) return;
    const volatile char *p = (const volatile char *)ctx->mmap_addr;
    const size_t page_sz = 4096;
    const size_t n_pages = ctx->file_size / page_sz;
    // Sequential stride across the entire mapped file
    uint64_t acc = 0;
    for (size_t i = 0; i < n_pages; ++i) {
        acc += (uint8_t)p[i * page_sz];
    }
    // Also touch the last byte
    if (ctx->file_size > 0) {
        acc += (uint8_t)p[ctx->file_size - 1];
    }
    (void)acc;
}

const gguf_tensor_info_t* tuff_gguf_get_tensor(const gguf_context_t *ctx, const char *name) {
    if (!ctx || !name) return NULL;
    for (uint64_t i = 0; i < ctx->n_tensors; ++i) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

const gguf_kv_t* tuff_gguf_get_kv(const gguf_context_t *ctx, const char *key) {
    if (!ctx || !key) return NULL;
    for (uint64_t i = 0; i < ctx->n_kv; ++i) {
        if (strcmp(ctx->kv_pairs[i].key, key) == 0) {
            return &ctx->kv_pairs[i];
        }
    }
    return NULL;
}

uint32_t tuff_gguf_get_kv_u32(const gguf_context_t *ctx, const char *key, uint32_t default_val) {
    const gguf_kv_t *kv = tuff_gguf_get_kv(ctx, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_UINT32) return kv->val.u32;
    if (kv->type == GGUF_TYPE_INT32) return (uint32_t)kv->val.i32;
    if (kv->type == GGUF_TYPE_UINT64) return (uint32_t)kv->val.u64;
    if (kv->type == GGUF_TYPE_INT64) return (uint32_t)kv->val.i64;
    return default_val;
}

float tuff_gguf_get_kv_f32(const gguf_context_t *ctx, const char *key, float default_val) {
    const gguf_kv_t *kv = tuff_gguf_get_kv(ctx, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_FLOAT32) return kv->val.f32;
    if (kv->type == GGUF_TYPE_FLOAT64) return (float)kv->val.f64;
    return default_val;
}

void tuff_gguf_free(gguf_context_t *ctx) {
    if (!ctx) return;
    if (ctx->kv_pairs) {
        for (uint64_t i = 0; i < ctx->n_kv; ++i) {
            gguf_kv_t *kv = &ctx->kv_pairs[i];
            if (kv->type == GGUF_TYPE_ARRAY && kv->val.arr.data) {
                if (kv->val.arr.item_type == GGUF_TYPE_STRING) {
                    char **strs = (char**)kv->val.arr.data;
                    for (uint64_t a = 0; a < kv->val.arr.len; ++a) {
                        free(strs[a]);
                    }
                }
                free(kv->val.arr.data);
            }
        }
        free(ctx->kv_pairs);
    }
    if (ctx->tensors) free(ctx->tensors);
    if (ctx->mmap_addr) munmap(ctx->mmap_addr, ctx->file_size);
    if (ctx->fd >= 0) close(ctx->fd);
    free(ctx);
}
