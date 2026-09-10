#ifndef GGUF_PARSER_H
#define GGUF_PARSER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ggml_types.h"

#define GGUF_MAGIC 0x46554747 // "GGUF" in little endian

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type_t;

typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type; // GGML type code (e.g. 0=F32, 8=Q8_0, 12=Q4_K, 14=Q6_K)
    uint64_t offset; // Byte offset relative to tensor_data_blob
    uint64_t nelements; // Total number of elements
    size_t size_bytes; // Exact size in bytes
} gguf_tensor_info_t;

typedef struct {
    char key[128];
    uint32_t type;
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
        char str[256];
        struct {
            uint32_t item_type;
            uint64_t len;
            void *data;
        } arr;
    } val;
} gguf_kv_t;

typedef struct {
    int fd;
    size_t file_size;
    void *mmap_addr;
    
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    
    gguf_kv_t *kv_pairs;
    gguf_tensor_info_t *tensors;
    const uint8_t *tensor_data_blob;
    uint32_t alignment;
} gguf_context_t;

// API functions (prefixed with tuff_gguf_ to prevent ELF collision with libggml.so)
gguf_context_t* tuff_gguf_parse_file(const char *filepath);
void tuff_gguf_warmup_memory(gguf_context_t *ctx);
const gguf_tensor_info_t* tuff_gguf_get_tensor(const gguf_context_t *ctx, const char *name);
const gguf_kv_t* tuff_gguf_get_kv(const gguf_context_t *ctx, const char *key);
uint32_t tuff_gguf_get_kv_u32(const gguf_context_t *ctx, const char *key, uint32_t default_val);
float tuff_gguf_get_kv_f32(const gguf_context_t *ctx, const char *key, float default_val);
void tuff_gguf_free(gguf_context_t *ctx);

#endif // GGUF_PARSER_H
