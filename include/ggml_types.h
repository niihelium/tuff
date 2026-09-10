#ifndef GGML_TYPES_H
#define GGML_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>

// Works in both C11 (_Static_assert) and CUDA/nvcc's C++ (static_assert).
#if defined(__cplusplus)
#define GGML_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define GGML_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif

#define QK8_0 32
#define QK_K 256
#define K_SCALE_SIZE 12

typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT
} ggml_type_t;

// block_q8_0: block size 32, 34 bytes
typedef struct {
    uint16_t d;         // fp16 delta
    int8_t   qs[QK8_0]; // 32 quantized values
} block_q8_0;
GGML_STATIC_ASSERT(sizeof(block_q8_0) == 34, "wrong q8_0 block size/padding");

// block_q4_K: block size 256, 144 bytes
typedef struct {
    uint16_t d;                    // fp16 super-block scale
    uint16_t dmin;                 // fp16 super-block min
    uint8_t  scales[K_SCALE_SIZE]; // 12-byte sub-block scale array
    uint8_t  qs[QK_K / 2];        // 128 bytes = 256 4-bit quants
} block_q4_K;
GGML_STATIC_ASSERT(sizeof(block_q4_K) == 144, "wrong q4_K block size/padding");

// block_q6_K: block size 256, 210 bytes
typedef struct {
    uint8_t  ql[QK_K / 2];      // 128 bytes = 256 lower 4 bits
    uint8_t  qh[QK_K / 4];      // 64 bytes = 256 upper 2 bits
    int8_t   scales[QK_K / 16]; // 16 bytes = scales
    uint16_t d;                // fp16 super-block scale
} block_q6_K;
GGML_STATIC_ASSERT(sizeof(block_q6_K) == 210, "wrong q6_K block size/padding");

typedef struct {
    const char *type_name;
    int32_t     block_size; // elements per block
    size_t      type_size;  // bytes per block
    bool        is_quantized;
} ggml_type_traits_t;

static inline const ggml_type_traits_t *ggml_get_type_traits(ggml_type_t type) {
    // Written with a switch (not C designated array initializers) so this header compiles
    // both as C11 and under nvcc's C++ pass (gpu.cu includes it via model.h).
    static const ggml_type_traits_t t_f32  = { "f32",  1,     sizeof(float),      false };
    static const ggml_type_traits_t t_f16  = { "f16",  1,     sizeof(uint16_t),   false };
    static const ggml_type_traits_t t_q8_0 = { "q8_0", QK8_0, sizeof(block_q8_0), true  };
    static const ggml_type_traits_t t_q4_k = { "q4_K", QK_K,  sizeof(block_q4_K), true  };
    static const ggml_type_traits_t t_q6_k = { "q6_K", QK_K,  sizeof(block_q6_K), true  };
    static const ggml_type_traits_t t_i8   = { "i8",   1,     sizeof(int8_t),     false };
    static const ggml_type_traits_t t_i16  = { "i16",  1,     sizeof(int16_t),    false };
    static const ggml_type_traits_t t_i32  = { "i32",  1,     sizeof(int32_t),    false };
    static const ggml_type_traits_t t_i64  = { "i64",  1,     sizeof(int64_t),    false };

    switch (type) {
        case GGML_TYPE_F32:  return &t_f32;
        case GGML_TYPE_F16:  return &t_f16;
        case GGML_TYPE_Q8_0: return &t_q8_0;
        case GGML_TYPE_Q4_K: return &t_q4_k;
        case GGML_TYPE_Q6_K: return &t_q6_k;
        case GGML_TYPE_I8:   return &t_i8;
        case GGML_TYPE_I16:  return &t_i16;
        case GGML_TYPE_I32:  return &t_i32;
        case GGML_TYPE_I64:  return &t_i64;
        default:             return NULL;
    }
}

static inline size_t ggml_row_size(ggml_type_t type, int64_t n_elements) {
    const ggml_type_traits_t *t = ggml_get_type_traits(type);
    if (!t) return 0;
    assert(n_elements % t->block_size == 0);
    return (size_t)(n_elements / t->block_size) * t->type_size;
}

#endif // GGML_TYPES_H
