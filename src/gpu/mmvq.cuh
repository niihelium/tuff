#pragma once

#include "common.cuh"
#include "ggml_types.h"

#define MMVQ_ROWS_PER_BLOCK 4

#define QK8_1 32
#define QI8_1 8                  // 32 int8 quants = 8 ints
#define QI4_K 32                 // QK_K/(4*QR4_K)
#define QR4_K 2
#define QI6_K 32                 // QK_K/(4*QR6_K)
#define QR6_K 2
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1

// block_q8_1 (ggml-common.h:258). ds.x = d (scale), ds.y = sum of the original f32 values.
typedef struct {
    __half2 ds;
    int8_t  qs[QK8_1];
} block_q8_1_c;
GGML_STATIC_ASSERT(sizeof(block_q8_1_c) == 36, "wrong q8_1 block size/padding");

static __device__ __forceinline__ int get_int_b2_c(const void *x, int i32) {
    const uint16_t *x16 = (const uint16_t *)x; // q6_K blocks are only 2-byte aligned
    return (int)((uint32_t)x16[2*i32 + 0] | ((uint32_t)x16[2*i32 + 1] << 16));
}

static __device__ __forceinline__ int get_int_b4_c(const void *x, int i32) {
    return ((const int *)x)[i32];
}


// vec_dot_q4_K_q8_1 (vecdotq.cuh:910) + vec_dot_q4_K_q8_1_impl_vmmq (vecdotq.cuh:508).
// bq4_K is the super-block, bq8_1 the aligned start of this super-block's 8 q8_1 blocks.
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_c(
        const block_q4_K * __restrict__ bq4_K, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int   v[2];
    int   u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int *q4 = (const int *)(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t *scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1_c *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int *q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0));
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    return h2f(bq4_K->d)*sumf_d - h2f(bq4_K->dmin)*sumf_m;
}

// vec_dot_q6_K_q8_1 (vecdotq.cuh:1002) + vec_dot_q6_K_q8_1_impl_mmvq (vecdotq.cuh:627).
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_c(
        const block_q6_K * __restrict__ bq6_K, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    const int bq8_offset   = 2*QR6_K*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4)*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2*((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2_c(bq6_K->ql, iqs);
    const int vh = get_int_b2_c(bq6_K->qh, (QI6_K/4)*(iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t *scales = bq6_K->scales + scale_offset;

    int   u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4_c(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return h2f(bq6_K->d) * sumf;
}

// H9: vec_dot_q8_0_q8_1 (vecdotq.cuh) for the DENSE Q8_0 projections — qkv, attn_gate, ssm_out,
// attn_q/k/v, attn_output and the shared expert. The scalar kernels these replace spend one fp32
// multiply per weight byte and reach 206–239 GB/s; the __dp4a form does four weights per
// instruction and measured 352 GB/s on the Q6_K output head (iteration 38).
#define QI8_0 8                  // 32 int8 quants = 8 ints per block
#define VDR_Q8_0_Q8_1_MMVQ 2     // ints per lane-iteration

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_c(
        const block_q8_0 * __restrict__ bq8_0, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        // block_q8_0 is 34 B, so qs is only 2-byte aligned — same reason q6_K uses get_int_b2_c.
        const int v = get_int_b2_c(bq8_0->qs, iqs + i);
        const int u = get_int_b4_c(bq8_1->qs, iqs + i);
        sumi = __dp4a(v, u, sumi);
    }
    return h2f(bq8_0->d) * __low2float(bq8_1->ds) * (float)sumi;
}

#ifndef MMVQ_ROWS_PER_BLOCK
#define MMVQ_ROWS_PER_BLOCK 4
#endif

