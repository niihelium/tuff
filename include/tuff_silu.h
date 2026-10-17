// SwiGLU activation with the SAME float operations on the CPU and the GPU.
//
// Lossless decode needs a routed expert to produce bit-identical output whether it runs on the
// GPU (cache hit) or the CPU (miss), so the output never depends on what the expert cache holds.
// The dot products already match (qwen_cpu_expert.c emulates the GPU lane math); the activation
// did not, because CUDA's expf and glibc's expf round differently. This exp is a fixed sequence
// of IEEE ops - explicit fma, explicitly rounded add/mul/div on the device (no contraction), and
// the host side compiled with -ffp-contract=off - so both sides return the same bits.
//
// exp: Cody-Waite range reduction x = n ln2 + r, |r| <= ln2/2, degree-7 polynomial, scale by 2^n
// through the exponent bits. ~1 ulp over the range SwiGLU sees; flushes to 0 / +inf outside.
#ifndef TUFF_SILU_H
#define TUFF_SILU_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#if defined(__CUDACC__)
#define TS_FN static __host__ __device__ __forceinline__
#else
#define TS_FN static inline
#endif

#if defined(__CUDA_ARCH__)
#define TS_ADD(a, b) __fadd_rn((a), (b))
#define TS_MUL(a, b) __fmul_rn((a), (b))
#define TS_DIV(a, b) __fdiv_rn((a), (b))
#define TS_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#define TS_RINT(a) rintf(a)
#else
#define TS_ADD(a, b) ((a) + (b))
#define TS_MUL(a, b) ((a) * (b))
#define TS_DIV(a, b) ((a) / (b))
#define TS_FMA(a, b, c) fmaf((a), (b), (c))
#define TS_RINT(a) rintf(a)
#endif

TS_FN float tuff_expf(float x) {
    if (x > 88.0f) return INFINITY;
    if (x < -87.0f) return 0.0f;
    const float n = TS_RINT(TS_MUL(x, 1.44269502162933349609375f));
    float r = TS_FMA(-n, 0.693145751953125f, x);           // ln2 hi
    r = TS_FMA(-n, 1.428606765330187045e-06f, r);          // ln2 lo
    float p = 1.9875691500e-4f;
    p = TS_FMA(p, r, 1.3981999507e-3f);
    p = TS_FMA(p, r, 8.3334519073e-3f);
    p = TS_FMA(p, r, 4.1665795894e-2f);
    p = TS_FMA(p, r, 1.6666665459e-1f);
    p = TS_FMA(p, r, 5.0000001201e-1f);
    p = TS_FMA(p, r, 1.0f);
    p = TS_FMA(p, r, 1.0f);
    const int32_t e = ((int32_t)n + 127) << 23;
    float s;
#if defined(__CUDA_ARCH__)
    s = __int_as_float(e);
#else
    memcpy(&s, &e, sizeof s);
#endif
    return TS_MUL(p, s);
}

// silu(g) * u, the routed experts' SwiGLU.
TS_FN float tuff_swiglu(float g, float u) {
    return TS_MUL(TS_DIV(g, TS_ADD(1.0f, tuff_expf(-g))), u);
}

#endif
