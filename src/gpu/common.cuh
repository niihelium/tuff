#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include "cuda_check.h"

__device__ __forceinline__ float h2f(uint16_t h) {
    __half raw = *reinterpret_cast<__half *>(&h);
    return __half2float(raw);
}

static __device__ __forceinline__ float warp_reduce_sum_f(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    return x;
}

static __device__ __forceinline__ float warp_reduce_max_f(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    return x;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit(void) {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait(void) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#else
__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem_src) {
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
}

__device__ __forceinline__ void cp_async_commit(void) {}

template <int N>
__device__ __forceinline__ void cp_async_wait(void) {}
#endif
