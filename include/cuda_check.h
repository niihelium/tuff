#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "\n[FATAL CUDA ERROR] %s (%d) at %s:%d\n", \
                cudaGetErrorString(_err), _err, __FILE__, __LINE__); \
        fflush(stderr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "\n[FATAL CUDA KERNEL ERROR] %s (%d) at %s:%d\n", \
                cudaGetErrorString(_err), _err, __FILE__, __LINE__); \
        fflush(stderr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define ALIGN256(bytes) (((bytes) + 255) & ~((size_t)255))

#endif // CUDA_CHECK_H
