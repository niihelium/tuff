#ifndef QWEN_CPU_EXPERT_H
#define QWEN_CPU_EXPERT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "lfm25_cpu_expert.h"
#include "ggml_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen_cpu_expert qwen_cpu_expert_t;

// nthreads <= 0: uses default (e.g. 8-12 worker threads)
qwen_cpu_expert_t* qwen_cpu_expert_create(int nthreads, int dim, int ffn_dim);
void qwen_cpu_expert_free(qwen_cpu_expert_t *ce);
int  qwen_cpu_expert_threads(const qwen_cpu_expert_t *ce);

void* qwen_cpu_expert_host_xq8(qwen_cpu_expert_t *ce);
float* qwen_cpu_expert_host_pre_y(qwen_cpu_expert_t *ce);
float* qwen_cpu_expert_d_pre_gu(qwen_cpu_expert_t *ce);
float* qwen_cpu_expert_d_pre_y(qwen_cpu_expert_t *ce);


// Evaluates missed experts on CPU for gate/up row dots, uploads pre_gu to GPU,
// runs GPU SwiGLU + Q8_1 quantize for missed experts, downloads act_q8,
// evaluates missed down row dots on CPU, and uploads pre_y to GPU.
void qwen_cpu_expert_eval(qwen_cpu_expert_t *ce, cudaStream_t stream,
                          const void *d_xq8, int down_type,
                          int dim, int ffn_dim,
                          unsigned miss_mask,
                          const void *const hg[8], const void *const hu[8], const void *const hd[8]);

// Cache-invariant miss path: bit-identical to the GPU hit kernels (see qwen_cpu_expert.c).
// xq8 / aq8: Q8_1 blocks (36 B per 32 values), as the GPU produced / qwen_exact_quant_q8_1 makes.
void qwen_exact_gateup_rows(const void *gw, const void *uw, const void *xq8, int dim, float *act, int r0, int r1);
void qwen_exact_quant_q8_1(const float *x, void *y, int n);
void qwen_exact_down_rows(int type, const void *dw, const void *aq8, int ffn, float *y, int r0, int r1);

#ifdef __cplusplus
}
#endif
#endif
