#ifndef LFM25_CPU_EXPERT_H
#define LFM25_CPU_EXPERT_H

// CPU evaluation of MoE experts whose slabs are not resident in the VRAM cache (hybrid miss path).
// Everything here reproduces, bit for bit, the per-row lane math of the GPU kernels (integer dp4a, fmaf chain, per-lane
// FADD accumulation in kbx order, xor-butterfly warp reduction), so a CPU-evaluated expert gives exactly the bytes the
// all-GPU path gives. SiLU (expf -> MUFU.EX2), the Q8_1 quantization and the weighted expert sum stay on the GPU.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Host mirror of block_q8_1_c (36 bytes): d (fp16), sum (fp16), 32 int8 quants.
typedef struct { uint16_t d; uint16_t s; int8_t qs[32]; } lfm25_q8blk_t;

// One row's dot product, exactly as the GPU lane math computes it (type: GGML_TYPE_Q4_K or GGML_TYPE_Q6_K).
// `row` points at nblk blocks of that type, `y` at nblk*8 Q8_1 blocks.
float lfm25_cpu_row_dot_scalar(int type, const void *row, const lfm25_q8blk_t *y, int nblk);
float lfm25_cpu_row_dot_avx2(int type, const void *row, const lfm25_q8blk_t *y, int nblk);

typedef struct lfm25_cpu_expert lfm25_cpu_expert_t;

// nthreads <= 0: LFM25_CPU_THREADS or the default. Returns NULL on failure.
lfm25_cpu_expert_t* lfm25_cpu_expert_create(int nthreads, int dim, int ffn_dim);
void lfm25_cpu_expert_free(lfm25_cpu_expert_t *ce);
int  lfm25_cpu_expert_threads(const lfm25_cpu_expert_t *ce);
// Forces the scalar reference kernels (test oracle).
void lfm25_cpu_expert_set_scalar(lfm25_cpu_expert_t *ce, int scalar);
// Pinned host copy target for the device Q8_1 of the layer input: dim/32*36 bytes.
void* lfm25_cpu_expert_host_xq8(lfm25_cpu_expert_t *ce);

// MoE FFN for one token where the experts in miss_mask (bit k = k-th selected expert) are evaluated on the CPU from the
// host weight slabs hg[k]/hu[k]/hd[k]; the others read device slots d_slot_ids[k] from the pools. Result is
// byte-identical to lfm25_gpu_moe_ffn on the same data. h_xq8: host copy of d_xq8 (the caller has synchronized).
// Launches on the stream set via lfm25_gpu_set_stream(stream) and synchronizes `stream` once (mid-FFN).
void lfm25_cpu_expert_moe_ffn(lfm25_cpu_expert_t *ce, cudaStream_t stream, const void *d_xq8, const void *h_xq8,
                              const int *d_slot_ids, const float *d_topk_w,
                              const void *d_gate, const void *d_up, const void *d_down, int down_type,
                              float *d_act, void *d_act_q8, float *d_dst, bool accum, int dim, int ffn_dim,
                              size_t gate_stride, size_t up_stride, size_t down_stride,
                              unsigned miss_mask, const void *const hg[4], const void *const hu[4], const void *const hd[4]);

#ifdef __cplusplus
}
#endif
#endif
