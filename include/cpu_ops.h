#ifndef CPU_OPS_H
#define CPU_OPS_H

#include "ggml_types.h"
#include "model.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dequantization routines
void dequantize_q8_0(const block_q8_0 *x, float *y, int64_t k);
void dequantize_q4_K(const block_q4_K *x, float *y, int64_t k);
void dequantize_q6_K(const block_q6_K *x, float *y, int64_t k);

// Generic row dequantizer using tensor GGML type
void dequantize_row(ggml_type_t type, const void *src, float *dst, int64_t k);

// Matrix-vector multiplication for quantized weights: dst = weight * src
// weight shape: (out_features, in_features)
//
// NOTE: this entry point keeps the strict scalar-fp32 semantics it always had -- it is the
// engine's exact reference (gpu.cu::gpu_selftest_q4k compares GPU kernels against it and
// asserts bit-equality). The fast AVX2 int8 path is NOT reached from here; use
// expert_mat_vec_cpu() below for the P1/P2 CPU expert path.
void mat_vec_cpu(float *dst, const tensor_t *weight, const float *src);

// ---------------------------------------------------------------------------
// P2: AVX2 int8 CPU expert kernels (Q8_K-quantized activation x Q4_K/Q6_K weights)
// ---------------------------------------------------------------------------

// Q8_K activation block -- llama.cpp's intermediate activation quantization type.
// Layout is byte-identical to ggml/src/ggml-common.h block_q8_K.
typedef struct {
    float   d;                // delta (= 1/iscale)
    int8_t  qs[QK_K];         // quants
    int16_t bsums[QK_K / 16]; // sums of quants in groups of 16
} block_q8_K;
GGML_STATIC_ASSERT(sizeof(block_q8_K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
                   "wrong q8_K block size/padding");

// Bytes needed to hold a `k`-element activation vector quantized to Q8_K. `k` must be a
// multiple of QK_K (256).
size_t q8_K_row_bytes(int64_t k);

// Quantize an fp32 activation vector to Q8_K. `dst` must hold q8_K_row_bytes(k) bytes.
// In the MoE path all experts of a layer share the same `norm_x`, so this is called ONCE
// per layer and the result reused across all selected experts.
void quantize_row_q8_K(const float *src, block_q8_K *dst, int64_t k);

// H22: f32 -> Q8_0 weight quantization (llama.cpp quantize_row_q8_0 numerics). Used to
// re-quantize the F32 router weights at load time behind TUFF_ROUTER_Q8.
void quantize_row_q8_0(const float *src, block_q8_0 *dst, int64_t k);

// dst[0..out_dim) = W * src, with W a quantized expert tensor of shape (in_dim, out_dim)
// stored row-major per output row (the GGUF expert layout).
//
//   type     : GGML_TYPE_Q4_K or GGML_TYPE_Q6_K (anything else -> exact scalar path)
//   src      : fp32 activation, in_dim elements. Only used by the exact path (and to
//              quantize internally when q8src == NULL).
//   q8src    : `src` pre-quantized with quantize_row_q8_K (may be NULL).
//   parallel : 1 -> parallelize over output rows with OpenMP; 0 -> fully serial
//              (use 0 when calling from a worker thread, to avoid nested-OMP oversubscription).
//
// Numerics: NOT bit-exact versus the scalar path -- the activation is quantized to 8 bits
// (see parity.md sec.3). Setting TUFF_EXACT=1 in the environment forces the original
// scalar fp32 kernels here instead (read once, cached).
void expert_mat_vec_cpu(float *dst, const void *w, ggml_type_t type,
                        int64_t in_dim, int64_t out_dim,
                        const float *src, const block_q8_K *q8src, int parallel);

// Fused Gate/Up + SwiGLU evaluation on CPU (Q4_K weights x Q8_K activation)
void cpu_expert_gateup_swiglu(float *act, const void *gw, const void *uw,
                              const float *src, const block_q8_K *q8x,
                              int64_t in_dim, int64_t ffn_dim);

// H10: row-range forms of the two kernels above, so one expert can be split across several
// worker threads instead of occupying one. Each output row is an independent dot product, so
// computing rows [r0, r1) on a different thread is **bit-identical** to computing them in the
// full-range call. AVX2 + q8_K activation only (the path the fast engine takes); they fail
// loudly rather than silently falling back, because a chunked call on the exact path would be
// a caller bug.
void expert_mat_vec_cpu_rows(float *dst, const void *w, ggml_type_t type,
                             int64_t in_dim, int64_t out_dim,
                             const block_q8_K *q8src, int64_t r0, int64_t r1);
void cpu_expert_gateup_swiglu_rows(float *act, const void *gw, const void *uw,
                                   const block_q8_K *q8x,
                                   int64_t in_dim, int64_t ffn_dim, int64_t r0, int64_t r1);

// H15: two-column (shared-miss) forms. A miss selected by both tokens is evaluated against
// both activations in one pass over the mmap'd weights: the row loop reads each weight row
// once and accumulates both columns' dots from it, so the DRAM weight stream is halved for
// the shared experts. Column c's output is bit-identical to the single-column call on that
// column alone (each output element is an independent dot product). The fallback (non-AVX2
// or exact mode) simply runs the single-column forms twice, which is correct if not faster.
void cpu_expert_gateup_swiglu_dual(float *act0, float *act1, const void *gw, const void *uw,
                                   const block_q8_K *q8x0, const block_q8_K *q8x1,
                                   int64_t in_dim, int64_t ffn_dim);
void expert_mat_vec_cpu_dual(float *dst0, float *dst1, const void *w, ggml_type_t type,
                             int64_t in_dim, int64_t out_dim,
                             const block_q8_K *q8x0, const block_q8_K *q8x1);
void expert_mat_vec_cpu_dual_rows(float *dst0, float *dst1, const void *w, ggml_type_t type,
                                  int64_t in_dim, int64_t out_dim,
                                  const block_q8_K *q8x0, const block_q8_K *q8x1,
                                  int64_t r0, int64_t r1);
void cpu_expert_gateup_swiglu_dual_rows(float *act0, float *act1, const void *gw, const void *uw,
                                        const block_q8_K *q8x0, const block_q8_K *q8x1,
                                        int64_t in_dim, int64_t ffn_dim, int64_t r0, int64_t r1);

// Convenience wrapper reading shape/type from a tensor_t (dims[0] = in, dims[1] = out).
void expert_mat_vec_cpu_t(float *dst, const tensor_t *weight,
                          const float *src, const block_q8_K *q8src, int parallel);

// Scalar-sequential Q6_K mat-vec (strict c-order accumulation, volatile sum), bit-identical
// to the single-thread CUDA kernel gpu.cu::mul_mat_vec_q6_K_scalar_kernel. Used for the
// routed down experts (which are Q6_K) so CPU-miss and GPU-hit produce identical output.
void mat_vec_q6_K_scalar(float *dst, const block_q6_K *x, const float *src, int64_t n_rows, int64_t in_dim);

// Activation and normalization operators
void rms_norm(float *dst, const float *src, const float *weight, int size, float eps);
void swiglu(float *dst, const float *gate, const float *up, int size);
void silu(float *dst, const float *src, int size);
void sigmoid(float *dst, const float *src, int size);
void softplus(float *dst, const float *src, int size);
void softmax(float *x, int size);
void l2_norm(float *x, int size, float eps);

// Rotary Position Embedding (Qwen3.5/3.6 interleaved mRoPE, GGML_ROPE_TYPE_IMROPE).
// Only the first `n_rot` of `head_dim` dimensions are rotated (NEOX-style pairing:
// (i, i+n_rot/2) for i in [0, n_rot/2)); dims [n_rot, head_dim) are left untouched.
// `sections` is rope.dimension_sections ([t,h,w,e] pair-count split); for a pure-text
// decode (no vision input) llama.cpp sets pos_t = pos_h = pos_w = pos and pos_e = 0
// (see llama-graph.cpp llm_graph_input_pos::set_input), which this function assumes.
void apply_mrope(float *q, float *k, int n_heads_q, int n_heads_k, int head_dim, int n_rot, int pos,
                const uint32_t sections[4], float freq_base);

#ifdef __cplusplus
}
#endif

#endif // CPU_OPS_H
