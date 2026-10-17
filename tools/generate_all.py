import re

with open('src/gpu.cu', 'r') as f:
    orig = f.read()
lines = orig.splitlines(keepends=True)

def find_line(pattern, start=0):
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    raise ValueError(f"Pattern '{pattern}' not found starting at {start}")

# 1. common.cuh
with open('src/gpu/common.cuh', 'w') as f:
    f.write("""#pragma once

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
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\\n"
                 :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit(void) {
    asm volatile("cp.async.commit_group;\\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait(void) {
    asm volatile("cp.async.wait_group %0;\\n" :: "n"(N));
}
#else
__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem_src) {
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
}

__device__ __forceinline__ void cp_async_commit(void) {}

template <int N>
__device__ __forceinline__ void cp_async_wait(void) {}
#endif
""")
print("1. common.cuh written")

# 2. mmvq.cuh
s_mmvq = find_line('#define QK8_1 32')
e_mmvq = find_line('static __device__ __forceinline__ float warp_reduce_sum_f', s_mmvq)
s_vd = find_line('// vec_dot_q4_K_q8_1', e_mmvq)
e_vd = find_line('// QK8_0 == QK8_1 == 32', s_vd)

with open('src/gpu/mmvq.cuh', 'w') as f:
    f.write("""#pragma once

#include "common.cuh"
#include "ggml_types.h"

#define MMVQ_ROWS_PER_BLOCK 4

""" + "".join(lines[s_mmvq:e_mmvq]) + "\n" + "".join(lines[s_vd:e_vd]))
print("2. mmvq.cuh written")

# 3. state.h and state.cu
state_h = """#pragma once

#include "gpu.h"
#include "cpu_ops.h"
#include "ggml_types.h"
#include "cuda_check.h"
#include "tuff_exact.h"
#include "moe_policy.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

#include "common.cuh"
#include "mmvq.cuh"

#define EXPERT_BATCH_MAX 8
#define GPU_IN_FLOATS   4096
#define GPU_OUT_FLOATS  262144

#ifdef __cplusplus
extern "C" {
#endif
extern moe_policy_t *g_moe_policy;
#ifdef __cplusplus
}
#endif

// Persistent scratch pointers
extern float *g_dx;
extern float *g_dy;

extern float *g_dstate_all;
extern size_t g_dstate_bytes;
extern float *g_dconv_all;
extern size_t g_dconv_bytes;

extern float *g_dpgate;
extern float *g_dq;
extern float *g_dk;
extern float *g_dv;
extern float *g_dout;
extern float *g_dp;

extern float *g_dpqkv;
extern float *g_dpz;
extern float *g_dpb;
extern float *g_dpa;
extern float *g_dgated;

extern float *g_shg;
extern float *g_shu;
extern float *g_shgi;

extern int    g_kv_quant_mode;
extern void  *g_dk_cache;
extern void  *g_dv_cache;
extern size_t g_kv_cache_bytes;
extern float *g_d_attn_scores;
extern size_t g_attn_scores_floats;
extern int    g_max_seq;

extern float *g_aq;
extern float *g_ak;
extern float *g_av;
extern float *g_aconc;

extern float *g_d_res_x;
extern float *g_d_norm_x;
extern int   *g_d_pos;

extern float *g_d_res_x2;
extern float *g_d_norm_x2;
extern void  *g_d_nx2_q8;
extern float *g_d_p2_qkv;
extern float *g_d_p2_z;
extern float *g_d_p2_b;
extern float *g_d_p2_a;
extern float *g_d_logits2;
extern int    g_n2_dim;
extern float *g_dstate_snap;
extern float *g_dconv_snap;

extern float   *g_d_res_batch;
extern float   *g_d_norm_batch;
extern void    *g_d_nx_batch_q8;
extern float   *g_d_p_batch_qkv;
extern float   *g_d_p_batch_z;
extern float   *g_d_p_batch_b;
extern float   *g_d_p_batch_a;
extern float   *g_d_logits_batch;
extern int32_t *g_d_best_token_batch;
extern int      g_batch_dim;
extern float   *g_dstate_snap_batch[8];
extern float   *g_dconv_snap_batch[8];
extern float   *g_d_shexp_batch;
extern float   *g_d_moe_buf_batch;

extern float *g_d_shexp_out2;
extern float *g_d_router_logits2;
extern float *g_d_router_logits;
extern cudaEvent_t g_router_d2h_event;
extern float *g_d_shexp_out;
extern float *g_d_shact;
extern float *g_d_shdown;

extern float *g_shg2;
extern float *g_shu2;
extern float *g_shgi2;
extern float *g_d_shact2;
extern float *g_d_shdown2;
extern float *g_d_miss_sum;
extern float *g_d_top_values;
extern int   *g_d_top_indices;
extern int   *g_d_hit_pos;
extern float *g_d_q_full;
extern float *g_d_k;
extern float *g_d_v;
extern float *g_d_q;
extern float *g_d_gate;
extern float *g_d_concat_out;
extern float *g_d_logits;
extern float *g_d_block_max;
extern int   *g_d_block_idx;
extern int   *g_d_best_token;
extern void  *g_d_out_normx_q8;
extern void  *g_d_shact_q8;
extern void  *g_d_dense_x_q8;
extern void  *g_d_norm_q8;
extern int    g_norm_q8_valid;
extern int    g_router_q8;

// Expert cache and execution buffers
extern char *g_pool;
extern size_t g_slot_bytes;
extern size_t g_tensor_off[3];
extern float *g_exgx;
extern float *g_exgy;
extern float *g_exgy2;
extern float *g_exdx;
extern float *g_exdy;
extern cudaStream_t g_expert_stream;
extern int g_expert_pipe_active;
extern int g_expert_down_active;
extern float *g_exgbat;
extern float *g_exubbat;
extern float *g_exdbat;
extern int *g_exslots;
extern size_t g_ex_stride_dx, g_ex_stride_dbat, g_ex_stride_slots;
extern size_t g_ex_stride_gxq8, g_ex_stride_dxq8;
extern int *g_d_exdst;
extern int g_batch_active;
extern void *g_exgx_q8;
extern void *g_exdx_q8;

// CUDA Graphs
extern cudaGraph_t     g_layer_graphs[64];
extern cudaGraphExec_t g_layer_execs[64];
extern int             g_layer_graphs_ready;
extern cudaStream_t    g_graph_stream;
extern unsigned long long g_graph_launches;
extern cudaGraph_t     g_layer_graphs2[64];
extern cudaGraphExec_t g_layer_execs2[64];
extern int             g_layer_graphs2_ready;

// MTP
extern float *g_d_mtp_concat;
extern float *g_d_mtp_x;
extern float *g_d_mtp_norm_x;
extern float *g_d_token_embd;

// Internal helpers & kernels shared between modules
int router_q8_enabled(void);
int tuff_h50(void);
int tuff_h49(void);
int mmvq_dual_enabled(void);

void launch_mat_vec_q8_0(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim);
void launch_mat_vec_q8_0_accum(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim);
void launch_mat_vec_q8_0_dual(const block_q8_0 *w0, float *y0, int out0,
                             const block_q8_0 *w1, float *y1, int out1,
                             const float *x, int in_dim);
void launch_mat_vec_f32_dual(const float *w0, float *y0, int out0,
                            const float *w1, float *y1, int out1,
                            const float *x, int in_dim);
void launch_mat_vec_f32(const float *w, const float *x, float *y, int in_dim, int out_dim);
void launch_mat_vec_q6_K(const block_q6_K *w, const float *x, float *y, int in_dim, int out_dim);
void launch_mat_vec_q4_K(const block_q4_K *w, const float *x, float *y, int in_dim, int out_dim, cudaStream_t stream = 0);
void launch_mat_vec_q6_K_scalar(const block_q6_K *w, const float *x, float *y, int in_dim, int out_dim, cudaStream_t stream = 0);
void launch_mul_mat_vec_q8_0_std(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim);
void launch_mul_mat_vec_f32_std(const float *w, const float *x, float *y, int in_dim, int out_dim);

block_q8_1_c *dense_quant_buf_ensure(int in_dim);
block_q8_1_c *norm_x_q8_ensure(int in_dim);
block_q8_1_c *dense_quantize_x(const float *x, int in_dim);
block_q8_1_c *dense_quantize_x_to(const float *x, int in_dim, void *dst);
void mmvq_q8_0_rows(const block_q8_0 *w, const block_q8_1_c *xq, float *y, int in_dim, int out_dim, bool accum = false);

void launch_rms_norm_dev(float *dst, const float *src, const float *weight, int dim, float eps, cudaStream_t stream = 0);
void launch_quantize_q8_1(const float *x, block_q8_1_c *y, int n, cudaStream_t stream = 0);
void launch_mmvq_q8_0_dense_ncols(const block_q8_0 *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols, cudaStream_t stream = 0);
void launch_mmvq_q6_K_dense_ncols(const block_q6_K *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols, cudaStream_t stream = 0);

void launch_mmvq_delta_quad(const block_q8_0 *qkv_w, float *g_dpqkv, int out_qkv,
                           const block_q8_0 *z_w, float *g_dpz, int out_z,
                           const float *beta_w, float *g_dpb, int out_beta,
                           const float *alpha_w, float *g_dpa, int out_alpha,
                           const block_q8_1_c *xq, const float *d_norm_x, int nblk, int in_dim);

void launch_mmvq_dense_triple(const block_q8_0 *q_dev, float *d_q_full, int q_full_dim,
                             const block_q8_0 *k_dev, float *d_k, int k_dim,
                             const block_q8_0 *v_dev, float *d_v, int v_dim,
                             const block_q8_1_c *xq, int nblk);

void launch_mmvq_shexp_swiglu_quant(
    const block_q8_0 *w_gate, const block_q8_0 *w_up,
    block_q8_1_c *dst_q8, int sh_ffn_dim,
    const block_q8_1_c *xq, int nblk,
    cudaStream_t stream = 0);

void kv_cache_ensure(int n_attn_layers, int n_head_kv, int max_seq, int head_dim);
void attn_scores_ensure(int n_head, int max_seq);

void launch_attention_kernel(
    void *k_cache, void *v_cache, const float *q, const float *k, const float *v,
    const float *gate, float *concat_out, float *scores_buf,
    int n_head, int n_head_kv, int max_seq, int head_dim,
    int layer_idx, int pos, int ctx_len, float scale,
    block_q8_1_c *out_q8, const int *d_pos);

void launch_fused_qk_split_norm_rope(
    float *d_q, float *d_gate, const float *d_q_full, const float *d_q_norm_w, int n_head,
    float *d_k_out, const float *d_k_in, const float *d_k_norm_w, int n_head_kv,
    int head_dim, float eps, int rope_dim_count, int pos,
    int sec0, int sec1, int sec2, int sec3, float rope_freq_base, const int *d_pos,
    cudaStream_t stream = 0);

void gpu_delta_fused_gdn_dev_p(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                               float *qkv, const float *z, const float *beta_raw, const float *alpha_raw,
                               int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                               int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);

void launch_topk_moe_resident(const float *logits, float *weights, int *exslots, int layer_idx, cudaStream_t stream = 0);

void launch_swiglu_ncols(float *act, const float *gate, const float *up, int count, cudaStream_t stream = 0);
void launch_scale_shexp(float *out, const float *down, const float *gate, int dim, cudaStream_t stream = 0);
void launch_argmax(const float *x, float *block_max, int *block_idx, int *best_token, int vocab_size, cudaStream_t stream = 0);

block_q8_1_c *ex_col_dxq8(int col);
float *ex_col_dx(int col);
float *ex_col_dbat(int col);
int   *ex_col_slots(int col);
const block_q8_1_c *ex_col_xq8(int col);

// Cross-module exported functions
extern "C" void gpu_residual_accum(float *d_dst, const float *d_src, int count);
extern "C" void gpu_fused_rms_norm_quant_dev(float *dst_norm_x, const float *src_x, const float *w, void *dst_q8, int in_dim, float eps);
extern "C" void gpu_rms_norm_dev(float *dst, const float *src, const float *weight, int dim, float eps);
extern "C" void gpu_delta_projections_dev(const tensor_t *attn_qkv, const tensor_t *attn_gate, const tensor_t *ssm_beta, const tensor_t *ssm_alpha, const float *d_norm_x, int in_dim);
extern "C" void gpu_delta_fused_gdn_dev(int layer_idx, const model_layer_t *layer, float *d_res_x, float eps, float scale_q, int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim, int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);
extern "C" void gpu_router_logits_dev_only(const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int expert_count);
extern "C" void gpu_reset_delta_state(void);
extern "C" void gpu_reset_kv(void);
extern "C" const float *gpu_get_shexp2_dev_ptr(void);
extern "C" void gpu_rms_norm_dev_n2(float *d_dst2, const float *d_src2, const float *d_weight, int dim, float eps);
extern "C" void gpu_delta_projections_n2(const tensor_t *attn_qkv, const tensor_t *attn_gate, const tensor_t *ssm_beta, const tensor_t *ssm_alpha, const float *d_norm_x2, int in_dim);
extern "C" void gpu_delta_fused_gdn_dev_col(int col, int layer_idx, const model_layer_t *layer, float *d_res_x, float eps, float scale_q, int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim, int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);
extern "C" void gpu_delta_snapshot_layer_n2(int layer_idx, int n_v_heads, int head_v_dim, int head_k_dim, int conv_kernel, int qkv_dim);
extern "C" void gpu_router_logits_dev_only_col(const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int expert_count, int col);
"""

with open('src/gpu/state.h', 'w') as f:
    f.write(state_h)

state_cu = """#include "state.h"

extern "C" moe_policy_t *g_moe_policy = NULL;

float *g_dx = NULL;
float *g_dy = NULL;

float *g_dstate_all = NULL;
size_t g_dstate_bytes = 0;
float *g_dconv_all = NULL;
size_t g_dconv_bytes = 0;

float *g_dpgate = NULL;
float *g_dq = NULL;
float *g_dk = NULL;
float *g_dv = NULL;
float *g_dout = NULL;
float *g_dp = NULL;

float *g_dpqkv = NULL;
float *g_dpz = NULL;
float *g_dpb = NULL;
float *g_dpa = NULL;
float *g_dgated = NULL;

float *g_shg = NULL;
float *g_shu = NULL;
float *g_shgi = NULL;

int    g_kv_quant_mode = KV_QUANT_Q8_0;
void  *g_dk_cache = NULL;
void  *g_dv_cache = NULL;
size_t g_kv_cache_bytes = 0;
float *g_d_attn_scores = NULL;
size_t g_attn_scores_floats = 0;
int    g_max_seq = 131072;

float *g_aq = NULL;
float *g_ak = NULL;
float *g_av = NULL;
float *g_aconc = NULL;

float *g_d_res_x = NULL;
float *g_d_norm_x = NULL;
int   *g_d_pos    = NULL;

float *g_d_res_x2  = NULL;
float *g_d_norm_x2 = NULL;
void  *g_d_nx2_q8  = NULL;
float *g_d_p2_qkv  = NULL;
float *g_d_p2_z    = NULL;
float *g_d_p2_b    = NULL;
float *g_d_p2_a    = NULL;
float *g_d_logits2 = NULL;
int    g_n2_dim    = 0;
float *g_dstate_snap = NULL;
float *g_dconv_snap  = NULL;

float   *g_d_res_batch = NULL;
float   *g_d_norm_batch = NULL;
void    *g_d_nx_batch_q8 = NULL;
float   *g_d_p_batch_qkv = NULL;
float   *g_d_p_batch_z = NULL;
float   *g_d_p_batch_b = NULL;
float   *g_d_p_batch_a = NULL;
float   *g_d_logits_batch = NULL;
int32_t *g_d_best_token_batch = NULL;
int      g_batch_dim = 0;
float   *g_dstate_snap_batch[8] = {NULL};
float   *g_dconv_snap_batch[8]  = {NULL};
float   *g_d_shexp_batch = NULL;
float   *g_d_moe_buf_batch = NULL;

float *g_d_shexp_out2 = NULL;
float *g_d_router_logits2 = NULL;
float *g_d_router_logits = NULL;
cudaEvent_t g_router_d2h_event = 0;
float *g_d_shexp_out = NULL;
float *g_d_shact = NULL;
float *g_d_shdown = NULL;

float *g_shg2 = NULL;
float *g_shu2 = NULL;
float *g_shgi2 = NULL;
float *g_d_shact2 = NULL;
float *g_d_shdown2 = NULL;
float *g_d_miss_sum = NULL;
float *g_d_top_values = NULL;
int   *g_d_top_indices = NULL;
int   *g_d_hit_pos = NULL;
float *g_d_q_full = NULL;
float *g_d_k = NULL;
float *g_d_v = NULL;
float *g_d_q = NULL;
float *g_d_gate = NULL;
float *g_d_concat_out = NULL;
float *g_d_logits = NULL;
float *g_d_block_max = NULL;
int   *g_d_block_idx = NULL;
int   *g_d_best_token = NULL;
void  *g_d_out_normx_q8 = NULL;
void  *g_d_shact_q8 = NULL;
void  *g_d_dense_x_q8 = NULL;
void  *g_d_norm_q8 = NULL;
int    g_norm_q8_valid = 0;
int    g_router_q8 = -1;

char *g_pool = NULL;
size_t g_slot_bytes = 0;
size_t g_tensor_off[3] = {0,0,0};

float *g_exgx = NULL;
float *g_exgy = NULL;
float *g_exgy2 = NULL;
float *g_exdx = NULL;
float *g_exdy = NULL;
cudaStream_t g_expert_stream = 0;
int g_expert_pipe_active = 0;
int g_expert_down_active = 0;
float *g_exgbat = NULL;
float *g_exubbat = NULL;
float *g_exdbat = NULL;
int *g_exslots = NULL;
size_t g_ex_stride_dx = 0, g_ex_stride_dbat = 0, g_ex_stride_slots = 0;
size_t g_ex_stride_gxq8 = 0, g_ex_stride_dxq8 = 0;
int *g_d_exdst = NULL;
int g_batch_active = 0;
void *g_exgx_q8 = NULL;
void *g_exdx_q8 = NULL;

cudaGraph_t     g_layer_graphs[64];
cudaGraphExec_t g_layer_execs[64];
int             g_layer_graphs_ready = 0;
cudaStream_t    g_graph_stream = NULL;
unsigned long long g_graph_launches = 0;
cudaGraph_t     g_layer_graphs2[64];
cudaGraphExec_t g_layer_execs2[64];
int             g_layer_graphs2_ready = 0;

float *g_d_mtp_concat = NULL;
float *g_d_mtp_x = NULL;
float *g_d_mtp_norm_x = NULL;
float *g_d_token_embd = NULL;

int router_q8_enabled(void) {
    if (g_router_q8 < 0) { const char *e = getenv("TUFF_ROUTER_Q8"); g_router_q8 = (e && e[0] == '1') ? 1 : 0; }
    return g_router_q8;
}

extern "C" void gpu_set_kv_quant(int mode) { g_kv_quant_mode = mode; }
extern "C" int  gpu_get_kv_quant(void)     { return g_kv_quant_mode; }
"""

with open('src/gpu/state.cu', 'w') as f:
    f.write(state_cu)
print("3. state.h and state.cu written")

# 4. expert_cache.cu
s_ec = find_line('#define EXPERT_CACHE_DEFAULT_SLOTS')
e_ec = find_line('__device__ __forceinline__ float h2f', s_ec)
ec_text = "".join(lines[s_ec:e_ec])
ec_text = re.sub(r'static char \*g_pool = NULL;[\s\S]*?static size_t g_tensor_off\[3\] = \{0,0,0\};[^\n]*\n', '', ec_text)
ec_text = re.sub(r'static float \*g_exgx = NULL;[\s\S]*?static void \*g_exdx_q8 = NULL;\n', '', ec_text)

with open('src/gpu/expert_cache.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + ec_text)
print("4. expert_cache.cu written")

# 5. dense_gemv.cu
s_dg1a = find_line('__global__ void mul_mat_vec_q8_0_kernel')
e_dg1a = find_line('__global__ void delta_state_update_kernel')
s_dg1b = find_line('__global__ void mul_mat_vec_f32_kernel')
e_dg1b = find_line('static void upload_dense', s_dg1b)

s_dg2 = find_line('void gpu_mat_vec(float *dst')
e_dg2 = find_line('void gpu_delta_projections(', s_dg2)
s_dg3 = find_line('__global__ void mul_mat_vec_q6_K_scalar_kernel')
e_dg3 = find_line('void gpu_expert_mat_vec_q6(', s_dg3)
s_q = find_line('__global__ void quantize_q8_1_kernel')
e_q = find_line('// vec_dot_q4_K_q8_1', s_q)

# Split out mmvq_q6_K_dense_argmax_warprow_kernel (lines 2783 to 2835) so it goes to output.cu
s_mm = find_line('// QK8_0 == QK8_1 == 32')
s_argmax = find_line('__global__ void mmvq_q6_K_dense_argmax_warprow_kernel(') - 2
e_argmax = find_line('// H7 stage 2a: the output head over `ncols` activation columns.')
e_mm = find_line('// One CUDA block computes ONE output row of ONE expert', s_mm)

argmax_kernel_code = "".join(lines[s_argmax:e_argmax])

s_sh = find_line('static void shared_proj_launch')
e_sh = find_line('template <int HEAD_DIM, int TILE_SIZE>', s_sh)
s_bench = find_line('extern "C" void gpu_bench_dense_batch(')
e_bench = find_line('extern "C" void gpu_router_logits_copy_out(', s_bench)

dg_text = (''.join(lines[s_dg1a:e_dg1a]) + '\n' +
           ''.join(lines[s_dg1b:e_dg1b]) + '\n' +
           ''.join(lines[s_dg2:e_dg2]) + '\n' +
           ''.join(lines[s_dg3:e_dg3]) + '\n' +
           ''.join(lines[s_q:e_q]) + '\n' +
           ''.join(lines[s_mm:s_argmax]) + '\n' +
           ''.join(lines[e_argmax:e_mm]) + '\n' +
           ''.join(lines[s_sh:e_sh]) + '\n' +
           ''.join(lines[s_bench:e_bench]))

funcs_to_unstatic = [
    'static void launch_mat_vec_q8_0(',
    'static void launch_mat_vec_q8_0_accum(',
    'static void launch_mat_vec_q8_0_dual(',
    'static void launch_mat_vec_f32_dual(',
    'static void launch_mat_vec_f32(',
    'static void launch_mat_vec_q6_K(',
    'static block_q8_1_c *dense_quant_buf_ensure(',
    'static block_q8_1_c *norm_x_q8_ensure(',
    'static block_q8_1_c *dense_quantize_x(',
    'static block_q8_1_c *dense_quantize_x_to(',
    'static void mmvq_q8_0_rows(',
    'static int tuff_h50(void)',
    'static int tuff_h49(void)',
]
for fn in funcs_to_unstatic:
    dg_text = dg_text.replace(fn, fn.replace('static ', ''))

extra_launchers_dg = """
void launch_quantize_q8_1(const float *x, block_q8_1_c *y, int n, cudaStream_t stream) {
    quantize_q8_1_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
    CUDA_CHECK_LAST();
}

void launch_mat_vec_q4_K(const block_q4_K *w, const float *x, float *y, int in_dim, int out_dim, cudaStream_t stream) {
    mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, stream>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mat_vec_q6_K_scalar(const block_q6_K *w, const float *x, float *y, int in_dim, int out_dim, cudaStream_t stream) {
    mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1, 0, stream>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mul_mat_vec_q8_0_std(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim) {
    mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mul_mat_vec_f32_std(const float *w, const float *x, float *y, int in_dim, int out_dim) {
    mul_mat_vec_f32_kernel<<<out_dim, 256>>>(w, x, y, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mmvq_shexp_swiglu_quant(
    const block_q8_0 *w_gate, const block_q8_0 *w_up,
    block_q8_1_c *dst_q8, int sh_ffn_dim,
    const block_q8_1_c *xq, int nblk,
    cudaStream_t stream) {
    dim3 grid((unsigned)((sh_ffn_dim + 31) / 32), 1u, 1u);
    dim3 block(32u, 32u, 1u);
    mmvq_q8_0_dense_shexp_swiglu_quant_kernel<<<grid, block, 0, stream>>>(
        w_gate, w_up, dst_q8, sh_ffn_dim, xq, nblk);
    CUDA_CHECK_LAST();
}

void launch_mmvq_q8_0_dense_ncols(const block_q8_0 *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols, cudaStream_t stream) {
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    switch (ncols) {
        case 1: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 2: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 3: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 3><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 4: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 5: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 5><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 6: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 6><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 7: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 7><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 8: mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 8><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        default: break;
    }
    CUDA_CHECK_LAST();
}

void launch_mmvq_q6_K_dense_ncols(const block_q6_K *w, const block_q8_1_c *xq, float *dst, int out_dim, int nblk, int ncols, cudaStream_t stream) {
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    switch (ncols) {
        case 1: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 2: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 3: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 3><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 4: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 5: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 5><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 6: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 6><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 7: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 7><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        case 8: mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 8><<<grid, block, 0, stream>>>(w, xq, dst, out_dim, nblk); break;
        default: break;
    }
    CUDA_CHECK_LAST();
}

void launch_mmvq_delta_quad(const block_q8_0 *qkv_w, float *g_dpqkv, int out_qkv,
                           const block_q8_0 *z_w, float *g_dpz, int out_z,
                           const float *beta_w, float *g_dpb, int out_beta,
                           const float *alpha_w, float *g_dpa, int out_alpha,
                           const block_q8_1_c *xq, const float *d_norm_x, int nblk, int in_dim) {
    const int total_out = out_qkv + out_z + out_beta + out_alpha;
    dim3 grid((unsigned)((total_out + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_delta_quad_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block>>>(
        qkv_w, g_dpqkv, out_qkv,
        z_w, g_dpz, out_z,
        beta_w, g_dpb, out_beta,
        alpha_w, g_dpa, out_alpha,
        xq, d_norm_x, nblk, in_dim);
    CUDA_CHECK_LAST();
}

void launch_mmvq_dense_triple(const block_q8_0 *q_dev, float *d_q_full, int q_full_dim,
                             const block_q8_0 *k_dev, float *d_k, int k_dim,
                             const block_q8_0 *v_dev, float *d_v, int v_dim,
                             const block_q8_1_c *xq, int nblk) {
    const int total_qkv = q_full_dim + k_dim + v_dim;
    dim3 qkv_grid((unsigned)((total_qkv + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 qkv_block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>(
        q_dev, d_q_full, q_full_dim,
        k_dev, d_k, k_dim,
        v_dev, d_v, v_dim,
        xq, nblk);
    CUDA_CHECK_LAST();
}
"""

with open('src/gpu/dense_gemv.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + dg_text + extra_launchers_dg)
print("5. dense_gemv.cu written")

# 6. delta.cu
s_d1 = find_line('__global__ void delta_state_update_kernel(')
e_d1 = find_line('__global__ void mul_mat_vec_f32_kernel', s_d1)
s_d2 = find_line('void gpu_reset_delta_state(void)')
e_d2 = find_line('void gpu_shutdown(model_t *m)', s_d2)
s_d3 = find_line('void gpu_delta_projections(float *qkv_mixed')
e_d3 = find_line('void gpu_expert_mat_vec(', s_d3)
s_d4 = find_line('__global__ void delta_conv_kernel(')
e_d4 = find_line('static void shared_proj_launch', s_d4)
s_d5 = find_line('void gpu_residual_reset(const float *host_x')
e_d5 = find_line('__global__ void fused_q_split_norm_rope_kernel(', s_d5)

delta_text = (''.join(lines[s_d1:e_d1]) + '\n' +
              ''.join(lines[s_d2:e_d2]) + '\n' +
              ''.join(lines[s_d3:e_d3]) + '\n' +
              ''.join(lines[s_d4:e_d4]) + '\n' +
              ''.join(lines[s_d5:e_d5]))

delta_text = re.sub(r'mmvq_q8_0_dense_delta_quad_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block>>>\([\s\S]*?\);\s*CUDA_CHECK_LAST\(\);',
                    r'launch_mmvq_delta_quad((const block_q8_0*)attn_qkv->dev, g_dpqkv, out_qkv, (const block_q8_0*)attn_gate->dev, g_dpz, out_z, (const float*)ssm_beta->dev, g_dpb, out_beta, (const float*)ssm_alpha->dev, g_dpa, out_alpha, xq, d_norm_x, nblk, in_dim);',
                    delta_text)
delta_text = re.sub(r'mul_mat_vec_q8_0_kernel\s*<<<\s*([^,]+),\s*256\s*>>>\s*\(\s*\(const block_q8_0\*\)([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\);',
                    r'launch_mul_mat_vec_q8_0_std((const block_q8_0*)\2, \3, \4, \5, \1);', delta_text)
delta_text = re.sub(r'mul_mat_vec_f32_kernel\s*<<<\s*([^,]+),\s*256\s*>>>\s*\(\s*\(const float\*\)([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\);',
                    r'launch_mul_mat_vec_f32_std((const float*)\2, \3, \4, \5, \1);', delta_text)

extra_launchers_delta = """
void launch_rms_norm_dev(float *dst, const float *src, const float *weight, int dim, float eps, cudaStream_t stream) {
    rms_norm_dev_kernel<<<1, 256, 0, stream>>>(dst, src, weight, dim, eps);
    CUDA_CHECK_LAST();
}
"""

with open('src/gpu/delta.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + delta_text + extra_launchers_delta)
print("6. delta.cu written")

# 7. attention.cu
s_a1 = find_line('template <int HEAD_DIM, int TILE_SIZE>', s_sh)
e_a1 = find_line('double gpu_vram_used_mb(void)', s_a1)
s_a2 = find_line('__global__ void fused_q_split_norm_rope_kernel(')
e_a2 = find_line('void gpu_router_logits_launch(', s_a2)

att_text = ''.join(lines[s_a1:e_a1]) + '\n' + ''.join(lines[s_a2:e_a2])
att_text = re.sub(r'#if defined\(__CUDA_ARCH__\) && __CUDA_ARCH__ >= 800[\s\S]*?#endif\n', '', att_text)
att_text = re.sub(r'dim3 qkv_grid\([\s\S]*?\);\s*dim3 qkv_block\([\s\S]*?\);\s*mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>\([\s\S]*?\);\s*CUDA_CHECK_LAST\(\);',
                  r'launch_mmvq_dense_triple((const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim, (const block_q8_0*)layer->attn_k.dev, g_d_k, (int)k_dim, (const block_q8_0*)layer->attn_v.dev, g_d_v, (int)v_dim, xq, nblk);',
                  att_text)
att_text = att_text.replace('static void kv_cache_ensure(', 'void kv_cache_ensure(')
att_text = att_text.replace('static void attn_scores_ensure(', 'void attn_scores_ensure(')
att_text = att_text.replace('static void launch_attention_kernel(', 'void launch_attention_kernel(')

extra_launchers_att = """
void launch_fused_qk_split_norm_rope(
    float *d_q, float *d_gate, const float *d_q_full, const float *d_q_norm_w, int n_head,
    float *d_k_out, const float *d_k_in, const float *d_k_norm_w, int n_head_kv,
    int head_dim, float eps, int rope_dim_count, int pos,
    int sec0, int sec1, int sec2, int sec3, float rope_freq_base, const int *d_pos,
    cudaStream_t stream) {
    fused_qk_split_norm_rope_kernel<<<n_head + n_head_kv, head_dim, 0, stream>>>(
        d_q, d_gate, d_q_full, d_q_norm_w, n_head,
        d_k_out, d_k_in, d_k_norm_w, n_head_kv,
        head_dim, eps, rope_dim_count, pos,
        sec0, sec1, sec2, sec3, rope_freq_base, d_pos);
    CUDA_CHECK_LAST();
}
"""

with open('src/gpu/attention.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + att_text + extra_launchers_att)
print("7. attention.cu written")

# 8. router.cu
s_r1 = find_line('void gpu_router_logits_launch(')
e_r1 = find_line('extern "C" void gpu_bench_dense_batch(', s_r1)
s_r2 = find_line('extern "C" void gpu_router_logits_copy_out(', e_r1)
e_r2 = find_line('__global__ void swiglu_dev_kernel(', s_r2)
s_r3 = find_line('template <int experts_per_thread>')
e_r3 = find_line('__global__ void moe_combine_resident_kernel(', s_r3)

topk_resident_launcher = """
void launch_topk_moe_resident(const float *logits, float *weights, int *exslots, int layer_idx, cudaStream_t stream) {
    topk_moe_resident_kernel<256, 8><<<1, 32, 0, stream>>>(logits, weights, exslots, layer_idx);
    CUDA_CHECK_LAST();
}
"""

with open('src/gpu/router.cu', 'w') as f:
    f.write('#include "state.h"\n\n' +
            ''.join(lines[s_r1:e_r1]) + '\n' +
            ''.join(lines[s_r2:e_r2]) + '\n' +
            ''.join(lines[s_r3:e_r3]) + '\n' +
            topk_resident_launcher)
print("8. router.cu written")

# 9. moe.cu
s_m1 = find_line('void gpu_expert_mat_vec(')
e_m1 = find_line('__global__ void mul_mat_vec_q6_K_scalar_kernel', s_m1)
s_m2 = find_line('void gpu_expert_mat_vec_q6(')
e_m2 = find_line('#define QK8_1 32', s_m2)
s_m3 = find_line('// One CUDA block computes ONE output row of ONE expert')
e_m3 = find_line('__global__ void delta_conv_kernel', s_m3)
s_m4 = find_line('__global__ void swiglu_dev_kernel(')
e_m4 = find_line('template <int experts_per_thread>', s_m4)
s_m5 = find_line('__global__ void moe_combine_resident_kernel(')
e_m5 = find_line('__global__ void argmax_block_kernel(', s_m5)

moe_text = (''.join(lines[s_m1:e_m1]) + '\n' +
            ''.join(lines[s_m2:e_m2]) + '\n' +
            ''.join(lines[s_m3:e_m3]) + '\n' +
            ''.join(lines[s_m4:e_m4]) + '\n' +
            ''.join(lines[s_m5:e_m5]))

moe_text = moe_text.replace('topk_moe_resident_kernel<256, 8><<<1, 32>>>(g_d_router_logits, g_d_top_values, g_exslots, layer_idx);',
                            'launch_topk_moe_resident(g_d_router_logits, g_d_top_values, g_exslots, layer_idx);')
moe_text = moe_text.replace('static const block_q8_1_c *ex_col_xq8(int col)', 'const block_q8_1_c *ex_col_xq8(int col)')
moe_text = moe_text.replace('static float *ex_col_dx(int col)', 'float *ex_col_dx(int col)')
moe_text = moe_text.replace('static float *ex_col_dbat(int col)', 'float *ex_col_dbat(int col)')
moe_text = moe_text.replace('static int   *ex_col_slots(int col)', 'int *ex_col_slots(int col)')
moe_text = moe_text.replace('static block_q8_1_c *ex_col_dxq8(int col)', 'block_q8_1_c *ex_col_dxq8(int col)')

moe_text = re.sub(r'quantize_q8_1_kernel\s*<<<\s*[^,]+,\s*256,\s*0,\s*([^>]+)>>>\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\);',
                  r'launch_quantize_q8_1(\2, \3, \4, \1);', moe_text)
moe_text = re.sub(r'quantize_q8_1_kernel\s*<<<\s*[^,]+,\s*256>>>\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\);',
                  r'launch_quantize_q8_1(\1, \2, \3);', moe_text)

moe_text = moe_text.replace(
    'mul_mat_vec_q4_K_kernel<<<out_dim, 1>>>((const block_q4_K*)dev_w, g_dx, g_dy, in_dim);',
    'launch_mat_vec_q4_K((const block_q4_K*)dev_w, g_dx, g_dy, in_dim, out_dim);')
moe_text = moe_text.replace(
    'mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1>>>((const block_q6_K*)dev_w, g_dx, g_dy, in_dim);',
    'launch_mat_vec_q6_K_scalar((const block_q6_K*)dev_w, g_dx, g_dy, in_dim, out_dim);')
moe_text = moe_text.replace(
    'mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q4_K*)gpu_expert_dev_ptr(slot, 0), g_exgx, g_exgy, in_dim);',
    'launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 0), g_exgx, g_exgy, in_dim, out_dim, g_expert_stream);')
moe_text = moe_text.replace(
    'mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q4_K*)gpu_expert_dev_ptr(slot, 1), g_exgx, g_exgy2, in_dim);',
    'launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 1), g_exgx, g_exgy2, in_dim, out_dim, g_expert_stream);')
moe_text = re.sub(
    r'mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1, 0, g_expert_stream>>>\(\s*\(const block_q6_K\*\)gpu_expert_dev_ptr\(slot, 2\),\s*g_exdx,\s*g_exdy,\s*in_dim\);',
    r'launch_mat_vec_q6_K_scalar((const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, g_exdy, in_dim, out_dim, g_expert_stream);', moe_text)
moe_text = re.sub(
    r'mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>\(\s*\(const block_q4_K\*\)gpu_expert_dev_ptr\(slot, 2\),\s*g_exdx,\s*g_exdy,\s*in_dim\);',
    r'launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, g_exdy, in_dim, out_dim, g_expert_stream);', moe_text)
moe_text = re.sub(
    r'mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>\(\s*\(const block_q4_K\*\)gpu_expert_dev_ptr\(slots\[k\], 0\),\s*g_exgx,\s*g_exgbat \+ \(size_t\)k \* out_dim,\s*in_dim\);',
    r'launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slots[k], 0), g_exgx, g_exgbat + (size_t)k * out_dim, in_dim, out_dim, g_expert_stream);', moe_text)
moe_text = re.sub(
    r'mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>\(\s*\(const block_q4_K\*\)gpu_expert_dev_ptr\(slots\[k\], 1\),\s*g_exgx,\s*g_exubbat \+ \(size_t\)k \* out_dim,\s*in_dim\);',
    r'launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slots[k], 1), g_exgx, g_exubbat + (size_t)k * out_dim, in_dim, out_dim, g_expert_stream);', moe_text)
moe_text = moe_text.replace(
    'mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim);',
    'launch_mat_vec_q6_K_scalar((const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim, out_dim, g_expert_stream);')
moe_text = moe_text.replace(
    'mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim);',
    'launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim, out_dim, g_expert_stream);')

moe_text = re.sub(r'mmvq_q8_0_dense_shexp_swiglu_quant_kernel<<<grid, block>>>\([\s\S]*?\);\s*CUDA_CHECK_LAST\(\);',
                  r'''launch_mmvq_shexp_swiglu_quant(
            (const block_q8_0*)layer->ffn_gate_shexp.dev,
            (const block_q8_0*)layer->ffn_up_shexp.dev,
            (block_q8_1_c*)g_d_shact_q8,
            sh_ffn_dim, xq, nblk);''', moe_text)

extra_launchers_moe = """
void launch_swiglu_ncols(float *act, const float *gate, const float *up, int count, cudaStream_t stream) {
    swiglu_dev_ncols_kernel<<<(count + 255) / 256, 256, 0, stream>>>(act, gate, up, count);
    CUDA_CHECK_LAST();
}

void launch_scale_shexp(float *out, const float *down, const float *gate, int dim, cudaStream_t stream) {
    scale_shexp_kernel<<<(dim + 255) / 256, 256, 0, stream>>>(out, down, gate, dim);
    CUDA_CHECK_LAST();
}
"""

with open('src/gpu/moe.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + moe_text + extra_launchers_moe)
print("9. moe.cu written")

# 10. output.cu
s_o = find_line('__global__ void argmax_block_kernel(')
e_o = find_line('extern "C" void gpu_n2_prepare(', s_o)

output_text = "".join(lines[s_o:e_o])
output_text = output_text.replace(
    'quantize_q8_1_kernel<<<(dim + 255) / 256, 256>>>(d_dst_norm_x, (block_q8_1_c *)d_dst_q8, dim);',
    'launch_quantize_q8_1(d_dst_norm_x, (block_q8_1_c *)d_dst_q8, dim);')

extra_launchers_output = """
void launch_argmax(const float *x, float *block_max, int *block_idx, int *best_token, int vocab_size, cudaStream_t stream) {
    argmax_block_kernel<<<256, 256, 0, stream>>>(x, block_max, block_idx, vocab_size);
    argmax_final_kernel<<<1, 256, 0, stream>>>(block_max, block_idx, best_token, 256);
    CUDA_CHECK_LAST();
}
"""

with open('src/gpu/output.cu', 'w') as f:
    f.write('#include "state.h"\n\n' +
            argmax_kernel_code + '\n' +
            output_text + '\n' +
            extra_launchers_output)
print("10. output.cu written")

# 11. verify_n2.cu
s_v = find_line('extern "C" void gpu_n2_prepare(')
e_v = find_line('extern "C" float *gpu_get_d_res_batch(', s_v)

v_text = "".join(lines[s_v:e_v])
v_text = v_text.replace(
    'rms_norm_dev_kernel<<<1, 256>>>(d_dst2,                d_src2,                d_weight, dim, eps);',
    'launch_rms_norm_dev(d_dst2, d_src2, d_weight, dim, eps);')
v_text = v_text.replace(
    'rms_norm_dev_kernel<<<1, 256>>>(d_dst2 + (size_t)dim,  d_src2 + (size_t)dim,  d_weight, dim, eps);',
    'launch_rms_norm_dev(d_dst2 + (size_t)dim, d_src2 + (size_t)dim, d_weight, dim, eps);')

v_text = re.sub(
    r'quantize_q8_1_kernel<<<\([^)]+\) / 256, 256>>>\(([^,]+),\s*([^,]+),\s*([^)]+)\);',
    r'launch_quantize_q8_1(\1, \2, \3);', v_text)

v_text = re.sub(r'static void n2_dense_q8_0\([\s\S]*?CUDA_CHECK_LAST\(\);\s*}',
                r'''static void n2_dense_q8_0(const void *dev_w, float *d_dst2, int in_dim, int out_dim) {
    const int nblk = in_dim / QK8_0;
    launch_mmvq_q8_0_dense_ncols((const block_q8_0*)dev_w, (const block_q8_1_c*)g_d_nx2_q8, d_dst2, out_dim, nblk, 2);
}''', v_text)

v_text = v_text.replace(
    'swiglu_dev_ncols_kernel<<<(2 * sh_ffn_dim + 255) / 256, 256>>>(g_d_shact2, g_shg2, g_shu2, 2 * sh_ffn_dim);',
    'launch_swiglu_ncols(g_d_shact2, g_shg2, g_shu2, 2 * sh_ffn_dim);')

v_text = v_text.replace(
    'scale_shexp_kernel<<<(dim + 255) / 256, 256>>>(d_shexp_out0, g_d_shdown2,       g_shgi2,     dim);',
    'launch_scale_shexp(d_shexp_out0, g_d_shdown2, g_shgi2, dim);')
v_text = v_text.replace(
    'scale_shexp_kernel<<<(dim + 255) / 256, 256>>>(d_shexp_out1, g_d_shdown2 + dim, g_shgi2 + 1, dim);',
    'launch_scale_shexp(d_shexp_out1, g_d_shdown2 + dim, g_shgi2 + 1, dim);')

v_text = re.sub(r'mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>\([\s\S]*?\);\s*CUDA_CHECK_LAST\(\);',
                r'launch_mmvq_q6_K_dense_ncols((const block_q6_K*)output_weight->dev, (const block_q8_1_c*)g_d_nx2_q8, g_d_logits2, vocab_size, nblk, 2);',
                v_text)

v_text = re.sub(r'argmax_block_kernel<<<256, 256>>>\([\s\S]*?argmax_final_kernel<<<1, 256>>>\([\s\S]*?256\);\s*CUDA_CHECK_LAST\(\);',
                r'launch_argmax(g_d_logits2 + (size_t)c * vocab_size, g_d_block_max, g_d_block_idx, g_d_best_token, vocab_size);', v_text)

with open('src/gpu/verify_n2.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + v_text)
print("11. verify_n2.cu written")

# 12. batch.cu
s_b = find_line('extern "C" float *gpu_get_d_res_batch(')
e_b = find_line('CUDA Graph Capture for Delta-Net Dense Layers') - 1

b_text = "".join(lines[s_b:e_b])

b_text = b_text.replace(
    'rms_norm_dev_kernel<<<1, 256>>>(d_dst + (size_t)c * dim, d_src + (size_t)c * dim, d_weight, dim, eps);',
    'launch_rms_norm_dev(d_dst + (size_t)c * dim, d_src + (size_t)c * dim, d_weight, dim, eps);')

b_text = re.sub(
    r'quantize_q8_1_kernel<<<\([^)]+\) / 256, 256>>>\(([^,]+),\s*([^,]+),\s*([^)]+)\);',
    r'launch_quantize_q8_1(\1, \2, \3);', b_text)

b_text = re.sub(r'static void batch_dense_q8_0\([\s\S]*?CUDA_CHECK_LAST\(\);\s*}',
                r'''static void batch_dense_q8_0(const void *dev_w, float *d_dst, int in_dim, int out_dim, int n) {
    const int nblk = in_dim / QK8_0;
    launch_mmvq_q8_0_dense_ncols((const block_q8_0*)dev_w, (const block_q8_1_c*)g_d_nx_batch_q8, d_dst, out_dim, nblk, n);
}''', b_text)

b_text = re.sub(r'static void batch_dense_q6_K\([\s\S]*?CUDA_CHECK_LAST\(\);\s*}',
                r'''static void batch_dense_q6_K(const void *dev_w, float *d_dst, int in_dim, int out_dim, int n) {
    const int nblk = in_dim / QK_K;
    launch_mmvq_q6_K_dense_ncols((const block_q6_K*)dev_w, (const block_q8_1_c*)g_d_nx_batch_q8, d_dst, out_dim, nblk, n);
}''', b_text)

old_b_argmax = """    for (int c = 0; c < n; ++c) {
        argmax_block_kernel<<<256, 256>>>(g_d_logits_batch + (size_t)c * vocab_size,
                                          g_d_block_max, g_d_block_idx, vocab_size);
        CUDA_CHECK_LAST();
        argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token_batch + c, 256);
        CUDA_CHECK_LAST();
    }"""

new_b_argmax = """    for (int c = 0; c < n; ++c) {
        launch_argmax(g_d_logits_batch + (size_t)c * vocab_size, g_d_block_max, g_d_block_idx, g_d_best_token_batch + c, vocab_size);
    }"""

b_text = b_text.replace(old_b_argmax, new_b_argmax)

with open('src/gpu/batch.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + b_text)
print("12. batch.cu written")

# 13. graphs.cu
s_g = find_line('static void cuda_graph_update_exec')
e_g = find_line('Native MTP (Multi-Token Prediction)', s_g) - 1

g_text = "".join(lines[s_g:e_g])
g_text = re.sub(r'mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>\([\s\S]*?\);',
                r'launch_mmvq_dense_triple((const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim, (const block_q8_0*)layer->attn_k.dev, g_d_k, (int)k_dim, (const block_q8_0*)layer->attn_v.dev, g_d_v, (int)v_dim, xq, nblk);',
                g_text)

g_text = re.sub(r'fused_qk_split_norm_rope_kernel<<<n_head \+ n_head_kv, head_dim>>>\([\s\S]*?\);',
                r'launch_fused_qk_split_norm_rope(g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head, g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv, head_dim, eps, (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1], m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, g_d_pos);',
                g_text)

# Remove static definitions of variables defined in state.cu
g_text = g_text.replace('static unsigned long long g_graph_launches = 0;', '')
g_text = g_text.replace('static cudaGraph_t     g_layer_graphs2[64];', '')
g_text = g_text.replace('static cudaGraphExec_t g_layer_execs2[64];', '')
g_text = g_text.replace('static int             g_layer_graphs2_ready = 0;', '')

with open('src/gpu/graphs.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + g_text)
print("13. graphs.cu written")

# 14. mtp.cu
s_mtp = find_line('__global__ void concat_norms_kernel')
e_mtp = find_line('void *gpu_host_alloc(size_t bytes)', s_mtp)

mtp_text = "".join(lines[s_mtp:e_mtp])
mtp_text = mtp_text.replace('accum_inplace_kernel<<<8, 256>>>(g_d_mtp_x, g_d_mtp_concat, dim);',
                            'gpu_residual_accum(g_d_mtp_x, g_d_mtp_concat, dim);')

with open('src/gpu/mtp.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + mtp_text)
print("14. mtp.cu written")

# 15. context.cu
s_c1 = find_line('static void upload_dense')
e_c1 = find_line('void gpu_reset_delta_state(void)', s_c1)
s_c2 = find_line('void gpu_shutdown(model_t *m)')
e_c2 = find_line('void gpu_mat_vec(', s_c2)
s_c3 = find_line('double gpu_vram_used_mb(void)')
e_c3 = find_line('void gpu_residual_reset(const float *host_x', s_c3)
s_c4 = find_line('void *gpu_host_alloc(size_t bytes)')
e_c4 = len(lines)

ctx_text = (''.join(lines[s_c1:e_c1]) + '\n' +
            ''.join(lines[s_c2:e_c2]) + '\n' +
            ''.join(lines[s_c3:e_c3]) + '\n' +
            ''.join(lines[s_c4:e_c4]))

with open('src/gpu/context.cu', 'w') as f:
    f.write('#include "state.h"\n\n' + ctx_text)
print("15. context.cu written")

print("ALL MODULES SUCCESSFULLY GENERATED!")
