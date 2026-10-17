#pragma once

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
#define BATCH_COLS_MAX   8
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
extern int *g_d_union_slots;
extern int *g_d_union_m;
extern int *g_d_union_cols;
extern int *g_d_union_dst;
extern float *g_d_router_logits_batch;

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
