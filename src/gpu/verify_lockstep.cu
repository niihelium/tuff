// Lossless lockstep verify (2026-10): the two columns of a verify pass through ONE launch per
// weight-reading trunk kernel, each weight block read once for both columns.
//
// Every kernel here is a two-column copy of a single-token kernel with the per-column arithmetic
// untouched: each lane accumulates the same blocks in the same order into its column's partial
// sum, then the same warp reduction. So column c's outputs are bit-identical to running the
// single-token kernel on column c alone - the speculative output stays byte-identical to plain
// decode (checked end to end by the spec == no-spec corpus test). What is per column stays per
// column and runs the single-token kernels on per-column buffers: the RMS norms, the GDN state
// step (column 0 writes, column 1 the no-write variant), RoPE + attention over the KV cache
// (column 1 after column 0, so it sees column 0's KV), the router copy-out.
#include "state.h"
#include "dflash.h"

extern "C" void gpu_delta_gdn_core(int layer_idx, const model_layer_t *L, float eps, float scale_q,
                                   float *qkv, const float *z, const float *beta_raw, const float *alpha_raw,
                                   block_q8_1_c *xq_out, int nowrite,
                                   int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                   int key_dim, int value_dim, int qkv_dim, int conv_kernel);
void launch_scale_shexp(float *out, const float *down, const float *gate, int dim, cudaStream_t stream);
extern "C" int *gpu_verify_d_vpos(void);
extern "C" float *gpu_get_d_res_verify(int dim);
extern "C" void gpu_verify_set_positions(int pos0, int n);

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

// mmvq_q8_0_dense_delta_quad_warprow_kernel, two columns.
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 4)
__global__ void quad_n2_kernel(
        const block_q8_0 * __restrict__ w_qkv, int out_qkv,
        const block_q8_0 * __restrict__ w_z, int out_z,
        const float * __restrict__ w_beta, int out_beta,
        const float * __restrict__ w_alpha, int out_alpha,
        float * __restrict__ y_qkv0, float * __restrict__ y_z0, float * __restrict__ y_b0, float * __restrict__ y_a0,
        float * __restrict__ y_qkv1, float * __restrict__ y_z1, float * __restrict__ y_b1, float * __restrict__ y_a1,
        const block_q8_1_c * __restrict__ xq0, const block_q8_1_c * __restrict__ xq1,
        const float * __restrict__ xf0, const float * __restrict__ xf1,
        int nblk, int in_dim) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    const int total_out = out_qkv + out_z + out_beta + out_alpha;
    if (global_row >= total_out) return;

    if (global_row < out_qkv + out_z) {
        const bool is_qkv = (global_row < out_qkv);
        const int row = is_qkv ? global_row : (global_row - out_qkv);
        const block_q8_0 * __restrict__ rw = is_qkv ? (w_qkv + (size_t)row * nblk) : (w_z + (size_t)row * nblk);
        const int kqs = vdr * (threadIdx.x % tpsb);
        float t0 = 0.0f, t1 = 0.0f;
        #pragma unroll 4
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            t0 += vec_dot_q8_0_q8_1_c(rw + kbx, xq0 + kbx, kqs);
            t1 += vec_dot_q8_0_q8_1_c(rw + kbx, xq1 + kbx, kqs);
        }
        t0 = warp_reduce_sum_f(t0);
        t1 = warp_reduce_sum_f(t1);
        if (threadIdx.x == 0) {
            if (is_qkv) { y_qkv0[row] = t0; y_qkv1[row] = t1; }
            else        { y_z0[row] = t0;   y_z1[row] = t1; }
        }
    } else {
        const int rem = global_row - (out_qkv + out_z);
        const bool is_beta = (rem < out_beta);
        const int row = is_beta ? rem : (rem - out_beta);
        const float * __restrict__ rw = is_beta ? (w_beta + (size_t)row * in_dim) : (w_alpha + (size_t)row * in_dim);
        const float4 * __restrict__ rw4 = reinterpret_cast<const float4*>(rw);
        const float4 * __restrict__ x40 = reinterpret_cast<const float4*>(xf0);
        const float4 * __restrict__ x41 = reinterpret_cast<const float4*>(xf1);
        const int n_f4 = in_dim / 4;
        float t0 = 0.0f, t1 = 0.0f;
        #pragma unroll 4
        for (int i = threadIdx.x; i < n_f4; i += 32) {
            float4 w_val = rw4[i];
            float4 a = x40[i];
            float4 b = x41[i];
            t0 += w_val.x * a.x + w_val.y * a.y + w_val.z * a.z + w_val.w * a.w;
            t1 += w_val.x * b.x + w_val.y * b.y + w_val.z * b.z + w_val.w * b.w;
        }
        t0 = warp_reduce_sum_f(t0);
        t1 = warp_reduce_sum_f(t1);
        if (threadIdx.x == 0) {
            if (is_beta) { y_b0[row] = t0; y_b1[row] = t1; }
            else         { y_a0[row] = t0; y_a1[row] = t1; }
        }
    }
}

// mmvq_q8_0_dense_triple_warprow_kernel, two columns.
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void triple_n2_kernel(
        const block_q8_0 * __restrict__ w0, int out0,
        const block_q8_0 * __restrict__ w1, int out1,
        const block_q8_0 * __restrict__ w2, int out2,
        float * __restrict__ a0, float * __restrict__ a1, float * __restrict__ a2,
        float * __restrict__ b0, float * __restrict__ b1, float * __restrict__ b2,
        const block_q8_1_c * __restrict__ xq0, const block_q8_1_c * __restrict__ xq1, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;
    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    if (global_row >= out0 + out1 + out2) return;
    const block_q8_0 * __restrict__ rw;
    float *d0, *d1;
    int row;
    if (global_row < out0) {
        row = global_row; rw = w0 + (size_t)row * nblk; d0 = a0; d1 = b0;
    } else if (global_row < out0 + out1) {
        row = global_row - out0; rw = w1 + (size_t)row * nblk; d0 = a1; d1 = b1;
    } else {
        row = global_row - (out0 + out1); rw = w2 + (size_t)row * nblk; d0 = a2; d1 = b2;
    }
    const int kqs = vdr * (threadIdx.x % tpsb);
    float t0 = 0.0f, t1 = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        t0 += vec_dot_q8_0_q8_1_c(rw + kbx, xq0 + kbx, kqs);
        t1 += vec_dot_q8_0_q8_1_c(rw + kbx, xq1 + kbx, kqs);
    }
    t0 = warp_reduce_sum_f(t0);
    t1 = warp_reduce_sum_f(t1);
    if (threadIdx.x == 0) { d0[row] = t0; d1[row] = t1; }
}

// mmvq_q8_0_dense_warprow_kernel, two columns (accum: dst += , else dst = ).
template <int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void rows_n2_kernel(
        const block_q8_0 * __restrict__ w,
        const block_q8_1_c * __restrict__ y0, const block_q8_1_c * __restrict__ y1,
        float * __restrict__ dst0, float * __restrict__ dst1,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;
    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;
    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float t0 = 0.0f, t1 = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        t0 += vec_dot_q8_0_q8_1_c(rw + kbx, y0 + kbx, kqs);
        t1 += vec_dot_q8_0_q8_1_c(rw + kbx, y1 + kbx, kqs);
    }
    t0 = warp_reduce_sum_f(t0);
    t1 = warp_reduce_sum_f(t1);
    if (threadIdx.x == 0) {
        if (accum) { dst0[row] += t0; dst1[row] += t1; }
        else       { dst0[row]  = t0; dst1[row]  = t1; }
    }
}

// mul_mat_vec_f32_fast_kernel, two columns (the router projection).
__global__ void f32_fast_n2_kernel(const float * __restrict__ w,
                                   const float * __restrict__ x0, const float * __restrict__ x1,
                                   float * __restrict__ y0, float * __restrict__ y1,
                                   int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;
    extern __shared__ float s_xf[];
    float *s0 = s_xf, *s1 = s_xf + in_dim;
    for (int i = tid; i < in_dim; i += total_threads) { s0[i] = x0[i]; s1[i] = x1[i]; }
    __syncthreads();
    if (row < out_dim) {
        const float * __restrict__ rw = w + (size_t)row * in_dim;
        float a = 0.0f, b = 0.0f;
        for (int c = lane_id; c < in_dim; c += 32) {
            const float wv = rw[c];
            a += wv * s0[c];
            b += wv * s1[c];
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, mask);
            b += __shfl_down_sync(0xffffffff, b, mask);
        }
        if (lane_id == 0) { y0[row] = a; y1[row] = b; }
    }
}

// mmvq_q8_0_dense_shexp_swiglu_quant_kernel, two columns.
__launch_bounds__(1024, 1)
__global__ void shexp_swiglu_quant_n2_kernel(
        const block_q8_0 * __restrict__ w_gate, const block_q8_0 * __restrict__ w_up,
        block_q8_1_c * __restrict__ q0, block_q8_1_c * __restrict__ q1,
        int sh_ffn_dim,
        const block_q8_1_c * __restrict__ xq0, const block_q8_1_c * __restrict__ xq1, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int row = block_idx * 32 + warp_id;
    __shared__ float s_act[2][32];
    if (row < sh_ffn_dim) {
        const block_q8_0 * __restrict__ rw_g = w_gate + (size_t)row * nblk;
        const block_q8_0 * __restrict__ rw_u = w_up   + (size_t)row * nblk;
        const int kqs = vdr * (lane_id % tpsb);
        float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
        for (int kbx = lane_id / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            const block_q8_1_c *b0 = xq0 + kbx;
            const block_q8_1_c *b1 = xq1 + kbx;
            g0 += vec_dot_q8_0_q8_1_c(rw_g + kbx, b0, kqs);
            u0 += vec_dot_q8_0_q8_1_c(rw_u + kbx, b0, kqs);
            g1 += vec_dot_q8_0_q8_1_c(rw_g + kbx, b1, kqs);
            u1 += vec_dot_q8_0_q8_1_c(rw_u + kbx, b1, kqs);
        }
        g0 = warp_reduce_sum_f(g0);
        u0 = warp_reduce_sum_f(u0);
        g1 = warp_reduce_sum_f(g1);
        u1 = warp_reduce_sum_f(u1);
        if (lane_id == 0) {
            float sg0 = g0 / (1.0f + expf(-g0));
            float sg1 = g1 / (1.0f + expf(-g1));
            s_act[0][warp_id] = sg0 * u0;
            s_act[1][warp_id] = sg1 * u1;
        }
    } else {
        if (lane_id == 0) { s_act[0][warp_id] = 0.0f; s_act[1][warp_id] = 0.0f; }
    }
    __syncthreads();
    if (warp_id < 2) {
        block_q8_1_c *dst_q8 = warp_id == 0 ? q0 : q1;
        float act_val = s_act[warp_id][lane_id];
        float amax = fabsf(act_val);
        float sum = act_val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float d = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(act_val / d);
        dst_q8[block_idx].qs[lane_id] = q;
        if (lane_id == 0) {
            dst_q8[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Per-column buffers and the lockstep layer body
// ---------------------------------------------------------------------------------------------
typedef struct {
    float *norm[2];                 // attn_norm output (fp32), then post-attention norm output
    block_q8_1_c *xq[2];            // attn_norm q8 (dense input)
    float *qkv[2], *z[2], *b[2], *a[2];
    block_q8_1_c *gq[2];            // gated q8 (GDN out) / attention out q8
    float *qf[2], *k[2], *v[2];     // attention projections
    float *rl[2];                   // router logits
    block_q8_1_c *shq[2];           // shared-expert act q8
    float *shdown[2], *shgi[2];
} ls_bufs_t;

static ls_bufs_t B;
static cudaGraphExec_t g_ls_exec[64];
static int g_ls_ready = 0;
static float *g_h_lrl[2] = {NULL, NULL};
static float *g_h_lnx[2] = {NULL, NULL};

static void ls_alloc(const model_t *m) {
    const int dim = (int)m->embedding_length;
    const int sh = (int)m->expert_shared_ffn_len;
    int qf = 1, kd = 1, vd = 1;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        const model_layer_t *L = &m->layers[l];
        if (L->type != LAYER_TYPE_ATTENTION) continue;
        if ((int)L->attn_q.info.dims[1] > qf) qf = (int)L->attn_q.info.dims[1];
        if ((int)L->attn_k.info.dims[1] > kd) kd = (int)L->attn_k.info.dims[1];
        if ((int)L->attn_v.info.dims[1] > vd) vd = (int)L->attn_v.info.dims[1];
    }
    const int qkv_dim = (int)m->ssm_qkv_dim > 0 ? (int)m->ssm_qkv_dim : 1;
    const int vdim = (int)m->ssm_value_dim > 0 ? (int)m->ssm_value_dim : 1;
    const int nv = (int)m->ssm_n_v_heads > 0 ? (int)m->ssm_n_v_heads : 1;
    const int gmax = vdim > 8192 ? vdim : 8192;
    for (int c = 0; c < 2; ++c) {
        CUDA_CHECK(cudaMalloc(&B.norm[c], (size_t)dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.xq[c], (size_t)(dim / QK8_1) * sizeof(block_q8_1_c)));
        CUDA_CHECK(cudaMalloc(&B.qkv[c], (size_t)qkv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.z[c], (size_t)vdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.b[c], (size_t)nv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.a[c], (size_t)nv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.gq[c], (size_t)(gmax / QK8_1) * sizeof(block_q8_1_c)));
        CUDA_CHECK(cudaMalloc(&B.qf[c], (size_t)qf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.k[c], (size_t)kd * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.v[c], (size_t)vd * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.rl[c], 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.shq[c], (size_t)(sh / QK8_1) * sizeof(block_q8_1_c)));
        CUDA_CHECK(cudaMalloc(&B.shdown[c], (size_t)dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B.shgi[c], sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&g_h_lrl[c], 256 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&g_h_lnx[c], (size_t)dim * sizeof(float)));
    }
}

static void ls_layer_body(model_t *m, uint32_t l, uint32_t attn_idx, uint32_t delta_idx,
                          float *res[2], float *shexp[2], block_q8_1_c *rq8[2], const int *d_vpos) {
    model_layer_t *layer = &m->layers[l];
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int nblk = dim / QK8_0;
    dim3 blk(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    {
        const int tk = dflash_tap_index(l);   // drafter taps: the residual entering this layer
        if (tk >= 0)
            for (int c = 0; c < 2; ++c)
                CUDA_CHECK(cudaMemcpyAsync(dflash_stage_dst(c, tk), res[c], (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice, 0));
    }
    for (int c = 0; c < 2; ++c)
        gpu_fused_rms_norm_quant_dev(B.norm[c], res[c], (const float*)layer->attn_norm.dev, B.xq[c], dim, eps);

    if (layer->type == LAYER_TYPE_ATTENTION) {
        const uint32_t head_dim = m->key_length, n_head = m->head_count, n_head_kv = m->head_count_kv;
        const float scale = 1.0f / sqrtf((float)head_dim);
        const int qf = (int)layer->attn_q.info.dims[1], kd = (int)layer->attn_k.info.dims[1], vd = (int)layer->attn_v.info.dims[1];
        dim3 grid((unsigned)((qf + kd + vd + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
        triple_n2_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, blk>>>(
            (const block_q8_0*)layer->attn_q.dev, qf, (const block_q8_0*)layer->attn_k.dev, kd,
            (const block_q8_0*)layer->attn_v.dev, vd,
            B.qf[0], B.k[0], B.v[0], B.qf[1], B.k[1], B.v[1], B.xq[0], B.xq[1], nblk);
        CUDA_CHECK_LAST();
        for (int c = 0; c < 2; ++c) {
            const int *pos = d_vpos + c;
            launch_fused_qk_split_norm_rope(g_d_q, g_d_gate, B.qf[c], (const float*)layer->attn_q_norm.dev, (int)n_head,
                                            B.k[c], B.k[c], (const float*)layer->attn_k_norm.dev, (int)n_head_kv, head_dim, eps,
                                            (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1],
                                            m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, (int*)pos);
            launch_attention_kernel(g_dk_cache, g_dv_cache, g_d_q, B.k[c], B.v[c], g_d_gate, NULL, g_d_attn_scores,
                                    n_head, n_head_kv, g_max_seq, head_dim, (int)attn_idx, 0, 0, scale, B.gq[c], (int*)pos);
        }
        const int in_o = (int)(n_head * head_dim);
        dim3 og((unsigned)((dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
        rows_n2_kernel<MMVQ_ROWS_PER_BLOCK, true><<<og, blk>>>((const block_q8_0*)layer->attn_output.dev,
                                                               B.gq[0], B.gq[1], res[0], res[1], dim, in_o / QK8_0);
        CUDA_CHECK_LAST();
    } else {
        const int out_qkv = (int)layer->attn_qkv.info.dims[1], out_z = (int)layer->attn_gate.info.dims[1];
        const int out_b = (int)layer->ssm_beta.info.dims[1], out_a = (int)layer->ssm_alpha.info.dims[1];
        dim3 grid((unsigned)((out_qkv + out_z + out_b + out_a + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
        quad_n2_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, blk>>>(
            (const block_q8_0*)layer->attn_qkv.dev, out_qkv, (const block_q8_0*)layer->attn_gate.dev, out_z,
            (const float*)layer->ssm_beta.dev, out_b, (const float*)layer->ssm_alpha.dev, out_a,
            B.qkv[0], B.z[0], B.b[0], B.a[0], B.qkv[1], B.z[1], B.b[1], B.a[1],
            B.xq[0], B.xq[1], B.norm[0], B.norm[1], nblk, dim);
        CUDA_CHECK_LAST();
        const float scale_q = 1.0f / sqrtf((float)m->ssm_head_k_dim);
        for (int c = 0; c < 2; ++c) {
            gpu_delta_gdn_core((int)delta_idx, layer, eps, scale_q, B.qkv[c], B.z[c], B.b[c], B.a[c], B.gq[c], c,
                               (int)m->ssm_n_k_heads, (int)m->ssm_n_v_heads, (int)m->ssm_head_k_dim, (int)m->ssm_head_v_dim,
                               (int)m->ssm_key_dim, (int)m->ssm_value_dim, (int)m->ssm_qkv_dim, (int)m->ssm_conv_kernel);
        }
        const int vdim = (int)m->ssm_value_dim;
        dim3 og((unsigned)((dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
        rows_n2_kernel<MMVQ_ROWS_PER_BLOCK, true><<<og, blk>>>((const block_q8_0*)layer->ssm_out.dev,
                                                               B.gq[0], B.gq[1], res[0], res[1], dim, vdim / QK8_0);
        CUDA_CHECK_LAST();
    }

    // Post-attention norm straight into the routed experts' q8 (the bytes the single-token path's
    // gate/up reads from g_d_norm_q8) and the fp32 the router / CPU misses read.
    for (int c = 0; c < 2; ++c)
        gpu_fused_rms_norm_quant_dev(B.norm[c], res[c], (const float*)layer->post_attention_norm.dev, rq8[c], dim, eps);

    // Router (f32, the single-token default) for both columns, then the D2H the host top-k reads.
    const int n_expert = (int)m->expert_count;
    if (router_q8_enabled() && layer->ffn_gate_inp.dev_q8) {
        for (int c = 0; c < 2; ++c)
            launch_mat_vec_q8_0((const block_q8_0*)layer->ffn_gate_inp.dev_q8, B.norm[c], B.rl[c], dim, n_expert);
    } else {
        dim3 rb(32, 8);
        dim3 rg((n_expert + 7) / 8);
        f32_fast_n2_kernel<<<rg, rb, (size_t)2 * dim * sizeof(float)>>>((const float*)layer->ffn_gate_inp.dev,
                                                                          B.norm[0], B.norm[1], B.rl[0], B.rl[1], dim, n_expert);
        CUDA_CHECK_LAST();
    }
    for (int c = 0; c < 2; ++c) {
        CUDA_CHECK(cudaMemcpyAsync(g_h_lrl[c], B.rl[c], (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost, 0));
        CUDA_CHECK(cudaMemcpyAsync(g_h_lnx[c], B.norm[c], (size_t)dim * sizeof(float), cudaMemcpyDeviceToHost, 0));
    }

    // Shared expert, both columns: gate/up+SwiGLU+quant and down in one weight read each.
    const int sh = (int)m->expert_shared_ffn_len;
    {
        dim3 grid((unsigned)((sh + 31) / 32), 1u, 1u);
        dim3 block(32u, 32u, 1u);
        shexp_swiglu_quant_n2_kernel<<<grid, block>>>((const block_q8_0*)layer->ffn_gate_shexp.dev,
                                                      (const block_q8_0*)layer->ffn_up_shexp.dev,
                                                      B.shq[0], B.shq[1], sh, rq8[0], rq8[1], nblk);
        CUDA_CHECK_LAST();
        dim3 og((unsigned)((dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
        float *dd0 = layer->ffn_gate_inp_shexp.dev ? B.shdown[0] : shexp[0];
        float *dd1 = layer->ffn_gate_inp_shexp.dev ? B.shdown[1] : shexp[1];
        rows_n2_kernel<MMVQ_ROWS_PER_BLOCK, false><<<og, blk>>>((const block_q8_0*)layer->ffn_down_shexp.dev,
                                                                B.shq[0], B.shq[1], dd0, dd1, dim, sh / QK8_0);
        CUDA_CHECK_LAST();
        if (layer->ffn_gate_inp_shexp.dev) {
            for (int c = 0; c < 2; ++c) {
                launch_mat_vec_f32((const float*)layer->ffn_gate_inp_shexp.dev, B.norm[c], B.shgi[c], dim, 1);
                launch_scale_shexp(shexp[c], B.shdown[c], B.shgi[c], dim, 0);
            }
        }
    }
}

extern "C" void gpu_capture_lockstep_graphs(model_t *m) {
    if (!m || g_ls_ready || !gpu_has_verify_graphs()) return;
    if (tuff_exact() || !mmvq_dual_enabled() || g_n_gpu_expert_layers > 0) return;
    const int dim = (int)m->embedding_length;
    gpu_n2_prepare(dim, (int)m->vocab_size);
    ls_alloc(m);
    gpu_verify_set_positions(0, 2);
    float *res0 = gpu_get_d_res_verify(dim);
    float *res[2] = { res0, res0 + dim };
    float *shexp[2] = { g_d_shexp_out, (float*)gpu_get_shexp2_dev_ptr() };
    block_q8_1_c *rq8[2] = { (block_q8_1_c*)gpu_get_expert_xq8_dev_ptr(0), (block_q8_1_c*)gpu_get_expert_xq8_dev_ptr(1) };
    const int *d_vpos = gpu_verify_d_vpos();
    uint32_t ai = 0, di = 0;
    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
        ls_layer_body(m, l, ai, di, res, shexp, rq8, d_vpos);
        CUDA_CHECK(cudaStreamEndCapture(0, &g));
        CUDA_CHECK(cudaGraphInstantiate(&g_ls_exec[l], g, NULL, NULL, 0));
        CUDA_CHECK(cudaGraphDestroy(g));
        if (m->layers[l].type == LAYER_TYPE_ATTENTION) ai++; else di++;
    }
    g_ls_ready = 1;
}

extern "C" int gpu_has_lockstep_graphs(void) { return g_ls_ready; }
extern "C" void gpu_launch_lockstep(uint32_t l) { CUDA_CHECK(cudaGraphLaunch(g_ls_exec[l], 0)); }
extern "C" float *gpu_lockstep_host_rl(int c) { return g_h_lrl[c]; }
extern "C" float *gpu_lockstep_host_nx(int c) { return g_h_lnx[c]; }
