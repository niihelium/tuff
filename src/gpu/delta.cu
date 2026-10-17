#include "state.h"

__global__ void delta_state_update_kernel(
        float * __restrict__ S,          // [n_v_heads][head_v_dim][head_k_dim], in-place
        const float * __restrict__ q_norm, // [n_k_heads][head_k_dim]
        const float * __restrict__ k_norm, // [n_k_heads][head_k_dim]
        const float * __restrict__ v_conv, // [n_v_heads][head_v_dim]
        const float * __restrict__ decay,  // [n_v_heads] == expf(gate)
        const float * __restrict__ beta,   // [n_v_heads]
        float * __restrict__ dnet_out,     // [n_v_heads][head_v_dim]
        float scale_q,
        int n_k_heads, int head_v_dim, int head_k_dim) {
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int hk = h % n_k_heads;

    __shared__ float s_kh[128];
    __shared__ float s_qs[128];

    const float * __restrict__ kh = k_norm + (size_t)hk * head_k_dim;
    const float * __restrict__ qh = q_norm + (size_t)hk * head_k_dim;

    if (i < head_k_dim) {
        s_kh[i] = kh[i];
        s_qs[i] = qh[i] * scale_q;
    }
    __syncthreads();

    const float * __restrict__ vh = v_conv + (size_t)h * head_v_dim;
    const float dcy = decay[h];
    const float bv  = beta[h];

    float * __restrict__ row = S + ((size_t)h * head_v_dim + i) * head_k_dim;
    float4 * __restrict__ row4 = reinterpret_cast<float4*>(row);
    const float4 * __restrict__ kh4 = reinterpret_cast<const float4*>(s_kh);
    const float4 * __restrict__ qs4 = reinterpret_cast<const float4*>(s_qs);

    float sk = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float4 r = row4[j];
        float4 k = kh4[j];
        r.x *= dcy; r.y *= dcy; r.z *= dcy; r.w *= dcy;
        sk += r.x * k.x + r.y * k.y + r.z * k.z + r.w * k.w;
        row4[j] = r;
    }

    const float d = (vh[i] - sk) * bv;

    float oh = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float4 r = row4[j];
        float4 k = kh4[j];
        float4 q = qs4[j];
        r.x += d * k.x; r.y += d * k.y; r.z += d * k.z; r.w += d * k.w;
        row4[j] = r;
        oh += r.x * q.x + r.y * q.y + r.z * q.z + r.w * q.w;
    }

    dnet_out[(size_t)h * head_v_dim + i] = oh;
}

// Iteration-9 fused per-head RMS-norm + silu(z) gating. One block per head (head_v_dim threads).
__global__ void delta_norm_gate_kernel(
        float * __restrict__ gated_norm,      // [n_v_heads][head_v_dim]
        const float * __restrict__ dnet_out,  // [n_v_heads][head_v_dim]
        const float * __restrict__ z,         // [value_dim]
        const float * __restrict__ w,         // ssm_norm weights [head_v_dim]
        int head_v_dim, float eps) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const float * __restrict__ src = dnet_out + (size_t)h * head_v_dim;
    float * __restrict__ g = gated_norm + (size_t)h * head_v_dim;
    const float * __restrict__ zs = z + (size_t)h * head_v_dim;

    float val = (tid < head_v_dim) ? src[tid] : 0.0f;
    float sq = val * val;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sq += __shfl_down_sync(0xffffffff, sq, mask);
    }

    __shared__ float s_scale;
    __shared__ float warp_sums[4];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    if (tid == 0) {
        float sum = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        s_scale = rsqrtf(sum / (float)head_v_dim + eps);
    }
    __syncthreads();

    if (tid < head_v_dim) {
        const float zi = zs[tid];
        g[tid] = (val * s_scale * w[tid]) * (zi / (1.0f + expf(-zi)));
    }
}

// F32 mat-vec. One block per output row.

void gpu_reset_delta_state(void) {
    if (g_dstate_all && g_dstate_bytes > 0) CUDA_CHECK(cudaMemset(g_dstate_all, 0, g_dstate_bytes));
    // P4a: the device conv-state mirror must also restart from zero.
    if (g_dconv_all && g_dconv_bytes > 0) CUDA_CHECK(cudaMemset(g_dconv_all, 0, g_dconv_bytes));
}

void gpu_get_recurrent_state_sizes(size_t *dstate_bytes_out, size_t *dconv_bytes_out) {
    if (dstate_bytes_out) *dstate_bytes_out = g_dstate_bytes;
    if (dconv_bytes_out)  *dconv_bytes_out  = g_dconv_bytes;
}

void gpu_save_recurrent_state(float *dstate_host_out, float *dconv_host_out) {
    if (g_dstate_all && g_dstate_bytes > 0 && dstate_host_out) {
        CUDA_CHECK(cudaMemcpy(dstate_host_out, g_dstate_all, g_dstate_bytes, cudaMemcpyDeviceToHost));
    }
    if (g_dconv_all && g_dconv_bytes > 0 && dconv_host_out) {
        CUDA_CHECK(cudaMemcpy(dconv_host_out, g_dconv_all, g_dconv_bytes, cudaMemcpyDeviceToHost));
    }
}

void gpu_restore_recurrent_state(const float *dstate_host_in, const float *dconv_host_in) {
    if (g_dstate_all && g_dstate_bytes > 0 && dstate_host_in) {
        CUDA_CHECK(cudaMemcpy(g_dstate_all, dstate_host_in, g_dstate_bytes, cudaMemcpyHostToDevice));
    }
    if (g_dconv_all && g_dconv_bytes > 0 && dconv_host_in) {
        CUDA_CHECK(cudaMemcpy(g_dconv_all, dconv_host_in, g_dconv_bytes, cudaMemcpyHostToDevice));
    }
}


void gpu_delta_projections(float *qkv_mixed, const tensor_t *attn_qkv,
                           float *z,        const tensor_t *attn_gate,
                           float *beta_raw, const tensor_t *ssm_beta,
                           float *alpha_raw,const tensor_t *ssm_alpha,
                           const float *norm_x, int in_dim, int keep_dev) {
    const int out_qkv   = (int)attn_qkv->info.dims[1];
    const int out_z     = (int)attn_gate->info.dims[1];
    const int out_beta  = (int)ssm_beta->info.dims[1];
    const int out_alpha = (int)ssm_alpha->info.dims[1];

    if (!g_dpqkv) CUDA_CHECK(cudaMalloc(&g_dpqkv, (size_t)out_qkv   * sizeof(float)));
    if (!g_dpz)   CUDA_CHECK(cudaMalloc(&g_dpz,   (size_t)out_z     * sizeof(float)));
    if (!g_dpb)   CUDA_CHECK(cudaMalloc(&g_dpb,   (size_t)out_beta  * sizeof(float)));
    if (!g_dpa)   CUDA_CHECK(cudaMalloc(&g_dpa,   (size_t)out_alpha * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(g_dx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    launch_mul_mat_vec_q8_0_std((const block_q8_0*)attn_qkv->dev, g_dx, g_dpqkv, in_dim, out_qkv);
    launch_mul_mat_vec_q8_0_std((const block_q8_0*)attn_gate->dev, g_dx, g_dpz, in_dim, out_z);
    launch_mul_mat_vec_f32_std((const float*)ssm_beta->dev, g_dx, g_dpb, in_dim, out_beta);
    launch_mul_mat_vec_f32_std((const float*)ssm_alpha->dev, g_dx, g_dpa, in_dim, out_alpha);
    CUDA_CHECK_LAST();

    // P4a: keep_dev skips the four D2H downloads so g_dpqkv/g_dpz/g_dpb/g_dpa stay
    // device-resident for gpu_delta_pre / gpu_delta_post. The host pointers may then be NULL.
    if (!keep_dev) {
        CUDA_CHECK(cudaMemcpy(qkv_mixed, g_dpqkv, (size_t)out_qkv   * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z,         g_dpz,   (size_t)out_z     * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(beta_raw,  g_dpb,   (size_t)out_beta  * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(alpha_raw, g_dpa,   (size_t)out_alpha * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// M4: per-expert Q4_K mat-vec from a device-resident expert slot. in_dim x out_dim Q4_K
// matrix in dev_w; reuses the g_dx/g_dy scratch so norm_x is uploaded once per call.

// WRITE_STATE=false: the speculative verify column 1 computes its conv output against the state
// left by column 0 but does not shift the state (the commit replays the shift if it is accepted),
// and keeps a copy of its input in x_save for that replay.
template <bool WRITE_STATE>
__global__ void delta_conv_kernel_t(
        float * __restrict__ state,        // [conv_kernel][qkv_dim] this layer, in-place shift
        const float * __restrict__ x,      // g_dpqkv (qkv_mixed) [qkv_dim]
        const float * __restrict__ w,      // conv weights [qkv_dim][conv_kernel]
        float * __restrict__ out,          // qkv_conv [qkv_dim] (may alias x when WRITE_STATE)
        float * __restrict__ x_save,       // copy of x, or NULL
        int qkv_dim, int conv_kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) return;
    float s[4];
#pragma unroll
    for (int k = 0; k < conv_kernel; ++k) s[k] = state[(size_t)k * qkv_dim + c];
    const float xin = x[c];
    if (x_save) x_save[c] = xin;
    if (WRITE_STATE) {
#pragma unroll
        for (int k = 0; k < conv_kernel - 1; ++k) state[(size_t)k * qkv_dim + c] = s[k + 1];
        state[(size_t)(conv_kernel - 1) * qkv_dim + c] = xin;
    }

    // Conv over the SHIFTED state (matches the host loop exactly, k in 0..conv_kernel-1).
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < conv_kernel; ++k) {
        const float sv = (k < conv_kernel - 1) ? s[k + 1] : xin;
        sum += sv * w[(size_t)c * conv_kernel + k];
    }
    if (out) out[c] = sum / (1.0f + expf(-sum));   // SiLU
}

__global__ void delta_conv_kernel(
        float * __restrict__ state, const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ out, int qkv_dim, int conv_kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) return;
    float s[4];
#pragma unroll
    for (int k = 0; k < conv_kernel; ++k) s[k] = state[(size_t)k * qkv_dim + c];
    const float xin = x[c];
#pragma unroll
    for (int k = 0; k < conv_kernel - 1; ++k) state[(size_t)k * qkv_dim + c] = s[k + 1];
    state[(size_t)(conv_kernel - 1) * qkv_dim + c] = xin;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < conv_kernel; ++k) {
        const float sv = (k < conv_kernel - 1) ? s[k + 1] : xin;
        sum += sv * w[(size_t)c * conv_kernel + k];
    }
    out[c] = sum / (1.0f + expf(-sum));   // SiLU
}

// P4a: per-head l2_norm of both q and k slices of qkv_conv in a single fused launch.
__global__ void delta_l2norm_dual_kernel(
        const float * __restrict__ in,     // qkv_conv [qkv_dim]
        float * __restrict__ out_q,        // q_norm [n_k_heads][head_k_dim]
        float * __restrict__ out_k,        // k_norm [n_k_heads][head_k_dim]
        int key_dim, int head_k_dim, float eps, int n_k_heads) {
    const int b = blockIdx.x;
    const int is_k = (b >= n_k_heads);
    const int h = is_k ? (b - n_k_heads) : b;
    const int tid = threadIdx.x;
    const size_t base_offset = is_k ? (size_t)key_dim : 0;
    const float * __restrict__ src = in + base_offset + (size_t)h * head_k_dim;
    float * __restrict__ out = is_k ? out_k : out_q;

    float val = (tid < head_k_dim) ? src[tid] : 0.0f;
    float sq = val * val;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sq += __shfl_down_sync(0xffffffff, sq, mask);
    }

    __shared__ float s_scale;
    __shared__ float warp_sums[4];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    if (tid == 0) {
        float sum = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        s_scale = rsqrtf(sum + eps);
    }
    __syncthreads();

    if (tid < head_k_dim) {
        out[(size_t)h * head_k_dim + tid] = val * s_scale;
    }
}

// P4a: 32-element gate stage. beta[h]=sigmoid(beta_raw), decay[h]=exp(softplus(alpha+dt_bias)
// * ssm_a) with the host's exact `> 20.0f` branch, gate[h] kept for the dump path.
__global__ void delta_gate_kernel(
        const float * __restrict__ beta_raw,
        const float * __restrict__ alpha_raw,
        const float * __restrict__ dt_bias,
        const float * __restrict__ ssm_a,
        float * __restrict__ beta,         // [n_v_heads]
        float * __restrict__ decay,        // [n_v_heads] == expf(gate)
        float * __restrict__ gate,         // [n_v_heads]
        int n_v_heads) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_v_heads) return;
    beta[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
    const float ab = alpha_raw[h] + dt_bias[h];
    const float softp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
    const float g = softp * ssm_a[h];
    gate[h] = g;
    decay[h] = expf(g);
}

// Fused Single-Token Autoregressive Gated Delta-Net Kernel (H5):
// Ingests qkv_conv, beta_raw, alpha_raw, dt_bias, ssm_a, z, ssm_norm_w.
// In a single 32-block x 128-thread launch:
// 1. Performs per-head L2 normalization on Q and K using warp/block shuffles.
// 2. Computes Sigmoid(beta) and Softplus(alpha + dt_bias)*ssm_a on thread 0.
// 3. Performs the Delta-Net recurrent state update on S in registers with float4.
// 4. Performs per-head RMS normalization on the state-update output.
// 5. Applies SiLU(z) gating and writes the final gated_norm directly for ssm_out matvec.
template <bool WRITE_S>
__launch_bounds__(128, 4)
__global__ void fused_gated_delta_net_ar_kernel_t(
        float * __restrict__ S,                 // [n_v_heads][head_v_dim][head_k_dim] (32*128*128)
        const float * __restrict__ qkv_conv,    // [qkv_dim] (8192)
        const float * __restrict__ beta_raw,    // [n_v_heads] (32)
        const float * __restrict__ alpha_raw,   // [n_v_heads] (32)
        const float * __restrict__ dt_bias,     // [n_v_heads] (32)
        const float * __restrict__ ssm_a,       // [n_v_heads] (32)
        const float * __restrict__ z,           // [value_dim] (4096)
        const float * __restrict__ ssm_norm_w,  // [head_v_dim] (128)
        float * __restrict__ gated_norm,        // [value_dim] (4096) or NULL
        block_q8_1_c * __restrict__ out_gated_q8,// [value_dim/32] or NULL
        float * __restrict__ dnet_out,          // [value_dim] (4096)
        float * __restrict__ out_dq,            // [key_dim] (2048)
        float * __restrict__ out_dk,            // [key_dim] (2048)
        float * __restrict__ out_dp,            // [64]
        float * __restrict__ out_dpgate,        // [32]
        float scale_q,
        float eps,
        int n_k_heads, int n_v_heads,
        int head_k_dim, int head_v_dim,
        int key_dim, int value_dim) {
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int hk = h % n_k_heads;

    __shared__ float s_kh[128];
    __shared__ float s_qs[128];
    __shared__ float s_warp_q[4];
    __shared__ float s_warp_k[4];
    __shared__ float s_dcy;
    __shared__ float s_beta;
    __shared__ float s_rms_scale;

    const int warp_id = i >> 5;
    const int lane_id = i & 31;

    // 1. Q and K L2-norm
    const float * __restrict__ q_raw = qkv_conv + (size_t)hk * head_k_dim;
    const float * __restrict__ k_raw = qkv_conv + (size_t)key_dim + (size_t)hk * head_k_dim;

    float qi = (i < head_k_dim) ? q_raw[i] : 0.0f;
    float ki = (i < head_k_dim) ? k_raw[i] : 0.0f;
    float sq_q = qi * qi;
    float sq_k = ki * ki;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sq_q += __shfl_down_sync(0xffffffff, sq_q, mask);
        sq_k += __shfl_down_sync(0xffffffff, sq_k, mask);
    }
    if (lane_id == 0) {
        s_warp_q[warp_id] = sq_q;
        s_warp_k[warp_id] = sq_k;
    }
    __syncthreads();

    if (i == 0) {
        float sum_q = s_warp_q[0] + s_warp_q[1] + s_warp_q[2] + s_warp_q[3];
        float sum_k = s_warp_k[0] + s_warp_k[1] + s_warp_k[2] + s_warp_k[3];
        s_warp_q[0] = rsqrtf(sum_q + eps);
        s_warp_k[0] = rsqrtf(sum_k + eps);

        float b_val = 1.0f / (1.0f + expf(-beta_raw[h]));
        float ab = alpha_raw[h] + dt_bias[h];
        float softp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
        float g = softp * ssm_a[h];
        float d_val = expf(g);
        s_beta = b_val;
        s_dcy = d_val;
        if (out_dp) { out_dp[h] = d_val; out_dp[32 + h] = b_val; }
        if (out_dpgate) { out_dpgate[h] = g; }
    }
    __syncthreads();

    float scale_q_norm = s_warp_q[0] * scale_q;
    float scale_k_norm = s_warp_k[0];
    if (i < head_k_dim) {
        s_qs[i] = qi * scale_q_norm;
        s_kh[i] = ki * scale_k_norm;
        if (out_dq && h < n_k_heads) out_dq[(size_t)hk * head_k_dim + i] = qi * s_warp_q[0];
        if (out_dk && h < n_k_heads) out_dk[(size_t)hk * head_k_dim + i] = ki * scale_k_norm;
    }
    __syncthreads();

    const float * __restrict__ v_raw = qkv_conv + (size_t)2 * key_dim + (size_t)h * head_v_dim;
    const float vh_i = v_raw[i];
    const float dcy = s_dcy;
    const float bv = s_beta;

    float * __restrict__ row = S + ((size_t)h * head_v_dim + i) * head_k_dim;
    float4 * __restrict__ row4 = reinterpret_cast<float4*>(row);
    const float4 * __restrict__ kh4 = reinterpret_cast<const float4*>(s_kh);
    const float4 * __restrict__ qs4 = reinterpret_cast<const float4*>(s_qs);

    float sk = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float4 r = row4[j];
        float4 k = kh4[j];
        sk += r.x * k.x + r.y * k.y + r.z * k.z + r.w * k.w;
    }
    sk *= dcy;

    const float d = (vh_i - sk) * bv;

    float oh = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float4 r = row4[j];
        float4 k = kh4[j];
        float4 q = qs4[j];
        r.x = r.x * dcy + d * k.x;
        r.y = r.y * dcy + d * k.y;
        r.z = r.z * dcy + d * k.z;
        r.w = r.w * dcy + d * k.w;
        if (WRITE_S) row4[j] = r;
        oh += r.x * q.x + r.y * q.y + r.z * q.z + r.w * q.w;
    }

    if (dnet_out) dnet_out[(size_t)h * head_v_dim + i] = oh;

    // 2. Per-head RMS-norm and SiLU(z) gating
    float sq_o = oh * oh;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sq_o += __shfl_down_sync(0xffffffff, sq_o, mask);
    }
    if (lane_id == 0) s_warp_q[warp_id] = sq_o;
    __syncthreads();

    if (i == 0) {
        float sum_o = s_warp_q[0] + s_warp_q[1] + s_warp_q[2] + s_warp_q[3];
        s_rms_scale = rsqrtf(sum_o / 128.0f + eps);
    }
    __syncthreads();

    if (!dnet_out && !gated_norm && !out_gated_q8) return;   // commit replay: state only

    const float * __restrict__ zs = z + (size_t)h * head_v_dim;
    float z_val = zs[i];
    float silu_z = z_val / (1.0f + expf(-z_val));
    float g_val = (oh * s_rms_scale * ssm_norm_w[i]) * silu_z;
    if (gated_norm) gated_norm[(size_t)h * head_v_dim + i] = g_val;

    if (out_gated_q8) {
        float amax = fabsf(g_val);
        float sum  = g_val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
            sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
        }
        const float d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(g_val / d);
        const int block_idx = h * 4 + warp_id;
        out_gated_q8[block_idx].qs[lane_id] = q;
        if (lane_id == 0) {
            out_gated_q8[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

// H17: explicit-pointer variant. qkv/z/beta_raw/alpha_raw are the input projections this layer
// reads (today's single-column path passes the globals g_dpqkv/g_dpz/g_dpb/g_dpa; the two-column
// verify trunk points them straight at g_d_p2_qkv + c*out_qkv etc., so the 240 D2D column
// publishes per verify step disappear). qkv is consumed destructively by delta_conv_kernel (it
// overwrites it in place with qkv_conv), so the caller must own it and not need it again.
static float *g_nw_x = NULL, *g_nw_conv = NULL, *g_nw_b = NULL, *g_nw_a = NULL;
static void delta_nw_ensure(int qkv_dim, int n_v_heads) {
    const int nl = 64;
    if (!g_nw_x)    CUDA_CHECK(cudaMalloc(&g_nw_x,    (size_t)nl * qkv_dim * sizeof(float)));
    if (!g_nw_conv) CUDA_CHECK(cudaMalloc(&g_nw_conv, (size_t)nl * qkv_dim * sizeof(float)));
    if (!g_nw_b)    CUDA_CHECK(cudaMalloc(&g_nw_b,    (size_t)nl * n_v_heads * sizeof(float)));
    if (!g_nw_a)    CUDA_CHECK(cudaMalloc(&g_nw_a,    (size_t)nl * n_v_heads * sizeof(float)));
}

static void delta_fused_gdn_impl(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                                 float *qkv, const float *z, const float *beta_raw, const float *alpha_raw,
                                 int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                 int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim, int nowrite);

void gpu_delta_fused_gdn_dev_p(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                               float *qkv, const float *z, const float *beta_raw, const float *alpha_raw,
                               int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                               int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    delta_fused_gdn_impl(layer_idx, L, d_res_x, eps, scale_q, qkv, z, beta_raw, alpha_raw,
                         n_k_heads, n_v_heads, head_k_dim, head_v_dim, key_dim, value_dim, qkv_dim, conv_kernel, dim, 0);
}

// The fused GDN step for speculative verify column 1 (see delta_fused_gdn_impl).
void gpu_delta_fused_gdn_dev_nw(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                                int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    delta_fused_gdn_impl(layer_idx, L, d_res_x, eps, scale_q, g_dpqkv, g_dpz, g_dpb, g_dpa,
                         n_k_heads, n_v_heads, head_k_dim, head_v_dim, key_dim, value_dim, qkv_dim, conv_kernel, dim, 1);
}

// Column 1 was accepted: replay its conv shift and recurrent state update from the saved inputs,
// with the same kernels (state-writing instantiations) a plain decode step runs, so the state is
// bit-identical to having decoded the token on its own.
void gpu_delta_commit_nw(int layer_idx, const model_layer_t *L, float eps, float scale_q,
                         int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                         int key_dim, int value_dim, int qkv_dim, int conv_kernel) {
    delta_nw_ensure(qkv_dim, n_v_heads);
    float *conv_state = g_dconv_all + (size_t)layer_idx * (size_t)conv_kernel * qkv_dim;
    const float *conv_w = (const float*)L->ssm_conv1d.dev;
    float *S = g_dstate_all + (size_t)layer_idx * n_v_heads * head_v_dim * head_k_dim;
    delta_conv_kernel_t<true><<<(qkv_dim + 255) / 256, 256>>>(
        conv_state, g_nw_x + (size_t)layer_idx * qkv_dim, conv_w, NULL, NULL, qkv_dim, conv_kernel);
    fused_gated_delta_net_ar_kernel_t<true><<<n_v_heads, head_v_dim>>>(
        S, g_nw_conv + (size_t)layer_idx * qkv_dim,
        g_nw_b + (size_t)layer_idx * n_v_heads, g_nw_a + (size_t)layer_idx * n_v_heads,
        (const float*)L->ssm_dt_bias.dev, (const float*)L->ssm_a.dev,
        g_dpz, (const float*)L->ssm_norm.dev,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        scale_q, eps, n_k_heads, n_v_heads, head_k_dim, head_v_dim, key_dim, value_dim);
    CUDA_CHECK_LAST();
}

static void delta_fused_gdn_impl(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                                 float *qkv, const float *z, const float *beta_raw, const float *alpha_raw,
                                 int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                 int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim, int nowrite) {
    if (!g_dgated) CUDA_CHECK(cudaMalloc(&g_dgated, (size_t)value_dim * sizeof(float)));
    if (!g_dout)   CUDA_CHECK(cudaMalloc(&g_dout,   (size_t)value_dim * sizeof(float)));
    if (!g_dq)     CUDA_CHECK(cudaMalloc(&g_dq,     (size_t)key_dim * sizeof(float)));
    if (!g_dk)     CUDA_CHECK(cudaMalloc(&g_dk,     (size_t)key_dim * sizeof(float)));
    if (!g_dp)     CUDA_CHECK(cudaMalloc(&g_dp,     (size_t)64 * sizeof(float)));
    if (!g_dpgate) CUDA_CHECK(cudaMalloc(&g_dpgate, (size_t)32 * sizeof(float)));

    float *conv_state = g_dconv_all + (size_t)layer_idx * (size_t)conv_kernel * qkv_dim;
    const float *conv_w = (const float*)L->ssm_conv1d.dev;
    float *S = g_dstate_all + (size_t)layer_idx * n_v_heads * head_v_dim * head_k_dim;
    block_q8_1_c *xq = norm_x_q8_ensure(value_dim);
    float *d_gated_target = (!tuff_exact() && mmvq_dual_enabled()) ? NULL : g_dgated;

    if (nowrite) {
        // Speculative verify column 1: everything against the state column 0 left, nothing
        // written back. The inputs the state update needs are saved for gpu_delta_commit_nw.
        delta_nw_ensure(qkv_dim, n_v_heads);
        float *cx = g_nw_x    + (size_t)layer_idx * qkv_dim;
        float *cc = g_nw_conv + (size_t)layer_idx * qkv_dim;
        delta_conv_kernel_t<false><<<(qkv_dim + 255) / 256, 256>>>(
            conv_state, qkv, conv_w, cc, cx, qkv_dim, conv_kernel);
        CUDA_CHECK(cudaMemcpyAsync(g_nw_b + (size_t)layer_idx * n_v_heads, beta_raw,  (size_t)n_v_heads * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(g_nw_a + (size_t)layer_idx * n_v_heads, alpha_raw, (size_t)n_v_heads * sizeof(float), cudaMemcpyDeviceToDevice));
        fused_gated_delta_net_ar_kernel_t<false><<<n_v_heads, head_v_dim>>>(
            S, cc, beta_raw, alpha_raw,
            (const float*)L->ssm_dt_bias.dev, (const float*)L->ssm_a.dev,
            z, (const float*)L->ssm_norm.dev,
            d_gated_target, xq, g_dout, g_dq, g_dk, g_dp, g_dpgate,
            scale_q, eps, n_k_heads, n_v_heads, head_k_dim, head_v_dim, key_dim, value_dim);
        CUDA_CHECK_LAST();
        if (!tuff_exact() && mmvq_dual_enabled()) {
            mmvq_q8_0_rows((const block_q8_0*)L->ssm_out.dev, xq, d_res_x, value_dim, dim, true);
        } else {
            launch_mat_vec_q8_0_accum((const block_q8_0*)L->ssm_out.dev, g_dgated, d_res_x, value_dim, dim);
        }
        return;
    }

    delta_conv_kernel<<<(qkv_dim + 255) / 256, 256>>>(
        conv_state, qkv, conv_w, qkv, qkv_dim, conv_kernel);

    fused_gated_delta_net_ar_kernel_t<true><<<n_v_heads, head_v_dim>>>(
        S, qkv, beta_raw, alpha_raw,
        (const float*)L->ssm_dt_bias.dev, (const float*)L->ssm_a.dev,
        z, (const float*)L->ssm_norm.dev,
        d_gated_target, xq, g_dout, g_dq, g_dk, g_dp, g_dpgate,
        scale_q, eps, n_k_heads, n_v_heads, head_k_dim, head_v_dim, key_dim, value_dim);
    CUDA_CHECK_LAST();

    if (!tuff_exact() && mmvq_dual_enabled()) {
        mmvq_q8_0_rows((const block_q8_0*)L->ssm_out.dev, xq, d_res_x, value_dim, dim, true);
    } else {
        launch_mat_vec_q8_0_accum((const block_q8_0*)L->ssm_out.dev, g_dgated, d_res_x, value_dim, dim);
    }
}

void gpu_delta_fused_gdn_dev(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                             int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                             int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    gpu_delta_fused_gdn_dev_p(layer_idx, L, d_res_x, eps, scale_q,
                              g_dpqkv, g_dpz, g_dpb, g_dpa,
                              n_k_heads, n_v_heads, head_k_dim, head_v_dim,
                              key_dim, value_dim, qkv_dim, conv_kernel, dim);
}

// P4a: run the whole pre-stage on device, consuming g_dpqkv/g_dpb/g_dpa (from
// gpu_delta_projections keep_dev=1) and producing q_norm (g_dq), k_norm (g_dk), v_conv (g_dpqkv + 2*key_dim),
// decay (g_dp[0..31]), beta (g_dp[32..63]) and gate (g_dpgate). All launches are on the
// default stream, so they are stream-ordered after the projections and before the state update.
void gpu_delta_pre(int layer_idx, const model_layer_t *L, float eps,
                   int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                   int key_dim, int value_dim, int qkv_dim, int conv_kernel) {
    // Ensure the state-update scratch is resident (matches gpu_delta_state_update's sizing).
    if (!g_dq)      CUDA_CHECK(cudaMalloc(&g_dq,   (size_t)16 * 128 * sizeof(float)));
    if (!g_dk)      CUDA_CHECK(cudaMalloc(&g_dk,   (size_t)16 * 128 * sizeof(float)));
    if (!g_dp)      CUDA_CHECK(cudaMalloc(&g_dp,   (size_t)64 * sizeof(float)));
    if (!g_dpgate)  CUDA_CHECK(cudaMalloc(&g_dpgate, (size_t)32 * sizeof(float)));

    float *state = g_dconv_all + (size_t)layer_idx * (size_t)conv_kernel * qkv_dim;
    const float *conv_w = (const float*)L->ssm_conv1d.dev;

    delta_conv_kernel<<<(qkv_dim + 255) / 256, 256>>>(
        state, g_dpqkv, conv_w, g_dpqkv, qkv_dim, conv_kernel);
    delta_l2norm_dual_kernel<<<2 * n_k_heads, head_k_dim>>>(
        g_dpqkv, g_dq, g_dk, key_dim, head_k_dim, eps, n_k_heads);
    delta_gate_kernel<<<1, 32>>>(
        g_dpb, g_dpa, (const float*)L->ssm_dt_bias.dev, (const float*)L->ssm_a.dev,
        g_dp + 32, g_dp, g_dpgate, n_v_heads);
    CUDA_CHECK_LAST();
}

// P4a: device-pointer variant of gpu_delta_state_update. q_norm/k_norm/v_conv/decay/beta are
// already device-resident (produced by gpu_delta_pre) in g_dq/g_dk/g_dpqkv/g_dp, so the five
// H2D cudaMemcpyAsync calls disappear. dnet_out==NULL keeps the result in g_dout for
// gpu_delta_post, as before.
void gpu_delta_state_update_dev(int layer_idx, float *dnet_out, float scale_q,
                                int n_v_heads, int n_k_heads,
                                int head_v_dim, int head_k_dim) {
    if (!g_dout) CUDA_CHECK(cudaMalloc(&g_dout, (size_t)32 * 128 * sizeof(float)));
    float *S = g_dstate_all + (size_t)layer_idx * n_v_heads * head_v_dim * head_k_dim;
    const float *v_conv_ptr = g_dpqkv + (size_t)2 * n_k_heads * head_k_dim;

    delta_state_update_kernel<<<n_v_heads, head_v_dim>>>(
        S, g_dq, g_dk, v_conv_ptr, g_dp, g_dp + 32, g_dout, scale_q,
        n_k_heads, head_v_dim, head_k_dim);
    CUDA_CHECK_LAST();

    if (dnet_out)
        CUDA_CHECK(cudaMemcpy(dnet_out, g_dout, (size_t)n_v_heads * head_v_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// P4a: expose the device-resident delta tensors to the engine's on-demand dump path. All
// pointers are valid only after gpu_delta_pre has run for the current layer. `state` points at
// the start of g_dstate_all; the engine adds its layer slice.
void gpu_delta_get_device(gpu_delta_dev_t *out) {
    out->qkv_conv = g_dpqkv;   // qkv_conv (overwritten by delta_conv_kernel), qkv_dim
    out->z        = g_dpz;     // value_dim
    out->q_norm   = g_dq;      // key_dim
    out->k_norm   = g_dk;      // key_dim
    out->v_conv   = g_dpqkv + (size_t)2 * 16 * 128; // value_dim (2 * key_dim)
    out->decay    = g_dp;      // n_v_heads
    out->beta     = g_dp + 32; // n_v_heads
    out->gate     = g_dpgate;  // n_v_heads
    out->state    = g_dstate_all;
}

void gpu_copy_dev_f32(const void *dev, float *host, size_t n) {
    if (dev && host && n) CUDA_CHECK(cudaMemcpy(host, dev, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_delta_state_update(int layer_idx, float *dnet_out,
                            const float *q_norm, const float *k_norm,
                            const float *v_conv, const float *decay, const float *beta,
                            float scale_q,
                            int n_v_heads, int n_k_heads,
                            int head_v_dim, int head_k_dim) {
    // Lazy-allocate per-call scratch on first use (sizes fixed by the model constants).
    if (!g_dq)   CUDA_CHECK(cudaMalloc(&g_dq,   (size_t)16 * 128 * sizeof(float)));
    if (!g_dk)   CUDA_CHECK(cudaMalloc(&g_dk,   (size_t)16 * 128 * sizeof(float)));
    if (!g_dv)   CUDA_CHECK(cudaMalloc(&g_dv,   (size_t)32 * 128 * sizeof(float)));
    if (!g_dout) CUDA_CHECK(cudaMalloc(&g_dout, (size_t)32 * 128 * sizeof(float)));
    if (!g_dp)   CUDA_CHECK(cudaMalloc(&g_dp,   (size_t)64 * sizeof(float)));

    float *S = g_dstate_all + (size_t)layer_idx * n_v_heads * head_v_dim * head_k_dim;

    CUDA_CHECK(cudaMemcpyAsync(g_dq,     q_norm, (size_t)n_k_heads * head_k_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_dk,     k_norm, (size_t)n_k_heads * head_k_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_dv,     v_conv, (size_t)n_v_heads * head_v_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_dp,     decay,  (size_t)n_v_heads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_dp + 32, beta,   (size_t)n_v_heads * sizeof(float), cudaMemcpyHostToDevice));

    delta_state_update_kernel<<<n_v_heads, head_v_dim>>>(
        S, g_dq, g_dk, g_dv, g_dp, g_dp + 32, g_dout, scale_q,
        n_k_heads, head_v_dim, head_k_dim);
    CUDA_CHECK_LAST();

    // Iteration-9: if the caller supplies dnet_out == NULL the result is kept on device in
    // g_dout (fused into gpu_delta_post for the ssm_out mat-vec), skipping the D2H round-trip.
    if (dnet_out)
        CUDA_CHECK(cudaMemcpy(dnet_out, g_dout, (size_t)n_v_heads * head_v_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-9: after gpu_delta_state_update(layer_idx, NULL, ...) leaves dnet_out in g_dout,
// compute gated_norm on device (per-head rms_norm + silu(z) gating) and run the ssm_out Q8_0
// mat-vec straight from it, downloading only attn_out. z must remain device-resident in g_dpz
// (kept by gpu_delta_projections). Bit-identical to the CPU rms_norm/silu + gpu_mat_vec path.
void gpu_delta_post(float *attn_out, const tensor_t *ssm_out,
                    const float *ssm_norm_w, float eps,
                    int n_v_heads, int head_v_dim, int value_dim, int out_dim) {
    if (!g_dgated) CUDA_CHECK(cudaMalloc(&g_dgated, (size_t)value_dim * sizeof(float)));

    delta_norm_gate_kernel<<<n_v_heads, head_v_dim>>>(
        g_dgated, g_dout, g_dpz, ssm_norm_w, head_v_dim, eps);
    CUDA_CHECK_LAST();

    launch_mul_mat_vec_q8_0_std((const block_q8_0*)ssm_out->dev, g_dgated, g_dy, value_dim, out_dim);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(attn_out, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-10: run the three shared-expert projections that all take norm_x as input
// (ffn_gate_shexp, ffn_up_shexp -> sh_ffn_dim; ffn_gate_inp_shexp -> 1) with a SINGLE upload of the
// shared input and back-to-back launches, then download all three. Each output element is produced
// by the exact same kernel / accumulation order as gpu_mat_vec, so the result is bit-identical; we
// only remove the 2 redundant norm_x uploads and their serialized syncs.

void gpu_residual_reset(const float *host_x, int dim) {
    if (host_x && g_d_res_x) {
        CUDA_CHECK(cudaMemcpyAsync(g_d_res_x, host_x, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    }
}

__global__ void rms_norm_dev_kernel(float * __restrict__ dst, const float * __restrict__ src,
                                    const float * __restrict__ w, int dim, float eps) {
    __shared__ float s_warp_sum[8];
    __shared__ float s_scale;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = src[i];
        local_sum += v * v;
    }
    local_sum = warp_reduce_sum_f(local_sum);

    if (lane_id == 0) {
        s_warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum8 = (lane_id < 8) ? s_warp_sum[lane_id] : 0.0f;
        sum8 = warp_reduce_sum_f(sum8);
        if (lane_id == 0) {
            s_scale = rsqrtf(sum8 / (float)dim + eps);
        }
    }
    __syncthreads();

    const float scale = s_scale;
    for (int i = tid; i < dim; i += 256) {
        dst[i] = src[i] * scale * w[i];
    }
}

void gpu_rms_norm_dev(float *d_dst, const float *d_src, const float *d_weight, int dim, float eps) {
    rms_norm_dev_kernel<<<1, 256>>>(d_dst, d_src, d_weight, dim, eps);
    CUDA_CHECK_LAST();
}

__global__ void residual_add_kernel(float * __restrict__ dst, const float * __restrict__ delta, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        dst[idx] += delta[idx];
    }
}

void gpu_residual_add(float *d_res_x, const float *d_delta, int dim) {
    residual_add_kernel<<<(dim + 255) / 256, 256>>>(d_res_x, d_delta, dim);
    CUDA_CHECK_LAST();
}

void gpu_delta_projections_dev(const tensor_t *attn_qkv,
                               const tensor_t *attn_gate,
                               const tensor_t *ssm_beta,
                               const tensor_t *ssm_alpha,
                               const float *d_norm_x, int in_dim) {
    const int out_qkv   = (int)attn_qkv->info.dims[1];
    const int out_z     = (int)attn_gate->info.dims[1];
    const int out_beta  = (int)ssm_beta->info.dims[1];
    const int out_alpha = (int)ssm_alpha->info.dims[1];

    if (!tuff_exact() && mmvq_dual_enabled()) {
        const block_q8_1_c *xq = (const block_q8_1_c *)g_d_dense_x_q8;
        const int nblk = in_dim / QK8_0;
        const int total_out = out_qkv + out_z + out_beta + out_alpha;
        dim3 grid((unsigned)((total_out + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
        dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
        launch_mmvq_delta_quad((const block_q8_0*)attn_qkv->dev, g_dpqkv, out_qkv, (const block_q8_0*)attn_gate->dev, g_dpz, out_z, (const float*)ssm_beta->dev, g_dpb, out_beta, (const float*)ssm_alpha->dev, g_dpa, out_alpha, xq, d_norm_x, nblk, in_dim);
    } else {
        launch_mat_vec_q8_0_dual((const block_q8_0*)attn_qkv->dev, g_dpqkv, out_qkv,
                                 (const block_q8_0*)attn_gate->dev, g_dpz,   out_z,
                                 d_norm_x, in_dim);
        launch_mat_vec_f32_dual ((const float*)ssm_beta->dev, g_dpb, out_beta,
                                 (const float*)ssm_alpha->dev, g_dpa, out_alpha,
                                 d_norm_x, in_dim);
    }
}

void gpu_delta_post_accum(float *d_res_x, const tensor_t *ssm_out,
                          const float *ssm_norm_w, float eps,
                          int n_v_heads, int head_v_dim, int value_dim, int out_dim) {
    if (!g_dgated) CUDA_CHECK(cudaMalloc(&g_dgated, (size_t)value_dim * sizeof(float)));

    delta_norm_gate_kernel<<<n_v_heads, head_v_dim>>>(
        g_dgated, g_dout, g_dpz, ssm_norm_w, head_v_dim, eps);
    CUDA_CHECK_LAST();

    launch_mat_vec_q8_0_accum((const block_q8_0*)ssm_out->dev, g_dgated, d_res_x, value_dim, out_dim);
}


void launch_rms_norm_dev(float *dst, const float *src, const float *weight, int dim, float eps, cudaStream_t stream) {
    rms_norm_dev_kernel<<<1, 256, 0, stream>>>(dst, src, weight, dim, eps);
    CUDA_CHECK_LAST();
}
