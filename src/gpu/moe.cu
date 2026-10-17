#include "state.h"

void gpu_expert_mat_vec(float *dst, const void *dev_w, const float *src, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_dx, src, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    launch_mat_vec_q4_K((const block_q4_K*)dev_w, g_dx, g_dy, in_dim, out_dim);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(dst, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Q6_K expert mat-vec (the routed down experts are Q6_K). Single thread per output row,
// strict scalar sequential accumulation matching the CPU mat_vec_q6_K_scalar() bit-for-bit
// (both use the same dequant value `d*s*q` and `sum += yv*x[c]` fma contraction).

void gpu_expert_mat_vec_q6(float *dst, const void *dev_w, const float *src, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_dx, src, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    launch_mat_vec_q6_K_scalar((const block_q6_K*)dev_w, g_dx, g_dy, in_dim, out_dim);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(dst, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Pipelined gate+up launch on the dedicated expert stream. Uploads norm_x once, then runs the
// Q4_K gate and up kernels (both dim x ffn) writing gate->g_exgy and up->g_exgy2. Returns
// immediately; call gpu_expert_gateup_wait_download() later to sync + fetch both. The caller
// must ensure the previous launch has been downloaded (so g_exgx/g_exgy/g_exgy2 are free).
void gpu_expert_gateup_launch(int slot, const float *norm_x, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exgx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 0), g_exgx, g_exgy, in_dim, out_dim, g_expert_stream);
    launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 1), g_exgx, g_exgy2, in_dim, out_dim, g_expert_stream);
    CUDA_CHECK_LAST();
    g_expert_pipe_active = 1;
}

// Sync the expert stream and download the pending gate/up results into gate_dst/up_dst.
void gpu_expert_gateup_wait_download(float *gate_dst, float *up_dst, int out_dim) {
    if (!g_expert_pipe_active) return;
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    CUDA_CHECK(cudaMemcpy(gate_dst, g_exgy,  (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(up_dst,   g_exgy2, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    g_expert_pipe_active = 0;
}

// Drop any pending async expert work (used at layer boundaries in the fallback path).
void gpu_expert_gateup_cancel(void) {
    if (g_expert_pipe_active || g_expert_down_active) {
        CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
        g_expert_pipe_active = 0;
        g_expert_down_active = 0;
    }
}

// Sync the expert stream WITHOUT clearing the active flags. Called before a slot fill so an
// in-flight gateup/down kernel never reads a slot's gate/up/down bytes while they are being
// overwritten by a synchronous (default-stream) fill memcpy — otherwise an LFRU eviction can
// corrupt a resident slot mid-read and diverge the oracle.
void gpu_expert_gateup_sync(void) {
    if (g_expert_pipe_active || g_expert_down_active) {
        CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    }
}

// Down expert mat-vec on the dedicated expert stream, after gate/up(k+1) has been queued so
// both overlap. Uses dedicated buffers (g_exdx/g_exdy) so it never collides with the gate/up
// pipeline buffers. `type` is the layer's down tensor type (Q4_K or Q6_K); the matching kernel
// is bit-exact with the CPU reference. in_dim=ffn, out_dim=dim.
void gpu_expert_down_launch(int slot, ggml_type_t type, const float *act_exp, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exdx, act_exp, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    if (type == GGML_TYPE_Q6_K)
        launch_mat_vec_q6_K_scalar((const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, g_exdy, in_dim, out_dim, g_expert_stream);
    else
        launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, g_exdy, in_dim, out_dim, g_expert_stream);
    CUDA_CHECK_LAST();
    g_expert_down_active = 1;
}

// Sync the expert stream and download the pending down result.
void gpu_expert_down_wait_download(float *down_dst, int out_dim) {
    if (!g_expert_down_active) return;
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    CUDA_CHECK(cudaMemcpy(down_dst, g_exdy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    g_expert_down_active = 0;
}

// ---------------------------------------------------------------------------
// Batched async expert pipeline (one sync+download for all gate/up, one for all down per layer).
// ---------------------------------------------------------------------------

// Upload norm_x once, then launch gate+up for every HIT expert (slots[k]>=0) writing to
// per-position buffers g_exgbat/g_exubbat. All share the same g_exgx input.
void gpu_expert_batch_gateup_launch(const int *slots, int n, const float *norm_x, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exgx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    g_batch_active = 1;
    for (int k = 0; k < n; ++k) {
        if (slots[k] < 0) continue;
        launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slots[k], 0), g_exgx, g_exgbat + (size_t)k * out_dim, in_dim, out_dim, g_expert_stream);
        launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slots[k], 1), g_exgx, g_exubbat + (size_t)k * out_dim, in_dim, out_dim, g_expert_stream);
    }
    CUDA_CHECK_LAST();
}

// Sync the expert stream and download all batched gate/up outputs into contiguous host arrays.
void gpu_expert_batch_gateup_sync_download(float *gate_out, float *up_out, int n, int out_dim) {
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    CUDA_CHECK(cudaMemcpy(gate_out, g_exgbat,  (size_t)n * out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(up_out,   g_exubbat, (size_t)n * out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    g_batch_active = 0;
}

// Launch one down mat-vec into the per-position down buffer (act_exp uploaded to g_exdx, which
// is stream-ordered so each kernel reads its own upload). Callable per expert; all launches
// accumulate on the stream and are synced+downloaded together at the end of the layer.
void gpu_expert_batch_down_launch(int k, int slot, ggml_type_t type, const float *act_exp, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exdx, act_exp, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    float *dst = g_exdbat + (size_t)k * out_dim;
    if (type == GGML_TYPE_Q6_K)
        launch_mat_vec_q6_K_scalar((const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim, out_dim, g_expert_stream);
    else
        launch_mat_vec_q4_K((const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim, out_dim, g_expert_stream);
    CUDA_CHECK_LAST();
    g_batch_active = 1;
}

// Sync the expert stream and download all batched down outputs into a contiguous host array.
// Positions where slots[k]<0 (MISS: no down kernel launched, host result already in down_out)
// are left untouched.
void gpu_expert_batch_down_sync_download(const int *slots, float *down_out, int n, int out_dim) {
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    CUDA_CHECK(cudaMemcpy(down_out, g_exdbat, (size_t)n * out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    g_batch_active = 0;
}

// Sync the expert stream WITHOUT clearing batch state (used before a MISS slot fill so an
// in-flight batched kernel never reads a slot being overwritten).
void gpu_expert_batch_sync(void) {
    if (g_batch_active || g_expert_pipe_active || g_expert_down_active) {
        CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    }
}

// ---------------------------------------------------------------------------
// P3: MMVQ port from llama.cpp (MIT) @ 84e908c.
//   - block_q8_1 layout + activation quantization: ggml/src/ggml-cuda/quantize.cu
//     (quantize_q8_1) and ggml/src/ggml-common.h.
//   - vec_dot_q4_K_q8_1 / vec_dot_q4_K_q8_1_impl_vmmq (vecdotq.cuh:910, 508) and
//     vec_dot_q6_K_q8_1 / vec_dot_q6_K_q8_1_impl_mmvq (vecdotq.cuh:1002, 627).
//   - launch shape + reduction from mul_mat_vec_q (mmvq.cu:480-700): nwarps warps cooperate
//     on ONE output row, __dp4a int8 dot, warp_reduce_sum + shared-mem cross-warp reduce.
//   - the (expert -> slot) indirection is tuff's d_slots[e], which is structurally
//     llama.cpp's `channel_x = ids[channel_dst]` (mmvq.cu:516).
//
// This path is NOT bit-identical to tuff's scalar CPU reference (Q8_1 activation
// quantization + tree reduction), by design: parity.md P0/P3 retires bit-equality against the
// scalar path in favour of the llama.cpp token oracle. TUFF_EXACT=1 restores the old
// bit-exact scalar kernels below.
// ---------------------------------------------------------------------------


// One CUDA block computes ONE output row of ONE expert: grid = (out_dim, n_experts),
// block = (32, nwarps). Weight rows come from the LFRU pool at slot = d_slots[blockIdx.y];
// the q8_1 activation row is shared (y_stride_blocks == 0, gate/up) or per-expert
// (y_stride_blocks == in_dim/QK8_1, down). Mirrors mul_mat_vec_q with rows_per_cuda_block=1.
template <int nwarps>
__launch_bounds__(nwarps*32, 1)
__global__ void mmvq_q4_K_expert_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy, int y_stride_blocks,
        float * __restrict__ dst, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;                    // threads per super-block = 16
    constexpr int blocks_per_iter = vdr*nwarps*32 / QI4_K;

    const int row = blockIdx.x;
    const int e   = blockIdx.y;
    const int tid = threadIdx.y*32 + threadIdx.x;

    const block_q4_K * __restrict__ rw =
        (const block_q4_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y = vy + (size_t)e*y_stride_blocks;

    const int kqs = vdr * (tid % tpsb);
    float tmp = 0.0f;
    for (int kbx = tid / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q4_K_q8_1_c(rw + kbx, y + kbx*(QK_K/QK8_1), kqs);
    }

    tmp = warp_reduce_sum_f(tmp);
    __shared__ float s[nwarps];
    if (threadIdx.x == 0) s[threadIdx.y] = tmp;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) t += s[w];
        dst[(size_t)e*out_dim + row] = t;
    }
}

template <int nwarps>
__launch_bounds__(nwarps*32, 1)
__global__ void mmvq_q6_K_expert_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy, int y_stride_blocks,
        float * __restrict__ dst, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;                    // threads per super-block = 32
    constexpr int blocks_per_iter = vdr*nwarps*32 / QI6_K;

    const int row = blockIdx.x;
    const int e   = blockIdx.y;
    const int tid = threadIdx.y*32 + threadIdx.x;

    const block_q6_K * __restrict__ rw =
        (const block_q6_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y = vy + (size_t)e*y_stride_blocks;

    const int kqs = vdr * (tid % tpsb);
    float tmp = 0.0f;
    for (int kbx = tid / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx*(QK_K/QK8_1), kqs);
    }

    tmp = warp_reduce_sum_f(tmp);
    __shared__ float s[nwarps];
    if (threadIdx.x == 0) s[threadIdx.y] = tmp;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) t += s[w];
        dst[(size_t)e*out_dim + row] = t;
    }
}

// GLU-fused gate+up (parity.md sec.2.4b, llama.cpp mmvq.cu:588/617/657 + common.cuh:1528's
// ggml_cuda_mm_fusion_args_device): one launch carries BOTH weight offsets, keeps tmp/tmp_gate
// side by side over the same q8_1 activation, and applies SwiGLU in the epilogue. Writes
// act[e][row] = silu(gate) * up directly to device memory, so the gate/up D2H, the host
// swiglu() and the act H2D all disappear.
template <int nwarps>
__launch_bounds__(nwarps*32, 1)
__global__ void mmvq_q4_K_gateup_swiglu_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes,
        unsigned long long off_gate, unsigned long long off_up,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy,
        float * __restrict__ act, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*nwarps*32 / QI4_K;

    const int row = blockIdx.x;
    const int e   = blockIdx.y;
    const int tid = threadIdx.y*32 + threadIdx.x;

    const char * __restrict__ base = pool + (unsigned long long)d_slots[e]*slot_bytes;
    const block_q4_K * __restrict__ wg = (const block_q4_K *)(base + off_gate) + (size_t)row*nblk;
    const block_q4_K * __restrict__ wu = (const block_q4_K *)(base + off_up)   + (size_t)row*nblk;

    const int kqs = vdr * (tid % tpsb);
    float tmp_gate = 0.0f;
    float tmp_up   = 0.0f;
    for (int kbx = tid / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c * __restrict__ yb = vy + kbx*(QK_K/QK8_1);
        tmp_gate += vec_dot_q4_K_q8_1_c(wg + kbx, yb, kqs);
        tmp_up   += vec_dot_q4_K_q8_1_c(wu + kbx, yb, kqs);
    }

    tmp_gate = warp_reduce_sum_f(tmp_gate);
    tmp_up   = warp_reduce_sum_f(tmp_up);
    __shared__ float sg[nwarps];
    __shared__ float su[nwarps];
    if (threadIdx.x == 0) { sg[threadIdx.y] = tmp_gate; su[threadIdx.y] = tmp_up; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) { g += sg[w]; u += su[w]; }
        // Identical formula to cpu_ops.c swiglu(): silu(gate) * up.
        act[(size_t)e*out_dim + row] = (g / (1.0f + expf(-g))) * u;
    }
}

template <int rows_per_block>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q4_K_gateup_swiglu_warprow_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes,
        unsigned long long off_gate, unsigned long long off_up,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy,
        float * __restrict__ act, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;                    // 16
    constexpr int blocks_per_iter = vdr*32 / QI4_K;

    const int warp_id = threadIdx.y;                    // 0..rows_per_block-1
    const int lane_id = threadIdx.x;                    // 0..31
    const int row     = blockIdx.x * rows_per_block + warp_id;
    const int e       = blockIdx.y;

    if (row >= out_dim) return;

    const char * __restrict__ base = pool + (unsigned long long)d_slots[e]*slot_bytes;
    const block_q4_K * __restrict__ wg = (const block_q4_K *)(base + off_gate) + (size_t)row*nblk;
    const block_q4_K * __restrict__ wu = (const block_q4_K *)(base + off_up)   + (size_t)row*nblk;

    const int kqs = vdr * (lane_id % tpsb);
    float tmp_gate = 0.0f;
    float tmp_up   = 0.0f;
    for (int kbx = lane_id / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c * __restrict__ yb = vy + kbx*(QK_K/QK8_1);
        tmp_gate += vec_dot_q4_K_q8_1_c(wg + kbx, yb, kqs);
        tmp_up   += vec_dot_q4_K_q8_1_c(wu + kbx, yb, kqs);
    }

    tmp_gate = warp_reduce_sum_f(tmp_gate);
    tmp_up   = warp_reduce_sum_f(tmp_up);

    if (lane_id == 0) {
        act[(size_t)e*out_dim + row] = (tmp_gate / (1.0f + expf(-tmp_gate))) * tmp_up;
    }
}

// Small-K variant (llama.cpp's `small_k` case, mmvq.cu:464 `rows_per_cuda_block = nwarps`).
// When one row needs <= 2 warp-iterations there is no point spreading it over 4 warps: instead
// each WARP owns its own output row, so a 128-thread block retires MMVQ_ROWS_PER_BLOCK rows with
// every thread useful and no cross-warp reduction. This matters on Ampere, where a 32-thread
// block would cap the SM at 16 blocks x 32 = 512 of 1536 resident threads (33% occupancy); the
// 128-thread form is thread-limited instead. Used for the down projection (in_dim = ffn = 512).
#define MMVQ_ROWS_PER_BLOCK 4

template <int rows_per_block>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q4_K_expert_warprow_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy, int y_stride_blocks,
        float * __restrict__ dst, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI4_K;   // per warp

    const int row = blockIdx.x*rows_per_block + threadIdx.y;
    if (row >= out_dim) return;                      // uniform across the warp
    const int e = blockIdx.y;

    const block_q4_K * __restrict__ rw =
        (const block_q4_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y = vy + (size_t)e*y_stride_blocks;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q4_K_q8_1_c(rw + kbx, y + kbx*(QK_K/QK8_1), kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) dst[(size_t)e*out_dim + row] = tmp;
}

template <int rows_per_block>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q6_K_expert_warprow_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy, int y_stride_blocks,
        float * __restrict__ dst, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI6_K;   // per warp

    const int row = blockIdx.x*rows_per_block + threadIdx.y;
    if (row >= out_dim) return;
    const int e = blockIdx.y;

    const block_q6_K * __restrict__ rw =
        (const block_q6_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y = vy + (size_t)e*y_stride_blocks;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx*(QK_K/QK8_1), kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) dst[(size_t)e*out_dim + row] = tmp;
}

// ---------------------------------------------------------------------------
// H15: ncols variants of the routed-expert kernels for the two-column verify trunk.
//
// A shared expert (selected by BOTH tokens) must still be evaluated against two
// different activations, but its weight stream can be read once for both — the same
// ncols trick that took the dense trunk from 15.1 ms to 3.4 ms per two-token step.
// These kernels read each weight row once and accumulate one dot per activation
// column. The per-column accumulation sequence is identical to the single-column
// kernels above (same vec_dot, same kqs, same kbx order, same warp reduction), so
// column c is bit-identical to the single-column result.
//
// The `dst` arrays map the compressed slot-table index e to the destination index
// in the column's hit-ordered output buffer (NULL = identity, i.e. the existing
// layout). For ncols == 1 only dst0/act0/vy0 are used; the shared experts use
// ncols == 2 with both columns' dst arrays.
// ---------------------------------------------------------------------------
template <int nwarps, int ncols>
__launch_bounds__(nwarps*32, 1)
__global__ void mmvq_q4_K_gateup_swiglu_ncols_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes,
        unsigned long long off_gate, unsigned long long off_up,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy0, const block_q8_1_c * __restrict__ vy1,
        const int * __restrict__ d_dst0, const int * __restrict__ d_dst1,
        float * __restrict__ act0, float * __restrict__ act1, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*nwarps*32 / QI4_K;

    const int row = blockIdx.x;
    const int e   = blockIdx.y;
    const int tid = threadIdx.y*32 + threadIdx.x;

    const char * __restrict__ base = pool + (unsigned long long)d_slots[e]*slot_bytes;
    const block_q4_K * __restrict__ wg = (const block_q4_K *)(base + off_gate) + (size_t)row*nblk;
    const block_q4_K * __restrict__ wu = (const block_q4_K *)(base + off_up)   + (size_t)row*nblk;

    const int kqs = vdr * (tid % tpsb);
    float tg0 = 0.0f, tu0 = 0.0f;
    float tg1 = 0.0f, tu1 = 0.0f;
    for (int kbx = tid / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c * __restrict__ yb0 = vy0 + kbx*(QK_K/QK8_1);
        tg0 += vec_dot_q4_K_q8_1_c(wg + kbx, yb0, kqs);
        tu0 += vec_dot_q4_K_q8_1_c(wu + kbx, yb0, kqs);
        if (ncols == 2) {
            const block_q8_1_c * __restrict__ yb1 = vy1 + kbx*(QK_K/QK8_1);
            tg1 += vec_dot_q4_K_q8_1_c(wg + kbx, yb1, kqs);
            tu1 += vec_dot_q4_K_q8_1_c(wu + kbx, yb1, kqs);
        }
    }

    tg0 = warp_reduce_sum_f(tg0);
    tu0 = warp_reduce_sum_f(tu0);
    __shared__ float sg[nwarps];
    __shared__ float su[nwarps];
    if (threadIdx.x == 0) { sg[threadIdx.y] = tg0; su[threadIdx.y] = tu0; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) { g += sg[w]; u += su[w]; }
        const int d0 = d_dst0 ? d_dst0[e] : e;
        act0[(size_t)d0*out_dim + row] = (g / (1.0f + expf(-g))) * u;
    }
    if (ncols == 2) {
        tg1 = warp_reduce_sum_f(tg1);
        tu1 = warp_reduce_sum_f(tu1);
        if (threadIdx.x == 0) { sg[threadIdx.y] = tg1; su[threadIdx.y] = tu1; }
        __syncthreads();
        if (tid == 0) {
            float g = 0.0f, u = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) { g += sg[w]; u += su[w]; }
            const int d1 = d_dst1 ? d_dst1[e] : e;
            act1[(size_t)d1*out_dim + row] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q4_K_gateup_swiglu_warprow_ncols_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes,
        unsigned long long off_gate, unsigned long long off_up,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy0, const block_q8_1_c * __restrict__ vy1,
        const int * __restrict__ d_dst0, const int * __restrict__ d_dst1,
        float * __restrict__ act0, float * __restrict__ act1, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI4_K;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row     = blockIdx.x * rows_per_block + warp_id;
    const int e       = blockIdx.y;
    if (row >= out_dim) return;

    const char * __restrict__ base = pool + (unsigned long long)d_slots[e]*slot_bytes;
    const block_q4_K * __restrict__ wg = (const block_q4_K *)(base + off_gate) + (size_t)row*nblk;
    const block_q4_K * __restrict__ wu = (const block_q4_K *)(base + off_up)   + (size_t)row*nblk;

    const int kqs = vdr * (lane_id % tpsb);
    float tg0 = 0.0f, tu0 = 0.0f;
    float tg1 = 0.0f, tu1 = 0.0f;
    for (int kbx = lane_id / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c * __restrict__ yb0 = vy0 + kbx*(QK_K/QK8_1);
        tg0 += vec_dot_q4_K_q8_1_c(wg + kbx, yb0, kqs);
        tu0 += vec_dot_q4_K_q8_1_c(wu + kbx, yb0, kqs);
        if (ncols == 2) {
            const block_q8_1_c * __restrict__ yb1 = vy1 + kbx*(QK_K/QK8_1);
            tg1 += vec_dot_q4_K_q8_1_c(wg + kbx, yb1, kqs);
            tu1 += vec_dot_q4_K_q8_1_c(wu + kbx, yb1, kqs);
        }
    }

    tg0 = warp_reduce_sum_f(tg0);
    tu0 = warp_reduce_sum_f(tu0);
    if (lane_id == 0) {
        const int d0 = d_dst0 ? d_dst0[e] : e;
        act0[(size_t)d0*out_dim + row] = (tg0 / (1.0f + expf(-tg0))) * tu0;
    }
    if (ncols == 2) {
        tg1 = warp_reduce_sum_f(tg1);
        tu1 = warp_reduce_sum_f(tu1);
        if (lane_id == 0) {
            const int d1 = d_dst1 ? d_dst1[e] : e;
            act1[(size_t)d1*out_dim + row] = (tg1 / (1.0f + expf(-tg1))) * tu1;
        }
    }
}

// Down projection, Q4_K. Reads the activation at dst[e] (in_dim/QK8_1 blocks per
// expert), writes dst[e]'s output row — for the shared ncols=2 form both columns.
template <int nwarps, int ncols>
__launch_bounds__(nwarps*32, 1)
__global__ void mmvq_q4_K_expert_ncols_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy0, const block_q8_1_c * __restrict__ vy1, int y_stride_blocks,
        const int * __restrict__ d_dst0, const int * __restrict__ d_dst1,
        float * __restrict__ dst0, float * __restrict__ dst1, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*nwarps*32 / QI4_K;

    const int row = blockIdx.x;
    const int e   = blockIdx.y;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int d0  = d_dst0 ? d_dst0[e] : e;

    const block_q4_K * __restrict__ rw =
        (const block_q4_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y0 = vy0 + (size_t)d0*y_stride_blocks;

    const int kqs = vdr * (tid % tpsb);
    float tmp0 = 0.0f, tmp1 = 0.0f;
    for (int kbx = tid / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp0 += vec_dot_q4_K_q8_1_c(rw + kbx, y0 + kbx*(QK_K/QK8_1), kqs);
        if (ncols == 2) {
            const int d1 = d_dst1 ? d_dst1[e] : e;
            tmp1 += vec_dot_q4_K_q8_1_c(rw + kbx, vy1 + (size_t)d1*y_stride_blocks + kbx*(QK_K/QK8_1), kqs);
        }
    }

    tmp0 = warp_reduce_sum_f(tmp0);
    __shared__ float s[nwarps];
    if (threadIdx.x == 0) s[threadIdx.y] = tmp0;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) t += s[w];
        dst0[(size_t)d0*out_dim + row] = t;
    }
    if (ncols == 2) {
        const int d1 = d_dst1 ? d_dst1[e] : e;
        tmp1 = warp_reduce_sum_f(tmp1);
        if (threadIdx.x == 0) s[threadIdx.y] = tmp1;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) t += s[w];
            dst1[(size_t)d1*out_dim + row] = t;
        }
    }
}

template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q4_K_expert_warprow_ncols_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy0, const block_q8_1_c * __restrict__ vy1, int y_stride_blocks,
        const int * __restrict__ d_dst0, const int * __restrict__ d_dst1,
        float * __restrict__ dst0, float * __restrict__ dst1, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI4_K;   // per warp

    const int row = blockIdx.x*rows_per_block + threadIdx.y;
    if (row >= out_dim) return;
    const int e = blockIdx.y;
    const int d0 = d_dst0 ? d_dst0[e] : e;

    const block_q4_K * __restrict__ rw =
        (const block_q4_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y0 = vy0 + (size_t)d0*y_stride_blocks;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp0 = 0.0f, tmp1 = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp0 += vec_dot_q4_K_q8_1_c(rw + kbx, y0 + kbx*(QK_K/QK8_1), kqs);
        if (ncols == 2) {
            const int d1 = d_dst1 ? d_dst1[e] : e;
            tmp1 += vec_dot_q4_K_q8_1_c(rw + kbx, vy1 + (size_t)d1*y_stride_blocks + kbx*(QK_K/QK8_1), kqs);
        }
    }
    tmp0 = warp_reduce_sum_f(tmp0);
    if (threadIdx.x == 0) dst0[(size_t)d0*out_dim + row] = tmp0;
    if (ncols == 2) {
        const int d1 = d_dst1 ? d_dst1[e] : e;
        tmp1 = warp_reduce_sum_f(tmp1);
        if (threadIdx.x == 0) dst1[(size_t)d1*out_dim + row] = tmp1;
    }
}

// Down projection, Q6_K (the deep layers' routed down experts).
template <int nwarps, int ncols>
__launch_bounds__(nwarps*32, 1)
__global__ void mmvq_q6_K_expert_ncols_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy0, const block_q8_1_c * __restrict__ vy1, int y_stride_blocks,
        const int * __restrict__ d_dst0, const int * __restrict__ d_dst1,
        float * __restrict__ dst0, float * __restrict__ dst1, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr*nwarps*32 / QI6_K;

    const int row = blockIdx.x;
    const int e   = blockIdx.y;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int d0  = d_dst0 ? d_dst0[e] : e;

    const block_q6_K * __restrict__ rw =
        (const block_q6_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y0 = vy0 + (size_t)d0*y_stride_blocks;

    const int kqs = vdr * (tid % tpsb);
    float tmp0 = 0.0f, tmp1 = 0.0f;
    for (int kbx = tid / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp0 += vec_dot_q6_K_q8_1_c(rw + kbx, y0 + kbx*(QK_K/QK8_1), kqs);
        if (ncols == 2) {
            const int d1 = d_dst1 ? d_dst1[e] : e;
            tmp1 += vec_dot_q6_K_q8_1_c(rw + kbx, vy1 + (size_t)d1*y_stride_blocks + kbx*(QK_K/QK8_1), kqs);
        }
    }

    tmp0 = warp_reduce_sum_f(tmp0);
    __shared__ float s[nwarps];
    if (threadIdx.x == 0) s[threadIdx.y] = tmp0;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) t += s[w];
        dst0[(size_t)d0*out_dim + row] = t;
    }
    if (ncols == 2) {
        const int d1 = d_dst1 ? d_dst1[e] : e;
        tmp1 = warp_reduce_sum_f(tmp1);
        if (threadIdx.x == 0) s[threadIdx.y] = tmp1;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) t += s[w];
            dst1[(size_t)d1*out_dim + row] = t;
        }
    }
}

template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q6_K_expert_warprow_ncols_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const block_q8_1_c * __restrict__ vy0, const block_q8_1_c * __restrict__ vy1, int y_stride_blocks,
        const int * __restrict__ d_dst0, const int * __restrict__ d_dst1,
        float * __restrict__ dst0, float * __restrict__ dst1, int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI6_K;   // per warp

    const int row = blockIdx.x*rows_per_block + threadIdx.y;
    if (row >= out_dim) return;
    const int e = blockIdx.y;
    const int d0 = d_dst0 ? d_dst0[e] : e;

    const block_q6_K * __restrict__ rw =
        (const block_q6_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;
    const block_q8_1_c * __restrict__ y0 = vy0 + (size_t)d0*y_stride_blocks;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp0 = 0.0f, tmp1 = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp0 += vec_dot_q6_K_q8_1_c(rw + kbx, y0 + kbx*(QK_K/QK8_1), kqs);
        if (ncols == 2) {
            const int d1 = d_dst1 ? d_dst1[e] : e;
            tmp1 += vec_dot_q6_K_q8_1_c(rw + kbx, vy1 + (size_t)d1*y_stride_blocks + kbx*(QK_K/QK8_1), kqs);
        }
    }
    tmp0 = warp_reduce_sum_f(tmp0);
    if (threadIdx.x == 0) dst0[(size_t)d0*out_dim + row] = tmp0;
    if (ncols == 2) {
        const int d1 = d_dst1 ? d_dst1[e] : e;
        tmp1 = warp_reduce_sum_f(tmp1);
        if (threadIdx.x == 0) dst1[(size_t)d1*out_dim + row] = tmp1;
    }
}


// nwarps so that (nblk super-blocks) x (threads per super-block) threads cover the row in one
// iteration, capped at 4 warps (llama.cpp's Ampere/GENERIC choice at ncols_dst=1, mmvq.cu:354).
static int mmvq_nwarps(int nblk, int threads_per_superblock) {
    int w = (nblk*threads_per_superblock + 31) / 32;
    if (w < 1) w = 1;
    if (w > 4) w = 4;
    return w;
}

static void launch_mmvq_q4_K_expert(unsigned long long off, const block_q8_1_c *vy, int y_stride_blocks,
                                    float *dst, int n, int out_dim, int in_dim, cudaStream_t st,
                                    const int *slots) {
    const int nblk = in_dim / QK_K;
    if (nblk * (QI4_K / VDR_Q4_K_Q8_1_MMVQ) <= 64) {   // small-K: one warp per row
        dim3 g((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1)/MMVQ_ROWS_PER_BLOCK), (unsigned)n, 1u);
        dim3 b(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
        mmvq_q4_K_expert_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<g, b, 0, st>>>(
            g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk);
        CUDA_CHECK_LAST();
        return;
    }
    const int nw = mmvq_nwarps(nblk, QI4_K / VDR_Q4_K_Q8_1_MMVQ);
    dim3 grid((unsigned)out_dim, (unsigned)n, 1u);
    dim3 block(32u, (unsigned)nw, 1u);
    switch (nw) {
        case 1: mmvq_q4_K_expert_kernel<1><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
        case 2: mmvq_q4_K_expert_kernel<2><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
        case 3: mmvq_q4_K_expert_kernel<3><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
        default:mmvq_q4_K_expert_kernel<4><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
    }
    CUDA_CHECK_LAST();
}

static void launch_mmvq_q6_K_expert(unsigned long long off, const block_q8_1_c *vy, int y_stride_blocks,
                                    float *dst, int n, int out_dim, int in_dim, cudaStream_t st,
                                    const int *slots) {
    const int nblk = in_dim / QK_K;
    if (nblk * (QI6_K / VDR_Q6_K_Q8_1_MMVQ) <= 64) {   // small-K: one warp per row
        dim3 g((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1)/MMVQ_ROWS_PER_BLOCK), (unsigned)n, 1u);
        dim3 b(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
        mmvq_q6_K_expert_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<g, b, 0, st>>>(
            g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk);
        CUDA_CHECK_LAST();
        return;
    }
    const int nw = mmvq_nwarps(nblk, QI6_K / VDR_Q6_K_Q8_1_MMVQ);
    dim3 grid((unsigned)out_dim, (unsigned)n, 1u);
    dim3 block(32u, (unsigned)nw, 1u);
    switch (nw) {
        case 1: mmvq_q6_K_expert_kernel<1><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
        case 2: mmvq_q6_K_expert_kernel<2><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
        case 3: mmvq_q6_K_expert_kernel<3><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
        default:mmvq_q6_K_expert_kernel<4><<<grid, block, 0, st>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy, y_stride_blocks, dst, out_dim, nblk); break;
    }
    CUDA_CHECK_LAST();
}

// P3b GLU-fused MoE stage. One launch computes gate+up+SwiGLU for every selected expert into
// the device-resident activation buffer g_exdx, a second quantizes it to q8_1 in place — no
// host round-trip between the gate/up and down projections. `in_dim` is the model's embedding
// dim, `ffn_dim` the expert FFN dim; both come from the model struct via the engine.
void gpu_expert_moe_gateup_swiglu(const int *slots, int n, const float *norm_x, int in_dim, int ffn_dim) {
    block_q8_1_c *xq8 = (block_q8_1_c *)g_exgx_q8;
    if (norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(g_exgx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
        launch_quantize_q8_1(g_exgx, xq8, in_dim, g_expert_stream);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaMemcpyAsync(g_exslots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));

    const int nblk = in_dim / QK_K;
    const int nw = mmvq_nwarps(nblk, QI4_K / VDR_Q4_K_Q8_1_MMVQ);
    dim3 grid((unsigned)ffn_dim, (unsigned)n, 1u);
    dim3 block(32u, (unsigned)nw, 1u);
    const unsigned long long sb = (unsigned long long)g_slot_bytes;
    const unsigned long long og = (unsigned long long)g_tensor_off[0];
    const unsigned long long ou = (unsigned long long)g_tensor_off[1];
    switch (nw) {
        case 1: mmvq_q4_K_gateup_swiglu_kernel<1><<<grid, block, 0, g_expert_stream>>>(g_pool, sb, og, ou, g_exslots, xq8, g_exdx, ffn_dim, nblk); break;
        case 2: mmvq_q4_K_gateup_swiglu_kernel<2><<<grid, block, 0, g_expert_stream>>>(g_pool, sb, og, ou, g_exslots, xq8, g_exdx, ffn_dim, nblk); break;
        case 3: mmvq_q4_K_gateup_swiglu_kernel<3><<<grid, block, 0, g_expert_stream>>>(g_pool, sb, og, ou, g_exslots, xq8, g_exdx, ffn_dim, nblk); break;
        default:mmvq_q4_K_gateup_swiglu_kernel<4><<<grid, block, 0, g_expert_stream>>>(g_pool, sb, og, ou, g_exslots, xq8, g_exdx, ffn_dim, nblk); break;
    }
    CUDA_CHECK_LAST();

    const int nq = n * ffn_dim;
    launch_quantize_q8_1(g_exdx, (block_q8_1_c *)g_exdx_q8, nq, g_expert_stream);
    CUDA_CHECK_LAST();
    g_batch_active = 1;
    // Sync here (not a download) so the engine's gu/down phase timers stay separable; the
    // activation itself never leaves the device.
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
}

// Down projection of the GLU-fused stage: reads the q8_1 activation already on device, writes
// down_out [n][out_dim] to the host.
void gpu_expert_moe_down(int n, ggml_type_t type, int in_dim, int out_dim, float *down_out) {
    block_q8_1_c *aq8 = (block_q8_1_c *)g_exdx_q8;
    if (type == GGML_TYPE_Q6_K)
        launch_mmvq_q6_K_expert((unsigned long long)g_tensor_off[2], aq8, in_dim/QK8_1, g_exdbat, n, out_dim, in_dim, g_expert_stream, g_exslots);
    else
        launch_mmvq_q4_K_expert((unsigned long long)g_tensor_off[2], aq8, in_dim/QK8_1, g_exdbat, n, out_dim, in_dim, g_expert_stream, g_exslots);
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
    CUDA_CHECK(cudaMemcpy(down_out, g_exdbat, (size_t)n * out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    g_batch_active = 0;
}

// ---------------------------------------------------------------------------
// Fused batch kernels: ONE grid launch computes gate+up (or down) for ALL selected experts of
// a layer, instead of n*2 (gate+up) or n (down) separate 1-thread-per-row kernel launches. Each
// output element (expert e, row r) is still computed by a SINGLE thread doing the exact scalar
// accumulation sequence of the per-expert kernel, so the result is bit-identical. `in_stride`
// is the per-expert input stride (0 => all experts share one input vector, as for gate/up;
// in_dim => each expert reads its own slice, as for down). Weights are read from the LFRU pool
// at slot = d_slots[e].
// ---------------------------------------------------------------------------
__global__ void mul_mat_vec_q4_K_batch_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const float * __restrict__ x, float * __restrict__ y,
        int n, int out_dim, int in_dim, int in_stride, int total) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = t / out_dim;
    const int row = t - e * out_dim;
    const int nblk = in_dim / QK_K;
    const block_q4_K * __restrict__ rw =
        (const block_q4_K*)(pool + (unsigned long long)d_slots[e] * slot_bytes + off) + (size_t)row * nblk;
    const float * __restrict__ xin = x + (size_t)e * in_stride;

    float sum = 0.0f;
    int cidx = 0;
    for (int b = 0; b < nblk; ++b) {
        const block_q4_K &blk = rw[b];
        const float d   = h2f(blk.d);
        const float mn  = h2f(blk.dmin);
        const uint8_t *q = blk.qs;
        int is = 0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t sc, m;
            if (is < 4) { sc = blk.scales[is] & 63; m = blk.scales[is + 4] & 63; }
            else        { sc = (blk.scales[is + 4] & 0xF) | ((blk.scales[is - 4] >> 6) << 4);
                           m  = (blk.scales[is + 4] >> 4)  | ((blk.scales[is]     >> 6) << 4); }
            const float d1 = d * (float)sc;
            const float m1 = mn * (float)m;
            if (is + 1 < 4) { sc = blk.scales[is + 1] & 63; m = blk.scales[is + 5] & 63; }
            else            { sc = (blk.scales[is + 5] & 0xF) | ((blk.scales[is - 3] >> 6) << 4);
                               m  = (blk.scales[is + 5] >> 4)  | ((blk.scales[is + 1] >> 6) << 4); }
            const float d2 = d * (float)sc;
            const float m2 = mn * (float)m;
            for (int l = 0; l < 32; ++l) {
                // Same no-FMA outer accumulation as mul_mat_vec_q4_K_kernel (matches mat_vec_q4_K()).
                const float tq = __fmaf_rn(d1, (float)(q[l] & 0xF), -m1);
                sum = __fadd_rn(sum, __fmul_rn(tq, xin[cidx]));
                cidx++;
            }
            for (int l = 0; l < 32; ++l) {
                const float tq = __fmaf_rn(d2, (float)(q[l] >> 4), -m2);
                sum = __fadd_rn(sum, __fmul_rn(tq, xin[cidx]));
                cidx++;
            }
            q += 32;
            is += 2;
        }
    }
    y[t] = sum;
}

__global__ void mul_mat_vec_q6_K_batch_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const float * __restrict__ x, float * __restrict__ y,
        int n, int out_dim, int in_dim, int in_stride, int total) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = t / out_dim;
    const int row = t - e * out_dim;
    const int nblk = in_dim / QK_K;
    const block_q6_K * __restrict__ rw =
        (const block_q6_K*)(pool + (unsigned long long)d_slots[e] * slot_bytes + off) + (size_t)row * nblk;
    const float * __restrict__ xin = x + (size_t)e * in_stride;

    float sum = 0.0f;
    for (int c = 0; c < in_dim; ++c) {
        const block_q6_K &b = rw[c / QK_K];
        const int chunk = (c % QK_K) / 128;
        const int p128  = (c % QK_K) % 128;
        const int l     = p128 % 32;
        const int slot  = p128 / 32;
        const uint8_t ql = b.ql[chunk * 64 + l + ((slot & 1) ? 32 : 0)];
        const uint8_t qh = b.qh[chunk * 32 + l];
        const uint8_t qn = ((slot < 2) ? (ql & 0xF) : (ql >> 4)) |
                           (((qh >> (slot * 2)) & 3) << 4);
        const int8_t q = (int8_t)(qn - 32);
        const int8_t s = b.scales[chunk * 8 + (l / 16) + slot * 2];
        const float yv = h2f(b.d) * (float)s * (float)q;
        sum = __fadd_rn(sum, __fmul_rn(yv, xin[c]));
    }
    y[t] = sum;
}

// Fused gate+up launch: upload norm_x (shared) + the per-position slot table, then ONE kernel
// per tensor (gate, up) computes all experts' outputs into g_exgbat/g_exubbat. in_stride=0
// (all experts share norm_x).
void gpu_expert_batch_gateup_launch_fused(const int *slots, int n, const float *norm_x, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exgx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    CUDA_CHECK(cudaMemcpyAsync(g_exslots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));

    if (!tuff_exact()) {
        // P3 fast path: quantize norm_x to q8_1 once, then one mmvq launch per tensor.
        block_q8_1_c *xq8 = (block_q8_1_c *)g_exgx_q8;
        launch_quantize_q8_1(g_exgx, xq8, in_dim, g_expert_stream);
        CUDA_CHECK_LAST();
        launch_mmvq_q4_K_expert((unsigned long long)g_tensor_off[0], xq8, 0, g_exgbat, n, out_dim, in_dim, g_expert_stream, g_exslots);
        launch_mmvq_q4_K_expert((unsigned long long)g_tensor_off[1], xq8, 0, g_exubbat, n, out_dim, in_dim, g_expert_stream, g_exslots);
        g_batch_active = 1;
        return;
    }

    // Threads per block = 256; each thread still computes ONE (expert,row) element with the exact
    // scalar accumulation, so results are bit-identical to 1-thread-per-block but occupancy is
    // thread-limited (not block-limited) -> much better latency hiding.
    const int B = 256;
    const int total = n * out_dim;
    const int blocks = (total + B - 1) / B;
    mul_mat_vec_q4_K_batch_kernel<<<blocks, B, 0, g_expert_stream>>>(
        g_pool, (unsigned long long)g_slot_bytes, (unsigned long long)g_tensor_off[0],
        g_exslots, g_exgx, g_exgbat, n, out_dim, in_dim, 0, total);
    mul_mat_vec_q4_K_batch_kernel<<<blocks, B, 0, g_expert_stream>>>(
        g_pool, (unsigned long long)g_slot_bytes, (unsigned long long)g_tensor_off[1],
        g_exslots, g_exgx, g_exubbat, n, out_dim, in_dim, 0, total);
    CUDA_CHECK_LAST();
    g_batch_active = 1;
}

// Fused down launch: upload all experts' act_exp (contiguous, [n*ffn]) + the slot table, then
// ONE kernel computes all down outputs into g_exdbat. in_stride=in_dim (per-expert input slice).
void gpu_expert_batch_down_launch_fused(int n, ggml_type_t type, const float *act_all, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exdx, act_all, (size_t)n * in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));

    if (!tuff_exact()) {
        const int nq = n * in_dim;
        block_q8_1_c *aq8 = (block_q8_1_c *)g_exdx_q8;
        launch_quantize_q8_1(g_exdx, aq8, nq, g_expert_stream);
        CUDA_CHECK_LAST();
        if (type == GGML_TYPE_Q6_K)
            launch_mmvq_q6_K_expert((unsigned long long)g_tensor_off[2], aq8, in_dim/QK8_1, g_exdbat, n, out_dim, in_dim, g_expert_stream, g_exslots);
        else
            launch_mmvq_q4_K_expert((unsigned long long)g_tensor_off[2], aq8, in_dim/QK8_1, g_exdbat, n, out_dim, in_dim, g_expert_stream, g_exslots);
        g_batch_active = 1;
        return;
    }

    const int B = 256;
    const int total = n * out_dim;
    const int blocks = (total + B - 1) / B;
    if (type == GGML_TYPE_Q6_K)
        mul_mat_vec_q6_K_batch_kernel<<<blocks, B, 0, g_expert_stream>>>(
            g_pool, (unsigned long long)g_slot_bytes, (unsigned long long)g_tensor_off[2],
            g_exslots, g_exdx, g_exdbat, n, out_dim, in_dim, in_dim, total);
    else
        mul_mat_vec_q4_K_batch_kernel<<<blocks, B, 0, g_expert_stream>>>(
            g_pool, (unsigned long long)g_slot_bytes, (unsigned long long)g_tensor_off[2],
            g_exslots, g_exdx, g_exdbat, n, out_dim, in_dim, in_dim, total);
    CUDA_CHECK_LAST();
    g_batch_active = 1;
}

// Self-test for the routed-expert mat-vec kernels. For a representative set of layers (layer 0,
// whose down is Q4_K, and the deepest Q6_K-down layer), run gate/up/down for ALL 256 experts
// through both the CPU (mat_vec_cpu, the exact engine reference) and the matching GPU kernel
// (Q4_K kernel for gate/up and for Q4_K-down layers; Q6_K scalar kernel for Q6_K-down layers)
// and assert 0 differing floats (bit-exact). Gated by TUFF_EXPERT_SELFTEST=1.
void gpu_selftest_q4k(const model_t *m) {
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;

    // Pick layers: 0 (Q4_K down) plus the first Q6_K-down layer if any.
    int q6_layer = -1;
    for (uint32_t i = 0; i < m->block_count; ++i)
        if ((ggml_type_t)m->layers[i].ffn_down_exps.info.type == GGML_TYPE_Q6_K) { q6_layer = (int)i; break; }
    const int layers[2] = { 0, q6_layer };
    const int nlay = q6_layer >= 0 ? 2 : 1;

    const char *names[3] = { "gate", "up", "down" };
    const int in_d[3]  = { dim, dim, ffn };
    const int out_d[3] = { ffn, ffn, dim };

    int total_diff = 0;
    for (int li = 0; li < nlay; ++li) {
        const model_layer_t *l = &m->layers[layers[li]];
        const tensor_t *src_t[3] = { &l->ffn_gate_exps, &l->ffn_up_exps, &l->ffn_down_exps };
        for (int t = 0; t < 3; ++t) {
            const ggml_type_t type = (ggml_type_t)src_t[t]->info.type;
            int first_bad_expert = -1;
            double worst = 0.0;
            int expert_diffs = 0;
            for (int expert = 0; expert < 256; ++expert) {
                tensor_t gt = *src_t[t];
                size_t tb = ggml_row_size(type, in_d[t]) * out_d[t];
                gt.data = (const uint8_t*)src_t[t]->data + (size_t)expert * tb;
                gt.info.dims[1] = out_d[t];
                gt.info.type = type;

                void *dev = NULL;
                CUDA_CHECK(cudaMalloc(&dev, tb));
                CUDA_CHECK(cudaMemcpy(dev, gt.data, tb, cudaMemcpyHostToDevice));

                float *xin = (float*)malloc(in_d[t] * sizeof(float));
                float *yc  = (float*)malloc(out_d[t] * sizeof(float));
                float *yg  = (float*)malloc(out_d[t] * sizeof(float));
                for (int i = 0; i < in_d[t]; ++i) xin[i] = sinf(i * 0.013f);

                mat_vec_cpu(yc, &gt, xin);
                if (type == GGML_TYPE_Q6_K)
                    gpu_expert_mat_vec_q6(yg, dev, xin, in_d[t], out_d[t]);
                else
                    gpu_expert_mat_vec(yg, dev, xin, in_d[t], out_d[t]);

                double maxdiff = 0.0;
                int bad = 0;
                for (int i = 0; i < out_d[t]; ++i) {
                    double d = fabs((double)yc[i] - (double)yg[i]);
                    if (d > maxdiff) maxdiff = d;
                    if (yc[i] != yg[i]) bad++;
                }
                if (maxdiff > worst) worst = maxdiff;
                if (first_bad_expert < 0 && bad > 0) first_bad_expert = expert;
                expert_diffs += bad;
                total_diff += bad;

                CUDA_CHECK(cudaFree(dev));
                free(xin); free(yc); free(yg);
            }
            fprintf(stderr, "[SELFTEST] l=%d %s (type=%d, 256 experts): worst_maxdiff=%.3e "
                            "total_diffs=%d first_bad_expert=%d\n",
                    layers[li], names[t], (int)type, worst, expert_diffs, first_bad_expert);
        }
    }
    fprintf(stderr, "[SELFTEST] TOTAL differing floats across all tested experts: %d\n", total_diff);
}

// ---------------------------------------------------------------------------
// P4a: device-resident gated delta-net pre-stage. The engine's host elementwise work
// (conv-state shift + conv1d + SiLU over 8192 channels, per-head l2_norm, and the 32-element
// sigmoid/softplus/decay gates) moves onto the GPU; q_norm/k_norm/v_conv/decay/beta are
// produced directly in the device scratch g_dq/g_dk/g_dv/g_dp so the state-update kernel needs
// no host round-trip. g_dpqkv (the qkv projection output) is consumed in place: it is
// overwritten with qkv_conv, whose q/k/v slices are then normed/copied below.
//
// The conv1d is one thread per channel (8192 threads). Each channel only ever touches its own
// column across all conv_kernel rows of the conv state (host does memmove(c_state,
// c_state+qkv_dim, 3*qkv_dim*f32) then appends qkv_mixed -- i.e. row k of the shifted state is
// old row k+1, and the last row is the fresh projection). So a thread reads its 4 state values
// into registers, writes back the shifted column, and computes the conv sum from the shifted
// values -- no cross-thread hazard, no __syncthreads().

__global__ void swiglu_dev_kernel(float * __restrict__ act, const float * __restrict__ gate,
                                  const float * __restrict__ up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        act[i] = silu_g * up[i];
    }
}

// H15: the same SwiGLU over both columns' gate/up buffers (each [2][n] contiguous).
// Column c's slice is bit-identical to swiglu_dev_kernel on that column alone.
__global__ void swiglu_dev_ncols_kernel(float * __restrict__ act, const float * __restrict__ gate,
                                        const float * __restrict__ up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        act[i] = silu_g * up[i];
    }
}

__global__ void scale_shexp_kernel(float * __restrict__ out, const float * __restrict__ down,
                                   const float * __restrict__ ginp, int dim) {
    float gate_val = 1.0f / (1.0f + expf(-ginp[0]));
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        out[i] = gate_val * down[i];
    }
}

void gpu_shared_expert_dev(float *d_shexp_out, const model_layer_t *layer, const float *d_norm_x, int in_dim, int sh_ffn_dim) {
    const int dim = in_dim;
    // H11: quantize norm_x ONCE per layer, into g_d_norm_q8, and run the shared expert's
    // gate/up from it. The routed experts then reuse g_d_norm_q8 instead of re-quantizing the
    // same vector (gpu_expert_moe_gateup_swiglu_dev_col checks g_norm_q8_valid). On the
    // f32-activation fallback (exact, or TUFF_MMVQ_DUAL=0) no q8_1 of norm_x exists, so the
    // flag stays 0 and the routed experts quantize for themselves, exactly as before.
    if (!tuff_exact() && mmvq_dual_enabled()) {
        block_q8_1_c *xq;
        if (g_norm_q8_valid) {
            xq = (block_q8_1_c *)g_d_norm_q8;
        } else {
            xq = dense_quantize_x_to(d_norm_x, in_dim, norm_x_q8_ensure(in_dim));
            g_norm_q8_valid = 1;
        }
        const int nblk = in_dim / QK8_0;
        if (!g_d_shact_q8) {
            CUDA_CHECK(cudaMalloc(&g_d_shact_q8, (size_t)(sh_ffn_dim / QK8_1) * sizeof(block_q8_1_c)));
        }
        dim3 grid((unsigned)(sh_ffn_dim / 32), 1u, 1u);
        dim3 block(32u, 32u, 1u);
        launch_mmvq_shexp_swiglu_quant(
            (const block_q8_0*)layer->ffn_gate_shexp.dev,
            (const block_q8_0*)layer->ffn_up_shexp.dev,
            (block_q8_1_c*)g_d_shact_q8,
            sh_ffn_dim, xq, nblk);

        mmvq_q8_0_rows((const block_q8_0*)layer->ffn_down_shexp.dev, (const block_q8_1_c*)g_d_shact_q8, g_d_shdown, sh_ffn_dim, dim, false);
    } else {
        launch_mat_vec_q8_0_dual((const block_q8_0*)layer->ffn_gate_shexp.dev, g_shg, sh_ffn_dim,
                                 (const block_q8_0*)layer->ffn_up_shexp.dev,   g_shu, sh_ffn_dim,
                                 d_norm_x, in_dim);
        swiglu_dev_kernel<<<(sh_ffn_dim + 255) / 256, 256>>>(g_d_shact, g_shg, g_shu, sh_ffn_dim);
        CUDA_CHECK_LAST();
        g_norm_q8_valid = 0;
        launch_mat_vec_q8_0((const block_q8_0*)layer->ffn_down_shexp.dev, g_d_shact, g_d_shdown, sh_ffn_dim, dim);
    }

    if (layer->ffn_gate_inp_shexp.dev) {
        launch_mat_vec_f32 ((const float*)layer->ffn_gate_inp_shexp.dev, d_norm_x, g_shgi, in_dim, 1);
        scale_shexp_kernel<<<(dim + 255) / 256, 256>>>(d_shexp_out, g_d_shdown, g_shgi, dim);
        CUDA_CHECK_LAST();
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_shexp_out, g_d_shdown, (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

void gpu_expert_moe_gateup_swiglu_dev_col(const int *slots, int n, const float *d_norm_x,
                                          int in_dim, int ffn_dim, int col) {
    int   *ex_slots = g_exslots + (size_t)col * g_ex_stride_slots;
    float *ex_dx    = g_exdx    + (size_t)col * g_ex_stride_dx;
    if (slots) {
        CUDA_CHECK(cudaMemcpyAsync(ex_slots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    }

    block_q8_1_c *xq8;
    if (col == 0 && g_norm_q8_valid) {
        // H11: the shared expert already quantized this layer's norm_x into g_d_norm_q8
        // (same quantize_q8_1_kernel, same layout). Reuse it instead of a second quantize.
        xq8 = (block_q8_1_c *)g_d_norm_q8;
    } else {
        xq8 = (block_q8_1_c *)((char*)g_exgx_q8 + (size_t)col * g_ex_stride_gxq8);
        launch_quantize_q8_1(d_norm_x, xq8, in_dim);
        CUDA_CHECK_LAST();
    }

    const int nblk = in_dim / QK_K;
    const unsigned long long sb = (unsigned long long)g_slot_bytes;
    const unsigned long long og = (unsigned long long)g_tensor_off[0];
    const unsigned long long ou = (unsigned long long)g_tensor_off[1];

    if (nblk <= 16) {
        dim3 g((unsigned)((ffn_dim + MMVQ_ROWS_PER_BLOCK - 1)/MMVQ_ROWS_PER_BLOCK), (unsigned)n, 1u);
        dim3 b(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
        mmvq_q4_K_gateup_swiglu_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<g, b>>>(
            g_pool, sb, og, ou, ex_slots, xq8, ex_dx, ffn_dim, nblk);
    } else {
        const int nw = mmvq_nwarps(nblk, QI4_K / VDR_Q4_K_Q8_1_MMVQ);
        dim3 grid((unsigned)ffn_dim, (unsigned)n, 1u);
        dim3 block(32u, (unsigned)nw, 1u);
        switch (nw) {
            case 1: mmvq_q4_K_gateup_swiglu_kernel<1><<<grid, block>>>(g_pool, sb, og, ou, ex_slots, xq8, ex_dx, ffn_dim, nblk); break;
            case 2: mmvq_q4_K_gateup_swiglu_kernel<2><<<grid, block>>>(g_pool, sb, og, ou, ex_slots, xq8, ex_dx, ffn_dim, nblk); break;
            case 3: mmvq_q4_K_gateup_swiglu_kernel<3><<<grid, block>>>(g_pool, sb, og, ou, ex_slots, xq8, ex_dx, ffn_dim, nblk); break;
            default:mmvq_q4_K_gateup_swiglu_kernel<4><<<grid, block>>>(g_pool, sb, og, ou, ex_slots, xq8, ex_dx, ffn_dim, nblk); break;
        }
    }
    CUDA_CHECK_LAST();

    const int nq = n * ffn_dim;
    launch_quantize_q8_1(ex_dx, (block_q8_1_c *)((char*)g_exdx_q8 + (size_t)col * g_ex_stride_dxq8), nq);
    CUDA_CHECK_LAST();
}

void gpu_expert_moe_gateup_swiglu_dev(const int *slots, int n, const float *d_norm_x, int in_dim, int ffn_dim) {
    gpu_expert_moe_gateup_swiglu_dev_col(slots, n, d_norm_x, in_dim, ffn_dim, 0);
}

void gpu_expert_moe_down_dev_col(int n, ggml_type_t type, int in_dim, int out_dim, int col) {
    block_q8_1_c *aq8 = (block_q8_1_c *)((char*)g_exdx_q8 + (size_t)col * g_ex_stride_dxq8);
    float *ex_dbat  = g_exdbat  + (size_t)col * g_ex_stride_dbat;
    // This column's slot table, not column 0's. Reading the wrong one made the
    // second token's down projection use the first token's experts; the output
    // collapsed to a repeated token and acceptance rose to 98% on the collapse.
    const int *ex_slots = g_exslots + (size_t)col * g_ex_stride_slots;
    if (type == GGML_TYPE_Q6_K)
        launch_mmvq_q6_K_expert((unsigned long long)g_tensor_off[2], aq8, in_dim/QK8_1, ex_dbat, n, out_dim, in_dim, 0, ex_slots);
    else
        launch_mmvq_q4_K_expert((unsigned long long)g_tensor_off[2], aq8, in_dim/QK8_1, ex_dbat, n, out_dim, in_dim, 0, ex_slots);
    CUDA_CHECK_LAST();
}

void gpu_expert_moe_down_dev(int n, ggml_type_t type, int in_dim, int out_dim) {
    gpu_expert_moe_down_dev_col(n, type, in_dim, out_dim, 0);
}

// ---------------------------------------------------------------------------
// H15: partitioned two-column routed experts.
//
// The two tokens' top-8 selections overlap 36-38%. A shared expert is evaluated
// against both activations with its weights read ONCE (ncols=2 kernels), the
// exclusive experts run the single-column path with an explicit destination-index
// table so the compressed slot list still lands each output at its original hit
// position. Every kernel above is bit-identical per column to the single-column
// kernels (same per-thread accumulation, warp reduction and per-element SwiGLU),
// so the combine sees exactly the values it always saw, in exactly its order.
// ---------------------------------------------------------------------------
const block_q8_1_c *ex_col_xq8(int col) {
    return (const block_q8_1_c *)((char*)g_exgx_q8 + (size_t)col * g_ex_stride_gxq8);
}
block_q8_1_c *ex_col_dxq8(int col) {
    return (block_q8_1_c *)((char*)g_exdx_q8 + (size_t)col * g_ex_stride_dxq8);
}
float *ex_col_dx(int col)  { return g_exdx   + (size_t)col * g_ex_stride_dx; }
float *ex_col_dbat(int col){ return g_exdbat + (size_t)col * g_ex_stride_dbat; }
int *ex_col_slots(int col){ return g_exslots + (size_t)col * g_ex_stride_slots; }

void gpu_expert_moe_normx_quant_col(const float *d_norm_x, int in_dim, int col) {
    if (g_expert_stream) {
        static cudaEvent_t ev = NULL;
        if (!ev) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev, 0));
        CUDA_CHECK(cudaStreamWaitEvent(g_expert_stream, ev, 0));
        launch_quantize_q8_1(d_norm_x, (block_q8_1_c*)ex_col_xq8(col), in_dim, g_expert_stream);
    } else {
        launch_quantize_q8_1(d_norm_x, (block_q8_1_c*)ex_col_xq8(col), in_dim);
    }
    CUDA_CHECK_LAST();
}

void gpu_expert_moe_normx_quant_host(const float *h_norm_x, int in_dim, int col) {
    float *gx = g_exgx + (size_t)col * (size_t)in_dim;
    CUDA_CHECK(cudaMemcpyAsync(gx, h_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    launch_quantize_q8_1(gx, (block_q8_1_c*)ex_col_xq8(col), in_dim, g_expert_stream);
    CUDA_CHECK_LAST();
}

cudaStream_t gpu_get_expert_stream(void) { return g_expert_stream; }

const void* gpu_get_expert_xq8_dev_ptr(int col) { return ex_col_xq8(col); }
float*      gpu_get_expert_dx_dev_ptr(int col)  { return ex_col_dx(col); }
void*       gpu_get_expert_dxq8_dev_ptr(int col){ return ex_col_dxq8(col); }
float*      gpu_get_expert_dbat_dev_ptr(int col){ return ex_col_dbat(col); }


template <int ncols>
static void launch_mmvq_q4_K_gateup_swiglu_ncols(const int *slots, int n,
                                                 const block_q8_1_c *vy0, const block_q8_1_c *vy1,
                                                 const int *d_dst0, const int *d_dst1,
                                                 float *act0, float *act1, int out_dim, int nblk) {
    const unsigned long long sb = (unsigned long long)g_slot_bytes;
    const unsigned long long og = (unsigned long long)g_tensor_off[0];
    const unsigned long long ou = (unsigned long long)g_tensor_off[1];
    if (nblk <= 16) {
        dim3 g((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1)/MMVQ_ROWS_PER_BLOCK), (unsigned)n, 1u);
        dim3 b(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
        mmvq_q4_K_gateup_swiglu_warprow_ncols_kernel<MMVQ_ROWS_PER_BLOCK, ncols><<<g, b>>>(
            g_pool, sb, og, ou, slots, vy0, vy1, d_dst0, d_dst1, act0, act1, out_dim, nblk);
    } else {
        const int nw = mmvq_nwarps(nblk, QI4_K / VDR_Q4_K_Q8_1_MMVQ);
        dim3 grid((unsigned)out_dim, (unsigned)n, 1u);
        dim3 block(32u, (unsigned)nw, 1u);
        switch (nw) {
            case 1: mmvq_q4_K_gateup_swiglu_ncols_kernel<1, ncols><<<grid, block>>>(g_pool, sb, og, ou, slots, vy0, vy1, d_dst0, d_dst1, act0, act1, out_dim, nblk); break;
            case 2: mmvq_q4_K_gateup_swiglu_ncols_kernel<2, ncols><<<grid, block>>>(g_pool, sb, og, ou, slots, vy0, vy1, d_dst0, d_dst1, act0, act1, out_dim, nblk); break;
            case 3: mmvq_q4_K_gateup_swiglu_ncols_kernel<3, ncols><<<grid, block>>>(g_pool, sb, og, ou, slots, vy0, vy1, d_dst0, d_dst1, act0, act1, out_dim, nblk); break;
            default:mmvq_q4_K_gateup_swiglu_ncols_kernel<4, ncols><<<grid, block>>>(g_pool, sb, og, ou, slots, vy0, vy1, d_dst0, d_dst1, act0, act1, out_dim, nblk); break;
        }
    }
    CUDA_CHECK_LAST();
}

template <int ncols>
static void launch_mmvq_expert_ncols(ggml_type_t type, const int *slots, int n,
                                     const block_q8_1_c *vy0, const block_q8_1_c *vy1, int y_stride_blocks,
                                     const int *d_dst0, const int *d_dst1,
                                     float *dst0, float *dst1, int out_dim, int nblk) {
    const unsigned long long off = (unsigned long long)g_tensor_off[2];
    if (type == GGML_TYPE_Q6_K) {
        if (nblk * (QI6_K / VDR_Q6_K_Q8_1_MMVQ) <= 64) {
            dim3 g((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1)/MMVQ_ROWS_PER_BLOCK), (unsigned)n, 1u);
            dim3 b(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
            mmvq_q6_K_expert_warprow_ncols_kernel<MMVQ_ROWS_PER_BLOCK, ncols><<<g, b>>>(
                g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk);
        } else {
            const int nw = mmvq_nwarps(nblk, QI6_K / VDR_Q6_K_Q8_1_MMVQ);
            dim3 grid((unsigned)out_dim, (unsigned)n, 1u);
            dim3 block(32u, (unsigned)nw, 1u);
            switch (nw) {
                case 1: mmvq_q6_K_expert_ncols_kernel<1, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
                case 2: mmvq_q6_K_expert_ncols_kernel<2, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
                case 3: mmvq_q6_K_expert_ncols_kernel<3, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
                default:mmvq_q6_K_expert_ncols_kernel<4, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
            }
        }
    } else {
        if (nblk * (QI4_K / VDR_Q4_K_Q8_1_MMVQ) <= 64) {
            dim3 g((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1)/MMVQ_ROWS_PER_BLOCK), (unsigned)n, 1u);
            dim3 b(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
            mmvq_q4_K_expert_warprow_ncols_kernel<MMVQ_ROWS_PER_BLOCK, ncols><<<g, b>>>(
                g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk);
        } else {
            const int nw = mmvq_nwarps(nblk, QI4_K / VDR_Q4_K_Q8_1_MMVQ);
            dim3 grid((unsigned)out_dim, (unsigned)n, 1u);
            dim3 block(32u, (unsigned)nw, 1u);
            switch (nw) {
                case 1: mmvq_q4_K_expert_ncols_kernel<1, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
                case 2: mmvq_q4_K_expert_ncols_kernel<2, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
                case 3: mmvq_q4_K_expert_ncols_kernel<3, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
                default:mmvq_q4_K_expert_ncols_kernel<4, ncols><<<grid, block>>>(g_pool, (unsigned long long)g_slot_bytes, off, slots, vy0, vy1, y_stride_blocks, d_dst0, d_dst1, dst0, dst1, out_dim, nblk); break;
            }
        }
    }
    CUDA_CHECK_LAST();
}

// Shared experts: gate/up/SwiGLU over both columns with one weight read. Both
// columns' normalized inputs are quantized here (caller may pass either column's
// base; the two are contiguous in the n2 trunk, but kept distinct for clarity).
void gpu_expert_moe_gateup_swiglu_dev_n2_shared(const int *slots, const int *dst0, const int *dst1, int n,
                                                const float *d_norm_x0, const float *d_norm_x1,
                                                int in_dim, int ffn_dim) {
    int *ex_slots = ex_col_slots(0);
    int *d_dst0 = g_d_exdst + 0 * EXPERT_BATCH_MAX;
    int *d_dst1 = g_d_exdst + 1 * EXPERT_BATCH_MAX;
    CUDA_CHECK(cudaMemcpyAsync(ex_slots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    if (dst0) CUDA_CHECK(cudaMemcpyAsync(d_dst0, dst0, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    if (dst1) CUDA_CHECK(cudaMemcpyAsync(d_dst1, dst1, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    gpu_expert_moe_normx_quant_col(d_norm_x0, in_dim, 0);
    gpu_expert_moe_normx_quant_col(d_norm_x1, in_dim, 1);
    launch_mmvq_q4_K_gateup_swiglu_ncols<2>(ex_slots, n,
        ex_col_xq8(0), ex_col_xq8(1),
        dst0 ? d_dst0 : NULL, dst1 ? d_dst1 : NULL,
        ex_col_dx(0), ex_col_dx(1), ffn_dim, in_dim / QK_K);
}

// One column's exclusive experts, single-column kernel with a destination table.
void gpu_expert_moe_gateup_swiglu_dev_col_dst(const int *slots, const int *dst, int n,
                                              const float *d_norm_x, int in_dim, int ffn_dim, int col) {
    int *ex_slots = ex_col_slots(col);
    int *d_dst = g_d_exdst + (2 + col) * EXPERT_BATCH_MAX;
    CUDA_CHECK(cudaMemcpyAsync(ex_slots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    if (dst) CUDA_CHECK(cudaMemcpyAsync(d_dst, dst, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    gpu_expert_moe_normx_quant_col(d_norm_x, in_dim, col);
    launch_mmvq_q4_K_gateup_swiglu_ncols<1>(ex_slots, n,
        ex_col_xq8(col), NULL,
        dst ? d_dst : NULL, NULL,
        ex_col_dx(col), NULL, ffn_dim, in_dim / QK_K);
}

// Quantize column col's activation buffer (the first n*ffn_dim floats) so the down
// kernels can read it. n must be the FULL hit count of the column (shared+exclusive),
// because the shared experts' act lands at their original hit positions.
void gpu_expert_moe_act_quant_col(int n, int ffn_dim, int col) {
    const int nq = n * ffn_dim;
    launch_quantize_q8_1(ex_col_dx(col), ex_col_dxq8(col), nq);
    CUDA_CHECK_LAST();
}

// Shared experts: down over both columns with one weight read. The slot table is
// re-uploaded here: the exclusive-col0 gateup overwrote region 0 of g_exslots.
void gpu_expert_moe_down_dev_n2_shared(const int *slots, const int *dst0, const int *dst1, int n,
                                       ggml_type_t type, int in_dim, int out_dim) {
    int *ex_slots = ex_col_slots(0);
    int *d_dst0 = g_d_exdst + 0 * EXPERT_BATCH_MAX;
    int *d_dst1 = g_d_exdst + 1 * EXPERT_BATCH_MAX;
    CUDA_CHECK(cudaMemcpyAsync(ex_slots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    launch_mmvq_expert_ncols<2>(type, ex_slots, n,
        ex_col_dxq8(0), ex_col_dxq8(1), in_dim/QK8_1,
        dst0 ? d_dst0 : NULL, dst1 ? d_dst1 : NULL,
        ex_col_dbat(0), ex_col_dbat(1), out_dim, in_dim / QK_K);
}

// One column's exclusive experts, single-column down with a destination table.
void gpu_expert_moe_down_dev_col_dst(const int *slots, const int *dst, int n,
                                     ggml_type_t type, int in_dim, int out_dim, int col) {
    int *ex_slots = ex_col_slots(col);
    int *d_dst = g_d_exdst + (2 + col) * EXPERT_BATCH_MAX;
    CUDA_CHECK(cudaMemcpyAsync(ex_slots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    launch_mmvq_expert_ncols<1>(type, ex_slots, n,
        ex_col_dxq8(col), NULL, in_dim/QK8_1,
        dst ? d_dst : NULL, NULL,
        ex_col_dbat(col), NULL, out_dim, in_dim / QK_K);
}

// ---------------------------------------------------------------------------
// Task Q1.2: Generalized Union Warprow Kernels across N columns (1 <= N <= 8)
// ---------------------------------------------------------------------------
template <int rows_per_block>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q4_K_gateup_swiglu_union_warprow_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes,
        unsigned long long off_gate, unsigned long long off_up,
        const int * __restrict__ d_slots,
        const int * __restrict__ d_m,
        const int * __restrict__ d_cols,
        const int * __restrict__ d_dst,
        float * __restrict__ ex_dx,
        size_t ex_stride_dx,
        void * __restrict__ ex_gx_q8,
        size_t ex_stride_gxq8,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI4_K;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row     = blockIdx.x * rows_per_block + warp_id;
    const int e       = blockIdx.y;
    if (row >= out_dim) return;

    const char * __restrict__ base = pool + (unsigned long long)d_slots[e]*slot_bytes;
    const block_q4_K * __restrict__ wg = (const block_q4_K *)(base + off_gate) + (size_t)row*nblk;
    const block_q4_K * __restrict__ wu = (const block_q4_K *)(base + off_up)   + (size_t)row*nblk;

    const int kqs = vdr * (lane_id % tpsb);
    const int m = d_m[e];
    float tg[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float tu[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kbx = lane_id / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q4_K *wg_b = wg + kbx;
        const block_q4_K *wu_b = wu + kbx;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < m) {
                int col = d_cols[e * 8 + j];
                const block_q8_1_c *vy = (const block_q8_1_c *)((const char *)ex_gx_q8 + (size_t)col * ex_stride_gxq8);
                const block_q8_1_c *yb = vy + kbx * (QK_K / QK8_1);
                tg[j] += vec_dot_q4_K_q8_1_c(wg_b, yb, kqs);
                tu[j] += vec_dot_q4_K_q8_1_c(wu_b, yb, kqs);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < m) {
            float g = warp_reduce_sum_f(tg[j]);
            float u = warp_reduce_sum_f(tu[j]);
            if (lane_id == 0) {
                int col = d_cols[e * 8 + j];
                int dst_k = d_dst[e * 8 + j];
                float *act_col = ex_dx + (size_t)col * ex_stride_dx;
                act_col[(size_t)dst_k * out_dim + row] = (g / (1.0f + expf(-g))) * u;
            }
        }
    }
}

template <int rows_per_block>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q4_K_expert_union_warprow_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const int * __restrict__ d_m,
        const int * __restrict__ d_cols,
        const int * __restrict__ d_dst,
        float * __restrict__ ex_dbat,
        size_t ex_stride_dbat,
        const void * __restrict__ ex_dxq8,
        size_t ex_stride_dxq8,
        int y_stride_blocks,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int tpsb = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI4_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;
    const int e = blockIdx.y;

    const block_q4_K * __restrict__ rw =
        (const block_q4_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;

    const int kqs = vdr * (threadIdx.x % tpsb);
    const int m = d_m[e];
    float tmp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q4_K *rw_b = rw + kbx;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < m) {
                int col = d_cols[e * 8 + j];
                int dst_k = d_dst[e * 8 + j];
                const block_q8_1_c *vy = (const block_q8_1_c *)((const char *)ex_dxq8 + (size_t)col * ex_stride_dxq8);
                const block_q8_1_c *y = vy + (size_t)dst_k * y_stride_blocks + kbx * (QK_K / QK8_1);
                tmp[j] += vec_dot_q4_K_q8_1_c(rw_b, y, kqs);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < m) {
            float t = warp_reduce_sum_f(tmp[j]);
            if (threadIdx.x == 0) {
                int col = d_cols[e * 8 + j];
                int dst_k = d_dst[e * 8 + j];
                float *dst_col = ex_dbat + (size_t)col * ex_stride_dbat;
                dst_col[(size_t)dst_k * out_dim + row] = t;
            }
        }
    }
}

template <int rows_per_block>
__launch_bounds__(rows_per_block*32, 1)
__global__ void mmvq_q6_K_expert_union_warprow_kernel(
        const char * __restrict__ pool, unsigned long long slot_bytes, unsigned long long off,
        const int * __restrict__ d_slots,
        const int * __restrict__ d_m,
        const int * __restrict__ d_cols,
        const int * __restrict__ d_dst,
        float * __restrict__ ex_dbat,
        size_t ex_stride_dbat,
        const void * __restrict__ ex_dxq8,
        size_t ex_stride_dxq8,
        int y_stride_blocks,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr*32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;
    const int e = blockIdx.y;

    const block_q6_K * __restrict__ rw =
        (const block_q6_K *)(pool + (unsigned long long)d_slots[e]*slot_bytes + off) + (size_t)row*nblk;

    const int kqs = vdr * (threadIdx.x % tpsb);
    const int m = d_m[e];
    float tmp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q6_K *rw_b = rw + kbx;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < m) {
                int col = d_cols[e * 8 + j];
                int dst_k = d_dst[e * 8 + j];
                const block_q8_1_c *vy = (const block_q8_1_c *)((const char *)ex_dxq8 + (size_t)col * ex_stride_dxq8);
                const block_q8_1_c *y = vy + (size_t)dst_k * y_stride_blocks + kbx * (QK_K / QK8_1);
                tmp[j] += vec_dot_q6_K_q8_1_c(rw_b, y, kqs);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < m) {
            float t = warp_reduce_sum_f(tmp[j]);
            if (threadIdx.x == 0) {
                int col = d_cols[e * 8 + j];
                int dst_k = d_dst[e * 8 + j];
                float *dst_col = ex_dbat + (size_t)col * ex_stride_dbat;
                dst_col[(size_t)dst_k * out_dim + row] = t;
            }
        }
    }
}

extern "C" void gpu_expert_moe_union_dev(const int *slots, const int *m_arr, const int *cols, const int *dst,
                                         int n_unique, const float *d_norm_batch, ggml_type_t down_type,
                                         int in_dim, int ffn_dim, int n_cols) {
    if (n_unique <= 0) return;
    CUDA_CHECK(cudaMemcpyAsync(g_d_union_slots, slots, (size_t)n_unique * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));
    CUDA_CHECK(cudaMemcpyAsync(g_d_union_m, m_arr, (size_t)n_unique * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));
    CUDA_CHECK(cudaMemcpyAsync(g_d_union_cols, cols, (size_t)n_unique * 8 * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));
    CUDA_CHECK(cudaMemcpyAsync(g_d_union_dst, dst, (size_t)n_unique * 8 * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));

    for (int c = 0; c < n_cols; ++c) {
        gpu_expert_moe_normx_quant_col(d_norm_batch + (size_t)c * in_dim, in_dim, c);
    }

    const unsigned long long sb = (unsigned long long)g_slot_bytes;
    const unsigned long long og = (unsigned long long)g_tensor_off[0];
    const unsigned long long ou = (unsigned long long)g_tensor_off[1];
    const unsigned long long od = (unsigned long long)g_tensor_off[2];

    dim3 g_gu((unsigned)((ffn_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), (unsigned)n_unique, 1u);
    dim3 b_gu(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q4_K_gateup_swiglu_union_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<g_gu, b_gu, 0, g_expert_stream>>>(
        g_pool, sb, og, ou, g_d_union_slots, g_d_union_m, g_d_union_cols, g_d_union_dst,
        g_exdx, g_ex_stride_dx, g_exgx_q8, g_ex_stride_gxq8, ffn_dim, in_dim / QK_K);
    CUDA_CHECK_LAST();

    for (int c = 0; c < n_cols; ++c) {
        launch_quantize_q8_1(ex_col_dx(c), ex_col_dxq8(c), 8 * ffn_dim, g_expert_stream);
    }

    dim3 g_dn((unsigned)((in_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), (unsigned)n_unique, 1u);
    dim3 b_dn(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    if (down_type == GGML_TYPE_Q6_K) {
        mmvq_q6_K_expert_union_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<g_dn, b_dn, 0, g_expert_stream>>>(
            g_pool, sb, od, g_d_union_slots, g_d_union_m, g_d_union_cols, g_d_union_dst,
            g_exdbat, g_ex_stride_dbat, g_exdx_q8, g_ex_stride_dxq8, ffn_dim / QK8_1, in_dim, ffn_dim / QK_K);
    } else {
        mmvq_q4_K_expert_union_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<g_dn, b_dn, 0, g_expert_stream>>>(
            g_pool, sb, od, g_d_union_slots, g_d_union_m, g_d_union_cols, g_d_union_dst,
            g_exdbat, g_ex_stride_dbat, g_exdx_q8, g_ex_stride_dxq8, ffn_dim / QK8_1, in_dim, ffn_dim / QK_K);
    }
    CUDA_CHECK_LAST();
}

extern "C" void gpu_expert_moe_download_down_hits(float *host_dst, int n_cols, int dim) {
    for (int c = 0; c < n_cols; ++c) {
        float *gpu_dbat_c = ex_col_dbat(c);
        CUDA_CHECK(cudaMemcpyAsync(host_dst + (size_t)c * 8 * dim,
                                   gpu_dbat_c,
                                   (size_t)8 * dim * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   g_expert_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(g_expert_stream));
}

// ---------------------------------------------------------------------------
// Fused device top-k MoE routing (Ported from llama.cpp ggml-cuda/topk-moe.cu)
// Warp-level softmax + iterative argmax top-k reduction.
// ---------------------------------------------------------------------------


__global__ void moe_combine_resident_kernel(
    float * __restrict__ res_x,
    const float * __restrict__ exdbat,
    const float * __restrict__ top_values,
    const float * __restrict__ shexp_out,
    int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float sum = (shexp_out ? shexp_out[i] : 0.0f);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        sum += top_values[k] * exdbat[k * dim + i];
    }
    res_x[i] += sum;
}

void gpu_expert_resident_layer_dev(int layer_idx, ggml_type_t down_type, const float *d_norm_x, float *d_res_x, const float *d_shexp_out, int in_dim, int ffn_dim, int n_expert) {
    launch_topk_moe_resident(g_d_router_logits, g_d_top_values, g_exslots, layer_idx);
    CUDA_CHECK_LAST();

    gpu_expert_moe_gateup_swiglu_dev(NULL, 8, d_norm_x, in_dim, ffn_dim);
    gpu_expert_moe_down_dev(8, down_type, ffn_dim, in_dim);

    moe_combine_resident_kernel<<<(in_dim + 255)/256, 256>>>(d_res_x, g_exdbat, g_d_top_values, d_shexp_out, in_dim);
    CUDA_CHECK_LAST();
}

__global__ void moe_combine_accum_kernel(
    float * __restrict__ res_x,
    const float * __restrict__ exdbat,      // [n_hits * dim]
    const int * __restrict__ hit_pos,        // [n_hits] position in top_values (0..7)
    int n_hits,
    const float * __restrict__ top_values,   // [8]
    const float * __restrict__ shexp_out,    // [dim]
    const float * __restrict__ miss_sum,     // [dim] or NULL
    int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float sum = (shexp_out ? shexp_out[i] : 0.0f);
    for (int h = 0; h < n_hits; ++h) {
        int k = hit_pos[h];
        float w = top_values[k];
        sum += w * exdbat[h * dim + i];
    }
    if (miss_sum) {
        sum += miss_sum[i];
    }
    res_x[i] += sum;
}

__global__ void moe_combine_accum_vec4_kernel(
    float4 * __restrict__ res_x,
    const float4 * __restrict__ exdbat,      // [n_hits * (dim/4)]
    const int * __restrict__ hit_pos,        // [n_hits] position in top_values (0..7)
    int n_hits,
    const float * __restrict__ top_values,   // [8]
    const float4 * __restrict__ shexp_out,    // [dim/4]
    const float4 * __restrict__ miss_sum,     // [dim/4] or NULL
    int dim4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim4) return;

    float4 res = res_x[i];
    float4 sh = shexp_out ? shexp_out[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float sum_x = sh.x;
    float sum_y = sh.y;
    float sum_z = sh.z;
    float sum_w = sh.w;

    for (int h = 0; h < n_hits; ++h) {
        int k = hit_pos[h];
        float w = top_values[k];
        float4 ex = exdbat[h * dim4 + i];
        sum_x += w * ex.x;
        sum_y += w * ex.y;
        sum_z += w * ex.z;
        sum_w += w * ex.w;
    }
    if (miss_sum) {
        float4 m = miss_sum[i];
        sum_x += m.x;
        sum_y += m.y;
        sum_z += m.z;
        sum_w += m.w;
    }
    res.x += sum_x;
    res.y += sum_y;
    res.z += sum_z;
    res.w += sum_w;
    res_x[i] = res;
}

void gpu_moe_combine_accum_col(float *d_res_x, const int *hit_pos, int n_hits,
                               const float *top_values, const float *d_shexp_out,
                               const float *host_miss_sum, int dim, int col) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_top_values, top_values, 8 * sizeof(float), cudaMemcpyHostToDevice));
    if (n_hits > 0) {
        CUDA_CHECK(cudaMemcpyAsync(g_d_hit_pos, hit_pos, (size_t)n_hits * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (host_miss_sum) {
        CUDA_CHECK(cudaMemcpyAsync(g_d_miss_sum, host_miss_sum, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (tuff_h50()) {
        const int dim4 = dim / 4;
        moe_combine_accum_vec4_kernel<<<(dim4 + 255) / 256, 256>>>(
            (float4*)d_res_x, (const float4*)(g_exdbat + (size_t)col * g_ex_stride_dbat),
            g_d_hit_pos, n_hits, g_d_top_values, (const float4*)d_shexp_out,
            host_miss_sum ? (const float4*)g_d_miss_sum : NULL, dim4);
    } else {
        moe_combine_accum_kernel<<<(dim + 255) / 256, 256>>>(
            d_res_x, g_exdbat + (size_t)col * g_ex_stride_dbat, g_d_hit_pos, n_hits,
            g_d_top_values, d_shexp_out, host_miss_sum ? g_d_miss_sum : NULL, dim);
    }
    CUDA_CHECK_LAST();
}

void gpu_moe_combine_accum(float *d_res_x, const int *hit_pos, int n_hits,
                           const float *top_values, const float *d_shexp_out,
                           const float *host_miss_sum, int dim) {
    gpu_moe_combine_accum_col(d_res_x, hit_pos, n_hits, top_values, d_shexp_out, host_miss_sum, dim, 0);
}

__global__ void qwen_moe_swiglu_pre_gu_kernel(const float * __restrict__ pre_gu,
                                              float * __restrict__ act,
                                              unsigned miss_mask,
                                              int ffn_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y; // 0..7
    if (row >= ffn_dim) return;
    if (!((miss_mask >> k) & 1u)) return;
    float g = pre_gu[(2 * k + 0) * ffn_dim + row];
    float u = pre_gu[(2 * k + 1) * ffn_dim + row];
    act[(size_t)k * ffn_dim + row] = (g / (1.0f + expf(-g))) * u;
}

void gpu_qwen_moe_swiglu_pre_gu(cudaStream_t stream, const float *d_pre_gu,
                                float *d_act, void *d_act_q8,
                                unsigned miss_mask, int ffn_dim) {
    if (miss_mask == 0u) return;
    dim3 grid((unsigned)((ffn_dim + 255) / 256), 8u, 1u);
    dim3 block(256u, 1u, 1u);
    qwen_moe_swiglu_pre_gu_kernel<<<grid, block, 0, stream>>>(d_pre_gu, d_act, miss_mask, ffn_dim);
    CUDA_CHECK_LAST();
    for (int k = 0; k < 8; ++k) {
        if (!((miss_mask >> k) & 1u)) continue;
        launch_quantize_q8_1(
            d_act + (size_t)k * ffn_dim, (block_q8_1_c*)d_act_q8 + (size_t)k * (ffn_dim / 32), ffn_dim, stream);
    }
    CUDA_CHECK_LAST();
}

__global__ void qwen_moe_combine_unified_kernel(
    float * __restrict__ res_x,
    const float * __restrict__ exdbat,        // [n_hits * dim]
    const int * __restrict__ hit_pos,         // [n_hits] maps hit h -> k in 0..7
    int n_hits,
    const float * __restrict__ pre_y,         // [8 * dim] (for missed k)
    unsigned miss_mask,
    const float * __restrict__ top_values,    // [8]
    const float * __restrict__ shexp_out,     // [dim] or NULL
    int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float y[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) y[k] = 0.0f;

    for (int h = 0; h < n_hits; ++h) {
        int k = hit_pos[h];
        y[k] = exdbat[(size_t)h * dim + i];
    }
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if ((miss_mask >> k) & 1u) {
            y[k] = pre_y[(size_t)k * dim + i];
        }
    }

    float sum = shexp_out ? shexp_out[i] : 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        sum += top_values[k] * y[k];
    }
    res_x[i] += sum;
}

void gpu_qwen_moe_combine_unified(float *d_res_x, const float *exdbat,
                                  const int *hit_pos, int n_hits,
                                  const float *d_pre_y, unsigned miss_mask,
                                  const float *top_values, const float *d_shexp_out,
                                  int dim, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_top_values, top_values, 8 * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (n_hits > 0) {
        CUDA_CHECK(cudaMemcpyAsync(g_d_hit_pos, hit_pos, (size_t)n_hits * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    dim3 grid((unsigned)((dim + 255) / 256), 1u, 1u);
    dim3 block(256u, 1u, 1u);
    qwen_moe_combine_unified_kernel<<<grid, block, 0, stream>>>(
        d_res_x, exdbat, g_d_hit_pos, n_hits, d_pre_y, miss_mask, g_d_top_values, d_shexp_out, dim);
    CUDA_CHECK_LAST();
}



void launch_swiglu_ncols(float *act, const float *gate, const float *up, int count, cudaStream_t stream) {
    swiglu_dev_ncols_kernel<<<(count + 255) / 256, 256, 0, stream>>>(act, gate, up, count);
    CUDA_CHECK_LAST();
}

void launch_scale_shexp(float *out, const float *down, const float *gate, int dim, cudaStream_t stream) {
    scale_shexp_kernel<<<(dim + 255) / 256, 256, 0, stream>>>(out, down, gate, dim);
    CUDA_CHECK_LAST();
}
