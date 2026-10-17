#include "state.h"

template <int HEAD_DIM, int TILE_SIZE>
__launch_bounds__(HEAD_DIM, 1)
__global__ void flash_attn_vec_online_kernel(
        float * __restrict__ k_cache,   // [n_attn_layers][n_head_kv][max_seq][head_dim]
        float * __restrict__ v_cache,
        const float * __restrict__ q,   // [n_head][head_dim]
        const float * __restrict__ k,   // [n_head_kv][head_dim]
        const float * __restrict__ v,   // [n_head_kv][head_dim]
        const float * __restrict__ gate,// [n_head][head_dim] or NULL
        float * __restrict__ concat_out,// [n_head][head_dim]
        int n_head, int n_head_kv, int max_seq, int head_dim,
        int layer_idx, int pos, int ctx_len, float scale) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int gqa_ratio = n_head / n_head_kv;
    const int kv_h = h / gqa_ratio;

    const size_t layer_stride = (size_t)n_head_kv * max_seq * head_dim;
    const size_t kv_stride   = (size_t)max_seq * head_dim;
    float * __restrict__ kc = k_cache + (size_t)layer_idx * layer_stride + (size_t)kv_h * kv_stride;
    float * __restrict__ vc = v_cache + (size_t)layer_idx * layer_stride + (size_t)kv_h * kv_stride;

    const float * __restrict__ qh = q + (size_t)h * head_dim;
    const float * __restrict__ kh = k + (size_t)kv_h * head_dim;
    const float * __restrict__ vh = v + (size_t)kv_h * head_dim;

    // Stage A: write this token's k/v rows into the device cache
    if (tid < head_dim) {
        kc[(size_t)pos * head_dim + tid] = kh[tid];
        vc[(size_t)pos * head_dim + tid] = vh[tid];
    }
    __syncthreads();

    // Stage B: FlashDecoding Online-Softmax Tile Loop (0 global score allocations)
    __shared__ float s_score[TILE_SIZE];
    __shared__ float s_warp_dot[TILE_SIZE][8];

    const float q_val = (tid < head_dim) ? (qh[tid] * scale) : 0.0f;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc  = 0.0f;

    for (int t0 = 0; t0 < ctx_len; t0 += TILE_SIZE) {
        const int tile_len = min(TILE_SIZE, ctx_len - t0);

        float dots[TILE_SIZE];
        #pragma unroll
        for (int tau = 0; tau < TILE_SIZE; ++tau) {
            if (tau < tile_len) {
                const int t = t0 + tau;
                const float * __restrict__ kt = kc + (size_t)t * head_dim;
                dots[tau] = (tid < head_dim) ? (q_val * kt[tid]) : 0.0f;
            } else {
                dots[tau] = 0.0f;
            }
        }

        #pragma unroll
        for (int tau = 0; tau < TILE_SIZE; ++tau) {
            float d = dots[tau];
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                d += __shfl_down_sync(0xffffffff, d, mask);
            }
            if (lane_id == 0 && warp_id < 8) {
                s_warp_dot[tau][warp_id] = d;
            }
        }
        __syncthreads();

        if (tid < tile_len) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < (HEAD_DIM / 32); ++w) {
                total += s_warp_dot[tid][w];
            }
            s_score[tid] = total;
        }
        __syncthreads();

        // Online Softmax update for this tile in registers
        float tile_max = -1e30f;
        #pragma unroll
        for (int tau = 0; tau < tile_len; ++tau) {
            tile_max = fmaxf(tile_max, s_score[tau]);
        }

        const float m_new = fmaxf(m_prev, tile_max);
        const float alpha = expf(m_prev - m_new);
        m_prev = m_new;
        l_prev = l_prev * alpha;
        o_acc  = o_acc  * alpha;

        #pragma unroll
        for (int tau = 0; tau < tile_len; ++tau) {
            const float p = expf(s_score[tau] - m_new);
            l_prev += p;
            const int t = t0 + tau;
            const float * __restrict__ vt = vc + (size_t)t * head_dim;
            const float v_val = (tid < head_dim) ? vt[tid] : 0.0f;
            o_acc += p * v_val;
        }
        __syncthreads();
    }

    // Stage C: Final Normalization & Q-Gate Sigmoid Gating
    if (tid < head_dim) {
        float out_val = (l_prev > 0.0f) ? (o_acc / l_prev) : 0.0f;
        if (gate) {
            float gz = gate[(size_t)h * head_dim + tid];
            out_val *= 1.0f / (1.0f + expf(-gz));
        }
        concat_out[(size_t)h * head_dim + tid] = out_val;
    }
}

static int flash_attn_enabled(void) {
    static int s_fa = -1;
    if (s_fa < 0) {
        const char *e = getenv("TUFF_FLASH_ATTN");
        s_fa = (e && e[0] == '0') ? 0 : 1;
    }
    return s_fa;
}

void kv_cache_ensure(int n_attn_layers, int n_head_kv, int max_seq, int head_dim) {
    size_t needed_bytes = 0;
    if (g_kv_quant_mode == KV_QUANT_Q8_0) {
        needed_bytes = (size_t)n_attn_layers * n_head_kv * max_seq * (head_dim / 32) * sizeof(block_q8_0);
    } else {
        needed_bytes = (size_t)n_attn_layers * n_head_kv * max_seq * head_dim * sizeof(float);
    }
    if (!g_dk_cache || g_kv_cache_bytes != needed_bytes) {
        if (g_dk_cache) { CUDA_CHECK(cudaFree(g_dk_cache)); CUDA_CHECK(cudaFree(g_dv_cache)); }
        CUDA_CHECK(cudaMalloc(&g_dk_cache, needed_bytes));
        CUDA_CHECK(cudaMalloc(&g_dv_cache, needed_bytes));
        CUDA_CHECK(cudaMemset(g_dk_cache, 0, needed_bytes));
        CUDA_CHECK(cudaMemset(g_dv_cache, 0, needed_bytes));
        g_kv_cache_bytes = needed_bytes;
    }
}

// ---------------------------------------------------------------------------
// H58: Ampere cp.async Asynchronous KV Pipeline Primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ void load_kv_tile_q8_async(uint8_t *s_tile, const block_q8_0 *g_base, int tile_start_t, int total_tokens, int tid) {
    if (tile_start_t >= total_tokens) return;
    const uint8_t *g_bytes = reinterpret_cast<const uint8_t*>(g_base + (size_t)tile_start_t * 8);
    const int valid_bytes = min(4352, (total_tokens - tile_start_t) * 272);
    const int valid_chunks = (valid_bytes + 15) / 16;
    if (tid < valid_chunks) {
        cp_async16(s_tile + tid * 16, g_bytes + tid * 16);
    }
    if (256 + tid < valid_chunks) {
        cp_async16(s_tile + (256 + tid) * 16, g_bytes + (256 + tid) * 16);
    }
}

static int g_async_kv_flag = -1;
static inline bool is_async_kv_enabled(void) {
    if (g_async_kv_flag < 0) {
        const char *e = getenv("TUFF_ASYNC_KV");
        g_async_kv_flag = (e && e[0] == '0') ? 0 : 1;
    }
    return g_async_kv_flag != 0;
}

// Optimized templated full-attention kernel supporting Q8_0 lossless quantization & native FP32
template <int KV_QUANT, bool ASYNC_PIPELINE>
__launch_bounds__(256, 4)
__global__ void attention_kernel(
        void * __restrict__ k_cache,   // [n_attn_layers][n_head_kv][max_seq][head_dim] or [n_attn_layers][n_head_kv][max_seq][head_dim/32] block_q8_0
        void * __restrict__ v_cache,
        const float * __restrict__ q,   // [n_head][head_dim]
        const float * __restrict__ k,   // [n_head_kv][head_dim]
        const float * __restrict__ v,   // [n_head_kv][head_dim]
        const float * __restrict__ gate,// [n_head][head_dim] or NULL
        float * __restrict__ concat_out,// [n_head][head_dim]
        float * __restrict__ scores_buf,// [n_head][max_seq] per-block score storage
        int n_head, int n_head_kv, int max_seq, int head_dim,
        int layer_idx, int pos, int ctx_len, float scale,
        block_q8_1_c * __restrict__ out_q8,
        const int * __restrict__ d_pos) {
    const int actual_pos = d_pos ? *d_pos : pos;
    const int actual_ctx_len = d_pos ? (*d_pos + 1) : ctx_len;

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int gqa_ratio = n_head / n_head_kv;
    const int kv_h = h / gqa_ratio;
    float * __restrict__ scores = scores_buf + (size_t)h * max_seq;

    const float * __restrict__ qh = q + (size_t)h * head_dim;
    const float * __restrict__ kh = k + (size_t)kv_h * head_dim;
    const float * __restrict__ vh = v + (size_t)kv_h * head_dim;

    __shared__ float s_qh[256];
    __shared__ float s_red[256];
    __shared__ uint8_t s_kv_tile[2][4352]; // 16 tokens x 272 bytes per buffer (8.5 KB total smem)

    if (KV_QUANT == KV_QUANT_Q8_0) {
        const int n_blocks_per_head = head_dim / 32; // 8
        const size_t layer_stride_q8 = (size_t)n_head_kv * max_seq * n_blocks_per_head;
        const size_t kv_stride_q8    = (size_t)max_seq * n_blocks_per_head;
        block_q8_0 * __restrict__ kc_q8 = (block_q8_0*)k_cache + (size_t)layer_idx * layer_stride_q8 + (size_t)kv_h * kv_stride_q8;
        block_q8_0 * __restrict__ vc_q8 = (block_q8_0*)v_cache + (size_t)layer_idx * layer_stride_q8 + (size_t)kv_h * kv_stride_q8;

        // Stage A: Cache Q in shared memory and quantize new K/V token into block_q8_0
        if (tid < head_dim) {
            s_qh[tid] = qh[tid];

            const int warp_id = tid >> 5;  // 0..7
            const int lane_id = tid & 31;  // 0..31
            const float k_val = kh[tid];
            const float v_val = vh[tid];

            float k_amax = fabsf(k_val);
            float v_amax = fabsf(v_val);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                k_amax = fmaxf(k_amax, __shfl_xor_sync(0xffffffff, k_amax, mask, 32));
                v_amax = fmaxf(v_amax, __shfl_xor_sync(0xffffffff, v_amax, mask, 32));
            }
            const float d_k = k_amax / 127.0f;
            const float d_v = v_amax / 127.0f;
            const int8_t q_k = (k_amax == 0.0f) ? 0 : (int8_t)roundf(k_val / d_k);
            const int8_t q_v = (v_amax == 0.0f) ? 0 : (int8_t)roundf(v_val / d_v);

            block_q8_0 *k_dst = kc_q8 + (size_t)actual_pos * n_blocks_per_head + warp_id;
            block_q8_0 *v_dst = vc_q8 + (size_t)actual_pos * n_blocks_per_head + warp_id;
            k_dst->qs[lane_id] = q_k;
            v_dst->qs[lane_id] = q_v;
            if (lane_id == 0) {
                k_dst->d = __half_as_ushort(__float2half(d_k));
                v_dst->d = __half_as_ushort(__float2half(d_v));
            }
        }
        __syncthreads();

        // Stage B: Compute scores[t] = (sum_d qh[d]*kt[d]) * scale
        if (!ASYNC_PIPELINE) {
            for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
                const block_q8_0 *kt_blocks = kc_q8 + (size_t)t * n_blocks_per_head;
                float dotp = 0.0f;
                #pragma unroll 8
                for (int b = 0; b < n_blocks_per_head; ++b) {
                    const block_q8_0 *kb = &kt_blocks[b];
                    const float d_k = __half2float(__ushort_as_half(kb->d));
                    float sum_b = 0.0f;
                    #pragma unroll 8
                    for (int i4 = 0; i4 < 8; ++i4) {
                        const int8_t q0 = kb->qs[i4 * 4 + 0];
                        const int8_t q1 = kb->qs[i4 * 4 + 1];
                        const int8_t q2 = kb->qs[i4 * 4 + 2];
                        const int8_t q3 = kb->qs[i4 * 4 + 3];
                        sum_b += (float)q0 * s_qh[b * 32 + i4 * 4 + 0]
                               + (float)q1 * s_qh[b * 32 + i4 * 4 + 1]
                               + (float)q2 * s_qh[b * 32 + i4 * 4 + 2]
                               + (float)q3 * s_qh[b * 32 + i4 * 4 + 3];
                    }
                    dotp += sum_b * d_k;
                }
                scores[t] = dotp * scale;
            }
            __syncthreads();
        } else {
            const int num_k_tiles = (actual_ctx_len + 15) / 16;
            if (num_k_tiles > 0) {
                load_kv_tile_q8_async(s_kv_tile[0], kc_q8, 0, actual_ctx_len, tid);
                cp_async_commit();
            }

            for (int tile = 0; tile < num_k_tiles; ++tile) {
                if (tile + 1 < num_k_tiles) {
                    load_kv_tile_q8_async(s_kv_tile[(tile + 1) & 1], kc_q8, (tile + 1) * 16, actual_ctx_len, tid);
                    cp_async_commit();
                }
                cp_async_wait<0>();
                __syncthreads();

                const uint8_t *cur_tile = s_kv_tile[tile & 1];
                const int t_in_tile = tid >> 4;
                const int k_idx     = tid & 15;
                const int t         = tile * 16 + t_in_tile;

                float part_sum = 0.0f;
                if (t < actual_ctx_len) {
                    const block_q8_0 *kb = reinterpret_cast<const block_q8_0*>(cur_tile + t_in_tile * 272) + (k_idx >> 1);
                    const float d_k = __half2float(__ushort_as_half(kb->d));
                    const int half_offset = (k_idx & 1) * 16;
                    const int b = k_idx >> 1;
                    float sum_16 = 0.0f;
                    #pragma unroll 4
                    for (int i4 = 0; i4 < 4; ++i4) {
                        const int8_t q0 = kb->qs[half_offset + i4 * 4 + 0];
                        const int8_t q1 = kb->qs[half_offset + i4 * 4 + 1];
                        const int8_t q2 = kb->qs[half_offset + i4 * 4 + 2];
                        const int8_t q3 = kb->qs[half_offset + i4 * 4 + 3];
                        sum_16 += (float)q0 * s_qh[b * 32 + half_offset + i4 * 4 + 0]
                                + (float)q1 * s_qh[b * 32 + half_offset + i4 * 4 + 1]
                                + (float)q2 * s_qh[b * 32 + half_offset + i4 * 4 + 2]
                                + (float)q3 * s_qh[b * 32 + half_offset + i4 * 4 + 3];
                    }
                    part_sum = sum_16 * d_k;
                }

                #pragma unroll
                for (int mask = 8; mask > 0; mask >>= 1) {
                    part_sum += __shfl_xor_sync(0xffffffff, part_sum, mask, 32);
                }

                if (k_idx == 0 && t < actual_ctx_len) {
                    scores[t] = part_sum * scale;
                }
                __syncthreads();
            }
        }

        // Stage C: Parallel Softmax over scores[0..actual_ctx_len)
        // 1. Parallel Max
        float local_max = -1e30f;
        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            local_max = fmaxf(local_max, scores[t]);
        }
        s_red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
            __syncthreads();
        }
        const float max_val = s_red[0];
        __syncthreads();

        // 2. Parallel Exp & Sum
        float local_sum = 0.0f;
        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            float ex = expf(scores[t] - max_val);
            scores[t] = ex;
            local_sum += ex;
        }
        s_red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) s_red[tid] += s_red[tid + stride];
            __syncthreads();
        }
        const float inv_sum = (s_red[0] > 0.0f) ? (1.0f / s_red[0]) : 0.0f;
        __syncthreads();

        // 3. Parallel Normalization
        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            scores[t] *= inv_sum;
        }
        __syncthreads();

        // Stage D: concat_out[h][d] = sum_t scores[t] * vc[t][d]
        const int warp_id = tid >> 5;  // 0..7
        const int lane_id = tid & 31;  // 0..31
        float acc = 0.0f;

        if (!ASYNC_PIPELINE) {
            #pragma unroll 4
            for (int t = 0; t < actual_ctx_len; ++t) {
                const block_q8_0 *vb = vc_q8 + (size_t)t * n_blocks_per_head + warp_id;
                const float d_v = __half2float(__ushort_as_half(vb->d));
                const int8_t q_v = vb->qs[lane_id];
                acc += scores[t] * ((float)q_v * d_v);
            }
        } else {
            const int num_v_tiles = (actual_ctx_len + 15) / 16;
            if (num_v_tiles > 0) {
                load_kv_tile_q8_async(s_kv_tile[0], vc_q8, 0, actual_ctx_len, tid);
                cp_async_commit();
            }

            for (int tile = 0; tile < num_v_tiles; ++tile) {
                if (tile + 1 < num_v_tiles) {
                    load_kv_tile_q8_async(s_kv_tile[(tile + 1) & 1], vc_q8, (tile + 1) * 16, actual_ctx_len, tid);
                    cp_async_commit();
                }
                cp_async_wait<0>();
                __syncthreads();

                const uint8_t *cur_tile = s_kv_tile[tile & 1];
                #pragma unroll 16
                for (int t_in_tile = 0; t_in_tile < 16; ++t_in_tile) {
                    const int t = tile * 16 + t_in_tile;
                    if (t < actual_ctx_len) {
                        const block_q8_0 *vb = reinterpret_cast<const block_q8_0*>(cur_tile + t_in_tile * 272) + warp_id;
                        const float d_v = __half2float(__ushort_as_half(vb->d));
                        const int8_t q_v = vb->qs[lane_id];
                        acc += scores[t] * ((float)q_v * d_v);
                    }
                }
                __syncthreads();
            }
        }

        if (gate) {
            float gz = gate[(size_t)h * head_dim + tid];
            acc *= 1.0f / (1.0f + expf(-gz));
        }
        if (concat_out) concat_out[(size_t)h * head_dim + tid] = acc;
        if (out_q8) {
            const int block_idx = h * n_blocks_per_head + warp_id;
            float amax = fabsf(acc);
            float sum  = acc;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
                sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
            }
            const float d = amax / 127.0f;
            const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(acc / d);
            out_q8[block_idx].qs[lane_id] = q;
            if (lane_id == 0) {
                out_q8[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
            }
        }
    } else {
        // Native FP32 path
        const size_t layer_stride = (size_t)n_head_kv * max_seq * head_dim;
        const size_t kv_stride   = (size_t)max_seq * head_dim;
        float * __restrict__ kc = (float*)k_cache + (size_t)layer_idx * layer_stride + (size_t)kv_h * kv_stride;
        float * __restrict__ vc = (float*)v_cache + (size_t)layer_idx * layer_stride + (size_t)kv_h * kv_stride;

        // Stage A: Cache Q in shared memory and write new K/V token into cache
        if (tid < head_dim) {
            s_qh[tid] = qh[tid];
            kc[(size_t)actual_pos * head_dim + tid] = kh[tid];
            vc[(size_t)actual_pos * head_dim + tid] = vh[tid];
        }
        __syncthreads();

        // Stage B: Vectorized float4 dot-product: scores[t] = (sum_d qh[d]*kt[d]) * scale
        const float4 * __restrict__ q4 = reinterpret_cast<const float4*>(s_qh);
        const int n_f4 = head_dim / 4;

        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            const float4 * __restrict__ kt4 = reinterpret_cast<const float4*>(kc + (size_t)t * head_dim);
            float dotp = 0.0f;
            #pragma unroll 8
            for (int d4 = 0; d4 < n_f4; ++d4) {
                float4 kv = kt4[d4];
                float4 qv = q4[d4];
                dotp += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
            scores[t] = dotp * scale;
        }
        __syncthreads();

        // Stage C: Parallel Softmax over scores[0..actual_ctx_len)
        // 1. Parallel Max
        float local_max = -1e30f;
        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            local_max = fmaxf(local_max, scores[t]);
        }
        s_red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
            __syncthreads();
        }
        const float max_val = s_red[0];
        __syncthreads();

        // 2. Parallel Exp & Sum
        float local_sum = 0.0f;
        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            float ex = expf(scores[t] - max_val);
            scores[t] = ex;
            local_sum += ex;
        }
        s_red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) s_red[tid] += s_red[tid + stride];
            __syncthreads();
        }
        const float inv_sum = (s_red[0] > 0.0f) ? (1.0f / s_red[0]) : 0.0f;
        __syncthreads();

        // 3. Parallel Normalization
        for (int t = tid; t < actual_ctx_len; t += blockDim.x) {
            scores[t] *= inv_sum;
        }
        __syncthreads();

        // Stage D: concat_out[h][d] = sum_t scores[t] * vc[t][d], one d per thread
        if (tid < head_dim) {
            float acc = 0.0f;
            #pragma unroll 4
            for (int t = 0; t < actual_ctx_len; ++t) {
                acc += scores[t] * vc[(size_t)t * head_dim + tid];
            }
            if (gate) {
                float gz = gate[(size_t)h * head_dim + tid];
                acc *= 1.0f / (1.0f + expf(-gz));
            }
            if (concat_out) concat_out[(size_t)h * head_dim + tid] = acc;
            if (out_q8) {
                const int warp_id = tid >> 5;
                const int lane_id = tid & 31;
                const int block_idx = h * (head_dim / 32) + warp_id;
                float amax = fabsf(acc);
                float sum  = acc;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
                    sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
                }
                const float d = amax / 127.0f;
                const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(acc / d);
                out_q8[block_idx].qs[lane_id] = q;
                if (lane_id == 0) {
                    out_q8[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
                }
            }
        }
    }
}

void launch_attention_kernel(
        void *k_cache, void *v_cache, const float *q, const float *k, const float *v,
        const float *gate, float *concat_out, float *scores_buf,
        int n_head, int n_head_kv, int max_seq, int head_dim,
        int layer_idx, int pos, int ctx_len, float scale,
        block_q8_1_c *out_q8, const int *d_pos) {
    if (g_kv_quant_mode == KV_QUANT_Q8_0) {
        if (is_async_kv_enabled()) {
            attention_kernel<KV_QUANT_Q8_0, true><<<n_head, head_dim>>>(
                k_cache, v_cache, q, k, v, gate, concat_out, scores_buf,
                n_head, n_head_kv, max_seq, head_dim, layer_idx, pos, ctx_len, scale, out_q8, d_pos);
        } else {
            attention_kernel<KV_QUANT_Q8_0, false><<<n_head, head_dim>>>(
                k_cache, v_cache, q, k, v, gate, concat_out, scores_buf,
                n_head, n_head_kv, max_seq, head_dim, layer_idx, pos, ctx_len, scale, out_q8, d_pos);
        }
    } else {
        attention_kernel<KV_QUANT_F32, false><<<n_head, head_dim>>>(
            k_cache, v_cache, q, k, v, gate, concat_out, scores_buf,
            n_head, n_head_kv, max_seq, head_dim, layer_idx, pos, ctx_len, scale, out_q8, d_pos);
    }
}

// Per-block attention score storage in global memory (bit-identical to the former shared-memory
// scores; frees the kernel from the dynamic-smem ceiling at long context).
void attn_scores_ensure(int n_head, int max_seq) {
    const size_t needed = (size_t)n_head * max_seq;
    if (!g_d_attn_scores || g_attn_scores_floats != needed) {
        if (g_d_attn_scores) CUDA_CHECK(cudaFree(g_d_attn_scores));
        CUDA_CHECK(cudaMalloc(&g_d_attn_scores, needed * sizeof(float)));
        g_attn_scores_floats = needed;
    }
}

void gpu_attention(int layer_idx, float *concat_out,
                   const float *q, const float *k, const float *v,
                   int n_attn_layers, int n_head, int n_head_kv, int head_dim,
                   int pos, int ctx_len, int max_seq, float scale) {
    // Lazy-allocate the persistent device KV caches on first use (sized to the model + max_seq).
    kv_cache_ensure(n_attn_layers, n_head_kv, max_seq, head_dim);
    attn_scores_ensure(n_head, max_seq);
    if (!g_aq)   CUDA_CHECK(cudaMalloc(&g_aq,   (size_t)n_head * head_dim * sizeof(float)));
    if (!g_ak)   CUDA_CHECK(cudaMalloc(&g_ak,   (size_t)n_head_kv * head_dim * sizeof(float)));
    if (!g_av)   CUDA_CHECK(cudaMalloc(&g_av,   (size_t)n_head_kv * head_dim * sizeof(float)));
    if (!g_aconc) CUDA_CHECK(cudaMalloc(&g_aconc, (size_t)n_head * head_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(g_aq, q, (size_t)n_head * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_ak, k, (size_t)n_head_kv * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_av, v, (size_t)n_head_kv * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_LAST();

    attn_scores_ensure(n_head, max_seq);
    launch_attention_kernel(
        g_dk_cache, g_dv_cache, g_aq, g_ak, g_av, NULL, g_aconc, g_d_attn_scores,
        n_head, n_head_kv, max_seq, head_dim, layer_idx, pos, ctx_len, scale, NULL, NULL);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(concat_out, g_aconc, (size_t)n_head * head_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Zero the persistent device KV caches. Called from engine_init so each engine run (which
// allocates a fresh zeroed host engine) starts from a zeroed KV cache, matching the host
// calloc semantics. No-op if not yet allocated.
void gpu_reset_kv(void) {
    if (g_dk_cache && g_kv_cache_bytes > 0) {
        CUDA_CHECK(cudaMemset(g_dk_cache, 0, g_kv_cache_bytes));
        CUDA_CHECK(cudaMemset(g_dv_cache, 0, g_kv_cache_bytes));
    }
}


__global__ void fused_q_split_norm_rope_kernel(
        float * __restrict__ q_out,
        float * __restrict__ gate_out,
        const float * __restrict__ q_full,
        const float * __restrict__ q_norm_w,
        int head_dim, float eps, int n_rot,
        int pos, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        float freq_base) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;

    const float * __restrict__ qh_raw = q_full + (size_t)h * 2 * head_dim;
    const float * __restrict__ gh_raw = qh_raw + head_dim;

    float q_val = (tid < head_dim) ? qh_raw[tid] : 0.0f;
    float g_val = (tid < head_dim) ? gh_raw[tid] : 0.0f;

    if (tid < head_dim && gate_out) {
        gate_out[(size_t)h * head_dim + tid] = g_val;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = q_val * q_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float scale = rsqrtf(s_sum[0] / (float)head_dim + eps);
    __syncthreads();

    float q_normed = q_val * scale * q_norm_w[tid];

    __shared__ float s_q[256];
    s_q[tid] = q_normed;

    __shared__ float cos_cache[32];
    __shared__ float sin_cache[32];

    const int half_rot = n_rot / 2;
    const int sect_dims = (int)(s0 + s1 + s2 + s3);
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);

    if (tid == 0) {
        float theta_t = (float)pos;
        float theta_h = (float)pos;
        float theta_w = (float)pos;
        float theta_e = 0.0f;
        for (int p = 0; p < half_rot; ++p) {
            int sector = p % sect_dims;
            float theta;
            if (sector % 3 == 1 && sector < 3 * (int)s1) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * (int)s2) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * (int)s0) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
            cos_cache[p] = cosf(theta);
            sin_cache[p] = sinf(theta);
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
    }
    __syncthreads();

    if (tid < half_rot) {
        int p = tid;
        float c = cos_cache[p], s = sin_cache[p];
        float x0 = s_q[p];
        float x1 = s_q[p + half_rot];
        s_q[p]            = x0 * c - x1 * s;
        s_q[p + half_rot] = x0 * s + x1 * c;
    }
    __syncthreads();

    if (tid < head_dim) {
        q_out[(size_t)h * head_dim + tid] = s_q[tid];
    }
}

__global__ void fused_k_norm_rope_kernel(
        float * __restrict__ k_out,
        const float * __restrict__ k_in,
        const float * __restrict__ k_norm_w,
        int head_dim, float eps, int n_rot,
        int pos, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        float freq_base) {
    const int kh = blockIdx.x;
    const int tid = threadIdx.x;

    const float * __restrict__ kh_raw = k_in + (size_t)kh * head_dim;
    float k_val = (tid < head_dim) ? kh_raw[tid] : 0.0f;

    __shared__ float s_sum[256];
    s_sum[tid] = k_val * k_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float scale = rsqrtf(s_sum[0] / (float)head_dim + eps);
    __syncthreads();

    float k_normed = k_val * scale * k_norm_w[tid];

    __shared__ float s_k[256];
    s_k[tid] = k_normed;

    __shared__ float cos_cache[32];
    __shared__ float sin_cache[32];

    const int half_rot = n_rot / 2;
    const int sect_dims = (int)(s0 + s1 + s2 + s3);
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);

    if (tid == 0) {
        float theta_t = (float)pos;
        float theta_h = (float)pos;
        float theta_w = (float)pos;
        float theta_e = 0.0f;
        for (int p = 0; p < half_rot; ++p) {
            int sector = p % sect_dims;
            float theta;
            if (sector % 3 == 1 && sector < 3 * (int)s1) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * (int)s2) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * (int)s0) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
            cos_cache[p] = cosf(theta);
            sin_cache[p] = sinf(theta);
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
    }
    __syncthreads();

    if (tid < half_rot) {
        int p = tid;
        float c = cos_cache[p], s = sin_cache[p];
        float x0 = s_k[p];
        float x1 = s_k[p + half_rot];
        s_k[p]            = x0 * c - x1 * s;
        s_k[p + half_rot] = x0 * s + x1 * c;
    }
    __syncthreads();

    if (tid < head_dim) {
        k_out[(size_t)kh * head_dim + tid] = s_k[tid];
    }
}

__global__ void fused_qk_split_norm_rope_kernel(
        float * __restrict__ q_out,
        float * __restrict__ gate_out,
        const float * __restrict__ q_full,
        const float * __restrict__ q_norm_w,
        int n_q_heads,
        float * __restrict__ k_out,
        const float * __restrict__ k_in,
        const float * __restrict__ k_norm_w,
        int n_k_heads,
        int head_dim, float eps, int n_rot,
        int pos, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        float freq_base,
        const int * __restrict__ d_pos) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_q = (b < n_q_heads);
    const int h = is_q ? b : (b - n_q_heads);

    const float * __restrict__ in_ptr;
    const float * __restrict__ norm_w;
    float * __restrict__ out_ptr;

    float val = 0.0f;
    if (is_q) {
        in_ptr = q_full + (size_t)h * (2 * head_dim);
        norm_w = q_norm_w;
        out_ptr = q_out + (size_t)h * head_dim;
        if (tid < head_dim) {
            val = in_ptr[tid];
            if (gate_out) gate_out[(size_t)h * head_dim + tid] = in_ptr[head_dim + tid];
        }
    } else {
        in_ptr = k_in + (size_t)h * head_dim;
        norm_w = k_norm_w;
        out_ptr = k_out + (size_t)h * head_dim;
        if (tid < head_dim) {
            val = in_ptr[tid];
        }
    }

    __shared__ float s_sum[256];
    s_sum[tid] = (tid < head_dim) ? (val * val) : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float scale = rsqrtf(s_sum[0] / (float)head_dim + eps);
    __syncthreads();

    float normed = (tid < head_dim) ? (val * scale * norm_w[tid]) : 0.0f;

    __shared__ float s_buf[256];
    s_buf[tid] = normed;

    __shared__ float cos_cache[32];
    __shared__ float sin_cache[32];

    const int half_rot = n_rot / 2;
    const int sect_dims = (int)(s0 + s1 + s2 + s3);
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);

    const int actual_pos = d_pos ? *d_pos : pos;
    if (tid == 0) {
        float theta_t = (float)actual_pos;
        float theta_h = (float)actual_pos;
        float theta_w = (float)actual_pos;
        float theta_e = 0.0f;
        for (int p = 0; p < half_rot; ++p) {
            int sector = p % sect_dims;
            float theta;
            if (sector % 3 == 1 && sector < 3 * (int)s1) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * (int)s2) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * (int)s0) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
            cos_cache[p] = cosf(theta);
            sin_cache[p] = sinf(theta);
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
    }
    __syncthreads();

    if (tid < half_rot) {
        int p = tid;
        float c = cos_cache[p], s = sin_cache[p];
        float x0 = s_buf[p];
        float x1 = s_buf[p + half_rot];
        s_buf[p]            = x0 * c - x1 * s;
        s_buf[p + half_rot] = x0 * s + x1 * c;
    }
    __syncthreads();

    if (tid < head_dim) {
        out_ptr[tid] = s_buf[tid];
    }
}

void gpu_attention_layer_dev(int layer_idx, float *d_res_x, const model_layer_t *layer,
                             const float *d_norm_x, int n_attn_layers,
                             int n_head, int n_head_kv, int head_dim,
                             int pos, int ctx_len, int max_seq, float scale, float eps, int dim,
                             int rope_dim_count, const uint32_t rope_sections[4], float rope_freq_base) {
    gpu_set_device_pos(pos);
    kv_cache_ensure(n_attn_layers, n_head_kv, max_seq, head_dim);
    attn_scores_ensure(n_head, max_seq);

    uint32_t q_full_dim = (uint32_t)layer->attn_q.info.dims[1]; // 8192
    uint32_t k_dim      = (uint32_t)layer->attn_k.info.dims[1]; // 512
    uint32_t v_dim      = (uint32_t)layer->attn_v.info.dims[1]; // 512

    if (!tuff_exact() && mmvq_dual_enabled()) {
        dense_quantize_x(d_norm_x, dim);
        const block_q8_1_c *xq = (const block_q8_1_c *)g_d_dense_x_q8;
        if (tuff_h50()) {
            const int nblk = dim / QK8_0;
            const int total_qkv = (int)q_full_dim + (int)k_dim + (int)v_dim;
            launch_mmvq_dense_triple((const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim, (const block_q8_0*)layer->attn_k.dev, g_d_k, (int)k_dim, (const block_q8_0*)layer->attn_v.dev, g_d_v, (int)v_dim, xq, nblk);
        } else {
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_q.dev, xq, g_d_q_full, (int)dim, (int)q_full_dim, false);
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_k.dev, xq, g_d_k,        (int)dim, (int)k_dim,      false);
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_v.dev, xq, g_d_v,        (int)dim, (int)v_dim,      false);
        }
    } else {
        launch_mat_vec_q8_0((const block_q8_0*)layer->attn_q.dev, d_norm_x, g_d_q_full, dim, q_full_dim);
        launch_mat_vec_q8_0_dual((const block_q8_0*)layer->attn_k.dev, g_d_k, k_dim,
                                 (const block_q8_0*)layer->attn_v.dev, g_d_v, v_dim,
                                 d_norm_x, dim);
    }

    if (tuff_h50()) {
        fused_qk_split_norm_rope_kernel<<<n_head + n_head_kv, head_dim>>>(
            g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head,
            g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv,
            head_dim, eps, rope_dim_count, pos, rope_sections[0], rope_sections[1],
            rope_sections[2], rope_sections[3], rope_freq_base, g_d_pos);
        CUDA_CHECK_LAST();
    } else {
        // Fused Split + RMSNorm + MRoPE single-pass kernels
        fused_q_split_norm_rope_kernel<<<n_head, head_dim>>>(
            g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev,
            head_dim, eps, rope_dim_count, pos, rope_sections[0], rope_sections[1],
            rope_sections[2], rope_sections[3], rope_freq_base);
        CUDA_CHECK_LAST();

        fused_k_norm_rope_kernel<<<n_head_kv, head_dim>>>(
            g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev,
            head_dim, eps, rope_dim_count, pos, rope_sections[0], rope_sections[1],
            rope_sections[2], rope_sections[3], rope_freq_base);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK_LAST();

    if (tuff_h50()) {
        block_q8_1_c *out_q8 = (block_q8_1_c*)dense_quant_buf_ensure(dim);
        launch_attention_kernel(
            g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, NULL, g_d_attn_scores,
            n_head, n_head_kv, max_seq, head_dim, layer_idx, pos, ctx_len, scale, out_q8, g_d_pos);
        CUDA_CHECK_LAST();

        mmvq_q8_0_rows((const block_q8_0*)layer->attn_output.dev, out_q8, d_res_x, n_head * head_dim, dim, true);
    } else {
        launch_attention_kernel(
            g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, g_d_concat_out, g_d_attn_scores,
            n_head, n_head_kv, max_seq, head_dim, layer_idx, pos, ctx_len, scale, NULL, g_d_pos);
        CUDA_CHECK_LAST();

        launch_mat_vec_q8_0_accum((const block_q8_0*)layer->attn_output.dev, g_d_concat_out, d_res_x, n_head * head_dim, dim);
    }
}


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
