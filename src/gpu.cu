#include "gpu.h"
#include "cpu_ops.h"
#include "ggml_types.h"
#include "cuda_check.h"
#include "tuff_exact.h"   // P0's shared TUFF_EXACT flag (cached, extern "C")

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

// Scratch buffers for the single synchronous gpu_mat_vec round-trip.
// Largest dense in_dim is 4096 (attn_output / ssm_out). The output projection is the only
// large out_dim: 248320 (vocab size). Cap at 262144 for both.
#define GPU_IN_FLOATS   4096
#define GPU_OUT_FLOATS  262144

static float *g_dx = NULL; // device scratch for the input vector (in_dim)
static float *g_dy = NULL; // device scratch for the output vector (out_dim)

// Delta-net state-update scratch. g_dstate_all is the persistent recurrent state for ALL
// delta layers [n_delta_layers][n_v_heads][head_v_dim][head_k_dim], allocated + zeroed in
// gpu_prepare (so it lives on device across tokens) and zeroed again by gpu_reset_delta_state
// (called from engine_init so each engine run starts from zero). The remaining buffers are
// per-call scratch (allocated lazily, freed in gpu_shutdown):
//   g_dq/g_dk = 16*128 floats, g_dv/g_dout = 32*128 floats, g_dp = 64 floats (decay+beta)
static float *g_dstate_all = NULL;
static size_t g_dstate_bytes = 0;
// P4a: persistent device mirror of the gated delta-net conv1d state
// [n_delta_layers][conv_kernel][qkv_dim]. The host engine keeps its own `conv_state` for the
// TUFF_CPU_DELTA reference path; this copy is owned by the device-resident delta layer
// (gpu_delta_pre), allocated + zeroed in gpu_prepare and re-zeroed by gpu_reset_delta_state.
static float *g_dconv_all = NULL;
static size_t g_dconv_bytes = 0;
// P4a: gate[32] device scratch (produced by gpu_delta_pre, consumed only by the on-demand
// dump path so llama.cpp layer-by-layer comparison keeps working on the fast path).
static float *g_dpgate = NULL;
static float *g_dq = NULL;
static float *g_dk = NULL;
static float *g_dv = NULL;
static float *g_dout = NULL;
static float *g_dp = NULL;

// Iteration-9 batched delta-projection scratch (all four projections share the same norm_x
// input, so we upload it once). Fixed sizes for this model: qkv=8192, gate=4096, beta/alpha=32.
static float *g_dpqkv = NULL;
static float *g_dpz   = NULL;
static float *g_dpb   = NULL;
static float *g_dpa   = NULL;
// Iteration-9 fused out-path: gated_norm computed on device (from dnet_out/z), fed straight
// into the ssm_out mat-vec so we skip the dnet_out download + gated_norm re-upload.
static float *g_dgated = NULL;
// Iteration-10: scratch for the batched shared-expert projections (gate_shexp/up_shexp share the
// norm_x input with gate_inp_shexp; one upload + back-to-back launches + one download).
static float *g_shg = NULL;   // shared gate output scratch
static float *g_shu = NULL;   // shared up output scratch
static float *g_shgi = NULL;  // shared gate_inp output scratch

// H19: persistent device KV cache for the full-attention layers (supports F32 and Q8_0 quant)
static int    g_kv_quant_mode = KV_QUANT_Q8_0;
void gpu_set_kv_quant(int mode) { g_kv_quant_mode = mode; }
int  gpu_get_kv_quant(void)     { return g_kv_quant_mode; }

static void  *g_dk_cache = NULL;
static void  *g_dv_cache = NULL;
static size_t g_kv_cache_bytes = 0;
static float *g_d_attn_scores = NULL;  // [n_head][max_seq] per-block attention scores
static size_t g_attn_scores_floats = 0;
static int    g_max_seq = 131072;
static void kv_cache_ensure(int n_attn_layers, int n_head_kv, int max_seq, int head_dim);
static void attn_scores_ensure(int n_head, int max_seq);
static float *g_aq = NULL;   // [n_head*head_dim] q scratch
static float *g_ak = NULL;   // [n_head_kv*head_dim] k scratch
static float *g_av = NULL;   // [n_head_kv*head_dim] v scratch
static float *g_aconc = NULL;// [n_head*head_dim] concat_out scratch

// P4b: persistent device residual stream buffer, normalized input buffer, and scratch
static float *g_d_res_x = NULL;        // [dim] persistent across tokens / layers
static float *g_d_norm_x = NULL;       // [dim]
static int   *g_d_pos    = NULL;       // [1] device-resident sequence position pointer for CUDA graphs

// H7 stage 2b: the two-column verify trunk (see the block comment further down).
// Column c of every [2][n] buffer lives at base + c*n.
static float *g_d_res_x2  = NULL;      // [2][dim] residual stream, both columns
static float *g_d_norm_x2 = NULL;      // [2][dim] normalized stream, both columns
static void  *g_d_nx2_q8  = NULL;      // [2][dim/32] q8_1 activations for the ncols kernels
static float *g_d_p2_qkv  = NULL;      // [2][out_qkv]
static float *g_d_p2_z    = NULL;      // [2][out_z]
static float *g_d_p2_b    = NULL;      // [2][out_beta]
static float *g_d_p2_a    = NULL;      // [2][out_alpha]
static float *g_d_logits2 = NULL;      // [2][vocab]
static int    g_n2_dim    = 0;
static float *g_dstate_snap = NULL;    // rollback copy of the GDN recurrent state
static float *g_dconv_snap  = NULL;    // rollback copy of the conv1d state
static float *g_d_shexp_out2 = NULL;   // second column's shared-expert output
static float *g_d_router_logits2 = NULL; // second column's router logits (graph-captured)
static float *g_d_router_logits = NULL;// [256]
static cudaEvent_t g_router_d2h_event = 0; // marks the router-logits + norm_x D2H (H6)
static float *g_d_shexp_out = NULL;    // [dim]
static float *g_d_shact = NULL;        // [sh_ffn_dim=512]
static float *g_d_shdown = NULL;       // [dim]
// H15: two-column variants of the shared-expert scratch. Column c of each [2][n]
// buffer lives at base + c*n, mirroring the g_d_*x2 layout of the verify trunk.
static float *g_shg2 = NULL;           // [2][sh_ffn_dim] shared gate output
static float *g_shu2 = NULL;           // [2][sh_ffn_dim] shared up output
static float *g_shgi2 = NULL;          // [2] shared gate_inp scalars
static float *g_d_shact2 = NULL;       // [2][sh_ffn_dim] SwiGLU activation
static float *g_d_shdown2 = NULL;      // [2][dim] shared down output
static float *g_d_miss_sum = NULL;     // [dim]
static float *g_d_top_values = NULL;   // [8]
static int   *g_d_top_indices = NULL;  // [8]
static int   *g_d_hit_pos = NULL;      // [8]
static float *g_d_q_full = NULL;       // [8192]
static float *g_d_k = NULL;            // [512]
static float *g_d_v = NULL;            // [512]
static float *g_d_q = NULL;            // [4096]
static float *g_d_gate = NULL;         // [4096]
static float *g_d_concat_out = NULL;   // [4096]
static float *g_d_logits = NULL;       // [vocab_size=248320]
static float *g_d_block_max = NULL;    // [256]
static int   *g_d_block_idx = NULL;    // [256]
static int   *g_d_best_token = NULL;   // [1]
static void  *g_d_out_normx_q8 = NULL; // [in_dim/32 blocks = 64] Q8_1 activations for output projection
static void  *g_d_shact_q8 = NULL;     // [sh_ffn_dim/32 blocks = 32] Q8_1 activations for shared expert down projection
// H9: quantized activation scratch for the dense Q8_0 projections (widest input 4096; 8192 covers all).
static void  *g_d_dense_x_q8 = NULL;
// H11: persistent q8_1 staging for the post-attention norm_x, written once per MoE layer by the
// shared expert and reused by the routed expert gate/up instead of re-quantizing the same vector.
static void  *g_d_norm_q8 = NULL;
// Set to 1 by gpu_shared_expert_dev when the mmvq path wrote g_d_norm_q8 for this layer, so the
// routed expert gate/up can consume it instead of re-quantizing. Stays 0 on the f32-activation
// path (exact or TUFF_MMVQ_DUAL=0), where no q8_1 of norm_x exists.
static int    g_norm_q8_valid = 0;
// H22: TUFF_ROUTER_Q8 re-quantizes the F32 router (ffn_gate_inp) to Q8_0 at load time and
// runs the router GEMV through the mmvq Q8_0 path (~316 GB/s) instead of the f32 kernel
// (206 GB/s). Numerics change (router logits -> expert selection), so it is OFF by default and
// gated on N. Cached once.
static int g_router_q8 = -1;
static int router_q8_enabled(void) {
    if (g_router_q8 < 0) { const char *e = getenv("TUFF_ROUTER_Q8"); g_router_q8 = (e && e[0] == '1') ? 1 : 0; }
    return g_router_q8;
}

// ---------------------------------------------------------------------------
// M4: per-expert LFRU VRAM cache for routed Q4_K experts.
// The device pool holds N_SLOTS slots; each slot is sized to hold one expert's three
// Q4_K tensors (gate, up, down) laid out contiguously. A host-side table tracks
// (layer, expert) -> slot with LFRU metadata (heat primary, last_used tiebreak). On a
// HIT the engine runs the three Q4_K mat-vecs on GPU from the resident slot; on a MISS
// it runs the CPU path for this token and (async or sync) fills a slot for future tokens.
// ---------------------------------------------------------------------------
#define EXPERT_CACHE_DEFAULT_SLOTS 2500

// Iteration-12 static expert split count (layers whose full expert sets are resident in VRAM).
int g_n_gpu_expert_layers = 0;

typedef struct {
    uint32_t layer;
    int      expert;
    uint8_t  valid;
    uint64_t heat;
    uint64_t last_used;
} expert_slot_meta_t;

static char *g_pool = NULL;                 // device pool
static size_t g_slot_bytes = 0;
static size_t g_tensor_off[3] = {0,0,0};    // gate/up/down offsets within a slot (down unused: CPU)
static int g_n_slots = 0;
static int g_total_experts = 0;             // block_count * expert_count (direct-map key space)
static expert_slot_meta_t *g_slots = NULL;  // host metadata
static int *g_expert_map = NULL;            // key(layer*256+expert) -> slot, or -1
static uint64_t g_tick = 0;
static uint64_t g_cache_hits = 0, g_cache_misses = 0;
static int g_cache_ready = 0;
static int g_cache_async = 0;
static cudaStream_t g_upload_stream = 0;

// H18c: TUFF_EXPERT_HIST=1 counts per-(layer,expert) routing accesses so the
// skew of the routing distribution can be measured before deciding whether a
// static pin of hot experts could beat LFRU's eviction churn. Counted for every
// lookup (hit or miss); host-only, indexed by layer*256+expert.
static uint64_t *g_expert_hist = NULL;
static int g_expert_hist_on = 0;

// Iteration-10: O(1) slot allocation. The LRU victim search used to be an O(n_slots=2200) scan on
// every miss (the dominant miss-fill latency); replace it with a doubly-linked LRU list (MRU at
// head, LRU at tail) plus a freelist for the initial fill, so alloc is O(1) in steady state.
static int *g_lru_next = NULL;   // [n_slots]
static int *g_lru_prev = NULL;   // [n_slots]
static int g_lru_head = -1;      // MRU
static int g_lru_tail = -1;      // LRU (eviction victim)
static int g_free_head = -1;     // freelist of never-used (valid==0) slots
static int g_valid_count = 0;

// Dedicated async-expert scratch + stream, so the pipelined gate/up launches overlap the CPU
// down of the previous expert without colliding with the default-stream g_dx/g_dy scratch.
static float *g_exgx = NULL;  // input (norm_x), in_dim floats
static float *g_exgy = NULL;  // gate output
static float *g_exgy2 = NULL; // up output
static float *g_exdx = NULL;  // down input (act_exp), ffn floats
static float *g_exdy = NULL;  // down output (down_exp), dim floats
static cudaStream_t g_expert_stream = 0;
static int g_expert_pipe_active = 0;
static int g_expert_down_active = 0;

// Batched async expert buffers. All routed experts of one layer share the same norm_x input
// (uploaded once to g_exgx); per-expert-position output buffers let us launch all gate/up and
// all down kernels on the stream and sync+download just twice per layer instead of per expert.
#define EXPERT_BATCH_MAX 8
static float *g_exgbat = NULL; // [EXPERT_BATCH_MAX * ffn] gate outputs
static float *g_exubbat = NULL; // [EXPERT_BATCH_MAX * ffn] up outputs
static float *g_exdbat = NULL; // [EXPERT_BATCH_MAX * dim] down outputs
static int *g_exslots = NULL; // [EXPERT_BATCH_MAX] per-position slot indices (fused batch)
// H7: per-column strides into the doubled fast-path expert scratch (see gpu_prepare).
static size_t g_ex_stride_dx = 0, g_ex_stride_dbat = 0, g_ex_stride_slots = 0;
static size_t g_ex_stride_gxq8 = 0, g_ex_stride_dxq8 = 0;
// H15: per-expert destination index tables for the partitioned (shared/exclusive) verify
// path. [4 * EXPERT_BATCH_MAX] ints: region k at base + k*EXPERT_BATCH_MAX, k = 0 shared
// col0, 1 shared col1, 2 exclusive col0, 3 exclusive col1.
static int *g_d_exdst = NULL;
static int g_batch_active = 0;
// P3: device-side q8_1 activations for the mmvq expert kernels (see the MMVQ section below).
// g_exgx_q8 holds the shared norm_x row [dim/QK8_1 blocks]; g_exdx_q8 the per-expert down
// inputs [EXPERT_BATCH_MAX * ffn/QK8_1 blocks]. Declared as void* here and used as
// block_q8_1_c* below (the struct is defined with the kernels).
static void *g_exgx_q8 = NULL;
static void *g_exdx_q8 = NULL;

static int g_cache_enabled(void) {
    return !getenv("TUFF_CPU_EXPERT") && g_cache_ready;
}

// P1: background cache-fill worker (defined after the LRU helpers below).
static void bgfill_start(void);
static void bgfill_stop(void);

void gpu_expert_cache_init_ex(model_t *m, int *p_max_seq, int fill_ram) {
    if (!m || g_pool) return;

    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)m->layers[0].ffn_gate_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn; // 512 rows * 1152 = 589824 (Q4_K)
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn; // same
    // Down is MIXED per layer: Q4_K (589824 B/expert) in most layers, Q6_K (860160 B/expert)
    // in a few deep layers. Size the slot's down region for the largest, so any layer fits and
    // all down (including the Q6_K deep layers) runs on the GPU.
    size_t down_bytes = 0;
    for (uint32_t i = 0; i < m->block_count; ++i) {
        const size_t b = ggml_row_size((ggml_type_t)m->layers[i].ffn_down_exps.info.type, ffn) * dim;
        if (b > down_bytes) down_bytes = b;
    }
    g_tensor_off[0] = 0;
    g_tensor_off[1] = gate_bytes;
    g_tensor_off[2] = gate_bytes + up_bytes;
    g_slot_bytes = gate_bytes + up_bytes + down_bytes;

    // Allocate scratch buffers first so free VRAM measurement is accurate
    if (!g_exgx)   CUDA_CHECK(cudaMalloc(&g_exgx,   (size_t)dim  * sizeof(float)));
    if (!g_exgy)   CUDA_CHECK(cudaMalloc(&g_exgy,   (size_t)ffn  * sizeof(float)));
    if (!g_exgy2)  CUDA_CHECK(cudaMalloc(&g_exgy2,  (size_t)ffn  * sizeof(float)));
    g_ex_stride_dx    = (size_t)EXPERT_BATCH_MAX * ffn;
    g_ex_stride_dbat  = (size_t)EXPERT_BATCH_MAX * dim;
    g_ex_stride_slots = (size_t)EXPERT_BATCH_MAX;
    g_ex_stride_gxq8  = (size_t)(dim / 32) * 36;
    g_ex_stride_dxq8  = (size_t)EXPERT_BATCH_MAX * (ffn / 32) * 36;
    if (!g_exdx)   CUDA_CHECK(cudaMalloc(&g_exdx,   2 * g_ex_stride_dx * sizeof(float)));
    if (!g_exdy)   CUDA_CHECK(cudaMalloc(&g_exdy,   (size_t)dim  * sizeof(float)));
    if (!g_exgbat) CUDA_CHECK(cudaMalloc(&g_exgbat, (size_t)EXPERT_BATCH_MAX * ffn * sizeof(float)));
    if (!g_exubbat) CUDA_CHECK(cudaMalloc(&g_exubbat, (size_t)EXPERT_BATCH_MAX * ffn * sizeof(float)));
    if (!g_exdbat) CUDA_CHECK(cudaMalloc(&g_exdbat, 2 * g_ex_stride_dbat * sizeof(float)));
    if (!g_exslots) CUDA_CHECK(cudaMalloc(&g_exslots, 2 * g_ex_stride_slots * sizeof(int)));
    if (!g_d_exdst) CUDA_CHECK(cudaMalloc(&g_d_exdst, 4 * EXPERT_BATCH_MAX * sizeof(int)));
    if (!g_exgx_q8) CUDA_CHECK(cudaMalloc(&g_exgx_q8, 2 * g_ex_stride_gxq8));
    if (!g_exdx_q8) CUDA_CHECK(cudaMalloc(&g_exdx_q8, 2 * g_ex_stride_dxq8));
    if (!g_expert_stream) CUDA_CHECK(cudaStreamCreate(&g_expert_stream));

    // H56: Ampere L2 Cache Persistence window (sm_86, 4 MB L2).
    // Pin 1 MB of L2 cache for hot inter-layer slots and activation scratch buffers.
    size_t l2_persisting_bytes = 0;
    cudaDeviceGetLimit(&l2_persisting_bytes, cudaLimitPersistingL2CacheSize);
    if (l2_persisting_bytes < 1024 * 1024) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1024 * 1024);
    }
    cudaStreamAttrValue l2_attr;
    memset(&l2_attr, 0, sizeof(l2_attr));
    l2_attr.accessPolicyWindow.base_ptr = (void*)g_exslots;
    l2_attr.accessPolicyWindow.num_bytes = 1024 * 1024;
    l2_attr.accessPolicyWindow.hitRatio = 1.0f;
    l2_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    l2_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(g_expert_stream, cudaStreamAttributeAccessPolicyWindow, &l2_attr);

    // Dynamic expert slot adaptation based on context size and available VRAM:
    const int n_attn = (int)m->n_attn_layers;
    const int n_head_kv = (int)m->head_count_kv;
    const int head_dim = m->key_length ? (int)m->key_length : ((int)m->embedding_length / (int)m->head_count);
    size_t kv_bytes_per_tok = 0;
    if (g_kv_quant_mode == KV_QUANT_Q8_0) {
        kv_bytes_per_tok = (size_t)2 * n_attn * n_head_kv * (head_dim / 32) * sizeof(block_q8_0);
    } else {
        kv_bytes_per_tok = (size_t)2 * n_attn * n_head_kv * head_dim * sizeof(float);
    }

    int ctx_len = p_max_seq ? *p_max_seq : 4096;
    if (ctx_len < 16) ctx_len = 4096;

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    int margin_mb = 70;
    const char *margin_env = getenv("TUFF_VRAM_MARGIN_MB");
    if (margin_env && margin_env[0] != '\0') margin_mb = atoi(margin_env);
    if (margin_mb < 16) margin_mb = 16;
    const size_t margin_bytes = (size_t)margin_mb * 1024 * 1024;

    const size_t usable_b = (free_b > margin_bytes) ? (free_b - margin_bytes) : 0;
    size_t kv_bytes = g_kv_cache_bytes;

    const char *s = getenv("TUFF_EXPERT_SLOTS");
    if (s && s[0] != '\0') {
        const int dynamic_slots = atoi(s);
        g_n_slots = (g_n_gpu_expert_layers > 0)
                  ? g_n_gpu_expert_layers * (int)m->expert_count + dynamic_slots
                  : dynamic_slots;
    } else {
        int auto_slots = (int)(usable_b / g_slot_bytes);
        // Clamp auto_slots to maximum total routed experts in model if more VRAM is available
        const int total_model_experts = (int)m->block_count * (int)m->expert_count;
        if (auto_slots > total_model_experts) auto_slots = total_model_experts;
        g_n_slots = (g_n_gpu_expert_layers > 0)
                  ? g_n_gpu_expert_layers * (int)m->expert_count + auto_slots
                  : auto_slots;
    }
    if (g_n_slots <= 0) g_n_slots = 1;

    // Fill RAM / VRAM inflation: if enabled, use any leftover VRAM (that cannot fit an expert)
    // to expand the context window to maximize VRAM utilization.
    if (fill_ram || getenv("TUFF_FILL_VRAM") || getenv("TUFF_FILL_RAM")) {
        size_t pool_bytes_used = (size_t)g_n_slots * g_slot_bytes;
        size_t total_allocated = pool_bytes_used + kv_bytes;
        if (usable_b > total_allocated && kv_bytes_per_tok > 0) {
            size_t leftover_b = usable_b - total_allocated;
            size_t extra_toks = leftover_b / kv_bytes_per_tok;
            if (extra_toks > 0) {
                int old_ctx = ctx_len;
                ctx_len += (int)extra_toks;
                if (p_max_seq) *p_max_seq = ctx_len;
                kv_bytes = (size_t)ctx_len * kv_bytes_per_tok;
                fprintf(stderr, "[VRAM-ADAPT] fill_ram: expanding context %d -> %d (+%d tokens, utilizing +%.2f MiB leftover VRAM)\n",
                        old_ctx, ctx_len, (int)extra_toks, (double)(extra_toks * kv_bytes_per_tok) / (1024.0 * 1024.0));
            }
        }
    }

    const size_t pool_bytes = (size_t)g_n_slots * g_slot_bytes;
    CUDA_CHECK(cudaMalloc((void**)&g_pool, pool_bytes));
    CUDA_CHECK(cudaMemset(g_pool, 0, pool_bytes));

    g_total_experts = (int)(m->block_count * m->expert_count);
    g_slots = (expert_slot_meta_t*)calloc((size_t)g_n_slots, sizeof(expert_slot_meta_t));
    g_expert_map = (int*)malloc((size_t)g_total_experts * sizeof(int));
    for (int i = 0; i < g_total_experts; ++i) g_expert_map[i] = -1;
    if (getenv("TUFF_EXPERT_HIST")) {
        g_expert_hist = (uint64_t*)calloc((size_t)g_total_experts, sizeof(uint64_t));
        g_expert_hist_on = g_expert_hist != NULL;
        fprintf(stderr, "[EXPERT-HIST] per-(layer,expert) routing histogram ON (%d keys)\n", g_total_experts);
    }

    // O(1) alloc: freelist of dynamic slots (valid==0); LRU list starts empty.
    g_lru_next = (int*)malloc((size_t)g_n_slots * sizeof(int));
    g_lru_prev = (int*)malloc((size_t)g_n_slots * sizeof(int));
    g_lru_head = g_lru_tail = -1;
    g_free_head = -1;
    g_valid_count = 0;
    const int n_res_slots = (g_n_gpu_expert_layers > 0) ? (g_n_gpu_expert_layers * (int)m->expert_count) : 0;
    for (int i = 0; i < g_n_slots; ++i) { g_lru_next[i] = -1; g_lru_prev[i] = -1; }
    for (int i = g_n_slots - 1; i >= n_res_slots; --i) { g_lru_next[i] = g_free_head; g_free_head = i; }

    g_cache_async = 1; // async fill uploads on a dedicated stream, synced before the batch launch
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_upload_stream, cudaStreamNonBlocking));

    g_cache_ready = 1;
    bgfill_start();
    fprintf(stderr, "[EXPERT-CACHE] Context: %d tokens (KV: %.1f MiB, %.1f KiB/tok) | Expert Slots: %d (Pool: %.1f MiB, %.2f MiB/slot) | Margin: %d MiB | Free VRAM: %.1f MiB\n",
            ctx_len, (double)kv_bytes / (1024.0 * 1024.0), (double)kv_bytes_per_tok / 1024.0,
            g_n_slots, (double)pool_bytes / (1024.0 * 1024.0), (double)g_slot_bytes / (1024.0 * 1024.0),
            margin_mb, (double)(free_b - pool_bytes - kv_bytes) / (1024.0 * 1024.0));
}

void gpu_expert_cache_init(model_t *m) {
    gpu_expert_cache_init_ex(m, NULL, 0);
}

static void lru_push_front(int slot);

static void gpu_expert_cache_fill_pinned(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;

    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)expert * up_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)expert * down_bytes;

    if (g_cache_async) {
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice));
    }
    expert_slot_meta_t *sl = &g_slots[slot];
    sl->valid = 1;
    sl->layer = layer;
    sl->expert = expert;
    sl->heat = 1;
    sl->last_used = 0;
    g_expert_map[layer * 256 + expert] = slot;
    g_valid_count++;
}

void gpu_expert_cache_prefill(model_t *m) {
    if (!g_cache_ready) return;
    if (g_n_gpu_expert_layers > 0) {
        const int n_layers = g_n_gpu_expert_layers < (int)m->block_count ? g_n_gpu_expert_layers : (int)m->block_count;
        int n_filled = 0;
        for (int l = 0; l < n_layers; ++l) {
            for (int e = 0; e < (int)m->expert_count; ++e) {
                int slot = l * (int)m->expert_count + e;
                gpu_expert_cache_fill_pinned(slot, m, (uint32_t)l, e);
                n_filled++;
            }
        }
        gpu_expert_cache_sync();
        fprintf(stderr, "[EXPERT-CACHE] pre-filled %d resident layers x %d experts = %d slots "
                        "(%.2f MiB resident expert VRAM)\n",
                n_layers, (int)m->expert_count, n_filled,
                (double)n_filled * g_slot_bytes / 1048576.0);
    } else if (g_n_slots > 0) {
        // Predictive startup prefill: distribute all available slots evenly across all layers
        const int per_layer = g_n_slots / (int)m->block_count;
        int slot = 0;
        for (int l = 0; l < (int)m->block_count && slot < g_n_slots; ++l) {
            for (int e = 0; e < per_layer && slot < g_n_slots; ++e) {
                gpu_expert_cache_fill_pinned(slot, m, (uint32_t)l, e);
                lru_push_front(slot);
                slot++;
            }
        }
        g_free_head = (slot < g_n_slots) ? slot : -1;
        gpu_expert_cache_sync();
        fprintf(stderr, "[EXPERT-CACHE] pre-filled %d dynamic slots across %d layers (%d per layer, %.2f MiB)\n",
                slot, (int)m->block_count, per_layer, (double)slot * g_slot_bytes / 1048576.0);
    }
}

// H18c: dump the routing histogram at teardown. Reports the overall Zipf-ness
// (fraction of accesses covered by the top-X% of distinct experts) and, per
// layer, what fraction of that layer's accesses its top-K experts cover.
static int hist_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) ? -1 : (x < y);
}

static void gpu_expert_hist_dump(void) {
    if (!g_expert_hist_on || !g_expert_hist) return;
    const int n_layer = g_total_experts / 256;
    const int n_expert = 256;
    uint64_t total = 0;
    int distinct = 0;
    for (int i = 0; i < g_total_experts; ++i) { total += g_expert_hist[i]; if (g_expert_hist[i]) distinct++; }
    if (total == 0) { fprintf(stderr, "[EXPERT-HIST] no accesses recorded\n"); return; }

    uint64_t *sorted = (uint64_t*)malloc((size_t)distinct * sizeof(uint64_t));
    int n = 0;
    for (int i = 0; i < g_total_experts; ++i) if (g_expert_hist[i]) sorted[n++] = g_expert_hist[i];
    qsort(sorted, (size_t)n, sizeof(uint64_t), hist_cmp);

    fprintf(stderr, "[EXPERT-HIST] accesses=%llu distinct_experts=%d (of %d)\n",
            (unsigned long long)total, distinct, g_total_experts);
    const int fracs[] = {1, 2, 5, 10, 25, 50};
    for (size_t f = 0; f < sizeof(fracs) / sizeof(fracs[0]); ++f) {
        int k = distinct * fracs[f] / 100;
        if (k <= 0) k = 1;
        uint64_t cov = 0;
        for (int i = 0; i < k && i < n; ++i) cov += sorted[i];
        fprintf(stderr, "[EXPERT-HIST]   top %2d%% of distinct experts (%d) cover %.1f%% of accesses\n",
                fracs[f], k, 100.0 * (double)cov / (double)total);
    }

    // Per-layer concentration: for the 6 most-accessed layers, the coverage of
    // their top-K experts (K = the active count 8, then 16/32/64).
    int *layer_perm = (int*)malloc((size_t)n_layer * sizeof(int));
    uint64_t *layer_total = (uint64_t*)calloc((size_t)n_layer, sizeof(uint64_t));
    for (int l = 0; l < n_layer; ++l) { layer_perm[l] = l;
        for (int e = 0; e < n_expert; ++e) layer_total[l] += g_expert_hist[l * n_expert + e]; }
    for (int i = 0; i < n_layer; ++i)
        for (int j = i + 1; j < n_layer; ++j)
            if (layer_total[layer_perm[j]] > layer_total[layer_perm[i]]) {
                int t = layer_perm[i]; layer_perm[i] = layer_perm[j]; layer_perm[j] = t;
            }
    fprintf(stderr, "[EXPERT-HIST] per-layer top-K coverage (of that layer's accesses):\n");
    for (int li = 0; li < n_layer && li < 6; ++li) {
        int l = layer_perm[li];
        uint64_t *es = (uint64_t*)malloc((size_t)n_expert * sizeof(uint64_t));
        for (int e = 0; e < n_expert; ++e) es[e] = g_expert_hist[l * n_expert + e];
        qsort(es, (size_t)n_expert, sizeof(uint64_t), hist_cmp);
        if (layer_total[l] == 0) { free(es); continue; }
        fprintf(stderr, "[EXPERT-HIST]   layer %2d: accesses=%llu  top8=%.1f%% top16=%.1f%% top32=%.1f%% top64=%.1f%%\n",
                l, (unsigned long long)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7]) / (double)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7] + es[8] + es[9] + es[10] + es[11] + es[12] + es[13] + es[14] + es[15]) / (double)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7] + es[8] + es[9] + es[10] + es[11] + es[12] + es[13] + es[14] + es[15] + es[16] + es[17] + es[18] + es[19] + es[20] + es[21] + es[22] + es[23] + es[24] + es[25] + es[26] + es[27] + es[28] + es[29] + es[30] + es[31]) / (double)layer_total[l],
                100.0 * (double)(es[0] + es[1] + es[2] + es[3] + es[4] + es[5] + es[6] + es[7] + es[8] + es[9] + es[10] + es[11] + es[12] + es[13] + es[14] + es[15] + es[16] + es[17] + es[18] + es[19] + es[20] + es[21] + es[22] + es[23] + es[24] + es[25] + es[26] + es[27] + es[28] + es[29] + es[30] + es[31] + es[32] + es[33] + es[34] + es[35] + es[36] + es[37] + es[38] + es[39] + es[40] + es[41] + es[42] + es[43] + es[44] + es[45] + es[46] + es[47] + es[48] + es[49] + es[50] + es[51] + es[52] + es[53] + es[54] + es[55] + es[56] + es[57] + es[58] + es[59] + es[60] + es[61] + es[62] + es[63]) / (double)layer_total[l]);
        free(es);
    }
    free(sorted);
    free(layer_perm);
    free(layer_total);
}

void gpu_expert_cache_free(void) {
    bgfill_stop();
    gpu_expert_hist_dump();
    if (g_pool)       { CUDA_CHECK(cudaFree(g_pool)); g_pool = NULL; }
    if (g_upload_stream) { CUDA_CHECK(cudaStreamDestroy(g_upload_stream)); g_upload_stream = 0; }
    if (g_exgx)      { CUDA_CHECK(cudaFree(g_exgx));   g_exgx = NULL; }
    if (g_exgy)      { CUDA_CHECK(cudaFree(g_exgy));   g_exgy = NULL; }
    if (g_exgy2)     { CUDA_CHECK(cudaFree(g_exgy2));  g_exgy2 = NULL; }
    if (g_exdx)      { CUDA_CHECK(cudaFree(g_exdx));   g_exdx = NULL; }
    if (g_exdy)      { CUDA_CHECK(cudaFree(g_exdy));   g_exdy = NULL; }
    if (g_exgbat)    { CUDA_CHECK(cudaFree(g_exgbat)); g_exgbat = NULL; }
    if (g_exubbat)   { CUDA_CHECK(cudaFree(g_exubbat)); g_exubbat = NULL; }
    if (g_exdbat)    { CUDA_CHECK(cudaFree(g_exdbat)); g_exdbat = NULL; }
    if (g_exslots)   { CUDA_CHECK(cudaFree(g_exslots)); g_exslots = NULL; }
    if (g_d_exdst)   { CUDA_CHECK(cudaFree(g_d_exdst)); g_d_exdst = NULL; }
    if (g_exgx_q8)   { CUDA_CHECK(cudaFree(g_exgx_q8)); g_exgx_q8 = NULL; }
    if (g_exdx_q8)   { CUDA_CHECK(cudaFree(g_exdx_q8)); g_exdx_q8 = NULL; }
    if (g_expert_stream) { CUDA_CHECK(cudaStreamDestroy(g_expert_stream)); g_expert_stream = 0; }
    free(g_slots); g_slots = NULL;
    free(g_expert_map); g_expert_map = NULL;
    free(g_lru_next); g_lru_next = NULL;
    free(g_lru_prev); g_lru_prev = NULL;
    g_lru_head = g_lru_tail = -1; g_free_head = -1; g_valid_count = 0;
    g_n_slots = 0; g_slot_bytes = 0; g_cache_ready = 0;
    free(g_expert_hist); g_expert_hist = NULL; g_expert_hist_on = 0;
}

// --- O(1) LRU + freelist bookkeeping (iteration-10) ---
// The LRU list holds all valid slots, MRU at g_lru_head, LRU at g_lru_tail. A slot that is valid
// is exactly one node in the LRU list. g_lru_next/prev[-1] are self-terminating sentinels.
static void lru_remove(int slot) {
    const int nxt = g_lru_next[slot], prv = g_lru_prev[slot];
    if (prv < 0) g_lru_head = nxt; else g_lru_next[prv] = nxt;
    if (nxt < 0) g_lru_tail = prv; else g_lru_prev[nxt] = prv;
    g_lru_next[slot] = g_lru_prev[slot] = -1;
}

static void lru_push_front(int slot) {
    g_lru_prev[slot] = -1;
    g_lru_next[slot] = g_lru_head;
    if (g_lru_head >= 0) g_lru_prev[g_lru_head] = slot;
    else g_lru_tail = slot;
    g_lru_head = slot;
}

static int freelist_pop(void) {
    const int s = g_free_head;
    if (s >= 0) g_free_head = g_lru_next[s]; // reuse next as free-list link
    return s;
}

static void expert_slot_claim(int slot, uint32_t layer, int expert) {
    expert_slot_meta_t *sl = &g_slots[slot];
    if (sl->valid) { g_expert_map[sl->layer * 256 + sl->expert] = -1; lru_remove(slot); }
    else g_valid_count++;
    sl->valid = 1;
    sl->layer = layer;
    sl->expert = expert;
    sl->heat = 1;
    sl->last_used = ++g_tick;
    g_expert_map[layer * 256 + expert] = slot;
    lru_push_front(slot);
}

// Look up (layer,expert). Returns slot index on HIT (resident, data on device), -1 on MISS.
// A HIT moves the slot to the MRU end of the LRU list (O(1)).
int gpu_expert_lookup(uint32_t layer, int expert) {
    if (!g_cache_ready) return -1;
    if (g_expert_hist_on && layer < (uint32_t)g_total_experts / 256)
        g_expert_hist[layer * 256 + expert]++;
    int slot = g_expert_map[layer * 256 + expert];
    if (slot >= 0) {
        g_slots[slot].heat++;
        g_slots[slot].last_used = ++g_tick;
        if (layer >= (uint32_t)g_n_gpu_expert_layers && g_lru_head != slot) {
            lru_remove(slot);
            lru_push_front(slot);
        }
        g_cache_hits++;
    } else {
        g_cache_misses++;
    }
    return slot;
}

// Device pointer for tensor `which` (0=gate,1=up,2=down) inside slot `slot`.
const void *gpu_expert_dev_ptr(int slot, int which) {
    return g_pool + (size_t)slot * g_slot_bytes + g_tensor_off[which];
}

void gpu_expert_cache_fill(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;

    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)expert * up_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)expert * down_bytes;

    if (g_cache_async) {
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice));
    }
    expert_slot_claim(slot, layer, expert);
}

// Split-fill for overlapping the miss upload behind the batch compute. gate+up are needed by the
// gate/up batch (Phase A); down is needed only by the down batch (Phase C). Filling gate+up first
// lets Phase A start while the larger down upload is deferred and overlapped with the gate/up
// compute + swiglu, shrinking the per-layer blocking fill. Both are async on the upload stream.
// fill_gateup claims the slot (so lookups hit); fill_down only uploads down into an already-claimed
// slot. Bit-identical: same bytes, same slots, just reordered async uploads.
void gpu_expert_cache_fill_gateup(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = ggml_row_size(gtype, dim) * ffn;
    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)expert * up_bytes;
    CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
    expert_slot_claim(slot, layer, expert);
}

void gpu_expert_cache_fill_down(int slot, const model_t *m, uint32_t layer, int expert) {
    const model_layer_t *L = &m->layers[layer];
    const int dim = (int)m->embedding_length;
    const int ffn = (int)m->expert_ffn_length;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;
    char *base = g_pool + (size_t)slot * g_slot_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)expert * down_bytes;
    CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
}

// Reserve a slot for a new expert (miss): free slot first, else LRU victim. Does NOT upload.
// O(1) via freelist + doubly-linked LRU list (iteration-10). Pure LRU naturally evicts stale
// domain working sets when the conversation shifts topic.
int gpu_expert_cache_alloc_slot(uint32_t layer, int expert) {
    int victim = freelist_pop();
    if (victim < 0) {
        victim = g_lru_tail;
    }
    return victim;
}

void gpu_expert_cache_stats(unsigned long long *hits, unsigned long long *misses) {
    if (hits) *hits = g_cache_hits;
    if (misses) *misses = g_cache_misses;
}

int gpu_expert_cache_active(void) { return g_cache_enabled(); }

// Ensure any in-flight async expert uploads are visible to the default stream before the
// engine reads a resident slot (cheap when the upload stream is idle). No-op if not async.
void gpu_expert_cache_sync(void) {
    if (g_cache_async && g_upload_stream) CUDA_CHECK(cudaStreamSynchronize(g_upload_stream));
}

// ---------------------------------------------------------------------------
// P1 (parity.md sec.4): background cache fills.
//
// The old miss policy pushed 1.95 MiB of expert weights over PCIe *from the decode thread*,
// in the critical path, per missing expert. That cost is not only PCIe: the GGUF weights are
// pageable mmap'd memory, so cudaMemcpyAsync stages them through a driver bounce buffer and
// blocks the calling thread while it does. Measured 21.0 ms/token = 35% of the token.
//
// Here the decode thread never copies. It only *reserves* a slot (host bookkeeping, O(1)) and
// hands (layer, expert, slot) to a dedicated fill thread, which does the staging memcpy +
// H2D on g_upload_stream and records an event. The decode thread later polls that event with
// cudaEventQuery (non-blocking) and only then publishes the slot into the lookup map.
//
// Race freedom, by construction:
//   * A reserved slot is removed from BOTH the freelist and the LRU list and its old
//     (layer,expert) key is erased from g_expert_map. So while a fill is in flight the slot is
//     unreachable: no lookup can return it (no kernel can read it) and no later reservation can
//     pick it as a victim.
//   * The slot is published (expert_slot_claim) only after cudaEventQuery reports the copy
//     complete, so any kernel that can subsequently see the slot runs after the copy.
//   * All slot-table / LRU / map mutation stays on the decode thread. The fill thread touches
//     only device memory of slots the decode thread promised not to touch, plus its own stream
//     and events. No lock is needed on the slot table.
//   * Reservations are issued only after the layer's expert kernels have completed
//     (gpu_expert_moe_down / gpu_expert_batch_down_sync_download both sync g_expert_stream),
//     so no in-flight kernel is reading the pool when a victim is chosen.
// ---------------------------------------------------------------------------
// P1 queue depth. Must fit a full token's fills with headroom (2 tokens' worth: up to
// 40 layers x g_fill_budget 2 = 80, plus slack). A power of two since the code indexes with
// `% BGFILL_QUEUE`. The old 8 gated issuance: g_bg_commit only advanced at the once-per-token
// drain, so at most 8 fills could be issued per token (measured 8.00/token) while a token
// routed up to 80 misses -> ~80% of routed experts took the CPU path every token. Deep enough
// that the queue never gates issuance; the commit boundary is now the sole ordering control.
#define BGFILL_QUEUE 256

typedef struct {
    const model_t *m;
    uint32_t layer;
    int      expert;
    int      slot;
    unsigned long token;   // issuing token index, recorded by the decode thread at reserve time
    cudaEvent_t ev;
} bgfill_req_t;

static bgfill_req_t g_bg[BGFILL_QUEUE];
static unsigned long g_bg_prod = 0;    // decode thread only (published with release)
static unsigned long g_bg_sub  = 0;    // fill thread only  (published with release)
static unsigned long g_bg_commit = 0;  // decode thread only
static unsigned long g_bg_token = 0;   // decode thread only: token index of the fills being issued
static pthread_t g_bg_thread;
static pthread_mutex_t g_bg_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_bg_cv  = PTHREAD_COND_INITIALIZER;
static int g_bg_running = 0;
static int g_bg_stop = 0;
static unsigned long long g_bg_issued = 0, g_bg_committed = 0;

static uint8_t *g_pinned_stage = NULL;

static void bgfill_do(const bgfill_req_t *r) {
    const model_layer_t *L = &r->m->layers[r->layer];
    const int dim = (int)r->m->embedding_length;
    const int ffn = (int)r->m->expert_ffn_length;
    const ggml_type_t gtype = (ggml_type_t)L->ffn_gate_exps.info.type;
    const ggml_type_t dtype = (ggml_type_t)L->ffn_down_exps.info.type;
    const size_t gate_bytes = ggml_row_size(gtype, dim) * ffn;
    const size_t up_bytes   = gate_bytes;
    const size_t down_bytes = ggml_row_size(dtype, ffn) * dim;
    char *base = g_pool + (size_t)r->slot * g_slot_bytes;
    const uint8_t *gate = (const uint8_t*)L->ffn_gate_exps.data + (size_t)r->expert * gate_bytes;
    const uint8_t *up   = (const uint8_t*)L->ffn_up_exps.data   + (size_t)r->expert * up_bytes;
    const uint8_t *down = (const uint8_t*)L->ffn_down_exps.data + (size_t)r->expert * down_bytes;

    if (g_pinned_stage) {
        const int idx = (int)(r - g_bg);
        uint8_t *stage = g_pinned_stage + (size_t)idx * g_slot_bytes;
        memcpy(stage + g_tensor_off[0], gate, gate_bytes);
        memcpy(stage + g_tensor_off[1], up,   up_bytes);
        memcpy(stage + g_tensor_off[2], down, down_bytes);
        CUDA_CHECK(cudaMemcpyAsync(base, stage, g_slot_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[0], gate, gate_bytes, cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[1], up,   up_bytes,   cudaMemcpyHostToDevice, g_upload_stream));
        CUDA_CHECK(cudaMemcpyAsync(base + g_tensor_off[2], down, down_bytes, cudaMemcpyHostToDevice, g_upload_stream));
    }
    CUDA_CHECK(cudaEventRecord(r->ev, g_upload_stream));
}

static void *bgfill_thread_fn(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_bg_mtx);
        while (!g_bg_stop && g_bg_sub == __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE))
            pthread_cond_wait(&g_bg_cv, &g_bg_mtx);
        const int stop = g_bg_stop && g_bg_sub == __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE);
        pthread_mutex_unlock(&g_bg_mtx);
        if (stop) break;
        bgfill_do(&g_bg[g_bg_sub % BGFILL_QUEUE]);
        __atomic_store_n(&g_bg_sub, g_bg_sub + 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void bgfill_start(void) {
    if (g_bg_running) return;
    if (!g_pinned_stage && g_slot_bytes > 0) {
        CUDA_CHECK(cudaMallocHost((void**)&g_pinned_stage, (size_t)BGFILL_QUEUE * g_slot_bytes));
    }
    for (int i = 0; i < BGFILL_QUEUE; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_bg[i].ev, cudaEventDisableTiming));
    }
    g_bg_prod = g_bg_sub = g_bg_commit = g_bg_token = 0;
    g_bg_stop = 0;
    if (pthread_create(&g_bg_thread, NULL, bgfill_thread_fn, NULL) != 0) {
        fprintf(stderr, "[FATAL ERROR] gpu: failed to create the background expert-fill thread\n");
        exit(EXIT_FAILURE);
    }
    g_bg_running = 1;
}

static void bgfill_stop(void) {
    if (!g_bg_running) return;
    pthread_mutex_lock(&g_bg_mtx);
    g_bg_stop = 1;
    pthread_cond_broadcast(&g_bg_cv);
    pthread_mutex_unlock(&g_bg_mtx);
    pthread_join(g_bg_thread, NULL);
    // Drain: everything the fill thread submitted has been recorded on g_upload_stream.
    if (g_upload_stream) CUDA_CHECK(cudaStreamSynchronize(g_upload_stream));
    for (int i = 0; i < BGFILL_QUEUE; ++i) {
        if (g_bg[i].ev) { CUDA_CHECK(cudaEventDestroy(g_bg[i].ev)); g_bg[i].ev = 0; }
    }
    g_bg_running = 0;
}

// Reserve a victim slot and make it unreachable (see the race-freedom note above). Decode
// thread only. Returns -1 when no slot can be spared (every slot is reserved in flight).
static int expert_slot_reserve(void) {
    int slot = freelist_pop();
    if (slot < 0) {
        slot = g_lru_tail;
        if (slot < 0) return -1;   // all slots reserved in flight
        lru_remove(slot);
    }
    expert_slot_meta_t *sl = &g_slots[slot];
    if (sl->valid) {
        g_expert_map[sl->layer * 256 + sl->expert] = -1;
        sl->valid = 0;
        g_valid_count--;
    }
    return slot;
}

int gpu_expert_cache_bgfill(const model_t *m, uint32_t layer, int expert) {
    if (!g_cache_ready || !g_bg_running) return 0;
    if (g_bg_prod - g_bg_commit >= BGFILL_QUEUE) return 0;   // queue full -> skip this fill
    const int slot = expert_slot_reserve();
    if (slot < 0) return 0;
    bgfill_req_t *r = &g_bg[g_bg_prod % BGFILL_QUEUE];
    r->m = m; r->layer = layer; r->expert = expert; r->slot = slot;
    r->token = g_bg_token;   // deterministic function of token history, not of copy latency
    pthread_mutex_lock(&g_bg_mtx);
    __atomic_store_n(&g_bg_prod, g_bg_prod + 1, __ATOMIC_RELEASE);
    pthread_cond_signal(&g_bg_cv);
    pthread_mutex_unlock(&g_bg_mtx);
    g_bg_issued++;
    return 1;
}

void gpu_expert_cache_bgfill_poll(void) {
    if (!g_bg_running) return;
    const unsigned long t = g_bg_token;
    const unsigned long prod = __atomic_load_n(&g_bg_prod, __ATOMIC_ACQUIRE);
    while (g_bg_commit < prod) {
        bgfill_req_t *r = &g_bg[g_bg_commit % BGFILL_QUEUE];
        if (r->token > t) break;
        while (g_bg_commit >= __atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE))
            sched_yield();
        CUDA_CHECK(cudaEventSynchronize(r->ev));
        expert_slot_claim(r->slot, r->layer, r->expert);
        g_bg_commit++;
        g_bg_committed++;
    }
    g_bg_token = t + 1;
}

void gpu_expert_cache_bgfill_poll_ready(void) {
    if (!g_bg_running) return;
    const unsigned long sub = __atomic_load_n(&g_bg_sub, __ATOMIC_ACQUIRE);
    while (g_bg_commit < sub) {
        bgfill_req_t *r = &g_bg[g_bg_commit % BGFILL_QUEUE];
        cudaError_t st = cudaEventQuery(r->ev);
        if (st == cudaSuccess) {
            expert_slot_claim(r->slot, r->layer, r->expert);
            g_bg_commit++;
            g_bg_committed++;
        } else if (st == cudaErrorNotReady) {
            break;
        } else {
            CUDA_CHECK(st);
        }
    }
}

void gpu_expert_cache_bgfill_stats(unsigned long long *issued, unsigned long long *committed) {
    if (issued) *issued = g_bg_issued;
    if (committed) *committed = g_bg_committed;
}

// Debug: read back a slot tensor region (which: 0=gate 1=up 2=down) and count bytes differing
// from a host reference. Returns the mismatch count, or -1 on invalid slot.
int gpu_expert_debug_check_down(int slot, const void *host_ref, size_t nbytes) {
    if (!g_cache_ready || slot < 0) return -1;
    if (!g_cache_ready || slot < 0) return -1;
    unsigned char *buf = (unsigned char*)malloc(nbytes);
    CUDA_CHECK(cudaMemcpy(buf, gpu_expert_dev_ptr(slot, 2), nbytes, cudaMemcpyDeviceToHost));
    const unsigned char *ref = (const unsigned char*)host_ref;
    int ndiff = 0;
    for (size_t i = 0; i < nbytes; ++i) if (buf[i] != ref[i]) ndiff++;
    free(buf);
    return ndiff;
}

__device__ __forceinline__ float h2f(uint16_t h) {
    __half raw = *reinterpret_cast<__half *>(&h);
    return __half2float(raw);
}

// Row-parallel dequant + dot for Q8_0. One block per output row; each thread accumulates a
// strided slice of in_dim, then a shared-memory reduction yields dst[row].
__global__ void mul_mat_vec_q8_0_kernel(const block_q8_0 * __restrict__ w,
                                        const float * __restrict__ x,
                                        float * __restrict__ y,
                                        int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nblk = in_dim / QK8_0;
    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

    float acc = 0.0f;
    for (int c = tid; c < in_dim; c += blockDim.x) {
        const block_q8_0 &b = rw[c / QK8_0];
        const float d = h2f(b.d);
        acc += (float)b.qs[c % QK8_0] * d * x[c];
    }

    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// Autoregressive gated delta-net per-head state update, offloaded from engine.c's CPU loop.
// Each (head, row) is fully independent given qh/kh/vh and the head's S slice, so the whole
// pass maps to one thread per (head, row). Per row this replicates the exact CPU math and
// accumulation order (j = 0..head_k_dim-1) of build_delta_net_autoregressive():
// Vectorized with float4, shared memory cached q/k, and fused loops for peak compute/bandwidth efficiency.
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
__global__ void mul_mat_vec_f32_kernel(const float * __restrict__ w,
                                       const float * __restrict__ x,
                                       float * __restrict__ y,
                                       int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * __restrict__ rw = w + (size_t)row * in_dim;

    float acc = 0.0f;
    for (int c = tid; c < in_dim; c += blockDim.x) {
        acc += rw[c] * x[c];
    }

    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// Q4_K mat-vec, matching mat_vec_q4_K() exactly (strict scalar sequential accumulation).
// One thread per output row iterates c in increasing order, replicating the CPU's volatile
// scalar sum bit-for-bit (both compile `sum += (dv*qn - mv)*x[c]` with fmad enabled, so the
// same mul->fma contractions occur). The Q4_K super-block layout (d/dmin fp16, 12-byte
// sub-block scales, both nibbles across 4 chunks of 64) is decoded via get_scale_min_k4().
__global__ void mul_mat_vec_q4_K_kernel(const block_q4_K * __restrict__ w,
                                        const float * __restrict__ x,
                                        float * __restrict__ y,
                                        int in_dim) {
    const int row = blockIdx.x;
    const int nblk = in_dim / QK_K;
    const block_q4_K * __restrict__ rw = w + (size_t)row * nblk;

    // Single thread per row, strict scalar sequential accumulation in the exact CPU order
    // (super-block -> 64-element chunk -> low nibble 32 -> high nibble 32). Iterated
    // incrementally (no per-element div/mod) so indexing is cheap while remaining bit-identical
    // to mat_vec_q4_K() and the previous explicit-div/mod kernel.
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
            // sub-block 0 (low nibble)
            if (is < 4) { sc = blk.scales[is] & 63; m = blk.scales[is + 4] & 63; }
            else        { sc = (blk.scales[is + 4] & 0xF) | ((blk.scales[is - 4] >> 6) << 4);
                           m  = (blk.scales[is + 4] >> 4)  | ((blk.scales[is]     >> 6) << 4); }
            const float d1 = d * (float)sc;
            const float m1 = mn * (float)m;
            // sub-block 1 (high nibble)
            if (is + 1 < 4) { sc = blk.scales[is + 1] & 63; m = blk.scales[is + 5] & 63; }
            else            { sc = (blk.scales[is + 5] & 0xF) | ((blk.scales[is - 3] >> 6) << 4);
                               m  = (blk.scales[is + 5] >> 4)  | ((blk.scales[is + 1] >> 6) << 4); }
            const float d2 = d * (float)sc;
            const float m2 = mn * (float)m;

            for (int l = 0; l < 32; ++l) {
                // The CPU mat_vec_q4_K() reference accumulates `sum += t*x` WITHOUT FMA contraction
                // (GCC keeps a separate mul+add here, rounding the product), so match it with an
                // explicit __fmul_rn + __fadd_rn instead of __fmaf_rn to stay bit-exact.
                const float t = __fmaf_rn(d1, (float)(q[l] & 0xF), -m1);
                sum = __fadd_rn(sum, __fmul_rn(t, x[cidx]));
                cidx++;
            }
            for (int l = 0; l < 32; ++l) {
                const float t = __fmaf_rn(d2, (float)(q[l] >> 4), -m2);
                sum = __fadd_rn(sum, __fmul_rn(t, x[cidx]));
                cidx++;
            }
            q += 32;
            is += 2;
        }
    }
    y[row] = sum;
}

// Q6_K mat-vec. Reproduces dequantize_q6_K() indexing exactly (including the sub-block
// scale array layout) so the output projection matches the CPU oracle token-for-token.
__global__ void mul_mat_vec_q6_K_kernel(const block_q6_K * __restrict__ w,
                                        const float * __restrict__ x,
                                        float * __restrict__ y,
                                        int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nblk = in_dim / QK_K;
    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;

    float acc = 0.0f;
    for (int c = tid; c < in_dim; c += blockDim.x) {
        const block_q6_K &b = rw[c / QK_K];
        const int p = c % QK_K;
        const int chunk  = p / 128;
        const int p128   = p % 128;
        const int l      = p128 % 32;
        const int slot   = p128 / 32;
        const float d    = h2f(b.d);
        const uint8_t ql_byte = b.ql[chunk * 64 + l + ((slot & 1) ? 32 : 0)];
        const uint8_t qh_byte = b.qh[chunk * 32 + l];
        const int qn = ((slot < 2) ? (ql_byte & 0xF) : (ql_byte >> 4)) |
                       (((qh_byte >> (slot * 2)) & 3) << 4);
        const int8_t q = (int8_t)(qn - 32);
        const int8_t s = b.scales[chunk * 8 + (l / 16) + slot * 2];
        acc += (float)d * (float)s * (float)q * x[c];
    }

    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// ---------------------------------------------------------------------------
// Warp-per-row Fast Mat-Vec Kernels (Warp Shuffles, Register-Only Accumulation)
// ---------------------------------------------------------------------------

__global__ void mul_mat_vec_q8_0_fast_kernel(const block_q8_0 * __restrict__ w,
                                             const float * __restrict__ x,
                                             float * __restrict__ y,
                                             int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_x[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_x[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const int nblk = in_dim / QK8_0;
        const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            const block_q8_0 &b0 = rw[b + 0];
            const block_q8_0 &b1 = rw[b + 1];
            const block_q8_0 &b2 = rw[b + 2];
            const block_q8_0 &b3 = rw[b + 3];

            float d0 = (lane_id == 0) ? h2f(b0.d) : 0.0f;
            float d1 = (lane_id == 0) ? h2f(b1.d) : 0.0f;
            float d2 = (lane_id == 0) ? h2f(b2.d) : 0.0f;
            float d3 = (lane_id == 0) ? h2f(b3.d) : 0.0f;

            d0 = __shfl_sync(0xffffffff, d0, 0);
            d1 = __shfl_sync(0xffffffff, d1, 0);
            d2 = __shfl_sync(0xffffffff, d2, 0);
            d3 = __shfl_sync(0xffffffff, d3, 0);

            sum += ((float)b0.qs[lane_id] * s_x[(b + 0) * QK8_0 + lane_id]) * d0;
            sum += ((float)b1.qs[lane_id] * s_x[(b + 1) * QK8_0 + lane_id]) * d1;
            sum += ((float)b2.qs[lane_id] * s_x[(b + 2) * QK8_0 + lane_id]) * d2;
            sum += ((float)b3.qs[lane_id] * s_x[(b + 3) * QK8_0 + lane_id]) * d3;
        }
        for (; b < nblk; ++b) {
            const block_q8_0 &blk = rw[b];
            float d = (lane_id == 0) ? h2f(blk.d) : 0.0f;
            d = __shfl_sync(0xffffffff, d, 0);
            sum += ((float)blk.qs[lane_id] * s_x[b * QK8_0 + lane_id]) * d;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__global__ void mul_mat_vec_f32_fast_kernel(const float * __restrict__ w,
                                            const float * __restrict__ x,
                                            float * __restrict__ y,
                                            int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xf[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xf[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const float * __restrict__ rw = w + (size_t)row * in_dim;
        float sum = 0.0f;
        for (int c = lane_id; c < in_dim; c += 32) {
            sum += rw[c] * s_xf[c];
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__global__ void mul_mat_vec_q6_K_fast_kernel(const block_q6_K * __restrict__ w,
                                             const float * __restrict__ x,
                                             float * __restrict__ y,
                                             int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xq6[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xq6[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const int nblk = in_dim >> 8;
        const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;

        float acc = 0.0f;
        const int l = lane_id & 31;
        const int l_div_16 = l >> 4;

        for (int c = lane_id; c < in_dim; c += 32) {
            const block_q6_K &b = rw[c >> 8];
            const int p = c & 255;
            const int chunk  = p >> 7;
            const int slot   = (p & 127) >> 5;
            const float d    = h2f(b.d);
            const uint8_t ql_byte = b.ql[(chunk << 6) + l + ((slot & 1) ? 32 : 0)];
            const uint8_t qh_byte = b.qh[(chunk << 5) + l];
            const int qn = ((slot < 2) ? (ql_byte & 0xF) : (ql_byte >> 4)) |
                           (((qh_byte >> (slot * 2)) & 3) << 4);
            const int8_t q = (int8_t)(qn - 32);
            const int8_t s = b.scales[(chunk << 3) + l_div_16 + (slot << 1)];
            acc += d * (float)s * (float)q * s_xq6[c];
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, mask);
        }

        if (lane_id == 0) {
            y[row] = acc;
        }
    }
}

// H9 (defined with the q8_1 machinery further down): __dp4a mmvq form of the dense Q8_0 GEMV.
void gpu_mmvq_q8_0_dense(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim, int accum);
void gpu_mmvq_q8_0_dense_dual(const block_q8_0 *w0, float *y0, int out0,
                              const block_q8_0 *w1, float *y1, int out1,
                              const float *x, int in_dim);

static void launch_mat_vec_q8_0(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim) {
    if (!tuff_exact()) { gpu_mmvq_q8_0_dense(w, x, y, in_dim, out_dim, 0); return; }
    dim3 block(32, 8);
    dim3 grid((out_dim + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_q8_0_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
    CUDA_CHECK_LAST();
}

__global__ void mul_mat_vec_q8_0_accum_fast_kernel(const block_q8_0 * __restrict__ w,
                                                   const float * __restrict__ x,
                                                   float * __restrict__ y,
                                                   int in_dim, int out_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xa[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xa[i] = x[i];
    }
    __syncthreads();

    if (row < out_dim) {
        const int nblk = in_dim / QK8_0;
        const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            const block_q8_0 &b0 = rw[b + 0];
            const block_q8_0 &b1 = rw[b + 1];
            const block_q8_0 &b2 = rw[b + 2];
            const block_q8_0 &b3 = rw[b + 3];

            float d0 = (lane_id == 0) ? h2f(b0.d) : 0.0f;
            float d1 = (lane_id == 0) ? h2f(b1.d) : 0.0f;
            float d2 = (lane_id == 0) ? h2f(b2.d) : 0.0f;
            float d3 = (lane_id == 0) ? h2f(b3.d) : 0.0f;

            d0 = __shfl_sync(0xffffffff, d0, 0);
            d1 = __shfl_sync(0xffffffff, d1, 0);
            d2 = __shfl_sync(0xffffffff, d2, 0);
            d3 = __shfl_sync(0xffffffff, d3, 0);

            sum += ((float)b0.qs[lane_id] * s_xa[(b + 0) * QK8_0 + lane_id]) * d0;
            sum += ((float)b1.qs[lane_id] * s_xa[(b + 1) * QK8_0 + lane_id]) * d1;
            sum += ((float)b2.qs[lane_id] * s_xa[(b + 2) * QK8_0 + lane_id]) * d2;
            sum += ((float)b3.qs[lane_id] * s_xa[(b + 3) * QK8_0 + lane_id]) * d3;
        }
        for (; b < nblk; ++b) {
            const block_q8_0 &blk = rw[b];
            float d = (lane_id == 0) ? h2f(blk.d) : 0.0f;
            d = __shfl_sync(0xffffffff, d, 0);
            sum += ((float)blk.qs[lane_id] * s_xa[b * QK8_0 + lane_id]) * d;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] += sum;
        }
    }
}

static void launch_mat_vec_q8_0_accum(const block_q8_0 *w, const float *x, float *y, int in_dim, int out_dim) {
    if (!tuff_exact()) { gpu_mmvq_q8_0_dense(w, x, y, in_dim, out_dim, 1); return; }
    dim3 block(32, 8);
    dim3 grid((out_dim + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_q8_0_accum_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
    CUDA_CHECK_LAST();
}

__global__ void mul_mat_vec_q8_0_dual_kernel(const block_q8_0 * __restrict__ w0, float * __restrict__ y0, int out0,
                                             const block_q8_0 * __restrict__ w1, float * __restrict__ y1, int out1,
                                             const float * __restrict__ x, int in_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int global_row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_x[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_x[i] = x[i];
    }
    __syncthreads();

    if (global_row < out0 + out1) {
        const bool is_first = (global_row < out0);
        const int row = is_first ? global_row : (global_row - out0);
        const block_q8_0 * __restrict__ w = is_first ? w0 : w1;
        float * __restrict__ y = is_first ? y0 : y1;

        const int nblk = in_dim / QK8_0;
        const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;

        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            const block_q8_0 &b0 = rw[b + 0];
            const block_q8_0 &b1 = rw[b + 1];
            const block_q8_0 &b2 = rw[b + 2];
            const block_q8_0 &b3 = rw[b + 3];

            float d0 = (lane_id == 0) ? h2f(b0.d) : 0.0f;
            float d1 = (lane_id == 0) ? h2f(b1.d) : 0.0f;
            float d2 = (lane_id == 0) ? h2f(b2.d) : 0.0f;
            float d3 = (lane_id == 0) ? h2f(b3.d) : 0.0f;

            d0 = __shfl_sync(0xffffffff, d0, 0);
            d1 = __shfl_sync(0xffffffff, d1, 0);
            d2 = __shfl_sync(0xffffffff, d2, 0);
            d3 = __shfl_sync(0xffffffff, d3, 0);

            sum += ((float)b0.qs[lane_id] * s_x[(b + 0) * QK8_0 + lane_id]) * d0;
            sum += ((float)b1.qs[lane_id] * s_x[(b + 1) * QK8_0 + lane_id]) * d1;
            sum += ((float)b2.qs[lane_id] * s_x[(b + 2) * QK8_0 + lane_id]) * d2;
            sum += ((float)b3.qs[lane_id] * s_x[(b + 3) * QK8_0 + lane_id]) * d3;
        }
        for (; b < nblk; ++b) {
            const block_q8_0 &blk = rw[b];
            float d = (lane_id == 0) ? h2f(blk.d) : 0.0f;
            d = __shfl_sync(0xffffffff, d, 0);
            sum += ((float)blk.qs[lane_id] * s_x[b * QK8_0 + lane_id]) * d;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

// TUFF_MMVQ_DUAL=0 keeps the f32-activation kernel for the paired projections (delta
// qkv+gate, shared expert gate+up) — the ones whose output feeds the recurrent state, where
// activation-quantization error compounds multiplicatively through decay. Diagnostic for the
// N = 60 -> 56 trade measured in iteration 39. Cached once and shared (H11 needs to know whether
// the shared expert produced a q8_1 of norm_x the routed experts can reuse).
extern "C" int mmvq_dual_enabled(void) {
    static int mmvq_dual = -1;
    if (mmvq_dual < 0) { const char *e = getenv("TUFF_MMVQ_DUAL"); mmvq_dual = (e && e[0] == '0') ? 0 : 1; }
    return mmvq_dual;
}

static void launch_mat_vec_q8_0_dual(const block_q8_0 *w0, float *y0, int out0,
                                    const block_q8_0 *w1, float *y1, int out1,
                                    const float *x, int in_dim) {
    if (!tuff_exact() && mmvq_dual_enabled()) { gpu_mmvq_q8_0_dense_dual(w0, y0, out0, w1, y1, out1, x, in_dim); return; }
    dim3 block(32, 8);
    dim3 grid((out0 + out1 + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_q8_0_dual_kernel<<<grid, block, smem>>>(w0, y0, out0, w1, y1, out1, x, in_dim);
    CUDA_CHECK_LAST();
}

__global__ void mul_mat_vec_f32_dual_kernel(const float * __restrict__ w0, float * __restrict__ y0, int out0,
                                            const float * __restrict__ w1, float * __restrict__ y1, int out1,
                                            const float * __restrict__ x, int in_dim) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int total_threads = blockDim.y * 32;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int global_row = blockIdx.x * blockDim.y + warp_id;

    extern __shared__ float s_xf[];
    for (int i = tid; i < in_dim; i += total_threads) {
        s_xf[i] = x[i];
    }
    __syncthreads();

    if (global_row < out0 + out1) {
        const bool is_first = (global_row < out0);
        const int row = is_first ? global_row : (global_row - out0);
        const float * __restrict__ w = is_first ? w0 : w1;
        float * __restrict__ y = is_first ? y0 : y1;

        const float * __restrict__ rw = w + (size_t)row * in_dim;
        float sum = 0.0f;
        for (int c = lane_id; c < in_dim; c += 32) {
            sum += rw[c] * s_xf[c];
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

static void launch_mat_vec_f32_dual(const float *w0, float *y0, int out0,
                                   const float *w1, float *y1, int out1,
                                   const float *x, int in_dim) {
    dim3 block(32, 8);
    dim3 grid((out0 + out1 + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_f32_dual_kernel<<<grid, block, smem>>>(w0, y0, out0, w1, y1, out1, x, in_dim);
    CUDA_CHECK_LAST();
}

static void launch_mat_vec_f32(const float *w, const float *x, float *y, int in_dim, int out_dim) {
    dim3 block(32, 8);
    dim3 grid((out_dim + 7) / 8);
    size_t smem = (size_t)in_dim * sizeof(float);
    mul_mat_vec_f32_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
    CUDA_CHECK_LAST();
}

static void upload_dense(tensor_t *t) {
    if (!t->data) return;
    const int type = (int)t->info.type;
    if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_F32 && type != GGML_TYPE_Q6_K) return;
    void *dev = NULL;
    CUDA_CHECK(cudaMalloc(&dev, t->info.size_bytes));
    CUDA_CHECK(cudaMemcpy(dev, t->data, t->info.size_bytes, cudaMemcpyHostToDevice));
    t->dev = dev;
}

void gpu_prepare_ex(model_t *m, int *p_max_seq, int fill_ram) {
    if (!m) return;
    // Not a knob: cudaDeviceScheduleSpin was measured in iteration 41 at +0.15% over four
    // order-balanced rounds — the driver's Auto policy already spins at this thread count.
    CUDA_CHECK(cudaFree(0));

    if (m->gguf && m->gguf->mmap_addr && m->gguf->file_size > 0) {
        cudaError_t err = cudaHostRegister(m->gguf->mmap_addr, m->gguf->file_size, cudaHostRegisterReadOnly);
        if (err == cudaSuccess) {
            fprintf(stderr, "[GPU-DMA] Successfully pinned %.2f GiB GGUF model memory with cudaHostRegister\n",
                    (double)m->gguf->file_size / (1024.0 * 1024.0 * 1024.0));
        } else {
            fprintf(stderr, "[GPU-DMA] Note: cudaHostRegister returned %d (%s), continuing with pageable fallback\n",
                    (int)err, cudaGetErrorString(err));
            cudaGetLastError();
        }
    }

    if (!g_dx) CUDA_CHECK(cudaMalloc(&g_dx, GPU_IN_FLOATS * sizeof(float)));
    if (!g_dy) CUDA_CHECK(cudaMalloc(&g_dy, GPU_OUT_FLOATS * sizeof(float)));

    // Persistent device copy of the gated delta-net recurrent state, so the state-update
    // kernel never round-trips it over PCIe. Sized to the model and zeroed here; re-zeroed
    // per engine run via gpu_reset_delta_state().
    g_dstate_bytes = (size_t)m->n_delta_layers * 32 * 128 * 128 * sizeof(float);
    if (g_dstate_bytes > 0) {
        if (!g_dstate_all) CUDA_CHECK(cudaMalloc(&g_dstate_all, g_dstate_bytes));
        CUDA_CHECK(cudaMemset(g_dstate_all, 0, g_dstate_bytes));
        if (!g_dstate_snap) CUDA_CHECK(cudaMalloc(&g_dstate_snap, g_dstate_bytes));
        CUDA_CHECK(cudaMemset(g_dstate_snap, 0, g_dstate_bytes));
    }

    // P4a: persistent device mirror of the conv1d state, following exactly the g_dstate_all
    // pattern above (sized to the model, zeroed here and re-zeroed per engine run).
    g_dconv_bytes = (size_t)m->n_delta_layers * m->ssm_conv_kernel * m->ssm_qkv_dim * sizeof(float);
    if (g_dconv_bytes > 0) {
        if (!g_dconv_all) CUDA_CHECK(cudaMalloc(&g_dconv_all, g_dconv_bytes));
        CUDA_CHECK(cudaMemset(g_dconv_all, 0, g_dconv_bytes));
        if (!g_dconv_snap) CUDA_CHECK(cudaMalloc(&g_dconv_snap, g_dconv_bytes));
        CUDA_CHECK(cudaMemset(g_dconv_snap, 0, g_dconv_bytes));
    }

    // output_weight (Q6_K) is dense -> uploaded.
    upload_dense(&m->output_weight);
    upload_dense(&m->output_norm);

    const int dim = (int)m->embedding_length;
    const int vocab_size = (int)m->vocab_size;
    const int sh_ffn = (int)m->expert_shared_ffn_len;

    if (!g_d_res_x)        CUDA_CHECK(cudaMalloc(&g_d_res_x, (size_t)dim * sizeof(float)));
    if (!g_d_norm_x)       CUDA_CHECK(cudaMalloc(&g_d_norm_x, (size_t)dim * sizeof(float)));
    if (!g_d_router_logits)CUDA_CHECK(cudaMalloc(&g_d_router_logits, 256 * sizeof(float)));
    if (!g_d_shexp_out)    CUDA_CHECK(cudaMalloc(&g_d_shexp_out, (size_t)dim * sizeof(float)));
    if (!g_d_shact)        CUDA_CHECK(cudaMalloc(&g_d_shact, (size_t)sh_ffn * sizeof(float)));
    if (!g_d_shdown)       CUDA_CHECK(cudaMalloc(&g_d_shdown, (size_t)dim * sizeof(float)));
    if (!g_shg)            CUDA_CHECK(cudaMalloc(&g_shg, (size_t)sh_ffn * sizeof(float)));
    if (!g_shu)            CUDA_CHECK(cudaMalloc(&g_shu, (size_t)sh_ffn * sizeof(float)));
    if (!g_shgi)           CUDA_CHECK(cudaMalloc(&g_shgi, (size_t)sh_ffn * sizeof(float)));
    if (!g_d_miss_sum)     CUDA_CHECK(cudaMalloc(&g_d_miss_sum, (size_t)dim * sizeof(float)));
    if (!g_d_top_values)   CUDA_CHECK(cudaMalloc(&g_d_top_values, 8 * sizeof(float)));
    if (!g_d_top_indices)  CUDA_CHECK(cudaMalloc(&g_d_top_indices, 8 * sizeof(int)));
    if (!g_d_hit_pos)      CUDA_CHECK(cudaMalloc(&g_d_hit_pos, 8 * sizeof(int)));
    if (!g_d_q_full)       CUDA_CHECK(cudaMalloc(&g_d_q_full, 8192 * sizeof(float)));
    if (!g_d_k)            CUDA_CHECK(cudaMalloc(&g_d_k, 512 * sizeof(float)));
    if (!g_d_v)            CUDA_CHECK(cudaMalloc(&g_d_v, 512 * sizeof(float)));
    if (!g_d_q)            CUDA_CHECK(cudaMalloc(&g_d_q, 4096 * sizeof(float)));
    if (!g_d_gate)         CUDA_CHECK(cudaMalloc(&g_d_gate, 4096 * sizeof(float)));
    if (!g_d_concat_out)   CUDA_CHECK(cudaMalloc(&g_d_concat_out, 4096 * sizeof(float)));
    if (!g_d_logits)       CUDA_CHECK(cudaMalloc(&g_d_logits, (size_t)vocab_size * sizeof(float)));
    if (!g_d_block_max)    CUDA_CHECK(cudaMalloc(&g_d_block_max, 65536 * sizeof(float)));
    if (!g_d_block_idx)    CUDA_CHECK(cudaMalloc(&g_d_block_idx, 65536 * sizeof(int)));
    if (!g_d_best_token)   CUDA_CHECK(cudaMalloc(&g_d_best_token, sizeof(int)));

    // Upfront delta scratch allocation so stream capture never triggers cudaMalloc
    if (!g_dpqkv)          CUDA_CHECK(cudaMalloc(&g_dpqkv, 9216 * sizeof(float)));
    if (!g_dpz)            CUDA_CHECK(cudaMalloc(&g_dpz,   4096 * sizeof(float)));
    if (!g_dpb)            CUDA_CHECK(cudaMalloc(&g_dpb,   64 * sizeof(float)));
    if (!g_dpa)            CUDA_CHECK(cudaMalloc(&g_dpa,   64 * sizeof(float)));
    if (!g_dq)             CUDA_CHECK(cudaMalloc(&g_dq,    2048 * sizeof(float)));
    if (!g_dk)             CUDA_CHECK(cudaMalloc(&g_dk,    2048 * sizeof(float)));
    if (!g_dv)             CUDA_CHECK(cudaMalloc(&g_dv,    4096 * sizeof(float)));
    if (!g_dp)             CUDA_CHECK(cudaMalloc(&g_dp,    64 * sizeof(float)));
    if (!g_dpgate)         CUDA_CHECK(cudaMalloc(&g_dpgate,32 * sizeof(float)));
    if (!g_dout)           CUDA_CHECK(cudaMalloc(&g_dout,  4096 * sizeof(float)));
    if (!g_dgated)         CUDA_CHECK(cudaMalloc(&g_dgated,4096 * sizeof(float)));

    for (uint32_t i = 0; i < m->block_count; ++i) {
        model_layer_t *l = &m->layers[i];
        upload_dense(&l->attn_norm);
        upload_dense(&l->post_attention_norm);
        upload_dense(&l->ffn_gate_inp);
        upload_dense(&l->ffn_gate_shexp);
        upload_dense(&l->ffn_up_shexp);
        upload_dense(&l->ffn_down_shexp);
        upload_dense(&l->ffn_gate_inp_shexp);
        if (l->type == LAYER_TYPE_ATTENTION) {
            upload_dense(&l->attn_q);
            upload_dense(&l->attn_k);
            upload_dense(&l->attn_v);
            upload_dense(&l->attn_output);
            upload_dense(&l->attn_q_norm);
            upload_dense(&l->attn_k_norm);
        } else {
            upload_dense(&l->attn_qkv);
            upload_dense(&l->attn_gate);
            upload_dense(&l->ssm_out);
            upload_dense(&l->ssm_alpha);
            upload_dense(&l->ssm_beta);
            upload_dense(&l->ssm_norm);   // iter-9: per-head rms_norm weights used by gpu_delta_post
            // P4a: the new gpu_delta_pre kernels read these directly on device.
            upload_dense(&l->ssm_conv1d); // F32 [qkv_dim][conv_kernel]
            upload_dense(&l->ssm_dt_bias);// F32 [n_v_heads]
            upload_dense(&l->ssm_a);      // F32 [n_v_heads]
        }
        if (l->nextn_eh_proj.data) upload_dense(&l->nextn_eh_proj);
        if (l->nextn_hnorm.data) upload_dense(&l->nextn_hnorm);
        if (l->nextn_enorm.data) upload_dense(&l->nextn_enorm);
        if (l->nextn_shared_head_norm.data) upload_dense(&l->nextn_shared_head_norm);

        // H22: re-quantize the F32 router to Q8_0 on the host and upload it alongside the F32
        // copy. The router is 2048 x 256 = 2 MiB/layer; Q8_0 is 557 KiB/layer, and the mmvq Q8_0
        // kernel runs ~316 GB/s vs the f32 kernel's 206 GB/s. The F32 dev copy is kept (the exact
        // path still uses it), so this only adds ~557 KiB x 40 of VRAM when enabled.
        if (router_q8_enabled() && l->ffn_gate_inp.data && l->ffn_gate_inp.info.type == GGML_TYPE_F32) {
            const int in_dim  = (int)l->ffn_gate_inp.info.dims[0]; // 2048
            const int out_dim = (int)l->ffn_gate_inp.info.dims[1]; // 256
            const size_t nblk_per_row = (size_t)(in_dim / QK8_0);
            const size_t q8_bytes = nblk_per_row * out_dim * sizeof(block_q8_0);
            block_q8_0 *hq8 = (block_q8_0*)malloc(q8_bytes);
            if (hq8) {
                const float *src = (const float*)l->ffn_gate_inp.data;
                for (int r = 0; r < out_dim; ++r) {
                    quantize_row_q8_0(src + (size_t)r * in_dim, hq8 + (size_t)r * nblk_per_row, in_dim);
                }
                CUDA_CHECK(cudaMalloc(&l->ffn_gate_inp.dev_q8, q8_bytes));
                CUDA_CHECK(cudaMemcpy(l->ffn_gate_inp.dev_q8, hq8, q8_bytes, cudaMemcpyHostToDevice));
                free(hq8);
            }
        }
    }

    const int max_seq_init = p_max_seq ? *p_max_seq : 4096;
    g_max_seq = max_seq_init;
    const int n_attn = (int)m->n_attn_layers;
    const int n_head_kv = (int)m->head_count_kv;
    const int head_dim = m->key_length ? (int)m->key_length : ((int)m->embedding_length / (int)m->head_count);
    const int n_head = (int)m->head_count;
    kv_cache_ensure(n_attn, n_head_kv, max_seq_init, head_dim);
    attn_scores_ensure(n_head, max_seq_init);

    gpu_expert_cache_init_ex(m, p_max_seq, fill_ram);
    gpu_expert_cache_prefill(m);
}

void gpu_prepare(model_t *m) {
    gpu_prepare_ex(m, NULL, 0);
}

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

void gpu_shutdown(model_t *m) {
    if (!m) return;
    if (m->output_weight.dev) { CUDA_CHECK(cudaFree(m->output_weight.dev)); m->output_weight.dev = NULL; }
    if (m->output_norm.dev)   { CUDA_CHECK(cudaFree(m->output_norm.dev));   m->output_norm.dev = NULL; }
    for (uint32_t i = 0; i < m->block_count; ++i) {
        model_layer_t *l = &m->layers[i];
        if (l->attn_norm.dev)           { CUDA_CHECK(cudaFree(l->attn_norm.dev));           l->attn_norm.dev = NULL; }
        if (l->post_attention_norm.dev) { CUDA_CHECK(cudaFree(l->post_attention_norm.dev)); l->post_attention_norm.dev = NULL; }
        if (l->ffn_gate_inp.dev)        { CUDA_CHECK(cudaFree(l->ffn_gate_inp.dev));        l->ffn_gate_inp.dev = NULL; }
        if (l->ffn_gate_inp.dev_q8)     { CUDA_CHECK(cudaFree(l->ffn_gate_inp.dev_q8));     l->ffn_gate_inp.dev_q8 = NULL; }
        if (l->ffn_gate_shexp.dev)      { CUDA_CHECK(cudaFree(l->ffn_gate_shexp.dev));      l->ffn_gate_shexp.dev = NULL; }
        if (l->ffn_up_shexp.dev)        { CUDA_CHECK(cudaFree(l->ffn_up_shexp.dev));        l->ffn_up_shexp.dev = NULL; }
        if (l->ffn_down_shexp.dev)      { CUDA_CHECK(cudaFree(l->ffn_down_shexp.dev));      l->ffn_down_shexp.dev = NULL; }
        if (l->ffn_gate_inp_shexp.dev)  { CUDA_CHECK(cudaFree(l->ffn_gate_inp_shexp.dev));  l->ffn_gate_inp_shexp.dev = NULL; }
        if (l->type == LAYER_TYPE_ATTENTION) {
            if (l->attn_q.dev)      { CUDA_CHECK(cudaFree(l->attn_q.dev));      l->attn_q.dev = NULL; }
            if (l->attn_k.dev)      { CUDA_CHECK(cudaFree(l->attn_k.dev));      l->attn_k.dev = NULL; }
            if (l->attn_v.dev)      { CUDA_CHECK(cudaFree(l->attn_v.dev));      l->attn_v.dev = NULL; }
            if (l->attn_output.dev) { CUDA_CHECK(cudaFree(l->attn_output.dev)); l->attn_output.dev = NULL; }
            if (l->attn_q_norm.dev) { CUDA_CHECK(cudaFree(l->attn_q_norm.dev)); l->attn_q_norm.dev = NULL; }
            if (l->attn_k_norm.dev) { CUDA_CHECK(cudaFree(l->attn_k_norm.dev)); l->attn_k_norm.dev = NULL; }
        } else {
            if (l->attn_qkv.dev)   { CUDA_CHECK(cudaFree(l->attn_qkv.dev));   l->attn_qkv.dev = NULL; }
            if (l->attn_gate.dev)  { CUDA_CHECK(cudaFree(l->attn_gate.dev));  l->attn_gate.dev = NULL; }
            if (l->ssm_out.dev)    { CUDA_CHECK(cudaFree(l->ssm_out.dev));    l->ssm_out.dev = NULL; }
            if (l->ssm_alpha.dev)  { CUDA_CHECK(cudaFree(l->ssm_alpha.dev));  l->ssm_alpha.dev = NULL; }
            if (l->ssm_beta.dev)   { CUDA_CHECK(cudaFree(l->ssm_beta.dev));   l->ssm_beta.dev = NULL; }
            if (l->ssm_norm.dev)   { CUDA_CHECK(cudaFree(l->ssm_norm.dev));   l->ssm_norm.dev = NULL; }
            if (l->ssm_conv1d.dev) { CUDA_CHECK(cudaFree(l->ssm_conv1d.dev)); l->ssm_conv1d.dev = NULL; }
            if (l->ssm_dt_bias.dev){ CUDA_CHECK(cudaFree(l->ssm_dt_bias.dev));l->ssm_dt_bias.dev = NULL; }
            if (l->ssm_a.dev)      { CUDA_CHECK(cudaFree(l->ssm_a.dev));      l->ssm_a.dev = NULL; }
        }
    }
    if (g_dx) { CUDA_CHECK(cudaFree(g_dx)); g_dx = NULL; }
    if (g_dy) { CUDA_CHECK(cudaFree(g_dy)); g_dy = NULL; }
    if (g_dstate_all)  { CUDA_CHECK(cudaFree(g_dstate_all));  g_dstate_all = NULL;  g_dstate_bytes = 0; }
    if (g_dstate_snap) { CUDA_CHECK(cudaFree(g_dstate_snap)); g_dstate_snap = NULL; }
    if (g_dconv_all)   { CUDA_CHECK(cudaFree(g_dconv_all));   g_dconv_all = NULL;   g_dconv_bytes = 0; }
    if (g_dconv_snap)  { CUDA_CHECK(cudaFree(g_dconv_snap));  g_dconv_snap = NULL; }
    if (g_dpgate)     { CUDA_CHECK(cudaFree(g_dpgate));     g_dpgate = NULL; }
    if (g_dq)     { CUDA_CHECK(cudaFree(g_dq));     g_dq = NULL; }
    if (g_dk)     { CUDA_CHECK(cudaFree(g_dk));     g_dk = NULL; }
    if (g_dv)     { CUDA_CHECK(cudaFree(g_dv));     g_dv = NULL; }
    if (g_dout)   { CUDA_CHECK(cudaFree(g_dout));   g_dout = NULL; }
    if (g_dp)     { CUDA_CHECK(cudaFree(g_dp));     g_dp = NULL; }
    if (g_dpqkv)  { CUDA_CHECK(cudaFree(g_dpqkv));  g_dpqkv = NULL; }
    if (g_dpz)    { CUDA_CHECK(cudaFree(g_dpz));    g_dpz = NULL; }
    if (g_dpb)    { CUDA_CHECK(cudaFree(g_dpb));    g_dpb = NULL; }
    if (g_dpa)    { CUDA_CHECK(cudaFree(g_dpa));    g_dpa = NULL; }
    if (g_dgated) { CUDA_CHECK(cudaFree(g_dgated)); g_dgated = NULL; }
    if (g_shg)    { CUDA_CHECK(cudaFree(g_shg));    g_shg = NULL; }
    if (g_shu)    { CUDA_CHECK(cudaFree(g_shu));    g_shu = NULL; }
    if (g_shgi)   { CUDA_CHECK(cudaFree(g_shgi));   g_shgi = NULL; }
    if (g_shg2)   { CUDA_CHECK(cudaFree(g_shg2));   g_shg2 = NULL; }
    if (g_shu2)   { CUDA_CHECK(cudaFree(g_shu2));   g_shu2 = NULL; }
    if (g_shgi2)  { CUDA_CHECK(cudaFree(g_shgi2));  g_shgi2 = NULL; }
    if (g_d_shact2){ CUDA_CHECK(cudaFree(g_d_shact2)); g_d_shact2 = NULL; }
    if (g_d_shdown2){ CUDA_CHECK(cudaFree(g_d_shdown2)); g_d_shdown2 = NULL; }
    if (g_dk_cache){ CUDA_CHECK(cudaFree(g_dk_cache)); g_dk_cache = NULL; }
    if (g_dv_cache){ CUDA_CHECK(cudaFree(g_dv_cache)); g_dv_cache = NULL; }
    g_kv_cache_bytes = 0;
    if (g_d_attn_scores){ CUDA_CHECK(cudaFree(g_d_attn_scores)); g_d_attn_scores = NULL; }
    g_attn_scores_floats = 0;
    if (g_aq)     { CUDA_CHECK(cudaFree(g_aq));     g_aq = NULL; }
    if (g_ak)     { CUDA_CHECK(cudaFree(g_ak));     g_ak = NULL; }
    if (g_av)     { CUDA_CHECK(cudaFree(g_av));     g_av = NULL; }
    if (g_aconc)  { CUDA_CHECK(cudaFree(g_aconc));  g_aconc = NULL; }

    if (g_d_res_x)        { CUDA_CHECK(cudaFree(g_d_res_x));        g_d_res_x = NULL; }
    if (g_d_norm_x)       { CUDA_CHECK(cudaFree(g_d_norm_x));       g_d_norm_x = NULL; }
    if (g_d_res_x2)       { CUDA_CHECK(cudaFree(g_d_res_x2));       g_d_res_x2 = NULL; }
    if (g_d_norm_x2)      { CUDA_CHECK(cudaFree(g_d_norm_x2));      g_d_norm_x2 = NULL; }
    if (g_d_nx2_q8)       { CUDA_CHECK(cudaFree(g_d_nx2_q8));       g_d_nx2_q8 = NULL; }
    if (g_d_p2_qkv)       { CUDA_CHECK(cudaFree(g_d_p2_qkv));       g_d_p2_qkv = NULL; }
    if (g_d_p2_z)         { CUDA_CHECK(cudaFree(g_d_p2_z));         g_d_p2_z = NULL; }
    if (g_d_p2_b)         { CUDA_CHECK(cudaFree(g_d_p2_b));         g_d_p2_b = NULL; }
    if (g_d_p2_a)         { CUDA_CHECK(cudaFree(g_d_p2_a));         g_d_p2_a = NULL; }
    if (g_d_logits2)      { CUDA_CHECK(cudaFree(g_d_logits2));      g_d_logits2 = NULL; }
    if (g_d_shexp_out2)   { CUDA_CHECK(cudaFree(g_d_shexp_out2));   g_d_shexp_out2 = NULL; }
    if (g_d_router_logits2){ CUDA_CHECK(cudaFree(g_d_router_logits2)); g_d_router_logits2 = NULL; }
    if (g_dstate_snap)    { CUDA_CHECK(cudaFree(g_dstate_snap));    g_dstate_snap = NULL; }
    if (g_dconv_snap)     { CUDA_CHECK(cudaFree(g_dconv_snap));     g_dconv_snap = NULL; }
    g_n2_dim = 0;
    if (g_d_router_logits){ CUDA_CHECK(cudaFree(g_d_router_logits));g_d_router_logits = NULL; }
    if (g_d_shexp_out)    { CUDA_CHECK(cudaFree(g_d_shexp_out));    g_d_shexp_out = NULL; }
    if (g_d_shact)        { CUDA_CHECK(cudaFree(g_d_shact));        g_d_shact = NULL; }
    if (g_d_shdown)       { CUDA_CHECK(cudaFree(g_d_shdown));       g_d_shdown = NULL; }
    if (g_d_miss_sum)     { CUDA_CHECK(cudaFree(g_d_miss_sum));     g_d_miss_sum = NULL; }
    if (g_d_top_values)   { CUDA_CHECK(cudaFree(g_d_top_values));   g_d_top_values = NULL; }
    if (g_d_top_indices)  { CUDA_CHECK(cudaFree(g_d_top_indices));  g_d_top_indices = NULL; }
    if (g_d_hit_pos)      { CUDA_CHECK(cudaFree(g_d_hit_pos));      g_d_hit_pos = NULL; }
    if (g_d_q_full)       { CUDA_CHECK(cudaFree(g_d_q_full));       g_d_q_full = NULL; }
    if (g_d_k)            { CUDA_CHECK(cudaFree(g_d_k));            g_d_k = NULL; }
    if (g_d_v)            { CUDA_CHECK(cudaFree(g_d_v));            g_d_v = NULL; }
    if (g_d_q)            { CUDA_CHECK(cudaFree(g_d_q));            g_d_q = NULL; }
    if (g_d_gate)         { CUDA_CHECK(cudaFree(g_d_gate));         g_d_gate = NULL; }
    if (g_d_concat_out)   { CUDA_CHECK(cudaFree(g_d_concat_out));   g_d_concat_out = NULL; }
    if (g_d_logits)       { CUDA_CHECK(cudaFree(g_d_logits));       g_d_logits = NULL; }
    if (g_d_block_max)    { CUDA_CHECK(cudaFree(g_d_block_max));    g_d_block_max = NULL; }
    if (g_d_block_idx)    { CUDA_CHECK(cudaFree(g_d_block_idx));    g_d_block_idx = NULL; }
    if (g_d_best_token)   { CUDA_CHECK(cudaFree(g_d_best_token));   g_d_best_token = NULL; }
    if (g_d_out_normx_q8) { CUDA_CHECK(cudaFree(g_d_out_normx_q8)); g_d_out_normx_q8 = NULL; }
    if (g_d_dense_x_q8)   { CUDA_CHECK(cudaFree(g_d_dense_x_q8));   g_d_dense_x_q8 = NULL; }
    if (g_d_norm_q8)      { CUDA_CHECK(cudaFree(g_d_norm_q8));      g_d_norm_q8 = NULL; }
    if (g_d_shact_q8)     { CUDA_CHECK(cudaFree(g_d_shact_q8));     g_d_shact_q8 = NULL; }

    gpu_expert_cache_free();
}

void gpu_mat_vec(float *dst, const tensor_t *w, const float *src) {
    const int in_dim  = (int)w->info.dims[0];
    const int out_dim = (int)w->info.dims[1];

    if (!w->dev) { // not uploaded -> CPU path (experts, token_embd, output, norms)
        mat_vec_cpu(dst, w, src);
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(g_dx, src, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    switch ((int)w->info.type) {
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>(static_cast<const block_q8_0 *>(w->dev), g_dx, g_dy, in_dim);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_kernel<<<out_dim, 256>>>(static_cast<const block_q6_K *>(w->dev), g_dx, g_dy, in_dim);
            break;
        case GGML_TYPE_F32:
            mul_mat_vec_f32_kernel<<<out_dim, 256>>>(static_cast<const float *>(w->dev), g_dx, g_dy, in_dim);
            break;
        default:
            fprintf(stderr, "[FATAL CUDA ERROR] gpu_mat_vec: unsupported type %d (in_dim=%d out_dim=%d)\n",
                    (int)w->info.type, in_dim, out_dim);
            exit(EXIT_FAILURE);
    }
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(dst, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-9: run the four delta-net projections (attn_qkv Q8_0 2048->8192, attn_gate Q8_0
// 2048->4096, ssm_beta F32 2048->32, ssm_alpha F32 2048->32) with a single upload of the shared
// input norm_x and back-to-back launches, then download all four results. Each output element is
// still produced by the exact same kernel / accumulation order as gpu_mat_vec, so the result is
// bit-identical; we only remove the 3 redundant norm_x uploads and their serialized syncs.
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
    mul_mat_vec_q8_0_kernel<<<out_qkv, 256>>>((const block_q8_0*)attn_qkv->dev, g_dx, g_dpqkv, in_dim);
    mul_mat_vec_q8_0_kernel<<<out_z,   256>>>((const block_q8_0*)attn_gate->dev, g_dx, g_dpz,   in_dim);
    mul_mat_vec_f32_kernel <<<out_beta,256>>>((const float*)ssm_beta->dev,       g_dx, g_dpb,   in_dim);
    mul_mat_vec_f32_kernel <<<out_alpha,256>>>((const float*)ssm_alpha->dev,     g_dx, g_dpa,   in_dim);
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
void gpu_expert_mat_vec(float *dst, const void *dev_w, const float *src, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_dx, src, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    mul_mat_vec_q4_K_kernel<<<out_dim, 1>>>((const block_q4_K*)dev_w, g_dx, g_dy, in_dim);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(dst, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Q6_K expert mat-vec (the routed down experts are Q6_K). Single thread per output row,
// strict scalar sequential accumulation matching the CPU mat_vec_q6_K_scalar() bit-for-bit
// (both use the same dequant value `d*s*q` and `sum += yv*x[c]` fma contraction).
__global__ void mul_mat_vec_q6_K_scalar_kernel(const block_q6_K * __restrict__ w,
                                               const float * __restrict__ x,
                                               float * __restrict__ y,
                                               int in_dim) {
    const int row = blockIdx.x;
    const int nblk = in_dim / QK_K;
    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;

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
        // __fmul_rn + __fadd_rn force NO fma, matching the CPU dequant+dot reference exactly.
        sum = __fadd_rn(sum, __fmul_rn(yv, x[c]));
    }
    y[row] = sum;
}

// Q6_K expert mat-vec (the routed down experts are Q6_K). Reuses the existing Q6_K kernel
// (row-parallel, 256 threads), which already matches the CPU oracle for the Q6_K output
// projection. in_dim x out_dim block_q6_K matrix in dev_w.
void gpu_expert_mat_vec_q6(float *dst, const void *dev_w, const float *src, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_dx, src, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1>>>((const block_q6_K*)dev_w, g_dx, g_dy, in_dim);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(dst, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Pipelined gate+up launch on the dedicated expert stream. Uploads norm_x once, then runs the
// Q4_K gate and up kernels (both dim x ffn) writing gate->g_exgy and up->g_exgy2. Returns
// immediately; call gpu_expert_gateup_wait_download() later to sync + fetch both. The caller
// must ensure the previous launch has been downloaded (so g_exgx/g_exgy/g_exgy2 are free).
void gpu_expert_gateup_launch(int slot, const float *norm_x, int in_dim, int out_dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_exgx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q4_K*)gpu_expert_dev_ptr(slot, 0), g_exgx, g_exgy, in_dim);
    mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q4_K*)gpu_expert_dev_ptr(slot, 1), g_exgx, g_exgy2, in_dim);
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
        mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1, 0, g_expert_stream>>>(
            (const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, g_exdy, in_dim);
    else
        mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>(
            (const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, g_exdy, in_dim);
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
        mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>(
            (const block_q4_K*)gpu_expert_dev_ptr(slots[k], 0), g_exgx, g_exgbat + (size_t)k * out_dim, in_dim);
        mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>(
            (const block_q4_K*)gpu_expert_dev_ptr(slots[k], 1), g_exgx, g_exubbat + (size_t)k * out_dim, in_dim);
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
        mul_mat_vec_q6_K_scalar_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q6_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim);
    else
        mul_mat_vec_q4_K_kernel<<<out_dim, 1, 0, g_expert_stream>>>((const block_q4_K*)gpu_expert_dev_ptr(slot, 2), g_exdx, dst, in_dim);
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

#define QK8_1 32
#define QI8_1 8                  // 32 int8 quants = 8 ints
#define QI4_K 32                 // QK_K/(4*QR4_K)
#define QR4_K 2
#define QI6_K 32                 // QK_K/(4*QR6_K)
#define QR6_K 2
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1

// block_q8_1 (ggml-common.h:258). ds.x = d (scale), ds.y = sum of the original f32 values.
typedef struct {
    __half2 ds;
    int8_t  qs[QK8_1];
} block_q8_1_c;
GGML_STATIC_ASSERT(sizeof(block_q8_1_c) == 36, "wrong q8_1 block size/padding");

static __device__ __forceinline__ int get_int_b2_c(const void *x, int i32) {
    const uint16_t *x16 = (const uint16_t *)x; // q6_K blocks are only 2-byte aligned
    return (int)((uint32_t)x16[2*i32 + 0] | ((uint32_t)x16[2*i32 + 1] << 16));
}

static __device__ __forceinline__ int get_int_b4_c(const void *x, int i32) {
    return ((const int *)x)[i32];
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

// Quantize a contiguous f32 vector to q8_1 (one thread per element; each 32-lane warp covers
// exactly one block). Ported from quantize.cu's quantize_q8_1. n must be a multiple of QK8_1
// and blockDim.x a multiple of 32.
__global__ void quantize_q8_1_kernel(const float * __restrict__ x, block_q8_1_c * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // No early return: every lane must reach the shuffles below, so out-of-range lanes
    // contribute a neutral 0 and simply skip the store.
    const float xi = i < n ? x[i] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    if (i >= n) return;
    y[i / QK8_1].qs[i % QK8_1] = q;
    if (i % QK8_1 == 0) y[i / QK8_1].ds = make_half2(__float2half(d), __float2half(sum));
}

// vec_dot_q4_K_q8_1 (vecdotq.cuh:910) + vec_dot_q4_K_q8_1_impl_vmmq (vecdotq.cuh:508).
// bq4_K is the super-block, bq8_1 the aligned start of this super-block's 8 q8_1 blocks.
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_c(
        const block_q4_K * __restrict__ bq4_K, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int   v[2];
    int   u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int *q4 = (const int *)(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t *scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1_c *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int *q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0));
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    return h2f(bq4_K->d)*sumf_d - h2f(bq4_K->dmin)*sumf_m;
}

// vec_dot_q6_K_q8_1 (vecdotq.cuh:1002) + vec_dot_q6_K_q8_1_impl_mmvq (vecdotq.cuh:627).
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_c(
        const block_q6_K * __restrict__ bq6_K, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    const int bq8_offset   = 2*QR6_K*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4)*(iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2*((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2_c(bq6_K->ql, iqs);
    const int vh = get_int_b2_c(bq6_K->qh, (QI6_K/4)*(iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t *scales = bq6_K->scales + scale_offset;

    int   u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4_c(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return h2f(bq6_K->d) * sumf;
}

// H9: vec_dot_q8_0_q8_1 (vecdotq.cuh) for the DENSE Q8_0 projections — qkv, attn_gate, ssm_out,
// attn_q/k/v, attn_output and the shared expert. The scalar kernels these replace spend one fp32
// multiply per weight byte and reach 206–239 GB/s; the __dp4a form does four weights per
// instruction and measured 352 GB/s on the Q6_K output head (iteration 38).
#define QI8_0 8                  // 32 int8 quants = 8 ints per block
#define VDR_Q8_0_Q8_1_MMVQ 2     // ints per lane-iteration

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_c(
        const block_q8_0 * __restrict__ bq8_0, const block_q8_1_c * __restrict__ bq8_1, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        // block_q8_0 is 34 B, so qs is only 2-byte aligned — same reason q6_K uses get_int_b2_c.
        const int v = get_int_b2_c(bq8_0->qs, iqs + i);
        const int u = get_int_b4_c(bq8_1->qs, iqs + i);
        sumi = __dp4a(v, u, sumi);
    }
    return h2f(bq8_0->d) * __low2float(bq8_1->ds) * (float)sumi;
}

#ifndef MMVQ_ROWS_PER_BLOCK
#define MMVQ_ROWS_PER_BLOCK 4
#endif

// QK8_0 == QK8_1 == 32, so weight block k pairs with activation block k.
template <int rows_per_block, bool accum>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_warprow_kernel(
        const block_q8_0 * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;                 // 4 lanes cover one block
    constexpr int blocks_per_iter = vdr * 32 / QI8_0; // 8 blocks per warp iteration

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q8_0_q8_1_c(rw + kbx, y + kbx, kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        if (accum) dst[row] += tmp;
        else       dst[row]  = tmp;
    }
}

// H7 stage 1: the same kernel over NCOLS activation vectors, reading each weight byte once.
// This is the whole premise of speculative decoding on this engine — if the second token is not
// close to free here, the trunk cannot verify γ+1 tokens for the price of one weight stream and
// H7 is dead before the expensive parts. Measured by --bench-dense-batch.
template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_ncols_kernel(
        const block_q8_0 * __restrict__ w,
        const block_q8_1_c * __restrict__ y,   // ncols activation rows, nblk blocks each
        float * __restrict__ dst,              // ncols output rows, out_dim each
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q8_0 * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) tmp[c] = 0.0f;

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_0 *bw = rw + kbx;      // one weight block, reused by every column
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            tmp[c] += vec_dot_q8_0_q8_1_c(bw, y + (size_t)c * nblk + kbx, kqs);
        }
    }
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float s = warp_reduce_sum_f(tmp[c]);
        if (threadIdx.x == 0) dst[(size_t)c * out_dim + row] = s;
    }
}

static block_q8_1_c *dense_quant_buf_ensure(int in_dim) {
    (void)in_dim;
    if (!g_d_dense_x_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_dense_x_q8, (size_t)(8192 / QK8_1) * sizeof(block_q8_1_c)));
    }
    return (block_q8_1_c *)g_d_dense_x_q8;
}

static block_q8_1_c *norm_x_q8_ensure(int in_dim) {
    (void)in_dim;
    if (!g_d_norm_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_norm_q8, (size_t)(8192 / QK8_1) * sizeof(block_q8_1_c)));
    }
    return (block_q8_1_c *)g_d_norm_q8;
}

extern "C" void *gpu_get_d_dense_x_q8(void) {
    return (void*)dense_quant_buf_ensure(2048);
}

extern "C" void *gpu_get_d_norm_q8(void) {
    return (void*)norm_x_q8_ensure(2048);
}

extern "C" void gpu_set_norm_q8_valid(int v) {
    g_norm_q8_valid = v;
}

static block_q8_1_c *dense_quantize_x(const float *x, int in_dim) {
    if (!g_d_dense_x_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_dense_x_q8, (size_t)(8192 / QK8_1) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq = (block_q8_1_c *)g_d_dense_x_q8;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(x, xq, in_dim);
    CUDA_CHECK_LAST();
    return xq;
}

// H11: quantize x into a caller-provided q8_1 buffer (used for the shared-expert norm_x, which
// the routed experts then reuse instead of re-quantizing). Same kernel, same layout as
// dense_quantize_x, only the destination differs.
static block_q8_1_c *dense_quantize_x_to(const float *x, int in_dim, void *dst) {
    block_q8_1_c *xq = (block_q8_1_c *)dst;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(x, xq, in_dim);
    CUDA_CHECK_LAST();
    return xq;
}

static void mmvq_q8_0_rows(const block_q8_0 *w, const block_q8_1_c *xq, float *y,
                           int in_dim, int out_dim, bool accum) {
    const int nblk = in_dim / QK8_0;
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    if (accum) mmvq_q8_0_dense_warprow_kernel<MMVQ_ROWS_PER_BLOCK, true ><<<grid, block>>>(w, xq, y, out_dim, nblk);
    else       mmvq_q8_0_dense_warprow_kernel<MMVQ_ROWS_PER_BLOCK, false><<<grid, block>>>(w, xq, y, out_dim, nblk);
    CUDA_CHECK_LAST();
}

void gpu_mmvq_q8_0_dense(const block_q8_0 *w, const float *x, float *y,
                         int in_dim, int out_dim, int accum) {
    mmvq_q8_0_rows(w, dense_quantize_x(x, in_dim), y, in_dim, out_dim, accum != 0);
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_dual_warprow_kernel(
        const block_q8_0 * __restrict__ w0,
        float * __restrict__ y0,
        int out0,
        const block_q8_0 * __restrict__ w1,
        float * __restrict__ y1,
        int out1,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    if (global_row >= out0 + out1) return;

    const bool is_w0 = global_row < out0;
    const int row = is_w0 ? global_row : (global_row - out0);
    const block_q8_0 * __restrict__ rw = is_w0 ? (w0 + (size_t)row * nblk) : (w1 + (size_t)row * nblk);
    float * __restrict__ dst = is_w0 ? y0 : y1;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q8_0_q8_1_c(rw + kbx, xq + kbx, kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 4)
__global__ void mmvq_q8_0_dense_delta_quad_warprow_kernel(
        const block_q8_0 * __restrict__ w_qkv,
        float * __restrict__ y_qkv,
        int out_qkv,
        const block_q8_0 * __restrict__ w_z,
        float * __restrict__ y_z,
        int out_z,
        const float * __restrict__ w_beta,
        float * __restrict__ y_beta,
        int out_beta,
        const float * __restrict__ w_alpha,
        float * __restrict__ y_alpha,
        int out_alpha,
        const block_q8_1_c * __restrict__ xq,
        const float * __restrict__ x_f32,
        int nblk,
        int in_dim) {
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
        float * __restrict__ dst = is_qkv ? y_qkv : y_z;

        const int kqs = vdr * (threadIdx.x % tpsb);
        float tmp = 0.0f;
        #pragma unroll 4
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            tmp += vec_dot_q8_0_q8_1_c(rw + kbx, xq + kbx, kqs);
        }
        tmp = warp_reduce_sum_f(tmp);
        if (threadIdx.x == 0) dst[row] = tmp;
    } else {
        const int rem = global_row - (out_qkv + out_z);
        const bool is_beta = (rem < out_beta);
        const int row = is_beta ? rem : (rem - out_beta);
        const float * __restrict__ rw = is_beta ? (w_beta + (size_t)row * in_dim) : (w_alpha + (size_t)row * in_dim);
        float * __restrict__ dst = is_beta ? y_beta : y_alpha;

        const float4 * __restrict__ rw4 = reinterpret_cast<const float4*>(rw);
        const float4 * __restrict__ x4  = reinterpret_cast<const float4*>(x_f32);
        const int n_f4 = in_dim / 4;

        float tmp = 0.0f;
        #pragma unroll 4
        for (int i = threadIdx.x; i < n_f4; i += 32) {
            float4 w_val = rw4[i];
            float4 x_val = x4[i];
            tmp += w_val.x * x_val.x + w_val.y * x_val.y + w_val.z * x_val.z + w_val.w * x_val.w;
        }
        tmp = warp_reduce_sum_f(tmp);
        if (threadIdx.x == 0) dst[row] = tmp;
    }
}

static int tuff_h49(void) {
    const char *e = getenv("TUFF_H49");
    return (e && e[0] == '1') ? 1 : 0;
}

static int tuff_h50(void) {
    const char *e = getenv("TUFF_H50");
    return (e && e[0] == '0') ? 0 : 1;
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_triple_warprow_kernel(
        const block_q8_0 * __restrict__ w0,
        float * __restrict__ y0,
        int out0,
        const block_q8_0 * __restrict__ w1,
        float * __restrict__ y1,
        int out1,
        const block_q8_0 * __restrict__ w2,
        float * __restrict__ y2,
        int out2,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int global_row = blockIdx.x * rows_per_block + threadIdx.y;
    if (global_row >= out0 + out1 + out2) return;

    const block_q8_0 * __restrict__ rw;
    float * __restrict__ dst;
    int row;
    if (global_row < out0) {
        row = global_row;
        rw = w0 + (size_t)row * nblk;
        dst = y0;
    } else if (global_row < out0 + out1) {
        row = global_row - out0;
        rw = w1 + (size_t)row * nblk;
        dst = y1;
    } else {
        row = global_row - (out0 + out1);
        rw = w2 + (size_t)row * nblk;
        dst = y2;
    }

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q8_0_q8_1_c(rw + kbx, xq + kbx, kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

// Shared expert fused gate+up+swiglu kernel: calculates both projections and SwiGLU in one pass
template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q8_0_dense_shexp_swiglu_kernel(
        const block_q8_0 * __restrict__ w_gate,
        const block_q8_0 * __restrict__ w_up,
        float * __restrict__ dst_act,
        int sh_ffn_dim,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= sh_ffn_dim) return;

    const block_q8_0 * __restrict__ rw_g = w_gate + (size_t)row * nblk;
    const block_q8_0 * __restrict__ rw_u = w_up   + (size_t)row * nblk;

    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp_g = 0.0f;
    float tmp_u = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q8_1_c *bx = xq + kbx;
        tmp_g += vec_dot_q8_0_q8_1_c(rw_g + kbx, bx, kqs);
        tmp_u += vec_dot_q8_0_q8_1_c(rw_u + kbx, bx, kqs);
    }
    tmp_g = warp_reduce_sum_f(tmp_g);
    tmp_u = warp_reduce_sum_f(tmp_u);
    if (threadIdx.x == 0) {
        float silu_g = tmp_g / (1.0f + expf(-tmp_g));
        dst_act[row] = silu_g * tmp_u;
    }
}

// Shared expert fused gate+up+swiglu+quantize kernel: calculates projections, SwiGLU, and Q8_1 quantization directly in registers (H50)
__launch_bounds__(1024, 1)
__global__ void mmvq_q8_0_dense_shexp_swiglu_quant_kernel(
        const block_q8_0 * __restrict__ w_gate,
        const block_q8_0 * __restrict__ w_up,
        block_q8_1_c * __restrict__ dst_q8,
        int sh_ffn_dim,
        const block_q8_1_c * __restrict__ xq,
        int nblk) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int tpsb = QI8_0 / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI8_0;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int row = block_idx * 32 + warp_id;

    __shared__ float s_act[32];

    if (row < sh_ffn_dim) {
        const block_q8_0 * __restrict__ rw_g = w_gate + (size_t)row * nblk;
        const block_q8_0 * __restrict__ rw_u = w_up   + (size_t)row * nblk;

        const int kqs = vdr * (lane_id % tpsb);
        float tmp_g = 0.0f;
        float tmp_u = 0.0f;
        for (int kbx = lane_id / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            const block_q8_1_c *bx = xq + kbx;
            tmp_g += vec_dot_q8_0_q8_1_c(rw_g + kbx, bx, kqs);
            tmp_u += vec_dot_q8_0_q8_1_c(rw_u + kbx, bx, kqs);
        }
        tmp_g = warp_reduce_sum_f(tmp_g);
        tmp_u = warp_reduce_sum_f(tmp_u);
        if (lane_id == 0) {
            float silu_g = tmp_g / (1.0f + expf(-tmp_g));
            s_act[warp_id] = silu_g * tmp_u;
        }
    } else {
        if (lane_id == 0) s_act[warp_id] = 0.0f;
    }
    __syncthreads();

    if (warp_id == 0) {
        float act_val = s_act[lane_id];
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

// Two matrices sharing one activation: quantize once, single unified grid launch.
void gpu_mmvq_q8_0_dense_dual(const block_q8_0 *w0, float *y0, int out0,
                              const block_q8_0 *w1, float *y1, int out1,
                              const float *x, int in_dim) {
    const block_q8_1_c *xq = dense_quantize_x(x, in_dim);
    const int nblk = in_dim / QK8_0;
    const int total_out = out0 + out1;
    dim3 grid((unsigned)((total_out + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_dual_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block>>>(w0, y0, out0, w1, y1, out1, xq, nblk);
    CUDA_CHECK_LAST();
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q6_K_dense_warprow_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
    }
    tmp = warp_reduce_sum_f(tmp);
    if (threadIdx.x == 0) dst[row] = tmp;
}

template <int rows_per_block>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q6_K_dense_argmax_warprow_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,
        float * __restrict__ dst,
        float * __restrict__ block_max,
        int * __restrict__ block_idx,
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    const bool valid = (row < out_dim);

    const block_q6_K * __restrict__ rw = valid ? (w + (size_t)row * nblk) : w;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp = 0.0f;
    if (valid) {
        for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
            tmp += vec_dot_q6_K_q8_1_c(rw + kbx, y + kbx * (QK_K / QK8_1), kqs);
        }
    }
    tmp = warp_reduce_sum_f(tmp);

    __shared__ float s_max[rows_per_block];
    __shared__ int   s_idx[rows_per_block];

    if (threadIdx.x == 0) {
        if (valid && dst) dst[row] = tmp;
        s_max[threadIdx.y] = valid ? tmp : -1e30f;
        s_idx[threadIdx.y] = valid ? row : 0;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float b_max = s_max[0];
        int   b_idx = s_idx[0];
        #pragma unroll
        for (int r = 1; r < rows_per_block; ++r) {
            if (s_max[r] > b_max) {
                b_max = s_max[r];
                b_idx = s_idx[r];
            }
        }
        block_max[blockIdx.x] = b_max;
        block_idx[blockIdx.x] = b_idx;
    }
}

// H7 stage 2a: the output head over `ncols` activation columns. 398 MB is the single largest
// read in the step, and both a draft and a verify need logits for more than one position; read
// once, score every column. Same argument as the Q8_0 form — see --bench-dense-batch.
template <int rows_per_block, int ncols>
__launch_bounds__(rows_per_block * 32, 1)
__global__ void mmvq_q6_K_dense_ncols_kernel(
        const block_q6_K * __restrict__ w,
        const block_q8_1_c * __restrict__ y,   // ncols activation rows, nblk*(QK_K/QK8_1) blocks each
        float * __restrict__ dst,              // ncols rows of out_dim
        int out_dim, int nblk) {
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int tpsb = QI6_K / vdr;
    constexpr int blocks_per_iter = vdr * 32 / QI6_K;
    constexpr int y_per_blk = QK_K / QK8_1;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= out_dim) return;

    const block_q6_K * __restrict__ rw = w + (size_t)row * nblk;
    const int kqs = vdr * (threadIdx.x % tpsb);
    float tmp[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) tmp[c] = 0.0f;

    for (int kbx = threadIdx.x / tpsb; kbx < nblk; kbx += blocks_per_iter) {
        const block_q6_K *bw = rw + kbx;
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            tmp[c] += vec_dot_q6_K_q8_1_c(bw, y + (size_t)c * nblk * y_per_blk + kbx * y_per_blk, kqs);
        }
    }
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float s = warp_reduce_sum_f(tmp[c]);
        if (threadIdx.x == 0) dst[(size_t)c * out_dim + row] = s;
    }
}

static void launch_mat_vec_q6_K(const block_q6_K *w, const float *x, float *y, int in_dim, int out_dim) {
    if (tuff_exact()) {
        dim3 block(32, 16);
        dim3 grid((out_dim + 15) / 16);
        size_t smem = (size_t)in_dim * sizeof(float);
        mul_mat_vec_q6_K_fast_kernel<<<grid, block, smem>>>(w, x, y, in_dim, out_dim);
        CUDA_CHECK_LAST();
        return;
    }

    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / 32) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(x, xq8, in_dim);
    CUDA_CHECK_LAST();

    const int nblk = in_dim / QK_K;
    dim3 grid_dim((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block_dim(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q6_K_dense_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid_dim, block_dim>>>(w, xq8, y, out_dim, nblk);
    CUDA_CHECK_LAST();
}

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
    CUDA_CHECK(cudaMemcpyAsync(g_exgx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice, g_expert_stream));
    CUDA_CHECK(cudaMemcpyAsync(g_exslots, slots, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, g_expert_stream));

    block_q8_1_c *xq8 = (block_q8_1_c *)g_exgx_q8;
    quantize_q8_1_kernel<<<(in_dim + 255)/256, 256, 0, g_expert_stream>>>(g_exgx, xq8, in_dim);
    CUDA_CHECK_LAST();

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
    quantize_q8_1_kernel<<<(nq + 255)/256, 256, 0, g_expert_stream>>>(g_exdx, (block_q8_1_c *)g_exdx_q8, nq);
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
        quantize_q8_1_kernel<<<(in_dim + 255)/256, 256, 0, g_expert_stream>>>(g_exgx, xq8, in_dim);
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
        quantize_q8_1_kernel<<<(nq + 255)/256, 256, 0, g_expert_stream>>>(g_exdx, aq8, nq);
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
__global__ void delta_conv_kernel(
        float * __restrict__ state,        // [conv_kernel][qkv_dim] this layer, in-place shift
        const float * __restrict__ x,      // g_dpqkv (qkv_mixed) [qkv_dim]
        const float * __restrict__ w,      // conv weights [qkv_dim][conv_kernel]
        float * __restrict__ out,          // qkv_conv (overwrites g_dpqkv) [qkv_dim]
        int qkv_dim, int conv_kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) return;
    float s[4];
#pragma unroll
    for (int k = 0; k < conv_kernel; ++k) s[k] = state[(size_t)k * qkv_dim + c];
    const float xin = x[c];
#pragma unroll
    for (int k = 0; k < conv_kernel - 1; ++k) state[(size_t)k * qkv_dim + c] = s[k + 1];
    state[(size_t)(conv_kernel - 1) * qkv_dim + c] = xin;

    // Conv over the SHIFTED state (matches the host loop exactly, k in 0..conv_kernel-1).
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
__launch_bounds__(128, 4)
__global__ void fused_gated_delta_net_ar_kernel(
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
        row4[j] = r;
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
void gpu_delta_fused_gdn_dev_p(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                               float *qkv, const float *z, const float *beta_raw, const float *alpha_raw,
                               int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                               int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    if (!g_dgated) CUDA_CHECK(cudaMalloc(&g_dgated, (size_t)value_dim * sizeof(float)));
    if (!g_dout)   CUDA_CHECK(cudaMalloc(&g_dout,   (size_t)value_dim * sizeof(float)));
    if (!g_dq)     CUDA_CHECK(cudaMalloc(&g_dq,     (size_t)key_dim * sizeof(float)));
    if (!g_dk)     CUDA_CHECK(cudaMalloc(&g_dk,     (size_t)key_dim * sizeof(float)));
    if (!g_dp)     CUDA_CHECK(cudaMalloc(&g_dp,     (size_t)64 * sizeof(float)));
    if (!g_dpgate) CUDA_CHECK(cudaMalloc(&g_dpgate, (size_t)32 * sizeof(float)));

    float *conv_state = g_dconv_all + (size_t)layer_idx * (size_t)conv_kernel * qkv_dim;
    const float *conv_w = (const float*)L->ssm_conv1d.dev;
    delta_conv_kernel<<<(qkv_dim + 255) / 256, 256>>>(
        conv_state, qkv, conv_w, qkv, qkv_dim, conv_kernel);

    float *S = g_dstate_all + (size_t)layer_idx * n_v_heads * head_v_dim * head_k_dim;
    block_q8_1_c *xq = norm_x_q8_ensure(value_dim);
    float *d_gated_target = (!tuff_exact() && mmvq_dual_enabled()) ? NULL : g_dgated;
    fused_gated_delta_net_ar_kernel<<<n_v_heads, head_v_dim>>>(
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

    mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>((const block_q8_0*)ssm_out->dev, g_dgated, g_dy, value_dim);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(attn_out, g_dy, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-10: run the three shared-expert projections that all take norm_x as input
// (ffn_gate_shexp, ffn_up_shexp -> sh_ffn_dim; ffn_gate_inp_shexp -> 1) with a SINGLE upload of the
// shared input and back-to-back launches, then download all three. Each output element is produced
// by the exact same kernel / accumulation order as gpu_mat_vec, so the result is bit-identical; we
// only remove the 2 redundant norm_x uploads and their serialized syncs.
static void shared_proj_launch(const tensor_t *w, const float *xin, float *yout, int in_dim) {
    const int out_dim = (int)w->info.dims[1];
    switch ((int)w->info.type) {
        case GGML_TYPE_Q8_0: mul_mat_vec_q8_0_kernel<<<out_dim, 256>>>((const block_q8_0*)w->dev, xin, yout, in_dim); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q6_K_kernel<<<out_dim, 256>>>((const block_q6_K*)w->dev, xin, yout, in_dim); break;
        case GGML_TYPE_F32:  mul_mat_vec_f32_kernel <<<out_dim, 256>>>((const float*)w->dev, xin, yout, in_dim); break;
        default:
            fprintf(stderr, "[FATAL CUDA ERROR] shared_proj_launch: unsupported type %d\n", (int)w->info.type);
            exit(EXIT_FAILURE);
    }
}

void gpu_shared_expert_projections(float *sh_gate, const tensor_t *gate_t,
                                   float *sh_up,   const tensor_t *up_t,
                                   float *sh_ginp, const tensor_t *ginp_t,
                                   const float *norm_x, int in_dim) {
    const int sh_ffn = (int)gate_t->info.dims[1];
    if (!g_shg)  CUDA_CHECK(cudaMalloc(&g_shg,  (size_t)sh_ffn * sizeof(float)));
    if (!g_shu)  CUDA_CHECK(cudaMalloc(&g_shu,  (size_t)sh_ffn * sizeof(float)));
    if (!g_shgi) CUDA_CHECK(cudaMalloc(&g_shgi, (size_t)sh_ffn * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(g_dx, norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice));
    shared_proj_launch(gate_t, g_dx, g_shg, in_dim);
    shared_proj_launch(up_t,   g_dx, g_shu, in_dim);
    shared_proj_launch(ginp_t, g_dx, g_shgi, in_dim);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(sh_gate, g_shg,  (size_t)sh_ffn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sh_up,   g_shu,  (size_t)sh_ffn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sh_ginp, g_shgi, sizeof(float), cudaMemcpyDeviceToHost));
}

// Iteration-13: bit-exact full-attention for Qwen3.5/3.6, offloaded from engine.c's scalar CPU
// per-head loop. Replicates the CPU math and fp accumulation order EXACTLY so the oracle stays
// 32/32 bit-identical:
//   per query head h: kv_h = h / gqa_ratio
//     for t in [0,ctx): dotp = sum_{d=0..head_dim-1} qh[d]*kt[d];  scores[t] = dotp*scale
//     softmax(scores)   -- CPU order: max over indices 0..ctx (from x[0]), then exp(x-max)
//                         accumulating sum in index order, then multiply by 1/sum
//     concat_out[h][d] = sum_{t=0..ctx-1} scores[t] * vt[t][d]    (per d, t increasing)
// One block per query head (head_dim threads). Stage A writes this token's k/v rows into the
// device cache (each block writes its own kv_h group redundantly with identical bytes, so the
// write is visible to that block's own reads below); Stage B computes scores (one t per thread,
// each an independent sequential dotp); Stage C thread 0 runs the sequential softmax; Stage D
// each d thread accumulates weighted V over t in order. Low parallelism is fine: attention is not
// the compute bottleneck; the goal is to move it off the CPU critical path. The gate sigmoid
// multiply stays on the host (elementwise, order-independent, avoids a device-expf divergence).
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

static void kv_cache_ensure(int n_attn_layers, int n_head_kv, int max_seq, int head_dim) {
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit(void) {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait(void) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#else
__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem_src) {
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
}

__device__ __forceinline__ void cp_async_commit(void) {}

template <int N>
__device__ __forceinline__ void cp_async_wait(void) {}
#endif

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

static void launch_attention_kernel(
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
static void attn_scores_ensure(int n_head, int max_seq) {
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

double gpu_vram_used_mb(void) {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return (double)(total_b - free_b) / (1024.0 * 1024.0);
}

// ===========================================================================
// P4b: Device-Resident Residual Stream and Fused Fast Path Implementation
// ===========================================================================

const float *gpu_get_d_norm_x(void) { return g_d_norm_x; }
float *gpu_get_d_res_x(void) { return g_d_res_x; }
const float *gpu_get_shexp_dev_ptr(void) {
    return g_d_shexp_out;
}

const float *gpu_get_d_top_values(void) {
    return g_d_top_values;
}

const int *gpu_get_d_exslots(void) {
    return g_exslots;
}

void gpu_copy_dev_i32(const int *dev_src, int *host_dst, int n) {
    CUDA_CHECK(cudaMemcpy(host_dst, dev_src, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
}

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
        mmvq_q8_0_dense_delta_quad_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<grid, block>>>(
            (const block_q8_0*)attn_qkv->dev, g_dpqkv, out_qkv,
            (const block_q8_0*)attn_gate->dev, g_dpz,   out_z,
            (const float*)ssm_beta->dev,       g_dpb,   out_beta,
            (const float*)ssm_alpha->dev,      g_dpa,   out_alpha,
            xq, d_norm_x, nblk, in_dim);
        CUDA_CHECK_LAST();
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
    kv_cache_ensure(n_attn_layers, n_head_kv, max_seq, head_dim);
    attn_scores_ensure(n_head, max_seq);

    uint32_t q_full_dim = (uint32_t)layer->attn_q.info.dims[1]; // 8192
    uint32_t k_dim      = (uint32_t)layer->attn_k.info.dims[1]; // 512
    uint32_t v_dim      = (uint32_t)layer->attn_v.info.dims[1]; // 512

    if (!tuff_exact() && mmvq_dual_enabled()) {
        const block_q8_1_c *xq = (const block_q8_1_c *)g_d_dense_x_q8;
        if (tuff_h50()) {
            const int nblk = dim / QK8_0;
            const int total_qkv = (int)q_full_dim + (int)k_dim + (int)v_dim;
            dim3 qkv_grid((unsigned)((total_qkv + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
            dim3 qkv_block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
            mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>(
                (const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim,
                (const block_q8_0*)layer->attn_k.dev, g_d_k,      (int)k_dim,
                (const block_q8_0*)layer->attn_v.dev, g_d_v,      (int)v_dim,
                xq, nblk);
            CUDA_CHECK_LAST();
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

void gpu_router_logits_launch(float *host_logits, float *host_norm_x, const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x, g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x, g_d_router_logits, in_dim, n_expert);
    }
    if (host_logits) {
        CUDA_CHECK(cudaMemcpyAsync(host_logits, g_d_router_logits, (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, 0));
}

extern "C" void gpu_router_logits_dev_only(const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x, g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x, g_d_router_logits, in_dim, n_expert);
    }
}

// H7: the router compute for column `col`. Column 1 lands in its own device buffer so BOTH
// columns' router work can sit inside one captured graph — with a single shared buffer the
// host had to drain column 0 before column 1 could be issued, which is exactly the launch
// serialization the graph exists to remove.
extern "C" void gpu_router_logits_dev_only_col(const tensor_t *ffn_gate_inp, const float *d_norm_x,
                                               int in_dim, int n_expert, int col) {
    if (col != 0 && !g_d_router_logits2)
        CUDA_CHECK(cudaMalloc(&g_d_router_logits2, (size_t)256 * sizeof(float)));
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x,
                            col ? g_d_router_logits2 : g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x,
                           col ? g_d_router_logits2 : g_d_router_logits, in_dim, n_expert);
    }
}

// Both columns' router logits and normalized activations, one burst of D2H after the graph.
extern "C" void gpu_router_logits_copy_out_n2(float *logits0, float *nx0, float *logits1, float *nx1,
                                              const float *d_norm_x2, int in_dim, int n_expert) {
    CUDA_CHECK(cudaMemcpyAsync(logits0, g_d_router_logits,  (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(nx0,     d_norm_x2,          (size_t)in_dim   * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(logits1, g_d_router_logits2, (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(nx1, d_norm_x2 + (size_t)in_dim, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// H7 stage 1 measurement: cost of a dense Q8_0 projection over 1, 2 and 4 activation columns.
// Weight bytes are read once per launch regardless of ncols, so if the card is bandwidth-bound
// the per-column cost should collapse. Reports wall time per launch and GB/s of weight bytes.
extern "C" void gpu_bench_dense_batch(const tensor_t *w, int in_dim, int out_dim, int reps) {
    if (!w || !w->dev) { fprintf(stderr, "[BENCH] tensor has no device copy\n"); return; }
    const int nblk = in_dim / QK8_0;
    const size_t wbytes = (size_t)out_dim * nblk * sizeof(block_q8_0);

    float *d_x = NULL; void *d_xq = NULL; float *d_y = NULL;
    CUDA_CHECK(cudaMalloc(&d_x,  (size_t)4 * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xq, (size_t)4 * nblk * sizeof(block_q8_1_c)));
    CUDA_CHECK(cudaMalloc(&d_y,  (size_t)4 * out_dim * sizeof(float)));

    float *h_x = (float*)malloc((size_t)4 * in_dim * sizeof(float));
    for (int i = 0; i < 4 * in_dim; ++i) h_x[i] = 0.02f * ((i % 97) - 48);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)4 * in_dim * sizeof(float), cudaMemcpyHostToDevice));
    for (int c = 0; c < 4; ++c) {
        quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(d_x + (size_t)c * in_dim,
                                                            (block_q8_1_c*)d_xq + (size_t)c * nblk, in_dim);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("[BENCH] dense Q8_0 projection %d x %d, %.1f MiB of weights, %d reps\n",
           in_dim, out_dim, wbytes / 1048576.0, reps);
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    double t1 = 0.0;
    for (int ncols = 1; ncols <= 4; ncols *= 2) {
        for (int warm = 0; warm < 3; ++warm) {
            if (ncols == 1) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 2) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 4) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
        }
        CUDA_CHECK(cudaStreamSynchronize(0));
        const double t0 = (double)clock() / CLOCKS_PER_SEC;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        for (int r = 0; r < reps; ++r) {
            if (ncols == 1) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 2) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            if (ncols == 4) mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>((const block_q8_0*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
        }
        CUDA_CHECK(cudaStreamSynchronize(0));
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        (void)t0;
        const double dt = (ts1.tv_sec - ts0.tv_sec) + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
        const double us = dt / reps * 1e6;
        if (ncols == 1) t1 = us;
        printf("  ncols=%d  %8.2f us/launch  %7.1f GB/s of weights  |  per column %7.2f us  (%.2fx of ncols=1)\n",
               ncols, us, wbytes / (us * 1e-6) / 1e9, us / ncols, us / t1);
    }
    free(h_x);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_xq)); CUDA_CHECK(cudaFree(d_y));
}

// H7 stage 2a measurement: the same question for the Q6_K output head, the single largest read
// in the step (398 MB). A draft and a verify both need logits for more than one position.
extern "C" void gpu_bench_head_batch(const tensor_t *w, int in_dim, int out_dim, int reps) {
    if (!w || !w->dev) { fprintf(stderr, "[BENCH] output head has no device copy\n"); return; }
    const int nblk = in_dim / QK_K;                 // q6_K super-blocks per row
    const int yblk = nblk * (QK_K / QK8_1);         // q8_1 blocks per activation column
    const size_t wbytes = (size_t)out_dim * nblk * sizeof(block_q6_K);

    float *d_x = NULL; void *d_xq = NULL; float *d_y = NULL;
    CUDA_CHECK(cudaMalloc(&d_x,  (size_t)4 * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xq, (size_t)4 * yblk * sizeof(block_q8_1_c)));
    CUDA_CHECK(cudaMalloc(&d_y,  (size_t)4 * out_dim * sizeof(float)));

    float *h_x = (float*)malloc((size_t)4 * in_dim * sizeof(float));
    for (int i = 0; i < 4 * in_dim; ++i) h_x[i] = 0.02f * ((i % 89) - 44);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)4 * in_dim * sizeof(float), cudaMemcpyHostToDevice));
    for (int c = 0; c < 4; ++c) {
        quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(d_x + (size_t)c * in_dim,
                                                            (block_q8_1_c*)d_xq + (size_t)c * yblk, in_dim);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("[BENCH] Q6_K output head %d x %d, %.1f MiB of weights, %d reps\n",
           in_dim, out_dim, wbytes / 1048576.0, reps);
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    double t1 = 0.0;
    for (int ncols = 1; ncols <= 4; ncols *= 2) {
        for (int pass = 0; pass < 2; ++pass) {          // pass 0 = warmup, pass 1 = timed
            const int n = pass ? reps : 2;
            struct timespec ts0, ts1;
            CUDA_CHECK(cudaStreamSynchronize(0));
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            for (int r = 0; r < n; ++r) {
                if (ncols == 1) mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 1><<<grid, block>>>((const block_q6_K*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
                if (ncols == 2) mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>((const block_q6_K*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
                if (ncols == 4) mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 4><<<grid, block>>>((const block_q6_K*)w->dev, (const block_q8_1_c*)d_xq, d_y, out_dim, nblk);
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            if (!pass) continue;
            const double dt = (ts1.tv_sec - ts0.tv_sec) + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
            const double us = dt / n * 1e6;
            if (ncols == 1) t1 = us;
            printf("  ncols=%d  %8.2f us/launch  %7.1f GB/s of weights  |  per column %7.2f us  (%.2fx of ncols=1)\n",
                   ncols, us, wbytes / (us * 1e-6) / 1e9, us / ncols, us / t1);
        }
    }
    free(h_x);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_xq)); CUDA_CHECK(cudaFree(d_y));
}

// H12: the D2H half of gpu_router_logits_launch, for layers whose router compute is already
// inside a captured graph. The copies stay outside the graph because their destinations are host
// buffers the decode thread reads immediately after the sync.
extern "C" void gpu_router_logits_copy_out(float *host_logits, float *host_norm_x,
                                           const float *d_norm_x, int in_dim, int n_expert) {
    if (host_logits) {
        CUDA_CHECK(cudaMemcpyAsync(host_logits, g_d_router_logits, (size_t)n_expert * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, 0));
}

void gpu_router_logits_wait(void) {
    CUDA_CHECK(cudaStreamSynchronize(0));
}

// H6: wait only for the router-logits + norm_x D2H (marked by the event), not the
// whole stream. The shared expert, already ordered after the copies on stream 0, keeps
// running while the host does top-k and submits the CPU-miss work.
void gpu_router_logits_wait_event(void) {
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventSynchronize(g_router_d2h_event));
}

void gpu_router_logits_dev(float *host_logits, float *host_norm_x, const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    gpu_router_logits_launch(host_logits, host_norm_x, ffn_gate_inp, d_norm_x, in_dim, n_expert);
    gpu_router_logits_wait();
}

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
        mmvq_q8_0_dense_shexp_swiglu_quant_kernel<<<grid, block>>>(
            (const block_q8_0*)layer->ffn_gate_shexp.dev,
            (const block_q8_0*)layer->ffn_up_shexp.dev,
            (block_q8_1_c*)g_d_shact_q8,
            sh_ffn_dim, xq, nblk);
        CUDA_CHECK_LAST();

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
        quantize_q8_1_kernel<<<(in_dim + 255)/256, 256>>>(d_norm_x, xq8, in_dim);
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
    quantize_q8_1_kernel<<<(nq + 255)/256, 256>>>(ex_dx, (block_q8_1_c *)((char*)g_exdx_q8 + (size_t)col * g_ex_stride_dxq8), nq);
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
static const block_q8_1_c *ex_col_xq8(int col) {
    return (const block_q8_1_c *)((char*)g_exgx_q8 + (size_t)col * g_ex_stride_gxq8);
}
static block_q8_1_c *ex_col_dxq8(int col) {
    return (block_q8_1_c *)((char*)g_exdx_q8 + (size_t)col * g_ex_stride_dxq8);
}
static float *ex_col_dx(int col)  { return g_exdx   + (size_t)col * g_ex_stride_dx; }
static float *ex_col_dbat(int col){ return g_exdbat + (size_t)col * g_ex_stride_dbat; }
static int   *ex_col_slots(int col){ return g_exslots + (size_t)col * g_ex_stride_slots; }

// One column's normalized input, quantized to q8_1 for the gate/up kernels.
void gpu_expert_moe_normx_quant_col(const float *d_norm_x, int in_dim, int col) {
    quantize_q8_1_kernel<<<(in_dim + 255)/256, 256>>>(d_norm_x, (block_q8_1_c*)ex_col_xq8(col), in_dim);
    CUDA_CHECK_LAST();
}

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
    quantize_q8_1_kernel<<<(nq + 255)/256, 256>>>(ex_col_dx(col), ex_col_dxq8(col), nq);
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
// Fused device top-k MoE routing (Ported from llama.cpp ggml-cuda/topk-moe.cu)
// Warp-level softmax + iterative argmax top-k reduction.
// ---------------------------------------------------------------------------

template <int experts_per_thread>
static __device__ void softmax_warp_inplace_f(float (&vals)[experts_per_thread], const int lane) {
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        max_val = fmaxf(max_val, vals[i]);
    }
    max_val = warp_reduce_max_f(max_val);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const float val = expf(vals[i] - max_val);
        vals[i] = val;
        sum += val;
    }
    sum = warp_reduce_sum_f(sum);
    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        vals[i] *= inv_sum;
    }
}

template <int n_experts, int n_expert_used>
__global__ void topk_moe_resident_kernel(
    const float * __restrict__ logits,
    float * __restrict__ weights,
    int * __restrict__ exslots,
    int layer_idx) {
    constexpr int experts_per_thread = (n_experts > 32) ? n_experts / 32 : 1;
    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += 32) {
        const int expert = i + threadIdx.x;
        wt[i / 32] = (expert < n_experts) ? logits[expert] : -INFINITY;
    }

    softmax_warp_inplace_f<experts_per_thread>(wt, threadIdx.x);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.0f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.0f;
    }

    for (int k = 0; k < n_expert_used; k++) {
        float max_val = wt[0];
        int max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * 32;
            if (expert < n_experts && wt[i] > max_val) {
                max_val = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val = __shfl_xor_sync(0xffffffff, max_val, mask, 32);
            const int expert = __shfl_xor_sync(0xffffffff, max_expert, mask, 32);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val = val;
                max_expert = expert;
            }
        }

        if ((max_expert & 31) == threadIdx.x) {
            wt[max_expert / 32] = -INFINITY;
            exslots[k] = layer_idx * 256 + max_expert;
            wt_sum += max_val;
        }

        if ((k & 31) == threadIdx.x) {
            output_weights[k / 32] = max_val;
        }
    }

    wt_sum = warp_reduce_sum_f(wt_sum);
    const float inv_sum = 1.0f / fmaxf(wt_sum, 1e-20f);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] *= inv_sum;
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * 32 + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i];
        }
    }
}

template <int n_experts, int n_expert_used>
__global__ void topk_moe_kernel(
    const float * __restrict__ logits,
    float * __restrict__ weights,
    int * __restrict__ indices) {
    constexpr int experts_per_thread = (n_experts > 32) ? n_experts / 32 : 1;
    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += 32) {
        const int expert = i + threadIdx.x;
        wt[i / 32] = (expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Step 1: Iterative argmax across all 256 raw logits
    float top_raw[n_expert_used];
    for (int k = 0; k < n_expert_used; k++) {
        float max_val = wt[0];
        int max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * 32;
            if (expert < n_experts && wt[i] > max_val) {
                max_val = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val = __shfl_xor_sync(0xffffffff, max_val, mask, 32);
            const int expert = __shfl_xor_sync(0xffffffff, max_expert, mask, 32);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val = val;
                max_expert = expert;
            }
        }

        if ((max_expert & 31) == threadIdx.x) {
            wt[max_expert / 32] = -INFINITY;
            indices[k] = max_expert;
        }

        top_raw[k] = max_val;
    }

    // Step 2: Exact softmax over the top-8 selected logits
    if (threadIdx.x < n_expert_used) {
        float max_l = top_raw[0];
#pragma unroll
        for (int k = 1; k < n_expert_used; k++) {
            max_l = fmaxf(max_l, top_raw[k]);
        }

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < n_expert_used; k++) {
            sum += expf(top_raw[k] - max_l);
        }

        const float inv_sum = 1.0f / sum;
        weights[threadIdx.x] = expf(top_raw[threadIdx.x] - max_l) * inv_sum;
    }
}

static cudaStream_t g_d2h_stream = NULL;
static cudaEvent_t  g_topk_done_event = NULL;

void gpu_router_topk_launch(int *host_top_indices, float *host_top_values, float *host_norm_x,
                            const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert) {
    if (!g_d2h_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_d2h_stream, cudaStreamNonBlocking));
    }
    if (!g_topk_done_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_topk_done_event, cudaEventDisableTiming));
    }
    if (router_q8_enabled() && ffn_gate_inp->dev_q8) {
        launch_mat_vec_q8_0((const block_q8_0*)ffn_gate_inp->dev_q8, d_norm_x, g_d_router_logits, in_dim, n_expert);
    } else {
        launch_mat_vec_f32((const float*)ffn_gate_inp->dev, d_norm_x, g_d_router_logits, in_dim, n_expert);
    }
    topk_moe_kernel<256, 8><<<1, 32>>>(g_d_router_logits, g_d_top_values, g_d_top_indices);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaEventRecord(g_topk_done_event, 0));
    CUDA_CHECK(cudaStreamWaitEvent(g_d2h_stream, g_topk_done_event, 0));

    if (host_top_indices) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_indices, g_d_top_indices, 8 * sizeof(int), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_top_values) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_values, g_d_top_values, 8 * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, g_d2h_stream));
}

void gpu_router_topk_copy_out(int *host_top_indices, float *host_top_values, float *host_norm_x,
                              const float *d_norm_x, int in_dim) {
    if (!g_d2h_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_d2h_stream, cudaStreamNonBlocking));
    }
    if (!g_topk_done_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_topk_done_event, cudaEventDisableTiming));
    }
    topk_moe_kernel<256, 8><<<1, 32>>>(g_d_router_logits, g_d_top_values, g_d_top_indices);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaEventRecord(g_topk_done_event, 0));
    CUDA_CHECK(cudaStreamWaitEvent(g_d2h_stream, g_topk_done_event, 0));

    if (host_top_indices) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_indices, g_d_top_indices, 8 * sizeof(int), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_top_values) {
        CUDA_CHECK(cudaMemcpyAsync(host_top_values, g_d_top_values, 8 * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (host_norm_x) {
        CUDA_CHECK(cudaMemcpyAsync(host_norm_x, d_norm_x, (size_t)in_dim * sizeof(float), cudaMemcpyDeviceToHost, g_d2h_stream));
    }
    if (!g_router_d2h_event)
        CUDA_CHECK(cudaEventCreateWithFlags(&g_router_d2h_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g_router_d2h_event, g_d2h_stream));
}

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
    topk_moe_resident_kernel<256, 8><<<1, 32>>>(g_d_router_logits, g_d_top_values, g_exslots, layer_idx);
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

__global__ void argmax_block_kernel(const float * __restrict__ x, float * __restrict__ block_max,
                                    int * __restrict__ block_idx, int n) {
    __shared__ float s_max[256];
    __shared__ int   s_idx[256];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float cur_max = -1e30f;
    int   cur_idx = 0;
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        if (v > cur_max) {
            cur_max = v;
            cur_idx = i;
        }
    }
    s_max[tid] = cur_max;
    s_idx[tid] = cur_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = s_max[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}

__global__ void argmax_final_kernel(const float * __restrict__ block_max,
                                    const int * __restrict__ block_idx,
                                    int * __restrict__ out_best_token,
                                    int n_blocks) {
    __shared__ float s_max[256];
    __shared__ int   s_idx[256];
    const int tid = threadIdx.x;

    float cur_max = -1e30f;
    int   cur_idx = 0;
    for (int i = tid; i < n_blocks; i += 256) {
        float v = block_max[i];
        if (v > cur_max) {
            cur_max = v;
            cur_idx = block_idx[i];
        }
    }
    s_max[tid] = cur_max;
    s_idx[tid] = cur_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_best_token = s_idx[0];
    }
}

#define HEAD_ROWS_PER_BLOCK 8

int32_t gpu_output_argmax_dev(const tensor_t *output_weight, const float *d_norm_x, int in_dim, int vocab_size, float *host_logits_capture) {
    const block_q8_1_c *xq = dense_quantize_x(d_norm_x, in_dim);
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    float *dst_logits = host_logits_capture ? g_d_logits : NULL;

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq, dst_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits_capture && dst_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits_capture, dst_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token, grid_blocks);
    CUDA_CHECK_LAST();

    int32_t best_token = 0;
    CUDA_CHECK(cudaMemcpy(&best_token, g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return best_token;
}

// Fused Layer RMS-Norm and Q8_1 Quantization Kernel (H48/H49)
__global__ void fused_rms_norm_quantize_q8_1_kernel(
        const float * __restrict__ x,
        const float * __restrict__ w,
        float * __restrict__ dst_norm_x,
        block_q8_1_c * __restrict__ y,
        int in_dim,
        float eps) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < in_dim; i += num_threads) {
        float xi = x[i];
        sum_sq += xi * xi;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask, 32);
    }
    __shared__ float s_warp_sum[8];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    if (lane_id == 0) s_warp_sum[warp_id] = sum_sq;
    __syncthreads();

    __shared__ float s_scale;
    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) total_sq += s_warp_sum[k];
        s_scale = rsqrtf(total_sq / (float)in_dim + eps);
    }
    __syncthreads();

    const float scale = s_scale;
    const int total_q_blocks = in_dim / QK8_1;
    for (int b = warp_id; b < total_q_blocks; b += 8) {
        const int idx = b * QK8_1 + lane_id;
        float norm_val = x[idx] * scale * w[idx];
        if (dst_norm_x) dst_norm_x[idx] = norm_val;
        if (y) {
            float amax = fabsf(norm_val);
            float sum = norm_val;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
                sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
            }
            const float d = amax / 127.0f;
            const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(norm_val / d);
            y[b].qs[lane_id] = q;
            if (lane_id == 0) {
                y[b].ds = make_half2(__float2half(d), __float2half(sum));
            }
        }
    }
}

extern "C" void gpu_fused_rms_norm_quant_dev(float *d_dst_norm_x, const float *d_src, const float *d_weight, void *d_dst_q8, int dim, float eps) {
    if (tuff_h49()) {
        fused_rms_norm_quantize_q8_1_kernel<<<1, 256>>>(d_src, d_weight, d_dst_norm_x, (block_q8_1_c *)d_dst_q8, dim, eps);
    } else {
        gpu_rms_norm_dev(d_dst_norm_x, d_src, d_weight, dim, eps);
        if (d_dst_q8) {
            quantize_q8_1_kernel<<<(dim + 255) / 256, 256>>>(d_dst_norm_x, (block_q8_1_c *)d_dst_q8, dim);
        }
    }
    CUDA_CHECK_LAST();
}

int32_t gpu_fused_output_norm_quant_argmax_dev(
        const tensor_t *output_weight,
        const float *d_res_x,
        const float *d_norm_w,
        int in_dim,
        int vocab_size,
        float eps,
        float *host_logits_capture) {
    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / QK8_1) * sizeof(block_q8_1_c)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    gpu_fused_rms_norm_quant_dev(g_d_norm_x, d_res_x, d_norm_w, xq8, in_dim, eps);

    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    float *dst_logits = host_logits_capture ? g_d_logits : NULL;

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq8, dst_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits_capture && dst_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits_capture, dst_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token, grid_blocks);
    CUDA_CHECK_LAST();

    int32_t best_token = 0;
    CUDA_CHECK(cudaMemcpy(&best_token, g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return best_token;
}

void gpu_output_logits_dev(const tensor_t *output_weight, const float *d_norm_x, int in_dim, int vocab_size, float *host_logits) {
    if (!g_d_logits) {
        CUDA_CHECK(cudaMalloc(&g_d_logits, (size_t)vocab_size * sizeof(float)));
    }
    const block_q8_1_c *xq = dense_quantize_x(d_norm_x, in_dim);
    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq, g_d_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits, g_d_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void gpu_fused_output_norm_quant_logits_dev(
        const tensor_t *output_weight,
        const float *d_res_x,
        const float *d_norm_w,
        int in_dim,
        int vocab_size,
        float eps,
        float *host_logits) {
    if (!g_d_out_normx_q8) {
        CUDA_CHECK(cudaMalloc(&g_d_out_normx_q8, (size_t)(in_dim / QK8_1) * sizeof(block_q8_1_c)));
    }
    if (!g_d_logits) {
        CUDA_CHECK(cudaMalloc(&g_d_logits, (size_t)vocab_size * sizeof(float)));
    }
    block_q8_1_c *xq8 = (block_q8_1_c *)g_d_out_normx_q8;
    gpu_fused_rms_norm_quant_dev(g_d_norm_x, d_res_x, d_norm_w, xq8, in_dim, eps);

    const int nblk = in_dim / QK_K;
    const int grid_blocks = (vocab_size + HEAD_ROWS_PER_BLOCK - 1) / HEAD_ROWS_PER_BLOCK;
    dim3 grid((unsigned)grid_blocks, 1u, 1u);
    dim3 block(32u, (unsigned)HEAD_ROWS_PER_BLOCK, 1u);

    mmvq_q6_K_dense_argmax_warprow_kernel<HEAD_ROWS_PER_BLOCK><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, xq8, g_d_logits, g_d_block_max, g_d_block_idx, vocab_size, nblk);
    CUDA_CHECK_LAST();

    if (host_logits) {
        CUDA_CHECK(cudaMemcpy(host_logits, g_d_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// ===========================================================================
// H7 stage 2b: the two-column (verify) trunk.
//
// Speculative decoding verifies [x_{t+1}, draft] in one pass. Both inputs are
// known before the pass starts, so the trunk can be evaluated for both
// positions while the weight stream is read once — the property measured in
// iteration 42 (a second column costs 0.06% on the Q8_0 projections and 1.46x
// on the Q6_K head).
//
// The residual and normalized streams become [2][dim]; column c lives at
// base + c*dim. That layout is what makes this affordable: every entry point
// that is already correct for one token (the GDN recurrence, the attention
// layer, the routed experts, the MoE combine) runs per column with no kernel
// change at all, by passing base + c*dim. Only the places where the weight
// stream is genuinely shared get new launchers — the delta-layer input
// projections (~1.05 GiB/token, the largest shareable read in the step) and
// the Q6_K output head (398 MB).
//
// The single-column path below is deliberately untouched: it is what the
// oracle gate pins, and speculation must not be able to perturb it.
// ===========================================================================

extern "C" void gpu_n2_prepare(int dim, int vocab_size) {
    if (g_n2_dim == dim && g_d_res_x2) return;
    g_n2_dim = dim;
    if (!g_d_res_x2)  CUDA_CHECK(cudaMalloc(&g_d_res_x2,  (size_t)2 * dim * sizeof(float)));
    if (!g_d_norm_x2) CUDA_CHECK(cudaMalloc(&g_d_norm_x2, (size_t)2 * dim * sizeof(float)));
    // The q8_1 staging buffer is sized by the widest activation the ncols kernels
    // see, which is the trunk width (dim), not any projection's output width.
    if (!g_d_nx2_q8)  CUDA_CHECK(cudaMalloc(&g_d_nx2_q8,  (size_t)2 * (dim / QK8_1) * sizeof(block_q8_1_c)));
    if (!g_d_logits2) CUDA_CHECK(cudaMalloc(&g_d_logits2, (size_t)2 * vocab_size * sizeof(float)));
}

// Second shared-expert output buffer, so both columns of a layer can be issued
// before the single router sync instead of one on each side of it.
extern "C" const float *gpu_get_shexp2_dev_ptr(void) {
    if (!g_d_shexp_out2 && g_n2_dim > 0)
        CUDA_CHECK(cudaMalloc(&g_d_shexp_out2, (size_t)g_n2_dim * sizeof(float)));
    return g_d_shexp_out2;
}

extern "C" float *gpu_get_d_res_x2(void)  { return g_d_res_x2; }
extern "C" float *gpu_get_d_norm_x2(void) { return g_d_norm_x2; }

extern "C" void gpu_residual_reset_n2(const float *h_x0, const float *h_x1, int dim) {
    CUDA_CHECK(cudaMemcpyAsync(g_d_res_x2,              h_x0, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(g_d_res_x2 + (size_t)dim, h_x1, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void gpu_rms_norm_dev_n2(float *d_dst2, const float *d_src2, const float *d_weight, int dim, float eps) {
    rms_norm_dev_kernel<<<1, 256>>>(d_dst2,                d_src2,                d_weight, dim, eps);
    rms_norm_dev_kernel<<<1, 256>>>(d_dst2 + (size_t)dim,  d_src2 + (size_t)dim,  d_weight, dim, eps);
    CUDA_CHECK_LAST();
}

// Quantize both normalized columns to q8_1 once; every ncols kernel below reads
// this staging buffer. Q8_0 and Q6_K forms both consume in_dim/32 blocks per
// column (QK_K/QK8_1 = 8 sub-blocks per super-block), so one buffer serves both.
extern "C" void gpu_n2_quantize_norm_x(const float *d_src2, int in_dim) {
    const int nblk = in_dim / QK8_1;
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(
        d_src2, (block_q8_1_c*)g_d_nx2_q8, in_dim);
    quantize_q8_1_kernel<<<(in_dim + 255) / 256, 256>>>(
        d_src2 + (size_t)in_dim, (block_q8_1_c*)g_d_nx2_q8 + nblk, in_dim);
    CUDA_CHECK_LAST();
}

// One weight read, two output columns. dst is [2][out_dim].
static void n2_dense_q8_0(const void *dev_w, float *d_dst2, int in_dim, int out_dim) {
    const int nblk = in_dim / QK8_0;
    dim3 grid((unsigned)((out_dim + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q8_0_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>(
        (const block_q8_0*)dev_w, (const block_q8_1_c*)g_d_nx2_q8, d_dst2, out_dim, nblk);
    CUDA_CHECK_LAST();
}

// The delta-layer input projections for both columns. attn_qkv (2048x8192) and
// attn_gate are the two big Q8_0 reads and are shared; ssm_beta/ssm_alpha are
// F32 and a few hundred KB, so they just run twice.
extern "C" void gpu_delta_projections_n2(const tensor_t *attn_qkv, const tensor_t *attn_gate,
                                         const tensor_t *ssm_beta, const tensor_t *ssm_alpha,
                                         const float *d_norm_x2, int in_dim) {
    const int out_qkv   = (int)attn_qkv->info.dims[1];
    const int out_z     = (int)attn_gate->info.dims[1];
    const int out_beta  = (int)ssm_beta->info.dims[1];
    const int out_alpha = (int)ssm_alpha->info.dims[1];

    if (!g_d_p2_qkv) CUDA_CHECK(cudaMalloc(&g_d_p2_qkv, (size_t)2 * out_qkv   * sizeof(float)));
    if (!g_d_p2_z)   CUDA_CHECK(cudaMalloc(&g_d_p2_z,   (size_t)2 * out_z     * sizeof(float)));
    if (!g_d_p2_b)   CUDA_CHECK(cudaMalloc(&g_d_p2_b,   (size_t)2 * out_beta  * sizeof(float)));
    if (!g_d_p2_a)   CUDA_CHECK(cudaMalloc(&g_d_p2_a,   (size_t)2 * out_alpha * sizeof(float)));

    gpu_n2_quantize_norm_x(d_norm_x2, in_dim);
    n2_dense_q8_0(attn_qkv->dev,  g_d_p2_qkv, in_dim, out_qkv);
    n2_dense_q8_0(attn_gate->dev, g_d_p2_z,   in_dim, out_z);

    launch_mat_vec_f32_dual((const float*)ssm_beta->dev,  g_d_p2_b, out_beta,
                            (const float*)ssm_alpha->dev, g_d_p2_a, out_alpha,
                            d_norm_x2, in_dim);
    launch_mat_vec_f32_dual((const float*)ssm_beta->dev,  g_d_p2_b + out_beta,  out_beta,
                            (const float*)ssm_alpha->dev, g_d_p2_a + out_alpha, out_alpha,
                            d_norm_x2 + (size_t)in_dim, in_dim);
}

// H15: the shared expert over BOTH columns in one weight read. The gate/up pair
// (Q8_0, dim->sh_ffn_dim), the SwiGLU, the down (Q8_0, sh_ffn_dim->dim) and the
// optional gate_inp scale all run over the two columns via the ncols=2 mmvq
// launcher, so each weight row is read once for both tokens instead of twice.
// Column c's output is bit-identical to gpu_shared_expert_dev on column c alone
// while the mmvq dual path is active: the ncols kernel performs the same per-thread
// accumulation and warp reduction over the same q8_1 blocks. When that path is
// disabled (TUFF_EXACT=1 or TUFF_MMVQ_DUAL=0) it falls back to two per-column
// calls, which the ncols kernels cannot reproduce bit-for-bit.
void gpu_shared_expert_dev_n2(float *d_shexp_out0, float *d_shexp_out1, const model_layer_t *layer,
                              const float *d_norm_x2, int in_dim, int sh_ffn_dim) {
    static int mmvq_dual = -1;
    if (mmvq_dual < 0) { const char *e = getenv("TUFF_MMVQ_DUAL"); mmvq_dual = (e && e[0] == '0') ? 0 : 1; }
    if (tuff_exact() || !mmvq_dual) {
        gpu_shared_expert_dev(d_shexp_out0, layer, d_norm_x2, in_dim, sh_ffn_dim);
        gpu_shared_expert_dev(d_shexp_out1, layer, d_norm_x2 + (size_t)in_dim, in_dim, sh_ffn_dim);
        return;
    }
    const int dim = in_dim;
    if (!g_shg2)      CUDA_CHECK(cudaMalloc(&g_shg2,      (size_t)2 * sh_ffn_dim * sizeof(float)));
    if (!g_shu2)      CUDA_CHECK(cudaMalloc(&g_shu2,      (size_t)2 * sh_ffn_dim * sizeof(float)));
    if (!g_shgi2)     CUDA_CHECK(cudaMalloc(&g_shgi2,     (size_t)2             * sizeof(float)));
    if (!g_d_shact2)  CUDA_CHECK(cudaMalloc(&g_d_shact2,  (size_t)2 * sh_ffn_dim * sizeof(float)));
    if (!g_d_shdown2) CUDA_CHECK(cudaMalloc(&g_d_shdown2, (size_t)2 * dim        * sizeof(float)));

    // gate/up share the normalized input; one q8_1 quantization of both columns.
    gpu_n2_quantize_norm_x(d_norm_x2, in_dim);
    n2_dense_q8_0(layer->ffn_gate_shexp.dev, g_shg2, in_dim, sh_ffn_dim);
    n2_dense_q8_0(layer->ffn_up_shexp.dev,   g_shu2, in_dim, sh_ffn_dim);
    swiglu_dev_ncols_kernel<<<(2 * sh_ffn_dim + 255) / 256, 256>>>(g_d_shact2, g_shg2, g_shu2, 2 * sh_ffn_dim);
    CUDA_CHECK_LAST();

    // The down projection's activation is the SwiGLU output, sh_ffn_dim per column.
    // The q8_1 staging buffer is free (gate/up consumed it); nblk = sh_ffn_dim/QK8_1.
    {
        const int nblk = sh_ffn_dim / QK8_1;
        quantize_q8_1_kernel<<<(sh_ffn_dim + 255) / 256, 256>>>(
            g_d_shact2, (block_q8_1_c*)g_d_nx2_q8, sh_ffn_dim);
        quantize_q8_1_kernel<<<(sh_ffn_dim + 255) / 256, 256>>>(
            g_d_shact2 + sh_ffn_dim, (block_q8_1_c*)g_d_nx2_q8 + nblk, sh_ffn_dim);
        CUDA_CHECK_LAST();
    }
    n2_dense_q8_0(layer->ffn_down_shexp.dev, g_d_shdown2, sh_ffn_dim, dim);

    if (layer->ffn_gate_inp_shexp.dev) {
        launch_mat_vec_f32((const float*)layer->ffn_gate_inp_shexp.dev, d_norm_x2,          g_shgi2,     in_dim, 1);
        launch_mat_vec_f32((const float*)layer->ffn_gate_inp_shexp.dev, d_norm_x2 + in_dim, g_shgi2 + 1, in_dim, 1);
        scale_shexp_kernel<<<(dim + 255) / 256, 256>>>(d_shexp_out0, g_d_shdown2,       g_shgi2,     dim);
        scale_shexp_kernel<<<(dim + 255) / 256, 256>>>(d_shexp_out1, g_d_shdown2 + dim, g_shgi2 + 1, dim);
        CUDA_CHECK_LAST();
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_shexp_out0, g_d_shdown2,       (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_shexp_out1, g_d_shdown2 + dim, (size_t)dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// H17: run the fused GDN step for column c straight off the batched projections. The old
// gpu_delta_select_column_n2 published column c into the single-column scratch g_dpqkv/g_dpz/
// g_dpb/g_dpa with four D2D copies per column per layer (240 copies per verify step), because
// delta_conv_kernel consumed g_dpqkv in place. Pointing the fused kernel directly at
// g_d_p2_qkv + c*out_qkv is safe: column c's projection is read exactly once (by the conv), so
// overwriting it with qkv_conv clobbers nothing the rest of the step still needs, and column 1's
// projection is untouched until its own pass. g_d_p2_* are allocated by gpu_delta_projections_n2.
extern "C" void gpu_delta_fused_gdn_dev_col(int c, int layer_idx, const model_layer_t *L, float *d_res_x,
                                            float eps, float scale_q,
                                            int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                            int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim) {
    if (!g_d_p2_qkv) CUDA_CHECK(cudaMalloc(&g_d_p2_qkv, (size_t)2 * qkv_dim   * sizeof(float)));
    if (!g_d_p2_z)   CUDA_CHECK(cudaMalloc(&g_d_p2_z,   (size_t)2 * value_dim * sizeof(float)));
    if (!g_d_p2_b)   CUDA_CHECK(cudaMalloc(&g_d_p2_b,   (size_t)2 * n_v_heads * sizeof(float)));
    if (!g_d_p2_a)   CUDA_CHECK(cudaMalloc(&g_d_p2_a,   (size_t)2 * n_v_heads * sizeof(float)));
    gpu_delta_fused_gdn_dev_p(layer_idx, L, d_res_x, eps, scale_q,
                              g_d_p2_qkv + (size_t)c * qkv_dim,
                              g_d_p2_z   + (size_t)c * value_dim,
                              g_d_p2_b   + (size_t)c * n_v_heads,
                              g_d_p2_a   + (size_t)c * n_v_heads,
                              n_k_heads, n_v_heads, head_k_dim, head_v_dim,
                              key_dim, value_dim, qkv_dim, conv_kernel, dim);
}

// --- Rollback of the recurrent state for a rejected draft ---
//
// The snapshot is taken BETWEEN the two columns of a layer, not before the
// pass. Before the pass the state has consumed tokens up to pos-1; a rejected
// draft needs the state as of pos, i.e. after column 0 but before column 1.
// Taking it per layer slice at that point costs the same 60 MB of D2D in total
// (~0.15 ms, ~1% of the step) and saves re-running the accepted token.
extern "C" void gpu_delta_snapshot_layer_n2(int layer_idx, int n_v_heads, int head_v_dim,
                                            int head_k_dim, int conv_kernel, int qkv_dim) {
    if (!g_dstate_all || g_dstate_bytes == 0) return;
    if (!g_dstate_snap) CUDA_CHECK(cudaMalloc(&g_dstate_snap, g_dstate_bytes));
    if (!g_dconv_snap && g_dconv_bytes > 0) CUDA_CHECK(cudaMalloc(&g_dconv_snap, g_dconv_bytes));

    const size_t state_floats = (size_t)n_v_heads * head_v_dim * head_k_dim;
    CUDA_CHECK(cudaMemcpyAsync(g_dstate_snap + (size_t)layer_idx * state_floats,
                               g_dstate_all  + (size_t)layer_idx * state_floats,
                               state_floats * sizeof(float), cudaMemcpyDeviceToDevice));

    if (g_dconv_all && g_dconv_snap) {
        const size_t conv_floats = (size_t)conv_kernel * qkv_dim;
        CUDA_CHECK(cudaMemcpyAsync(g_dconv_snap + (size_t)layer_idx * conv_floats,
                                   g_dconv_all  + (size_t)layer_idx * conv_floats,
                                   conv_floats * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

extern "C" void gpu_delta_rollback_n2(void) {
    if (g_dstate_all && g_dstate_snap && g_dstate_bytes > 0)
        CUDA_CHECK(cudaMemcpyAsync(g_dstate_all, g_dstate_snap, g_dstate_bytes, cudaMemcpyDeviceToDevice));
    if (g_dconv_all && g_dconv_snap && g_dconv_bytes > 0)
        CUDA_CHECK(cudaMemcpyAsync(g_dconv_all, g_dconv_snap, g_dconv_bytes, cudaMemcpyDeviceToDevice));
}

// The output head over both columns: 398 MB read once, two argmaxes out.
// Measured at 1.46x a single column (iteration 42 stage 2a), not 1.0x.
extern "C" void gpu_output_argmax_n2(const tensor_t *output_weight, const float *d_norm_x2,
                                     int in_dim, int vocab_size, int32_t *out_tokens2) {
    gpu_n2_quantize_norm_x(d_norm_x2, in_dim);

    const int nblk = in_dim / QK_K;
    dim3 grid((unsigned)((vocab_size + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
    dim3 block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
    mmvq_q6_K_dense_ncols_kernel<MMVQ_ROWS_PER_BLOCK, 2><<<grid, block>>>(
        (const block_q6_K*)output_weight->dev, (const block_q8_1_c*)g_d_nx2_q8,
        g_d_logits2, vocab_size, nblk);
    CUDA_CHECK_LAST();

    for (int c = 0; c < 2; ++c) {
        argmax_block_kernel<<<256, 256>>>(g_d_logits2 + (size_t)c * vocab_size,
                                          g_d_block_max, g_d_block_idx, vocab_size);
        CUDA_CHECK_LAST();
        argmax_final_kernel<<<1, 256>>>(g_d_block_max, g_d_block_idx, g_d_best_token, 256);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(&out_tokens2[c], g_d_best_token, sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
}

// ===========================================================================
// CUDA Graph Capture for Delta-Net Dense Layers
// ===========================================================================

static cudaGraph_t     g_layer_graphs[64];
static cudaGraphExec_t g_layer_execs[64];
static int             g_layer_graphs_ready = 0;
static cudaStream_t    g_graph_stream = NULL;

static void cuda_graph_update_exec(cudaGraphExec_t *instance, cudaGraph_t graph) {
#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(*instance, graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(*instance, graph, &errorNode, &result_info);
#endif
    if (stat == cudaErrorGraphExecUpdateFailure) {
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(*instance));
        *instance = NULL;
        CUDA_CHECK(cudaGraphInstantiate(instance, graph, NULL, NULL, 0));
    } else {
        CUDA_CHECK(stat);
    }
}

int gpu_has_delta_graphs(void) {
    return g_layer_graphs_ready;
}

void gpu_set_device_pos(int pos) {
    if (!g_d_pos) {
        CUDA_CHECK(cudaMalloc(&g_d_pos, sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpyAsync(g_d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, 0));
}

// 🔴 This function duplicates the forward token body for all 40 layers (30 Delta-Net + 10 Attention)
// to build persistent CUDA Graphs. Attention layers utilize device-resident position pointer g_d_pos.
void gpu_capture_delta_graphs(model_t *m) {
    if (!m) return;
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int sh_ffn_dim = (int)m->expert_shared_ffn_len;

    if (!g_graph_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_graph_stream, cudaStreamNonBlocking));
    }
    if (!g_d_pos) {
        CUDA_CHECK(cudaMalloc(&g_d_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(g_d_pos, 0, sizeof(int)));
    }

    uint32_t delta_layer_idx = 0;
    uint32_t attn_layer_idx = 0;

    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];

        if (layer->type == LAYER_TYPE_ATTENTION) {
            uint32_t head_dim  = m->key_length;
            uint32_t n_head    = m->head_count;
            uint32_t n_head_kv = m->head_count_kv;
            float scale = 1.0f / sqrtf((float)head_dim);
            uint32_t q_full_dim = (uint32_t)layer->attn_q.info.dims[1];
            uint32_t k_dim      = (uint32_t)layer->attn_k.info.dims[1];
            uint32_t v_dim      = (uint32_t)layer->attn_v.info.dims[1];
            kv_cache_ensure((int)m->n_attn_layers, (int)n_head_kv, g_max_seq, (int)head_dim);
            attn_scores_ensure(n_head, g_max_seq);

            // Dry-run allocation
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            const block_q8_1_c *xq = (const block_q8_1_c *)g_d_dense_x_q8;
            const int nblk = dim / QK8_0;
            const int total_qkv = (int)q_full_dim + (int)k_dim + (int)v_dim;
            dim3 qkv_grid((unsigned)((total_qkv + MMVQ_ROWS_PER_BLOCK - 1) / MMVQ_ROWS_PER_BLOCK), 1u, 1u);
            dim3 qkv_block(32u, (unsigned)MMVQ_ROWS_PER_BLOCK, 1u);
            mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>(
                (const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim,
                (const block_q8_0*)layer->attn_k.dev, g_d_k,      (int)k_dim,
                (const block_q8_0*)layer->attn_v.dev, g_d_v,      (int)v_dim,
                xq, nblk);
            fused_qk_split_norm_rope_kernel<<<n_head + n_head_kv, head_dim>>>(
                g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head,
                g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv,
                head_dim, eps, (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1],
                m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, g_d_pos);
            block_q8_1_c *out_q8 = (block_q8_1_c*)dense_quant_buf_ensure(dim);
            launch_attention_kernel(
                g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, NULL, g_d_attn_scores,
                n_head, n_head_kv, g_max_seq, head_dim, (int)attn_layer_idx, 0, 0, scale, out_q8, g_d_pos);
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_output.dev, out_q8, g_d_res_x, n_head * head_dim, dim, true);
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamSynchronize(0));

            // Capture Attention Graph on stream 0
            if (g_layer_graphs[l]) {
                CUDA_CHECK(cudaGraphDestroy(g_layer_graphs[l]));
                g_layer_graphs[l] = NULL;
            }
            CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            mmvq_q8_0_dense_triple_warprow_kernel<MMVQ_ROWS_PER_BLOCK><<<qkv_grid, qkv_block>>>(
                (const block_q8_0*)layer->attn_q.dev, g_d_q_full, (int)q_full_dim,
                (const block_q8_0*)layer->attn_k.dev, g_d_k,      (int)k_dim,
                (const block_q8_0*)layer->attn_v.dev, g_d_v,      (int)v_dim,
                xq, nblk);
            fused_qk_split_norm_rope_kernel<<<n_head + n_head_kv, head_dim>>>(
                g_d_q, g_d_gate, g_d_q_full, (const float*)layer->attn_q_norm.dev, (int)n_head,
                g_d_k, g_d_k, (const float*)layer->attn_k_norm.dev, (int)n_head_kv,
                head_dim, eps, (int)m->rope_dim_count, 0, m->rope_dim_sections[0], m->rope_dim_sections[1],
                m->rope_dim_sections[2], m->rope_dim_sections[3], m->rope_freq_base, g_d_pos);
            launch_attention_kernel(
                g_dk_cache, g_dv_cache, g_d_q, g_d_k, g_d_v, g_d_gate, NULL, g_d_attn_scores,
                n_head, n_head_kv, g_max_seq, head_dim, (int)attn_layer_idx, 0, 0, scale, out_q8, g_d_pos);
            mmvq_q8_0_rows((const block_q8_0*)layer->attn_output.dev, out_q8, g_d_res_x, n_head * head_dim, dim, true);
            gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs[l]));

            if (g_layer_execs[l]) {
                cuda_graph_update_exec(&g_layer_execs[l], g_layer_graphs[l]);
            } else {
                CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs[l], g_layer_graphs[l], NULL, NULL, 0));
            }
            attn_layer_idx++;
        } else {
            // Delta-Net Layer
            uint32_t key_dim     = m->ssm_key_dim;
            uint32_t value_dim   = m->ssm_value_dim;
            uint32_t qkv_dim     = m->ssm_qkv_dim;
            uint32_t n_k_heads   = m->ssm_n_k_heads;
            uint32_t n_v_heads   = m->ssm_n_v_heads;
            uint32_t head_k_dim  = m->ssm_head_k_dim;
            uint32_t head_v_dim  = m->ssm_head_v_dim;
            uint32_t conv_kernel = m->ssm_conv_kernel;
            float scale_q = 1.0f / sqrtf((float)head_k_dim);

            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dim, eps);
            }
            gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate, &layer->ssm_beta, &layer->ssm_alpha, g_d_norm_x, dim);
            gpu_delta_fused_gdn_dev((int)delta_layer_idx, layer, g_d_res_x, eps, scale_q, (int)n_k_heads, (int)n_v_heads, (int)head_k_dim, (int)head_v_dim, (int)key_dim, (int)value_dim, (int)qkv_dim, (int)conv_kernel, dim);
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, dim, eps);
            }
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamSynchronize(0));

            // Capture graph on stream 0
            if (g_layer_graphs[l]) {
                CUDA_CHECK(cudaGraphDestroy(g_layer_graphs[l]));
                g_layer_graphs[l] = NULL;
            }

            CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dense_quant_buf_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->attn_norm.dev, dim, eps);
            }
            gpu_delta_projections_dev(&layer->attn_qkv, &layer->attn_gate, &layer->ssm_beta, &layer->ssm_alpha, g_d_norm_x, dim);
            gpu_delta_fused_gdn_dev((int)delta_layer_idx, layer, g_d_res_x, eps, scale_q, (int)n_k_heads, (int)n_v_heads, (int)head_k_dim, (int)head_v_dim, (int)key_dim, (int)value_dim, (int)qkv_dim, (int)conv_kernel, dim);
            if (!tuff_exact() && mmvq_dual_enabled()) {
                gpu_fused_rms_norm_quant_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, norm_x_q8_ensure(dim), dim, eps);
            } else {
                gpu_rms_norm_dev(g_d_norm_x, g_d_res_x, (const float*)layer->post_attention_norm.dev, dim, eps);
            }
            gpu_router_logits_dev_only(&layer->ffn_gate_inp, g_d_norm_x, dim, (int)m->expert_count);
            CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs[l]));

            if (g_layer_execs[l]) {
                cuda_graph_update_exec(&g_layer_execs[l], g_layer_graphs[l]);
            } else {
                CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs[l], g_layer_graphs[l], NULL, NULL, 0));
            }

            delta_layer_idx++;
        }
    }
    g_layer_graphs_ready = 1;
    gpu_reset_delta_state();
    gpu_reset_kv();
}

static unsigned long long g_graph_launches = 0;

void gpu_launch_delta_layer_graph(uint32_t delta_layer_idx) {
    if (g_layer_graphs_ready && delta_layer_idx < 64 && g_layer_execs[delta_layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs[delta_layer_idx], 0));
        g_graph_launches++;
    }
}

void gpu_launch_layer_graph(uint32_t layer_idx) {
    if (g_layer_graphs_ready && layer_idx < 64 && g_layer_execs[layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs[layer_idx], 0));
        g_graph_launches++;
    }
}

unsigned long long gpu_graph_launch_count(void) { return g_graph_launches; }

// ---------------------------------------------------------------------------
// H7: the same capture for the two-column verify trunk.
//
// The verify path issues roughly 50 launches per delta layer (two rms_norms, the
// batched projections, two column publishes, two GDN runs, the state snapshot,
// two rms_norms, two router computes and two shared experts) against the single
// graph launch the one-token path costs. Over 30 delta layers that is ~1500
// launches per verify step; iteration 40 priced ~1000 launches at ~4.5 ms of host
// time, which is the same order as the gap to break-even. Collapsing them is
// ordinary plumbing, and it is the last item between H7 and a positive result.
// ---------------------------------------------------------------------------
static cudaGraph_t     g_layer_graphs2[64];
static cudaGraphExec_t g_layer_execs2[64];
static int             g_layer_graphs2_ready = 0;

extern "C" int gpu_has_delta_graphs_n2(void) { return g_layer_graphs2_ready; }

// One delta layer of the verify trunk, both columns. Used for the dry run (to force
// every lazy allocation) and then captured verbatim.
static void n2_delta_layer_body(model_t *m, model_layer_t *layer, uint32_t delta_layer_idx) {
    const int dim = (int)m->embedding_length;
    const float eps = m->rms_norm_eps;
    const int sh_ffn_dim = (int)m->expert_shared_ffn_len;
    const float scale_q = 1.0f / sqrtf((float)m->ssm_head_k_dim);
    float *d_res2  = g_d_res_x2;
    float *d_norm2 = g_d_norm_x2;
    const float *shexp[2] = { g_d_shexp_out, gpu_get_shexp2_dev_ptr() };

    gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->attn_norm.dev, dim, eps);
    gpu_delta_projections_n2(&layer->attn_qkv, &layer->attn_gate,
                             &layer->ssm_beta, &layer->ssm_alpha, d_norm2, dim);
    for (int c = 0; c < 2; ++c) {
        // H17: no column publish — the fused GDN points straight at the batched projections.
        gpu_delta_fused_gdn_dev_col(c, (int)delta_layer_idx, layer, d_res2 + (size_t)c * dim, eps, scale_q,
                                    (int)m->ssm_n_k_heads, (int)m->ssm_n_v_heads,
                                    (int)m->ssm_head_k_dim, (int)m->ssm_head_v_dim,
                                    (int)m->ssm_key_dim, (int)m->ssm_value_dim,
                                    (int)m->ssm_qkv_dim, (int)m->ssm_conv_kernel, dim);
        if (c == 0) {
            gpu_delta_snapshot_layer_n2((int)delta_layer_idx, (int)m->ssm_n_v_heads,
                                        (int)m->ssm_head_v_dim, (int)m->ssm_head_k_dim,
                                        (int)m->ssm_conv_kernel, (int)m->ssm_qkv_dim);
        }
    }
    gpu_rms_norm_dev_n2(d_norm2, d_res2, (const float*)layer->post_attention_norm.dev, dim, eps);
    for (int c = 0; c < 2; ++c) {
        gpu_router_logits_dev_only_col(&layer->ffn_gate_inp, d_norm2 + (size_t)c * dim,
                                       dim, (int)m->expert_count, c);
    }
    // H15: one weight read for both columns; must match verify_moe_stage_n2's
    // live path exactly or the captured graph and the live path diverge.
    gpu_shared_expert_dev_n2((float*)shexp[0], (float*)shexp[1], layer, d_norm2, dim, sh_ffn_dim);
}

extern "C" void gpu_capture_delta_graphs_n2(model_t *m) {
    if (!m || g_layer_graphs2_ready) return;
    gpu_n2_prepare((int)m->embedding_length, (int)m->vocab_size);

    uint32_t delta_layer_idx = 0;
    for (uint32_t l = 0; l < m->block_count - m->nextn_predict_layers; ++l) {
        model_layer_t *layer = &m->layers[l];
        if (layer->type == LAYER_TYPE_ATTENTION) continue;

        // Dry run: every lazy cudaMalloc on this path must happen before capture.
        n2_delta_layer_body(m, layer, delta_layer_idx);
        CUDA_CHECK(cudaStreamSynchronize(0));

        if (g_layer_graphs2[delta_layer_idx]) {
            CUDA_CHECK(cudaGraphDestroy(g_layer_graphs2[delta_layer_idx]));
            g_layer_graphs2[delta_layer_idx] = NULL;
        }
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed));
        n2_delta_layer_body(m, layer, delta_layer_idx);
        CUDA_CHECK(cudaStreamEndCapture(0, &g_layer_graphs2[delta_layer_idx]));

        if (g_layer_execs2[delta_layer_idx]) {
            cuda_graph_update_exec(&g_layer_execs2[delta_layer_idx], g_layer_graphs2[delta_layer_idx]);
        } else {
            CUDA_CHECK(cudaGraphInstantiate(&g_layer_execs2[delta_layer_idx], g_layer_graphs2[delta_layer_idx], NULL, NULL, 0));
        }
        delta_layer_idx++;
    }
    g_layer_graphs2_ready = 1;
    // The dry runs advanced the recurrent state; put it back.
    gpu_reset_delta_state();
}

extern "C" void gpu_launch_delta_layer_graph_n2(uint32_t delta_layer_idx) {
    if (g_layer_graphs2_ready && delta_layer_idx < 64 && g_layer_execs2[delta_layer_idx]) {
        CUDA_CHECK(cudaGraphLaunch(g_layer_execs2[delta_layer_idx], 0));
        g_graph_launches++;
    }
}

// ---------------------------------------------------------------------------
// Native MTP (Multi-Token Prediction) Speculative Drafting Engine
// ---------------------------------------------------------------------------
static float *g_d_mtp_concat = NULL;
static float *g_d_mtp_x = NULL;
static float *g_d_mtp_norm_x = NULL;
static float *g_d_token_embd = NULL;

__global__ void concat_norms_kernel(float * __restrict__ concat,
                                   const float * __restrict__ enorm,
                                   const float * __restrict__ hnorm,
                                   int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        concat[i] = enorm[i];
        concat[dim + i] = hnorm[i];
    }
}

__global__ void accum_inplace_kernel(float * __restrict__ dst, const float * __restrict__ src, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] += src[i];
    }
}

int32_t gpu_mtp_draft_dev(const model_t *m, float *d_res_x, int32_t token_id, int pos, int max_seq) {
    if (!m || m->nextn_predict_layers == 0 || m->block_count <= 40) return -1;
    const model_layer_t *l40 = &m->layers[40];
    if (!l40->nextn_eh_proj.dev) return -1;

    int dim = (int)m->embedding_length; // 2048
    float eps = m->rms_norm_eps;

    if (!g_d_mtp_concat) {
        CUDA_CHECK(cudaMalloc(&g_d_mtp_concat, 2 * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_mtp_x, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_mtp_norm_x, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_d_token_embd, dim * sizeof(float)));
    }

    // 1. Fetch token embedding for token_id onto device
    float host_embd[2048];
    const uint8_t *embd_base = (const uint8_t*)m->token_embd.data;
    size_t embd_row_bytes = ggml_row_size((ggml_type_t)m->token_embd.info.type, dim);
    dequantize_row((ggml_type_t)m->token_embd.info.type, embd_base + (size_t)token_id * embd_row_bytes, host_embd, dim);
    CUDA_CHECK(cudaMemcpy(g_d_token_embd, host_embd, dim * sizeof(float), cudaMemcpyHostToDevice));

    if (getenv("TUFF_DEBUG_MTP") && pos == 1) {
        fprintf(stderr, "[MTP DEBUG] l40->nextn_enorm.dev=%p, l40->nextn_hnorm.dev=%p, l40->nextn_eh_proj.dev=%p\n",
                l40->nextn_enorm.dev, l40->nextn_hnorm.dev, l40->nextn_eh_proj.dev);
        fprintf(stderr, "[MTP DEBUG] host_embd[0..3]=%.4f %.4f %.4f %.4f\n",
                host_embd[0], host_embd[1], host_embd[2], host_embd[3]);
    }

    if (getenv("TUFF_DEBUG_MTP") && pos == 1) {
        float r[4], nw[4];
        CUDA_CHECK(cudaMemcpy(r, d_res_x, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nw, l40->nextn_hnorm.dev, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[MTP DEBUG] d_res_x[0..3]: %.4f %.4f %.4f %.4f, nextn_hnorm[0..3]: %.4f %.4f %.4f %.4f\n",
                r[0], r[1], r[2], r[3], nw[0], nw[1], nw[2], nw[3]);
    }

    // 2. RMS-norm token embedding and hidden state
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_token_embd, (const float*)l40->nextn_enorm.dev, dim, eps);
    gpu_rms_norm_dev(g_d_mtp_x, d_res_x, (const float*)l40->nextn_hnorm.dev, dim, eps);

    // 3. Concat [enorm, hnorm] -> 4096
    concat_norms_kernel<<<16, 256>>>(g_d_mtp_concat, g_d_mtp_norm_x, g_d_mtp_x, dim);
    CUDA_CHECK_LAST();

    // 4. eh_proj: (4096 -> 2048) -> g_d_mtp_x
    launch_mat_vec_q8_0((const block_q8_0*)l40->nextn_eh_proj.dev, g_d_mtp_concat, g_d_mtp_x, 2 * dim, dim);

    // 5. Layer 40 Attention block
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_mtp_x, (const float*)l40->attn_norm.dev, dim, eps);
    gpu_attention_layer_dev(10, g_d_mtp_x, l40, g_d_mtp_norm_x,
                            (int)m->n_attn_layers, (int)m->head_count, (int)m->head_count_kv,
                            (int)m->key_length, pos, pos + 1, max_seq,
                            1.0f / sqrtf((float)m->key_length), eps, dim,
                            (int)m->rope_dim_count, m->rope_dim_sections, m->rope_freq_base);

    // 6. Layer 40 Post-attention norm + Shared expert
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_mtp_x, (const float*)l40->post_attention_norm.dev, dim, eps);
    gpu_shared_expert_dev(g_d_mtp_concat, l40, g_d_mtp_norm_x, dim, (int)m->expert_shared_ffn_len);
    accum_inplace_kernel<<<8, 256>>>(g_d_mtp_x, g_d_mtp_concat, dim);
    CUDA_CHECK_LAST();

    // 7. Output Norm (shared_head_norm or output_norm)
    const float *head_norm_w = l40->nextn_shared_head_norm.dev ? (const float*)l40->nextn_shared_head_norm.dev : (const float*)m->output_norm.dev;
    gpu_rms_norm_dev(g_d_mtp_norm_x, g_d_mtp_x, head_norm_w, dim, eps);

    // 8. Argmax against output_weight
    int32_t draft_token = gpu_output_argmax_dev(&m->output_weight, g_d_mtp_norm_x, dim, (int)m->vocab_size, NULL);
    return draft_token;
}


// ---------------------------------------------------------------------------
// Pinned host buffers for the per-layer decode round trips (declared in gpu.h).
// cudaMemcpyAsync on pageable memory blocks the calling thread while the driver stages the
// bytes; on pinned memory the DMA engine reads host memory directly. The decode loop performs
// ~3 such copies per layer (norm_x D2H, router logits D2H, CPU-miss accumulator H2D).
void *gpu_host_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess || !p) {
        cudaGetLastError();          // clear the sticky error, then degrade gracefully
        return malloc(bytes);
    }
    return p;
}

void gpu_host_free(void *p) {
    if (!p) return;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, p) == cudaSuccess && attr.type == cudaMemoryTypeHost) {
        cudaFreeHost(p);
    } else {
        cudaGetLastError();
        free(p);
    }
}
