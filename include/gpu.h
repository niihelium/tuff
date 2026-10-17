#ifndef GPU_H
#define GPU_H

#include "model.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// M2 GPU dense path. All dense weights (Q8_0/F32, not experts/token_embd) are uploaded
// once via gpu_prepare(); gpu_mat_vec() routes any tensor with a device copy to CUDA and
// falls back to mat_vec_cpu() for everything else (experts, token_embd, output). Everything
// is synchronous on the default stream for M2; overlap is a later milestone.

void gpu_prepare(model_t *m);
void gpu_prepare_ex(model_t *m, int *p_max_seq, int fill_ram);
void gpu_shutdown(model_t *m);

// Full round-trip: upload src, launch the row-parallel dequant+dot kernel, download dst.
void gpu_mat_vec(float *dst, const tensor_t *w, const float *src);

// Iteration-9: the four delta-net projections sharing norm_x run as one upload + back-to-back
// launches + one download (bit-identical to four gpu_mat_vec round-trips). P4a: when keep_dev
// is nonzero the four D2H downloads are skipped so g_dpqkv/g_dpz/g_dpb/g_dpa stay
// device-resident for gpu_delta_pre / gpu_delta_post (the host pointers may then be NULL).
void gpu_delta_projections(float *qkv_mixed, const tensor_t *attn_qkv,
                           float *z,        const tensor_t *attn_gate,
                           float *beta_raw, const tensor_t *ssm_beta,
                           float *alpha_raw,const tensor_t *ssm_alpha,
                           const float *norm_x, int in_dim, int keep_dev);

// P4a: device-resident handles to the delta-net tensors produced by gpu_delta_pre (for the
// on-demand TUFF_DUMP_DIR path). `state` is g_dstate_all; the engine adds its layer slice.
typedef struct {
    const float *qkv_conv;  // conv1d+SiLU output, qkv_dim (overwritten in g_dpqkv)
    const float *z;         // value_dim
    const float *q_norm;    // key_dim
    const float *k_norm;    // key_dim
    const float *v_conv;    // value_dim
    const float *decay;     // n_v_heads
    const float *beta;      // n_v_heads
    const float *gate;      // n_v_heads
    const float *state;     // full g_dstate_all
} gpu_delta_dev_t;
void gpu_delta_get_device(gpu_delta_dev_t *out);
void gpu_copy_dev_f32(const void *dev, float *host, size_t n);

// Iteration-9 fused delta out-path: per-head rms_norm + silu(z) gating and the ssm_out mat-vec
// run on device (dnet_out/z stay resident from the state update + projections), downloading only
// attn_out. Pass dnet_out == NULL to gpu_delta_state_update to keep it resident.
void gpu_delta_post(float *attn_out, const tensor_t *ssm_out,
                    const float *ssm_norm_w, float eps,
                    int n_v_heads, int head_v_dim, int value_dim, int out_dim);

int mmvq_dual_enabled(void);

// Iteration-10: batch the three shared-expert projections that share the norm_x input
// (gate_shexp + up_shexp -> sh_ffn_dim, gate_inp_shexp -> scalar) into one upload + launch +
// download burst. Bit-identical to three gpu_mat_vec round-trips.
void gpu_shared_expert_projections(float *sh_gate, const tensor_t *gate_t,
                                   float *sh_up,   const tensor_t *up_t,
                                   float *sh_ginp, const tensor_t *ginp_t,
                                   const float *norm_x, int in_dim);

// Current GPU memory usage (total - free) in MiB, from cudaMemGetInfo.
double gpu_vram_used_mb(void);

// Q4_K expert mat-vec from a device-resident block_q4_K matrix (in_dim x out_dim).
// dev_w points at the first block on device; uses the shared g_dx/g_dy scratch.
void gpu_expert_mat_vec(float *dst, const void *dev_w, const float *src, int in_dim, int out_dim);

// Q6_K expert mat-vec (the routed down experts are Q6_K).
void gpu_expert_mat_vec_q6(float *dst, const void *dev_w, const float *src, int in_dim, int out_dim);

// Pipelined async gate+up: launch on a dedicated stream (returns immediately), then later
// gpu_expert_gateup_wait_download() syncs and fetches both outputs. The caller must have
// downloaded/cancelled the previous launch before re-launching (scratch is single-buffered).
void gpu_expert_gateup_launch(int slot, const float *norm_x, int in_dim, int out_dim);
void gpu_expert_gateup_wait_download(float *gate_dst, float *up_dst, int out_dim);
void gpu_expert_gateup_cancel(void);
void gpu_expert_gateup_sync(void);

// Down expert mat-vec (Q4_K, [in_dim=ffn x out_dim=dim]) from the resident slot on the expert
// stream. Launch after gate/up(k+1) so both overlap; wait+download later.
void gpu_expert_down_launch(int slot, ggml_type_t type, const float *act_exp, int in_dim, int out_dim);
void gpu_expert_down_wait_download(float *down_dst, int out_dim);

// KV Cache quantization modes
enum kv_quant_mode_t {
    KV_QUANT_F32  = 0,
    KV_QUANT_Q8_0 = 1,
    KV_QUANT_Q4_0 = 2
};
void gpu_set_kv_quant(int mode);
int  gpu_get_kv_quant(void);

// Batched async expert pipeline: upload norm_x to g_exgx first, then launch all gate/up (HIT
// experts only, slots[k]>=0) and all down kernels on the expert stream, syncing+downloading
// once for gate/up and once for down per layer. `slots` is the per-position slot table.
void gpu_expert_batch_gateup_launch(const int *slots, int n, const float *norm_x, int in_dim, int out_dim);
void gpu_expert_batch_gateup_sync_download(float *gate_out, float *up_out, int n, int out_dim);
void gpu_expert_batch_down_launch(int k, int slot, ggml_type_t type, const float *act_exp, int in_dim, int out_dim);
void gpu_expert_batch_down_sync_download(const int *slots, float *down_out, int n, int out_dim);
void gpu_expert_batch_sync(void);

// Fused batch launches: ONE kernel per tensor (gate, up, down) computes ALL selected experts'
// outputs for a layer (vs n*2 gate/up + n down separate 1-thread-per-row launches). Bit-identical
// to the per-expert kernels (each output element still computed by a single scalar thread).
void gpu_expert_batch_gateup_launch_fused(const int *slots, int n, const float *norm_x, int in_dim, int out_dim);
void gpu_expert_batch_down_launch_fused(int n, ggml_type_t type, const float *act_all, int in_dim, int out_dim);

// P3 (parity.md sec.4): tuff_exact() (include/tuff_exact.h, TUFF_EXACT=1) keeps the
// original bit-exact scalar expert kernels (1 thread per output row, no activation
// quantization). Default (0) uses the llama.cpp mmvq port: q8_1 activations + __dp4a +
// warp-per-row cooperation, which is NOT bit-identical to the scalar path by design.
//
// P3b GLU-fused routed-expert stage (parity.md sec.2.4b). gateup_swiglu computes gate+up and
// applies SwiGLU in the kernel epilogue, leaving the activation device-resident and already
// quantized to q8_1; moe_down then runs the down projection from it and downloads down_out
// [n][out_dim]. Replaces gateup_launch_fused + host swiglu() + down_launch_fused, removing two
// PCIe transfers and the host stage per layer. Not available under TUFF_EXACT=1.
void gpu_expert_moe_gateup_swiglu(const int *slots, int n, const float *norm_x, int in_dim, int ffn_dim);
void gpu_expert_moe_down(int n, ggml_type_t type, int in_dim, int out_dim, float *down_out);

void gpu_qwen_moe_swiglu_pre_gu(cudaStream_t stream, const float *d_pre_gu,
                                float *d_act, void *d_act_q8,
                                unsigned miss_mask, int ffn_dim);
void gpu_qwen_moe_combine_unified(float *d_res_x, const float *exdbat,
                                  const int *hit_pos, int n_hits,
                                  const float *d_pre_y, unsigned miss_mask,
                                  const float *top_values, const float *d_shexp_out,
                                  int dim, cudaStream_t stream);

const void* gpu_get_expert_xq8_dev_ptr(int col);
float*      gpu_get_expert_dx_dev_ptr(int col);
void*       gpu_get_expert_dxq8_dev_ptr(int col);
float*      gpu_get_expert_dbat_dev_ptr(int col);
void        gpu_expert_moe_normx_quant_col(const float *d_norm_x, int in_dim, int col);
void        gpu_expert_moe_normx_quant_host(const float *h_norm_x, int in_dim, int col);
cudaStream_t gpu_get_expert_stream(void);




// Self-test of the Q4_K expert kernel vs the CPU reference (TUFF_EXPERT_SELFTEST=1).
void gpu_selftest_q4k(const model_t *m);
// H7 stage 1: cost of a dense Q8_0 projection over 1/2/4 activation columns.
void gpu_bench_dense_batch(const tensor_t *w, int in_dim, int out_dim, int reps);
void gpu_bench_head_batch(const tensor_t *w, int in_dim, int out_dim, int reps);

// --- M4 per-expert Q4_K LFRU VRAM cache ---
// Allocated/freed by gpu_prepare/gpu_shutdown. The engine calls gpu_expert_lookup() per
// selected expert: >=0 is a HIT (run gpu_expert_mat_vec on the resident slot); -1 is a MISS
// (run the CPU path, then gpu_expert_cache_alloc_slot()+gpu_expert_cache_fill() so future
// tokens hit). Disable entirely with TUFF_CPU_EXPERT=1. TUFF_EXPERT_SLOTS overrides
// the slot count. Fill uploads are always async on a dedicated stream, synced before the batch
// launch reads the slots.
void gpu_expert_cache_init(model_t *m);
void gpu_expert_cache_init_ex(model_t *m, int *p_max_seq, int fill_ram);
void gpu_expert_cache_free(void);

// Iteration-12 static expert split: the number of layers whose FULL expert sets are made
// resident in VRAM (pre-filled at gpu_prepare, never missing). Layers >= this count run their
// routed experts on the CPU (overlapped with GPU dense work). 0 disables the split (legacy
// all-LFRU behavior). Set by main.c from --n-gpu-expert-layers before gpu_prepare.
extern int g_n_gpu_expert_layers;
void gpu_expert_cache_prefill(model_t *m);
int  gpu_expert_lookup(uint32_t layer, int expert);
const void *gpu_expert_dev_ptr(int slot, int which); // which: 0=gate 1=up 2=down
int  gpu_expert_cache_alloc_slot(uint32_t layer, int expert);
void gpu_expert_cache_fill(int slot, const model_t *m, uint32_t layer, int expert);
void gpu_expert_cache_fill_gateup(int slot, const model_t *m, uint32_t layer, int expert);
void gpu_expert_cache_fill_down(int slot, const model_t *m, uint32_t layer, int expert);
void gpu_expert_cache_stats(unsigned long long *hits, unsigned long long *misses);
int  gpu_expert_cache_active(void);
void gpu_expert_cache_sync(void);
struct moe_policy* gpu_get_moe_policy(void);
int  gpu_expert_is_resident(uint32_t layer, int expert);
int  gpu_expert_cache_les_prefetch(const model_t *m);

// --- P1 background cache fills (parity.md sec.4 P1) ---
// The decode thread must never copy expert weights itself: the GGUF mapping is pageable, so
// cudaMemcpyAsync stages it through a driver bounce buffer and blocks the caller (measured
// 21.0 ms/token). gpu_expert_cache_bgfill() only *reserves* a slot (O(1) host bookkeeping,
// the slot is made unreachable to lookups and to further reservations) and queues the copy on
// a dedicated fill thread; it returns 1 if the fill was queued, 0 if the queue is full or no
// slot can be spared. gpu_expert_cache_bgfill_poll() is the (non-blocking) completion side:
// it cudaEventQuery()s finished fills and publishes their slots into the lookup map. Neither
// call ever waits on PCIe. See the block comment in gpu.cu for the race-freedom argument.
int  gpu_expert_cache_bgfill(const model_t *m, uint32_t layer, int expert);
// Lossless speculation (per-layer cache partition, see expert_cache.cu): the logical token clock
// every cache operation is stamped with, the split logical/physical fill, GPU-side readiness
// waits, and per-layer snapshots for undoing a rejected verify column.
void     gpu_expert_cache_set_token(uint64_t tok);
uint64_t gpu_expert_cache_get_token(void);
void     gpu_expert_cache_token_end(void);
void     gpu_expert_cache_layer_begin(uint32_t layer);
int      gpu_expert_cache_fill_reserve(const model_t *m, uint32_t layer, int expert, int group);
int      gpu_expert_cache_resolve_hit_slot(const model_t *m, uint32_t layer, int slot, int expert);
void     gpu_expert_cache_fill_push(int group);
void     gpu_expert_cache_fill_commit(int group);
void     gpu_expert_cache_fill_drop(int group);
void     gpu_expert_cache_wait_slots(const int *slots, int n);
unsigned long long gpu_expert_cache_les_rejects(void);
void     gpu_expert_cache_wait_stats(double *s, unsigned long long *n);
void     gpu_expert_cache_streamwait_stats(unsigned long long *n, unsigned long long *gap, unsigned long long *behind);
typedef struct expert_layer_snap expert_layer_snap_t;
expert_layer_snap_t *gpu_expert_cache_snap_alloc(void);
void     gpu_expert_cache_snap_layer(expert_layer_snap_t *sn, uint32_t layer);
void     gpu_expert_cache_restore_layer(const expert_layer_snap_t *sn, uint32_t layer);
void     gpu_expert_cache_snap_free(expert_layer_snap_t *sn);
void gpu_expert_cache_bgfill_poll(void);
void gpu_expert_cache_bgfill_poll_ready(void);
void gpu_expert_cache_bgfill_stats(unsigned long long *issued, unsigned long long *committed);
int  gpu_expert_debug_check_down(int slot, const void *host_ref, size_t nbytes);

// Zero the persistent device delta-net state. Called from engine_init so each engine run
// (which allocates a fresh zeroed host engine) starts from a zeroed recurrent state. No-op
// if gpu_prepare has not been called or the model has no delta layers.
void gpu_reset_delta_state(void);
void gpu_get_recurrent_state_sizes(size_t *dstate_bytes_out, size_t *dconv_bytes_out);
void gpu_save_recurrent_state(float *dstate_host_out, float *dconv_host_out);
void gpu_restore_recurrent_state(const float *dstate_host_in, const float *dconv_host_in);

// Iteration-13: run the full-attention layer's per-head scoring / softmax / weighted-V on GPU,
// keeping the KV cache on device (allocated lazily in gpu_attention, freed in gpu_shutdown).
// q is [n_head][head_dim], k/v are [n_head_kv][head_dim] (post-RMSNorm + post-mrope), the result
// concat_out [n_head][head_dim] is downloaded. The gate sigmoid multiply is applied on the host.
// Bit-identical to engine.c's CPU scalar per-head loop. Disable with TUFF_CPU_ATTN=1.
void gpu_attention(int layer_idx, float *concat_out,
                   const float *q, const float *k, const float *v,
                   int n_attn_layers, int n_head, int n_head_kv, int head_dim,
                   int pos, int ctx_len, int max_seq, float scale);

// Zero the persistent device KV caches. Called from engine_init per engine run.
void gpu_reset_kv(void);

// Offload the gated delta-net per-head recurrent state update (decay, sk, d, S update, oh)
// for delta layer layer_idx to CUDA. The state lives persistently on device (allocated in
// gpu_prepare), so only the small per-token inputs (q_norm/k_norm/v_conv/decay/beta) are
// uploaded and dnet_out [n_v_heads][head_v_dim] is downloaded. decay[] must equal
// expf(gate[h]) computed on the host so the pure-FMA kernel is bit-identical to the CPU
// reference. q_norm/k_norm are [n_k_heads][head_k_dim], v_conv is [n_v_heads][head_v_dim].
void gpu_delta_state_update(int layer_idx, float *dnet_out,
                            const float *q_norm, const float *k_norm,
                            const float *v_conv, const float *decay, const float *beta,
                            float scale_q,
                            int n_v_heads, int n_k_heads,
                            int head_v_dim, int head_k_dim);

// P4a: device-resident delta pre-stage. Runs the conv-state shift + conv1d + SiLU, per-head
// l2_norm, and the sigmoid/softplus/decay gate stage on GPU, consuming g_dpqkv/g_dpb/g_dpa
// (kept resident by gpu_delta_projections keep_dev=1) and producing q_norm/k_norm/v_conv/
// decay/beta directly in device scratch for the state-update kernel.
void gpu_delta_pre(int layer_idx, const model_layer_t *L, float eps,
                   int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                   int key_dim, int value_dim, int qkv_dim, int conv_kernel);

// P4a: device-pointer variant of gpu_delta_state_update. q_norm/k_norm/v_conv/decay/beta are
// read from the persistent device scratch g_dq/g_dk/g_dv/g_dp (produced by gpu_delta_pre), so
// the five H2D uploads disappear. dnet_out==NULL keeps the result resident for gpu_delta_post.
void gpu_delta_state_update_dev(int layer_idx, float *dnet_out, float scale_q,
                                int n_v_heads, int n_k_heads,
                                int head_v_dim, int head_k_dim);

// ===========================================================================
// P4b: Device-resident residual stream & fused fast path.
// The residual stream x and normalized norm_x live on device throughout decode.
// ===========================================================================
void gpu_residual_reset(const float *host_x, int dim);
void gpu_rms_norm_dev(float *d_dst, const float *d_src, const float *d_weight, int dim, float eps);
void gpu_fused_rms_norm_quant_dev(float *d_dst_norm_x, const float *d_src, const float *d_weight, void *d_dst_q8, int dim, float eps);
void *gpu_get_d_dense_x_q8(void);
void *gpu_get_d_norm_q8(void);
void gpu_set_norm_q8_valid(int v);
void gpu_residual_add(float *d_res_x, const float *d_delta, int dim);
const float *gpu_get_d_norm_x(void);
float *gpu_get_d_res_x(void);

// Device delta projections from device norm_x
void gpu_delta_projections_dev(const tensor_t *attn_qkv,
                               const tensor_t *attn_gate,
                               const tensor_t *ssm_beta,
                               const tensor_t *ssm_alpha,
                               const float *d_norm_x, int in_dim);

// Fused delta post accumulating directly into device residual stream
void gpu_delta_post_accum(float *d_res_x, const tensor_t *ssm_out,
                          const float *ssm_norm_w, float eps,
                          int n_v_heads, int head_v_dim, int value_dim, int out_dim);

// Fused gated delta-net autoregressive step (H5)
void gpu_delta_fused_gdn_dev(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                             int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                             int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);

// Full attention layer end-to-end on device
void gpu_attention_layer_dev(int layer_idx, float *d_res_x, const model_layer_t *layer,
                             const float *d_norm_x, int n_attn_layers,
                             int n_head, int n_head_kv, int head_dim,
                             int pos, int ctx_len, int max_seq, float scale, float eps, int dim,
                             int rope_dim_count, const uint32_t rope_sections[4], float rope_freq_base);

// Router logits projection and norm_x copy from device to host
void gpu_router_logits_dev(float *host_logits, float *host_norm_x, const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert);
void gpu_router_logits_launch(float *host_logits, float *host_norm_x, const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert);
void gpu_router_topk_launch(int *host_top_indices, float *host_top_values, float *host_norm_x,
                            const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert);
void gpu_router_logits_dev_only(const tensor_t *ffn_gate_inp, const float *d_norm_x, int in_dim, int n_expert);
void gpu_router_logits_wait(void);
// H6: event-scoped wait (router logits + norm_x D2H only, not the whole stream).
void gpu_router_logits_wait_event(void);

// CUDA Graph execution for all 40 layers (Delta-Net + Attention)
int  gpu_has_delta_graphs(void);
void gpu_set_device_pos(int pos);
void gpu_capture_delta_graphs(model_t *m);
void gpu_launch_delta_layer_graph(uint32_t delta_layer_idx);
void gpu_launch_layer_graph(uint32_t layer_idx);
unsigned long long gpu_graph_launch_count(void);
// H12: D2H of the router logits and norm_x for layers whose router compute is inside a graph.
void gpu_router_logits_copy_out(float *host_logits, float *host_norm_x,
                                const float *d_norm_x, int in_dim, int n_expert);
// H25: D2H of top-8 indices, top-8 values, and norm_x for layers whose router compute is inside a graph.
void gpu_router_topk_copy_out(int *host_top_indices, float *host_top_values, float *host_norm_x,
                              const float *d_norm_x, int in_dim);

// Shared expert projection + SwiGLU + down + scale on device
void gpu_shared_expert_dev(float *d_shexp_out, const model_layer_t *layer, const float *d_norm_x, int in_dim, int sh_ffn_dim);
// H15: both columns of the shared expert with one weight read (ncols=2 dense Q8_0
// launcher). Column c is bit-identical to gpu_shared_expert_dev on column c alone.
void gpu_shared_expert_dev_n2(float *d_shexp_out0, float *d_shexp_out1, const model_layer_t *layer,
                              const float *d_norm_x2, int in_dim, int sh_ffn_dim);
const float *gpu_get_shexp_dev_ptr(void);
const float *gpu_get_d_top_values(void);
const int   *gpu_get_d_exslots(void);
void gpu_copy_dev_i32(const int *dev_src, int *host_dst, int n);

// Device routed experts (GLU-fused gate/up + down without host round-trip)
void gpu_expert_moe_gateup_swiglu_dev(const int *slots, int n, const float *d_norm_x, int in_dim, int ffn_dim);
// H7: same three stages against column `col` of the doubled expert scratch, so a
// verify pass can leave both tokens' GPU expert work in flight across the CPU
// pool join. col=0 is exactly the single-token behaviour.
void gpu_expert_moe_gateup_swiglu_dev_col(const int *slots, int n, const float *d_norm_x, int in_dim, int ffn_dim, int col);
void gpu_expert_moe_down_dev_col(int n, ggml_type_t type, int in_dim, int out_dim, int col);
void gpu_moe_combine_accum_col(float *d_res_x, const int *hit_pos, int n_hits,
                               const float *top_values, const float *d_shexp_out,
                               const float *host_miss_sum, int dim, int col);
void gpu_expert_moe_down_dev(int n, ggml_type_t type, int in_dim, int out_dim);
// H15: partitioned two-column routed experts. The shared set runs ncols=2 kernels
// (one weight read, both columns); the exclusive sets run the single-column path
// with a destination-index table so the compressed slot lists still land each
// output at its original hit position. Bit-identical per column to the single
// column path. `dst` arrays map slot-list index -> hit position (NULL = identity).
void gpu_expert_moe_gateup_swiglu_dev_n2_shared(const int *slots, const int *dst0, const int *dst1, int n,
                                                const float *d_norm_x0, const float *d_norm_x1,
                                                int in_dim, int ffn_dim);
void gpu_expert_moe_gateup_swiglu_dev_col_dst(const int *slots, const int *dst, int n,
                                              const float *d_norm_x, int in_dim, int ffn_dim, int col);
void gpu_expert_moe_act_quant_col(int n, int ffn_dim, int col);
void gpu_expert_moe_down_dev_n2_shared(const int *slots, const int *dst0, const int *dst1, int n,
                                       ggml_type_t type, int in_dim, int out_dim);
void gpu_expert_moe_down_dev_col_dst(const int *slots, const int *dst, int n,
                                     ggml_type_t type, int in_dim, int out_dim, int col);

// Fused MoE combine and residual add on device
void gpu_moe_combine_accum(float *d_res_x, const int *hit_pos, int n_hits,
                           const float *top_values, const float *d_shexp_out,
                           const float *host_miss_sum, int dim);

// Autonomous device MoE execution for VRAM-resident layers
void gpu_expert_resident_layer_dev(int layer_idx, ggml_type_t down_type, const float *d_norm_x, float *d_res_x, const float *d_shexp_out, int in_dim, int ffn_dim, int n_expert);

// Device argmax reduction over vocab_size logits
int32_t gpu_output_argmax_dev(const tensor_t *output_weight, const float *d_norm_x, int in_dim, int vocab_size, float *host_logits_capture);
int32_t gpu_fused_output_norm_quant_argmax_dev(const tensor_t *output_weight, const float *d_res_x, const float *d_norm_w, int in_dim, int vocab_size, float eps, float *host_logits_capture);

// Materialize vocab_size logits for stochastic sampling (H60)
void gpu_output_logits_dev(const tensor_t *output_weight, const float *d_norm_x, int in_dim, int vocab_size, float *host_logits);
void gpu_fused_output_norm_quant_logits_dev(const tensor_t *output_weight, const float *d_res_x, const float *d_norm_w, int in_dim, int vocab_size, float eps, float *host_logits);

// Native MTP (Multi-Token Prediction) Speculative Drafting Engine
int32_t gpu_mtp_draft_dev(const model_t *m, float *d_res_x, int32_t token_id, int pos, int max_seq);

// --- H7 stage 2b: the two-column verify trunk ---
// The residual/normalized streams become [2][dim] with column c at base + c*dim,
// so every single-column entry point above still applies per column. Only the
// shared-weight reads get batched launchers: the delta-layer input projections
// and the Q6_K output head. See the block comment in gpu.cu.
void  gpu_n2_prepare(int dim, int vocab_size);
float *gpu_get_d_res_x2(void);
float *gpu_get_d_norm_x2(void);
void  gpu_residual_reset_n2(const float *h_x0, const float *h_x1, int dim);
void  gpu_rms_norm_dev_n2(float *d_dst2, const float *d_src2, const float *d_weight, int dim, float eps);
void  gpu_n2_quantize_norm_x(const float *d_src2, int in_dim);
void  gpu_delta_projections_n2(const tensor_t *attn_qkv, const tensor_t *attn_gate,
                               const tensor_t *ssm_beta, const tensor_t *ssm_alpha,
                               const float *d_norm_x2, int in_dim);
// H17: run the fused GDN step for column c straight off the batched projections. Replaces
// the old gpu_delta_select_column_n2 + gpu_delta_fused_gdn_dev pair and its 4 D2D column
// publishes per layer (240 per verify step); column c's qkv is consumed in place by the
// conv and never needed again.
void  gpu_delta_fused_gdn_dev_col(int c, int layer_idx, const model_layer_t *L, float *d_res_x,
                                  float eps, float scale_q,
                                  int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                  int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);
void  gpu_output_argmax_n2(const tensor_t *output_weight, const float *d_norm_x2,
                           int in_dim, int vocab_size, int32_t *out_tokens2);
// Snapshot taken between the two columns of a layer (state as of the accepted
// token); gpu_delta_rollback_n2 restores it when the draft is rejected.
void  gpu_delta_snapshot_layer_n2(int layer_idx, int n_v_heads, int head_v_dim,
                                  int head_k_dim, int conv_kernel, int qkv_dim);
void  gpu_delta_rollback_n2(void);
void  gpu_router_logits_dev_only_col(const tensor_t *ffn_gate_inp, const float *d_norm_x,
                                     int in_dim, int n_expert, int col);
void  gpu_router_logits_copy_out_n2(float *logits0, float *nx0, float *logits1, float *nx1,
                                    const float *d_norm_x2, int in_dim, int n_expert);
// Two-column delta-layer graphs: ~50 launches per layer collapse into one.
int   gpu_has_delta_graphs_n2(void);
// Lossless two-token verify (see graphs.cu): no-write delta graphs for column 1 and the commit
// replay of an accepted column 1's recurrent-state updates.
void  gpu_capture_verify_graphs(model_t *m);
int   gpu_has_verify_graphs(void);
void  gpu_launch_layer_graph_nw(uint32_t layer_idx);
void  gpu_launch_verify_commit(void);
void  gpu_launch_verify_commit_slot(int slot);
void  gpu_verify_set_positions(int pos0, int n);
void  gpu_verify_select_pos(int c);
void  gpu_memcpy_d2d(void *dst, const void *src, size_t bytes);
void  gpu_memcpy_h2d(void *dst, const void *src, size_t bytes);
void  gpu_capture_verify_col_graphs(model_t *m);
int   gpu_has_verify_col_graphs(void);
void  gpu_launch_verify_col(int c, uint32_t l);
float *gpu_verify_host_rl(int c);
float *gpu_verify_host_nx(int c);
void  gpu_fused_output_argmax_async(const tensor_t *output_weight, const float *d_res_x, const float *d_norm_w,
                                    int in_dim, int vocab_size, float eps, int32_t *h_tok);
int32_t *gpu_pinned_tokens(void);
void  gpu_stream_sync(void);
void  gpu_capture_lockstep_graphs(model_t *m);
int   gpu_has_lockstep_graphs(void);
void  gpu_launch_lockstep(uint32_t l);
void  gpu_launch_lockstep_n(int ncols, uint32_t l);
float *gpu_lockstep_host_rl(int c);
float *gpu_lockstep_host_nx(int c);
float *gpu_lockstep_shexp(int c);
void  gpu_lockstep_output_argmax_async_n(const tensor_t *output_weight, float * const res[4],
                                         const float *d_norm_w, int in_dim, int vocab_size,
                                         float eps, int ncols, int32_t *h_tok);
void *gpu_event_new(void);
void  gpu_event_record(void *ev);
void  gpu_event_sync(void *ev);
float *gpu_get_d_res_verify(int dim);
void  gpu_expert_save_norm_q8_col(int in_dim, int col);
void  gpu_expert_moe_gateup_swiglu_dev_saved(const int *slots, int n, int in_dim, int ffn_dim, int col);
void  gpu_delta_fused_gdn_dev_nw(int layer_idx, const model_layer_t *L, float *d_res_x, float eps, float scale_q,
                                 int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                 int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);
void  gpu_delta_commit_nw(int layer_idx, const model_layer_t *L, float eps, float scale_q,
                          int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                          int key_dim, int value_dim, int qkv_dim, int conv_kernel);
void  gpu_delta_commit_nw_slot(int slot, int layer_idx, const model_layer_t *L, float eps, float scale_q,
                               int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                               int key_dim, int value_dim, int qkv_dim, int conv_kernel);
void  gpu_capture_delta_graphs_n2(model_t *m);
void  gpu_launch_delta_layer_graph_n2(uint32_t delta_layer_idx);
const float *gpu_get_shexp2_dev_ptr(void);

// Generalised N-column batch verify operations (Task Q1.1, 1 <= N <= 8)
void  gpu_batch_prepare(int dim, int vocab_size, int max_batch);
float *gpu_get_d_res_batch(void);
float *gpu_get_d_norm_batch(void);
float *gpu_get_d_shexp_batch(void);
float *gpu_get_d_moe_buf_batch(void);
void  gpu_upload_moe_buf(int c, const float *h_buf, int dim);
void  gpu_residual_reset_batch(const float *h_x_all, int dim, int n);
void  gpu_residual_accum(float *d_dst, const float *d_src, int count);
void  gpu_rms_norm_dev_batch(float *d_dst, const float *d_src, const float *d_weight, int dim, float eps, int n);
void  gpu_batch_quantize_norm_x(const float *d_src, int in_dim, int n);
void  gpu_delta_projections_batch(const tensor_t *attn_qkv, const tensor_t *attn_gate,
                                  const tensor_t *ssm_beta, const tensor_t *ssm_alpha,
                                  const float *d_norm_batch, int in_dim, int n);
void  gpu_delta_fused_gdn_dev_col_batch(int c, int layer_idx, const model_layer_t *L, float *d_res_x,
                                        float eps, float scale_q,
                                        int n_k_heads, int n_v_heads, int head_k_dim, int head_v_dim,
                                        int key_dim, int value_dim, int qkv_dim, int conv_kernel, int dim);
void  gpu_delta_snapshot_layer_batch(int layer_idx, int col, int n_v_heads, int head_v_dim,
                                     int head_k_dim, int conv_kernel, int qkv_dim);
void  gpu_delta_rollback_batch(int n_keep);
void  gpu_output_argmax_batch(const tensor_t *output_weight, const float *d_norm_batch,
                              int in_dim, int vocab_size, int n, int32_t *out_tokens,
                              float *host_logits_capture);

// Task Q1.2: Batched router logits and expert union across N columns
void  gpu_router_logits_batch(float *host_logits, const tensor_t *ffn_gate_inp,
                              const float *d_norm_batch, int in_dim, int n_expert, int n);
void  gpu_expert_moe_union_dev(const int *slots, const int *m_arr, const int *cols, const int *dst,
                               int n_unique, const float *d_norm_batch, ggml_type_t down_type,
                               int in_dim, int ffn_dim, int n_cols);
void  gpu_expert_moe_download_down_hits(float *host_dst, int n_cols, int dim);

// Page-locked host allocation for the per-layer decode round trips (norm_x, router logits, the
// CPU-miss accumulator). A pageable cudaMemcpyAsync is not asynchronous: the driver stages it
// through an internal pinned buffer and the calling thread blocks. These buffers are touched
// ~3x per layer, 40 layers per token. Falls back to malloc if pinning fails.
void *gpu_host_alloc(size_t bytes);
void  gpu_host_free(void *p);

#ifdef __cplusplus
}
#endif

#endif // GPU_H
