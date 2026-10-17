#ifndef ENGINE_H
#define ENGINE_H

#include "model.h"
#include "cpu_ops.h"
#include "tokenizer.h"
#include "sampler.h"
#include "qwen_cpu_expert.h"

typedef struct {
    model_t *model;

    // Recurrent state for the gated delta-net layers: [n_delta_layers][n_v_heads][head_v_dim][head_k_dim]
    float *delta_state;
    // Conv1d sliding-window state: [n_delta_layers][ssm_conv_kernel][ssm_qkv_dim]
    float *conv_state;

    // KV Cache for the full-attention layers: [n_attn_layers][head_count_kv][max_seq][key_length]
    float *k_cache;
    float *v_cache;
    int max_seq_len;

    // Profiling accumulators (seconds), printed by engine_free when TUFF_PROFILE=1.
    double t_gpu, t_expert, t_delta, t_attn, t_misc, t_total;
    // Fine-grained per-layer accumulators (seconds) for iteration-8 latency profiling.
    double t_gu, t_fill, t_down, t_resid, t_dense, t_miss;   // expert phases + dense
    double t_expert_sync, t_expert_fillwait;
    double t_pref;          // P1: background-fill enqueue + completion poll (was: PCIe miss-fill)
    // P1: CPU-computed expert misses. t_cpumoe_pub is the q8_K quantization of norm_x + the
    // batch publish; t_cpumoe is the join at the end of the MoE stage (i.e. the part of the CPU
    // expert work that did NOT hide behind the GPU).
    double t_cpumoe, t_cpumoe_pub;
    // Iteration-9 delta-branch fine-grained timing.
    double t_d_mat, t_d_conv, t_d_state, t_d_norm, t_d_out;

    // Pre-allocated scratch buffers
    float *x, *norm_x, *moe_buf, *router_logits;
    int   *top_indices;
    float *top_values, *gate_exp, *up_exp, *act_exp, *down_exp, *sh_gate_exp;
    int   *exp_slots;
    float *gate_arr, *up_arr, *act_arr, *down_arr, *down_hit, *sh_up_exp;
    block_q8_K *q8_normx;
    float *sh_act_exp, *sh_down_exp;

    // H7: second-column host buffers for the two-token verify pass. The verify
    // stage drains both columns' router logits with one sync, so each column
    // needs its own landing buffer rather than reusing the ones above.
    float      *norm_x2, *router_logits2, *down_arr2, *moe_buf2;
    block_q8_K *q8_normx2;

    // Speculative MTP/N-gram draft tokens for the current step (up to 4 candidates)
    int32_t last_draft_tokens[4];
    int     last_draft_count;
    int32_t last_draft_token;

    // H17: the accepted token sequence, kept only when the n-gram draft source is active
    // (g_spec_draft != SPEC_DRAFT_MTP). It mirrors engine_prefix_cache_t.tokens for the
    // benchmark loop in main.c, which runs speculative decode without a prefix cache.
    int32_t *hist_tokens;
    int      hist_len;
    int      hist_cap;

    // External background drafter (e.g. Qwen 3.5 0.8B)
    struct drafter_s *drafter;

    // Stochastic sampler state (H60)
    sampler_t sampler;
    float    *host_logits; // Lazy-allocated only when temp > 0

    // Qwen bit-identical CPU miss evaluator
    qwen_cpu_expert_t *qwen_cpu_exp;

    // Batched verify / speculative state (Task Q1.1, Q1.2)
    int      batch_n;
    float   *batch_x;
    float   *router_logits_batch;

    // Lossless two-token verify (engine.c, verify2_*): a pass leaves its expert-cache and
    // recurrent-state effects open until the caller says how many columns it keeps
    // (engine_rollback), or until the next engine call (both kept).
    int      vpend;
    uint64_t vtok0;              // logical expert-cache token of column 0
    struct expert_layer_snap *vsnap;
    float   *d_res_v;            // device [2][dim]: the pass's per-column residual streams
    void    *vev[2];             // per-column router-copy events
    double   t_verify;
    double   tv_trunk, tv_sync, tv_host, tv_pub, tv_hits, tv_pool, tv_comb, tv_head;
    unsigned long long n_verify, n_verify_kept2;
} engine_t;

int engine_forward_batch(engine_t *eng, const int32_t *tokens, int n, int pos0, int32_t *argmax_out, float *logits_out);
int engine_rollback(engine_t *eng, int n_keep);
// 1 when engine_forward_batch(n=2) runs the lossless two-token verify (bit-identical to two
// engine_forward_token calls, expert-cache trajectory included).
int engine_exact_verify_available(const engine_t *eng);

void engine_set_sampler(engine_t *eng, const sampler_params_t *params);

// Set from main.c by --spec-mtp. The MTP draft head is only evaluated when this is non-zero;
// it is a per-token cost (one attention block + shared expert + a second 248320-row output
// projection) that ordinary decode must not pay.
extern int g_spec_mtp;

// H17/H29: --spec-draft selects the draft source; --spec-draft-len sets candidate depth (1..3).
typedef enum {
    SPEC_DRAFT_MTP = 0,
    SPEC_DRAFT_NGRAM = 1,
    SPEC_DRAFT_NGRAM_MTP = 2,
    SPEC_DRAFT_EXTERNAL = 3,
    SPEC_DRAFT_NONE = 4,
} spec_draft_t;
extern int g_spec_draft;
extern int g_spec_draft_len;
extern int g_spec_ngram_n;   // max n-gram order (e.g. 3 or 4) for SPEC_DRAFT_NGRAM*
extern int g_spec_ngram_min; // min n-gram fallback order (e.g. 2) for SPEC_DRAFT_NGRAM*

void engine_set_drafter(engine_t *eng, struct drafter_s *d);

typedef struct {
    int pos;                      // sequence position (tokens processed)
    int32_t *tokens;              // token IDs [pos]
    float   *delta_state;         // host delta state
    size_t   delta_state_bytes;
    float   *conv_state;          // host conv state
    size_t   conv_state_bytes;
    float   *gpu_dstate;          // GPU delta state snapshot
    size_t   gpu_dstate_bytes;
    float   *gpu_dconv;           // GPU conv state snapshot
    size_t   gpu_dconv_bytes;
} engine_state_t;

#define ENGINE_MAX_CHECKPOINTS 64

typedef struct {
    int32_t *tokens;              // currently active token sequence in engine
    int      n_tokens;            // number of tokens currently in engine state (pos)
    int      cap_tokens;
    engine_state_t checkpoints[ENGINE_MAX_CHECKPOINTS];
    int      n_checkpoints;
} engine_prefix_cache_t;

typedef struct {
    int total_prompt_tokens;
    int cached_tokens;
    int new_tokens_prefilled;
    double prefill_time_s;
    double prefill_tok_per_sec;
    int32_t first_generated_token;
} prefill_result_t;

// H7: speculative decoding accounting for one --spec-mtp run.
typedef struct {
    unsigned long long steps;          // verify passes executed
    unsigned long long drafted;        // drafts offered to a verify pass
    unsigned long long accepted;       // drafts the verify pass confirmed
    unsigned long long tokens_emitted; // tokens the loop produced overall
} spec_stats_t;

engine_t* engine_init(model_t *model, int max_seq_len);
int32_t engine_forward_token(engine_t *eng, int32_t token_id, int pos);
int32_t engine_get_last_draft_token(const engine_t *eng);

// Evaluate the trunk at two consecutive positions in one pass. tok0 is the
// accepted token at `pos`, tok1 the draft at `pos+1`. out_next[0] is the
// model's own continuation of tok0 (it decides whether the draft was right);
// out_next[1] continues tok1 and is meaningful only when out_next[0] == tok1.
void engine_forward_verify(engine_t *eng, int32_t tok0, int32_t tok1, int pos, int32_t out_next[2]);

// One speculative step: consumes `token` at `pos`, emits 1 or 2 tokens into
// out_tokens and returns how many. `*next_token` receives the token to feed the
// following step. Falls back to a plain forward when no draft is available.
int engine_spec_step(engine_t *eng, int32_t token, int pos,
                     int32_t out_tokens[4], int32_t *next_token, spec_stats_t *st);
void engine_reset_context(engine_t *eng);
void engine_free(engine_t *eng);

// State checkpointing & Prefix Caching
void engine_state_save_to(const engine_t *eng, int pos, const int32_t *tokens, engine_state_t *st);
void engine_state_restore_from(engine_t *eng, const engine_state_t *st);
void engine_state_free_members(engine_state_t *st);

engine_prefix_cache_t* engine_prefix_cache_create(int max_seq_len);
void engine_prefix_cache_free(engine_prefix_cache_t *cache);
void engine_prefix_cache_reset(engine_prefix_cache_t *cache);
void engine_prefix_cache_save_checkpoint(engine_prefix_cache_t *cache, const engine_t *eng);
prefill_result_t engine_prefill_with_cache(engine_t *eng, engine_prefix_cache_t *cache,
                                          const int32_t *prompt_tokens, int n_prompt);

#endif // ENGINE_H
