// ABI between the qwen35 host engine (C) and the decode megakernel (CUDA).
//
// One persistent cooperative CTA per SM.  Weights are Q8_0 repacked by the engine into
// the kernel's row layout: K bytes of (q + 128) followed by K/32 fp16 scales, so a row is
// one contiguous span and the fp16 scales stay bit-identical to the GGUF.
#ifndef QWEN35_KERNEL_H
#define QWEN35_KERNEL_H

#include <stdint.h>

#define Q35_MAX_LAYERS 24
#define Q35_VOCAB      248320
#define Q35_LIN_H      16
#define Q35_LIN_DK     128
#define Q35_LIN_DV     128
#define Q35_LIN_QK     (Q35_LIN_H * Q35_LIN_DK)      // 2048
#define Q35_LIN_V      (Q35_LIN_H * Q35_LIN_DV)      // 2048
#define Q35_CONV_DIM   (2 * Q35_LIN_QK + Q35_LIN_V)  // 6144
#define Q35_CONV_K     4
#define Q35_CONV_RING  8
#define Q35_GDN_IN_ROWS (Q35_CONV_DIM + Q35_LIN_V + 2 * Q35_LIN_H) // 8224
#define Q35_NQ         8
#define Q35_NKV        2
#define Q35_HD         256
#define Q35_ROT        64
#define Q35_ATT_IN_ROWS (Q35_NQ * 2 * Q35_HD + 2 * Q35_NKV * Q35_HD) // 5120
#define Q35_MAX_SPLITS 42
#define Q35_PART_STRIDE (Q35_HD + 2)
#define Q35_PROJ_MAX   8224
#define Q35_NTMAX      4        // n-tiles in the batch GEMV activation layout (32 entries)

typedef struct {
    const void *w_in;       // GDN: [8224,HID]  attn: [5120,HID]  (q|gate per head, k, v)
    const void *w_out;      // [HID, 2048]
    const void *w_gu;       // [2*INTER, HID], rows interleaved gate0, up0, gate1, up1, ...
    const void *w_down;     // [HID, INTER]
    // Fragment layout for the tensor-core batch GEMV: [group][kt][544], 16-row groups.
    // The gate/up matrix packs rows 0..7 = gate[8g..8g+7], rows 8..15 = up[8g..8g+7].
    const void *w_in_f;
    const void *w_out_f;
    const void *w_gu_f;
    const void *w_down_f;
    const float *in_norm;   // [HID], GGUF values (already 1 + HF offset)
    const float *post_norm; // [HID]
    // GDN only
    const float *conv_w;    // [6144, 4]
    const float *ssm_a;     // [16], already -exp(A_log)
    const float *dt_bias;   // [16]
    const float *gdn_norm;  // [128]
    float *conv_hist;       // [8, 6144] fp32 ring, [ring][channel] (slot 0)
    float *state;           // [16][128 v][128 k] fp32 (slot 0)
    long long conv_stride;  // elements per slot in conv_hist
    long long state_stride; // elements per slot in state
    // attention only
    const float *q_norm;    // [256]
    const float *k_norm;    // [256]
    void *k_cache;          // [2][max_seq][256] f16 (slot 0)
    void *v_cache;          // [2][max_seq][256] f16 (slot 0)
    long long kv_stride;    // elements per slot in k_cache/v_cache
    long long is_attn;
} q35_layer_ptrs_t;

typedef struct {
    q35_layer_ptrs_t layer[Q35_MAX_LAYERS];
    const void *embed;      // tied Q8_0 token_embd / LM head, kernel row layout
    const void *lm_head;    // == embed
    const void *embed_f;    // tied token_embd in the batch GEMV fragment layout
    void *vecf;             // f16 batch-GEMV activations, 32 entries x K, fragment-major
    const float *final_norm;
    const float *rope_cos;  // [max_seq][32]
    const float *rope_sin;
    // Flag-in-data activation buffers.  Each 64-bit word is (epoch << 32) | fp32 bits.
    unsigned long long *xl;     // [HID]
    unsigned long long *projl;  // [8224]
    unsigned long long *corel;  // [2048]
    unsigned long long *attnl;  // [2048]
    unsigned long long *actl;   // [INTER]
    unsigned long long *lml;    // [grid][2]
    float *attn_part;           // [NKV][MAX_SPLITS][4][PART_STRIDE]
    unsigned *attn_cnt;         // [BT][NKV]
    long long max_seq;
    int hid;
    int inter;
    int nslots;
} q35_params_t;

typedef struct {
    const int *forced;
    int n_forced;
    int lm_from;
    int n_steps;
    int start_pos;
    int prof_step;
    unsigned epoch_base;
    unsigned long long *tokens_out;  // [n_steps]
    float *logits_out;               // [n_steps][VOCAB] optional
    unsigned long long *prof;        // optional
    // Batched decode: one entry per slot (decode step).  n_entries <= BT tier.
    const int *slot_ids;
    const int *positions;
    int n_entries;
    float inv_temperature;           // 0 => greedy
    const int *stop_ids;
    int n_stop;
    int hid;
    int *out_info;                   // [2]: steps done, epoch after last op
    unsigned long long *step_ts;     // optional [n_steps+1]
    unsigned seed;
    float *resid_out;                // optional [n_steps][HID]
    unsigned long long *core_dump;
    unsigned long long *x2_dump;     // optional [n_steps][24][HID] after the mixer residual add
    unsigned long long *act_dump;
    float *q_dump;
    unsigned long long *proj_dump;   // optional [n_steps][24][8224]
                   // optional [ctx][4][256] post-RoPE Q of the first attention task
    // optional [n_steps][24][INTER] after SwiGLU
   // optional [n_steps][24][2048] raw recurrence output
} q35_args_t;

#ifdef __cplusplus
extern "C" {
#endif
int  q35_mk_set_params(const q35_params_t *p);
int  q35_mk_launch(const q35_args_t *a, int grid, void *stream);
int  q35_mk_launch_batch(const q35_args_t *a, int bt, int grid, void *stream);
int  q35_mk_smem_bytes(int hid);
int  q35_mk_watchdog_fired(int *blk, unsigned *ep);   // kind of the last watchdog event (0 = none)
#ifdef __cplusplus
}
#endif

#endif
