#include "state.h"

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
