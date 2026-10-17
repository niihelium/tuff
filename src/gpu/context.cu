#include "state.h"

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

    gpu_batch_prepare((int)m->embedding_length, (int)m->vocab_size, 8);

    gpu_expert_cache_init_ex(m, p_max_seq, fill_ram);
    gpu_expert_cache_prefill(m);
}

void gpu_prepare(model_t *m) {
    gpu_prepare_ex(m, NULL, 0);
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

    if (g_d_res_batch)        { CUDA_CHECK(cudaFree(g_d_res_batch));        g_d_res_batch = NULL; }
    if (g_d_norm_batch)       { CUDA_CHECK(cudaFree(g_d_norm_batch));       g_d_norm_batch = NULL; }
    if (g_d_nx_batch_q8)      { CUDA_CHECK(cudaFree(g_d_nx_batch_q8));      g_d_nx_batch_q8 = NULL; }
    if (g_d_p_batch_qkv)      { CUDA_CHECK(cudaFree(g_d_p_batch_qkv));      g_d_p_batch_qkv = NULL; }
    if (g_d_p_batch_z)        { CUDA_CHECK(cudaFree(g_d_p_batch_z));        g_d_p_batch_z = NULL; }
    if (g_d_p_batch_b)        { CUDA_CHECK(cudaFree(g_d_p_batch_b));        g_d_p_batch_b = NULL; }
    if (g_d_p_batch_a)        { CUDA_CHECK(cudaFree(g_d_p_batch_a));        g_d_p_batch_a = NULL; }
    if (g_d_logits_batch)     { CUDA_CHECK(cudaFree(g_d_logits_batch));     g_d_logits_batch = NULL; }
    if (g_d_best_token_batch) { CUDA_CHECK(cudaFree(g_d_best_token_batch)); g_d_best_token_batch = NULL; }
    if (g_d_shexp_batch)      { CUDA_CHECK(cudaFree(g_d_shexp_batch));      g_d_shexp_batch = NULL; }
    if (g_d_moe_buf_batch)    { CUDA_CHECK(cudaFree(g_d_moe_buf_batch));    g_d_moe_buf_batch = NULL; }
    for (int i = 0; i < 8; ++i) {
        if (g_dstate_snap_batch[i]) { CUDA_CHECK(cudaFreeHost(g_dstate_snap_batch[i])); g_dstate_snap_batch[i] = NULL; }
        if (g_dconv_snap_batch[i])  { CUDA_CHECK(cudaFreeHost(g_dconv_snap_batch[i]));  g_dconv_snap_batch[i]  = NULL; }
    }
    g_batch_dim = 0;

    gpu_expert_cache_free();
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
