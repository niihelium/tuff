// Standalone test for lfm25_gpu_moe_ffn against llama.cpp-derived fixtures.
// Usage: lfm25-moe-ffn-test <model.gguf> <fixtures.bin>
#include "lfm25_model.h"
#include "lfm25_engine.h"
#include "cuda_check.h"
#include "ggml_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

// Link stub: cpu_ops.c references tuff_exact(), which normally lives in main.c.
extern "C" int tuff_exact(void) { return 0; }

static int read_exact(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n ? 0 : -1; }

static double max_abs(const std::vector<float> &v) {
    double m = 0; for (float x : v) m = fmax(m, fabs((double)x)); return m;
}
static double rel_err(const std::vector<float> &got, const std::vector<float> &ref) {
    double d = 0;
    for (size_t i = 0; i < ref.size(); i++) d = fmax(d, fabs((double)got[i] - ref[i]));
    double m = max_abs(ref);
    return m > 0 ? d / m : d;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model.gguf> <fixtures.bin>\n", argv[0]); return 2; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    char magic[8]; uint32_t hdr[4];
    if (read_exact(f, magic, 8) || read_exact(f, hdr, sizeof(hdr)) || memcmp(magic, "LFMMOE01", 8) != 0) {
        fprintf(stderr, "bad fixture header/magic\n"); return 2;
    }
    const uint32_t n_cases = hdr[0], dim = hdr[1], ffn = hdr[2], K = hdr[3];
    if (dim != 2048 || (ffn != 1792 && ffn != 1536) || K != 4) {
        fprintf(stderr, "unexpected fixture dims: dim=%u ffn_dim=%u k=%u\n", dim, ffn, K); return 2;
    }

    lfm25_model_t *model = lfm25_model_load(argv[1]);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    if (model->embedding_length != dim || model->expert_ffn_dim != ffn) {
        fprintf(stderr, "model/fixture mismatch: dim=%u expert_ffn_dim=%u\n", model->embedding_length, model->expert_ffn_dim); return 2;
    }
    const bool pool_mode = model->expert_count > 32;
    int ctx = 256;
    if (lfm25_gpu_prepare(model, &ctx, 0) != 0) { fprintf(stderr, "lfm25_gpu_prepare failed\n"); return 2; }

    // Pool mode (64 experts): 4 slots each for gate / up / down; down slot sized for the larger of Q4_K/Q6_K.
    auto row_b = [](const lfm25_tensor_t &t, int64_t n) { return ggml_row_size((ggml_type_t)t.info.type, n); };
    size_t gate_slot = 0, up_slot = 0, down_slot = 0;
    char *d_pg = nullptr, *d_pu = nullptr, *d_pd = nullptr;
    if (pool_mode) {
        for (uint32_t l = 0; l < model->block_count; l++) {
            const lfm25_layer_t *L = &model->layers[l];
            if (L->ffn_type != LFM25_FFN_MOE) continue;
            gate_slot = std::max(gate_slot, (size_t)ffn * row_b(L->ffn_gate_exps, dim));
            up_slot   = std::max(up_slot,   (size_t)ffn * row_b(L->ffn_up_exps, dim));
            down_slot = std::max(down_slot, (size_t)dim * row_b(L->ffn_down_exps, ffn));
        }
        CUDA_CHECK(cudaMalloc(&d_pg, 4 * gate_slot));
        CUDA_CHECK(cudaMalloc(&d_pu, 4 * up_slot));
        CUDA_CHECK(cudaMalloc(&d_pd, 4 * down_slot));
    }

    float *d_x, *d_act, *d_dst, *d_tw; int *d_ids; void *d_xq8, *d_actq8;
    CUDA_CHECK(cudaMalloc(&d_x, dim * 4));
    CUDA_CHECK(cudaMalloc(&d_act, (size_t)K * ffn * 4));
    CUDA_CHECK(cudaMalloc(&d_dst, dim * 4));
    CUDA_CHECK(cudaMalloc(&d_tw, K * 4));
    CUDA_CHECK(cudaMalloc(&d_ids, K * 4));
    CUDA_CHECK(cudaMalloc(&d_xq8, (size_t)(dim / 32) * 36));
    CUDA_CHECK(cudaMalloc(&d_actq8, (size_t)(K * ffn / 32) * 36));

    std::vector<float> x(dim), wref(K), actref((size_t)K * ffn), outref(dim), actgot((size_t)K * ffn), outgot(dim);
    int32_t idref[4];
    uint32_t pass = 0;
    double max_act = 0, max_out = 0;
    bool accum_ok = true;

    for (uint32_t c = 0; c < n_cases; c++) {
        uint32_t ls[2];
        if (read_exact(f, ls, sizeof(ls)) || read_exact(f, x.data(), dim * 4) || read_exact(f, idref, K * 4) ||
            read_exact(f, wref.data(), K * 4) || read_exact(f, actref.data(), (size_t)K * ffn * 4) ||
            read_exact(f, outref.data(), dim * 4)) {
            fprintf(stderr, "truncated fixture at case %u\n", c); return 2;
        }
        const uint32_t layer = ls[0], step = ls[1];
        if (layer >= model->block_count || model->layers[layer].ffn_type != LFM25_FFN_MOE) {
            fprintf(stderr, "case %u: layer %u is not MoE\n", c, layer); return 2;
        }
        const lfm25_layer_t *L = &model->layers[layer];
        const int down_type = (int)L->ffn_down_exps.info.type;

        CUDA_CHECK(cudaMemcpy(d_x, x.data(), dim * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids, idref, K * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tw, wref.data(), K * 4, cudaMemcpyHostToDevice));
        lfm25_gpu_quantize_q8_1(d_x, d_xq8, (int)dim);
        const void *g_ptr, *u_ptr, *d_ptr; size_t gs, us, ds; const int *ids_used = d_ids;
        if (!pool_mode) {
            g_ptr = L->ffn_gate_exps.dev; u_ptr = L->ffn_up_exps.dev; d_ptr = L->ffn_down_exps.dev;
            gs = (size_t)ffn * row_b(L->ffn_gate_exps, dim);
            us = (size_t)ffn * row_b(L->ffn_up_exps, dim);
            ds = (size_t)dim * row_b(L->ffn_down_exps, ffn);
        } else {
            const size_t gb = (size_t)ffn * row_b(L->ffn_gate_exps, dim);
            const size_t ub = (size_t)ffn * row_b(L->ffn_up_exps, dim);
            const size_t db = (size_t)dim * row_b(L->ffn_down_exps, ffn);
            for (int k = 0; k < 4; k++) {
                const size_t e = (size_t)idref[k];
                CUDA_CHECK(cudaMemcpy(d_pg + k * gate_slot, (const char*)L->ffn_gate_exps.data + e * gb, gb, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_pu + k * up_slot,   (const char*)L->ffn_up_exps.data + e * ub, ub, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_pd + k * down_slot, (const char*)L->ffn_down_exps.data + e * db, db, cudaMemcpyHostToDevice));
            }
            const int slots[4] = {0, 1, 2, 3};
            CUDA_CHECK(cudaMemcpy(d_ids, slots, K * 4, cudaMemcpyHostToDevice));
            g_ptr = d_pg; u_ptr = d_pu; d_ptr = d_pd; gs = gate_slot; us = up_slot; ds = down_slot;
        }
        lfm25_gpu_moe_ffn(d_xq8, ids_used, d_tw, g_ptr, u_ptr, d_ptr,
                          down_type, d_act, d_actq8, d_dst, false, (int)dim, (int)ffn, gs, us, ds);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actgot.data(), d_act, (size_t)K * ffn * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outgot.data(), d_dst, dim * 4, cudaMemcpyDeviceToHost));

        const double ae = rel_err(actgot, actref), oe = rel_err(outgot, outref);
        max_act = fmax(max_act, ae); max_out = fmax(max_out, oe);
        if (ae <= 1e-5 && oe <= 1e-4) pass++;
        else printf("FAIL layer %u step %u down=%s: act_err=%.3g out_err=%.3g\n",
                    layer, step, down_type == GGML_TYPE_Q4_K ? "Q4_K" : "Q6_K", ae, oe);

        if (c == 0) {
            std::vector<float> r(dim), acc(dim);
            for (uint32_t i = 0; i < dim; i++) r[i] = 0.25f * sinf(0.01f * (float)i) + 0.1f;
            CUDA_CHECK(cudaMemcpy(d_dst, r.data(), dim * 4, cudaMemcpyHostToDevice));
            lfm25_gpu_moe_ffn(d_xq8, ids_used, d_tw, g_ptr, u_ptr, d_ptr,
                              down_type, d_act, d_actq8, d_dst, true, (int)dim, (int)ffn, gs, us, ds);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(acc.data(), d_dst, dim * 4, cudaMemcpyDeviceToHost));
            std::vector<float> expect(dim);
            for (uint32_t i = 0; i < dim; i++) expect[i] = r[i] + outref[i];
            double d = 0;
            for (uint32_t i = 0; i < dim; i++) d = fmax(d, fabs((double)acc[i] - expect[i]));
            accum_ok = d / max_abs(outref) <= 1e-4;
        }
    }
    fclose(f);

    printf("moe_ffn: %u/%u cases pass, max act_err = %.3g, max out_err = %.3g, accum %s\n",
           pass, n_cases, max_act, max_out, accum_ok ? "ok" : "FAIL");

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_act)); CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_tw)); CUDA_CHECK(cudaFree(d_ids)); CUDA_CHECK(cudaFree(d_xq8)); CUDA_CHECK(cudaFree(d_actq8));
    if (pool_mode) { cudaFree(d_pg); cudaFree(d_pu); cudaFree(d_pd); }
    lfm25_gpu_shutdown(model);
    lfm25_model_free(model);
    return (pass == n_cases && accum_ok) ? 0 : 1;
}
