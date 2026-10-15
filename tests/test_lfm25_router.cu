// Standalone test for lfm25_gpu_router_top4 against llama.cpp-derived fixtures.
// Usage: lfm25-router-test <model.gguf> <fixtures.bin>
#include "lfm25_model.h"
#include "lfm25_engine.h"
#include "cuda_check.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vector>

// Link stub: cpu_ops.c references tuff_exact(), which normally lives in main.c.
extern "C" int tuff_exact(void) { return 0; }

static int read_exact(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n ? 0 : -1; }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model.gguf> <fixtures.bin>\n", argv[0]); return 2; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    char magic[8]; uint32_t hdr[4];
    if (read_exact(f, magic, 8) || read_exact(f, hdr, sizeof(hdr)) || memcmp(magic, "LFMRTR01", 8) != 0) {
        fprintf(stderr, "bad fixture header/magic\n"); return 2;
    }
    const uint32_t n_cases = hdr[0], dim = hdr[1], n_expert = hdr[2], K = hdr[3];
    if (dim != 2048 || n_expert != 32 || K != 4) {
        fprintf(stderr, "unexpected fixture dims: dim=%u n_expert=%u k=%u\n", dim, n_expert, K); return 2;
    }

    lfm25_model_t *model = lfm25_model_load(argv[1]);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    if (model->embedding_length != dim || model->expert_count != n_expert) {
        fprintf(stderr, "model/fixture mismatch: dim=%u experts=%u\n", model->embedding_length, model->expert_count); return 2;
    }

    // Upload only the router tensors of each MoE layer.
    std::vector<float *> d_w(model->block_count, nullptr), d_b(model->block_count, nullptr);
    for (uint32_t l = 0; l < model->block_count; l++) {
        const lfm25_layer_t *L = &model->layers[l];
        if (L->ffn_type != LFM25_FFN_MOE) continue;
        CUDA_CHECK(cudaMalloc(&d_w[l], (size_t)n_expert * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[l], n_expert * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_w[l], L->ffn_gate_inp.data, (size_t)n_expert * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b[l], L->exp_probs_b.data, n_expert * sizeof(float), cudaMemcpyHostToDevice));
    }

    float *d_x, *d_tw, *d_lg; int *d_ids;
    CUDA_CHECK(cudaMalloc(&d_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tw, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lg, n_expert * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids, K * sizeof(int)));

    std::vector<float> x(dim), wref(K), lref(n_expert), wgot(K), lgot(n_expert);
    int32_t idref[4], idgot[4];
    uint32_t pass = 0;
    double max_dw = 0, max_dl = 0;

    for (uint32_t c = 0; c < n_cases; c++) {
        uint32_t ls[2];
        if (read_exact(f, ls, sizeof(ls)) || read_exact(f, x.data(), dim * 4) || read_exact(f, idref, K * 4) ||
            read_exact(f, wref.data(), K * 4) || read_exact(f, lref.data(), n_expert * 4)) {
            fprintf(stderr, "truncated fixture at case %u\n", c); return 2;
        }
        const uint32_t layer = ls[0], step = ls[1];
        if (layer >= model->block_count || !d_w[layer]) { fprintf(stderr, "case %u: layer %u is not MoE\n", c, layer); return 2; }

        CUDA_CHECK(cudaMemcpy(d_x, x.data(), dim * 4, cudaMemcpyHostToDevice));
        lfm25_gpu_router_top4(d_x, d_w[layer], d_b[layer], d_ids, d_tw, d_lg, (int)dim, (int)n_expert);
        CUDA_CHECK(cudaMemcpy(idgot, d_ids, K * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(wgot.data(), d_tw, K * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lgot.data(), d_lg, n_expert * 4, cudaMemcpyDeviceToHost));

        double dw = 0, dl = 0;
        for (uint32_t k = 0; k < K; k++) dw = fmax(dw, fabs((double)wgot[k] - wref[k]));
        for (uint32_t e = 0; e < n_expert; e++) dl = fmax(dl, fabs((double)lgot[e] - lref[e]));
        bool ok = dw <= 1e-5 && dl <= 1e-4;
        for (uint32_t k = 0; k < K; k++) if (idgot[k] != idref[k]) ok = false;
        if (dw > max_dw) max_dw = dw;
        if (dl > max_dl) max_dl = dl;
        if (ok) pass++;
        else printf("FAIL layer %u step %u: ids exp [%d %d %d %d] got [%d %d %d %d]; w exp [%.7f %.7f %.7f %.7f] got [%.7f %.7f %.7f %.7f]; |dw|=%.3g |dlogit|=%.3g\n",
                    layer, step, idref[0], idref[1], idref[2], idref[3], idgot[0], idgot[1], idgot[2], idgot[3],
                    wref[0], wref[1], wref[2], wref[3], wgot[0], wgot[1], wgot[2], wgot[3], dw, dl);
    }
    fclose(f);

    printf("router: %u/%u cases pass, max |dw| = %.3g, max |dlogit| = %.3g\n", pass, n_cases, max_dw, max_dl);

    for (uint32_t l = 0; l < model->block_count; l++) { cudaFree(d_w[l]); cudaFree(d_b[l]); }
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_tw)); CUDA_CHECK(cudaFree(d_lg)); CUDA_CHECK(cudaFree(d_ids));
    lfm25_model_free(model);
    return pass == n_cases ? 0 : 1;
}
