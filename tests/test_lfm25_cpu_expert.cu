// Byte-identity test of the hybrid CPU-miss MoE FFN (lfm25_cpu_expert_moe_ffn) against the all-GPU lfm25_gpu_moe_ffn.
// For random (layer, 4 experts, activations) cases and EVERY one of the 15 non-empty miss masks, d_act, d_act_q8 and
// d_dst must be memcmp-identical. Also checks the scalar reference row dot against the AVX2 one on random blocks.
// Usage: lfm25-cpu-expert-test <model.gguf> [reps_per_layer=3]
#include "lfm25_model.h"
#include "lfm25_engine.h"
#include "lfm25_cpu_expert.h"
#include "cuda_check.h"
#include "ggml_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <random>

extern "C" int tuff_exact(void) { return 0; }   // cpu_ops.c link stub

static uint16_t rand_half(std::mt19937 &r) {   // finite, moderate magnitude
    uint32_t sign = r() & 1u, exp = 8 + (r() % 14), man = r() & 0x3ffu;   // 2^-7 .. 2^6
    return (uint16_t)((sign << 15) | (exp << 10) | man);
}

// scalar vs AVX2 row dots on random blocks (random bytes, extreme int8 values included)
static int rowdot_check(std::mt19937 &r) {
    int bad = 0, total = 0;
    for (int type = 0; type < 2; ++type) {
        for (int nblk = 1; nblk <= 8; ++nblk) {
            for (int rep = 0; rep < 40; ++rep) {
                const size_t bsz = type == 0 ? sizeof(block_q4_K) : sizeof(block_q6_K);
                std::vector<uint8_t> w(bsz * nblk);
                for (auto &b : w) b = (uint8_t)r();
                for (int k = 0; k < nblk; ++k) {
                    if (type == 0) {
                        block_q4_K *b = (block_q4_K *)&w[k * bsz]; b->d = rand_half(r); b->dmin = rand_half(r);
                    } else {
                        block_q6_K *b = (block_q6_K *)&w[k * bsz]; b->d = rand_half(r);
                    }
                }
                std::vector<lfm25_q8blk_t> y((size_t)nblk * 8);
                for (auto &q : y) {
                    q.d = rand_half(r); q.s = rand_half(r);
                    for (int i = 0; i < 32; ++i) q.qs[i] = (int8_t)(r() % (rep % 4 == 0 ? 256 : 255) - (rep % 4 == 0 ? 128 : 127));
                }
                float a = lfm25_cpu_row_dot_scalar(type == 0 ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K, w.data(), y.data(), nblk);
                float b = lfm25_cpu_row_dot_avx2(type == 0 ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K, w.data(), y.data(), nblk);
                total++;
                if (memcmp(&a, &b, 4) != 0) {
                    if (bad < 5) printf("rowdot mismatch type=%s nblk=%d rep=%d scalar=%a avx2=%a\n", type ? "Q6_K" : "Q4_K", nblk, rep, a, b);
                    bad++;
                }
            }
        }
    }
    printf("cpu_expert: scalar-vs-avx2 row dots: %d/%d identical\n", total - bad, total);
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [reps_per_layer]\n", argv[0]); return 2; }
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    std::mt19937 rng(12345);
    int bad_rows = rowdot_check(rng);

    lfm25_model_t *model = lfm25_model_load(argv[1]);
    if (!model || !model->is_moe) { fprintf(stderr, "model load failed / not MoE\n"); return 2; }
    const int dim = (int)model->embedding_length, ffn = (int)model->expert_ffn_dim, K = 4;
    const int n_exp = (int)model->expert_count;

    auto row_b = [](const lfm25_tensor_t &t, int64_t n) { return ggml_row_size((ggml_type_t)t.info.type, n); };
    size_t gate_slot = 0, up_slot = 0, down_slot = 0;
    for (uint32_t l = 0; l < model->block_count; l++) {
        const lfm25_layer_t *L = &model->layers[l];
        if (L->ffn_type != LFM25_FFN_MOE) continue;
        gate_slot = std::max(gate_slot, (size_t)ffn * row_b(L->ffn_gate_exps, dim));
        up_slot   = std::max(up_slot,   (size_t)ffn * row_b(L->ffn_up_exps, dim));
        down_slot = std::max(down_slot, (size_t)dim * row_b(L->ffn_down_exps, ffn));
    }
    char *d_pg, *d_pu, *d_pd;
    CUDA_CHECK(cudaMalloc(&d_pg, 4 * gate_slot)); CUDA_CHECK(cudaMalloc(&d_pu, 4 * up_slot)); CUDA_CHECK(cudaMalloc(&d_pd, 4 * down_slot));
    float *d_x, *d_act, *d_dst, *d_tw; int *d_slots; void *d_xq8, *d_actq8;
    CUDA_CHECK(cudaMalloc(&d_x, dim * 4)); CUDA_CHECK(cudaMalloc(&d_act, (size_t)K * ffn * 4));
    CUDA_CHECK(cudaMalloc(&d_dst, dim * 4)); CUDA_CHECK(cudaMalloc(&d_tw, K * 4)); CUDA_CHECK(cudaMalloc(&d_slots, K * 4));
    CUDA_CHECK(cudaMalloc(&d_xq8, (size_t)(dim / 32) * 36)); CUDA_CHECK(cudaMalloc(&d_actq8, (size_t)(K * ffn / 32) * 36));
    const int slots[4] = {0, 1, 2, 3};
    CUDA_CHECK(cudaMemcpy(d_slots, slots, sizeof(slots), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    lfm25_gpu_set_stream(stream);
    lfm25_cpu_expert_t *ce = lfm25_cpu_expert_create(0, dim, ffn);
    if (!ce) { fprintf(stderr, "cpu expert create failed\n"); return 2; }
    printf("cpu_expert: %d threads, dim=%d ffn=%d experts=%d\n", lfm25_cpu_expert_threads(ce), dim, ffn, n_exp);

    const size_t act_bytes = (size_t)K * ffn * 4, q8_bytes = (size_t)(K * ffn / 32) * 36;
    std::vector<float> x(dim), resid(dim), tw(K);
    std::vector<uint8_t> ref_act(act_bytes), ref_q8(q8_bytes), ref_dst(dim * 4), got_act(act_bytes), got_q8(q8_bytes), got_dst(dim * 4);
    int total = 0, pass = 0, cases = 0;
    std::normal_distribution<float> nd(0.f, 1.f);

    for (uint32_t l = 0; l < model->block_count; l++) {
        const lfm25_layer_t *L = &model->layers[l];
        if (L->ffn_type != LFM25_FFN_MOE) continue;
        const int down_type = (int)L->ffn_down_exps.info.type;
        const size_t gb = (size_t)ffn * row_b(L->ffn_gate_exps, dim), ub = (size_t)ffn * row_b(L->ffn_up_exps, dim),
                     db = (size_t)dim * row_b(L->ffn_down_exps, ffn);
        for (int rep = 0; rep < reps; ++rep) {
            // experts: 4 distinct random
            int ex[4];
            for (int k = 0; k < 4; ++k) { bool dup; do { ex[k] = (int)(rng() % n_exp); dup = false; for (int j = 0; j < k; ++j) dup |= ex[j] == ex[k]; } while (dup); }
            const void *hg[4], *hu[4], *hd[4];
            for (int k = 0; k < 4; ++k) {
                hg[k] = (const char *)L->ffn_gate_exps.data + (size_t)ex[k] * gb;
                hu[k] = (const char *)L->ffn_up_exps.data + (size_t)ex[k] * ub;
                hd[k] = (const char *)L->ffn_down_exps.data + (size_t)ex[k] * db;
                CUDA_CHECK(cudaMemcpy(d_pg + k * gate_slot, hg[k], gb, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_pu + k * up_slot, hu[k], ub, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_pd + k * down_slot, hd[k], db, cudaMemcpyHostToDevice));
            }
            // activations: gaussian with a few outliers and a random per-case scale
            const float scale = 0.2f + 3.f * (float)(rng() % 100) / 100.f;
            for (int i = 0; i < dim; ++i) { x[i] = scale * nd(rng); if (rng() % 97 == 0) x[i] *= 12.f; }
            float s = 0; for (int k = 0; k < K; ++k) { tw[k] = 0.1f + (float)(rng() % 1000) / 1000.f; s += tw[k]; }
            for (int k = 0; k < K; ++k) tw[k] /= s;
            for (int i = 0; i < dim; ++i) resid[i] = 0.25f * nd(rng);
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), dim * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tw, tw.data(), K * 4, cudaMemcpyHostToDevice));
            lfm25_gpu_quantize_q8_1(d_x, d_xq8, dim);
            CUDA_CHECK(cudaMemcpyAsync(lfm25_cpu_expert_host_xq8(ce), d_xq8, (size_t)(dim / 32) * 36, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto fetch = [&](std::vector<uint8_t> &a, std::vector<uint8_t> &q, std::vector<uint8_t> &o) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(a.data(), d_act, act_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(q.data(), d_actq8, q8_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(o.data(), d_dst, dim * 4, cudaMemcpyDeviceToHost));
            };
            CUDA_CHECK(cudaMemcpy(d_dst, resid.data(), dim * 4, cudaMemcpyHostToDevice));
            lfm25_gpu_moe_ffn(d_xq8, d_slots, d_tw, d_pg, d_pu, d_pd, down_type, d_act, d_actq8, d_dst, true, dim, ffn, gate_slot, up_slot, down_slot);
            fetch(ref_act, ref_q8, ref_dst);
            cases++;

            for (unsigned mask = 1; mask <= 15; ++mask) {
                // every mask runs the AVX2 kernels; one mask per case additionally runs the scalar oracle
                for (int scalar = 0; scalar < 2; ++scalar) {
                    if (scalar && (mask != 1 + (unsigned)(cases % 15))) continue;
                    lfm25_cpu_expert_set_scalar(ce, scalar);
                    CUDA_CHECK(cudaMemset(d_act, 0xAB, act_bytes)); CUDA_CHECK(cudaMemset(d_actq8, 0xCD, q8_bytes));
                    CUDA_CHECK(cudaMemcpy(d_dst, resid.data(), dim * 4, cudaMemcpyHostToDevice));
                    lfm25_cpu_expert_moe_ffn(ce, stream, d_xq8, lfm25_cpu_expert_host_xq8(ce), d_slots, d_tw, d_pg, d_pu, d_pd, down_type,
                                             d_act, d_actq8, d_dst, true, dim, ffn, gate_slot, up_slot, down_slot, mask, hg, hu, hd);
                    fetch(got_act, got_q8, got_dst);
                    total++;
                    const bool ok = !memcmp(got_act.data(), ref_act.data(), act_bytes) && !memcmp(got_q8.data(), ref_q8.data(), q8_bytes) &&
                                    !memcmp(got_dst.data(), ref_dst.data(), dim * 4);
                    if (ok) pass++;
                    else {
                        printf("MISMATCH layer %u rep %d mask %u %s down=%s:", l, rep, mask, scalar ? "scalar" : "avx2", down_type == GGML_TYPE_Q4_K ? "Q4_K" : "Q6_K");
                        int shown = 0;
                        for (size_t i = 0; i < (size_t)K * ffn && shown < 3; ++i)
                            if (memcmp(&got_act[i * 4], &ref_act[i * 4], 4)) { float a, b; memcpy(&a, &got_act[i*4], 4); memcpy(&b, &ref_act[i*4], 4);
                                printf(" act[k=%zu,row=%zu] cpu=%a gpu=%a", i / ffn, i % ffn, a, b); shown++; }
                        for (int i = 0; i < dim && shown < 5; ++i)
                            if (memcmp(&got_dst[i * 4], &ref_dst[i * 4], 4)) { float a, b; memcpy(&a, &got_dst[i*4], 4); memcpy(&b, &ref_dst[i*4], 4);
                                printf(" dst[%d] cpu=%a gpu=%a", i, a, b); shown++; }
                        printf("\n");
                    }
                }
            }
        }
    }
    lfm25_cpu_expert_free(ce);
    printf("cpu_expert: %d/%d cases byte-identical (%d layer/expert-set cases x 15 miss masks + scalar-oracle runs)\n", pass, total, cases);
    printf("cpu_expert: %d/%d cases pass\n", pass + (bad_rows == 0), total + 1);
    cudaFree(d_pg); cudaFree(d_pu); cudaFree(d_pd);
    lfm25_model_free(model);
    return (pass == total && bad_rows == 0) ? 0 : 1;
}
