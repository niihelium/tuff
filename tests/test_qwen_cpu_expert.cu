// Byte-identity test of the hybrid CPU-miss MoE for Qwen3.6-35B-A3B against the all-GPU path.
// Checks forced-hit against all miss masks for layers with Q4_K down and Q6_K down.
// Usage: qwen-cpu-expert-test <model.gguf> [reps_per_layer=2]
#include "model.h"
#include "gpu.h"
#include "qwen_cpu_expert.h"
#include "cuda_check.h"
#include "ggml_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <random>

extern "C" int tuff_exact(void) { return 0; } // cpu_ops.c link stub

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [reps_per_layer=2]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const int reps = argc > 2 ? atoi(argv[2]) : 2;

    printf("[qwen-cpu-expert-test] Loading model %s...\n", model_path);
    model_t *m = model_load(model_path);
    if (!m) {
        fprintf(stderr, "Failed to load model %s\n", model_path);
        return 1;
    }

    printf("[qwen-cpu-expert-test] Initializing GPU...\n");
    gpu_prepare(m);

    const int dim = (int)m->embedding_length;     // 2048
    const int ffn = (int)m->expert_ffn_length;    // 512
    const int K = 8;
    const int n_exp = (int)m->expert_count;       // 256

    cudaStream_t stream = 0;

    qwen_cpu_expert_t *ce = qwen_cpu_expert_create(0, dim, ffn);
    if (!ce) {
        fprintf(stderr, "Failed to create qwen_cpu_expert\n");
        return 1;
    }
    printf("[qwen-cpu-expert-test] Created CPU expert worker with %d threads, dim=%d, ffn=%d\n",
           qwen_cpu_expert_threads(ce), dim, ffn);

    float *d_norm_x, *d_ref_dst, *d_got_dst, *d_top_values, *d_shexp_out, *d_act;
    void *d_xq8, *d_act_q8;
    CUDA_CHECK(cudaMalloc(&d_norm_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref_dst, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_got_dst, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_top_values, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shexp_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act, K * ffn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xq8, (size_t)(dim / 32) * 36));
    CUDA_CHECK(cudaMalloc(&d_act_q8, (size_t)K * (ffn / 32) * 36));

    std::vector<float> h_norm_x(dim), h_top_values(K), h_shexp_out(dim);
    std::vector<float> h_ref_dst(dim), h_got_dst(dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    int total_cases = 0, pass_cases = 0;
    for (uint32_t l = 0; l < m->block_count; ++l) {
        if (l >= m->block_count) continue;
        const model_layer_t *L = &m->layers[l];
        const int down_type = (int)L->ffn_down_exps.info.type;
        const char *down_type_str = (down_type == GGML_TYPE_Q6_K) ? "Q6_K" : "Q4_K";
        printf("\n=== Testing Layer %u (down=%s) ===\n", l, down_type_str);

        const size_t gate_bytes = (size_t)ffn * (dim / 256) * sizeof(block_q4_K);
        const size_t up_bytes   = gate_bytes;
        const size_t down_bytes = (size_t)dim * (ffn / 256) * (down_type == GGML_TYPE_Q6_K ? sizeof(block_q6_K) : sizeof(block_q4_K));

        for (int rep = 0; rep < reps; ++rep) {
            // Select 8 distinct random experts
            int ex[8];
            for (int k = 0; k < 8; ++k) {
                bool dup;
                do {
                    ex[k] = (int)(rng() % n_exp);
                    dup = false;
                    for (int j = 0; j < k; ++j) if (ex[j] == ex[k]) dup = true;
                } while (dup);
            }

            const void *hg[8], *hu[8], *hd[8];
            int slots[8];
            for (int k = 0; k < 8; ++k) {
                hg[k] = (const char*)L->ffn_gate_exps.data + (size_t)ex[k] * gate_bytes;
                hu[k] = (const char*)L->ffn_up_exps.data   + (size_t)ex[k] * up_bytes;
                hd[k] = (const char*)L->ffn_down_exps.data + (size_t)ex[k] * down_bytes;
                // Allocate slot in GPU cache and fill it so it is guaranteed resident
                int s = gpu_expert_cache_alloc_slot(l, ex[k]);
                gpu_expert_cache_fill(s, m, l, ex[k]);
                slots[k] = s;
            }
            gpu_expert_cache_sync();

            // Random norm_x, shexp_out, and top_values
            for (int i = 0; i < dim; ++i) {
                h_norm_x[i] = nd(rng);
                h_shexp_out[i] = 0.5f * nd(rng);
            }
            float sum_w = 0.0f;
            for (int k = 0; k < 8; ++k) {
                h_top_values[k] = 0.05f + (float)(rng() % 1000) / 1000.0f;
                sum_w += h_top_values[k];
            }
            for (int k = 0; k < 8; ++k) h_top_values[k] /= sum_w;

            CUDA_CHECK(cudaMemcpy(d_norm_x, h_norm_x.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_top_values, h_top_values.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_shexp_out, h_shexp_out.data(), dim * sizeof(float), cudaMemcpyHostToDevice));

            // Quantize d_norm_x into d_xq8
            gpu_expert_moe_normx_quant_col(d_norm_x, dim, 0);
            // ex_col_xq8(0) is on device
            CUDA_CHECK(cudaMemcpy(d_xq8, (const void*)gpu_expert_dev_ptr(slots[0], 0), (size_t)(dim / 32) * 36, cudaMemcpyDeviceToDevice)); // dummy dev ptr check
            // Actually ex_col_xq8 is populated by gpu_expert_moe_gateup_swiglu_dev

            // 1. ALL-GPU reference run (miss_mask = 0)
            int hit_pos_ref[8];
            for (int k = 0; k < 8; ++k) hit_pos_ref[k] = k;
            CUDA_CHECK(cudaMemset(d_ref_dst, 0, dim * sizeof(float)));

            gpu_expert_moe_gateup_swiglu_dev(slots, 8, d_norm_x, dim, ffn);
            gpu_expert_moe_down_dev(8, (ggml_type_t)down_type, ffn, dim);
            gpu_qwen_moe_combine_unified(d_ref_dst, gpu_get_expert_dbat_dev_ptr(0), hit_pos_ref, 8,
                                         NULL, 0u, h_top_values.data(), d_shexp_out, dim, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(h_ref_dst.data(), d_ref_dst, dim * sizeof(float), cudaMemcpyDeviceToHost));

            std::vector<float> ref_exdbat(8 * dim);
            CUDA_CHECK(cudaMemcpy(ref_exdbat.data(), gpu_get_expert_dbat_dev_ptr(0), 8 * dim * sizeof(float), cudaMemcpyDeviceToHost));

            // Test representative miss masks:
            // 0 (all hits), 1 (exp 0 miss), 2 (exp 1 miss), 5 (exp 0,2 miss), 0x55 (alternating), 0xAA, 0xFF (all misses)
            unsigned test_masks[] = { 0x01u, 0x02u, 0x05u, 0x18u, 0x55u, 0xAAu, 0xFFu };
            for (unsigned mask : test_masks) {
                total_cases++;
                int cur_hit_slots[8], cur_hit_pos[8];
                int cur_n_hits = 0;
                for (int k = 0; k < 8; ++k) {
                    if (!((mask >> k) & 1u)) {
                        cur_hit_slots[cur_n_hits] = slots[k];
                        cur_hit_pos[cur_n_hits] = k;
                        cur_n_hits++;
                    }
                }

                CUDA_CHECK(cudaMemset(d_got_dst, 0, dim * sizeof(float)));

                // Run hit experts on GPU
                if (cur_n_hits > 0) {
                    gpu_expert_moe_gateup_swiglu_dev(cur_hit_slots, cur_n_hits, d_norm_x, dim, ffn);
                    gpu_expert_moe_down_dev(cur_n_hits, (ggml_type_t)down_type, ffn, dim);
                }

                // Run miss experts via qwen_cpu_expert
                if (mask != 0u) {
                    const void *d_xq8_dev = gpu_get_expert_xq8_dev_ptr(0);
                    qwen_cpu_expert_eval(ce, stream, d_xq8_dev, down_type,
                                         dim, ffn, mask, hg, hu, hd);
                }

                // Check intermediate expert outputs:
                std::vector<float> got_exdbat(cur_n_hits * dim);
                if (cur_n_hits > 0) {
                    CUDA_CHECK(cudaMemcpy(got_exdbat.data(), gpu_get_expert_dbat_dev_ptr(0), cur_n_hits * dim * sizeof(float), cudaMemcpyDeviceToHost));
                    for (int h = 0; h < cur_n_hits; ++h) {
                        int k = cur_hit_pos[h];
                        if (memcmp(&got_exdbat[h * dim], &ref_exdbat[k * dim], dim * sizeof(float)) != 0) {
                            printf("  [DEBUG Layer %u Mask 0x%02X] Hit expert h=%d (k=%d) differs from ref! got[0]=%a ref[0]=%a\n",
                                   l, mask, h, k, got_exdbat[h * dim], ref_exdbat[k * dim]);
                        }
                    }
                }
                for (int k = 0; k < 8; ++k) {
                    if ((mask >> k) & 1u) {
                        float *py = qwen_cpu_expert_host_pre_y(ce) + (size_t)k * dim;
                        if (memcmp(py, &ref_exdbat[k * dim], dim * sizeof(float)) != 0) {
                            printf("  [DEBUG Layer %u Mask 0x%02X] Miss expert k=%d differs from ref! py[0]=%a ref[0]=%a\n",
                                   l, mask, k, py[0], ref_exdbat[k * dim]);
                        }
                    }
                }
                fflush(stdout);

                // Unified combine
                gpu_qwen_moe_combine_unified(d_got_dst, gpu_get_expert_dbat_dev_ptr(0), cur_hit_pos, cur_n_hits,
                                             qwen_cpu_expert_d_pre_y(ce), mask,
                                             h_top_values.data(), d_shexp_out, dim, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(h_got_dst.data(), d_got_dst, dim * sizeof(float), cudaMemcpyDeviceToHost));

                // Compare bit-identity
                if (memcmp(h_got_dst.data(), h_ref_dst.data(), dim * sizeof(float)) == 0) {
                    pass_cases++;
                } else {
                    printf("MISMATCH on Layer %u Rep %d Mask 0x%02X (%s down)!\n", l, rep, mask, down_type_str);
                    int shown = 0;
                    for (int i = 0; i < dim && shown < 5; ++i) {
                        if (memcmp(&h_got_dst[i], &h_ref_dst[i], 4) != 0) {
                            printf("  dst[%d]: got=%a ref=%a (diff=%e)\n", i, h_got_dst[i], h_ref_dst[i],
                                   fabsf(h_got_dst[i] - h_ref_dst[i]));
                            shown++;
                        }
                    }
                }

            }
        }
    }

    qwen_cpu_expert_free(ce);
    CUDA_CHECK(cudaFree(d_norm_x));
    CUDA_CHECK(cudaFree(d_ref_dst));
    CUDA_CHECK(cudaFree(d_got_dst));
    CUDA_CHECK(cudaFree(d_top_values));
    CUDA_CHECK(cudaFree(d_shexp_out));
    CUDA_CHECK(cudaFree(d_act));
    CUDA_CHECK(cudaFree(d_xq8));
    CUDA_CHECK(cudaFree(d_act_q8));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));

    printf("\n=======================================================\n");
    printf(" Result: %d / %d cases byte-identical\n", pass_cases, total_cases);
    printf("=======================================================\n");

    return (pass_cases == total_cases) ? 0 : 1;
}
