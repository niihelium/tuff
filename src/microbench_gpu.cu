#include "model.h"
#include "gpu.h"
#include "cupti_profiler.h"
#include "tuff_exact.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {
static int g_tuff_exact = -1;
int tuff_exact(void) {
    if (g_tuff_exact < 0) {
        const char *e = getenv("TUFF_EXACT");
        g_tuff_exact = (e && e[0] == '1') ? 1 : 0;
    }
    return g_tuff_exact;
}

int  tuff_logit_capture_enable(int on) { (void)on; return 0; }
int  tuff_logit_capture_is_enabled(void)   { return 0; }
void tuff_logit_capture(const float *logits, uint32_t n) { (void)logits; (void)n; }
const float* tuff_logit_capture_take(uint32_t *n_out) { if (n_out) *n_out = 0; return NULL; }
}

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    printf("====================================================================================\n");
    printf("                       Tuff GPU Kernel Microbenchmark Suite                   \n");
    printf("====================================================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute: %d.%d, SMs: %d, Global VRAM: %zu MiB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1048576);

    const int dim = 2048;
    const int n_runs = 500;
    const int warmup = 50;

    printf("Iterations per benchmark: %d (Warmup: %d)\n\n", n_runs, warmup);

    // -----------------------------------------------------------------------
    // Benchmark 1: RMS-Norm & Activation Quantization (Pre/Post Layer)
    // -----------------------------------------------------------------------
    {
        float *d_x, *d_w, *d_dst;
        void *d_xq;
        cudaMalloc(&d_x, dim * sizeof(float));
        cudaMalloc(&d_w, dim * sizeof(float));
        cudaMalloc(&d_dst, dim * sizeof(float));
        cudaMalloc(&d_xq, (dim / 32) * 36);

        cudaMemset(d_x, 0x3f, dim * sizeof(float));
        cudaMemset(d_w, 0x3f, dim * sizeof(float));

        for (int i = 0; i < warmup; ++i) {
            gpu_fused_rms_norm_quant_dev(d_dst, d_x, d_w, d_xq, dim, 1e-6f);
        }
        cudaDeviceSynchronize();

        double t0 = get_time_sec();
        for (int i = 0; i < n_runs; ++i) {
            gpu_fused_rms_norm_quant_dev(d_dst, d_x, d_w, d_xq, dim, 1e-6f);
        }
        cudaDeviceSynchronize();
        double t1 = get_time_sec();
        double us = (t1 - t0) / n_runs * 1e6;
        printf("[BENCH 1] Layer RMS-Norm + Q8_1 Quantize (dim=%4d): %6.2f us/launch\n", dim, us);

        cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst); cudaFree(d_xq);
    }

    // -----------------------------------------------------------------------
    // Benchmark 2: Residual Stream Vector Addition
    // -----------------------------------------------------------------------
    {
        float *d_res_x, *d_delta;
        cudaMalloc(&d_res_x, dim * sizeof(float));
        cudaMalloc(&d_delta, dim * sizeof(float));
        cudaMemset(d_res_x, 0, dim * sizeof(float));
        cudaMemset(d_delta, 0x3f, dim * sizeof(float));

        for (int i = 0; i < warmup; ++i) gpu_residual_add(d_res_x, d_delta, dim);
        cudaDeviceSynchronize();

        double t0 = get_time_sec();
        for (int i = 0; i < n_runs; ++i) gpu_residual_add(d_res_x, d_delta, dim);
        cudaDeviceSynchronize();
        double t1 = get_time_sec();
        double us = (t1 - t0) / n_runs * 1e6;
        printf("[BENCH 2] Residual Stream Vector Add     (dim=%4d): %6.2f us/launch\n", dim, us);

        cudaFree(d_res_x); cudaFree(d_delta);
    }

    // -----------------------------------------------------------------------
    // Benchmark 3: Vectorized MoE 128-bit Combine Accumulator
    // -----------------------------------------------------------------------
    {
        float *d_res_x, *d_shexp;
        cudaMalloc(&d_res_x, dim * sizeof(float));
        cudaMalloc(&d_shexp, dim * sizeof(float));
        cudaMemset(d_res_x, 0, dim * sizeof(float));
        cudaMemset(d_shexp, 0x3f, dim * sizeof(float));

        int hit_pos[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        float top_values[8] = {0.2f, 0.15f, 0.15f, 0.12f, 0.1f, 0.1f, 0.09f, 0.09f};

        for (int i = 0; i < warmup; ++i) {
            gpu_moe_combine_accum(d_res_x, hit_pos, 8, top_values, d_shexp, NULL, dim);
        }
        cudaDeviceSynchronize();

        double t0 = get_time_sec();
        for (int i = 0; i < n_runs; ++i) {
            gpu_moe_combine_accum(d_res_x, hit_pos, 8, top_values, d_shexp, NULL, dim);
        }
        cudaDeviceSynchronize();
        double t1 = get_time_sec();
        double us = (t1 - t0) / n_runs * 1e6;
        printf("[BENCH 3] Vectorized MoE Combine (8 hits, dim=%4d): %6.2f us/launch\n", dim, us);

        cudaFree(d_res_x); cudaFree(d_shexp);
    }

    printf("\n====================================================================================\n");
    printf("                    All Isolated Kernel Microbenchmarks Ready                        \n");
    printf("====================================================================================\n");
    return 0;
}
