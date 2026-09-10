// P2 self-test + microbenchmark for the AVX2 int8 CPU expert kernels (src/cpu_ops.c).
//
// Runs against REAL expert tensors from the GGUF -- never synthetic random weights:
//   - accuracy: AVX2 (Q8_K activation) vs the exact scalar fp32 path, per tensor,
//     reporting max |abs err|, max relative err and cosine similarity.
//   - throughput: weight bytes/s for both kernels on the real expert shapes, in a
//     "cold" sweep across many distinct experts (streamed from RAM, the P1 scenario)
//     and a "hot" single-expert loop (cache-resident, pure kernel rate).
//
// Build target: cpu-expert-bench.  Usage:
//   ./build-p2/cpu-expert-bench <model.gguf> [--experts N] [--reps N] [--threads N]
//
// Every number printed is measured by this program. Nothing here is estimated.

#include "model.h"
#include "cpu_ops.h"
#include "tuff_exact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// This target does not link src/main.c, which holds the engine's definition of P0's
// tuff_exact(). Provide the same semantics locally so cpu_ops.c resolves.
int tuff_exact(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TUFF_EXACT");
        cached = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }
    return cached;
}

// ---------------------------------------------------------------------------

typedef struct {
    const char  *name;      // "gate" / "up" / "down"
    const void  *base;      // start of the expert-major weight blob
    ggml_type_t  type;
    int64_t      in_dim;
    int64_t      out_dim;
    size_t       expert_bytes;
    int64_t      n_experts;
} tcase_t;

static tcase_t make_case(const char *name, const tensor_t *t, int64_t n_experts) {
    tcase_t c;
    c.name    = name;
    c.base    = t->data;
    c.type    = (ggml_type_t)t->info.type;
    c.in_dim  = t->info.dims[0];
    c.out_dim = t->info.dims[1];
    // dims[2] is the expert count for the 3-D expert tensors; fall back to the model's.
    if (t->info.n_dims >= 3 && t->info.dims[2] > 0) n_experts = t->info.dims[2];
    c.n_experts    = n_experts;
    c.expert_bytes = ggml_row_size(c.type, c.in_dim) * (size_t)c.out_dim;
    return c;
}

static const void *expert_ptr(const tcase_t *c, int64_t e) {
    return (const uint8_t*)c->base + (size_t)e * c->expert_bytes;
}

// Fault in / warm the pages of `n` experts so timing measures the kernel + RAM traffic,
// not page faults from the mmap'd GGUF. Returns a checksum so it cannot be optimized away.
static uint64_t touch_experts(const tcase_t *c, int64_t n) {
    uint64_t sum = 0;
    for (int64_t e = 0; e < n; ++e) {
        const uint8_t *p = (const uint8_t*)expert_ptr(c, e);
        for (size_t off = 0; off < c->expert_bytes; off += 64) sum += p[off];
    }
    return sum;
}

static void fill_activation(float *x, int64_t n, unsigned seed) {
    // Deterministic, roughly unit-scaled -- shaped like a post-rms_norm activation.
    uint32_t s = seed | 1u;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = (float)((s >> 8) & 0xFFFFFF) / (float)0x1000000; // [0,1)
        x[i] = (u - 0.5f) * 2.0f;
    }
}

// ---------------------------------------------------------------------------
// accuracy
// ---------------------------------------------------------------------------

// A realistic MoE input: the real token embedding for `token_id`, rms_norm'd with the
// layer's real post-attention norm weights -- i.e. exactly the vector shape (scale and
// distribution) that norm_x has in the engine. Synthetic uniform noise would understate
// the Q8_K quantization error, which depends on the activation's peak-to-RMS ratio.
static void real_activation(const model_t *m, const model_layer_t *L, float *x, int token_id) {
    const int64_t dim = m->embedding_length;
    const uint8_t *base = (const uint8_t*)m->token_embd.data;
    const ggml_type_t et = (ggml_type_t)m->token_embd.info.type;
    dequantize_row(et, base + (size_t)token_id * ggml_row_size(et, dim), x, dim);
    if (!L->post_attention_norm.data || L->post_attention_norm.info.type != GGML_TYPE_F32) {
        fprintf(stderr, "[FATAL] post_attention_norm missing/not F32 for the tested layer\n");
        exit(1);
    }
    rms_norm(x, x, (const float*)L->post_attention_norm.data, (int)dim, m->rms_norm_eps);
}

// `xs` holds the activation(s): one shared vector when x_stride == 0 (gate/up -- all experts
// of a layer see the same norm_x, which is why it is quantized once), or one vector per
// expert when x_stride == in_dim (down -- each expert has its own swiglu output).
static void accuracy_case(const tcase_t *c, int layer, int64_t n_experts_tested,
                          const float *xs, int64_t x_stride) {
    float *ref   = (float*)malloc((size_t)c->out_dim * sizeof(float));
    float *refq  = (float*)malloc((size_t)c->out_dim * sizeof(float));
    float *fast  = (float*)malloc((size_t)c->out_dim * sizeof(float));
    float *xdeq  = (float*)malloc((size_t)c->in_dim  * sizeof(float));
    block_q8_K *q8 = (block_q8_K*)malloc(q8_K_row_bytes(c->in_dim));

    double worst_abs = 0.0, worst_rel = 0.0, worst_cos = 1.0;
    double worst_rel_l2 = 0.0, worst_rel_sig = 0.0;
    double worst_kern_l2 = 0.0;   // AVX2 vs exact scalar fed the SAME dequantized Q8_K input
    int64_t worst_expert = -1;

    for (int64_t e = 0; e < n_experts_tested; ++e) {
        const void *w = expert_ptr(c, e);
        const float *x = xs + (size_t)e * (size_t)x_stride;
        if (e == 0 || x_stride != 0) {
            quantize_row_q8_K(x, q8, c->in_dim);
            // Q8_K -> fp32, so the scalar kernel can be fed exactly the values the AVX2
            // kernel actually multiplies. The residual difference is then purely the
            // kernel's (int8 accumulation + reassociated fp reduction), not the activation
            // quantization -- this separates "port is wrong" from "8-bit input".
            for (int64_t b = 0; b < c->in_dim / QK_K; ++b)
                for (int j = 0; j < QK_K; ++j)
                    xdeq[b * QK_K + j] = q8[b].d * (float)q8[b].qs[j];
        }
        // exact scalar reference: force it regardless of TUFF_EXACT by calling the
        // scalar entry points directly through mat_vec_cpu's tensor view.
        tensor_t vt;
        memset(&vt, 0, sizeof(vt));
        vt.data = w;
        vt.info.type = (uint32_t)c->type;
        vt.info.n_dims = 2;
        vt.info.dims[0] = c->in_dim;
        vt.info.dims[1] = c->out_dim;
        if (c->type == GGML_TYPE_Q6_K) {
            mat_vec_q6_K_scalar(ref,  (const block_q6_K*)w, x,    c->out_dim, c->in_dim);
            mat_vec_q6_K_scalar(refq, (const block_q6_K*)w, xdeq, c->out_dim, c->in_dim);
        } else {
            mat_vec_cpu(ref,  &vt, x);
            mat_vec_cpu(refq, &vt, xdeq);
        }

        expert_mat_vec_cpu(fast, w, c->type, c->in_dim, c->out_dim, x, q8, 1);

        {
            double e2 = 0.0, r2 = 0.0;
            for (int64_t i = 0; i < c->out_dim; ++i) {
                double d = (double)fast[i] - (double)refq[i];
                e2 += d * d; r2 += (double)refq[i] * (double)refq[i];
            }
            double kl2 = r2 > 0 ? sqrt(e2 / r2) : 0.0;
            if (kl2 > worst_kern_l2) worst_kern_l2 = kl2;
        }

        // amax of the reference row -- used to define "significant" outputs. A raw
        // element-wise relative error is unbounded here because expert outputs pass
        // through zero; max_rel_sig only counts elements >= 1% of the row's amax.
        double amax = 0.0;
        for (int64_t i = 0; i < c->out_dim; ++i) if (fabs((double)ref[i]) > amax) amax = fabs((double)ref[i]);
        const double sig = 0.01 * amax;

        double maxa = 0.0, maxr = 0.0, maxr_sig = 0.0;
        double dot = 0.0, na = 0.0, nb = 0.0, err2 = 0.0, ref2 = 0.0;
        for (int64_t i = 0; i < c->out_dim; ++i) {
            double a = ref[i], b = fast[i];
            double ad = fabs(a - b);
            if (ad > maxa) maxa = ad;
            double den = fabs(a);
            if (den > 1e-6 && ad / den > maxr) maxr = ad / den;
            if (den >= sig && den > 0.0 && ad / den > maxr_sig) maxr_sig = ad / den;
            dot += a * b; na += a * a; nb += b * b;
            err2 += (a - b) * (a - b); ref2 += a * a;
        }
        if (maxr_sig > worst_rel_sig) worst_rel_sig = maxr_sig;
        double cs = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 1.0;
        double rel_l2 = ref2 > 0 ? sqrt(err2 / ref2) : 0.0;

        if (maxr > worst_rel) { worst_rel = maxr; worst_expert = e; }
        if (maxa > worst_abs) worst_abs = maxa;
        if (cs < worst_cos) worst_cos = cs;
        if (rel_l2 > worst_rel_l2) worst_rel_l2 = rel_l2;
    }

    printf("  l%-2d %-4s %-4s in=%-5lld out=%-5lld e=%-3lld | max_abs=%.3e  "
           "max_rel=%.3e (e=%lld)  max_rel_sig=%.3e  rel_l2=%.3e  min_cos=%.9f  kernel_only_l2=%.3e\n",
           layer, c->name, c->type == GGML_TYPE_Q6_K ? "Q6_K" : "Q4_K",
           (long long)c->in_dim, (long long)c->out_dim, (long long)n_experts_tested,
           worst_abs, worst_rel, (long long)worst_expert, worst_rel_sig, worst_rel_l2, worst_cos,
           worst_kern_l2);

    free(ref); free(refq); free(fast); free(xdeq); free(q8);
}

// ---------------------------------------------------------------------------
// throughput
// ---------------------------------------------------------------------------

typedef enum { KERN_SCALAR, KERN_AVX2 } kern_t;

// Returns GB/s of *weight bytes* processed.
static double bench_case(const tcase_t *c, kern_t k, int threads, int64_t n_experts,
                         int reps, const float *x, const block_q8_K *q8, int cold,
                         double *out_seconds) {
    float *dst = (float*)malloc((size_t)c->out_dim * sizeof(float));
    const int parallel = threads > 1;
    if (parallel) omp_set_num_threads(threads);

    // warm-up (also faults in the pages)
    for (int64_t e = 0; e < (cold ? n_experts : 1); ++e) {
        const void *w = expert_ptr(c, cold ? e : 0);
        if (k == KERN_AVX2) expert_mat_vec_cpu(dst, w, c->type, c->in_dim, c->out_dim, x, q8, parallel);
        else if (c->type == GGML_TYPE_Q6_K)
            mat_vec_q6_K_scalar(dst, (const block_q6_K*)w, x, c->out_dim, c->in_dim);
        else {
            tensor_t vt; memset(&vt, 0, sizeof(vt));
            vt.data = w; vt.info.type = (uint32_t)c->type;
            vt.info.n_dims = 2; vt.info.dims[0] = c->in_dim; vt.info.dims[1] = c->out_dim;
            mat_vec_cpu(dst, &vt, x);
        }
    }

    const double t0 = omp_get_wtime();
    int64_t calls = 0;
    for (int r = 0; r < reps; ++r) {
        for (int64_t e = 0; e < (cold ? n_experts : 1); ++e) {
            const void *w = expert_ptr(c, cold ? e : 0);
            if (k == KERN_AVX2) {
                expert_mat_vec_cpu(dst, w, c->type, c->in_dim, c->out_dim, x, q8, parallel);
            } else if (c->type == GGML_TYPE_Q6_K) {
                mat_vec_q6_K_scalar(dst, (const block_q6_K*)w, x, c->out_dim, c->in_dim);
            } else {
                tensor_t vt; memset(&vt, 0, sizeof(vt));
                vt.data = w; vt.info.type = (uint32_t)c->type;
                vt.info.n_dims = 2; vt.info.dims[0] = c->in_dim; vt.info.dims[1] = c->out_dim;
                mat_vec_cpu(dst, &vt, x);
            }
            calls++;
        }
    }
    const double dt = omp_get_wtime() - t0;

    // Keep dst alive.
    volatile float sink = dst[0]; (void)sink;
    free(dst);

    if (out_seconds) *out_seconds = dt / (double)calls;
    return (double)calls * (double)c->expert_bytes / dt / 1e9;
}

// The scalar path always runs its own `#pragma omp parallel for` over rows (that is its
// shipped behaviour), so a "1 thread" scalar measurement needs the OMP thread count pinned.
static double bench_scalar(const tcase_t *c, int threads, int64_t n_experts, int reps,
                           const float *x, int cold, double *out_seconds) {
    omp_set_num_threads(threads);
    double r = bench_case(c, KERN_SCALAR, threads, n_experts, reps, x, NULL, cold, out_seconds);
    return r;
}

// Plain streaming-read bandwidth over the same GGUF region, as an in-program reference
// point for the kernel GB/s numbers (is the kernel at the memory roofline, or is the
// region simply cache-resident?).
static double bench_membw(const void *base, size_t bytes, int threads, int reps) {
    omp_set_num_threads(threads);
    const uint64_t *p = (const uint64_t*)base;
    const int64_t n = (int64_t)(bytes / 8);
    unsigned long long sink = 0;

    #pragma omp parallel for schedule(static) reduction(+:sink) if(threads > 1)
    for (int64_t i = 0; i < n; ++i) sink += p[i];

    const double t0 = omp_get_wtime();
    for (int r = 0; r < reps; ++r) {
        #pragma omp parallel for schedule(static) reduction(+:sink) if(threads > 1)
        for (int64_t i = 0; i < n; ++i) sink += p[i];
    }
    const double dt = omp_get_wtime() - t0;
    if (sink == 0xdeadbeefULL) fprintf(stderr, "impossible\n"); // keep the sum live
    return (double)reps * (double)bytes / dt / 1e9;
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const char *path = NULL;
    int64_t n_experts_bench = 64;   // distinct experts swept in the "cold" benchmark
    int     reps = 3;
    int     acc_experts = 8;
    int     mt_threads = 8;         // physical cores on the target CPU

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--experts") && i + 1 < argc) n_experts_bench = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--acc-experts") && i + 1 < argc) acc_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) mt_threads = atoi(argv[++i]);
        else if (argv[i][0] != '-') path = argv[i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!path) {
        fprintf(stderr, "usage: %s <model.gguf> [--experts N] [--reps N] [--acc-experts N] [--threads N]\n", argv[0]);
        return 1;
    }

    printf("[P2] loading %s ...\n", path);
    fflush(stdout);
    model_t *m = model_load(path);
    if (!m) { fprintf(stderr, "model_load failed\n"); return 1; }

#if defined(__AVX2__)
    printf("[P2] built with __AVX2__=1, omp_get_max_threads=%d, TUFF_EXACT=%s\n",
           omp_get_max_threads(), getenv("TUFF_EXACT") ? getenv("TUFF_EXACT") : "(unset)");
#else
    printf("[P2] WARNING: built WITHOUT __AVX2__ -- the fast path degenerates to scalar\n");
#endif

    // Layers to exercise: layer 0 (Q4_K down) plus every Q6_K-down layer in the model.
    int q6_layers[8]; int n_q6 = 0;
    for (uint32_t i = 0; i < m->block_count; ++i) {
        if ((ggml_type_t)m->layers[i].ffn_down_exps.info.type == GGML_TYPE_Q6_K &&
            n_q6 < (int)(sizeof(q6_layers)/sizeof(q6_layers[0])))
            q6_layers[n_q6++] = (int)i;
    }
    printf("[P2] model: %u blocks, %u experts, dim=%u, expert_ffn=%u; Q6_K-down layers:",
           m->block_count, m->expert_count, m->embedding_length, m->expert_ffn_length);
    for (int i = 0; i < n_q6; ++i) printf(" %d", q6_layers[i]);
    printf("\n");

    int layers[9]; int n_lay = 0;
    layers[n_lay++] = 0;
    for (int i = 0; i < n_q6; ++i) layers[n_lay++] = q6_layers[i];

    // ---- accuracy -------------------------------------------------------
    printf("\n[P2] ACCURACY -- AVX2 (Q8_K activation) vs exact scalar fp32, real GGUF tensors,\n"
           "     real activations (token embedding -> post-attention rms_norm -> gate/up -> swiglu)\n");
    for (int li = 0; li < n_lay; ++li) {
        const model_layer_t *L = &m->layers[layers[li]];
        tcase_t cs[3] = {
            make_case("gate", &L->ffn_gate_exps, (int64_t)m->expert_count),
            make_case("up",   &L->ffn_up_exps,   (int64_t)m->expert_count),
            make_case("down", &L->ffn_down_exps, (int64_t)m->expert_count),
        };
        const int64_t dim = m->embedding_length;
        const int64_t ffn = cs[2].in_dim;

        float *x = (float*)malloc((size_t)dim * sizeof(float));
        real_activation(m, L, x, 1000);

        accuracy_case(&cs[0], layers[li], acc_experts, x, 0);
        accuracy_case(&cs[1], layers[li], acc_experts, x, 0);

        // Real down-projection inputs: swiglu(gate_e * x, up_e * x) per expert, computed
        // with the exact scalar kernels so the down accuracy figure is not polluted by
        // the gate/up error.
        float *acts = (float*)malloc((size_t)acc_experts * (size_t)ffn * sizeof(float));
        float *g = (float*)malloc((size_t)ffn * sizeof(float));
        float *u = (float*)malloc((size_t)ffn * sizeof(float));
        for (int e = 0; e < acc_experts; ++e) {
            tensor_t vg, vu;
            memset(&vg, 0, sizeof(vg)); memset(&vu, 0, sizeof(vu));
            vg.data = expert_ptr(&cs[0], e); vg.info.type = (uint32_t)cs[0].type;
            vg.info.n_dims = 2; vg.info.dims[0] = cs[0].in_dim; vg.info.dims[1] = cs[0].out_dim;
            vu = vg; vu.data = expert_ptr(&cs[1], e);
            mat_vec_cpu(g, &vg, x);
            mat_vec_cpu(u, &vu, x);
            swiglu(acts + (size_t)e * ffn, g, u, (int)ffn);
        }
        accuracy_case(&cs[2], layers[li], acc_experts, acts, ffn);

        free(g); free(u); free(acts); free(x);
    }

    // ---- memory-bandwidth reference -------------------------------------
    {
        const model_layer_t *L = &m->layers[0];
        tcase_t g = make_case("gate", &L->ffn_gate_exps, (int64_t)m->expert_count);
        int64_t ne = n_experts_bench < g.n_experts ? n_experts_bench : g.n_experts;
        size_t bytes = (size_t)ne * g.expert_bytes;
        (void)touch_experts(&g, ne);
        double b1 = bench_membw(g.base, bytes, 1, 3);
        double bm = bench_membw(g.base, bytes, mt_threads, 3);
        printf("\n[P2] MEMORY REFERENCE -- plain streaming read of %.1f MiB of expert weights "
               "(same region the kernels walk):\n      1T %6.2f GB/s | %dT %6.2f GB/s\n",
               bytes / 1048576.0, b1, mt_threads, bm);
        omp_set_num_threads(1);
    }

    // ---- throughput -----------------------------------------------------
    printf("\n[P2] THROUGHPUT -- GB/s of weight bytes (cold = %lld distinct experts swept per rep,\n"
           "     hot = one expert repeated; reps=%d). Scalar numbers are the shipped fp32 kernels.\n",
           (long long)n_experts_bench, reps);

    for (int li = 0; li < n_lay; ++li) {
        const model_layer_t *L = &m->layers[layers[li]];
        tcase_t cs[3] = {
            make_case("gate", &L->ffn_gate_exps, (int64_t)m->expert_count),
            make_case("up",   &L->ffn_up_exps,   (int64_t)m->expert_count),
            make_case("down", &L->ffn_down_exps, (int64_t)m->expert_count),
        };
        // gate and up are the same shape/type; benchmark gate + down only.
        int which[2] = { 0, 2 };
        for (int wi = 0; wi < 2; ++wi) {
            tcase_t *c = &cs[which[wi]];
            int64_t ne = n_experts_bench < c->n_experts ? n_experts_bench : c->n_experts;

            uint64_t chk = touch_experts(c, ne);

            float *x = (float*)malloc((size_t)c->in_dim * sizeof(float));
            fill_activation(x, c->in_dim, 999u);
            block_q8_K *q8 = (block_q8_K*)malloc(q8_K_row_bytes(c->in_dim));
            quantize_row_q8_K(x, q8, c->in_dim);

            double s1_us = 0, sm_us = 0, a1_us = 0, am_us = 0, ah_us = 0;
            double s1 = bench_scalar(c, 1, ne, reps, x, 1, &s1_us);
            double sm = bench_scalar(c, mt_threads, ne, reps, x, 1, &sm_us);
            omp_set_num_threads(1);
            double a1 = bench_case(c, KERN_AVX2, 1, ne, reps, x, q8, 1, &a1_us);
            double am = bench_case(c, KERN_AVX2, mt_threads, ne, reps, x, q8, 1, &am_us);
            omp_set_num_threads(1);
            double ah = bench_case(c, KERN_AVX2, 1, ne, reps * 8, x, q8, 0, &ah_us);

            printf("  l%-2d %-4s %-4s in=%-5lld out=%-5lld %7.0f KiB/expert | "
                   "scalar 1T %6.2f GB/s (%7.1f us) | scalar %dT %6.2f GB/s (%7.1f us) | "
                   "avx2 1T %6.2f GB/s (%6.1f us) | avx2 %dT %6.2f GB/s (%6.1f us) | "
                   "avx2 1T hot %6.2f GB/s (%6.1f us) | chk=%llu\n",
                   layers[li], c->name, c->type == GGML_TYPE_Q6_K ? "Q6_K" : "Q4_K",
                   (long long)c->in_dim, (long long)c->out_dim, c->expert_bytes / 1024.0,
                   s1, s1_us * 1e6, mt_threads, sm, sm_us * 1e6,
                   a1, a1_us * 1e6, mt_threads, am, am_us * 1e6,
                   ah, ah_us * 1e6, (unsigned long long)chk);
            fflush(stdout);

            free(x); free(q8);
        }
    }

    // ---- the actual P1 workload, measured end to end --------------------
    // What P1 will hand this kernel: `misses` experts per token, each requiring
    // gate + up + swiglu + down, with norm_x quantized to Q8_K ONCE per layer (8 experts).
    // This is measured, not extrapolated from the per-call latencies above.
    printf("\n[P2] P1 WORKLOAD -- full experts (gate + up + swiglu + down) computed end to end,\n"
           "     Q8_K quantization of norm_x amortized over each group of %u experts:\n",
           m->expert_used_count);
    {
        const int64_t dim = m->embedding_length;
        const int64_t ffn = m->expert_ffn_length;
        const int per_layer = (int)m->expert_used_count;

        for (int li = 0; li < n_lay; ++li) {
            const model_layer_t *L = &m->layers[layers[li]];
            tcase_t cg = make_case("gate", &L->ffn_gate_exps, (int64_t)m->expert_count);
            tcase_t cu = make_case("up",   &L->ffn_up_exps,   (int64_t)m->expert_count);
            tcase_t cd = make_case("down", &L->ffn_down_exps, (int64_t)m->expert_count);

            float *x = (float*)malloc((size_t)dim * sizeof(float));
            real_activation(m, L, x, 1000);
            block_q8_K *q8x = (block_q8_K*)malloc(q8_K_row_bytes(dim));
            block_q8_K *q8a = (block_q8_K*)malloc(q8_K_row_bytes(ffn));
            float *g = (float*)malloc((size_t)ffn * sizeof(float));
            float *u = (float*)malloc((size_t)ffn * sizeof(float));
            float *a = (float*)malloc((size_t)ffn * sizeof(float));
            float *d = (float*)malloc((size_t)dim * sizeof(float));

            const size_t bytes_per_expert = 2 * cg.expert_bytes + cd.expert_bytes;
            const int    n_miss = 96;   // 8 experts x 40 layers at the measured 70% hit rate

            for (int tsel = 0; tsel < 2; ++tsel) {
                const int threads = tsel == 0 ? 1 : mt_threads;
                omp_set_num_threads(threads);
                const int par = threads > 1;

                for (int warm = 0; warm < 2; ++warm) {  // pass 0 warms, pass 1 is timed
                    double t0 = omp_get_wtime();
                    for (int i = 0; i < n_miss; ++i) {
                        if (i % per_layer == 0) quantize_row_q8_K(x, q8x, dim);
                        int64_t e = (int64_t)((i * 37) % (int)cg.n_experts); // spread over the blob
                        expert_mat_vec_cpu(g, expert_ptr(&cg, e), cg.type, dim, ffn, x, q8x, par);
                        expert_mat_vec_cpu(u, expert_ptr(&cu, e), cu.type, dim, ffn, x, q8x, par);
                        swiglu(a, g, u, (int)ffn);
                        quantize_row_q8_K(a, q8a, ffn);
                        expert_mat_vec_cpu(d, expert_ptr(&cd, e), cd.type, ffn, dim, a, q8a, par);
                    }
                    double dt = omp_get_wtime() - t0;
                    if (warm == 1) {
                        double mb = (double)n_miss * bytes_per_expert / 1e6;
                        printf("  l%-2d (down=%s) %3d experts = %6.1f MB of weights: "
                               "%7.2f ms @ %2dT  -> %6.2f GB/s, %6.1f us/expert\n",
                               layers[li], cd.type == GGML_TYPE_Q6_K ? "Q6_K" : "Q4_K",
                               n_miss, mb, dt * 1e3, threads,
                               (double)n_miss * bytes_per_expert / dt / 1e9, dt / n_miss * 1e6);
                    }
                }
            }
            free(x); free(q8x); free(q8a); free(g); free(u); free(a); free(d);
            fflush(stdout);
        }
        omp_set_num_threads(1);
    }

    // ---- realistic per-token budget -------------------------------------
    // One expert = gate + up + down. Report the measured cost of a full 8-expert layer
    // and of a whole-token miss load, using the multithreaded cold numbers just measured.
    printf("\n[P2] Q8_K activation quantization cost (once per layer, shared by all experts):\n");
    {
        int64_t k = m->embedding_length;
        float *x = (float*)malloc((size_t)k * sizeof(float));
        fill_activation(x, k, 7u);
        block_q8_K *q8 = (block_q8_K*)malloc(q8_K_row_bytes(k));
        const int n = 100000;
        double t0 = omp_get_wtime();
        for (int i = 0; i < n; ++i) quantize_row_q8_K(x, q8, k);
        double dt = omp_get_wtime() - t0;
        printf("  quantize_row_q8_K(k=%lld): %.3f us/call\n", (long long)k, dt / n * 1e6);
        free(x); free(q8);
    }

    model_free(m);
    return 0;
}
