#include "model.h"
#include "engine.h"
#include "gpu.h"
#include "tokenizer.h"
#include "tuff_exact.h"
#include "chat.h"
#include "drafter.h"
#include "cupti_profiler.h"
#include "tuff_vtable.h"
#include "lfm25.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <sys/resource.h>

#define TF_MAX_BATCH 8   // == LFM25_MAX_BATCH

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --model <path_to_gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>              model GGUF path\n"
        "  --prompt, -p <str>          prompt string for single generation\n"
        "  -c, --ctx-size, --max-seq   maximum context sequence length (default: 4096)\n"
        "  --fill-ram, --fill-vram     inflate context to maximize VRAM utilization\n"
        "  -n <tokens>                 number of tokens to generate (default: 128)\n"
        "  --warmup <tokens>           warmup tokens (default: 32)\n"
        "  --runs <n>                  benchmark runs (default: 3)\n"
        "  --tf-batch <B>              with --teacher-force: feed the reference in chunks of B tokens (LFM2.5 batched forward)\n"
        "  --tf-rollback               with --tf-batch B>=2: roll back to 1+(chunk%%B) tokens after every chunk and re-feed\n"
        "\n"
        "Interactive TUI Chat Options:\n"
        "  --chat, -i, --interactive   launch interactive terminal chat mode\n"
        "  --system <prompt>           system prompt\n"
        "  --log, --log-file <path>    log conversation to markdown file\n"
        "\n"
        "Speculative Decoding Options:\n"
        "  --spec-draft <mode>         speculation draft: ngram, mtp, ngram+mtp, external\n"
        "  --spec-ngram <n>            n-gram order for ngram draft (default: 3)\n"
        "  --spec-ngram-min <n>        minimum n-gram fallback order (default: 2)\n"
        "  --spec-k <K>                LFM engines: max draft tokens per verify step, 1..7 (default: 4; LFM25_SPEC_K=<k> fixes k)\n"
        "  --spec-mtp                  enable MTP head speculation\n"
        "Memory / KV Cache Options:\n"
        "  --kv-quant <mode>           KV cache quant mode: f32 (native), q8_0 (default), q4_0\n"
        "\n"
        "Diagnostic / Benchmark Options:\n"
        "  --inspect                   print model/hparam inventory and exit\n"
        "  --bench-dense-batch         time dense Q8_0 projections\n"
        "  --dump-tokens               oracle mode: print token IDs to stdout\n"
        "  --dump-logits <path>        numeric-gate mode: dump final logit vectors\n"
        "  --teacher-force <file>      feed reference token IDs, report per-step top-1 agreement\n"
        "                              (with --dump-logits: also dump every step's logits)\n"
        "\n"
        "Environment:\n"
        "  TUFF_KV_QUANT=<mode>     KV cache quant: f32, q8_0 (default), q4_0\n"
        "  TUFF_EXACT=1             force bit-exact reference kernels; unset/0 uses fast path\n"
        "  TUFF_EXPERT_SLOTS=<n>    override expert cache pool slot count\n"
        "  TUFF_VRAM_MARGIN_MB=<n>  safety headroom in MiB (default: 256)\n",
        prog);
}

// ---------------------------------------------------------------------------
// P0: cached TUFF_EXACT flag (declared in include/tuff_exact.h).
// getenv() is a linear scan of environ, so it is read exactly once and cached. The benign
// race on first touch is harmless: every racing reader computes the same value.
static int g_tuff_exact = -1;

int tuff_exact(void) {
    int v = g_tuff_exact;
    if (v < 0) {
        const char *e = getenv("TUFF_EXACT");
        v = (e && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        g_tuff_exact = v;
    }
    return v;
}

// ---------------------------------------------------------------------------
// P0: final-logit capture sink (declared in include/tuff_exact.h).
// Disarmed by default; tuff_logit_capture() then costs a single predicate.
static float   *g_logit_buf = NULL;   // NULL => capture disabled
static uint32_t g_logit_cap = 0;      // allocated floats
static uint32_t g_logit_n   = 0;      // floats in the last captured vector (0 => none pending)
static int      g_logit_on  = 0;

int tuff_logit_capture_enable(int on) {
    g_logit_on = on ? 1 : 0;
    g_logit_n = 0;
    if (!g_logit_on) {
        free(g_logit_buf);
        g_logit_buf = NULL;
        g_logit_cap = 0;
    }
    return 0;
}

int tuff_logit_capture_is_enabled(void) {
    return g_logit_on;
}

void tuff_logit_capture(const float *logits, uint32_t n) {
    if (!g_logit_on || !logits || n == 0) return;
    if (n > g_logit_cap) {
        float *nb = (float*)realloc(g_logit_buf, (size_t)n * sizeof(float));
        if (!nb) {
            fprintf(stderr, "[Tuff] FATAL: out of memory capturing %u logits\n", n);
            exit(EXIT_FAILURE);
        }
        g_logit_buf = nb;
        g_logit_cap = n;
    }
    memcpy(g_logit_buf, logits, (size_t)n * sizeof(float));
    g_logit_n = n;
}

// Consumes the pending vector: returns NULL if no capture happened since the previous take,
// which is how --dump-logits detects a missing engine-side hook instead of writing zeros.
const float* tuff_logit_capture_take(uint32_t *n_out) {
    uint32_t n = g_logit_n;
    g_logit_n = 0;
    if (n_out) *n_out = n;
    return n ? g_logit_buf : NULL;
}

// getrusage ru_maxrss is in KB on Linux.
static double rss_mib(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
    return (double)ru.ru_maxrss / 1024.0;
}

// Read current VmRSS from /proc/self/status (MiB). -1 if unavailable.
static double vmrss_mib(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1.0;
    char line[256];
    double vmrss = -1.0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmRSS: %lf kB", &vmrss) == 1) {
            vmrss /= 1024.0;
            break;
        }
    }
    fclose(f);
    return vmrss;
}

// ---------------------------------------------------------------------------------------------------------------
// Lossless n-gram (prompt-lookup) speculative decoding for the LFM2.5 engines (Task 5.2). Greedy only.
// Draft = continuation of the most recent earlier occurrence of the last n tokens (n = nmax..nmin) of the token
// history; verification = one engine_forward_batch([t, d1..dk]) whose per-position argmax is bit-identical to k+1
// consecutive forward_token calls, so the emitted stream equals the non-speculative one exactly. A step without a
// draft uses the (CUDA-graphed) engine_forward_token, never forward_batch(1).
// Adaptive-k policy (LFM25_SPEC_K=<1..7> fixes k instead): with a draft of length dl and a per-position acceptance
// estimate p (EMA of accepted/attempted per match-length bucket, with a prior), choose k in 0..dl maximising
//   throughput(k) = (1 + p + p^2 + ... + p^k) / cost(k+1 tokens in one batch)      (k >= 1)
//   throughput(0) = 1 / cost(one graphed single token)
// where cost() is a table measured once at startup. Ties and no-gain cases choose k = 0 (no speculation).
// ---------------------------------------------------------------------------------------------------------------
#define SPEC_KMAX 7
#define SPEC_NBUCKET 6
#define SPEC_DECAY 0.85
#define SPEC_PRIOR_W 2.0
static const double SPEC_PRIOR_P[SPEC_NBUCKET] = {0.35, 0.45, 0.55, 0.70, 0.85, 0.92};

static double g_spec_cost[SPEC_KMAX + 2];   // [1] = graphed single token, [n] = forward_batch(n), seconds
static int    g_spec_cost_measured = 0;

typedef struct {
    const engine_vtable_t *vt;
    void *eng;
    int32_t *hist;
    int n_hist, cap;
    int nmax, nmin, kmax, k_fixed, ctx;
    double acc[SPEC_NBUCKET], att[SPEC_NBUCKET];
    double gain, t_first;   // guard: seconds saved (+) / lost (-) by batch steps vs single-token decoding
    unsigned long long guarded, n_single;
    double m1, mb;   // live cost scales (EMA): m1 = measured/table single-token time; mb = measured batch time / (table batch time x m1)
    unsigned long long steps, drafted, accepted, no_spec_with_draft, hist_k[SPEC_KMAX + 1];
} spec_t;

static int spec_cmp_double(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

// Static cost table (also the fallback whenever measuring fails): batch(n) = 0.9 n single-token times (the measured
// 8B/24B MoE verify cost, where experts run per token and only the trunk is shared), single = 1 unit, scaled to a
// nominal ~30 ms/token (only ratios matter; the live cost scale corrects the absolute level).
static void spec_static_costs(void) {
    for (int n = 1; n <= SPEC_KMAX + 1; ++n) g_spec_cost[n] = 0.9 * n;
    g_spec_cost[1] = 1.0;
    for (int n = 1; n <= SPEC_KMAX + 1; ++n) g_spec_cost[n] *= 0.030;
    g_spec_cost_measured = 0;
}

// Measure the verify-cost table on a scratch engine (the real engine's state stays untouched). Dense engines measure
// the real table; MoE engines keep the static one (measuring on a scratch engine perturbs the 24B expert cache) and
// rely on the live cost scale. Any failure (context too small, engine init / forward error, non-positive timing)
// falls back to the static table, so no cost is ever zero (the policy divides by them).
static void spec_measure_costs(const engine_vtable_t *vt, void *model, int ctx, const sampler_params_t *sp,
                               const int32_t *prompt, int n_prompt, int kmax, int do_measure) {
    spec_static_costs();
    if (!do_measure) return;
    int sum_n = 0;
    for (int n = 2; n <= kmax + 1; ++n) sum_n += n;
    int reps = 5;
    while (reps >= 1 && n_prompt + 12 + (reps + 1) * sum_n + 8 > ctx) reps--;
    if (reps < 1) {
        fprintf(stderr, "[Tuff] spec: context too small to measure the cost table; using the static table\n");
        return;
    }
    void *e = vt->engine_init(model, ctx);
    if (!e) {
        fprintf(stderr, "[Tuff] spec: scratch engine init failed; using the static cost table\n");
        return;
    }
    vt->engine_set_sampler(e, sp);
    double cost[SPEC_KMAX + 2] = {0};
    int ok = 1;
    int pos = 0;
    int32_t t = prompt[0];
    for (int i = 0; ok && i < n_prompt; ++i) { t = vt->engine_forward_token(e, prompt[i], pos++); if (t < 0) ok = 0; }
    int32_t toks[SPEC_KMAX + 1], am[SPEC_KMAX + 1];
    for (int i = 0; ok && i < 3; ++i) { t = vt->engine_forward_token(e, prompt[i % n_prompt], pos++); if (t < 0) ok = 0; }   // untimed warm-up
    double t0 = omp_get_wtime();
    for (int i = 0; ok && i < 8; ++i) { t = vt->engine_forward_token(e, prompt[i % n_prompt], pos++); if (t < 0) ok = 0; }
    cost[1] = (omp_get_wtime() - t0) / 8.0;
    for (int n = 2; ok && n <= kmax + 1; ++n) {
        double ts[5] = {0};
        for (int r = 0; ok && r < reps + 1; ++r) {
            for (int j = 0; j < n; ++j) toks[j] = prompt[(r + j) % n_prompt];
            double a = omp_get_wtime();
            if (vt->engine_forward_batch(e, toks, n, pos, am, NULL) != 0) { ok = 0; break; }
            double b = omp_get_wtime();
            pos += n;
            if (r > 0) ts[r - 1] = b - a;   // first call per n is untimed
        }
        if (!ok) break;
        qsort(ts, (size_t)reps, sizeof(double), spec_cmp_double);
        cost[n] = ts[reps / 2];
    }
    vt->engine_free(e);
    for (int n = 1; ok && n <= kmax + 1; ++n) if (!(cost[n] > 0.0)) ok = 0;
    if (!ok) {
        fprintf(stderr, "[Tuff] spec: cost-table measurement failed; using the static cost table\n");
        return;
    }
    for (int n = 1; n <= kmax + 1; ++n) g_spec_cost[n] = cost[n];
    g_spec_cost_measured = 1;
}

static void spec_print_policy(const spec_t *s) {
    fprintf(stderr, "[Tuff] spec: lossless n-gram speculation, orders %d..%d, k_max %d, %s\n", s->nmax, s->nmin, s->kmax,
            s->k_fixed ? "FIXED k (LFM25_SPEC_K)" : "adaptive k");
    if (!s->k_fixed) {
        fprintf(stderr, "[Tuff] spec policy: per step pick k in 0..draft_len maximising (1+p+..+p^k)/cost(k+1) vs 1/cost(1 graphed token); "
                        "p = acceptance EMA (decay %.2f, prior weight %.0f) per match-length bucket; costs = startup table x live measured/table scale (EMA) for single and batch steps\n", SPEC_DECAY, SPEC_PRIOR_W);
        fprintf(stderr, "[Tuff] spec cost table (%s, ms): single %.3f", g_spec_cost_measured ? "measured" : "static, nominal ms",
                g_spec_cost[1] * 1e3);
        for (int n = 2; n <= s->kmax + 1; ++n) fprintf(stderr, ", batch%d %.3f (%.2fx single-token time)", n, g_spec_cost[n] * 1e3,
                                                     g_spec_cost[n] / (g_spec_cost[1] * n));
        fprintf(stderr, "\n");
    }
}

static void spec_init(spec_t *s, const engine_vtable_t *vt, int ctx, int nmax, int nmin, int kmax, int k_fixed) {
    memset(s, 0, sizeof(*s));
    s->vt = vt; s->ctx = ctx; s->nmax = nmax; s->nmin = nmin; s->kmax = kmax; s->k_fixed = k_fixed;
    s->cap = ctx + 16;
    s->hist = (int32_t*)malloc((size_t)s->cap * sizeof(int32_t));
}

// Reset per-run state for a fresh engine.
static void spec_begin_run(spec_t *s, void *eng) {
    s->eng = eng; s->n_hist = 0; s->m1 = 1.0;
    s->mb = g_spec_cost_measured ? 1.0 : 0.6;   // static table: optimistic start, the live scale corrects it
    s->gain = 0.0; s->t_first = -1.0; s->guarded = 0; s->n_single = 0;
    for (int b = 0; b < SPEC_NBUCKET; ++b) { s->acc[b] = 0.0; s->att[b] = 0.0; }
    s->steps = s->drafted = s->accepted = s->no_spec_with_draft = 0;
    memset(s->hist_k, 0, sizeof(s->hist_k));
}

static void spec_push(spec_t *s, int32_t t) {
    if (s->n_hist < s->cap) s->hist[s->n_hist++] = t;
}

static int spec_bucket(int ml) {
    return ml <= 2 ? 0 : ml == 3 ? 1 : ml <= 5 ? 2 : ml <= 8 ? 3 : ml <= 16 ? 4 : 5;
}

// Prompt lookup: returns the number of draft tokens written to d (0 = no match); *ml = match length (capped at 32).
static int spec_lookup(const spec_t *s, int32_t *d, int kcap, int *ml_out) {
    const int32_t *h = s->hist;
    const int L = s->n_hist;
    for (int n = s->nmax; n >= s->nmin; --n) {
        if (L < n + 1) continue;
        for (int j = L - 2; j >= n - 1; --j) {   // j = last index of the candidate occurrence; j+1 <= L-1 exists
            int i = 0;
            while (i < n && h[j - i] == h[L - 1 - i]) i++;
            if (i < n) continue;
            int ml = n;
            while (ml < 32 && j - ml >= 0 && h[j - ml] == h[L - 1 - ml]) ml++;
            int k = L - 1 - j;
            if (k > kcap) k = kcap;
            for (int q = 0; q < k; ++q) d[q] = h[j + 1 + q];
            *ml_out = ml;
            return k;
        }
    }
    return 0;
}

static double spec_p(const spec_t *s, int b) {
    return (s->acc[b] + SPEC_PRIOR_P[b] * SPEC_PRIOR_W) / (s->att[b] + SPEC_PRIOR_W);
}

static int spec_choose_k(const spec_t *s, int dl, int b) {
    if (s->k_fixed) return dl < s->k_fixed ? dl : s->k_fixed;
    const double p = spec_p(s, b);
    int best = 0;
    double best_rate = 1.0 / (g_spec_cost[1] * s->m1);
    double pw = 1.0, sum = 1.0;
    for (int k = 1; k <= dl; ++k) {
        pw *= p; sum += pw;
        double rate = sum / (g_spec_cost[k + 1] * s->m1 * s->mb);
        if (rate > best_rate) { best_rate = rate; best = k; }
    }
    return best;
}

// One decoding step from token t at position pos with rem >= 1 tokens still wanted. Writes 1..rem tokens to out
// (out[0] follows t); the state afterwards is "t and out[0..n-2] processed", out[n-1] is the next input token.
static int spec_step(spec_t *s, int32_t t, int pos, int rem, int32_t *out) {
    const engine_vtable_t *vt = s->vt;
    if (s->t_first < 0.0) s->t_first = omp_get_wtime();
    int kcap = s->kmax < rem - 1 ? s->kmax : rem - 1;
    if (kcap > s->ctx - 1 - pos) kcap = s->ctx - 1 - pos;
    int32_t d[SPEC_KMAX];
    int ml = 0, dl = 0, k = 0, b = 0;
    if (kcap > 0 && (g_spec_cost_measured || s->k_fixed || s->n_single >= 8)) {   // static table: calibrate m1 on 8 single steps first
        dl = spec_lookup(s, d, kcap, &ml);
        if (dl > 0) {
            b = spec_bucket(ml);
            k = spec_choose_k(s, dl, b);
            // Loss guard: the worst case of a batch step (every draft rejected: one token for cost(k+1)) must fit in the
            // budget  net gain so far + 0.5% of the elapsed time + one single-token time;  k shrinks until it does, so
            // probing is rate-limited to ~0.5% of the run when drafts keep failing and free when they pay.
            if (k > 0 && !s->k_fixed) {
                const double c1 = g_spec_cost[1] * s->m1;
                const double budget = s->gain + 0.005 * (omp_get_wtime() - s->t_first) + 1.0 * c1;
                while (k > 0 && g_spec_cost[k + 1] * s->m1 * s->mb - c1 > budget) k--;
                if (k == 0) s->guarded++;
            }
            if (k == 0) {   // draft skipped: let the evidence fade toward the prior so the bucket gets re-probed
                s->acc[b] *= 0.9; s->att[b] *= 0.9;
                s->no_spec_with_draft++;
            }
        }
    }
    s->steps++;
    s->hist_k[k]++;
    const double t0 = omp_get_wtime();
    if (k == 0) {
        out[0] = vt->engine_forward_token(s->eng, t, pos);
        if (out[0] < 0) {
            fprintf(stderr, "[Tuff] FATAL: engine_forward_token failed at position %d\n", pos);
            exit(3);
        }
        spec_push(s, out[0]);
        if (!s->k_fixed) {
            const double x = (omp_get_wtime() - t0) / g_spec_cost[1];
            if (!g_spec_cost_measured && s->n_single == 0) s->m1 = x;   // static table is in nominal units: adopt the first reading
            else { const double a = s->n_single < 19 ? 1.0 / (double)(s->n_single + 1) : 0.05; s->m1 += a * (x - s->m1); }
            if (s->m1 < 1e-9) s->m1 = 1e-9;   // the policy divides by it
            s->n_single++;
        }
        return 1;
    }
    int32_t toks[SPEC_KMAX + 1], am[SPEC_KMAX + 1];
    toks[0] = t;
    for (int i = 0; i < k; ++i) toks[1 + i] = d[i];
    if (vt->engine_forward_batch(s->eng, toks, k + 1, pos, am, NULL) != 0) {
        fprintf(stderr, "[Tuff] FATAL: speculative engine_forward_batch failed at position %d (k=%d)\n", pos, k);
        exit(1);
    }
    int m = 0;
    while (m < k && d[m] == am[m]) m++;
    for (int i = 0; i <= m; ++i) { out[i] = am[i]; spec_push(s, am[i]); }
    if (m < k && vt->engine_rollback(s->eng, m + 1) != 0) {
        fprintf(stderr, "[Tuff] FATAL: speculative engine_rollback(%d) failed at position %d\n", m + 1, pos);
        exit(1);
    }
    if (!s->k_fixed) {
        const double dt = omp_get_wtime() - t0;
        s->mb = 0.8 * s->mb + 0.2 * (dt / (g_spec_cost[k + 1] * s->m1));
        s->gain += (double)(m + 1) * g_spec_cost[1] * s->m1 - dt;
    }
    s->drafted += (unsigned)k;
    s->accepted += (unsigned)m;
    s->acc[b] = s->acc[b] * SPEC_DECAY + m;
    s->att[b] = s->att[b] * SPEC_DECAY + m + (m < k ? 1 : 0);
    return m + 1;
}

static void spec_print_stats(const spec_t *s, unsigned long long tokens) {
    fprintf(stderr, "[Tuff] spec stats: steps=%llu drafted=%llu accepted=%llu acceptance=%.1f%% tokens/step=%.3f "
                    "(draft available but skipped: %llu) k-hist:",
            s->steps, s->drafted, s->accepted, s->drafted ? 100.0 * (double)s->accepted / (double)s->drafted : 0.0,
            s->steps ? (double)tokens / (double)s->steps : 0.0, s->no_spec_with_draft);
    for (int k = 0; k <= s->kmax; ++k) fprintf(stderr, " k%d=%llu", k, s->hist_k[k]);
    if (!s->k_fixed) fprintf(stderr, " | live cost scale single=%.3f batch/single=%.2f, net gain %.1f ms, loss-guard skips %llu", s->m1, s->mb, s->gain * 1e3, s->guarded);
    fprintf(stderr, "\n");
}

static int arch_is_lfm(tuff_arch_t a) { return a == TUFF_ARCH_LFM2 || a == TUFF_ARCH_LFM2MOE; }

// The context window must hold every position a run will feed: `need` positions into a window of `ctx`.
static int ctx_fits(const char *what, long need, int ctx) {
    if (need <= (long)ctx) return 1;
    fprintf(stderr, "[Tuff] ERROR: %s needs %ld context positions but the context window (-c) is %d; "
                    "raise -c to at least %ld or reduce the prompt/-n\n", what, need, ctx, need);
    return 0;
}

// A forward step that returned an error (-1) must abort the run, never feed garbage tokens onward.
static void fwd_check(int is_lfm, int32_t t, int pos) {
    if (is_lfm && t < 0) {
        fprintf(stderr, "[Tuff] FATAL: engine_forward_token failed at position %d\n", pos);
        exit(3);
    }
}

int main(int argc, char **argv) {
    const char *model_path = "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf";
    const char *prompt = "Hello";
    bool prompt_specified = false;
    bool chat_mode = false;
    const char *system_prompt = "You are a helpful, respectful and honest AI assistant.";
    const char *log_filepath = NULL;
    int max_context_len = 131072;
    int max_gen_tokens = 1024;
    bool fill_ram = false;
    bool inspect_mode = false;
    bool bench_dense_batch = false;
    bool dump_tokens_mode = false;
    bool spec_mtp_mode = true;
    bool spec_measure_mode = false;
    bool spec_ngram_requested = false;   // explicit --spec-draft ngram (LFM engines only act on an explicit request)
    int  spec_k_max = 4;                 // --spec-k (LFM engines)
    const char *spec_draft_model = NULL;
    const char *dump_logits_path = NULL; // non-NULL => --dump-logits mode
    int n_predict = -1; // resolved below based on mode
    int n_warmup = 32;
    int n_runs = 3;
    int n_gpu_expert_layers = 0; // 0 => auto (computed below); set g_n_gpu_expert_layers in gpu.h
    sampler_params_t sampler_params = sampler_params_default();
    const float default_temp = sampler_params.temp;

    const char *teacher_force_path = NULL; // non-NULL => --teacher-force mode
    int tf_batch = 1;                      // --tf-batch B: teacher-force in chunks of B tokens through engine_forward_batch
    bool tf_rollback = false;              // --tf-rollback: after each chunk roll back to 1 + (chunk % B) tokens and re-feed

    const char *kvq_env = getenv("TUFF_KV_QUANT");
    if (kvq_env && kvq_env[0] != '\0') {
        if (strcmp(kvq_env, "f32") == 0 || strcmp(kvq_env, "fp32") == 0) {
            gpu_set_kv_quant(KV_QUANT_F32);
        } else if (strcmp(kvq_env, "q8_0") == 0 || strcmp(kvq_env, "q8") == 0) {
            gpu_set_kv_quant(KV_QUANT_Q8_0);
        } else if (strcmp(kvq_env, "q4_0") == 0 || strcmp(kvq_env, "q4") == 0) {
            gpu_set_kv_quant(KV_QUANT_Q4_0);
        }
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--chat") == 0 || strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            chat_mode = true;
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if ((strcmp(argv[i], "--log") == 0 || strcmp(argv[i], "--log-file") == 0) && i + 1 < argc) {
            log_filepath = argv[++i];
        } else if ((strcmp(argv[i], "--max-seq") == 0 || strcmp(argv[i], "-c") == 0 ||
                    strcmp(argv[i], "--ctx-size") == 0 || strcmp(argv[i], "--ctx") == 0 ||
                    strcmp(argv[i], "--context-size") == 0) && i + 1 < argc) {
            max_context_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fill-ram") == 0 || strcmp(argv[i], "--fill-vram") == 0 ||
                   strcmp(argv[i], "--fill_ram") == 0 || strcmp(argv[i], "--fill_vram") == 0) {
            fill_ram = true;
        } else if (strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc) {
            max_gen_tokens = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "-p") == 0) && i + 1 < argc) {
            prompt = argv[++i];
            prompt_specified = true;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            n_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--spec-draft-model") == 0 && i + 1 < argc) {
            spec_draft_model = argv[++i];
            spec_mtp_mode = true;
            g_spec_mtp = 1;
            g_spec_draft = SPEC_DRAFT_EXTERNAL;
        } else if (strcmp(argv[i], "--spec-mtp") == 0) {
            spec_mtp_mode = true;
            g_spec_mtp = 1;
        } else if (strncmp(argv[i], "--spec-draft=", 13) == 0) {
            spec_mtp_mode = true;
            g_spec_mtp = 1;
            const char *s = argv[i] + 13;
            spec_ngram_requested = (strcmp(s, "ngram") == 0);
            if (strcmp(s, "ngram") == 0) {
                g_spec_draft = SPEC_DRAFT_NGRAM;
            } else if (strcmp(s, "ngram+mtp") == 0 || strcmp(s, "mtp+ngram") == 0) {
                g_spec_draft = SPEC_DRAFT_NGRAM_MTP;
            } else if (strcmp(s, "mtp") == 0) {
                g_spec_draft = SPEC_DRAFT_MTP;
            } else if (strcmp(s, "external") == 0) {
                g_spec_draft = SPEC_DRAFT_EXTERNAL;
            } else if (strcmp(s, "none") == 0 || strcmp(s, "off") == 0 || strcmp(s, "0") == 0) {
                g_spec_draft = SPEC_DRAFT_NONE;
                spec_mtp_mode = false;
                g_spec_mtp = 0;
            } else {
                fprintf(stderr, "Error: --spec-draft must be ngram, mtp, ngram+mtp, external or none (got '%s')\n", s);
                return 2;
            }
        } else if (strcmp(argv[i], "--no-spec") == 0) {
            spec_ngram_requested = false;
            g_spec_draft = SPEC_DRAFT_NONE;
            spec_mtp_mode = false;
            g_spec_mtp = 0;
        } else if (strcmp(argv[i], "--spec-draft") == 0 && i + 1 < argc) {
            spec_mtp_mode = true;
            g_spec_mtp = 1;
            const char *s = argv[++i];
            spec_ngram_requested = (strcmp(s, "ngram") == 0);
            if (strcmp(s, "ngram") == 0) {
                g_spec_draft = SPEC_DRAFT_NGRAM;
            } else if (strcmp(s, "ngram+mtp") == 0 || strcmp(s, "mtp+ngram") == 0) {
                g_spec_draft = SPEC_DRAFT_NGRAM_MTP;
            } else if (strcmp(s, "mtp") == 0) {
                g_spec_draft = SPEC_DRAFT_MTP;
            } else if (strcmp(s, "external") == 0) {
                g_spec_draft = SPEC_DRAFT_EXTERNAL;
            } else if (strcmp(s, "none") == 0 || strcmp(s, "off") == 0 || strcmp(s, "0") == 0) {
                g_spec_draft = SPEC_DRAFT_NONE;
                spec_mtp_mode = false;
                g_spec_mtp = 0;
            } else {
                fprintf(stderr, "Error: --spec-draft must be ngram, mtp, ngram+mtp, external or none (got '%s')\n", s);
                return 2;
            }
        } else if ((strcmp(argv[i], "--spec-ngram") == 0 || strcmp(argv[i], "--spec-ngram-max") == 0) && i + 1 < argc) {
            g_spec_ngram_n = atoi(argv[++i]);
            if (g_spec_ngram_n < 1 || g_spec_ngram_n > 8) {
                fprintf(stderr, "Error: --spec-ngram must be between 1 and 8 (got %d)\n", g_spec_ngram_n);
                return 2;
            }
        } else if (strcmp(argv[i], "--spec-ngram-min") == 0 && i + 1 < argc) {
            g_spec_ngram_min = atoi(argv[++i]);
            if (g_spec_ngram_min < 1 || g_spec_ngram_min > 8) {
                fprintf(stderr, "Error: --spec-ngram-min must be between 1 and 8 (got %d)\n", g_spec_ngram_min);
                return 2;
            }
        } else if ((strcmp(argv[i], "--spec-draft-len") == 0 || strcmp(argv[i], "--draft-len") == 0) && i + 1 < argc) {
            g_spec_draft_len = atoi(argv[++i]);
            if (g_spec_draft_len < 1 || g_spec_draft_len > 3) {
                fprintf(stderr, "Error: --spec-draft-len must be between 1 and 3 (got %d)\n", g_spec_draft_len);
                return 2;
            }
        } else if (strcmp(argv[i], "--spec-k") == 0 && i + 1 < argc) {
            spec_k_max = atoi(argv[++i]);
            if (spec_k_max < 1 || spec_k_max > SPEC_KMAX) {
                fprintf(stderr, "Error: --spec-k must be between 1 and %d (got %d)\n", SPEC_KMAX, spec_k_max);
                return 2;
            }
        } else if (strcmp(argv[i], "--spec-measure") == 0) {
            spec_mtp_mode = true;
            spec_measure_mode = true;
            g_spec_mtp = 1;
        } else if (strcmp(argv[i], "--n-gpu-expert-layers") == 0 && i + 1 < argc) {
            n_gpu_expert_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--inspect") == 0) {
            inspect_mode = true;
        } else if (strcmp(argv[i], "--bench-dense-batch") == 0) {
            bench_dense_batch = true;
        } else if (strcmp(argv[i], "--dump-tokens") == 0) {
            dump_tokens_mode = true;
        } else if (strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits_path = argv[++i];
        } else if (strcmp(argv[i], "--dump-logits") == 0) {
            fprintf(stderr, "Error: --dump-logits requires a path argument\n");
            print_usage(argv[0]);
        } else if (strncmp(argv[i], "--kv-quant=", 11) == 0) {
            const char *s = argv[i] + 11;
            if (strcmp(s, "f32") == 0 || strcmp(s, "fp32") == 0) {
                gpu_set_kv_quant(KV_QUANT_F32);
            } else if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8") == 0) {
                gpu_set_kv_quant(KV_QUANT_Q8_0);
            } else if (strcmp(s, "q4_0") == 0 || strcmp(s, "q4") == 0) {
                gpu_set_kv_quant(KV_QUANT_Q4_0);
            } else {
                fprintf(stderr, "Error: --kv-quant must be f32, q8_0, or q4_0 (got '%s')\n", s);
                return 2;
            }
        } else if (strcmp(argv[i], "--kv-quant") == 0 && i + 1 < argc) {
            const char *s = argv[++i];
            if (strcmp(s, "f32") == 0 || strcmp(s, "fp32") == 0) {
                gpu_set_kv_quant(KV_QUANT_F32);
            } else if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8") == 0) {
                gpu_set_kv_quant(KV_QUANT_Q8_0);
            } else if (strcmp(s, "q4_0") == 0 || strcmp(s, "q4") == 0) {
                gpu_set_kv_quant(KV_QUANT_Q4_0);
            } else {
                fprintf(stderr, "Error: --kv-quant must be f32, q8_0, or q4_0 (got '%s')\n", s);
                return 2;
            }
        } else if (strcmp(argv[i], "--tf-batch") == 0 && i + 1 < argc) {
            tf_batch = atoi(argv[++i]);
            if (tf_batch < 1 || tf_batch > TF_MAX_BATCH) {
                fprintf(stderr, "Error: --tf-batch must be 1..%d\n", TF_MAX_BATCH);
                return 2;
            }
        } else if (strcmp(argv[i], "--tf-rollback") == 0) {
            tf_rollback = true;
        } else if (strcmp(argv[i], "--teacher-force") == 0 && i + 1 < argc) {
            teacher_force_path = argv[++i];
        } else if (strcmp(argv[i], "--teacher-force") == 0) {
            fprintf(stderr, "Error: --teacher-force requires a path argument\n");
            print_usage(argv[0]);
            return 2;
        } else if ((strcmp(argv[i], "--temp") == 0 || strcmp(argv[i], "--temperature") == 0) && i + 1 < argc) {
            sampler_params.temp = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--top-p") == 0 || strcmp(argv[i], "--top_p") == 0) && i + 1 < argc) {
            sampler_params.top_p = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "--top_k") == 0) && i + 1 < argc) {
            sampler_params.top_k = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--min-p") == 0 || strcmp(argv[i], "--min_p") == 0) && i + 1 < argc) {
            sampler_params.min_p = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--repeat-penalty") == 0 || strcmp(argv[i], "--repeat_penalty") == 0) && i + 1 < argc) {
            sampler_params.repeat_penalty = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--repeat-last-n") == 0 || strcmp(argv[i], "--repeat_last_n") == 0) && i + 1 < argc) {
            sampler_params.repeat_last_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            sampler_params.seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (max_context_len < 1 || max_context_len > 262144) {
        fprintf(stderr, "Error: --max-seq must be between 1 and 262144 (got %d)\n", max_context_len);
        return 2;
    }

    if (!prompt_specified && !inspect_mode && !bench_dense_batch &&
        !dump_tokens_mode && !dump_logits_path && !teacher_force_path) {
        chat_mode = true;
    }

    if (n_predict < 0) n_predict = (dump_tokens_mode || dump_logits_path) ? 32 : 128;
    if (n_runs < 1 || n_warmup < 0) {
        fprintf(stderr, "Error: --runs must be >= 1 and --warmup >= 0 (got --runs %d --warmup %d)\n", n_runs, n_warmup);
        return 2;
    }
    if (n_predict <= 0) {
        fprintf(stderr, "Error: -n must be >= 1 (got %d)\n", n_predict);
        return 2;
    }

    // Resolve TUFF_EXACT once, before any worker thread exists, and state it up front so
    // every log line is attributable to a path (parity.md sec.3).
    fprintf(stderr, "[Tuff] TUFF_EXACT=%d (%s)\n", tuff_exact(),
            tuff_exact() ? "bit-exact reference kernels" : "fast paths permitted");

    // Iteration-12 static expert split: --n-gpu-expert-layers N pre-fills the full expert sets
    // of the first N layers into VRAM (guaranteed resident, never miss). Default 0 = no prefill;
    // the engine still overlaps resident/LFRU-hit experts on the GPU with miss experts on the CPU
    // within each layer. Negative/absent = 0.
    g_n_gpu_expert_layers = n_gpu_expert_layers < 0 ? 0 : n_gpu_expert_layers; // extern in gpu.h
    fprintf(stderr, "[Tuff] expert split: %d layers pre-resident on GPU "
                    "(0 = none, hit/miss CPU overlap on)\n", g_n_gpu_expert_layers);

    tuff_arch_t arch = tuff_detect_arch(model_path);
    const engine_vtable_t *vtable = tuff_get_vtable(arch);
    if (!vtable) {
        fprintf(stderr, "[Tuff] ERROR: Unsupported or unrecognized architecture in %s\n", model_path);
        return 1;
    }
    fprintf(stderr, "[Tuff] Detected architecture: %s (%s engine)\n",
            tuff_arch_to_string(arch), vtable->name);

    const int is_lfm = arch_is_lfm(arch);

    // The LFM2.5 engines decode greedily only (the sampler is stored but never consulted).
    if (is_lfm && sampler_params.temp > 0.0f) {
        if (default_temp <= 0.0f) {
            fprintf(stderr, "[Tuff] ERROR: sampling not implemented for LFM2.5 (--temp %g); use --temp 0\n", (double)sampler_params.temp);
            return 2;
        }
        fprintf(stderr, "[Tuff] WARNING: LFM2.5 decodes greedily; --temp %g (default %g) is ignored\n",
                (double)sampler_params.temp, (double)default_temp);
    }

    // LFM engines: lossless n-gram speculation is opt-in via an explicit --spec-draft ngram (greedy only).
    bool lfm_spec = false;
    int spec_nmax = g_spec_ngram_n, spec_nmin = g_spec_ngram_min, spec_kfix = 0;
    if (spec_ngram_requested && vtable->engine_forward_batch && vtable->engine_rollback) {
        {
            lfm_spec = true;
            if (spec_nmin > spec_nmax) spec_nmin = spec_nmax;
            const char *ks = getenv("LFM25_SPEC_K");
            if (ks && ks[0]) {
                spec_kfix = atoi(ks);
                if (spec_kfix < 1 || spec_kfix > SPEC_KMAX) {
                    fprintf(stderr, "Error: LFM25_SPEC_K must be between 1 and %d (got '%s')\n", SPEC_KMAX, ks);
                    return 2;
                }
                if (spec_kfix > spec_k_max) spec_k_max = spec_kfix;
            }
        }
    }

    fprintf(stderr, "[Tuff] Loading GGUF model from: %s\n", model_path);
    void *model = vtable->model_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }

    // Iteration 11: warm the model weight bytes (especially the lazy expert region) into
    // RAM before any timed decode, so miss-fill reads hit the page cache instead of disk.
    {
        double rss_before = vmrss_mib();
        double tw0 = omp_get_wtime();
        tuff_gguf_warmup_memory(vtable->model_get_gguf(model));
        double tw1 = omp_get_wtime();
        double rss_after = vmrss_mib();
        fprintf(stderr, "[Tuff] Page-warmup: %.2f s (one-time, pre-decode); VmRSS %.0f MiB -> %.0f MiB\n",
                tw1 - tw0, rss_before, rss_after);
    }

    if (inspect_mode) {
        vtable->model_print_inspection(model);
        vtable->model_free(model);
        return 0;
    }

    // M2: upload dense weights to VRAM once, before any decode.
    if (vtable->gpu_prepare(model, &max_context_len, fill_ram ? 1 : 0) != 0) {
        fprintf(stderr, "[Tuff] ERROR: GPU preparation failed for %s (model not supported by the %s engine)\n", model_path, vtable->name);
        vtable->model_free(model);
        return 1;
    }

    if (arch == TUFF_ARCH_QWEN35MOE) {
        model_t *qm = (model_t*)model;
        // M4 sub-step A self-test for the Q4_K expert kernel (TUFF_EXPERT_SELFTEST=1).
        if (getenv("TUFF_EXPERT_SELFTEST")) gpu_selftest_q4k(qm);

        // H7 stage 1: is the second token free in a dense projection?
        if (bench_dense_batch) {
            for (uint32_t l = 0; l < qm->block_count; ++l) {
                model_layer_t *L = &qm->layers[l];
                if (L->type == LAYER_TYPE_ATTENTION || !L->attn_qkv.dev) continue;
                gpu_bench_dense_batch(&L->attn_qkv, (int)qm->embedding_length,
                                      (int)qm->ssm_qkv_dim, 200);
                break;
            }
            gpu_bench_head_batch(&qm->output_weight, (int)qm->embedding_length,
                                 (int)qm->vocab_size, 50);
            vtable->model_free(model);
            return 0;
        }
    }

    tokenizer_t *tok = tokenizer_init(vtable->model_get_gguf(model));

    if (chat_mode) {
        if (arch != TUFF_ARCH_QWEN35MOE) {
            fprintf(stderr, "[Tuff] Interactive chat is currently only supported for Qwen models.\n");
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 1;
        }
        chat_config_t cfg;
        cfg.model_path = model_path;
        cfg.system_prompt = system_prompt;
        cfg.log_filepath = log_filepath;
        cfg.max_context_len = max_context_len;
        cfg.max_gen_tokens = max_gen_tokens;
        cfg.spec_mtp = spec_mtp_mode;
        cfg.n_gpu_expert_layers = g_n_gpu_expert_layers;
        cfg.sampler = sampler_params;

        int rc = run_interactive_chat((model_t*)model, tok, &cfg);
        tokenizer_free(tok);
        vtable->gpu_shutdown(model);
        vtable->model_free(model);
        return rc;
    }

    int32_t *prompt_tokens = (int32_t*)malloc((size_t)max_context_len * sizeof(int32_t));
    if (!prompt_tokens) {
        fprintf(stderr, "[Tuff] FATAL: failed to allocate prompt_tokens for context length %d\n", max_context_len);
        tokenizer_free(tok);
        vtable->gpu_shutdown(model);
        vtable->model_free(model);
        return 1;
    }
    int n_prompt = tokenizer_tokenize(tok, prompt, prompt_tokens, max_context_len);
    if (n_prompt == 0) {
        prompt_tokens[0] = tok->bos_id;
        n_prompt = 1;
    }

    fprintf(stderr, "[Tuff] Prompt tokens (%d): ", n_prompt);
    for (int i = 0; i < n_prompt; ++i) fprintf(stderr, "%d ", prompt_tokens[i]);
    fprintf(stderr, "\n");

    // LFM engines index a fixed KV window: refuse up front a run that would feed positions beyond -c (the engine also
    // rejects out-of-range positions, but failing here is immediate and covers every mode).
    if (is_lfm && !teacher_force_path) {
        long need;
        const char *what;
        if (dump_logits_path) { need = (long)n_prompt + n_predict; what = "--dump-logits (prompt + -n)"; }
        else if (dump_tokens_mode) { need = (long)n_prompt + n_predict; what = "--dump-tokens (prompt + -n)"; }
        else { need = (long)n_prompt + n_warmup + n_predict; what = "generation/bench (prompt + --warmup + -n)"; }
        if (!ctx_fits(what, need, max_context_len)) {
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }
    }

    if (teacher_force_path) {
        FILE *tf = fopen(teacher_force_path, "r");
        if (!tf) {
            fprintf(stderr, "[Tuff] FATAL: cannot open teacher-force file: %s\n", teacher_force_path);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }
        // At most max_context_len reference tokens are ever used; heap-allocated (the default window is 131072).
        int32_t *ref_tokens = (int32_t*)malloc((size_t)max_context_len * sizeof(int32_t));
        const uint32_t tf_vocab_chk = vtable->model_get_vocab_size(model);
        int n_ref = 0, tf_bad = 0, tf_lineno = 0;
        char line[256];
        while (ref_tokens && fgets(line, sizeof(line), tf) && n_ref < max_context_len) {
            int t = 0;
            tf_lineno++;
            if (sscanf(line, "%d", &t) == 1) {
                if (t < 0 || (uint32_t)t >= tf_vocab_chk) {
                    fprintf(stderr, "[Tuff] FATAL: teacher-force file %s line %d: token id %d is outside the vocabulary [0, %u)\n",
                            teacher_force_path, tf_lineno, t, tf_vocab_chk);
                    tf_bad = 1;
                    break;
                }
                ref_tokens[n_ref++] = (int32_t)t;
            }
        }
        fclose(tf);
        if (!ref_tokens) {
            fprintf(stderr, "[Tuff] FATAL: out of memory for the teacher-force reference\n");
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }
        if (tf_bad) {
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }
        if (n_ref == 0) {
            fprintf(stderr, "[Tuff] FATAL: teacher-force file contained 0 tokens\n");
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }

        int target_steps = (n_predict > 0 && n_predict < n_ref) ? n_predict : n_ref;
        fprintf(stderr, "[Tuff] --teacher-force: testing %d steps against %s\n",
                target_steps, teacher_force_path);
        if (is_lfm && !ctx_fits("--teacher-force (prompt + reference steps)", (long)n_prompt + target_steps, max_context_len)) {
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }

        // With --dump-logits, the teacher-forced pass also writes the logit vector of every
        // step (same CLBRLGT1 format as the free-running dump). Because every step is fed the
        // reference token, vector i lines up 1:1 with reference token i, so the dump can be
        // compared numerically against llama.cpp / an FP64 reference past the first divergence.
        FILE *tf_dump = NULL;
        uint32_t tf_vocab = vtable->model_get_vocab_size(model);
        uint32_t tf_written = 0;
        if (dump_logits_path) {
            tf_dump = fopen(dump_logits_path, "wb");
            uint32_t zero = 0;
            if (!tf_dump || fwrite("CLBRLGT1", 1, 8, tf_dump) != 8 ||
                fwrite(&zero, sizeof(uint32_t), 1, tf_dump) != 1 ||
                fwrite(&tf_vocab, sizeof(uint32_t), 1, tf_dump) != 1) {
                fprintf(stderr, "[Tuff] FATAL: cannot write %s\n", dump_logits_path);
                if (tf_dump) fclose(tf_dump);
                tokenizer_free(tok);
                vtable->gpu_shutdown(model);
                vtable->model_free(model);
                return 3;
            }
        }

        void *eng = vtable->engine_init(model, max_context_len);
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed\n");
            if (tf_dump) { fclose(tf_dump); remove(dump_logits_path); }
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }
        vtable->engine_set_sampler(eng, &sampler_params);

        if (tf_rollback && tf_batch < 2) {
            fprintf(stderr, "[Tuff] FATAL: --tf-rollback requires --tf-batch >= 2\n");
            vtable->engine_free(eng);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }
        if (tf_batch > 1) {
            if (!vtable->engine_forward_batch || (tf_rollback && !vtable->engine_rollback)) {
                fprintf(stderr, "[Tuff] FATAL: --tf-batch/--tf-rollback are not supported by the %s engine\n", vtable->name);
                vtable->engine_free(eng);
                if (tf_dump) { fclose(tf_dump); remove(dump_logits_path); }
                tokenizer_free(tok);
                vtable->gpu_shutdown(model);
                vtable->model_free(model);
                return 2;
            }
            // Chunked teacher forcing. The token stream is prompt ++ reference[0..target_steps); the logits row computed
            // for stream position q selected reference step i = q - (n_prompt - 1), exactly as in the single-token loop.
            // Each chunk is run through forward_batch; with --tf-rollback the engine is then rolled back to
            // n_keep = 1 + (chunk_index % B) (clamped to the chunk length) and the next chunk starts at p + n_keep, so the
            // tail positions are computed again. Rows are consumed only for the n_keep accepted positions, i.e. every
            // dumped row is the one from the LAST time its position was computed, in stream order.
            int total = n_prompt + target_steps;
            int32_t *seq = (int32_t*)malloc((size_t)total * sizeof(int32_t));
            float *rows = (float*)malloc((size_t)tf_batch * tf_vocab * sizeof(float));
            if (!seq || !rows) { fprintf(stderr, "[Tuff] FATAL: out of memory\n"); return 2; }
            for (int q = 0; q < n_prompt; ++q) seq[q] = prompt_tokens[q];
            for (int q = 0; q < target_steps; ++q) seq[n_prompt + q] = ref_tokens[q];

            int matches = 0, first_divergence = -1, dump_ok = 1, n_batches = 0, n_rollbacks = 0;
            int p = 0, chunk = 0;
            while (p < total) {
                int n = total - p < tf_batch ? total - p : tf_batch;
                int32_t am[TF_MAX_BATCH];
                if (vtable->engine_forward_batch(eng, seq + p, n, p, am, rows) != 0) {
                    fprintf(stderr, "[Tuff] FATAL: engine_forward_batch failed at position %d (n=%d)\n", p, n);
                    return 3;
                }
                n_batches++;
                int keep = n;
                if (tf_rollback) {
                    keep = 1 + (chunk % tf_batch);
                    if (keep > n) keep = n;
                    if (vtable->engine_rollback(eng, keep) != 0) {
                        fprintf(stderr, "[Tuff] FATAL: engine_rollback(%d) failed at position %d\n", keep, p);
                        return 3;
                    }
                    if (keep < n) n_rollbacks++;
                }
                for (int j = 0; j < keep; ++j) {
                    int i = p + j - (n_prompt - 1);
                    if (i < 0 || i >= target_steps) continue;
                    const float *lg = rows + (size_t)j * tf_vocab;
                    int32_t expected = ref_tokens[i];
                    int32_t pred_token = am[j];
                    if (tf_dump && dump_ok) {
                        if (fwrite(lg, sizeof(float), tf_vocab, tf_dump) != tf_vocab) dump_ok = 0;
                        else tf_written++;
                    }
                    if (pred_token == expected) {
                        matches++;
                    } else {
                        if (first_divergence < 0) first_divergence = i;
                        char exp_piece[128];
                        snprintf(exp_piece, sizeof exp_piece, "%s", tokenizer_decode(tok, expected));
                        fprintf(stderr, "[TEACHER-FORCING] Mismatch at step %d: expected %d (%s, logit=%.4f), got %d (%s, logit=%.4f), diff=%.6f\n",
                                i, expected, exp_piece, lg[expected], pred_token, tokenizer_decode(tok, pred_token), lg[pred_token],
                                lg[pred_token] - lg[expected]);
                    }
                }
                p += keep;
                chunk++;
            }
            free(seq);
            free(rows);
            fprintf(stderr, "[Tuff] --tf-batch %d%s: %d batches, %d rollbacks\n", tf_batch, tf_rollback ? " --tf-rollback" : "",
                    n_batches, n_rollbacks);

            if (tf_dump) {
                if (dump_ok && (fseek(tf_dump, 8, SEEK_SET) != 0 ||
                                fwrite(&tf_written, sizeof(uint32_t), 1, tf_dump) != 1)) dump_ok = 0;
                if (fclose(tf_dump) != 0) dump_ok = 0;
                if (!dump_ok) {
                    fprintf(stderr, "[Tuff] FATAL: failed writing teacher-forced logits to %s; removing it\n", dump_logits_path);
                    remove(dump_logits_path);
                    vtable->engine_free(eng);
                    tokenizer_free(tok);
                    vtable->gpu_shutdown(model);
                    vtable->model_free(model);
                    return 3;
                }
                fprintf(stderr, "[Tuff] wrote %u x %u teacher-forced logits to %s\n", tf_written, tf_vocab, dump_logits_path);
            }
            fprintf(stderr, "\n========================================================================\n");
            fprintf(stderr, " Teacher-Forced Verification Result: %d / %d (%.2f%%) top-1 agreement\n",
                    matches, target_steps, 100.0 * (double)matches / (double)target_steps);
            if (matches == target_steps) {
                fprintf(stderr, " [TEACHER-FORCING] PASS: 100%% per-step agreement over %d tokens\n", target_steps);
            } else {
                fprintf(stderr, " [TEACHER-FORCING] FAIL: first mismatch at step %d\n", first_divergence);
            }
            fprintf(stderr, "========================================================================\n");
            vtable->engine_free(eng);
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return (matches == target_steps) ? 0 : 1;
        }

        // Armed before the prompt so the vector that selected step 0 is captured too.
        tuff_logit_capture_enable(1);
        int pos = 0;
        int32_t pred_token = prompt_tokens[0];
        for (int i = 0; i < n_prompt; ++i) {
            pred_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
            fwd_check(is_lfm, pred_token, pos - 1);
        }

        int matches = 0;
        int first_divergence = -1;
        int dump_ok = 1;
        for (int i = 0; i < target_steps; ++i) {
            int32_t expected = ref_tokens[i];
            uint32_t vn = 0;
            const float *lg = tuff_logit_capture_take(&vn);
            if (tf_dump && dump_ok) {
                if (!lg || vn != tf_vocab || fwrite(lg, sizeof(float), vn, tf_dump) != vn) dump_ok = 0;
                else tf_written++;
            }
            int match = (pred_token == expected);
            if (match) {
                matches++;
            } else {
                if (first_divergence < 0) first_divergence = i;
                float exp_logit = (lg && expected < (int32_t)vn) ? lg[expected] : 0.0f;
                float got_logit = (lg && pred_token < (int32_t)vn) ? lg[pred_token] : 0.0f;
                // tokenizer_decode returns a shared buffer: copy the first piece before the second call.
                char exp_piece[128];
                snprintf(exp_piece, sizeof exp_piece, "%s", tokenizer_decode(tok, expected));
                fprintf(stderr, "[TEACHER-FORCING] Mismatch at step %d: expected %d (%s, logit=%.4f), got %d (%s, logit=%.4f), diff=%.6f\n",
                        i, expected, exp_piece, exp_logit,
                        pred_token, tokenizer_decode(tok, pred_token), got_logit,
                        got_logit - exp_logit);
            }
            // Teacher-force: next step receives expected reference token instead of pred_token
            pred_token = vtable->engine_forward_token(eng, expected, pos++);
            fwd_check(is_lfm, pred_token, pos - 1);
        }
        tuff_logit_capture_enable(0);

        if (tf_dump) {
            if (dump_ok && (fseek(tf_dump, 8, SEEK_SET) != 0 ||
                            fwrite(&tf_written, sizeof(uint32_t), 1, tf_dump) != 1)) dump_ok = 0;
            if (fclose(tf_dump) != 0) dump_ok = 0;
            if (!dump_ok) {
                fprintf(stderr, "[Tuff] FATAL: failed writing teacher-forced logits to %s; removing it\n",
                        dump_logits_path);
                remove(dump_logits_path);
                vtable->engine_free(eng);
                tokenizer_free(tok);
                vtable->gpu_shutdown(model);
                vtable->model_free(model);
                return 3;
            }
            fprintf(stderr, "[Tuff] wrote %u x %u teacher-forced logits to %s\n",
                    tf_written, tf_vocab, dump_logits_path);
        }

        fprintf(stderr, "\n========================================================================\n");
        fprintf(stderr, " Teacher-Forced Verification Result: %d / %d (%.2f%%) top-1 agreement\n",
                matches, target_steps, 100.0 * (double)matches / (double)target_steps);
        if (matches == target_steps) {
            fprintf(stderr, " [TEACHER-FORCING] PASS: 100%% per-step agreement over %d tokens\n", target_steps);
        } else {
            fprintf(stderr, " [TEACHER-FORCING] FAIL: first mismatch at step %d\n", first_divergence);
        }
        fprintf(stderr, "========================================================================\n");

        vtable->engine_free(eng);
        free(ref_tokens);
        free(prompt_tokens);
        tokenizer_free(tok);
        vtable->gpu_shutdown(model);
        vtable->model_free(model);
        return (matches == target_steps) ? 0 : 1;
    }

    if (dump_logits_path) {
        // Numeric-gate mode (parity.md P0). Same deterministic greedy pass as --dump-tokens
        // (no warmup, no timing protocol, diagnostics on stderr), but each generated token's
        // final logit vector is written to a binary file so two runs -- e.g. TUFF_EXACT=1
        // versus the fast path -- can be compared numerically by scripts/verify_tolerance.py.
        fprintf(stderr, "[Tuff] --dump-logits: generating %d tokens -> %s\n",
                n_predict, dump_logits_path);

        tuff_logit_capture_enable(1);

        void *eng = vtable->engine_init(model, max_context_len);
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed\n");
            tuff_logit_capture_enable(0);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];
        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
        }

        // The vector left behind by the last prompt forward is the one that selected the
        // first generated token, so the i-th dumped vector lines up 1:1 with the i-th token
        // ID printed by --dump-tokens.
        uint32_t vocab = 0;
        const float *lg = tuff_logit_capture_take(&vocab);
        if (!lg || vocab == 0) {
            fprintf(stderr,
                "[Tuff] FATAL: --dump-logits captured no logits.\n"
                "             The engine-side hook is missing. src/engine.c must call\n"
                "                 tuff_logit_capture(logits, vocab_size);\n"
                "             immediately after `mat_vec(logits, &m->output_weight, norm_x);`\n"
                "             (and #include \"tuff_exact.h\"). Refusing to write a file:\n"
                "             an empty or zero-filled dump would compare as a silent PASS.\n");
            vtable->engine_free(eng);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }
        if (vocab != vtable->model_get_vocab_size(model)) {
            fprintf(stderr, "[Tuff] FATAL: captured logit width %u != model vocab_size %u\n",
                    vocab, vtable->model_get_vocab_size(model));
            vtable->engine_free(eng);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }

        FILE *fp = fopen(dump_logits_path, "wb");
        if (!fp) {
            fprintf(stderr, "[Tuff] FATAL: cannot open %s for writing: ", dump_logits_path);
            perror(NULL);
            vtable->engine_free(eng);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }

        // Header: magic (8 B, no NUL) + uint32 n_tokens + uint32 vocab_size. n_tokens is
        // rewritten at close with the count actually produced.
        uint32_t n_tokens_hdr = 0;
        int hdr_ok = (fwrite("CLBRLGT1", 1, 8, fp) == 8) &&
                     (fwrite(&n_tokens_hdr, sizeof(uint32_t), 1, fp) == 1) &&
                     (fwrite(&vocab, sizeof(uint32_t), 1, fp) == 1);

        uint32_t n_written = 0;
        int io_ok = hdr_ok;
        for (int i = 0; io_ok && i < n_predict; ++i) {
            if (fwrite(lg, sizeof(float), vocab, fp) != vocab) { io_ok = 0; break; }
            n_written++;
            fprintf(stderr, "%s%d", i ? " " : "[Tuff] token IDs: ", curr_token);

            curr_token = vtable->engine_forward_token(eng, curr_token, pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
            uint32_t vn = 0;
            lg = tuff_logit_capture_take(&vn);
            if (i + 1 < n_predict && (!lg || vn != vocab)) {
                fprintf(stderr, "\n[Tuff] FATAL: logit capture dropped out at token %d "
                                "(got width %u, expected %u)\n", i + 1, vn, vocab);
                io_ok = 0;
            }
        }
        fprintf(stderr, "\n");

        if (io_ok && n_written == (uint32_t)n_predict) {
            if (fseek(fp, 8, SEEK_SET) != 0 ||
                fwrite(&n_written, sizeof(uint32_t), 1, fp) != 1) {
                io_ok = 0;
            }
        } else {
            io_ok = 0;
        }
        if (fclose(fp) != 0) io_ok = 0;

        tuff_logit_capture_enable(0);
        vtable->engine_free(eng);
        tokenizer_free(tok);
        vtable->gpu_shutdown(model);
        vtable->model_free(model);

        if (!io_ok) {
            fprintf(stderr, "[Tuff] FATAL: failed writing %s (%u/%d vectors); removing it "
                            "so nothing downstream reads a truncated dump.\n",
                    dump_logits_path, n_written, n_predict);
            remove(dump_logits_path);
            return 3;
        }
        fprintf(stderr, "[Tuff] wrote %u x %u float32 logits (%zu bytes payload) to %s\n",
                n_written, vocab, (size_t)n_written * vocab * sizeof(float), dump_logits_path);
        return 0;
    }

    if (dump_tokens_mode) {
        // Oracle mode: a single deterministic greedy pass. One token ID per line on stdout;
        // everything else on stderr, so scripts/verify_vs_llama.sh can diff pure token streams.
        fprintf(stderr, "[Tuff] --dump-tokens: generating %d tokens (%s reference oracle)...\n",
                n_predict, vtable->name);

        if (lfm_spec && !spec_kfix) spec_measure_costs(vtable, model, max_context_len, &sampler_params, prompt_tokens, n_prompt, spec_k_max,
                                                   arch == TUFF_ARCH_LFM2 || getenv("LFM25_SPEC_MEASURE") != NULL);
        void *eng = vtable->engine_init(model, max_context_len);
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed\n");
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];
        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
        }
        if (lfm_spec) {
            spec_t sp;
            spec_init(&sp, vtable, max_context_len, spec_nmax, spec_nmin, spec_k_max, spec_kfix);
            spec_begin_run(&sp, eng);
            spec_print_policy(&sp);
            for (int i = 0; i < n_prompt; ++i) spec_push(&sp, prompt_tokens[i]);
            spec_push(&sp, curr_token);
            printf("%d\n", curr_token);
            fflush(stdout);
            int printed = 1;
            while (printed < n_predict) {
                int32_t out[SPEC_KMAX + 1];
                int n = spec_step(&sp, curr_token, pos, n_predict - printed, out);
                pos += n;
                for (int q = 0; q < n; ++q) printf("%d\n", out[q]);
                fflush(stdout);
                printed += n;
                curr_token = out[n - 1];
            }
            spec_print_stats(&sp, (unsigned long long)(printed - 1));
            free(sp.hist);
        } else {
        for (int i = 0; i < n_predict; ++i) {
            printf("%d\n", curr_token);
            fflush(stdout);
            curr_token = vtable->engine_forward_token(eng, curr_token, pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
        }
        }
        vtable->engine_free(eng);
        tokenizer_free(tok);
        vtable->gpu_shutdown(model);
        vtable->model_free(model);
        return 0;
    }

    // Speed-measurement mode: PLAN.md sec.4 protocol — 32 warmup tokens discarded, then a
    // timed block, >=3 runs, report mean +/- spread. All diagnostics on stderr; only the
    // generated text streams to stdout, consistent with keeping stdout clean for the oracle.
    if (n_runs < 3) {
        fprintf(stderr, "[Tuff] WARNING: --runs %d < 3; PLAN.md sec.4 requires >=3 runs "
                        "for a reported number to be honest.\n", n_runs);
    }

    double *tok_per_sec = (double*)malloc((size_t)n_runs * sizeof(double));

    drafter_t *drafter = NULL;
    if (spec_draft_model) {
        drafter = drafter_init(spec_draft_model, 8, 0);
    }

    spec_t lfm_sp;
    if (lfm_spec) {
        if (!spec_kfix) spec_measure_costs(vtable, model, max_context_len, &sampler_params, prompt_tokens, n_prompt, spec_k_max,
                                                   arch == TUFF_ARCH_LFM2 || getenv("LFM25_SPEC_MEASURE") != NULL);
        spec_init(&lfm_sp, vtable, max_context_len, spec_nmax, spec_nmin, spec_k_max, spec_kfix);
        spec_print_policy(&lfm_sp);
    }

    for (int run = 0; run < n_runs; ++run) {
        void *eng = vtable->engine_init(model, max_context_len);
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed (run %d)\n", run + 1);
            if (lfm_spec) free(lfm_sp.hist);
            if (drafter) drafter_free(drafter);
            free(tok_per_sec);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 3;
        }
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];

        int32_t *warmup_seq = (int32_t*)malloc((size_t)(n_prompt + n_warmup + 1) * sizeof(int32_t));
        memcpy(warmup_seq, prompt_tokens, (size_t)n_prompt * sizeof(int32_t));

        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
        }
        if (lfm_spec) {
            spec_begin_run(&lfm_sp, eng);
            for (int i = 0; i < n_prompt; ++i) spec_push(&lfm_sp, prompt_tokens[i]);
            spec_push(&lfm_sp, curr_token);
        }

        for (int i = 0; i < n_warmup; ++i) {
            warmup_seq[n_prompt + i] = curr_token;
            curr_token = vtable->engine_forward_token(eng, curr_token, pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
            if (lfm_spec) spec_push(&lfm_sp, curr_token);
        }

        if (drafter && arch == TUFF_ARCH_QWEN35MOE) {
            drafter_prefill(drafter, warmup_seq, n_prompt + n_warmup);
            engine_set_drafter((engine_t*)eng, drafter);
        }
        free(warmup_seq);

        if (run == 0) fprintf(stderr, "[Tuff] Generated text (run 0, includes warmup+timed): ");
        if (getenv("TUFF_TRACE") && run == 0) {
            cupti_profiler_start();
        }
        double t0 = omp_get_wtime();
        int n_generated = 0;
        int n_drafts = 0, n_accepted = 0;
        // --spec-mtp now DECODES speculatively (H7). Each step verifies the draft in a
        // two-column trunk pass and emits one or two verified tokens; --spec-measure keeps
        // the old behaviour, which only compares the draft with the token the model chose
        // anyway and throws it away. Both emit only tokens the real trunk produced, so the
        // tok/s from either is comparable with the non-spec number.
        spec_stats_t sstat;
        memset(&sstat, 0, sizeof(sstat));
        int32_t pending_draft = -1;
        while (n_generated < n_predict) {
            if (run == 0) {
                fputs(tokenizer_decode(tok, curr_token), stdout);
                fflush(stdout);
            }

            if (lfm_spec) {
                int32_t out[SPEC_KMAX + 1];
                int n = spec_step(&lfm_sp, curr_token, pos, n_predict - n_generated, out);
                if (run == 0) {
                    for (int q = 0; q < n - 1; ++q) {
                        fputs(tokenizer_decode(tok, out[q]), stdout);
                        fflush(stdout);
                    }
                }
                curr_token = out[n - 1];
                pos += n;
                n_generated += n;
                continue;
            }

            if (spec_mtp_mode && !spec_measure_mode && arch == TUFF_ARCH_QWEN35MOE) {
                int32_t emitted[4], next_tok = -1;
                int n = engine_spec_step((engine_t*)eng, curr_token, pos, emitted, &next_tok, &sstat);
                if (run == 0) {
                    for (int k = 0; k < n - 1; ++k) {
                        fputs(tokenizer_decode(tok, emitted[k]), stdout);
                        fflush(stdout);
                    }
                }
                curr_token = emitted[n - 1];
                pos += n;
                n_generated += n;
                continue;
            }

            int32_t next_tok = vtable->engine_forward_token(eng, curr_token, pos++);
            fwd_check(is_lfm, next_tok, pos - 1);
            n_generated++;

            if (spec_measure_mode && arch == TUFF_ARCH_QWEN35MOE) {
                if (pending_draft >= 0) {
                    n_drafts++;
                    if (pending_draft == next_tok) n_accepted++;
                }
                pending_draft = engine_get_last_draft_token((const engine_t*)eng);
            }
            curr_token = next_tok;
        }
        double t1 = omp_get_wtime();
        if (run == 0) fprintf(stderr, "\n");
        if (getenv("TUFF_TRACE") && run == 0) {
            cupti_profiler_stop_and_print();
        }

        double elapsed = t1 - t0;
        tok_per_sec[run] = (elapsed > 0.0) ? (double)n_generated / elapsed : 0.0;

        if (spec_mtp_mode && !spec_measure_mode && sstat.drafted > 0) {
            const char *src = (g_spec_draft == SPEC_DRAFT_NGRAM) ? "ngram" :
                              (g_spec_draft == SPEC_DRAFT_NGRAM_MTP) ? "ngram+mtp" :
                              (g_spec_draft == SPEC_DRAFT_EXTERNAL) ? "external" : "mtp";
            fprintf(stderr, "[Tuff] run %d/%d: %d warmup + %llu timed tokens in %.3f s -> %.2f tok/s "
                            "[%s speculative: alpha=%.1f%% (%llu/%llu), %.3f tokens/step over %llu steps]\n",
                    run + 1, n_runs, n_warmup, sstat.tokens_emitted, elapsed, tok_per_sec[run],
                    src,
                    100.0 * (double)sstat.accepted / (double)sstat.drafted,
                    sstat.accepted, sstat.drafted,
                    (double)sstat.tokens_emitted / (double)sstat.steps, sstat.steps);
        } else if (spec_measure_mode && n_drafts > 0) {
            const char *src = (g_spec_draft == SPEC_DRAFT_NGRAM) ? "ngram" :
                              (g_spec_draft == SPEC_DRAFT_NGRAM_MTP) ? "ngram+mtp" :
                              (g_spec_draft == SPEC_DRAFT_EXTERNAL) ? "external" : "mtp";
            fprintf(stderr, "[Tuff] run %d/%d: %d warmup + %d timed tokens in %.3f s -> %.2f tok/s [%s draft acceptance (measured, not used): %.1f%% (%d/%d)]\n",
                    run + 1, n_runs, n_warmup, n_predict, elapsed, tok_per_sec[run],
                    src,
                    100.0 * (double)n_accepted / (double)n_drafts, n_accepted, n_drafts);
        } else {
            fprintf(stderr, "[Tuff] run %d/%d: %d warmup + %d timed tokens in %.3f s -> %.2f tok/s\n",
                    run + 1, n_runs, n_warmup, n_predict, elapsed, tok_per_sec[run]);
        }

        if (lfm_spec) spec_print_stats(&lfm_sp, (unsigned long long)n_generated);
        vtable->engine_free(eng);
    }
    if (lfm_spec) free(lfm_sp.hist);
    if (drafter) drafter_free(drafter);

    double mean = 0.0;
    for (int r = 0; r < n_runs; ++r) mean += tok_per_sec[r];
    mean /= n_runs;

    double min_v = tok_per_sec[0], max_v = tok_per_sec[0];
    for (int r = 1; r < n_runs; ++r) {
        if (tok_per_sec[r] < min_v) min_v = tok_per_sec[r];
        if (tok_per_sec[r] > max_v) max_v = tok_per_sec[r];
    }
    double spread = (max_v - min_v) / 2.0;

    fprintf(stderr, "\n========================================================================\n");
    fprintf(stderr, " Pipeline Generation Speed: %.2f +/- %.2f tok/s (mean of %d runs, %d warmup + %d timed)\n",
            mean, spread, n_runs, n_warmup, n_predict);
    fprintf(stderr, " RSS: %.1f MiB\n", rss_mib());
    fprintf(stderr, " VRAM: %.1f MiB used\n", gpu_vram_used_mb());
    if (arch == TUFF_ARCH_QWEN35MOE) {
        unsigned long long hits = 0, misses = 0;
        gpu_expert_cache_stats(&hits, &misses);
        double hr = (hits + misses) > 0 ? 100.0 * (double)hits / (double)(hits + misses) : 0.0;
        fprintf(stderr, " Expert-cache hit rate: %.1f%% (%llu hits / %llu misses)\n", hr, hits, misses);
    }
    fprintf(stderr, "========================================================================\n");

    free(tok_per_sec);
    tokenizer_free(tok);
    vtable->gpu_shutdown(model);
    vtable->model_free(model);
    return 0;
}
