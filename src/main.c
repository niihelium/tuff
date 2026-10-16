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
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <limits.h>
#include "lfm25_model.h"

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
        "Job runner (LFM2.5 engines): load the model once, run many jobs\n"
        "  --jobs <file>               one job per line: [NAME=value ...] <tuff arguments> [--out <f>] [--err <f>] [--status <f>];\n"
        "                              other arguments on this command line are defaults for every job; reloads the model only when\n"
        "                              the model path or LFM25_FORCE_EXPERT_CACHE (on <=32-expert MoE) changes. Env: TUFF_TIMING=1\n"
        "                              prints per-phase wall times.\n"
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
// --jobs mode re-reads TUFF_EXACT for every job (the job may set it).
static void tuff_exact_reset(void) { g_tuff_exact = -1; }

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
// Job runner infrastructure (`tuff --jobs <file>`): one process loads the model once and runs many jobs.
// ---------------------------------------------------------------------------------------------------------------
static int         g_in_jobs = 0;        // 1 while inside the --jobs runner (exit() must not take the runner down silently)
static const char *g_cur_status = NULL;  // --status path of the running job

// exit() for fatal conditions inside a run: in jobs mode the job's exit code is recorded (last stderr line + status file)
// before the whole runner terminates (a fatal engine error leaves CUDA state we do not trust for the next job).
static void job_exit(int code) {
    if (g_in_jobs) {
        fflush(stdout);
        fprintf(stderr, "JOB_EXIT=%d\n", code);
        if (g_cur_status) { FILE *f = fopen(g_cur_status, "w"); if (f) { fprintf(f, "%d\n", code); fclose(f); } }
    }
    exit(code);
}

// TUFF_TIMING=1: one line per phase on stderr, "[Tuff timing] <phase>: <seconds since the previous mark> (t=<seconds since job start>)".
static int    g_tm_on = 0;
static double g_tm_last = 0.0, g_tm_start = 0.0;

static double tm_mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void tm_reset(void) {
    const char *e = getenv("TUFF_TIMING");
    g_tm_on = e && e[0] == '1';
    g_tm_start = g_tm_last = tm_mono();
}

static void tm_mark(const char *what) {
    if (!g_tm_on) return;
    double now = tm_mono();
    fprintf(stderr, "[Tuff timing] %s: %.3f s (t=%.3f)\n", what, now - g_tm_last, now - g_tm_start);
    g_tm_last = now;
}

// Seconds since this process was started (dynamic loading + static init + CUDA runtime load happen before main()).
static double proc_age_sec(void) {
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) return -1.0;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    const char *p = strrchr(buf, ')');
    if (!p) return -1.0;
    unsigned long long start = 0;
    int field = 3;   // fields after "(comm)" start at 3 (state)
    for (p += 2; *p && field < 22; ++p) if (*p == ' ') field++;
    if (sscanf(p, "%llu", &start) != 1) return -1.0;
    FILE *u = fopen("/proc/uptime", "r");
    if (!u) return -1.0;
    double up = 0.0;
    int ok = fscanf(u, "%lf", &up);
    fclose(u);
    if (ok != 1) return -1.0;
    return up - (double)start / (double)sysconf(_SC_CLK_TCK);
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
            job_exit(3);
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
        job_exit(1);
    }
    int m = 0;
    while (m < k && d[m] == am[m]) m++;
    for (int i = 0; i <= m; ++i) { out[i] = am[i]; spec_push(s, am[i]); }
    if (m < k && vt->engine_rollback(s->eng, m + 1) != 0) {
        fprintf(stderr, "[Tuff] FATAL: speculative engine_rollback(%d) failed at position %d\n", m + 1, pos);
        job_exit(1);
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
        job_exit(3);
    }
}


// ---------------------------------------------------------------------------------------------------------------
// Job configuration: everything the command line (or a job line) can say.
// ---------------------------------------------------------------------------------------------------------------
typedef struct job_cfg {
    const char *model_path, *prompt, *system_prompt, *log_filepath, *spec_draft_model, *dump_logits_path, *teacher_force_path;
    bool prompt_specified, chat_mode, fill_ram, inspect_mode, bench_dense_batch, dump_tokens_mode;
    bool spec_mtp_mode, spec_measure_mode, spec_ngram_requested, tf_rollback;
    int max_context_len, max_gen_tokens, spec_k_max, n_predict, n_warmup, n_runs, n_gpu_expert_layers, tf_batch;
    sampler_params_t sampler_params;
    float default_temp;
    int kv_quant;
    // process-wide knobs of the Qwen/draft machinery, applied by cfg_apply_globals()
    int g_spec_mtp, g_spec_draft, g_spec_draft_len, g_spec_ngram_n, g_spec_ngram_min;
    // job-only options (--jobs mode)
    const char *jobs_path, *out_path, *err_path, *status_path;
    // derived by job_prepare()
    tuff_arch_t arch;
    const engine_vtable_t *vtable;
    int is_lfm;
    bool lfm_spec;
    int spec_nmax, spec_nmin, spec_kfix;
} job_cfg_t;

// Defaults of the process-wide knobs, captured once before any job touches them.
static struct { int mtp, draft, draft_len, ngram_n, ngram_min; } g_glob_def;

static void cfg_init(job_cfg_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->model_path = "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf";
    cfg->prompt = "Hello";
    cfg->system_prompt = "You are a helpful, respectful and honest AI assistant.";
    cfg->max_context_len = 131072;
    cfg->max_gen_tokens = 1024;
    cfg->spec_mtp_mode = true;
    cfg->spec_k_max = 4;
    cfg->n_predict = -1;
    cfg->n_warmup = 32;
    cfg->n_runs = 3;
    cfg->tf_batch = 1;
    cfg->sampler_params = sampler_params_default();
    cfg->default_temp = cfg->sampler_params.temp;
    cfg->kv_quant = KV_QUANT_Q8_0;
    cfg->g_spec_mtp = g_glob_def.mtp; cfg->g_spec_draft = g_glob_def.draft; cfg->g_spec_draft_len = g_glob_def.draft_len;
    cfg->g_spec_ngram_n = g_glob_def.ngram_n; cfg->g_spec_ngram_min = g_glob_def.ngram_min;
    const char *kvq_env = getenv("TUFF_KV_QUANT");
    if (kvq_env && kvq_env[0] != '\0') {
        if (strcmp(kvq_env, "f32") == 0 || strcmp(kvq_env, "fp32") == 0) cfg->kv_quant = KV_QUANT_F32;
        else if (strcmp(kvq_env, "q8_0") == 0 || strcmp(kvq_env, "q8") == 0) cfg->kv_quant = KV_QUANT_Q8_0;
        else if (strcmp(kvq_env, "q4_0") == 0 || strcmp(kvq_env, "q4") == 0) cfg->kv_quant = KV_QUANT_Q4_0;
    }
}

static void cfg_apply_globals(const job_cfg_t *cfg) {
    gpu_set_kv_quant(cfg->kv_quant);
    g_spec_mtp = cfg->g_spec_mtp; g_spec_draft = cfg->g_spec_draft; g_spec_draft_len = cfg->g_spec_draft_len;
    g_spec_ngram_n = cfg->g_spec_ngram_n; g_spec_ngram_min = cfg->g_spec_ngram_min;
}

// Returns 1 to continue, 0 to stop with exit code *rc (usage errors, --help).
static int parse_args(int argc, char **argv, job_cfg_t *cfg, int *rc) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (strcmp(argv[i], "--chat") == 0 || strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            cfg->chat_mode = true;
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            cfg->system_prompt = argv[++i];
        } else if ((strcmp(argv[i], "--log") == 0 || strcmp(argv[i], "--log-file") == 0) && i + 1 < argc) {
            cfg->log_filepath = argv[++i];
        } else if ((strcmp(argv[i], "--max-seq") == 0 || strcmp(argv[i], "-c") == 0 ||
                    strcmp(argv[i], "--ctx-size") == 0 || strcmp(argv[i], "--ctx") == 0 ||
                    strcmp(argv[i], "--context-size") == 0) && i + 1 < argc) {
            cfg->max_context_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fill-ram") == 0 || strcmp(argv[i], "--fill-vram") == 0 ||
                   strcmp(argv[i], "--fill_ram") == 0 || strcmp(argv[i], "--fill_vram") == 0) {
            cfg->fill_ram = true;
        } else if (strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc) {
            cfg->max_gen_tokens = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "-p") == 0) && i + 1 < argc) {
            cfg->prompt = argv[++i];
            cfg->prompt_specified = true;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cfg->n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg->n_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            cfg->n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--spec-draft-model") == 0 && i + 1 < argc) {
            cfg->spec_draft_model = argv[++i];
            cfg->spec_mtp_mode = true;
            cfg->g_spec_mtp = 1;
            cfg->g_spec_draft = SPEC_DRAFT_EXTERNAL;
        } else if (strcmp(argv[i], "--spec-mtp") == 0) {
            cfg->spec_mtp_mode = true;
            cfg->g_spec_mtp = 1;
        } else if (strncmp(argv[i], "--spec-draft=", 13) == 0) {
            cfg->spec_mtp_mode = true;
            cfg->g_spec_mtp = 1;
            const char *s = argv[i] + 13;
            cfg->spec_ngram_requested = (strcmp(s, "ngram") == 0);
            if (strcmp(s, "ngram") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_NGRAM;
            } else if (strcmp(s, "ngram+mtp") == 0 || strcmp(s, "mtp+ngram") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_NGRAM_MTP;
            } else if (strcmp(s, "mtp") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_MTP;
            } else if (strcmp(s, "external") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_EXTERNAL;
            } else if (strcmp(s, "none") == 0 || strcmp(s, "off") == 0 || strcmp(s, "0") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_NONE;
                cfg->spec_mtp_mode = false;
                cfg->g_spec_mtp = 0;
            } else {
                fprintf(stderr, "Error: --spec-draft must be ngram, mtp, ngram+mtp, external or none (got '%s')\n", s);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--no-spec") == 0) {
            cfg->spec_ngram_requested = false;
            cfg->g_spec_draft = SPEC_DRAFT_NONE;
            cfg->spec_mtp_mode = false;
            cfg->g_spec_mtp = 0;
        } else if (strcmp(argv[i], "--spec-draft") == 0 && i + 1 < argc) {
            cfg->spec_mtp_mode = true;
            cfg->g_spec_mtp = 1;
            const char *s = argv[++i];
            cfg->spec_ngram_requested = (strcmp(s, "ngram") == 0);
            if (strcmp(s, "ngram") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_NGRAM;
            } else if (strcmp(s, "ngram+mtp") == 0 || strcmp(s, "mtp+ngram") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_NGRAM_MTP;
            } else if (strcmp(s, "mtp") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_MTP;
            } else if (strcmp(s, "external") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_EXTERNAL;
            } else if (strcmp(s, "none") == 0 || strcmp(s, "off") == 0 || strcmp(s, "0") == 0) {
                cfg->g_spec_draft = SPEC_DRAFT_NONE;
                cfg->spec_mtp_mode = false;
                cfg->g_spec_mtp = 0;
            } else {
                fprintf(stderr, "Error: --spec-draft must be ngram, mtp, ngram+mtp, external or none (got '%s')\n", s);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if ((strcmp(argv[i], "--spec-ngram") == 0 || strcmp(argv[i], "--spec-ngram-max") == 0) && i + 1 < argc) {
            cfg->g_spec_ngram_n = atoi(argv[++i]);
            if (cfg->g_spec_ngram_n < 1 || cfg->g_spec_ngram_n > 8) {
                fprintf(stderr, "Error: --spec-ngram must be between 1 and 8 (got %d)\n", cfg->g_spec_ngram_n);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--spec-ngram-min") == 0 && i + 1 < argc) {
            cfg->g_spec_ngram_min = atoi(argv[++i]);
            if (cfg->g_spec_ngram_min < 1 || cfg->g_spec_ngram_min > 8) {
                fprintf(stderr, "Error: --spec-ngram-min must be between 1 and 8 (got %d)\n", cfg->g_spec_ngram_min);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if ((strcmp(argv[i], "--spec-draft-len") == 0 || strcmp(argv[i], "--draft-len") == 0) && i + 1 < argc) {
            cfg->g_spec_draft_len = atoi(argv[++i]);
            if (cfg->g_spec_draft_len < 1 || cfg->g_spec_draft_len > 3) {
                fprintf(stderr, "Error: --spec-draft-len must be between 1 and 3 (got %d)\n", cfg->g_spec_draft_len);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--spec-k") == 0 && i + 1 < argc) {
            cfg->spec_k_max = atoi(argv[++i]);
            if (cfg->spec_k_max < 1 || cfg->spec_k_max > SPEC_KMAX) {
                fprintf(stderr, "Error: --spec-k must be between 1 and %d (got %d)\n", SPEC_KMAX, cfg->spec_k_max);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--spec-measure") == 0) {
            cfg->spec_mtp_mode = true;
            cfg->spec_measure_mode = true;
            cfg->g_spec_mtp = 1;
        } else if (strcmp(argv[i], "--n-gpu-expert-layers") == 0 && i + 1 < argc) {
            cfg->n_gpu_expert_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--inspect") == 0) {
            cfg->inspect_mode = true;
        } else if (strcmp(argv[i], "--bench-dense-batch") == 0) {
            cfg->bench_dense_batch = true;
        } else if (strcmp(argv[i], "--dump-tokens") == 0) {
            cfg->dump_tokens_mode = true;
        } else if (strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            cfg->dump_logits_path = argv[++i];
        } else if (strcmp(argv[i], "--dump-logits") == 0) {
            fprintf(stderr, "Error: --dump-logits requires a path argument\n");
            print_usage(argv[0]);
        } else if (strncmp(argv[i], "--kv-quant=", 11) == 0) {
            const char *s = argv[i] + 11;
            if (strcmp(s, "f32") == 0 || strcmp(s, "fp32") == 0) {
                cfg->kv_quant = KV_QUANT_F32;
            } else if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8") == 0) {
                cfg->kv_quant = KV_QUANT_Q8_0;
            } else if (strcmp(s, "q4_0") == 0 || strcmp(s, "q4") == 0) {
                cfg->kv_quant = KV_QUANT_Q4_0;
            } else {
                fprintf(stderr, "Error: --kv-quant must be f32, q8_0, or q4_0 (got '%s')\n", s);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--kv-quant") == 0 && i + 1 < argc) {
            const char *s = argv[++i];
            if (strcmp(s, "f32") == 0 || strcmp(s, "fp32") == 0) {
                cfg->kv_quant = KV_QUANT_F32;
            } else if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8") == 0) {
                cfg->kv_quant = KV_QUANT_Q8_0;
            } else if (strcmp(s, "q4_0") == 0 || strcmp(s, "q4") == 0) {
                cfg->kv_quant = KV_QUANT_Q4_0;
            } else {
                fprintf(stderr, "Error: --kv-quant must be f32, q8_0, or q4_0 (got '%s')\n", s);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--tf-batch") == 0 && i + 1 < argc) {
            cfg->tf_batch = atoi(argv[++i]);
            if (cfg->tf_batch < 1 || cfg->tf_batch > TF_MAX_BATCH) {
                fprintf(stderr, "Error: --tf-batch must be 1..%d\n", TF_MAX_BATCH);
                { *rc = 2; { *rc = 0; return 0; } }
            }
        } else if (strcmp(argv[i], "--tf-rollback") == 0) {
            cfg->tf_rollback = true;
        } else if (strcmp(argv[i], "--teacher-force") == 0 && i + 1 < argc) {
            cfg->teacher_force_path = argv[++i];
        } else if (strcmp(argv[i], "--teacher-force") == 0) {
            fprintf(stderr, "Error: --teacher-force requires a path argument\n");
            print_usage(argv[0]);
            { *rc = 2; { *rc = 0; return 0; } }
        } else if ((strcmp(argv[i], "--temp") == 0 || strcmp(argv[i], "--temperature") == 0) && i + 1 < argc) {
            cfg->sampler_params.temp = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--top-p") == 0 || strcmp(argv[i], "--top_p") == 0) && i + 1 < argc) {
            cfg->sampler_params.top_p = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "--top_k") == 0) && i + 1 < argc) {
            cfg->sampler_params.top_k = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--min-p") == 0 || strcmp(argv[i], "--min_p") == 0) && i + 1 < argc) {
            cfg->sampler_params.min_p = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--repeat-penalty") == 0 || strcmp(argv[i], "--repeat_penalty") == 0) && i + 1 < argc) {
            cfg->sampler_params.repeat_penalty = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--repeat-last-n") == 0 || strcmp(argv[i], "--repeat_last_n") == 0) && i + 1 < argc) {
            cfg->sampler_params.repeat_last_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->sampler_params.seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) {
            cfg->jobs_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            cfg->out_path = argv[++i];
        } else if (strcmp(argv[i], "--err") == 0 && i + 1 < argc) {
            cfg->err_path = argv[++i];
        } else if (strcmp(argv[i], "--status") == 0 && i + 1 < argc) {
            cfg->status_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            { *rc = 0; return 0; }
        }
    }
    return 1;
}

// Argument validation + everything that happens before the model is loaded (architecture, sampler/spec policy).
// Returns -1 to continue, else the exit code.
static int job_prepare(job_cfg_t *cfg) {
    cfg_apply_globals(cfg);
    if (cfg->max_context_len < 1 || cfg->max_context_len > 262144) {
        fprintf(stderr, "Error: --max-seq must be between 1 and 262144 (got %d)\n", cfg->max_context_len);
        return 2;
    }

    if (!cfg->prompt_specified && !cfg->inspect_mode && !cfg->bench_dense_batch &&
        !cfg->dump_tokens_mode && !cfg->dump_logits_path && !cfg->teacher_force_path) {
        cfg->chat_mode = true;
    }

    if (cfg->n_predict < 0) cfg->n_predict = (cfg->dump_tokens_mode || cfg->dump_logits_path) ? 32 : 128;
    if (cfg->n_runs < 1 || cfg->n_warmup < 0) {
        fprintf(stderr, "Error: --runs must be >= 1 and --warmup >= 0 (got --runs %d --warmup %d)\n", cfg->n_runs, cfg->n_warmup);
        return 2;
    }
    if (cfg->n_predict <= 0) {
        fprintf(stderr, "Error: -n must be >= 1 (got %d)\n", cfg->n_predict);
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
    g_n_gpu_expert_layers = cfg->n_gpu_expert_layers < 0 ? 0 : cfg->n_gpu_expert_layers; // extern in gpu.h
    fprintf(stderr, "[Tuff] expert split: %d layers pre-resident on GPU "
                    "(0 = none, hit/miss CPU overlap on)\n", g_n_gpu_expert_layers);

    cfg->arch = tuff_detect_arch(cfg->model_path);
    cfg->vtable = tuff_get_vtable(cfg->arch);
    if (!cfg->vtable) {
        fprintf(stderr, "[Tuff] ERROR: Unsupported or unrecognized architecture in %s\n", cfg->model_path);
        return 1;
    }
    fprintf(stderr, "[Tuff] Detected architecture: %s (%s engine)\n",
            tuff_arch_to_string(cfg->arch), cfg->vtable->name);

    cfg->is_lfm = arch_is_lfm(cfg->arch);

    // The LFM2.5 engines decode greedily only (the sampler is stored but never consulted).
    if (cfg->is_lfm && cfg->sampler_params.temp > 0.0f) {
        if (cfg->default_temp <= 0.0f) {
            fprintf(stderr, "[Tuff] ERROR: sampling not implemented for LFM2.5 (--temp %g); use --temp 0\n", (double)cfg->sampler_params.temp);
            return 2;
        }
        fprintf(stderr, "[Tuff] WARNING: LFM2.5 decodes greedily; --temp %g (default %g) is ignored\n",
                (double)cfg->sampler_params.temp, (double)cfg->default_temp);
    }

    // LFM engines: lossless n-gram speculation is opt-in via an explicit --spec-draft ngram (greedy only).
    cfg->lfm_spec = false;
    cfg->spec_nmax = cfg->g_spec_ngram_n; cfg->spec_nmin = cfg->g_spec_ngram_min; cfg->spec_kfix = 0;
    if (cfg->spec_ngram_requested && cfg->vtable->engine_forward_batch && cfg->vtable->engine_rollback) {
        {
            cfg->lfm_spec = true;
            if (cfg->spec_nmin > cfg->spec_nmax) cfg->spec_nmin = cfg->spec_nmax;
            const char *ks = getenv("LFM25_SPEC_K");
            if (ks && ks[0]) {
                cfg->spec_kfix = atoi(ks);
                if (cfg->spec_kfix < 1 || cfg->spec_kfix > SPEC_KMAX) {
                    fprintf(stderr, "Error: LFM25_SPEC_K must be between 1 and %d (got '%s')\n", SPEC_KMAX, ks);
                    return 2;
                }
                if (cfg->spec_kfix > cfg->spec_k_max) cfg->spec_k_max = cfg->spec_kfix;
            }
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------
// Model lifetime: loaded once, reused by every job whose "model key" matches.
// Model key = (model path, LFM25_FORCE_EXPERT_CACHE=1 on an LFM2.5 MoE with <= 32 experts). Reasons: the path picks the
// weights; lfm25_gpu_prepare() uploads the expert tensors of a <=32-expert MoE only when the expert cache is NOT forced
// (src/lfm25_gpu.cu), and the VRAM they occupy would otherwise be missing / wasted for the other mode. Everything else
// read from the environment (LFM25_*, TUFF_*) is read per engine_init()/per run, not at model-load time.
// ---------------------------------------------------------------------------------------------------------------
typedef struct {
    void *model;
    const engine_vtable_t *vt;
    char *path;
    int prepared;
    int force_matters;   // LFM25_FORCE_EXPERT_CACHE changes the weight upload (MoE with <= 32 experts)
    int force;           // value of the flag the weights were uploaded with
} loaded_model_t;

static int env_force_expert_cache(void) {
    const char *f = getenv("LFM25_FORCE_EXPERT_CACHE");
    return f && f[0] == '1';
}

static int model_reusable(const loaded_model_t *lm, const job_cfg_t *cfg) {
    if (!lm->model || lm->vt != cfg->vtable) return 0;
    if (strcmp(lm->path, cfg->model_path) != 0) return 0;
    if (lm->force_matters && lm->force != env_force_expert_cache()) return 0;
    return 1;
}

// Returns -1 on success, else the exit code. `prepare` = upload the weights to VRAM (everything but --inspect).
static int load_model(job_cfg_t *cfg, int prepare, loaded_model_t *lm) {
    const engine_vtable_t *vtable = cfg->vtable;
    memset(lm, 0, sizeof(*lm));
    fprintf(stderr, "[Tuff] Loading GGUF model from: %s\n", cfg->model_path);
    void *model = vtable->model_load(cfg->model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", cfg->model_path);
        return 1;
    }
    tm_mark("model_load (gguf parse + mmap)");

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
    tm_mark("page_warmup");

    lm->model = model; lm->vt = vtable; lm->path = strdup(cfg->model_path);
    if (cfg->is_lfm) {
        const lfm25_model_t *lmod = (const lfm25_model_t*)model;
        lm->force_matters = lmod->is_moe && lmod->expert_count <= 32;
        lm->force = env_force_expert_cache();
    }
    if (!prepare) return -1;

    // M2: upload dense weights to VRAM once, before any decode.
    if (vtable->gpu_prepare(model, &cfg->max_context_len, cfg->fill_ram ? 1 : 0) != 0) {
        fprintf(stderr, "[Tuff] ERROR: GPU preparation failed for %s (model not supported by the %s engine)\n", cfg->model_path, vtable->name);
        vtable->model_free(model);
        free(lm->path);
        memset(lm, 0, sizeof(*lm));
        return 1;
    }
    lm->prepared = 1;
    tm_mark("gpu_prepare (cuda init + weight upload)");
    return -1;
}

static void unload_model(loaded_model_t *lm) {
    if (!lm->model) return;
    if (lm->prepared) lm->vt->gpu_shutdown(lm->model);
    lm->vt->model_free(lm->model);
    free(lm->path);
    memset(lm, 0, sizeof(*lm));
}


static int run_job(const engine_vtable_t *vtable, void *model, job_cfg_t *cfg) {
    const char *prompt = cfg->prompt;
    const char *system_prompt = cfg->system_prompt;
    const char *log_filepath = cfg->log_filepath;
    const char *model_path = cfg->model_path;
    int max_context_len = cfg->max_context_len;
    int max_gen_tokens = cfg->max_gen_tokens;
    const bool chat_mode = cfg->chat_mode;
    const bool bench_dense_batch = cfg->bench_dense_batch;
    const bool dump_tokens_mode = cfg->dump_tokens_mode;
    const bool spec_mtp_mode = cfg->spec_mtp_mode;
    const bool spec_measure_mode = cfg->spec_measure_mode;
    const char *spec_draft_model = cfg->spec_draft_model;
    const char *dump_logits_path = cfg->dump_logits_path;
    const int n_warmup = cfg->n_warmup, n_runs = cfg->n_runs, n_predict = cfg->n_predict;
    sampler_params_t sampler_params = cfg->sampler_params;
    const char *teacher_force_path = cfg->teacher_force_path;
    const int tf_batch = cfg->tf_batch;
    const bool tf_rollback = cfg->tf_rollback;
    const tuff_arch_t arch = cfg->arch;
    const int is_lfm = cfg->is_lfm;
    const bool lfm_spec = cfg->lfm_spec;
    const int spec_nmax = cfg->spec_nmax, spec_nmin = cfg->spec_nmin, spec_kfix = cfg->spec_kfix, spec_k_max = cfg->spec_k_max;
    (void)bench_dense_batch; (void)spec_draft_model;

    if (cfg->inspect_mode) {
        vtable->model_print_inspection(model);
        return 0;
    }
    if (n_runs < 1 || n_warmup < 0 || n_predict < 1) return 2;   // job_prepare() rejects these already

    spec_static_costs();   // the cost table is per-job state: never inherit a previous job's measurement

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
            return 0;
        }
    }

    tokenizer_t *tok = tokenizer_init(vtable->model_get_gguf(model));
    tm_mark("tokenizer_init");

    if (chat_mode) {
        if (arch != TUFF_ARCH_QWEN35MOE) {
            fprintf(stderr, "[Tuff] Interactive chat is currently only supported for Qwen models.\n");
            tokenizer_free(tok);
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
        return rc;
    }

    int32_t *prompt_tokens = (int32_t*)malloc((size_t)max_context_len * sizeof(int32_t));
    if (!prompt_tokens) {
        fprintf(stderr, "[Tuff] FATAL: failed to allocate prompt_tokens for context length %d\n", max_context_len);
        tokenizer_free(tok);
        return 1;
    }
    int n_prompt = tokenizer_tokenize(tok, prompt, prompt_tokens, max_context_len);
    if (n_prompt <= 0) {
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
            return 2;
        }
    }

    if (teacher_force_path) {
        FILE *tf = fopen(teacher_force_path, "r");
        if (!tf) {
            fprintf(stderr, "[Tuff] FATAL: cannot open teacher-force file: %s\n", teacher_force_path);
            tokenizer_free(tok);
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
            return 2;
        }
        if (tf_bad) {
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            return 2;
        }
        if (n_ref == 0) {
            fprintf(stderr, "[Tuff] FATAL: teacher-force file contained 0 tokens\n");
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            return 2;
        }

        int target_steps = (n_predict > 0 && n_predict < n_ref) ? n_predict : n_ref;
        fprintf(stderr, "[Tuff] --teacher-force: testing %d steps against %s\n",
                target_steps, teacher_force_path);
        if (is_lfm && !ctx_fits("--teacher-force (prompt + reference steps)", (long)n_prompt + target_steps, max_context_len)) {
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
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
                return 3;
            }
        }

        void *eng = vtable->engine_init(model, max_context_len);
        tm_mark("engine_init");
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed\n");
            if (tf_dump) { fclose(tf_dump); remove(dump_logits_path); }
            free(ref_tokens);
            free(prompt_tokens);
            tokenizer_free(tok);
            return 3;
        }
        vtable->engine_set_sampler(eng, &sampler_params);

        if (tf_rollback && tf_batch < 2) {
            fprintf(stderr, "[Tuff] FATAL: --tf-rollback requires --tf-batch >= 2\n");
            vtable->engine_free(eng);
            tokenizer_free(tok);
            return 2;
        }
        if (tf_batch > 1) {
            if (!vtable->engine_forward_batch || (tf_rollback && !vtable->engine_rollback)) {
                fprintf(stderr, "[Tuff] FATAL: --tf-batch/--tf-rollback are not supported by the %s engine\n", vtable->name);
                vtable->engine_free(eng);
                if (tf_dump) { fclose(tf_dump); remove(dump_logits_path); }
                tokenizer_free(tok);
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
            if (!seq || !rows) { fprintf(stderr, "[Tuff] FATAL: out of memory\n"); free(seq); free(rows); vtable->engine_free(eng); return 2; }
            for (int q = 0; q < n_prompt; ++q) seq[q] = prompt_tokens[q];
            for (int q = 0; q < target_steps; ++q) seq[n_prompt + q] = ref_tokens[q];

            int matches = 0, first_divergence = -1, dump_ok = 1, n_batches = 0, n_rollbacks = 0;
            int p = 0, chunk = 0;
            while (p < total) {
                int n = total - p < tf_batch ? total - p : tf_batch;
                int32_t am[TF_MAX_BATCH];
                if (vtable->engine_forward_batch(eng, seq + p, n, p, am, rows) != 0) {
                    fprintf(stderr, "[Tuff] FATAL: engine_forward_batch failed at position %d (n=%d)\n", p, n);
                    vtable->engine_free(eng);
                    return 3;
                }
                n_batches++;
                int keep = n;
                if (tf_rollback) {
                    keep = 1 + (chunk % tf_batch);
                    if (keep > n) keep = n;
                    if (vtable->engine_rollback(eng, keep) != 0) {
                        fprintf(stderr, "[Tuff] FATAL: engine_rollback(%d) failed at position %d\n", keep, p);
                        vtable->engine_free(eng);
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
            tm_mark("teacher-forced decode (tf-batch)");
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

        tm_mark("prefill (prompt forward, incl. graph capture)");
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
        tm_mark("teacher-forced decode");

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
        tm_mark("engine_init");
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed\n");
            tuff_logit_capture_enable(0);
            tokenizer_free(tok);
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
        tm_mark("prefill (prompt forward, incl. graph capture)");
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
            return 3;
        }
        if (vocab != vtable->model_get_vocab_size(model)) {
            fprintf(stderr, "[Tuff] FATAL: captured logit width %u != model vocab_size %u\n",
                    vocab, vtable->model_get_vocab_size(model));
            vtable->engine_free(eng);
            tokenizer_free(tok);
            return 3;
        }

        FILE *fp = fopen(dump_logits_path, "wb");
        if (!fp) {
            fprintf(stderr, "[Tuff] FATAL: cannot open %s for writing: ", dump_logits_path);
            perror(NULL);
            vtable->engine_free(eng);
            tokenizer_free(tok);
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
        tm_mark("decode (dump-logits)");

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
        tm_mark("engine_init");
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed\n");
            tokenizer_free(tok);
            return 3;
        }
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];
        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
            fwd_check(is_lfm, curr_token, pos - 1);
        }
        tm_mark("prefill (prompt forward, incl. graph capture)");
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
        tm_mark("decode (dump-tokens)");
        vtable->engine_free(eng);
        tm_mark("engine_free");
        tokenizer_free(tok);
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
        tm_mark("engine_init");
        if (!eng) {
            fprintf(stderr, "[Tuff] FATAL: engine initialization failed (run %d)\n", run + 1);
            if (lfm_spec) free(lfm_sp.hist);
            if (drafter) drafter_free(drafter);
            free(tok_per_sec);
            tokenizer_free(tok);
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
        tm_mark("prefill (prompt forward, incl. graph capture)");
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

        tm_mark("warmup");
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
        tm_mark("decode (timed)");
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
        tm_mark("engine_free");
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
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// --jobs <file>: one process, one model load, many jobs.
// A job line is: [NAME=value ...] <ordinary tuff arguments> [--out <file>] [--err <file>] [--status <file>]
// The effective command line of a job is  <arguments given to this tuff invocation, minus --jobs> + <job arguments>,
// run with the environment of this process + the job's NAME=value assignments (everything a previous job assigned is
// restored first, so no variable leaks from one job to the next). stdout/stderr of the job go to --out/--err (appended
// with a last stderr line "JOB_EXIT=<code>"); --status additionally gets the exit code. Blank lines and lines starting
// with '#' are skipped. Jobs run strictly in order; the process exits 0 iff every job exited 0 (else with the first
// nonzero job exit code); the per-job codes are what scripts should consult.
// ---------------------------------------------------------------------------------------------------------------
typedef struct { char *name; char *old; } env_save_t;   // old == NULL => was unset

// Shell-like split: whitespace separated, '...' literal, "..." with \" and \\ escapes. Returns token count or -1.
static int split_line(const char *s, char ***out) {
    int cap = 16, n = 0;
    char **v = (char**)malloc((size_t)cap * sizeof(char*));
    size_t len = strlen(s);
    char *cur = (char*)malloc(len + 1);
    for (const char *p = s; *p;) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        size_t k = 0;
        while (*p && !isspace((unsigned char)*p)) {
            if (*p == '\'') {
                p++;
                while (*p && *p != '\'') cur[k++] = *p++;
                if (!*p) { free(cur); for (int i = 0; i < n; ++i) free(v[i]); free(v); return -1; }
                p++;
            } else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && (p[1] == '"' || p[1] == '\\')) p++;
                    cur[k++] = *p++;
                }
                if (!*p) { free(cur); for (int i = 0; i < n; ++i) free(v[i]); free(v); return -1; }
                p++;
            } else {
                cur[k++] = *p++;
            }
        }
        cur[k] = 0;
        if (n == cap) { cap *= 2; v = (char**)realloc(v, (size_t)cap * sizeof(char*)); }
        v[n++] = strdup(cur);
    }
    free(cur);
    *out = v;
    return n;
}

static int is_env_assignment(const char *t) {
    if (!(isalpha((unsigned char)t[0]) || t[0] == '_')) return 0;
    const char *p = t;
    while (isalnum((unsigned char)*p) || *p == '_') p++;
    return *p == '=';
}

static void env_restore(env_save_t *sv, int *n) {
    for (int i = 0; i < *n; ++i) {
        if (sv[i].old) setenv(sv[i].name, sv[i].old, 1); else unsetenv(sv[i].name);
        free(sv[i].name); free(sv[i].old);
    }
    *n = 0;
}

static const char* tok_after(char **v, int n, const char *opt) {   // value following the LAST occurrence of opt, or NULL
    const char *r = NULL;
    for (int i = 0; i + 1 < n; ++i) if (strcmp(v[i], opt) == 0) r = v[i + 1];
    return r;
}

static int run_jobs(const char *jobs_path, int cli_argc, char **cli_argv) {
    FILE *jf = fopen(jobs_path, "r");
    if (!jf) { fprintf(stderr, "[Tuff] FATAL: cannot open jobs file %s\n", jobs_path); return 2; }
    g_in_jobs = 1;
    int saved_out = dup(STDOUT_FILENO), saved_err = dup(STDERR_FILENO);
    loaded_model_t lm; memset(&lm, 0, sizeof(lm));
    env_save_t *sv = (env_save_t*)calloc(256, sizeof(env_save_t));
    int nsv = 0, overall = 0, njob = 0, nfail = 0;
    char *line = NULL; size_t lcap = 0;
    while (getline(&line, &lcap, jf) >= 0) {
        char **tk = NULL;
        int nt = split_line(line, &tk);
        if (nt == 0 || (nt > 0 && tk[0][0] == '#')) { for (int i = 0; i < nt; ++i) free(tk[i]); free(tk); continue; }
        njob++;
        int rc = 0;
        const char *out_path = NULL, *err_path = NULL;
        char **av = NULL; int ac = 0;
        if (nt < 0) {
            fprintf(stderr, "[Tuff] jobs: line %d: unterminated quote\n", njob);
            rc = 2;
        } else {
            // environment: back to the runner's own, then this job's assignments
            env_restore(sv, &nsv);
            int ia = 0;
            while (ia < nt && is_env_assignment(tk[ia]) && nsv < 256) {
                char *eq = strchr(tk[ia], '=');
                size_t nl = (size_t)(eq - tk[ia]);
                char *name = strndup(tk[ia], nl);
                const char *old = getenv(name);
                sv[nsv].name = name; sv[nsv].old = old ? strdup(old) : NULL; nsv++;
                setenv(name, eq + 1, 1);
                ia++;
            }
            av = (char**)malloc((size_t)(1 + cli_argc + nt) * sizeof(char*));
            av[ac++] = (char*)"tuff";
            for (int i = 0; i < cli_argc; ++i) av[ac++] = cli_argv[i];
            for (int i = ia; i < nt; ++i) av[ac++] = tk[i];
            out_path = tok_after(av, ac, "--out");
            err_path = tok_after(av, ac, "--err");
            g_cur_status = tok_after(av, ac, "--status");
        }
        int fo = -1, fe = -1;
        fflush(stdout); fflush(stderr);
        if (out_path) { fo = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fo >= 0) dup2(fo, STDOUT_FILENO); }
        if (err_path) { fe = open(err_path, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fe >= 0) dup2(fe, STDERR_FILENO); }

        if (rc == 0) {
            job_cfg_t cfg;
            cfg_init(&cfg);
            int prc = 0;
            tuff_exact_reset();
            tuff_logit_capture_enable(0);   // a previous job that failed mid-capture must not leave the sink armed
            tm_reset();
            if (!parse_args(ac, av, &cfg, &prc)) rc = prc;
            else if ((prc = job_prepare(&cfg)) >= 0) rc = prc;
            else if (!arch_is_lfm(cfg.arch)) {
                fprintf(stderr, "[Tuff] ERROR: --jobs supports the LFM2.5 engines only (got %s)\n", tuff_arch_to_string(cfg.arch));
                rc = 2;
            } else {
                if (lm.model && !model_reusable(&lm, &cfg)) {
                    fprintf(stderr, "[Tuff] jobs: model key changed (%s, force-expert-cache=%d) -> reloading the model\n",
                            cfg.model_path, env_force_expert_cache());
                    unload_model(&lm);
                    tm_mark("model_unload");
                }
                if (!lm.model) {
                    prc = load_model(&cfg, 1, &lm);
                    if (prc >= 0) rc = prc;
                } else {
                    fprintf(stderr, "[Tuff] jobs: reusing the loaded model %s\n", lm.path);
                }
                if (rc == 0) {
                    rc = run_job(cfg.vtable, lm.model, &cfg);
                    fflush(stdout);
                }
            }
        }
        fflush(stdout);
        fprintf(stderr, "JOB_EXIT=%d\n", rc);
        if (g_cur_status) { FILE *f = fopen(g_cur_status, "w"); if (f) { fprintf(f, "%d\n", rc); fclose(f); } }
        fflush(stderr);
        if (fo >= 0) { dup2(saved_out, STDOUT_FILENO); close(fo); }
        if (fe >= 0) { dup2(saved_err, STDERR_FILENO); close(fe); }
        fprintf(stderr, "[Tuff] job %d: exit %d%s%s\n", njob, rc, err_path ? " err=" : "", err_path ? err_path : "");
        if (rc != 0) { nfail++; if (!overall) overall = rc; }
        g_cur_status = NULL;
        free(av);
        for (int i = 0; i < nt; ++i) free(tk[i]);
        free(tk);
    }
    free(line);
    fclose(jf);
    env_restore(sv, &nsv);
    free(sv);
    unload_model(&lm);
    close(saved_out); close(saved_err);
    fprintf(stderr, "[Tuff] jobs: %d job(s), %d failed\n", njob, nfail);
    return overall;
}

int main(int argc, char **argv) {
    g_glob_def.mtp = g_spec_mtp; g_glob_def.draft = g_spec_draft; g_glob_def.draft_len = g_spec_draft_len;
    g_glob_def.ngram_n = g_spec_ngram_n; g_glob_def.ngram_min = g_spec_ngram_min;
    tm_reset();
    if (g_tm_on) fprintf(stderr, "[Tuff timing] process start -> main(): %.3f s\n", proc_age_sec());

    job_cfg_t cfg;
    cfg_init(&cfg);
    int rc = 0;
    if (!parse_args(argc, argv, &cfg, &rc)) return rc;

    if (cfg.jobs_path) {
        // Arguments other than --jobs act as defaults for every job line.
        char **rest = (char**)malloc((size_t)argc * sizeof(char*));
        int nr = 0;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) { i++; continue; }
            rest[nr++] = argv[i];
        }
        int jrc = run_jobs(cfg.jobs_path, nr, rest);
        free(rest);
        return jrc;
    }

    if ((rc = job_prepare(&cfg)) >= 0) return rc;
    tm_mark("args + arch detect");

    loaded_model_t lm;
    if ((rc = load_model(&cfg, !cfg.inspect_mode, &lm)) >= 0) return rc;
    rc = run_job(cfg.vtable, lm.model, &cfg);
    unload_model(&lm);
    tm_mark("gpu_shutdown + model_free");
    if (g_tm_on) fprintf(stderr, "[Tuff timing] total main(): %.3f s (process age %.3f s)\n", tm_mono() - g_tm_start, proc_age_sec());
    return rc;
}
