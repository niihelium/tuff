// qwen35-batch-bench: Phase 2 engine benchmark and G4 invariance gate.
//
//   --invariance B : G4.  Each slot is run alone through the batch path (n = 1, BT = 1
//                    tier) and then together with B slots (any slot index, mixed prompt
//                    lengths); every per-step logit row must match bitwise.
//   --bench B N    : B slots, N decode steps each, median of 3 rounds -> aggregate tok/s.
//
// usage: qwen35-batch-bench --model M.gguf --invariance B [--ctx C] [--prompt-len L]
//        qwen35-batch-bench --model M.gguf --bench B --steps N [--ctx C] [--prompt-len L] [--rounds R]
#include "qwen35.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

int main(int argc, char **argv) {
    const char *model = NULL;
    int inv = 0, bench = 0, bench_B = 1, nsteps = 0, ctx = 2048, plen = 24, rounds = 3;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "missing value for %s\n", a); return 2; }
        if (!strcmp(a, "--model")) model = v;
        else if (!strcmp(a, "--invariance")) inv = atoi(v);
        else if (!strcmp(a, "--bench")) { bench = 1; bench_B = atoi(v); }
        else if (!strcmp(a, "--steps")) nsteps = atoi(v);
        else if (!strcmp(a, "--ctx")) ctx = atoi(v);
        else if (!strcmp(a, "--prompt-len")) plen = atoi(v);
        else if (!strcmp(a, "--rounds")) rounds = atoi(v);
        else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
        i++;
    }
    if (!model || (!inv && !bench)) {
        fprintf(stderr, "usage: %s --model M --invariance B | --bench B --steps N [--ctx C]\n", argv[0]);
        return 2;
    }
    int B = inv ? inv : bench_B;
    if (B < 1 || B > 32) { fprintf(stderr, "B must be 1..32\n"); return 2; }
    if (nsteps <= 0 && !inv) { fprintf(stderr, "need --steps N (or --bench B N)\n"); return 2; }
    q35_model *m = q35_model_load(model);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    if (q35_prepare(m, &ctx) != 0) { fprintf(stderr, "prepare failed\n"); return 1; }
    const int V = q35_model_vocab(m);
    q35_engine *e = q35_engine_init(m, ctx, B);
    if (!e) { fprintf(stderr, "engine init failed\n"); return 1; }

    // prompt and teacher-forced continuation token lists, deterministic and distinct per slot
    int *prompt = (int *)malloc(sizeof(int) * (size_t)B * plen);
    int *cont = (int *)malloc(sizeof(int) * (size_t)B * nsteps);
    unsigned s = 12345;
    for (int b = 0; b < B; b++)
        for (int i = 0; i < plen; i++) { s = s * 1664525u + 1013904223u; prompt[b * plen + i] = 1000 + (s % 100000); }
    for (int b = 0; b < B; b++)
        for (int i = 0; i < nsteps; i++) { s = s * 1664525u + 1013904223u; cont[b * nsteps + i] = 1000 + (s % 100000); }

    if (getenv("Q35_DUP")) {
        // two slots with identical state must produce bitwise identical logits
        int nb = B < 2 ? 2 : B;
        q35_engine *e2 = e;
        (void)e2;
        int *sl = (int *)malloc(sizeof(int) * nb), *po = (int *)malloc(sizeof(int) * nb);
        int *tk = (int *)malloc(sizeof(int) * nb);
        for (int b = 0; b < nb; b++) q35_engine_reset(e, b);
        for (int i = 0; i < plen; i++) {
            for (int b = 0; b < nb; b++) { sl[b] = b; po[b] = i; tk[b] = 1000 + i; }
            q35_engine_decode_batch(e, sl, po, tk, nb, NULL, NULL);
        }
        for (int b = 0; b < nb; b++) { sl[b] = b; po[b] = plen; tk[b] = 4242; }
        float *lg = (float *)malloc((size_t)nb * V * sizeof(float));
        q35_engine_decode_batch(e, sl, po, tk, nb, NULL, lg);
        int bad = 0;
        for (int b = 1; b < nb; b++)
            if (memcmp(lg, lg + (size_t)b * V, sizeof(float) * V)) {
                float mx = 0;
                for (int q = 0; q < V; q++) mx = fmaxf(mx, fabsf(lg[q] - lg[(size_t)b * V + q]));
                printf("DUP slot %d differs from slot 0: max %.6g\n", b, mx);
                bad++;
            }
        printf("DUP n=%d: %s\n", nb, bad ? "FAIL" : "PASS (all slots bitwise equal)");
        return bad ? 1 : 0;
    }
    if (inv) {
        // G4: a slot's logits are bitwise identical whether it runs alone (BT = 1 tier) or
        // with 1..B-1 other slots, at any slot index, with mixed prompt lengths and mixed
        // prompt/decode positions.  The reference is the batch path itself at n = 1.
        int nst = nsteps > 0 ? nsteps : 8;
        if (nst > nsteps && nsteps > 0) nst = nsteps;
        int *plen_s = (int *)malloc(sizeof(int) * B);
        for (int b = 0; b < B; b++) plen_s[b] = 8 + (b * 5) % 17;   // mixed prompt lengths
        int maxplen = 0;
        for (int b = 0; b < B; b++) if (plen_s[b] > maxplen) maxplen = plen_s[b];
        int32_t *ptok = (int32_t *)malloc(sizeof(int32_t) * (size_t)B * maxplen);
        unsigned s2 = 99991;
        for (int b = 0; b < B; b++)
            for (int i = 0; i < maxplen; i++) { s2 = s2 * 1664525u + 1013904223u; ptok[b * maxplen + i] = 1000 + (s2 % 100000); }
        int32_t *ctok = (int32_t *)malloc(sizeof(int32_t) * (size_t)B * nst);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < nst; i++) { s2 = s2 * 1664525u + 1013904223u; ctok[b * nst + i] = 1000 + (s2 % 100000); }

        const double Vf = (double)V * sizeof(float);
        const size_t rows = (size_t)B * (size_t)nst;
        float *solo = (float *)malloc(rows * Vf);
        float *bat = (float *)malloc(rows * Vf);
        int32_t am;
        // solo: each slot alone, n = 1 (BT = 1 tier), its own prompt length
        for (int b = 0; b < B; b++) {
            q35_engine_reset(e, b);
            for (int i = 0; i < plen_s[b]; i++) {
                int sl = b, po = i;
                if (q35_engine_decode_batch(e, &sl, &po, &ptok[b * maxplen + i], 1, &am, NULL) != 0) {
                    fprintf(stderr, "solo prefill failed\n"); return 1;
                }
            }
            for (int k = 0; k < nst; k++) {
                int sl = b, po = plen_s[b] + k;
                if (q35_engine_decode_batch(e, &sl, &po, &ctok[b * nst + k], 1, &am,
                                            solo + ((size_t)b * nst + k) * V) != 0) {
                    fprintf(stderr, "solo step %d/%d failed\n", b, k); return 1;
                }
            }
        }
        int *order = (int *)malloc(sizeof(int) * B);
        int *slots = (int *)malloc(sizeof(int) * B);
        int *pos = (int *)malloc(sizeof(int) * B);
        int *toks = (int *)malloc(sizeof(int) * B);
        int fails = 0;
        for (int pass = 0; pass < 2; pass++) {
            for (int b = 0; b < B; b++) order[b] = pass ? ((b * 7 + 3) % B) : b;
            for (int b = 0; b < B; b++) q35_engine_reset(e, b);
            // prompts interleaved across slots, one token per slot per launch
            for (int i = 0; i < maxplen; i++) {
                int n = 0;
                for (int j = 0; j < B; j++) {
                    const int b2 = order[j];
                    if (i < plen_s[b2]) { slots[n] = b2; pos[n] = i; toks[n] = ptok[b2 * maxplen + i]; n++; }
                }
                if (n && q35_engine_decode_batch(e, slots, pos, toks, n, NULL, NULL) != 0) {
                    fprintf(stderr, "batch prefill failed\n"); return 1;
                }
            }
            for (int k = 0; k < nst; k++) {
                for (int i = 0; i < B; i++) {
                    const int b = order[i];
                    slots[i] = b;
                    pos[i] = plen_s[b] + k;
                    toks[i] = ctok[b * nst + k];
                }
                if (q35_engine_decode_batch(e, slots, pos, toks, B, NULL, bat) != 0) {
                    fprintf(stderr, "batch step %d failed\n", k); return 1;
                }
                for (int i = 0; i < B; i++) {
                    const int b = order[i];
                    const float *ref = solo + ((size_t)b * nst + k) * V;
                    if (memcmp(bat + (size_t)i * V, ref, Vf) != 0) {
                        float mx = 0;
                        for (int q = 0; q < V; q++) mx = fmaxf(mx, fabsf(bat[(size_t)i * V + q] - ref[q]));
                        int bj = -1;
                        float bm = 1e30f;
                        for (int j = 0; j < B; j++) {
                            float m2 = 0;
                            const float *sj = solo + ((size_t)j * nst + k) * V;
                            for (int q = 0; q < V; q++)
                                m2 = fmaxf(m2, fabsf(bat[(size_t)i * V + q] - sj[q]));
                            if (m2 < bm) { bm = m2; bj = j; }
                        }
                        fprintf(stderr, "G4 FAIL pass=%d step=%d slot=%d (index %d) max|dlogit|=%.6g best_slot=%d (%.3g)\n",
                                pass, k, b, i, mx, bj, bm);
                        if (++fails >= 4) break;
                    }
                }
            }
        }
        printf("G4 %s: %d slots x %d steps, mixed prompt lengths (8..%d), two slot orders, %s\n",
               fails ? "FAIL" : "PASS", B, nst, maxplen,
               fails ? "batch logits differ from the solo batch-path run"
                     : "bitwise identical (solo BT=1 tier vs B-slot tier)");
        return fails ? 1 : 0;
    }

    // bench: feed prompts, then N batched decode steps repeatedly
    int32_t am[32];
    for (int b = 0; b < B; b++) {
        q35_engine_reset(e, b);
        q35_engine_steps(e, b, prompt + b * plen, plen, 0, &am[b], NULL);
    }
    int *slots = (int *)malloc(sizeof(int) * B);
    int *pos = (int *)malloc(sizeof(int) * B);
    int *toks = (int *)malloc(sizeof(int) * B);
    for (int b = 0; b < B; b++) slots[b] = b;
    int nfail = 0;
    double *rates = (double *)malloc(sizeof(double) * (rounds > 0 ? rounds : 1));
    for (int r = 0; r < rounds; r++) {
        // fresh state so every round does the same work
        for (int b = 0; b < B; b++) {
            q35_engine_reset(e, b);
            q35_engine_steps(e, b, prompt + b * plen, plen, 0, &am[b], NULL);
        }
        double t0 = now_s();
        for (int k = 0; k < nsteps; k++) {
            for (int b = 0; b < B; b++) { pos[b] = plen + k; toks[b] = cont[b * nsteps + k]; }
            if (q35_engine_decode_batch(e, slots, pos, toks, B, am, NULL) != 0) {
                fprintf(stderr, "bench step failed (step %d)\n", k);
                if (getenv("Q35_BENCH_CONTINUE")) { nfail++; continue; }   // watchdog recovery test
                return 1;
            }
        }
        double dt = now_s() - t0;
        rates[r] = (double)B * nsteps / dt;
        printf("round %d: %d slots x %d steps in %.3f s -> %.1f tok/s\n", r + 1, B, nsteps, dt, rates[r]);
    }
    qsort(rates, rounds, sizeof(double), cmp_int);
    if (nfail) printf("bench steps that returned an error: %d\n", nfail);
    printf("P2 engine: B=%d aggregate %.1f tok/s (median of %d)\n", B, rates[rounds / 2], rounds);
    return 0;
}
