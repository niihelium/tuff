// qwen35-det-probe: synthetic determinism bisect for the batched decode path.
//
// From a fresh reset, runs ONE batch launch (n entries, all at pos 0, fixed tokens) R times
// and checks that every repeat produces the same intermediate buffers.  With --op K the
// kernel stops after op K of layer 0 (Q35_DBG_OPS), so sweeping K finds the first op whose
// output is not reproducible.  --op 0 runs the whole model and compares logits.
//
// usage: qwen35-det-probe --model M.gguf [--n N] [--reps R] [--op K] [--dup]
//   --dup : all n entries get the same token (entries must then match each other too)
// Prints one line per op: "op K n N: <buf> distinct=<d> ..." then "__RESULT PASS|FAIL".
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qwen35.h"

static uint64_t fnv(const void *p, size_t n, size_t stride, size_t take) {
    // hash `take` bytes out of every `stride` bytes (lets us drop the epoch tag of 8-byte ll words)
    const unsigned char *c = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i + stride <= n; i += stride)
        for (size_t j = 0; j < take; j++) { h ^= c[i + j]; h *= 1099511628211ull; }
    return h;
}

static unsigned char *slurp(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)malloc(*sz);
    if (fread(b, 1, *sz, f) != *sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    const char *model = NULL;
    int n = 1, reps = 5, op = 0, dup = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "0";
        if (!strcmp(a, "--model")) { model = v; i++; }
        else if (!strcmp(a, "--n")) { n = atoi(v); i++; }
        else if (!strcmp(a, "--reps")) { reps = atoi(v); i++; }
        else if (!strcmp(a, "--op")) { op = atoi(v); i++; }
        else if (!strcmp(a, "--dup")) dup = 1;
    }
    if (!model || n < 1 || n > 32 || reps < 2) {
        fprintf(stderr, "usage: %s --model M [--n N] [--reps R>=2] [--op K] [--dup]\n", argv[0]);
        return 2;
    }
    int ctx = 256;
    q35_model *m = q35_model_load(model);
    if (!m || q35_prepare(m, &ctx) != 0) { fprintf(stderr, "load failed\n"); return 1; }
    const int V = q35_model_vocab(m);
    q35_engine *e = q35_engine_init(m, ctx, n);
    if (!e) { fprintf(stderr, "engine init failed\n"); return 1; }

    char buf[16];
    snprintf(buf, sizeof buf, "%d", op);
    if (op > 0) { setenv("Q35_DBG_OPS", buf, 1); setenv("Q35_DUMP", "1", 1); }
    else { unsetenv("Q35_DBG_OPS"); unsetenv("Q35_DUMP"); }

    int sl[32], po[32];
    int32_t tk[32], am[32];
    for (int b = 0; b < n; b++) { sl[b] = b; po[b] = 0; tk[b] = dup ? 4242 : 1000 + 7919 * b; }

    // ll buffers: keep the low 4 bytes (value), drop the epoch tag
    const char *files[] = {"/tmp/q35_vecf.bin", "/tmp/q35_projl.bin", "/tmp/q35_xl.bin",
                           "/tmp/q35_corel.bin", "/tmp/q35_actl.bin"};
    const int stride[] = {1, 8, 8, 8, 8}, take[] = {1, 4, 4, 4, 4};
    const int nf = op > 0 ? 5 : 0;
    uint64_t h[64][6];
    float *lg = (float *)malloc((size_t)n * V * sizeof(float));
    float *lg0 = (float *)malloc((size_t)n * V * sizeof(float));
    float maxd = 0;
    for (int r = 0; r < reps && r < 64; r++) {
        for (int b = 0; b < n; b++) q35_engine_reset(e, b);
        if (q35_engine_decode_batch(e, sl, po, tk, n, am, op > 0 ? NULL : lg) != 0) {
            printf("op %d n %d: launch failed at rep %d\n__RESULT FAIL\n", op, n, r);
            return 1;
        }
        for (int f = 0; f < nf; f++) {
            size_t sz = 0;
            unsigned char *d = slurp(files[f], &sz);
            h[r][f] = d ? fnv(d, sz, stride[f], take[f]) : 0;
            free(d);
        }
        if (op == 0) {
            h[r][5] = fnv(lg, (size_t)n * V * 4, 1, 1);
            if (r == 0) memcpy(lg0, lg, (size_t)n * V * 4);
            else for (size_t q = 0; q < (size_t)n * V; q++) maxd = fmaxf(maxd, fabsf(lg[q] - lg0[q]));
        }
    }
    const char *names[] = {"vecf", "projl", "xl", "corel", "actl", "logits"};
    int bad = 0;
    for (int f = 0; f < 6; f++) {
        if ((op > 0) != (f < 5)) continue;
        int distinct = 0;
        uint64_t seen[64];
        for (int r = 0; r < reps && r < 64; r++) {
            int k = 0;
            while (k < distinct && seen[k] != h[r][f]) k++;
            if (k == distinct) seen[distinct++] = h[r][f];
        }
        printf("op %d n %d: %-6s distinct=%d/%d\n", op, n, names[f], distinct, reps);
        if (distinct > 1) bad++;
    }
    if (op == 0) {
        printf("op 0 n %d: logits max|d| across reps %.6g\n", n, maxd);
        if (dup && n > 1) {
            float md = 0;
            for (int b = 1; b < n; b++)
                for (int q = 0; q < V; q++) md = fmaxf(md, fabsf(lg[(size_t)b * V + q] - lg[q]));
            printf("op 0 n %d: dup entries max|d| vs entry 0 %.6g\n", n, md);
            if (md > 0) bad++;
        }
    }
    printf("__RESULT %s op=%d n=%d\n", bad ? "FAIL" : "PASS", op, n);
    return bad ? 1 : 0;
}
