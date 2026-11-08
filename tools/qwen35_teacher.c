// qwen35-teacher: feed a token-id list through the tuf qwen35 megakernel and dump the
// raw fp32 logits after every position.  This is the G1/G3 gate tool: its output is
// compared against tools/qwen35_ref.py (fp32 GGUF reference) and it can run one token
// per launch or a whole chunk in one launch (bitwise-identity check).
//
// usage: qwen35-teacher --model M.gguf --tokens ids.txt --logits-out f32
//                       [--chunk N] [--margins-out f] [--tokens-out f] [--dump-argmax]
//
// Env Q35_TEACHER_BATCH=N (2..32): run on the BATCH path (tensor-core f16-activation kernel)
// instead of the T=1 kernel.  The teacher-forced sequence is slot 0, entry 0 of a batch of N;
// slots 1..N-1 are fed other tokens of the same list (shifted by 13*slot, so they sit at the
// same position with different history) in the same launches.  N=1 is the batch path with a
// single entry.  --chunk and --greedy are ignored in this mode.
#include "qwen35.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *model_path = NULL, *tokens_path = NULL, *logits_out = NULL;
    const char *margins_out = NULL, *tokens_out = NULL, *greedy_out = NULL;
    int chunk = 1, greedy = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--temperature")) { i++; continue; }
        if (!v) { fprintf(stderr, "missing value for %s\n", a); return 2; }
        if (!strcmp(a, "--model")) model_path = v;
        else if (!strcmp(a, "--tokens")) tokens_path = v;
        else if (!strcmp(a, "--logits-out")) logits_out = v;
        else if (!strcmp(a, "--margins-out")) margins_out = v;
        else if (!strcmp(a, "--tokens-out")) tokens_out = v;
        else if (!strcmp(a, "--chunk")) chunk = atoi(v);
        else if (!strcmp(a, "--greedy")) greedy = atoi(v);
        else if (!strcmp(a, "--greedy-out")) greedy_out = v;
        else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
        i++;
    }
    if (!model_path || !tokens_path || !logits_out) {
        fprintf(stderr, "usage: qwen35-teacher --model M --tokens ids --logits-out f [--chunk N]\n");
        return 2;
    }
    int32_t *toks = (int32_t *)malloc(sizeof(int32_t) * 65536);
    int n = 0;
    FILE *f = fopen(tokens_path, "r");
    if (!f) { perror(tokens_path); return 1; }
    while (n < 65536 && fscanf(f, "%d", &toks[n]) == 1) n++;
    fclose(f);
    if (!n) { fprintf(stderr, "empty token list\n"); return 1; }
    if (chunk < 1) chunk = 1;
    if (chunk > 16) chunk = 16;

    q35_model *m = q35_model_load(model_path);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    int ctx = n + (greedy > 0 ? greedy : 0) + 16;
    if (q35_prepare(m, &ctx) != 0) { fprintf(stderr, "prepare failed\n"); return 1; }
    const char *tb = getenv("Q35_TEACHER_BATCH");
    const int batchN = tb ? atoi(tb) : 0;
    if (tb && (batchN < 1 || batchN > 32)) { fprintf(stderr, "Q35_TEACHER_BATCH must be 1..32\n"); return 2; }
    if (batchN > 0 && greedy > 0) { fprintf(stderr, "--greedy is not supported with Q35_TEACHER_BATCH\n"); return 2; }
    q35_engine *e = q35_engine_init(m, ctx, batchN > 0 ? batchN : 1);
    if (!e) { fprintf(stderr, "engine init failed\n"); return 1; }
    const int V = q35_model_vocab(m);
    float *logits = (float *)malloc(sizeof(float) * (size_t)V * n);
    int32_t *am = (int32_t *)malloc(sizeof(int32_t) * n);
    if (batchN > 0) {
        float *bl = (float *)malloc(sizeof(float) * (size_t)V * batchN);
        int32_t bam[32];
        int bslots[32], bpos[32], btok[32];
        for (int s2 = 0; s2 < batchN; s2++) q35_engine_reset(e, s2);
        for (int p = 0; p < n; p++) {
            for (int s2 = 0; s2 < batchN; s2++) {
                bslots[s2] = s2;
                bpos[s2] = p;
                btok[s2] = toks[(p + 13 * s2) % n];
            }
            if (q35_engine_decode_batch(e, bslots, bpos, btok, batchN, bam, bl) != 0) {
                fprintf(stderr, "batch step failed at %d\n", p);
                return 1;
            }
            memcpy(logits + (size_t)p * V, bl, sizeof(float) * (size_t)V);
            am[p] = bam[0];
        }
        free(bl);
        chunk = n;   // skip the T=1 loop below
    }
    for (int p = 0; p < n; p += chunk) {
        if (batchN > 0) break;
        int c = n - p < chunk ? n - p : chunk;
        if (q35_engine_steps(e, 0, toks + p, c, p, am + p, logits + (size_t)p * V) != 0) {
            fprintf(stderr, "step failed at %d\n", p);
            return 1;
        }
    }
    FILE *fl = fopen(logits_out, "wb");
    if (!fl) { perror(logits_out); return 1; }
    fwrite(logits, sizeof(float), (size_t)V * n, fl);
    fclose(fl);
    if (margins_out) {
        FILE *fm = fopen(margins_out, "w");
        for (int p = 0; p < n; p++) {
            const float *lg = logits + (size_t)p * V;
            int b0 = 0, b1 = 1;
            for (int i = 1; i < V; i++) {
                if (lg[i] > lg[b0]) { b1 = b0; b0 = i; }
                else if (i != b0 && (b1 == b0 || lg[i] > lg[b1])) b1 = i;
            }
            fprintf(fm, "%d %d %.6f\n", p, b0, lg[b0] - lg[b1]);
        }
        fclose(fm);
    }
    if (tokens_out) {
        FILE *ft = fopen(tokens_out, "w");
        for (int p = 0; p < n; p++) fprintf(ft, "%d\n", am[p]);
        fclose(ft);
    }
    if (greedy > 0 && greedy_out) {
        FILE *fg = fopen(greedy_out, "w");
        if (!fg) { perror(greedy_out); return 1; }
        int32_t last = am[n - 1];
        for (int g = 0; g < greedy; g++) {
            fprintf(fg, "%d\n", last);
            if (g + 1 == greedy) break;
            int32_t nxt = -1;
            if (q35_engine_step(e, 0, last, n + g, &nxt, NULL) != 0) {
                fprintf(stderr, "greedy step failed at %d\n", n + g);
                fclose(fg);
                return 1;
            }
            last = nxt;
        }
        fclose(fg);
    }
    fprintf(stderr, "fed %d tokens (%s), vocab %d\n", n, batchN > 0 ? "batch path" : "T=1 path", V);
    free(logits); free(am); free(toks);
    q35_engine_free(e);
    q35_model_free(m);
    return 0;
}
