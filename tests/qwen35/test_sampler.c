// G6 helper: deterministic fixed-logit sampler driver.  The independent reference and the
// chi-square check live in tests/qwen35/sampler_ref.py.

#include "q35_sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float LOGITS[8] = { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 0.0f, -1.0f };

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s hist SEED N | greedy | det SEED\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "hist") == 0 && argc >= 4) {
        uint64_t seed = strtoull(argv[2], NULL, 10);
        int n = atoi(argv[3]);
        q35_sampling p;
        q35_sampling_defaults(&p);
        p.temp = 1.0f;
        p.top_k = 8;
        p.top_p = 1.0f;
        p.min_p = 0.0f;
        p.penalty_last_n = 0;
        uint64_t st = q35_rng_seed(seed, 0, 0);
        int cnt[8] = {0};
        for (int i = 0; i < n; i++) {
            float lg[8];
            memcpy(lg, LOGITS, sizeof(lg));
            int t = q35_sampling_sample(lg, 8, &p, NULL, 0, &st);
            if (t < 0 || t >= 8) { fprintf(stderr, "bad token %d\n", t); return 1; }
            cnt[t]++;
        }
        printf("{\"n\":%d,\"counts\":[", n);
        for (int i = 0; i < 8; i++) printf("%s%d", i ? "," : "", cnt[i]);
        printf("]}\n");
        return 0;
    }
    if (strcmp(argv[1], "greedy") == 0) {
        q35_sampling p;
        q35_sampling_defaults(&p);
        p.temp = 0.0f;
        p.repeat_penalty = 1.5f;
        p.freq_penalty = 0.5f;
        p.presence_penalty = 1.0f;
        p.penalty_last_n = 4;
        int32_t hist[] = { 0, 0, 1, 2 };
        float lg[8] = { 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f };
        int t = q35_sampling_greedy(lg, 8, &p, hist, 4);
        printf("{\"greedy\":%d}\n", t);
        return 0;
    }
    if (strcmp(argv[1], "det") == 0 && argc >= 3) {
        uint64_t seed = strtoull(argv[2], NULL, 10);
        q35_sampling p;
        q35_sampling_defaults(&p);
        p.temp = 0.9f;
        p.top_k = 8;
        p.top_p = 1.0f;
        p.min_p = 0.0f;
        p.penalty_last_n = 0;
        uint64_t st = q35_rng_seed(seed, 0, 0);
        int seq[20];
        for (int i = 0; i < 20; i++) {
            float lg[8];
            memcpy(lg, LOGITS, sizeof(lg));
            seq[i] = q35_sampling_sample(lg, 8, &p, NULL, 0, &st);
        }
        printf("{\"seq\":[");
        for (int i = 0; i < 20; i++) printf("%s%d", i ? "," : "", seq[i]);
        printf("]}\n");
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 2;
}
