#include "model.h"
#include "gpu.h"

extern "C" {
#include "engine.h"
#include "sampler.h"

int tuff_exact(void) { return 0; }
int tuff_logit_capture_is_enabled(void) { return 0; }
void tuff_logit_capture(const float *logits, uint32_t vocab_size) { (void)logits; (void)vocab_size; }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];

    printf("[test_qwen_batch] Loading model %s...\n", model_path);
    model_t *m = model_load(model_path);
    if (!m) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    int max_seq = 1024;
    printf("[test_qwen_batch] Initializing GPU...\n");
    gpu_prepare_ex(m, &max_seq, 0);

    engine_t *eng = engine_init(m, max_seq);
    if (!eng) {
        fprintf(stderr, "Failed to init engine\n");
        return 1;
    }
    sampler_params_t sp = sampler_params_default();
    sp.temp = 0.0f;
    engine_set_sampler(eng, &sp);

    for (uint32_t l = 0; l < m->block_count; ++l) {
        if (m->layers[l].type == LAYER_TYPE_ATTENTION) {
            printf("[test_qwen_batch] Layer %u is ATTENTION\n", l);
        }
    }

    // 1. Reference single-token decode for 8 tokens starting from prompt [9419]
    int32_t prompt = 9419;
    printf("[test_qwen_batch] Running single-token reference (8 steps)...\n");
    int32_t ref_tokens[8];
    int32_t ref_inputs[8];
    ref_inputs[0] = prompt;
    int32_t cur = prompt;
    for (int p = 0; p < 8; ++p) {
        ref_tokens[p] = engine_forward_token(eng, cur, p);
        printf("  pos %d (in=%d) -> pred=%d\n", p, cur, ref_tokens[p]);
        if (p < 7) {
            ref_inputs[p + 1] = ref_tokens[p];
            cur = ref_tokens[p];
        }
    }
    engine_free(eng);

    // 2. Test engine_forward_batch with N = 1
    printf("\n[test_qwen_batch] Testing engine_forward_batch N=1 at pos 0...\n");
    eng = engine_init(m, max_seq);
    engine_set_sampler(eng, &sp);

    int32_t b1_tokens[1] = { prompt };
    int32_t b1_out[1] = { 0 };
    if (engine_forward_batch(eng, b1_tokens, 1, 0, b1_out, NULL) != 0) {
        fprintf(stderr, "engine_forward_batch failed for N=1\n");
        return 1;
    }
    printf("  N=1: in=%d -> pred=%d (expected %d)\n", prompt, b1_out[0], ref_tokens[0]);
    if (b1_out[0] != ref_tokens[0]) {
        fprintf(stderr, "FAIL: N=1 output %d != expected %d\n", b1_out[0], ref_tokens[0]);
        return 1;
    }
    printf("  N=1 PASS!\n");
    engine_free(eng);

    // 3. Test engine_forward_batch with N = 2
    printf("\n[test_qwen_batch] Testing engine_forward_batch N=2 at pos 0...\n");
    eng = engine_init(m, max_seq);
    engine_set_sampler(eng, &sp);

    int32_t b2_out[2] = { 0, 0 };
    if (engine_forward_batch(eng, ref_inputs, 2, 0, b2_out, NULL) != 0) {
        fprintf(stderr, "engine_forward_batch failed for N=2\n");
        return 1;
    }
    for (int c = 0; c < 2; ++c) {
        printf("  col %d: in=%d -> pred=%d (expected %d)\n", c, ref_inputs[c], b2_out[c], ref_tokens[c]);
        if (b2_out[c] != ref_tokens[c]) {
            fprintf(stderr, "FAIL: N=2 col %d mismatch!\n", c);
            return 1;
        }
    }
    printf("  N=2 PASS!\n");
    engine_free(eng);

    // 4. Test engine_forward_batch with N = 4
    printf("\n[test_qwen_batch] Testing engine_forward_batch N=4 at pos 0...\n");
    eng = engine_init(m, max_seq);
    engine_set_sampler(eng, &sp);

    int32_t b4_out[4] = { 0 };
    if (engine_forward_batch(eng, ref_inputs, 4, 0, b4_out, NULL) != 0) {
        fprintf(stderr, "engine_forward_batch failed for N=4\n");
        return 1;
    }
    for (int c = 0; c < 4; ++c) {
        printf("  col %d: in=%d -> pred=%d (expected %d)\n", c, ref_inputs[c], b4_out[c], ref_tokens[c]);
        if (b4_out[c] != ref_tokens[c]) {
            fprintf(stderr, "FAIL: N=4 col %d mismatch!\n", c);
            return 1;
        }
    }
    printf("  N=4 PASS!\n");
    engine_free(eng);

    // 5. Test engine_forward_batch with N = 8
    printf("\n[test_qwen_batch] Testing engine_forward_batch N=8 at pos 0...\n");
    eng = engine_init(m, max_seq);
    engine_set_sampler(eng, &sp);

    int32_t b8_out[8] = { 0 };
    float *b8_logits = (float*)malloc((size_t)8 * m->vocab_size * sizeof(float));
    if (engine_forward_batch(eng, ref_inputs, 8, 0, b8_out, b8_logits) != 0) {
        fprintf(stderr, "engine_forward_batch failed for N=8\n");
        return 1;
    }
    int b8_failed = 0;
    for (int c = 0; c < 8; ++c) {
        const float *lg = b8_logits + (size_t)c * m->vocab_size;
        printf("  col %d: in=%d -> pred=%d (logit=%.4f) (expected %d, logit=%.4f) %s\n",
               c, ref_inputs[c], b8_out[c], lg[b8_out[c]], ref_tokens[c], lg[ref_tokens[c]],
               (b8_out[c] == ref_tokens[c]) ? "OK" : "MISMATCH");
        if (b8_out[c] != ref_tokens[c]) b8_failed = 1;
    }
    free(b8_logits);
    if (b8_failed) {
        fprintf(stderr, "FAIL: N=8 mismatch!\n");
        return 1;
    }
    printf("  N=8 PASS!\n");
    engine_free(eng);

    // 6. Test Rollback: run N=4 at pos 0, rollback to n_keep=2, then run N=2 at pos 2
    printf("\n[test_qwen_batch] Testing engine_rollback (N=4 -> keep 2 -> forward 2 at pos 2)...\n");
    eng = engine_init(m, max_seq);
    engine_set_sampler(eng, &sp);

    int32_t r4_out[4] = { 0 };
    if (engine_forward_batch(eng, ref_inputs, 4, 0, r4_out, NULL) != 0) {
        fprintf(stderr, "forward_batch failed\n");
        return 1;
    }
    printf("  Forwarded 4 tokens at pos 0: [%d, %d, %d, %d]\n", r4_out[0], r4_out[1], r4_out[2], r4_out[3]);
    printf("  Rolling back to n_keep = 2...\n");
    if (engine_rollback(eng, 2) != 0) {
        fprintf(stderr, "engine_rollback(2) failed\n");
        return 1;
    }

    int32_t r2_out[2] = { 0, 0 };
    if (engine_forward_batch(eng, ref_inputs + 2, 2, 2, r2_out, NULL) != 0) {
        fprintf(stderr, "forward_batch after rollback failed\n");
        return 1;
    }
    printf("  Re-forwarded 2 tokens at pos 2: col 0 -> %d (expected %d), col 1 -> %d (expected %d)\n",
           r2_out[0], ref_tokens[2], r2_out[1], ref_tokens[3]);
    if (r2_out[0] != ref_tokens[2] || r2_out[1] != ref_tokens[3]) {
        fprintf(stderr, "FAIL: Rollback re-forward mismatch!\n");
        return 1;
    }
    printf("  Rollback PASS!\n");

    engine_free(eng);
    gpu_shutdown(m);
    model_free(m);

    printf("\n========================================================================\n");
    printf(" [QWEN-BATCH-TEST] ALL TESTS PASSED (N=1, N=2, N=4, N=8, ROLLBACK)\n");
    printf("========================================================================\n");
    return 0;
}
