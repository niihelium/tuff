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
    const char *spec_draft_model = NULL;
    const char *dump_logits_path = NULL; // non-NULL => --dump-logits mode
    int n_predict = -1; // resolved below based on mode
    int n_warmup = 32;
    int n_runs = 3;
    int n_gpu_expert_layers = 0; // 0 => auto (computed below); set g_n_gpu_expert_layers in gpu.h
    sampler_params_t sampler_params = sampler_params_default();

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
            g_spec_draft = SPEC_DRAFT_NONE;
            spec_mtp_mode = false;
            g_spec_mtp = 0;
        } else if (strcmp(argv[i], "--spec-draft") == 0 && i + 1 < argc) {
            spec_mtp_mode = true;
            g_spec_mtp = 1;
            const char *s = argv[++i];
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
    vtable->gpu_prepare(model, &max_context_len, fill_ram ? 1 : 0);

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

    if (teacher_force_path) {
        FILE *tf = fopen(teacher_force_path, "r");
        if (!tf) {
            fprintf(stderr, "[Tuff] FATAL: cannot open teacher-force file: %s\n", teacher_force_path);
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }
        int32_t ref_tokens[65536];
        int n_ref = 0;
        char line[256];
        while (fgets(line, sizeof(line), tf) && n_ref < max_context_len) {
            int t = 0;
            if (sscanf(line, "%d", &t) == 1) {
                ref_tokens[n_ref++] = (int32_t)t;
            }
        }
        fclose(tf);
        if (n_ref == 0) {
            fprintf(stderr, "[Tuff] FATAL: teacher-force file contained 0 tokens\n");
            tokenizer_free(tok);
            vtable->gpu_shutdown(model);
            vtable->model_free(model);
            return 2;
        }

        int target_steps = (n_predict > 0 && n_predict < n_ref) ? n_predict : n_ref;
        fprintf(stderr, "[Tuff] --teacher-force: testing %d steps against %s\n",
                target_steps, teacher_force_path);

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
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];
        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
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

        void *eng = vtable->engine_init(model, max_context_len);
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];
        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
        }
        for (int i = 0; i < n_predict; ++i) {
            printf("%d\n", curr_token);
            fflush(stdout);
            curr_token = vtable->engine_forward_token(eng, curr_token, pos++);
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

    for (int run = 0; run < n_runs; ++run) {
        void *eng = vtable->engine_init(model, max_context_len);
        vtable->engine_set_sampler(eng, &sampler_params);
        int pos = 0;
        int32_t curr_token = prompt_tokens[0];

        int32_t *warmup_seq = (int32_t*)malloc((size_t)(n_prompt + n_warmup + 1) * sizeof(int32_t));
        memcpy(warmup_seq, prompt_tokens, (size_t)n_prompt * sizeof(int32_t));

        for (int i = 0; i < n_prompt; ++i) {
            curr_token = vtable->engine_forward_token(eng, prompt_tokens[i], pos++);
        }

        for (int i = 0; i < n_warmup; ++i) {
            warmup_seq[n_prompt + i] = curr_token;
            curr_token = vtable->engine_forward_token(eng, curr_token, pos++);
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

        vtable->engine_free(eng);
    }
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
