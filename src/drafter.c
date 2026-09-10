#include "drafter.h"
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>

struct drafter_s {
    struct llama_model   *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    int32_t vocab_size;

    // Asynchronous worker thread state
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t cv_in;
    pthread_cond_t cv_out;

    int32_t req_token;
    int     req_pos;
    int32_t res_draft;
    bool    has_req;
    bool    has_res;
    bool    shutdown;
};

static void* drafter_worker_fn(void *arg) {
    drafter_t *d = (drafter_t*)arg;

    while (1) {
        pthread_mutex_lock(&d->mtx);
        while (!d->has_req && !d->shutdown) {
            pthread_cond_wait(&d->cv_in, &d->mtx);
        }
        if (d->shutdown) {
            pthread_mutex_unlock(&d->mtx);
            break;
        }

        int32_t tok = d->req_token;
        int     pos = d->req_pos;
        d->has_req = false;
        pthread_mutex_unlock(&d->mtx);

        // Run decode of the single token on CPU with exact position
        struct llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0]     = tok;
        batch.pos[0]       = pos;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;
        int32_t next_draft = -1;

        if (llama_decode(d->ctx, batch) == 0) {
            const float *logits = llama_get_logits_ith(d->ctx, -1);
            if (logits) {
                float max_val = logits[0];
                int32_t max_tok = 0;
                for (int32_t i = 1; i < d->vocab_size; ++i) {
                    if (logits[i] > max_val) {
                        max_val = logits[i];
                        max_tok = i;
                    }
                }
                next_draft = max_tok;
            }
        }
        llama_batch_free(batch);

        pthread_mutex_lock(&d->mtx);
        d->res_draft = next_draft;
        d->has_res = true;
        pthread_cond_signal(&d->cv_out);
        pthread_mutex_unlock(&d->mtx);
    }
    return NULL;
}

drafter_t* drafter_init(const char *model_path, int n_threads, int n_gpu_layers) {
    if (!model_path) return NULL;

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    static ggml_backend_dev_t devices[2];
    if (n_gpu_layers == 0) {
        devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        devices[1] = NULL;
        mparams.devices = devices;
    }

    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[DRAFTER ERROR] Failed to load draft model from: %s\n", model_path);
        return NULL;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t vocab_size = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_ubatch = 128;
    cparams.offload_kqv = false;
    cparams.op_offload = false;
    cparams.n_threads = n_threads > 0 ? n_threads : 4;
    cparams.n_threads_batch = cparams.n_threads;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[DRAFTER ERROR] Failed to create draft llama_context\n");
        llama_model_free(model);
        return NULL;
    }

    drafter_t *d = (drafter_t*)calloc(1, sizeof(drafter_t));
    d->model = model;
    d->ctx = ctx;
    d->vocab = vocab;
    d->vocab_size = vocab_size;
    d->res_draft = -1;

    pthread_mutex_init(&d->mtx, NULL);
    pthread_cond_init(&d->cv_in, NULL);
    pthread_cond_init(&d->cv_out, NULL);

    if (pthread_create(&d->thread, NULL, drafter_worker_fn, d) != 0) {
        fprintf(stderr, "[DRAFTER ERROR] Failed to create background worker thread\n");
        llama_free(ctx);
        llama_model_free(model);
        free(d);
        return NULL;
    }

    fprintf(stderr, "[DRAFTER] Loaded background drafter model (%s, vocab: %d, %d threads, %d GPU layers)\n",
            model_path, vocab_size, cparams.n_threads, n_gpu_layers);
    return d;
}

void drafter_prefill(drafter_t *d, const int32_t *prompt_tokens, int n_prompt) {
    if (!d || !prompt_tokens || n_prompt <= 0) return;

    llama_memory_seq_rm(llama_get_memory(d->ctx), -1, -1, -1);
    llama_memory_clear(llama_get_memory(d->ctx), true);

    // Synchronous prefill of prompt tokens
    struct llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i]     = prompt_tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_prompt - 1);
    }
    batch.n_tokens = n_prompt;

    if (llama_decode(d->ctx, batch) != 0) {
        fprintf(stderr, "[DRAFTER WARNING] Prompt prefill decode failed\n");
    } else {
        const float *logits = llama_get_logits_ith(d->ctx, -1);
        if (logits) {
            float max_val = logits[0];
            int32_t max_tok = 0;
            for (int32_t i = 1; i < d->vocab_size; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    max_tok = i;
                }
            }
            d->res_draft = max_tok;
            d->has_res = true;
        }
    }
    llama_batch_free(batch);
}

void drafter_submit_accepted(drafter_t *d, int32_t accepted_token, int pos) {
    if (!d) return;
    pthread_mutex_lock(&d->mtx);
    d->req_token = accepted_token;
    d->req_pos   = pos;
    d->has_req   = true;
    d->has_res   = false;
    pthread_cond_signal(&d->cv_in);
    pthread_mutex_unlock(&d->mtx);
}

int32_t drafter_poll_draft(drafter_t *d) {
    if (!d) return -1;
    pthread_mutex_lock(&d->mtx);
    while (!d->has_res && !d->shutdown) {
        pthread_cond_wait(&d->cv_out, &d->mtx);
    }
    int32_t draft = d->res_draft;
    pthread_mutex_unlock(&d->mtx);
    return draft;
}

void drafter_rollback(drafter_t *d, int valid_pos) {
    if (!d || valid_pos < 0) return;
    llama_memory_seq_rm(llama_get_memory(d->ctx), 0, valid_pos, -1);
}

void drafter_free(drafter_t *d) {
    if (!d) return;
    pthread_mutex_lock(&d->mtx);
    d->shutdown = true;
    pthread_cond_signal(&d->cv_in);
    pthread_mutex_unlock(&d->mtx);

    pthread_join(d->thread, NULL);

    pthread_mutex_destroy(&d->mtx);
    pthread_cond_destroy(&d->cv_in);
    pthread_cond_destroy(&d->cv_out);

    if (d->ctx) llama_free(d->ctx);
    if (d->model) llama_model_free(d->model);
    free(d);
}
