#ifndef CHAT_H
#define CHAT_H

#include "model.h"
#include "engine.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    char *log_filepath;
    FILE *log_file;
    bool  enabled;
    int   turn_count;
} chat_logger_t;

typedef struct {
    const char *model_path;
    const char *system_prompt;
    const char *log_filepath;
    int         max_context_len;
    int         max_gen_tokens;
    bool        spec_mtp;
    int         n_gpu_expert_layers;
    sampler_params_t sampler;
} chat_config_t;

typedef struct {
    const char *user_text;
    const char *assistant_text;
    int   prompt_tokens;
    int   cached_tokens;
    int   gen_tokens;
    double prefill_time_s;
    double prefill_tok_per_sec;
    double gen_time_s;
    double gen_tok_per_sec;
    double ttft_ms;
    double total_time_s;
} chat_turn_stats_t;

// Logger functions
chat_logger_t* chat_logger_init(const char *filepath, const char *model_path, const model_t *model);
void chat_logger_log_turn(chat_logger_t *logger, int turn_idx, const chat_turn_stats_t *stats);
void chat_logger_free(chat_logger_t *logger);

// Main TUI chat entrypoint
int run_interactive_chat(model_t *model, tokenizer_t *tok, const chat_config_t *config);

#endif // CHAT_H
