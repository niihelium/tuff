#define _GNU_SOURCE
#include "chat.h"
#include "gpu.h"
#include "tuff_exact.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <omp.h>
#include <sys/resource.h>

// ANSI Colors and Styling
#define ANSI_RESET       "\033[0m"
#define ANSI_BOLD        "\033[1m"
#define ANSI_DIM         "\033[2m"
#define ANSI_ITALIC      "\033[3m"
#define ANSI_UNDERLINE   "\033[4m"
#define ANSI_RED         "\033[31m"
#define ANSI_GREEN       "\033[32m"
#define ANSI_YELLOW      "\033[33m"
#define ANSI_BLUE        "\033[34m"
#define ANSI_MAGENTA     "\033[35m"
#define ANSI_CYAN        "\033[36m"
#define ANSI_WHITE       "\033[37m"
#define ANSI_BG_DARK     "\033[48;5;235m"
#define ANSI_BRIGHT_CYAN "\033[96m"
#define ANSI_BRIGHT_GREEN "\033[92m"

static volatile sig_atomic_t g_interrupted = 0;

static void sigint_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

static double chat_rss_mib(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
    return (double)ru.ru_maxrss / 1024.0;
}

// ---------------------------------------------------------------------------
// Chat Logger Implementation
// ---------------------------------------------------------------------------

chat_logger_t* chat_logger_init(const char *filepath, const char *model_path, const model_t *model) {
    chat_logger_t *logger = (chat_logger_t*)calloc(1, sizeof(chat_logger_t));
    if (!logger) return NULL;

    char default_path[256];
    if (!filepath || filepath[0] == '\0') {
        time_t now = time(NULL);
        struct tm *tm = localtime(&now);
        snprintf(default_path, sizeof(default_path), "tuff_chat_%04d%02d%02d_%02d%02d%02d.log",
                 tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec);
        filepath = default_path;
    }

    logger->log_filepath = strdup(filepath);
    logger->log_file = fopen(filepath, "a");
    if (!logger->log_file) {
        fprintf(stderr, "%s[Logger Warning] Could not open log file %s for writing.%s\n",
                ANSI_YELLOW, filepath, ANSI_RESET);
        logger->enabled = false;
        return logger;
    }

    logger->enabled = true;
    logger->turn_count = 0;

    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(logger->log_file, "================================================================================\n");
    fprintf(logger->log_file, " Tuff Chat Session Log\n");
    fprintf(logger->log_file, " Session Start: %s\n", time_str);
    fprintf(logger->log_file, " Model Path:    %s\n", model_path ? model_path : "unknown");
    if (model) {
        fprintf(logger->log_file, " Architecture:  Vocab=%u, Dim=%u, Layers=%u (Delta=%u, Attn=%u), Experts=%u/%u\n",
                model->vocab_size, model->embedding_length, model->block_count,
                model->n_delta_layers, model->n_attn_layers, model->expert_used_count, model->expert_count);
    }
    fprintf(logger->log_file, "================================================================================\n\n");
    fflush(logger->log_file);

    return logger;
}

void chat_logger_log_turn(chat_logger_t *logger, int turn_idx, const chat_turn_stats_t *stats) {
    if (!logger || !logger->enabled || !logger->log_file || !stats) return;

    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));

    double hit_rate = (stats->prompt_tokens > 0) ?
                      (100.0 * (double)stats->cached_tokens / (double)stats->prompt_tokens) : 0.0;

    fprintf(logger->log_file, "--------------------------------------------------------------------------------\n");
    fprintf(logger->log_file, "TURN %d  [%s]\n", turn_idx, time_str);
    fprintf(logger->log_file, "User Prompt:\n%s\n\n", stats->user_text ? stats->user_text : "");
    fprintf(logger->log_file, "Assistant Response:\n%s\n\n", stats->assistant_text ? stats->assistant_text : "");
    fprintf(logger->log_file, "Performance Metrics:\n");
    fprintf(logger->log_file, "  - Total Prompt Tokens:  %d\n", stats->prompt_tokens);
    fprintf(logger->log_file, "  - Cached Prefix Tokens: %d (%.1f%% cache hit)\n", stats->cached_tokens, hit_rate);
    fprintf(logger->log_file, "  - New Prefilled Tokens: %d\n", stats->prompt_tokens - stats->cached_tokens);
    fprintf(logger->log_file, "  - Prefill Time / Speed: %.4f s (%.2f tok/s)\n", stats->prefill_time_s, stats->prefill_tok_per_sec);
    fprintf(logger->log_file, "  - Time to First Token:  %.1f ms\n", stats->ttft_ms);
    fprintf(logger->log_file, "  - Generated Tokens:     %d\n", stats->gen_tokens);
    fprintf(logger->log_file, "  - Generation Time:      %.4f s (%.2f tok/s)\n", stats->gen_time_s, stats->gen_tok_per_sec);
    fprintf(logger->log_file, "  - Total Turn Latency:   %.4f s\n", stats->total_time_s);
    fprintf(logger->log_file, "  - Memory Usage:         RAM=%.1f MiB, VRAM=%.1f MiB\n", chat_rss_mib(), gpu_vram_used_mb());
    fprintf(logger->log_file, "--------------------------------------------------------------------------------\n\n");
    fflush(logger->log_file);
}

void chat_logger_free(chat_logger_t *logger) {
    if (!logger) return;
    if (logger->log_file) {
        time_t now = time(NULL);
        char time_str[64];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
        fprintf(logger->log_file, "================================================================================\n");
        fprintf(logger->log_file, " Session Ended: %s (Total turns: %d)\n", time_str, logger->turn_count);
        fprintf(logger->log_file, "================================================================================\n");
        fclose(logger->log_file);
    }
    free(logger->log_filepath);
    free(logger);
}

// ---------------------------------------------------------------------------
// Conversation Message Buffer
// ---------------------------------------------------------------------------

typedef struct {
    char *role;    // "system", "user", "assistant"
    char *content;
} message_entry_t;

typedef struct {
    message_entry_t *entries;
    int count;
    int cap;
} conversation_t;

static conversation_t* conv_create(void) {
    conversation_t *c = (conversation_t*)calloc(1, sizeof(conversation_t));
    c->cap = 32;
    c->entries = (message_entry_t*)calloc(c->cap, sizeof(message_entry_t));
    return c;
}

static void conv_add(conversation_t *c, const char *role, const char *content) {
    if (c->count >= c->cap) {
        c->cap *= 2;
        c->entries = (message_entry_t*)realloc(c->entries, (size_t)c->cap * sizeof(message_entry_t));
    }
    c->entries[c->count].role = strdup(role);
    c->entries[c->count].content = strdup(content);
    c->count++;
}

static void conv_clear_dialogue(conversation_t *c) {
    if (c->count <= 0) return;
    int start = (strcmp(c->entries[0].role, "system") == 0) ? 1 : 0;
    for (int i = start; i < c->count; ++i) {
        free(c->entries[i].role);
        free(c->entries[i].content);
    }
    c->count = start;
}

static void conv_free(conversation_t *c) {
    if (!c) return;
    for (int i = 0; i < c->count; ++i) {
        free(c->entries[i].role);
        free(c->entries[i].content);
    }
    free(c->entries);
    free(c);
}

// ---------------------------------------------------------------------------
// TUI Display Utilities
// ---------------------------------------------------------------------------

static void print_banner(const model_t *model, const chat_config_t *cfg, const char *log_path) {
    printf("\n");
    printf("%s%s╔════════════════════════════════════════════════════════════════════════════════╗%s\n", ANSI_BOLD, ANSI_CYAN, ANSI_RESET);
    printf("%s%s║                       🐦 TUFF INTERACTIVE CHAT TUI                      ║%s\n", ANSI_BOLD, ANSI_CYAN, ANSI_RESET);
    printf("%s%s╚════════════════════════════════════════════════════════════════════════════════╝%s\n", ANSI_BOLD, ANSI_CYAN, ANSI_RESET);
    printf("%s• Model:%s       %s\n", ANSI_DIM, ANSI_RESET, cfg->model_path ? cfg->model_path : "default");
    printf("%s• Specs:%s       Vocab: %u | Layers: %u (Delta: %u, Attn: %u) | Experts: %u/%u\n",
           ANSI_DIM, ANSI_RESET, model->vocab_size, model->block_count,
           model->n_delta_layers, model->n_attn_layers, model->expert_used_count, model->expert_count);
    printf("%s• Memory:%s      RSS: %.1f MiB | VRAM: %.1f MiB\n",
           ANSI_DIM, ANSI_RESET, chat_rss_mib(), gpu_vram_used_mb());
    printf("%s• Cache:%s       Prefix Caching: %sENABLED (Hybrid Delta-Net/KV)%s\n",
           ANSI_DIM, ANSI_RESET, ANSI_GREEN, ANSI_RESET);
    if (log_path && log_path[0] != '\0') {
        printf("%s• Logging:%s     %s%s%s\n", ANSI_DIM, ANSI_RESET, ANSI_YELLOW, log_path, ANSI_RESET);
    }
    printf("%s• Help:%s        Type %s/help%s to view commands, %s/clear%s to reset chat, %s/exit%s to quit.\n",
           ANSI_DIM, ANSI_RESET, ANSI_BOLD, ANSI_RESET, ANSI_BOLD, ANSI_RESET, ANSI_BOLD, ANSI_RESET);
    printf("%s────────────────────────────────────────────────────────────────────────────────%s\n\n", ANSI_DIM, ANSI_RESET);
}

static void print_help(void) {
    printf("\n%s%s╭─ Available Slash Commands ───────────────────────────────────────────────────╮%s\n", ANSI_BOLD, ANSI_YELLOW, ANSI_RESET);
    printf("│ %s/help%s             Show this help menu                                     │\n", ANSI_BOLD, ANSI_RESET);
    printf("│ %s/clear%s, %s/reset%s     Reset conversation history & prefix cache               │\n", ANSI_BOLD, ANSI_RESET, ANSI_BOLD, ANSI_RESET);
    printf("│ %s/stats%s            Display current context size, cache stats, memory usage │\n", ANSI_BOLD, ANSI_RESET);
    printf("│ %s/system <prompt>%s  View or update system prompt                            │\n", ANSI_BOLD, ANSI_RESET);
    printf("│ %s/save <filepath>%s  Save entire markdown conversation transcript to a file  │\n", ANSI_BOLD, ANSI_RESET);
    printf("│ %s/log <on|off|path>%s Toggle or set log file                                  │\n", ANSI_BOLD, ANSI_RESET);
    printf("│ %s/exit%s, %s/quit%s      Exit interactive chat                                   │\n", ANSI_BOLD, ANSI_RESET, ANSI_BOLD, ANSI_RESET);
    printf("%s╰───────────────────────────────────────────────────────────────────────────────╯%s\n\n", ANSI_YELLOW, ANSI_RESET);
}

static void print_stats(const engine_prefix_cache_t *cache, int max_seq, int total_turns, int total_gen_tokens, double total_gen_time) {
    double avg_speed = (total_gen_time > 0) ? ((double)total_gen_tokens / total_gen_time) : 0.0;
    double ctx_pct = (max_seq > 0) ? (100.0 * (double)cache->n_tokens / (double)max_seq) : 0.0;

    printf("\n%s%s╭─ Engine & Session Statistics ─────────────────────────────────────────────────╮%s\n", ANSI_BOLD, ANSI_CYAN, ANSI_RESET);
    printf("│ • Current Context:       %s%4d / %d tokens (%.1f%%)%s                             │\n",
           ANSI_BOLD, cache->n_tokens, max_seq, ctx_pct, ANSI_RESET);
    printf("│ • Prefix Checkpoints:    %s%d saved state snapshot(s)%s                          │\n",
           ANSI_BOLD, cache->n_checkpoints, ANSI_RESET);
    printf("│ • Total Turns Completed: %s%d%s                                                  │\n",
           ANSI_BOLD, total_turns, ANSI_RESET);
    printf("│ • Total Tokens Generated:%s%d%s                                                  │\n",
           ANSI_BOLD, total_gen_tokens, ANSI_RESET);
    printf("│ • Average Generation:    %s%.2f tok/s%s                                         │\n",
           ANSI_BOLD, avg_speed, ANSI_RESET);
    printf("│ • Host RAM (RSS):        %s%.1f MiB%s                                         │\n",
           ANSI_BOLD, chat_rss_mib(), ANSI_RESET);
    printf("│ • GPU VRAM Used:         %s%.1f MiB%s                                         │\n",
           ANSI_BOLD, gpu_vram_used_mb(), ANSI_RESET);
    printf("%s╰───────────────────────────────────────────────────────────────────────────────╯%s\n\n", ANSI_CYAN, ANSI_RESET);
}

static void print_turn_footer(const chat_turn_stats_t *s, int ctx_used, int max_ctx) {
    double hit_rate = (s->prompt_tokens > 0) ?
                      (100.0 * (double)s->cached_tokens / (double)s->prompt_tokens) : 0.0;
    double ctx_pct = (max_ctx > 0) ? (100.0 * (double)ctx_used / (double)max_ctx) : 0.0;

    printf("\n%s%s╭─ ⚡ Response Stats ────────────────────────────────────────────────────────────╮%s\n", ANSI_DIM, ANSI_CYAN, ANSI_RESET);
    printf("%s│ Speed: %s%s%.2f tok/s%s%s | Generated: %s%d tok%s%s in %.2fs | TTFT: %s%.1f ms%s%s             │\n",
           ANSI_DIM, ANSI_BOLD, ANSI_BRIGHT_GREEN, s->gen_tok_per_sec, ANSI_RESET, ANSI_DIM,
           ANSI_BOLD, s->gen_tokens, ANSI_RESET, ANSI_DIM, s->gen_time_s,
           ANSI_BOLD, s->ttft_ms, ANSI_RESET, ANSI_DIM);
    printf("%s│ Prefill: %d tok (%s%.1f%% cached%s%s: %d cached, %d new @ %.1f tok/s in %.3fs) │\n",
           ANSI_DIM, s->prompt_tokens, ANSI_BOLD, hit_rate, ANSI_RESET, ANSI_DIM,
           s->cached_tokens, s->prompt_tokens - s->cached_tokens, s->prefill_tok_per_sec, s->prefill_time_s);
    printf("%s│ Context: %d/%d tokens (%.1f%%) | RAM: %.0f MiB | VRAM: %.0f MiB                │\n",
           ANSI_DIM, ctx_used, max_ctx, ctx_pct, chat_rss_mib(), gpu_vram_used_mb());
    printf("%s╰───────────────────────────────────────────────────────────────────────────────╯%s\n\n", ANSI_DIM, ANSI_RESET);
}

static void save_transcript_markdown(const conversation_t *c, const char *filepath) {
    FILE *f = fopen(filepath, "w");
    if (!f) {
        printf("%s[Error] Could not open %s for saving transcript.%s\n", ANSI_RED, filepath, ANSI_RESET);
        return;
    }
    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(f, "# Tuff Chat Transcript\n\n");
    fprintf(f, "*Exported on: %s*\n\n---\n\n", time_str);

    for (int i = 0; i < c->count; ++i) {
        if (strcmp(c->entries[i].role, "system") == 0) {
            fprintf(f, "> **System Prompt:**\n> %s\n\n", c->entries[i].content);
        } else if (strcmp(c->entries[i].role, "user") == 0) {
            fprintf(f, "### 👤 User:\n%s\n\n", c->entries[i].content);
        } else if (strcmp(c->entries[i].role, "assistant") == 0) {
            fprintf(f, "### 🤖 Assistant:\n%s\n\n---\n\n", c->entries[i].content);
        }
    }
    fclose(f);
    printf("%s[Success] Transcript saved to %s%s\n\n", ANSI_GREEN, filepath, ANSI_RESET);
}

// ---------------------------------------------------------------------------
// Stop Token Detection
// ---------------------------------------------------------------------------

static bool is_stop_token(const tokenizer_t *tok, int32_t token_id) {
    if (!tok || token_id < 0 || token_id >= (int32_t)tok->vocab_size) return true;
    if (token_id == tok->eos_id) return true;

    const char *piece = tokenizer_decode(tok, token_id);
    if (!piece) return true;

    if (strcmp(piece, "<|im_end|>") == 0 ||
        strcmp(piece, "<|endoftext|>") == 0 ||
        strcmp(piece, "<|im_start|>") == 0) {
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// Interactive Chat Loop
// ---------------------------------------------------------------------------

int run_interactive_chat(model_t *model, tokenizer_t *tok, const chat_config_t *config) {
    if (!model || !tok) return 1;

    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    int max_seq = (config && config->max_context_len > 0) ? config->max_context_len : 4096;
    int max_gen = (config && config->max_gen_tokens > 0) ? config->max_gen_tokens : 1024;
    const char *sys_prompt = (config && config->system_prompt && config->system_prompt[0]) ?
                             config->system_prompt : "You are a helpful, respectful and honest AI assistant.";

    chat_logger_t *logger = chat_logger_init(config ? config->log_filepath : NULL,
                                             config ? config->model_path : NULL, model);

    print_banner(model, config, logger ? logger->log_filepath : NULL);

    engine_t *eng = engine_init(model, max_seq);
    if (config) {
        engine_set_sampler(eng, &config->sampler);
    }
    engine_prefix_cache_t *cache = engine_prefix_cache_create(max_seq);

    conversation_t *conv = conv_create();
    conv_add(conv, "system", sys_prompt);

    int turn_index = 0;
    int total_gen_tokens = 0;
    double total_gen_time = 0.0;

    int32_t *history_tokens = (int32_t*)malloc((size_t)max_seq * sizeof(int32_t));
    int history_token_count = 0;

    // Tokenize initial system prompt
    char sys_block[4096];
    snprintf(sys_block, sizeof(sys_block), "<|im_start|>system\n%s<|im_end|>\n", sys_prompt);
    history_token_count = tokenizer_tokenize(tok, sys_block, history_tokens, max_seq);

    int32_t *user_turn_tokens = (int32_t*)malloc((size_t)max_seq * sizeof(int32_t));
    char *user_input_buf = (char*)malloc(16384);
    char *resp_buf = (char*)malloc(65536);

    while (1) {
        g_interrupted = 0;
        printf("%s%s👤 You > %s", ANSI_BOLD, ANSI_BRIGHT_GREEN, ANSI_RESET);
        fflush(stdout);

        if (!fgets(user_input_buf, 16384, stdin)) {
            printf("\n%sExiting chat. Goodbye!%s\n", ANSI_DIM, ANSI_RESET);
            break;
        }

        size_t ulen = strlen(user_input_buf);
        while (ulen > 0 && (user_input_buf[ulen - 1] == '\n' || user_input_buf[ulen - 1] == '\r' || user_input_buf[ulen - 1] == ' ')) {
            user_input_buf[--ulen] = '\0';
        }

        if (ulen == 0) continue;

        // Check for Slash Commands
        if (user_input_buf[0] == '/') {
            if (strcmp(user_input_buf, "/exit") == 0 || strcmp(user_input_buf, "/quit") == 0) {
                printf("%sExiting chat. Goodbye!%s\n", ANSI_DIM, ANSI_RESET);
                break;
            } else if (strcmp(user_input_buf, "/help") == 0) {
                print_help();
                continue;
            } else if (strcmp(user_input_buf, "/clear") == 0 || strcmp(user_input_buf, "/reset") == 0) {
                conv_clear_dialogue(conv);
                engine_reset_context(eng);
                engine_prefix_cache_reset(cache);
                snprintf(sys_block, sizeof(sys_block), "<|im_start|>system\n%s<|im_end|>\n", conv->entries[0].content);
                history_token_count = tokenizer_tokenize(tok, sys_block, history_tokens, max_seq);
                printf("%s%s[Chat reset] Conversation history and prefix cache cleared.%s\n\n", ANSI_BOLD, ANSI_YELLOW, ANSI_RESET);
                continue;
            } else if (strcmp(user_input_buf, "/stats") == 0) {
                print_stats(cache, max_seq, turn_index, total_gen_tokens, total_gen_time);
                continue;
            } else if (strncmp(user_input_buf, "/system", 7) == 0) {
                const char *p = user_input_buf + 7;
                while (*p == ' ') p++;
                if (*p == '\0') {
                    printf("%sSystem Prompt:%s %s\n\n", ANSI_BOLD, ANSI_RESET, conv->entries[0].content);
                } else {
                    free(conv->entries[0].content);
                    conv->entries[0].content = strdup(p);
                    conv_clear_dialogue(conv);
                    engine_reset_context(eng);
                    engine_prefix_cache_reset(cache);
                    snprintf(sys_block, sizeof(sys_block), "<|im_start|>system\n%s<|im_end|>\n", p);
                    history_token_count = tokenizer_tokenize(tok, sys_block, history_tokens, max_seq);
                    printf("%s%s[System Prompt Updated & Context Reset]%s %s\n\n", ANSI_BOLD, ANSI_YELLOW, ANSI_RESET, p);
                }
                continue;
            } else if (strncmp(user_input_buf, "/save", 5) == 0) {
                const char *path = user_input_buf + 5;
                while (*path == ' ') path++;
                if (*path == '\0') path = "chat_transcript.md";
                save_transcript_markdown(conv, path);
                continue;
            } else if (strncmp(user_input_buf, "/log", 4) == 0) {
                const char *arg = user_input_buf + 4;
                while (*arg == ' ') arg++;
                if (strcmp(arg, "off") == 0) {
                    if (logger) logger->enabled = false;
                    printf("%s[Logger Disabled]%s\n\n", ANSI_YELLOW, ANSI_RESET);
                } else if (strcmp(arg, "on") == 0) {
                    if (logger) logger->enabled = true;
                    printf("%s[Logger Enabled] Logging to %s%s\n\n", ANSI_YELLOW, logger ? logger->log_filepath : "none", ANSI_RESET);
                } else if (*arg != '\0') {
                    if (logger) chat_logger_free(logger);
                    logger = chat_logger_init(arg, config ? config->model_path : NULL, model);
                    printf("%s[Logger Switched] Logging to %s%s\n\n", ANSI_YELLOW, arg, ANSI_RESET);
                } else {
                    printf("%s[Logger Status] %s (File: %s)%s\n\n", ANSI_YELLOW,
                           (logger && logger->enabled) ? "ENABLED" : "DISABLED",
                           logger ? logger->log_filepath : "none", ANSI_RESET);
                }
                continue;
            } else {
                printf("%sUnknown command '%s'. Type /help for available commands.%s\n\n", ANSI_YELLOW, user_input_buf, ANSI_RESET);
                continue;
            }
        }

        // Format user turn
        char user_turn_text[18000];
        snprintf(user_turn_text, sizeof(user_turn_text), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_input_buf);
        int n_user_tok = tokenizer_tokenize(tok, user_turn_text, user_turn_tokens, max_seq);

        if (n_user_tok <= 0) {
            printf("%s[Error] Failed to tokenize user input.%s\n\n", ANSI_RED, ANSI_RESET);
            continue;
        }

        // Check context overflow
        if (history_token_count + n_user_tok >= max_seq - 64) {
            printf("%s[Warning] Context limit reached (%d/%d tokens). Auto-clearing older history...%s\n",
                   ANSI_YELLOW, history_token_count + n_user_tok, max_seq, ANSI_RESET);
            conv_clear_dialogue(conv);
            engine_reset_context(eng);
            engine_prefix_cache_reset(cache);
            snprintf(sys_block, sizeof(sys_block), "<|im_start|>system\n%s<|im_end|>\n", conv->entries[0].content);
            history_token_count = tokenizer_tokenize(tok, sys_block, history_tokens, max_seq);
        }

        // Append user turn tokens to prompt token buffer
        int n_prompt = history_token_count + n_user_tok;
        int32_t *full_prompt_tokens = (int32_t*)malloc((size_t)n_prompt * sizeof(int32_t));
        memcpy(full_prompt_tokens, history_tokens, (size_t)history_token_count * sizeof(int32_t));
        memcpy(full_prompt_tokens + history_token_count, user_turn_tokens, (size_t)n_user_tok * sizeof(int32_t));

        // Add user message to conversation history
        conv_add(conv, "user", user_input_buf);
        turn_index++;
        if (logger) logger->turn_count = turn_index;

        // Prefill with prefix caching
        double t_turn_start = omp_get_wtime();
        prefill_result_t pf_res = engine_prefill_with_cache(eng, cache, full_prompt_tokens, n_prompt);
        double t_first_tok = omp_get_wtime();
        double ttft_ms = (t_first_tok - t_turn_start) * 1000.0;

        printf("\n%s%s🤖 Tuff > %s", ANSI_BOLD, ANSI_BRIGHT_CYAN, ANSI_RESET);
        fflush(stdout);

        // Update history token buffer with prompt tokens
        memcpy(history_tokens, full_prompt_tokens, (size_t)n_prompt * sizeof(int32_t));
        history_token_count = n_prompt;
        free(full_prompt_tokens);

        // Autoregressive generation
        int gen_pos = n_prompt;
        int32_t curr_token = pf_res.first_generated_token;
        int n_generated = 0;
        resp_buf[0] = '\0';
        size_t resp_len = 0;

        double t_gen_start = omp_get_wtime();

        while (n_generated < max_gen && gen_pos < max_seq) {
            if (g_interrupted) {
                printf(" %s[Interrupted by user]%s", ANSI_YELLOW, ANSI_RESET);
                break;
            }

            if (is_stop_token(tok, curr_token)) {
                break;
            }

            const char *piece = tokenizer_decode(tok, curr_token);
            if (piece) {
                fputs(piece, stdout);
                fflush(stdout);

                size_t plen = strlen(piece);
                if (resp_len + plen < 65535) {
                    memcpy(resp_buf + resp_len, piece, plen);
                    resp_len += plen;
                    resp_buf[resp_len] = '\0';
                }
            }

            // Append generated token to prefix cache sequence
            if (gen_pos >= cache->cap_tokens) {
                cache->cap_tokens = gen_pos + 1024;
                cache->tokens = (int32_t*)realloc(cache->tokens, (size_t)cache->cap_tokens * sizeof(int32_t));
            }
            cache->tokens[gen_pos] = curr_token;
            cache->n_tokens = gen_pos + 1;

            if (history_token_count < max_seq) {
                history_tokens[history_token_count++] = curr_token;
            }

            int32_t next_tok = engine_forward_token(eng, curr_token, gen_pos++);
            curr_token = next_tok;
            n_generated++;
        }

        double t_gen_end = omp_get_wtime();
        double gen_elapsed = t_gen_end - t_gen_start;
        double gen_tok_per_sec = (n_generated > 0 && gen_elapsed > 0) ?
                                 ((double)n_generated / gen_elapsed) : 0.0;

        printf("\n");

        // Append <|im_end|>\n to history tokens so next turn starts with clean ChatML
        int32_t im_end_tokens[16];
        int n_im_end = tokenizer_tokenize(tok, "<|im_end|>\n", im_end_tokens, 16);
        for (int i = 0; i < n_im_end && history_token_count < max_seq; ++i) {
            // Forward im_end through engine to keep state synchronized at turn boundary
            if (gen_pos < max_seq) {
                engine_forward_token(eng, im_end_tokens[i], gen_pos++);
                if (gen_pos >= cache->cap_tokens) {
                    cache->cap_tokens = gen_pos + 1024;
                    cache->tokens = (int32_t*)realloc(cache->tokens, (size_t)cache->cap_tokens * sizeof(int32_t));
                }
                cache->tokens[gen_pos - 1] = im_end_tokens[i];
                cache->n_tokens = gen_pos;
            }
            history_tokens[history_token_count++] = im_end_tokens[i];
        }

        // Save turn state checkpoint for fast prefix continuation on future turns
        engine_prefix_cache_save_checkpoint(cache, eng);

        // Add assistant response to conversation history
        conv_add(conv, "assistant", resp_buf);

        // Performance metrics
        total_gen_tokens += n_generated;
        total_gen_time += gen_elapsed;

        chat_turn_stats_t stats;
        stats.user_text = user_input_buf;
        stats.assistant_text = resp_buf;
        stats.prompt_tokens = n_prompt;
        stats.cached_tokens = pf_res.cached_tokens;
        stats.gen_tokens = n_generated;
        stats.prefill_time_s = pf_res.prefill_time_s;
        stats.prefill_tok_per_sec = pf_res.prefill_tok_per_sec;
        stats.gen_time_s = gen_elapsed;
        stats.gen_tok_per_sec = gen_tok_per_sec;
        stats.ttft_ms = ttft_ms;
        stats.total_time_s = t_gen_end - t_turn_start;

        // Print stats footer
        print_turn_footer(&stats, cache->n_tokens, max_seq);

        // Log turn
        if (logger) {
            chat_logger_log_turn(logger, turn_index, &stats);
        }
    }

    free(user_input_buf);
    free(resp_buf);
    free(history_tokens);
    free(user_turn_tokens);
    conv_free(conv);
    engine_prefix_cache_free(cache);
    engine_free(eng);
    if (logger) chat_logger_free(logger);

    return 0;
}
