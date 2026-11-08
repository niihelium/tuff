// Backend B: Qwen3.5 dense request handling, continuous-batching scheduler and slots.
//
// One GPU thread owns every engine call.  At each launch boundary it admits queued
// requests into idle slots (longest exact-prefix reuse first, then LRU), issues at most one
// chunked-prefill launch (<= 16 tokens, the engine's step limit) and one batched decode
// launch covering all decoding slots, then streams whatever became available.  A slot is
// freed on stop, EOS, context/max-token exhaustion or client disconnect.

#define _GNU_SOURCE

#include "q35_server.h"
#include "q35_template.h"
#include "q35_toolparse.h"
#include "q35_sample.h"
#include "qwen35.h"

#include <errno.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int send_all_fd(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    size_t sent = 0;
    while (sent < len) {
        ssize_t r = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        sent += (size_t)r;
    }
    return 0;
}

static void jesc(q35_buf *b, const char *s) {
    q35_buf_append(b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
        case '"': q35_buf_append(b, "\\\"", 2); break;
        case '\\': q35_buf_append(b, "\\\\", 2); break;
        case '\n': q35_buf_append(b, "\\n", 2); break;
        case '\r': q35_buf_append(b, "\\r", 2); break;
        case '\t': q35_buf_append(b, "\\t", 2); break;
        case '\b': q35_buf_append(b, "\\b", 2); break;
        case '\f': q35_buf_append(b, "\\f", 2); break;
        default:
            if (*p < 0x20) {
                char t[8];
                snprintf(t, sizeof(t), "\\u%04x", *p);
                q35_buf_append(b, t, 6);
            } else {
                q35_buf_append(b, (const char *)p, 1);
            }
        }
    }
    q35_buf_append(b, "\"", 1);
}

static void respond_raw(int fd, int status, const char *status_text, const char *ctype,
                        const void *body, size_t body_len) {
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Connection: close\r\n\r\n", status, status_text, ctype, body_len);
    send_all_fd(fd, hdr, (size_t)n);
    if (body_len) send_all_fd(fd, body, body_len);
}

static void respond_json_q(int fd, int status, const char *status_text, const q35_buf *b) {
    respond_raw(fd, status, status_text, "application/json", b->data ? b->data : "", b->len);
}

static void respond_error(int fd, int status, const char *msg) {
    q35_buf b;
    q35_buf_init(&b);
    q35_buf_append_str(&b, "{\"error\":{\"message\":");
    jesc(&b, msg);
    q35_buf_append_str(&b, ",\"type\":\"invalid_request_error\"}}");
    respond_json_q(fd, status, status == 400 ? "Bad Request" : "Internal Server Error", &b);
    q35_buf_free(&b);
}

static void respond_stream_headers(int fd, const char *ignored) {
    q35_buf h;
    q35_buf_init(&h);
    q35_buf_append_str(&h,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n");
    if (ignored && ignored[0]) q35_buf_appendf(&h, "X-Tuf-Ignored: %s\r\n", ignored);
    q35_buf_append_str(&h, "Connection: close\r\n\r\n");
    send_all_fd(fd, h.data, h.len);
    q35_buf_free(&h);
}

// Forward UTF-8 validation: length of the valid prefix (stops at any malformed byte).
static size_t utf8_valid_prefix(const unsigned char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = s[i];
        size_t need;
        if (c < 0x80) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else return i;
        if (i + need > n) return i;
        for (size_t k = 1; k < need; k++)
            if ((s[i + k] & 0xC0) != 0x80) return i;
        i += need;
    }
    return n;
}

// A token piece may be a partial UTF-8 sequence (byte-level BPE).  llama-server
// validates/truncates the `token` and `bytes` fields before emitting JSON; do the same.
static const char *valid_token_text(const char *s, char *tmp, size_t cap) {
    if (!s) return "";
    size_t n = strlen(s);
    size_t v = utf8_valid_prefix((const unsigned char *)s, n);
    if (v == n) return s;
    if (v >= cap) v = cap - 1;
    memcpy(tmp, s, v);
    tmp[v] = '\0';
    return tmp;
}

// ---------------------------------------------------------------------------
// Jobs and slots
// ---------------------------------------------------------------------------

#define Q35_MAX_STOP 8
#define Q35_PREFILL_CHUNK 16

enum { FIN_STOP = 0, FIN_LENGTH = 1, FIN_TOOL_CALLS = 2, FIN_CANCELLED = 3 };

typedef struct q35_job {
    int32_t *tokens;
    int      n_tokens;
    int      max_tokens;
    int      chat, stream, ignore_eos;
    q35_sampling samp;
    char    *stops[Q35_MAX_STOP];
    int      n_stops;
    int      thinking, extract_reasoning, parse_tools;
    qj_val  *root;
    const qj_val *tools;
    char    *model;
    char    *id;
    const char *object;
    int64_t  created;
    int      fd;

    int      cancelled, done, rc, finish;
    int      slot;
    int      n_prompt, n_prompt_cached, n_gen;
    q35_buf  raw;
    q35_out  parsed;
    size_t   emitted_reasoning, emitted_content;
    int      emitted_calls;
    int      initial_sent;
    q35_logprobs *lps;
    int32_t *lp_ids;
    int      lp_cap;
    int      n_lp;
    int      t_start_set;
    double   t_start, t_first, t_end;

    q35_buf  response;       // built by the GPU thread for non-streaming requests
    char    *ignored;        // X-Tuf-Ignored value (comma separated)

    struct q35_job *next;
} q35_job;

typedef struct {
    int          busy;
    q35_job     *job;
    int32_t     *seq;        // tokens whose state has been fed to the engine
    int          seq_len;
    int          seq_cap;
    int          phase;      // 0 idle, 1 prefill, 2 decode
    int          pending;    // token to feed at the next decode launch
    int          prefilled;  // prompt tokens already fed in this job
    int          n_prompt;
    int          gen_count;
    uint64_t     rng;
    int          need_logits;
    unsigned long long last_used;
} q35_slot;

struct q35_server {
    q35_server_cfg cfg;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    pthread_t       gpu;
    int             stop;
    q35_job        *q_head, *q_tail;
    q35_slot       *slots;
    int            *d_slots;
    int            *d_pos;
    int32_t        *d_tok;
    float          *logits;      // n_slots * V
    float          *raw_copy;    // V, for logprobs
    qj_val         *default_kwargs;
    unsigned long long use_counter;
    uint64_t         job_counter;
};

// ---------------------------------------------------------------------------
// JSON request helpers
// ---------------------------------------------------------------------------

static void free_job(q35_job *j) {
    if (!j) return;
    free(j->tokens);
    for (int i = 0; i < j->n_stops; i++) free(j->stops[i]);
    qj_free(j->root);
    free(j->model);
    free(j->id);
    q35_buf_free(&j->raw);
    q35_out_free(&j->parsed);
    q35_buf_free(&j->response);
    free(j->lps);
    free(j->lp_ids);
    free(j->ignored);
    free(j);
}

static void add_ignored(q35_job *j, const char *field) {
    q35_buf b;
    q35_buf_init(&b);
    if (j->ignored) q35_buf_appendf(&b, "%s,", j->ignored);
    q35_buf_append_str(&b, field);
    free(j->ignored);
    j->ignored = b.data;
}

// Fields accepted by llama-server but not implemented here.
static const char *const k_ignored_fields[] = {
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers", "xtc_probability", "xtc_threshold", "typical_p",
    "mirostat", "mirostat_tau", "mirostat_eta", "top_n_sigma", "adaptive_target",
    "adaptive_decay", "dynatemp_range", "dynatemp_exponent", "min_keep", "samplers",
    "logit_bias", "n_probs", "timings_per_token", "cache_prompt", "slot_id",
    NULL
};

static int parse_sampling(const qj_val *root, q35_sampling *p, int is_completion) {
    q35_sampling_defaults(p);
    const qj_val *v;
    if ((v = qj_get(root, "temperature"))) p->temp = (float)qj_num(v, p->temp);
    if ((v = qj_get(root, "top_k"))) p->top_k = (int32_t)qj_int(v, p->top_k);
    if ((v = qj_get(root, "top_p"))) p->top_p = (float)qj_num(v, p->top_p);
    if ((v = qj_get(root, "min_p"))) p->min_p = (float)qj_num(v, p->min_p);
    if ((v = qj_get(root, "repeat_penalty"))) p->repeat_penalty = (float)qj_num(v, p->repeat_penalty);
    if ((v = qj_get(root, "frequency_penalty"))) p->freq_penalty = (float)qj_num(v, 0);
    if ((v = qj_get(root, "presence_penalty"))) p->presence_penalty = (float)qj_num(v, 0);
    if ((v = qj_get(root, "repeat_last_n"))) p->penalty_last_n = (int32_t)qj_int(v, p->penalty_last_n);
    if ((v = qj_get(root, "seed"))) p->seed = (uint64_t)qj_int(v, 0);
    if (is_completion) {
        if ((v = qj_get(root, "logprobs"))) {
            if (v->type == QJ_NUM) {
                p->logprobs = v->num != 0;
                p->top_logprobs = (int32_t)qj_int(v, 0);
            } else if (v->type == QJ_BOOL) {
                p->logprobs = v->boolean;
                p->top_logprobs = p->logprobs ? 20 : 0;
            }
        }
    } else {
        if ((v = qj_get(root, "logprobs"))) p->logprobs = qj_bool(v, 0);
        if ((v = qj_get(root, "top_logprobs"))) p->top_logprobs = (int32_t)qj_int(v, 0);
        if (!p->logprobs) p->top_logprobs = 0;
        else if (p->top_logprobs <= 0) p->top_logprobs = 20;
    }
    if (p->top_logprobs > 20) p->top_logprobs = 20;
    if (p->top_k < 0) p->top_k = 0;
    if (p->penalty_last_n < 0) p->penalty_last_n = 0;
    if (p->min_p < 0) p->min_p = 0;
    if (p->top_p <= 0 || p->top_p > 1) p->top_p = 1.0f;
    return 0;
}

static void resolve_template_opts(q35_server *s, const qj_val *root,
                                  int *thinking, int *add_gen) {
    int th = 0, ag = 1;
    q35_template_kwargs(s->default_kwargs, &th, &ag, 0, 1);
    q35_template_kwargs(qj_get(root, "chat_template_kwargs"), &th, &ag, th, ag);
    const qj_val *agv = qj_get(root, "add_generation_prompt");
    if (agv) ag = qj_bool(agv, ag);
    const char *eff = qj_str(qj_get(root, "reasoning_effort"));
    if (eff && strcmp(eff, "none") == 0) th = 0;
    *thinking = th;
    *add_gen = ag;
}

static int parse_request_common(q35_server *s, const qj_val *root, q35_job *j,
                                int chat, char *err, size_t errlen) {
    const qj_val *v;
    j->max_tokens = -1;
    if ((v = qj_get(root, "max_tokens")) || (v = qj_get(root, "max_completion_tokens")))
        j->max_tokens = (int)qj_int(v, -1);
    j->stream = qj_bool(qj_get(root, "stream"), 0);
    j->ignore_eos = qj_bool(qj_get(root, "ignore_eos"), 0);
    j->chat = chat;
    if ((v = qj_get(root, "model")) && v->type == QJ_STR) j->model = strdup(v->str);
    j->created = (int64_t)time(NULL);

    if ((v = qj_get(root, "n")) && qj_int(v, 1) != 1) {
        snprintf(err, errlen, "n must be 1");
        return -1;
    }
    if (qj_get(root, "grammar") || qj_get(root, "json_schema")) {
        snprintf(err, errlen, "grammar and json_schema are not supported by backend B");
        return -1;
    }
    if ((v = qj_get(root, "response_format")) && v->type == QJ_OBJ) {
        const char *t = qj_str(qj_get(v, "type"));
        if (t && (strcmp(t, "json_schema") == 0 || strcmp(t, "json_object") == 0)) {
            snprintf(err, errlen, "response_format %s is not supported by backend B", t);
            return -1;
        }
    }
    for (int i = 0; k_ignored_fields[i]; i++)
        if (qj_get(root, k_ignored_fields[i])) add_ignored(j, k_ignored_fields[i]);

    parse_sampling(root, &j->samp, !chat);

    // stop: string or array of strings
    if ((v = qj_get(root, "stop"))) {
        if (v->type == QJ_STR) {
            j->stops[0] = strdup(v->str);
            j->n_stops = 1;
        } else if (v->type == QJ_ARR) {
            for (int i = 0; i < v->n && j->n_stops < Q35_MAX_STOP; i++) {
                const char *st = qj_str(v->items[i]);
                if (st && st[0]) j->stops[j->n_stops++] = strdup(st);
            }
        }
    }

    // chat_template_kwargs: server default, then request override
    int thinking = 0, add_gen = 1;
    resolve_template_opts(s, root, &thinking, &add_gen);
    j->thinking = thinking;
    j->extract_reasoning = s->cfg.reasoning_format != 0;

    j->tools = qj_get(root, "tools");
    j->parse_tools = j->tools && j->tools->type == QJ_ARR && j->tools->n > 0;
    if ((v = qj_get(root, "tool_choice"))) {
        if (v->type == QJ_STR && strcmp(v->str, "none") == 0) j->parse_tools = 0;
    }
    if (!j->parse_tools) j->tools = NULL;
    return 0;
}

static char *render_prompt(q35_server *s, const qj_val *root, int chat, char *err, size_t errlen) {
    const qj_val *messages = qj_get(root, "messages");
    q35_msgs msgs;
    if (chat) {
        if (!messages || messages->type != QJ_ARR || messages->n == 0) {
            snprintf(err, errlen, "expected a non-empty \"messages\" array");
            return NULL;
        }
        if (q35_messages_from_json(messages, &msgs, err, errlen) != 0) return NULL;
    } else {
        const char *prompt = qj_str(qj_get(root, "prompt"));
        if (!prompt) {
            snprintf(err, errlen, "expected a string \"prompt\" field");
            return NULL;
        }
        memset(&msgs, 0, sizeof(msgs));
        msgs.n = 1;
        msgs.items = (q35_msg *)calloc(1, sizeof(q35_msg));
        msgs.items[0].role = strdup("user");
        msgs.items[0].content = strdup(prompt);
    }
    int thinking = 0, add_gen = 1;
    resolve_template_opts(s, root, &thinking, &add_gen);
    if (chat) {
        const qj_val *tools = qj_get(root, "tools");
        char *out = q35_render_chat(&msgs, tools && tools->type == QJ_ARR ? tools : NULL,
                                    add_gen, thinking, err, errlen);
        q35_messages_free(&msgs);
        return out;
    }
    // /v1/completions: the prompt is used verbatim.
    char *out = msgs.items[0].content;
    msgs.items[0].content = NULL;
    q35_messages_free(&msgs);
    return out;
}

// ---------------------------------------------------------------------------
// Response building
// ---------------------------------------------------------------------------

static int parse_now(q35_server *s, q35_job *j) {
    (void)s;
    q35_out_free(&j->parsed);
    q35_parse_output(j->raw.data ? j->raw.data : "", j->raw.len, j->tools,
                     j->thinking, j->extract_reasoning, &j->parsed);
    if (j->parsed.content) {
        size_t n = strlen(j->parsed.content);
        j->parsed.content[utf8_valid_prefix((const unsigned char *)j->parsed.content, n)] = '\0';
    }
    if (j->parsed.reasoning) {
        size_t n = strlen(j->parsed.reasoning);
        j->parsed.reasoning[utf8_valid_prefix((const unsigned char *)j->parsed.reasoning, n)] = '\0';
    }
    return 0;
}

static const char *finish_name(int fin) {
    if (fin == FIN_LENGTH) return "length";
    if (fin == FIN_TOOL_CALLS) return "tool_calls";
    return "stop";
}

static void append_bytes(q35_buf *b, const char *s) {
    q35_buf_append(b, "[", 1);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++)
        q35_buf_appendf(b, "%s%u", p == (const unsigned char *)(s ? s : "") ? "" : ",", *p);
    q35_buf_append(b, "]", 1);
}

static void append_logprobs_obj(q35_server *s, q35_buf *b, q35_job *j, int only_index) {
    int start = only_index >= 0 ? only_index : 0;
    int end = only_index >= 0 ? only_index + 1 : j->n_gen;
    if (start < 0 || start >= end || !j->lps) { q35_buf_append_str(b, "null"); return; }
    q35_buf_append_str(b, "{\"content\":[");
    int first = 1;
    for (int i = start; i < end; i++) {
        if (i >= j->n_lp) break;
        q35_logprobs *lp = &j->lps[i];
        if (!first) q35_buf_append_str(b, ",");
        first = 0;
        q35_buf_appendf(b, "{\"id\":%d,\"token\":", j->lp_ids ? j->lp_ids[i] : 0);
        char tbuf[4096];
        const char *txt = valid_token_text(tokenizer_decode(s->cfg.tok, j->lp_ids ? j->lp_ids[i] : 0), tbuf, sizeof(tbuf));
        jesc(b, txt);
        q35_buf_append_str(b, ",\"bytes\":");
        append_bytes(b, txt ? txt : "");
        q35_buf_appendf(b, ",\"logprob\":%.9g,\"top_logprobs\":[", lp->chosen_logprob);
        for (int k = 0; k < lp->n_top; k++) {
            if (k) q35_buf_append_str(b, ",");
            q35_buf_appendf(b, "{\"id\":%d,\"token\":", lp->top[k].id);
            char tbuf2[4096];
            const char *tt = valid_token_text(tokenizer_decode(s->cfg.tok, lp->top[k].id), tbuf2, sizeof(tbuf2));
            jesc(b, tt);
            q35_buf_append_str(b, ",\"bytes\":");
            append_bytes(b, tt);
            q35_buf_appendf(b, ",\"logprob\":%.9g}", lp->top[k].logprob);
        }
        q35_buf_append_str(b, "]}");
    }
    q35_buf_append_str(b, "]}");
}

static void append_tool_calls(q35_buf *b, q35_job *j) {
    q35_buf_append_str(b, "[");
    for (int i = 0; i < j->parsed.n_calls; i++) {
        if (i) q35_buf_append_str(b, ",");
        q35_buf_appendf(b, "{\"id\":\"call_%s_%d\",\"type\":\"function\",\"function\":{\"name\":",
                        j->id ? j->id : "0", i);
        jesc(b, j->parsed.calls[i].name ? j->parsed.calls[i].name : "");
        q35_buf_append_str(b, ",\"arguments\":");
        jesc(b, j->parsed.calls[i].arguments ? j->parsed.calls[i].arguments : "{}");
        q35_buf_append_str(b, "}}");
    }
    q35_buf_append_str(b, "]");
}

static void append_usage(q35_buf *b, q35_job *j) {
    double prompt_ms = (j->t_first - j->t_start) * 1000.0;
    double gen_ms = (j->t_end - j->t_first) * 1000.0;
    if (prompt_ms < 0.001) prompt_ms = 0.001;
    if (gen_ms < 0.001) gen_ms = 0.001;
    int total = j->n_prompt + j->n_gen;
    q35_buf_appendf(b,
        ",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
        "\"prompt_tokens_details\":{\"cached_tokens\":%d},"
        "\"completion_tokens_details\":{\"reasoning_tokens\":%d}}"
        ",\"timings\":{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.3f,"
        "\"prompt_per_token_ms\":%.3f,\"prompt_per_second\":%.2f,"
        "\"predicted_n\":%d,\"predicted_ms\":%.3f,\"predicted_per_token_ms\":%.3f,"
        "\"predicted_per_second\":%.2f}",
        j->n_prompt, j->n_gen, total, j->n_prompt_cached, 0,
        j->n_prompt_cached, j->n_prompt - j->n_prompt_cached, prompt_ms,
        prompt_ms / (j->n_prompt ? j->n_prompt : 1),
        j->n_prompt > 0 ? (j->n_prompt / (prompt_ms / 1000.0)) : 0.0,
        j->n_gen, gen_ms, j->n_gen ? gen_ms / j->n_gen : 0.0,
        j->n_gen ? (j->n_gen / (gen_ms / 1000.0)) : 0.0);
}

static void build_final_response(q35_server *s, q35_job *j) {
    parse_now(s, j);
    int calls = j->parse_tools && j->parsed.n_calls > 0;
    q35_buf *b = &j->response;
    q35_buf_append_str(b, "{\"id\":");
    jesc(b, j->id ? j->id : "x");
    q35_buf_append_str(b, ",\"object\":");
    jesc(b, j->object);
    q35_buf_appendf(b, ",\"created\":%lld,\"model\":", (long long)j->created);
    jesc(b, j->model ? j->model : (s->cfg.model_name ? s->cfg.model_name : ""));
    if (j->chat) {
        q35_buf_append_str(b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"");
        if (j->parsed.reasoning && j->parsed.reasoning[0]) {
            q35_buf_append_str(b, ",\"reasoning_content\":");
            jesc(b, j->parsed.reasoning);
        }
        q35_buf_append_str(b, ",\"content\":");
        jesc(b, j->parsed.content ? j->parsed.content : "");
        if (calls) { q35_buf_append_str(b, ",\"tool_calls\":"); append_tool_calls(b, j); }
        q35_buf_append_str(b, "},\"finish_reason\":");
        jesc(b, finish_name(j->finish));
        q35_buf_append_str(b, ",\"logprobs\":");
        append_logprobs_obj(s, b, j, -1);
        q35_buf_append_str(b, "}]");
    } else {
        q35_buf_append_str(b, ",\"choices\":[{\"text\":");
        jesc(b, j->parsed.content ? j->parsed.content : "");
        q35_buf_append_str(b, ",\"index\":0,\"logprobs\":");
        append_logprobs_obj(s, b, j, -1);
        q35_buf_append_str(b, ",\"finish_reason\":");
        jesc(b, finish_name(j->finish));
        q35_buf_append_str(b, "}]");
    }
    append_usage(b, j);
    if (j->ignored) { q35_buf_append_str(b, ",\"__tuf_ignored\":"); jesc(b, j->ignored); }
    q35_buf_append_str(b, "}");
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

static void sse_send(q35_job *j, q35_buf *payload) {
    q35_buf out;
    q35_buf_init(&out);
    q35_buf_append_str(&out, "data: ");
    q35_buf_append(&out, payload->data ? payload->data : "", payload->len);
    q35_buf_append_str(&out, "\n\n");
    if (send_all_fd(j->fd, out.data, out.len) != 0) j->cancelled = 1;
    q35_buf_free(&out);
}

static void sse_delta(q35_server *s, q35_job *j, const char *content,
                      const char *reasoning, const q35_tc *tc, int tc_index,
                      const char *finish, int lp_index) {
    q35_buf p;
    q35_buf_init(&p);
    q35_buf_append_str(&p, "{\"id\":");
    jesc(&p, j->id ? j->id : "x");
    q35_buf_append_str(&p, ",\"object\":");
    jesc(&p, j->chat && j->stream ? "chat.completion.chunk" : "text_completion");
    q35_buf_appendf(&p, ",\"created\":%lld,\"model\":", (long long)j->created);
    jesc(&p, j->model ? j->model : (s->cfg.model_name ? s->cfg.model_name : ""));
    q35_buf_append_str(&p, ",\"choices\":[{\"index\":0,\"delta\":{");
    if (j->chat) {
        int first = 1;
        if (content) {
            q35_buf_append_str(&p, "\"content\":");
            jesc(&p, content);
            first = 0;
        }
        if (reasoning) {
            if (!first) q35_buf_append_str(&p, ",");
            q35_buf_append_str(&p, "\"reasoning_content\":");
            jesc(&p, reasoning);
            first = 0;
        }
        if (tc) {
            if (!first) q35_buf_append_str(&p, ",");
            q35_buf_appendf(&p, "\"tool_calls\":[{\"index\":%d,\"id\":\"call_%s_%d\","
                                 "\"type\":\"function\",\"function\":{\"name\":",
                            tc_index, j->id ? j->id : "0", tc_index);
            jesc(&p, tc->name ? tc->name : "");
            q35_buf_append_str(&p, ",\"arguments\":");
            jesc(&p, tc->arguments ? tc->arguments : "{}");
            q35_buf_append_str(&p, "}}]");
        }
    } else {
        q35_buf_append_str(&p, "\"text\":");
        jesc(&p, content ? content : "");
    }
    q35_buf_append_str(&p, "},\"finish_reason\":");
    if (finish) jesc(&p, finish);
    else q35_buf_append_str(&p, "null");
    if (lp_index >= 0 && j->samp.logprobs) {
        q35_buf_append_str(&p, ",\"logprobs\":");
        append_logprobs_obj(s, &p, j, lp_index);
    }
    q35_buf_append_str(&p, "}]}");
    sse_send(j, &p);
    q35_buf_free(&p);
}

static void sse_role(q35_server *s, q35_job *j) {
    (void)s;
    q35_buf p;
    q35_buf_init(&p);
    q35_buf_append_str(&p, "{\"id\":");
    jesc(&p, j->id ? j->id : "x");
    q35_buf_append_str(&p, ",\"object\":\"chat.completion.chunk\",\"created\":");
    q35_buf_appendf(&p, "%lld,\"model\":", (long long)j->created);
    jesc(&p, j->model ? j->model : "");
    q35_buf_append_str(&p, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\","
                             "\"content\":null},\"finish_reason\":null}]}");
    sse_send(j, &p);
    q35_buf_free(&p);
}

static void stream_flush(q35_server *s, q35_job *j) {
    if (!j->stream || j->cancelled) return;
    if (!j->initial_sent) { sse_role(s, j); j->initial_sent = 1; }
    if (!j->raw.len) return;
    q35_out o;
    q35_parse_output(j->raw.data, j->raw.len, j->tools, j->thinking,
                     j->extract_reasoning, &o);
    // reasoning delta
    if (o.reasoning) {
        size_t have = j->emitted_reasoning, rlen = strlen(o.reasoning);
        if (rlen > have) {
            size_t avail = rlen - have;
            size_t v = utf8_valid_prefix((const unsigned char *)o.reasoning + have, avail);
            if (v > 0) {
                char *tmp = (char *)malloc(v + 1);
                memcpy(tmp, o.reasoning + have, v);
                tmp[v] = '\0';
                sse_delta(s, j, NULL, tmp, NULL, 0, NULL, -1);
                free(tmp);
                j->emitted_reasoning = have + v;
            }
        }
    }
    // content delta
    if (o.content) {
        size_t have = j->emitted_content, clen = strlen(o.content);
        if (clen > have) {
            size_t avail = clen - have;
            size_t v = utf8_valid_prefix((const unsigned char *)o.content + have, avail);
            if (v > 0) {
                char *tmp = (char *)malloc(v + 1);
                memcpy(tmp, o.content + have, v);
                tmp[v] = '\0';
                int lp = (j->samp.logprobs && j->n_gen > 0) ? j->n_gen - 1 : -1;
                sse_delta(s, j, tmp, NULL, NULL, 0, NULL, lp);
                free(tmp);
                j->emitted_content = have + v;
            }
        }
    }
    // completed tool calls
    while (j->emitted_calls < o.n_calls) {
        sse_delta(s, j, NULL, NULL, &o.calls[j->emitted_calls], j->emitted_calls, NULL,
                  (j->samp.logprobs && j->n_gen > 0) ? j->n_gen - 1 : -1);
        j->emitted_calls++;
    }
    q35_out_free(&o);
}

static void stream_final(q35_server *s, q35_job *j) {
    if (!j->stream || j->cancelled) return;
    stream_flush(s, j);
    int calls = j->parse_tools && j->parsed.n_calls > 0;
    const char *fin = calls ? "tool_calls" : finish_name(j->finish);
    sse_delta(s, j, NULL, NULL, NULL, 0, fin, -1);
    q35_buf p;
    q35_buf_init(&p);
    q35_buf_append_str(&p, "{\"id\":");
    jesc(&p, j->id ? j->id : "x");
    q35_buf_append_str(&p, ",\"object\":\"chat.completion.chunk\",\"created\":");
    q35_buf_appendf(&p, "%lld,\"model\":", (long long)j->created);
    jesc(&p, j->model ? j->model : "");
    q35_buf_append_str(&p, ",\"choices\":[]");
    append_usage(&p, j);
    q35_buf_append_str(&p, "}");
    sse_send(j, &p);
    q35_buf_free(&p);
    send_all_fd(j->fd, "data: [DONE]\n\n", 15);
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

static void slot_reset_seq(q35_slot *sl, int n) {
    if (n < 0) n = 0;
    if (n > sl->seq_len) n = sl->seq_len;
    sl->seq_len = n;
}

static int slot_stride_ok(q35_server *s, q35_slot *sl) {
    return sl->seq_cap == s->cfg.slot_ctx;
}

static int common_prefix(const int32_t *a, int na, const int32_t *b, int nb) {
    int n = na < nb ? na : nb;
    int i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

static q35_slot *pick_slot(q35_server *s, q35_job *j, int *out_reuse) {
    q35_slot *best = NULL;
    int best_prefix = -1;
    unsigned long long best_used = 0;
    for (int i = 0; i < s->cfg.n_slots; i++) {
        q35_slot *sl = &s->slots[i];
        if (sl->busy) continue;
        // GDN state cannot rewind: only the slot's *whole* previous sequence may be
        // reused.  A new prompt that merely shares a prefix with a longer old sequence
        // must start from a reset slot.
        int p = 0;
        if (sl->seq_len <= j->n_tokens &&
            common_prefix(j->tokens, sl->seq_len, sl->seq, sl->seq_len) == sl->seq_len) {
            p = sl->seq_len;
        }
        if (p >= j->n_tokens) p = j->n_tokens - 1;
        if (p < 0) p = 0;
        if (p > best_prefix || (p == best_prefix && sl->last_used < best_used)) {
            best = sl;
            best_prefix = p;
            best_used = sl->last_used;
        }
    }
    *out_reuse = best_prefix;
    return best;
}

static void emit_piece(q35_server *s, q35_job *j, q35_slot *sl) {
    (void)sl;
    int32_t tok = sl->pending;
    if (!j->ignore_eos && tok == s->cfg.tok->eos_id) {
        j->finish = FIN_STOP;
        return;
    }
    const char *piece = tokenizer_decode(s->cfg.tok, tok);
    if (piece && *piece) q35_buf_append_str(&j->raw, piece);
    j->n_gen++;

    // stop strings: cut the text at the first occurrence and finish.
    for (int i = 0; i < j->n_stops; i++) {
        const char *p = strstr(j->raw.data ? j->raw.data : "", j->stops[i]);
        if (p) {
            size_t pos = (size_t)(p - j->raw.data);
            j->raw.data[pos] = '\0';
            j->raw.len = pos;
            j->finish = FIN_STOP;
            return;
        }
    }
    if (j->n_gen >= j->max_tokens) j->finish = FIN_LENGTH;
}

static void finish_job(q35_server *s, q35_slot *sl, int fin) {
    q35_job *j = sl->job;
    if (!j) return;
    if (fin < 0) fin = FIN_STOP;
    j->finish = fin;
    j->t_end = now_s();
    parse_now(s, j);
    if (j->parse_tools && j->parsed.n_calls > 0) j->finish = FIN_TOOL_CALLS;
    if (!j->stream) build_final_response(s, j);
    else stream_final(s, j);
    (void)s;
    pthread_mutex_lock(&s->mu);
    j->done = 1;
    j->rc = 0;
    sl->busy = 0;
    sl->job = NULL;
    sl->phase = 0;
    sl->last_used = ++s->use_counter;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

static void cancel_job(q35_server *s, q35_job *j) {
    j->done = 1;
    j->rc = -2;
    j->finish = FIN_CANCELLED;
    if (j->slot >= 0) {
        q35_slot *sl = &s->slots[j->slot];
        sl->busy = 0;
        sl->job = NULL;
        sl->phase = 0;
        sl->last_used = ++s->use_counter;
    }
    j->slot = -1;
    pthread_cond_broadcast(&s->cv);
}

static int job_admit(q35_server *s, q35_job *j) {
    int reuse = 0;
    q35_slot *sl = pick_slot(s, j, &reuse);
    if (!sl) return -1;
    if (!slot_stride_ok(s, sl)) return -1;
    j->slot = (int)(sl - s->slots);
    if (getenv("Q35_SRV_DEBUG"))
        fprintf(stderr, "[q35-srv] admit slot=%d reuse=%d n=%d seq=%d job=%s\n",
                j->slot, reuse, j->n_tokens, sl->seq_len, j->id ? j->id : "?");
    sl->busy = 1;
    sl->job = j;
    sl->n_prompt = j->n_tokens;
    sl->gen_count = 0;
    sl->pending = -1;
    sl->prefilled = reuse;
    sl->rng = q35_rng_seed(j->samp.seed, j->slot, s->job_counter++);
    sl->need_logits = q35_sampling_needs_logits(&j->samp);
    if (reuse == 0) q35_engine_reset(s->cfg.engine, j->slot);
    slot_reset_seq(sl, reuse);
    j->n_prompt_cached = reuse;

    if (j->max_tokens <= 0) j->max_tokens = s->cfg.max_gen;
    if (j->max_tokens > s->cfg.slot_ctx - j->n_tokens) {
        j->max_tokens = s->cfg.slot_ctx - j->n_tokens;
    }
    if (j->max_tokens < 1) j->max_tokens = 1;

    if (j->n_tokens - 1 > reuse) {
        sl->phase = 1;   // prefill [reuse, n_tokens-1)
    } else {
        sl->phase = 2;   // feed the last prompt token in the decode launch
        sl->pending = j->tokens[j->n_tokens - 1];
    }
    j->t_start = now_s();
    j->t_first = j->t_start;
    return 0;
}

static q35_job *queue_pop(q35_server *s) {
    q35_job *j = s->q_head;
    if (!j) return NULL;
    s->q_head = j->next;
    if (!s->q_head) s->q_tail = NULL;
    j->next = NULL;
    return j;
}

static void *gpu_thread_main(void *arg) {
    q35_server *s = (q35_server *)arg;
    const int V = q35_model_vocab(s->cfg.model);
    int launches_since_admit = 1 << 30;
    int rr = 0;

    for (;;) {
        pthread_mutex_lock(&s->mu);
        if (s->stop) { pthread_mutex_unlock(&s->mu); break; }

        // Drop queued requests that were cancelled before admission.
        q35_job **pp = &s->q_head;
        while (*pp) {
            q35_job *j = *pp;
            if (j->cancelled) {
                *pp = j->next;
                if (s->q_tail == j) s->q_tail = NULL;
                j->next = NULL;
                cancel_job(s, j);
            } else {
                pp = &j->next;
            }
        }

        // Admit at launch boundaries (always when a slot is idle and work is queued).
        int idle_exists = 0;
        for (int i = 0; i < s->cfg.n_slots; i++) if (!s->slots[i].busy) { idle_exists = 1; break; }
        if (launches_since_admit >= s->cfg.steps_per_launch || (s->q_head && idle_exists)) {
            for (;;) {
                q35_job *j = s->q_head;
                if (!j) break;
                int idle = 0;
                for (int i = 0; i < s->cfg.n_slots; i++) if (!s->slots[i].busy) { idle = 1; break; }
                if (!idle) break;
                if (j->cancelled) {
                    queue_pop(s);
                    cancel_job(s, j);
                    continue;
                }
                queue_pop(s);
                if (job_admit(s, j) != 0) {
                    cancel_job(s, j);
                    continue;
                }
            }
            launches_since_admit = 0;
        }

        // Pick a prefill chunk (round robin over prefilling slots).
        q35_slot *pf = NULL;
        for (int k = 0; k < s->cfg.n_slots; k++) {
            q35_slot *sl = &s->slots[(rr + k) % s->cfg.n_slots];
            if (sl->busy && sl->phase == 1) { pf = sl; rr = (rr + k + 1) % s->cfg.n_slots; break; }
        }

        // Collect decoding entries.
        int n = 0;
        int any_logits = 0;
        for (int i = 0; i < s->cfg.n_slots; i++) {
            q35_slot *sl = &s->slots[i];
            if (sl->busy && sl->phase == 2) {
                s->d_slots[n] = i;
                s->d_pos[n] = sl->seq_len;
                s->d_tok[n] = sl->pending;
                if (sl->need_logits || sl->job->samp.logprobs) any_logits = 1;
                n++;
            }
        }
        int have_work = n > 0 || pf;
        pthread_mutex_unlock(&s->mu);

        if (!have_work) {
            pthread_mutex_lock(&s->mu);
            if (!s->stop && !s->q_head) pthread_cond_wait(&s->cv, &s->mu);
            launches_since_admit = 1 << 30;
            pthread_mutex_unlock(&s->mu);
            continue;
        }

        // One prefill chunk, interleaved with the decode launch below.
        if (pf) {
            q35_job *j = pf->job;
            int remain = j->n_tokens - 1 - pf->prefilled;
            if (remain > Q35_PREFILL_CHUNK) remain = Q35_PREFILL_CHUNK;
            if (remain > 0) {
                const int32_t *src = j->tokens + pf->prefilled;
                if (q35_engine_steps(s->cfg.engine, j->slot, src, remain,
                                     pf->prefilled, NULL, NULL) != 0) {
                    finish_job(s, pf, FIN_STOP);
                    continue;
                }
                for (int k = 0; k < remain; k++) {
                    if (pf->seq_len < pf->seq_cap) pf->seq[pf->seq_len++] = src[k];
                }
                pf->prefilled += remain;
            }
            if (pf->prefilled >= j->n_tokens - 1) {
                pf->phase = 2;
                pf->pending = j->tokens[j->n_tokens - 1];
            }
        }

        if (n > 0) {
            int32_t argmax[32];
            float *lg = any_logits ? s->logits : NULL;
            int rc = q35_engine_decode_batch(s->cfg.engine, s->d_slots, s->d_pos, s->d_tok,
                                             n, argmax, lg);
            if (rc != 0) {
                for (int i = 0; i < n; i++) {
                    q35_slot *sl = &s->slots[s->d_slots[i]];
                    if (sl->busy && sl->phase == 2) finish_job(s, sl, FIN_STOP);
                }
                continue;
            }
            // Sample and emit outside the lock; only the GPU thread touches slot state.
            for (int i = 0; i < n; i++) {
                q35_slot *sl = &s->slots[s->d_slots[i]];
                if (!sl->busy || sl->phase != 2) continue;
                q35_job *j = sl->job;
                if (j->cancelled) continue;
                int32_t chosen;
                float *row = NULL;
                if (lg) {
                    row = lg + (size_t)i * V;
                    if (j->samp.logprobs) memcpy(s->raw_copy, row, sizeof(float) * (size_t)V);
                    if (j->samp.temp > 0.0f)
                        chosen = q35_sampling_sample(row, V, &j->samp, sl->seq, sl->seq_len, &sl->rng);
                    else if (q35_sampling_needs_logits(&j->samp))
                        chosen = q35_sampling_greedy(row, V, &j->samp, sl->seq, sl->seq_len);
                    else
                        chosen = argmax[i];
                } else {
                    chosen = argmax[i];
                }
                if (j->samp.logprobs) {
                    if (j->n_gen >= j->lp_cap) {
                        j->lp_cap = j->lp_cap ? j->lp_cap * 2 : 64;
                        j->lps = (q35_logprobs *)realloc(j->lps, sizeof(q35_logprobs) * (size_t)j->lp_cap);
                    }
                    q35_logprobs lp;
                    q35_compute_logprobs(s->raw_copy, V, chosen, j->samp.top_logprobs, &lp);
                    j->lp_ids = (int32_t *)realloc(j->lp_ids, sizeof(int32_t) * (size_t)j->lp_cap);
                    j->lp_ids[j->n_lp] = chosen;
                    j->lps[j->n_lp++] = lp;
                }
                // Advance the engine sequence with the token that was just fed.
                if (sl->seq_len < sl->seq_cap) sl->seq[sl->seq_len++] = sl->pending;
                sl->pending = chosen;
                if (j->n_gen == 0 && j->t_first == j->t_start) j->t_first = now_s();

                emit_piece(s, j, sl);
                if (j->finish >= 0) {
                    // The sampled token is not fed after a stop/length finish.
                    finish_job(s, sl, j->finish);
                    continue;
                }
                // Context exhausted?
                if (sl->seq_len + 1 >= s->cfg.slot_ctx) {
                    finish_job(s, sl, FIN_LENGTH);
                    continue;
                }
                stream_flush(s, j);
            }
        }

        // Cancellations noticed after the launch.
        pthread_mutex_lock(&s->mu);
        for (int i = 0; i < s->cfg.n_slots; i++) {
            q35_slot *sl = &s->slots[i];
            if (sl->busy && sl->job && sl->job->cancelled) {
                cancel_job(s, sl->job);
                i = -1;
            }
        }
        launches_since_admit++;
        pthread_mutex_unlock(&s->mu);
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

q35_server *q35_server_create(const q35_server_cfg *cfg) {
    q35_server *s = (q35_server *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = *cfg;
    if (s->cfg.steps_per_launch < 1) s->cfg.steps_per_launch = 16;
    s->cfg.reasoning_format = cfg->reasoning_format;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
    s->slots = (q35_slot *)calloc((size_t)cfg->n_slots, sizeof(q35_slot));
    s->d_slots = (int *)malloc(sizeof(int) * (size_t)cfg->n_slots);
    s->d_pos = (int *)malloc(sizeof(int) * (size_t)cfg->n_slots);
    s->d_tok = (int32_t *)malloc(sizeof(int32_t) * (size_t)cfg->n_slots);
    const int V = q35_model_vocab(cfg->model);
    s->logits = (float *)malloc(sizeof(float) * (size_t)V * (size_t)cfg->n_slots);
    s->raw_copy = (float *)malloc(sizeof(float) * (size_t)V);
    for (int i = 0; i < cfg->n_slots; i++) {
        s->slots[i].seq_cap = cfg->slot_ctx;
        s->slots[i].seq = (int32_t *)calloc((size_t)cfg->slot_ctx, sizeof(int32_t));
    }
    if (cfg->chat_template_kwargs && cfg->chat_template_kwargs[0])
        s->default_kwargs = qj_parse(cfg->chat_template_kwargs, strlen(cfg->chat_template_kwargs));
    if (pthread_create(&s->gpu, NULL, gpu_thread_main, s) != 0) {
        q35_server_free(s);
        return NULL;
    }
    return s;
}

void q35_server_free(q35_server *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mu);
    s->stop = 1;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
    pthread_join(s->gpu, NULL);
    for (int i = 0; i < s->cfg.n_slots; i++) free(s->slots[i].seq);
    free(s->slots);
    free(s->d_slots);
    free(s->d_pos);
    free(s->d_tok);
    free(s->logits);
    free(s->raw_copy);
    qj_free(s->default_kwargs);
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
    free(s);
}

// ---------------------------------------------------------------------------
// Submission / waiting
// ---------------------------------------------------------------------------

static int submit_and_wait(q35_server *s, q35_job *j) {
    pthread_mutex_lock(&s->mu);
    j->next = NULL;
    if (s->q_tail) s->q_tail->next = j; else s->q_head = j;
    s->q_tail = j;
    pthread_cond_broadcast(&s->cv);
    // Wait, polling for a client disconnect so queued/running jobs can be freed early.
    struct timespec ts;
    for (;;) {
        if (j->done) break;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 50 * 1000 * 1000;
        if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&s->cv, &s->mu, &ts);
        if (j->done) break;
        struct pollfd pfd = { .fd = j->fd, .events = POLLHUP | POLLERR };
        int pr = poll(&pfd, 1, 0);
        if (pr > 0 && (pfd.revents & (POLLHUP | POLLERR))) j->cancelled = 1;
    }
    int rc = j->rc;
    pthread_mutex_unlock(&s->mu);
    return rc;
}

static q35_job *job_new(void) {
    q35_job *j = (q35_job *)calloc(1, sizeof(*j));
    q35_buf_init(&j->raw);
    q35_buf_init(&j->response);
    j->slot = -1;
    j->max_tokens = -1;
    j->finish = -1;
    return j;
}

// Common request front-end for both chat and completion.
static void handle_generate(q35_server *s, int fd, const char *body, size_t len, int chat) {
    qj_val *root = qj_parse(body ? body : "", len);
    if (!root) { respond_error(fd, 400, "Invalid JSON body"); return; }
    q35_job *j = job_new();
    j->root = root;
    char err[256] = {0};
    if (parse_request_common(s, root, j, chat, err, sizeof(err)) != 0) {
        respond_error(fd, 400, err);
        free_job(j);
        return;
    }
    char *prompt = render_prompt(s, root, chat, err, sizeof(err));
    if (!prompt) {
        respond_error(fd, 400, err);
        free_job(j);
        return;
    }
    int max_tok = s->cfg.slot_ctx + 1;
    j->tokens = (int32_t *)malloc(sizeof(int32_t) * (size_t)max_tok);
    int n = tokenizer_tokenize(s->cfg.tok, prompt, j->tokens, max_tok);
    free(prompt);
    if (n <= 0 || n > s->cfg.slot_ctx) {
        respond_error(fd, 400, "prompt does not fit in the per-slot context");
        free_job(j);
        return;
    }
    if (n < 2) {
        // A single-token prompt has no position to prefill; pad is not meaningful, reject.
        respond_error(fd, 400, "prompt is too short");
        free_job(j);
        return;
    }
    j->n_tokens = n;
    j->n_prompt = n;
    j->fd = fd;
    j->id = (char *)malloc(64);
    static unsigned long long counter = 0;
    snprintf(j->id, 64, "%s%lld%llu", chat ? "chatcmpl-" : "cmpl-", (long long)time(NULL),
             counter++);
    j->object = chat ? "chat.completion" : "text_completion";
    if (j->max_tokens < 0) j->max_tokens = s->cfg.max_gen;

    if (j->stream) respond_stream_headers(fd, j->ignored);
    if (j->samp.logprobs) j->lp_cap = j->max_tokens;
    if (j->lp_cap > 0) {
        j->lps = (q35_logprobs *)calloc((size_t)j->lp_cap, sizeof(q35_logprobs));
        j->lp_ids = (int32_t *)calloc((size_t)j->lp_cap, sizeof(int32_t));
    }

    int rc = submit_and_wait(s, j);
    if (j->stream) {
        if (j->ignored) {
            // Close the connection; the ignored field list is in the final JSON chunk only
            // when the client asked for non-streaming, so also stash it in the last chunk.
        }
    } else if (rc == 0) {
        q35_buf hdr;
        q35_buf_init(&hdr);
        if (j->ignored) {
            q35_buf_appendf(&hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                  "Content-Length: %zu\r\nX-Tuf-Ignored: %s\r\n"
                                  "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                            j->response.len, j->ignored);
        } else {
            q35_buf_appendf(&hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                  "Content-Length: %zu\r\n"
                                  "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                            j->response.len);
        }
        send_all_fd(fd, hdr.data, hdr.len);
        send_all_fd(fd, j->response.data ? j->response.data : "", j->response.len);
        q35_buf_free(&hdr);
    } else if (rc != -2) {
        respond_error(fd, 500, "generation failed");
    }
    free_job(j);
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

void q35_http_chat(q35_server *s, int fd, const char *body, size_t len) {
    handle_generate(s, fd, body, len, 1);
}

void q35_http_completion(q35_server *s, int fd, const char *body, size_t len) {
    handle_generate(s, fd, body, len, 0);
}

void q35_http_tokenize(q35_server *s, int fd, const char *body, size_t len) {
    qj_val *root = qj_parse(body ? body : "", len);
    if (!root) { respond_error(fd, 400, "Invalid JSON body"); return; }
    const char *text = qj_str(qj_get(root, "content"));
    if (!text) text = qj_str(qj_get(root, "prompt"));
    if (!text) { respond_error(fd, 400, "expected {\"content\":\"...\"}"); qj_free(root); return; }
    int cap = s->cfg.slot_ctx + 1;
    int32_t *t = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    int n = tokenizer_tokenize(s->cfg.tok, text, t, cap);
    q35_buf b;
    q35_buf_init(&b);
    q35_buf_append_str(&b, "{\"tokens\":[");
    for (int i = 0; i < n; i++) q35_buf_appendf(&b, "%s%d", i ? "," : "", t[i]);
    q35_buf_append_str(&b, "]}");
    respond_json_q(fd, 200, "OK", &b);
    q35_buf_free(&b);
    free(t);
    qj_free(root);
}

void q35_http_detokenize(q35_server *s, int fd, const char *body, size_t len) {
    qj_val *root = qj_parse(body ? body : "", len);
    const qj_val *arr = qj_get(root, "tokens");
    if (!root || !arr || arr->type != QJ_ARR) {
        respond_error(fd, 400, "expected {\"tokens\":[...]}");
        qj_free(root);
        return;
    }
    q35_buf txt;
    q35_buf_init(&txt);
    for (int i = 0; i < arr->n; i++) {
        int id = (int)qj_int(arr->items[i], -1);
        if (id >= 0 && id < (int)s->cfg.tok->vocab_size) {
            const char *d = tokenizer_decode(s->cfg.tok, id);
            if (d) q35_buf_append_str(&txt, d);
        }
    }
    q35_buf b;
    q35_buf_init(&b);
    q35_buf_append_str(&b, "{\"content\":");
    jesc(&b, txt.data ? txt.data : "");
    q35_buf_append_str(&b, "}");
    respond_json_q(fd, 200, "OK", &b);
    q35_buf_free(&b);
    q35_buf_free(&txt);
    qj_free(root);
}

void q35_http_apply_template(q35_server *s, int fd, const char *body, size_t len) {
    qj_val *root = qj_parse(body ? body : "", len);
    if (!root) { respond_error(fd, 400, "Invalid JSON body"); return; }
    char err[256] = {0};
    const qj_val *messages = qj_get(root, "messages");
    if (!messages || messages->type != QJ_ARR || messages->n == 0) {
        respond_error(fd, 400, "expected messages");
        qj_free(root);
        return;
    }
    q35_msgs msgs;
    if (q35_messages_from_json(messages, &msgs, err, sizeof(err)) != 0) {
        respond_error(fd, 400, err);
        qj_free(root);
        return;
    }
    int thinking = 0, add_gen = 1;
    resolve_template_opts(s, root, &thinking, &add_gen);
    const qj_val *tools = qj_get(root, "tools");
    char *prompt = q35_render_chat(&msgs, tools && tools->type == QJ_ARR ? tools : NULL,
                                   add_gen, thinking, err, sizeof(err));
    q35_messages_free(&msgs);
    if (!prompt) { respond_error(fd, 400, err); qj_free(root); return; }
    q35_buf b;
    q35_buf_init(&b);
    q35_buf_append_str(&b, "{\"prompt\":");
    jesc(&b, prompt);
    q35_buf_append_str(&b, "}");
    respond_json_q(fd, 200, "OK", &b);
    q35_buf_free(&b);
    free(prompt);
    qj_free(root);
}

void q35_http_props(q35_server *s, int fd) {
    q35_buf b;
    q35_buf_init(&b);
    q35_buf_appendf(&b, "{\"n_ctx\":%d,\"total_slots\":%d,\"chat_template\":\"qwen35\",\"model\":",
                    s->cfg.slot_ctx, s->cfg.n_slots);
    jesc(&b, s->cfg.model_name ? s->cfg.model_name : "");
    q35_buf_append_str(&b, "}");
    respond_json_q(fd, 200, "OK", &b);
    q35_buf_free(&b);
}
