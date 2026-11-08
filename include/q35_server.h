#ifndef Q35_SERVER_H
#define Q35_SERVER_H

// Backend B: Qwen3.5 dense HTTP side (request parsing, continuous-batching scheduler,
// streaming, template/tools, sampler and logprobs).  server.c only owns the socket and
// routes these paths here.

#include <stddef.h>
#include "qwen35.h"
#include "tokenizer.h"
#include "q35_json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q35_server q35_server;

typedef struct {
    q35_model     *model;
    tokenizer_t   *tok;
    q35_engine    *engine;
    const char    *model_name;
    const char    *system_prompt;       // NULL: no default system message
    const char    *chat_template_kwargs;// server --chat-template-kwargs, JSON text
    int            n_slots;
    int            slot_ctx;
    int            max_gen;
    int            steps_per_launch;    // default 16
    int            reasoning_format;    // 0 = none (inline), 1 = deepseek (default)
} q35_server_cfg;

q35_server *q35_server_create(const q35_server_cfg *cfg);
void        q35_server_free(q35_server *s);

// HTTP handlers (body already read; fd owned by the caller).
void q35_http_chat(q35_server *s, int fd, const char *body, size_t len);
void q35_http_completion(q35_server *s, int fd, const char *body, size_t len);
void q35_http_tokenize(q35_server *s, int fd, const char *body, size_t len);
void q35_http_detokenize(q35_server *s, int fd, const char *body, size_t len);
void q35_http_apply_template(q35_server *s, int fd, const char *body, size_t len);
void q35_http_props(q35_server *s, int fd);

#ifdef __cplusplus
}
#endif

#endif
