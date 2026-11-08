#ifndef Q35_TEMPLATE_H
#define Q35_TEMPLATE_H

// Qwen3.5 chat template renderer (backend B).
//
// This is a dedicated C implementation of the `tokenizer.chat_template` stored in the
// Qwen3.5 GGUFs, not a general Jinja engine.  It covers exactly the constructs that
// template uses: system/tools preamble, user, assistant (content, reasoning_content,
// tool_calls), grouped tool results, old-turn reasoning stripping, add_generation_prompt
// and enable_thinking.  Byte-for-byte parity with llama-server --jinja is gated by G8.

#include "q35_json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char         *role;              // owned
    char         *content;           // owned, rendered text parts concatenated
    char         *reasoning_content; // owned, may be NULL
    qj_val       *tool_calls;        // borrowed from the request tree, may be NULL
} q35_msg;

typedef struct {
    q35_msg *items;
    int      n;
} q35_msgs;

// Convert the OpenAI `messages` array into the renderer's representation.  Content
// whitespace is trimmed and tool_call `arguments` strings are parsed to JSON objects,
// matching llama.cpp's server preprocessing.
int  q35_messages_from_json(const qj_val *arr, q35_msgs *out, char *err, size_t errlen);
void q35_messages_free(q35_msgs *m);

// enable_thinking default (used by /apply-template and as the server default).
int  q35_template_kwargs(const qj_val *kwargs, int *enable_thinking, int *add_generation_prompt,
                         int def_thinking, int def_add_generation_prompt);

// Render.  `tools` is the OpenAI tools array or NULL.  Returns a malloc'd UTF-8 string or
// NULL with a message in err.
char *q35_render_chat(const q35_msgs *m, const qj_val *tools, int add_generation_prompt,
                      int enable_thinking, char *err, size_t errlen);

// Trim helpers (Python str.strip() semantics for ASCII whitespace).
char *q35_trim_dup(const char *s);

#ifdef __cplusplus
}
#endif

#endif
