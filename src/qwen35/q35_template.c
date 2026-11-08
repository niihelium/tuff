#include "q35_template.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

char *q35_trim_dup(const char *s) {
    if (!s) return strdup("");
    while (*s && is_ws(*s)) s++;
    size_t n = strlen(s);
    while (n > 0 && is_ws(s[n - 1])) n--;
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

// Concatenate a message `content` (string, array of text parts, or null).
static char *render_content(const qj_val *v) {
    if (!v || v->type == QJ_NULL) return strdup("");
    if (v->type == QJ_STR) return strdup(v->str);
    if (v->type == QJ_ARR) {
        q35_buf b;
        q35_buf_init(&b);
        for (int i = 0; i < v->n; i++) {
            const qj_val *item = v->items[i];
            if (!item) continue;
            if (item->type == QJ_STR) { q35_buf_append_str(&b, item->str); continue; }
            if (item->type != QJ_OBJ) continue;
            const qj_val *text = qj_get(item, "text");
            if (text) {
                if (text->type == QJ_STR) q35_buf_append_str(&b, text->str);
                else { q35_buf t; q35_buf_init(&t); qj_dump_string(&t, text); q35_buf_append_str(&b, t.data ? t.data : ""); q35_buf_free(&t); }
            }
        }
        return b.data ? b.data : strdup("");
    }
    return strdup("");
}

int q35_messages_from_json(const qj_val *arr, q35_msgs *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    if (!arr || arr->type != QJ_ARR) {
        snprintf(err, errlen, "messages must be an array");
        return -1;
    }
    out->n = arr->n;
    out->items = (q35_msg *)calloc((size_t)(arr->n ? arr->n : 1), sizeof(q35_msg));
    for (int i = 0; i < arr->n; i++) {
        const qj_val *m = arr->items[i];
        if (!m || m->type != QJ_OBJ) {
            snprintf(err, errlen, "message %d is not an object", i);
            return -1;
        }
        const char *role = qj_str(qj_get(m, "role"));
        const char *r = role ? role : "";
        // llama.cpp maps the developer role to system.
        if (strcmp(r, "developer") == 0) r = "system";
        q35_msg *dst = &out->items[i];
        dst->role = strdup(r);
        char *content = render_content(qj_get(m, "content"));
        dst->content = q35_trim_dup(content);
        free(content);
        const qj_val *rc = qj_get(m, "reasoning_content");
        dst->reasoning_content = (rc && rc->type == QJ_STR) ? q35_trim_dup(rc->str) : NULL;
        dst->tool_calls = qj_get(m, "tool_calls");
        // llama.cpp parses string `arguments` into JSON before rendering; invalid JSON is
        // a request error there and here.
        if (dst->tool_calls && dst->tool_calls->type == QJ_ARR) {
            for (int k = 0; k < dst->tool_calls->n; k++) {
                qj_val *tc = dst->tool_calls->items[k];
                if (!tc || tc->type != QJ_OBJ) continue;
                qj_val *fn = qj_get(tc, "function");
                if (!fn) fn = tc;
                qj_val *args = qj_get(fn, "arguments");
                if (args && args->type == QJ_STR) {
                    qj_val *parsed = qj_parse(args->str, strlen(args->str));
                    if (!parsed) {
                        snprintf(err, errlen, "tool_call arguments is not valid JSON");
                        return -1;
                    }
                    // Replace the string value in place with the parsed object.
                    free(args->str);
                    *args = *parsed;
                    free(parsed);
                }
            }
        }
    }
    return 0;
}

void q35_messages_free(q35_msgs *m) {
    if (!m || !m->items) return;
    for (int i = 0; i < m->n; i++) {
        free(m->items[i].role);
        free(m->items[i].content);
        free(m->items[i].reasoning_content);
    }
    free(m->items);
    m->items = NULL;
    m->n = 0;
}

int q35_template_kwargs(const qj_val *kwargs, int *enable_thinking, int *add_generation_prompt,
                        int def_thinking, int def_add_generation_prompt) {
    int et = def_thinking, agp = def_add_generation_prompt;
    if (kwargs && kwargs->type == QJ_OBJ) {
        const qj_val *v = qj_get(kwargs, "enable_thinking");
        if (v) et = qj_bool(v, et);
        v = qj_get(kwargs, "add_generation_prompt");
        if (v) agp = qj_bool(v, agp);
    }
    if (enable_thinking) *enable_thinking = et;
    if (add_generation_prompt) *add_generation_prompt = agp;
    return 0;
}

// `content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')`
static char *derive_reasoning(const char *content) {
    const char *close = strstr(content, "</think>");
    if (!close) return NULL;
    size_t head = (size_t)(close - content);
    while (head > 0 && content[head - 1] == '\n') head--;
    // last '<think>' in [0, head)
    const char *open = NULL;
    const char *scan = content;
    while ((scan = strstr(scan, "<think>")) != NULL && (size_t)(scan - content) < head) {
        open = scan;
        scan += 7;
    }
    const char *start = open ? open + 7 : content;
    while (start < content + head && *start == '\n') start++;
    size_t n = (size_t)(content + head - start);
    char *out = (char *)malloc(n + 1);
    memcpy(out, start, n);
    out[n] = '\0';
    return out;
}

static void append_system_tools(q35_buf *b, const qj_val *tools) {
    q35_buf_append_str(b, "<|im_start|>system\n");
    q35_buf_append_str(b, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (int i = 0; i < tools->n; i++) {
        q35_buf_append_str(b, "\n");
        qj_dump(b, tools->items[i]);
    }
    q35_buf_append_str(b, "\n</tools>");
    q35_buf_append_str(b,
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
        "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
        "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...>"
        "</function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
        "function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your "
        "current knowledge and do not tell the user about function calls\n</IMPORTANT>");
}

char *q35_render_chat(const q35_msgs *m, const qj_val *tools, int add_generation_prompt,
                      int enable_thinking, char *err, size_t errlen) {
    if (!m || m->n <= 0) {
        snprintf(err, errlen, "messages must not be empty");
        return NULL;
    }
    const int has_tools = tools && tools->type == QJ_ARR && tools->n > 0;
    q35_buf b;
    q35_buf_init(&b);

    const int first_system = strcmp(m->items[0].role, "system") == 0;
    if (has_tools) {
        append_system_tools(&b, tools);
        if (first_system) {
            char *c = q35_trim_dup(m->items[0].content);
            if (c && c[0]) {
                q35_buf_append_str(&b, "\n\n");
                q35_buf_append_str(&b, c);
            }
            free(c);
        }
        q35_buf_append_str(&b, "<|im_end|>\n");
    } else if (first_system) {
        char *c = q35_trim_dup(m->items[0].content);
        q35_buf_append_str(&b, "<|im_start|>system\n");
        q35_buf_append_str(&b, c ? c : "");
        q35_buf_append_str(&b, "<|im_end|>\n");
        free(c);
    }

    // last_query_index: last user message that is not a tool response.
    int last_query = -1;
    for (int i = m->n - 1; i >= 0; i--) {
        if (strcmp(m->items[i].role, "user") != 0) continue;
        const char *c = m->items[i].content ? m->items[i].content : "";
        size_t L = strlen(c);
        int starts = strncmp(c, "<tool_response>", 15) == 0;
        int ends = L >= 16 && strcmp(c + L - 16, "</tool_response>") == 0;
        if (!(starts && ends)) { last_query = i; break; }
    }
    if (last_query < 0) {
        snprintf(err, errlen, "No user query found in messages");
        q35_buf_free(&b);
        return NULL;
    }

    for (int i = 0; i < m->n; i++) {
        const q35_msg *msg = &m->items[i];
        const char *role = msg->role;
        const char *content = msg->content ? msg->content : "";
        char *trimmed = q35_trim_dup(content);
        if (strcmp(role, "system") == 0) {
            if (i != 0) {
                snprintf(err, errlen, "System message must be at the beginning");
                free(trimmed);
                q35_buf_free(&b);
                return NULL;
            }
            // Emitted above.
        } else if (strcmp(role, "user") == 0) {
            q35_buf_append_str(&b, "<|im_start|>user\n");
            q35_buf_append_str(&b, trimmed);
            q35_buf_append_str(&b, "<|im_end|>\n");
        } else if (strcmp(role, "assistant") == 0) {
            char *reasoning = msg->reasoning_content ? q35_trim_dup(msg->reasoning_content)
                                                     : derive_reasoning(trimmed);
            char *body = trimmed;
            if (!msg->reasoning_content && reasoning) {
                const char *close = strstr(trimmed, "</think>");
                const char *after = close + 8;
                while (*after == '\n') after++;
                body = strdup(after);
            }
            q35_buf_append_str(&b, "<|im_start|>assistant\n");
            if (i > last_query) {
                q35_buf_append_str(&b, "<think>\n");
                q35_buf_append_str(&b, reasoning ? reasoning : "");
                q35_buf_append_str(&b, "\n</think>\n\n");
                q35_buf_append_str(&b, body);
            } else {
                q35_buf_append_str(&b, body);
            }
            free(reasoning);
            if (body != trimmed) free(body);
            if (msg->tool_calls && msg->tool_calls->type == QJ_ARR && msg->tool_calls->n > 0) {
                int first_call = 1;
                for (int k = 0; k < msg->tool_calls->n; k++) {
                    const qj_val *tc = msg->tool_calls->items[k];
                    if (!tc || tc->type != QJ_OBJ) continue;
                    const qj_val *fn = qj_get(tc, "function");
                    if (!fn) fn = tc;
                    const char *name = qj_str(qj_get(fn, "name"));
                    if (!name) name = "";
                    if (first_call) {
                        char *ct = q35_trim_dup(content);
                        if (ct && ct[0]) q35_buf_append_str(&b, "\n\n<tool_call>\n<function=");
                        else q35_buf_append_str(&b, "<tool_call>\n<function=");
                        free(ct);
                        first_call = 0;
                    } else {
                        q35_buf_append_str(&b, "\n<tool_call>\n<function=");
                    }
                    q35_buf_append_str(&b, name);
                    q35_buf_append_str(&b, ">\n");
                    const qj_val *args = qj_get(fn, "arguments");
                    if (args && args->type == QJ_OBJ) {
                        for (int a = 0; a < args->n; a++) {
                            const qj_val *av = args->items[a];
                            q35_buf_append_str(&b, "<parameter=");
                            q35_buf_append_str(&b, args->keys[a]);
                            q35_buf_append_str(&b, ">\n");
                            if (av->type == QJ_OBJ || av->type == QJ_ARR) qj_dump(&b, av);
                            else qj_dump_string(&b, av);
                            q35_buf_append_str(&b, "\n</parameter>\n");
                        }
                    }
                    q35_buf_append_str(&b, "</function>\n</tool_call>");
                }
            }
            q35_buf_append_str(&b, "<|im_end|>\n");
        } else if (strcmp(role, "tool") == 0) {
            const q35_msg *prev = i > 0 ? &m->items[i - 1] : NULL;
            const q35_msg *next = (i + 1 < m->n) ? &m->items[i + 1] : NULL;
            if (prev && strcmp(prev->role, "tool") != 0) q35_buf_append_str(&b, "<|im_start|>user");
            q35_buf_append_str(&b, "\n<tool_response>\n");
            q35_buf_append_str(&b, trimmed);
            q35_buf_append_str(&b, "\n</tool_response>");
            if ((next && strcmp(next->role, "tool") != 0) || !next)
                q35_buf_append_str(&b, "<|im_end|>\n");
        } else {
            snprintf(err, errlen, "Unexpected message role: %s", role);
            free(trimmed);
            q35_buf_free(&b);
            return NULL;
        }
        free(trimmed);
    }

    if (add_generation_prompt) {
        q35_buf_append_str(&b, "<|im_start|>assistant\n");
        if (enable_thinking) q35_buf_append_str(&b, "<think>\n");
        else q35_buf_append_str(&b, "<think>\n\n</think>\n\n");
    }
    return b.data ? b.data : strdup("");
}
