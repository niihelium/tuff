#include "q35_toolparse.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static const char *find_str(const char *p, size_t n, const char *needle) {
    size_t m = strlen(needle);
    if (m == 0 || n < m) return NULL;
    for (size_t i = 0; i + m <= n; i++)
        if (memcmp(p + i, needle, m) == 0) return p + i;
    return NULL;
}

// Look up a parameter's declared type for a tool.  Returns 1 when the schema says the
// value is a plain string, 0 when it is any other type, -1 when unknown.
static int param_is_string(const qj_val *tools, const char *tool, const char *param) {
    if (!tools || tools->type != QJ_ARR) return -1;
    for (int i = 0; i < tools->n; i++) {
        const qj_val *fn = qj_get(tools->items[i], "function");
        if (!fn) fn = tools->items[i];
        const char *name = qj_str(qj_get(fn, "name"));
        if (!name || strcmp(name, tool) != 0) continue;
        const qj_val *params = qj_get(fn, "parameters");
        const qj_val *props = qj_get(params, "properties");
        const qj_val *ps = qj_get(props, param);
        if (!ps) return -1;
        const qj_val *type = qj_get(ps, "type");
        if (!type) return -1;
        if (type->type == QJ_STR) return strcmp(type->str, "string") == 0 ? 1 : 0;
        if (type->type == QJ_ARR) {
            for (int k = 0; k < type->n; k++)
                if (qj_str(type->items[k]) && strcmp(qj_str(type->items[k]), "string") != 0) return 0;
            return 1;
        }
        return -1;
    }
    return -1;
}

// Append `{"key": value, ...}` like llama.cpp's tagged-format mapper (", " / ": ").
typedef struct {
    q35_buf b;
    int     n;
} args_buf;

static void args_add(args_buf *a, const char *key, const char *value, size_t vlen,
                     const qj_val *tools, const char *tool) {
    if (a->n == 0) q35_buf_append_str(&a->b, "{");
    else q35_buf_append_str(&a->b, ",");
    a->n++;
    q35_buf keyv = {0};
    qj_val k;
    memset(&k, 0, sizeof(k));
    k.type = QJ_STR;
    k.str = (char *)key;
    qj_dump_compact(&keyv, &k);
    q35_buf_append(&a->b, keyv.data, keyv.len);
    q35_buf_free(&keyv);
    q35_buf_append_str(&a->b, ":");
    int is_str = param_is_string(tools, tool, key);
    if (is_str == 1) {
        qj_val s;
        memset(&s, 0, sizeof(s));
        s.type = QJ_STR;
        s.str = (char *)value;
        // The value bytes are not NUL-terminated at vlen; copy when needed.
        char *tmp = NULL;
        if (value[vlen] != '\0') {
            tmp = (char *)malloc(vlen + 1);
            memcpy(tmp, value, vlen);
            tmp[vlen] = '\0';
            s.str = tmp;
        }
        qj_dump_compact(&a->b, &s);
        free(tmp);
    } else {
        qj_val *parsed = NULL;
        if (vlen > 0) {
            char *tmp = (char *)malloc(vlen + 1);
            memcpy(tmp, value, vlen);
            tmp[vlen] = '\0';
            parsed = qj_parse(tmp, vlen);
            free(tmp);
        }
        if (parsed) {
            qj_dump_compact(&a->b, parsed);
            qj_free(parsed);
        } else {
            qj_val s;
            memset(&s, 0, sizeof(s));
            s.type = QJ_STR;
            char *tmp = (char *)malloc(vlen + 1);
            memcpy(tmp, value, vlen);
            tmp[vlen] = '\0';
            s.str = tmp;
            qj_dump_compact(&a->b, &s);
            free(tmp);
        }
    }
}

size_t q35_parse_tool_block(const char *p, size_t len, const qj_val *tools, q35_tc *out) {
    memset(out, 0, sizeof(*out));
    size_t i = 0;
    const char *open = "<tool_call>";
    const size_t olen = strlen(open);
    if (len < olen || memcmp(p, open, olen) != 0) return 0;
    i = olen;
    // Template emits one newline after <tool_call>; tolerate spaces too.
    while (i < len && (p[i] == ' ' || p[i] == '\t' || p[i] == '\r')) i++;
    if (i < len && p[i] == '\n') i++;
    const char *fopen = "<function=";
    const size_t flen = strlen(fopen);
    if (i + flen > len || memcmp(p + i, fopen, flen) != 0) return 0;
    i += flen;
    size_t name_start = i;
    while (i < len && p[i] != '>') i++;
    if (i >= len) return 0;
    out->name = (char *)malloc(i - name_start + 1);
    memcpy(out->name, p + name_start, i - name_start);
    out->name[i - name_start] = '\0';
    i++; // '>'
    if (i < len && p[i] == '\n') i++;

    args_buf a;
    memset(&a, 0, sizeof(a));
    q35_buf_init(&a.b);

    const char *popen = "<parameter=";
    const size_t plen = strlen(popen);
    const char *pclose = "</parameter>";
    const size_t pclen = strlen(pclose);
    const char *fclose = "</function>";
    const size_t fclen = strlen(fclose);

    for (;;) {
        // optional whitespace/newlines between parameters
        size_t save = i;
        while (i < len && (p[i] == '\n' || p[i] == '\r' || p[i] == ' ' || p[i] == '\t')) i++;
        if (i + fclen <= len && memcmp(p + i, fclose, fclen) == 0) {
            i += fclen;
            break;
        }
        if (i + plen > len || memcmp(p + i, popen, plen) != 0) {
            i = save;
            if (a.b.data) q35_buf_free(&a.b);
            free(out->name);
            out->name = NULL;
            return 0;
        }
        i += plen;
        size_t kstart = i;
        while (i < len && p[i] != '>') i++;
        if (i >= len) {
            if (a.b.data) q35_buf_free(&a.b);
            free(out->name);
            out->name = NULL;
            return 0;
        }
        char *key = (char *)malloc(i - kstart + 1);
        memcpy(key, p + kstart, i - kstart);
        key[i - kstart] = '\0';
        i++;
        if (i < len && p[i] == '\n') i++;
        // Value runs until "\n</parameter>".
        const char *end = NULL;
        for (size_t k = i; k + 1 + pclen <= len; k++) {
            if (p[k] == '\n' && memcmp(p + k + 1, pclose, pclen) == 0) {
                end = p + k;
                break;
            }
        }
        if (!end) {
            free(key);
            if (a.b.data) q35_buf_free(&a.b);
            free(out->name);
            out->name = NULL;
            return 0;
        }
        size_t vlen = (size_t)(end - (p + i));
        args_add(&a, key, p + i, vlen, tools, out->name);
        free(key);
        i = (size_t)(end - p) + 1 + pclen;
        // Skip the newline after </parameter> when present.
        if (i < len && p[i] == '\n') i++;
    }
    if (a.b.data) {
        q35_buf_append_str(&a.b, "}");
        out->arguments = a.b.data;
    } else {
        out->arguments = strdup("{}");
    }

    // optional newline between </function> and </tool_call>
    while (i < len && (p[i] == '\n' || p[i] == '\r' || p[i] == ' ' || p[i] == '\t')) i++;
    const char *fclose_tc = "</tool_call>";
    const size_t tclen = strlen(fclose_tc);
    if (i + tclen > len || memcmp(p + i, fclose_tc, tclen) != 0) {
        free(out->arguments);
        out->arguments = NULL;
        free(out->name);
        out->name = NULL;
        return 0;
    }
    i += tclen;
    return i;
}

static void out_push(q35_out *o, q35_tc tc) {
    o->calls = (q35_tc *)realloc(o->calls, sizeof(q35_tc) * (size_t)(o->n_calls + 1));
    o->calls[o->n_calls++] = tc;
}

void q35_parse_output(const char *raw, size_t len, const qj_val *tools,
                      int thinking_enabled, int extract_reasoning, q35_out *out) {
    memset(out, 0, sizeof(*out));
    const char *tool_marker = "<tool_call>";
    const char *close_think = "</think>";
    const char *open_think = "<think>";
    size_t rest_off = 0;
    q35_buf reasoning;
    q35_buf_init(&reasoning);

    if (extract_reasoning) {
        const char *tc = find_str(raw, len, tool_marker);
        const char *ct = find_str(raw, len, close_think);
        size_t reason_end = 0;
        int have_reason = 0;
        if (ct && (!tc || ct < tc)) {
            reason_end = (size_t)(ct - raw);
            rest_off = reason_end + strlen(close_think);
            have_reason = 1;
        } else if (tc) {
            reason_end = (size_t)(tc - raw);
            rest_off = reason_end;
            have_reason = 1;
        } else if (thinking_enabled) {
            reason_end = len;
            rest_off = len;
            have_reason = 1;
        }
        if (have_reason) {
            size_t s = 0;
            if (reason_end >= strlen(open_think) && memcmp(raw, open_think, strlen(open_think)) == 0)
                s = strlen(open_think);
            while (s < reason_end && is_ws(raw[s])) s++;
            size_t e = reason_end;
            // The "\n</think>" terminator includes its newline; "<tool_call>" does not,
            // so llama.cpp keeps a newline that precedes a tool call.
            if (ct && (!tc || ct < tc) && e > s && raw[e - 1] == '\n') e--;
            q35_buf_append(&reasoning, raw + s, e - s);
        }
        // Drop the newline(s) the template puts after </think>.
        while (rest_off < len && raw[rest_off] == '\n') rest_off++;
    }
    out->reasoning = reasoning.data ? reasoning.data : strdup("");

    const char *rest = raw + rest_off;
    size_t rest_len = len - rest_off;
    const int tools_present = tools && tools->type == QJ_ARR && tools->n > 0;
    const char *tc = tools_present ? find_str(rest, rest_len, tool_marker) : NULL;
    if (!tc) {
        out->content = (char *)malloc(rest_len + 1);
        memcpy(out->content, rest, rest_len);
        out->content[rest_len] = '\0';
        return;
    }
    size_t content_len = (size_t)(tc - rest);
    while (content_len > 0 && rest[content_len - 1] == '\n') content_len--;
    out->content = (char *)malloc(content_len + 1);
    memcpy(out->content, rest, content_len);
    out->content[content_len] = '\0';

    size_t off = (size_t)(tc - rest);
    while (off < rest_len) {
        // skip whitespace between calls
        while (off < rest_len && is_ws(rest[off])) off++;
        if (off >= rest_len) break;
        q35_tc block;
        size_t used = q35_parse_tool_block(rest + off, rest_len - off, tools, &block);
        if (used == 0) break;
        out_push(out, block);
        off += used;
    }
}

void q35_out_free(q35_out *o) {
    if (!o) return;
    free(o->reasoning);
    free(o->content);
    for (int i = 0; i < o->n_calls; i++) {
        free(o->calls[i].name);
        free(o->calls[i].arguments);
    }
    free(o->calls);
    memset(o, 0, sizeof(*o));
}
