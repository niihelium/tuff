// tuff-server: OpenAI-compatible HTTP API for the Tuff inference engine.
//
// A self-contained HTTP/1.1 server written in C11 using only POSIX sockets and
// pthreads (no external HTTP library). It drives the engine through the same
// plain decode sequence used by src/main.c: engine_forward_token for both
// prefill and generation. Because the engine owns global GPU state and a pinned
// CPU worker pool, every model invocation is serialized under a single mutex;
// the accept loop is threaded (one detached thread per connection), but only
// one request ever touches the engine at a time.
//
// Endpoints:
//   GET  /health                 -> {"status":"ok"}
//   GET  /v1/models              -> OpenAI model list
//   POST /v1/completions         -> text completion (stream via "stream":true)
//   POST /v1/chat/completions    -> ChatML chat completion (stream via "stream":true)
//
// Sampling is greedy-only: temperature / top_p are accepted and ignored.

#define _GNU_SOURCE

#include "model.h"
#include "engine.h"
#include "gpu.h"
#include "tokenizer.h"
#include "tuff_exact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <sys/resource.h>

// ---------------------------------------------------------------------------
// P0 shims: tuff_exact() and the final-logit capture sink are normally
// defined in src/main.c, which this target does not link. Provide the same
// semantics locally so engine.c / cpu_ops.c / gpu.cu resolve (the tuff_exact
// pattern mirrors src/cpu_expert_bench.c).
// ---------------------------------------------------------------------------

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

int tuff_logit_capture_enable(int on) { (void)on; return 0; }
int tuff_logit_capture_is_enabled(void) { return 0; }
void tuff_logit_capture(const float *logits, uint32_t n) { (void)logits; (void)n; }
const float *tuff_logit_capture_take(uint32_t *n_out) { if (n_out) *n_out = 0; return NULL; }

// ---------------------------------------------------------------------------
// Dynamic string buffer
// ---------------------------------------------------------------------------

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} buf_t;

static void buf_init(buf_t *b) { b->data = NULL; b->len = 0; b->cap = 0; }

static void buf_reserve(buf_t *b, size_t extra) {
    size_t need = b->len + extra + 1;
    if (need <= b->cap) return;
    size_t ncap = b->cap ? b->cap : 64;
    while (ncap < need) ncap *= 2;
    char *nd = (char *)realloc(b->data, ncap);
    if (!nd) { fprintf(stderr, "[tuff-server] FATAL: out of memory\n"); exit(EXIT_FAILURE); }
    b->data = nd;
    b->cap = ncap;
}

static void buf_append(buf_t *b, const char *s, size_t n) {
    if (n == 0) return;
    buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void buf_append_str(buf_t *b, const char *s) { buf_append(b, s, strlen(s)); }

static void buf_appendf(buf_t *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) { va_end(ap2); return; }
    buf_reserve(b, (size_t)need);
    vsnprintf(b->data + b->len, (size_t)need + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)need;
}

static void buf_free(buf_t *b) { free(b->data); b->data = NULL; b->len = 0; b->cap = 0; }

// ---------------------------------------------------------------------------
// Minimal JSON
// ---------------------------------------------------------------------------

static void jskip_ws(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) (*p)++;
}

// Parse a JSON string literal; returns a malloc'd decoded string (may be "").
static char *jparse_string(const char **p, const char *end) {
    jskip_ws(p, end);
    if (*p >= end || **p != '"') return NULL;
    (*p)++;
    buf_t b; buf_init(&b);
    const char *q = *p;
    while (q < end && *q != '"') {
        unsigned char c = (unsigned char)*q;
        if (c == '\\') {
            q++;
            if (q >= end) { buf_free(&b); return NULL; }
            switch (*q) {
            case '"':  buf_append(&b, "\"", 1); break;
            case '\\': buf_append(&b, "\\", 1); break;
            case '/':  buf_append(&b, "/", 1); break;
            case 'b':  buf_append(&b, "\b", 1); break;
            case 'f':  buf_append(&b, "\f", 1); break;
            case 'n':  buf_append(&b, "\n", 1); break;
            case 'r':  buf_append(&b, "\r", 1); break;
            case 't':  buf_append(&b, "\t", 1); break;
            case 'u': {
                if (q + 4 >= end) { buf_free(&b); return NULL; }
                unsigned cp = 0;
                for (int i = 0; i < 4; i++) {
                    char h = *++q;
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                    else { buf_free(&b); return NULL; }
                }
                if (cp >= 0xD800 && cp <= 0xDBFF && q + 6 < end && q[1] == '\\' && q[2] == 'u') {
                    unsigned lo = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = q[3 + i];
                        lo <<= 4;
                        if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                        else { lo = 0; break; }
                    }
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                        q += 6;
                    }
                }
                char tmp[4]; int n = 0;
                if (cp < 0x80) { tmp[n++] = (char)cp; }
                else if (cp < 0x800) { tmp[n++] = (char)(0xC0 | (cp >> 6)); tmp[n++] = (char)(0x80 | (cp & 0x3F)); }
                else if (cp < 0x10000) { tmp[n++] = (char)(0xE0 | (cp >> 12)); tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[n++] = (char)(0x80 | (cp & 0x3F)); }
                else { tmp[n++] = (char)(0xF0 | (cp >> 18)); tmp[n++] = (char)(0x80 | ((cp >> 12) & 0x3F)); tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[n++] = (char)(0x80 | (cp & 0x3F)); }
                buf_append(&b, tmp, (size_t)n);
                break;
            }
            default: buf_free(&b); return NULL;
            }
            q++;
        } else if (c < 0x20) {
            buf_free(&b); return NULL;
        } else {
            buf_append(&b, (const char *)q, 1);
            q++;
        }
    }
    if (q >= end) { buf_free(&b); return NULL; }
    *p = q + 1;
    char *out = (char *)malloc(b.len + 1);
    if (out) {
        if (b.len) memcpy(out, b.data, b.len);
        out[b.len] = '\0';
    }
    buf_free(&b);
    return out;
}

static long jparse_int(const char **p, const char *end) {
    jskip_ws(p, end);
    long v = 0; int neg = 0;
    if (*p < end && **p == '-') { neg = 1; (*p)++; }
    while (*p < end && **p >= '0' && **p <= '9') { v = v * 10 + (**p - '0'); (*p)++; }
    return neg ? -v : v;
}

// Skip an arbitrary JSON value (string/number/bool/null/object/array).
static int jskip_value(const char **p, const char *end) {
    jskip_ws(p, end);
    if (*p >= end) return -1;
    char c = **p;
    if (c == '"') { char *s = jparse_string(p, end); free(s); return s ? 0 : -1; }
    if (c == '{' || c == '[') {
        char open = c, close = (c == '{') ? '}' : ']';
        (*p)++;
        int depth = 1;
        while (*p < end && depth > 0) {
            char d = **p;
            if (d == '"') { char *s = jparse_string(p, end); free(s); if (!s) return -1; continue; }
            if (d == open) depth++;
            else if (d == close) depth--;
            (*p)++;
        }
        return depth == 0 ? 0 : -1;
    }
    while (*p < end && **p != ',' && **p != '}' && **p != ']') (*p)++;
    return 0;
}

typedef struct {
    char *role;
    char *content;
} chat_msg_t;

typedef struct {
    chat_msg_t *items;
    int n;
    int cap;
} chat_msgs_t;

static void msgs_add(chat_msgs_t *m, char *role, char *content) {
    if (m->n >= m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->items = (chat_msg_t *)realloc(m->items, (size_t)m->cap * sizeof(chat_msg_t));
    }
    m->items[m->n].role = role;
    m->items[m->n].content = content;
    m->n++;
}

// Parse a "messages" array: [{"role":"...","content":"..."}, ...]
static int jparse_messages(const char **p, const char *end, chat_msgs_t *msgs) {
    jskip_ws(p, end);
    if (*p >= end || **p != '[') return -1;
    (*p)++;
    for (;;) {
        jskip_ws(p, end);
        if (*p >= end) return -1;
        if (**p == ']') { (*p)++; break; }
        if (**p == ',') { (*p)++; continue; }
        if (**p != '{') { if (jskip_value(p, end)) return -1; continue; }
        (*p)++;
        char *role = NULL, *content = NULL;
        for (;;) {
            jskip_ws(p, end);
            if (*p >= end) { free(role); free(content); return -1; }
            if (**p == '}') { (*p)++; break; }
            if (**p == ',') { (*p)++; continue; }
            char *key = jparse_string(p, end);
            if (!key) { free(role); free(content); return -1; }
            jskip_ws(p, end);
            if (*p >= end || **p != ':') { free(key); free(role); free(content); return -1; }
            (*p)++;
            if (strcmp(key, "role") == 0) { free(role); role = jparse_string(p, end); }
            else if (strcmp(key, "content") == 0) {
                jskip_ws(p, end);
                if (*p < end && **p == '"') {
                    free(content);
                    content = jparse_string(p, end);
                } else if (*p < end && **p == '[') {
                    (*p)++;
                    buf_t cb; buf_init(&cb);
                    while (*p < end && **p != ']') {
                        jskip_ws(p, end);
                        if (*p < end && **p == '{') {
                            (*p)++;
                            while (*p < end && **p != '}') {
                                jskip_ws(p, end);
                                if (**p == ',') { (*p)++; continue; }
                                char *ckey = jparse_string(p, end);
                                if (!ckey) break;
                                jskip_ws(p, end);
                                if (*p < end && **p == ':') (*p)++;
                                if (strcmp(ckey, "text") == 0) {
                                    char *txt = jparse_string(p, end);
                                    if (txt) { buf_append_str(&cb, txt); free(txt); }
                                } else {
                                    jskip_value(p, end);
                                }
                                free(ckey);
                            }
                            if (*p < end && **p == '}') (*p)++;
                        } else if (*p < end && **p == ',') {
                            (*p)++;
                        } else {
                            jskip_value(p, end);
                        }
                    }
                    if (*p < end && **p == ']') (*p)++;
                    free(content);
                    content = cb.data ? cb.data : strdup("");
                } else {
                    jskip_value(p, end);
                }
            }
            else if (jskip_value(p, end)) { free(key); free(role); free(content); return -1; }
            free(key);
        }
        msgs_add(msgs, role ? role : strdup(""), content ? content : strdup(""));
    }
    return 0;
}

static double jparse_float(const char **p, const char *end) {
    jskip_ws(p, end);
    char buf[64];
    int n = 0;
    while (*p < end && n < 63 && ((**p >= '0' && **p <= '9') || **p == '.' || **p == '-' || **p == '+' || **p == 'e' || **p == 'E')) {
        buf[n++] = **p;
        (*p)++;
    }
    buf[n] = '\0';
    return (n > 0) ? atof(buf) : 0.0;
}

typedef struct {
    char *model;
    char *prompt;
    int   has_prompt;
    long  max_tokens;   // -1 = not specified
    int   stream;
    char *reasoning_effort; // "low", "medium", "high", "none" (or NULL)
    chat_msgs_t messages;
    sampler_params_t sampler;
} req_t;

static void req_init(req_t *r) {
    memset(r, 0, sizeof(*r));
    r->max_tokens = -1;
    r->sampler = sampler_params_default();
}

static void req_free(req_t *r) {
    free(r->model);
    free(r->prompt);
    free(r->reasoning_effort);
    for (int i = 0; i < r->messages.n; i++) {
        free(r->messages.items[i].role);
        free(r->messages.items[i].content);
    }
    free(r->messages.items);
}

static int parse_request(const char *body, size_t len, req_t *req) {
    const char *p = body, *end = body + len;
    jskip_ws(&p, end);
    if (p >= end || *p != '{') return -1;
    p++;
    for (;;) {
        jskip_ws(&p, end);
        if (p >= end) return -1;
        if (*p == '}') { p++; break; }
        if (*p == ',') { p++; continue; }
        char *key = jparse_string(&p, end);
        if (!key) return -1;
        jskip_ws(&p, end);
        if (p >= end || *p != ':') { free(key); return -1; }
        p++;
        if (strcmp(key, "model") == 0) { req->model = jparse_string(&p, end); }
        else if (strcmp(key, "prompt") == 0) { req->prompt = jparse_string(&p, end); req->has_prompt = 1; }
        else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_completion_tokens") == 0) {
            req->max_tokens = jparse_int(&p, end);
        }
        else if (strcmp(key, "reasoning_effort") == 0) {
            req->reasoning_effort = jparse_string(&p, end);
        }
        else if (strcmp(key, "stream") == 0) {
            jskip_ws(&p, end);
            if (p + 4 <= end && strncmp(p, "true", 4) == 0) { req->stream = 1; p += 4; }
            else if (p + 5 <= end && strncmp(p, "false", 5) == 0) { p += 5; }
            else if (jskip_value(&p, end)) { free(key); return -1; }
        }
        else if (strcmp(key, "temperature") == 0) {
            req->sampler.temp = (float)jparse_float(&p, end);
        }
        else if (strcmp(key, "top_p") == 0) {
            req->sampler.top_p = (float)jparse_float(&p, end);
        }
        else if (strcmp(key, "top_k") == 0) {
            req->sampler.top_k = (int32_t)jparse_int(&p, end);
        }
        else if (strcmp(key, "min_p") == 0) {
            req->sampler.min_p = (float)jparse_float(&p, end);
        }
        else if (strcmp(key, "presence_penalty") == 0 || strcmp(key, "frequency_penalty") == 0 || strcmp(key, "repeat_penalty") == 0) {
            float pen = (float)jparse_float(&p, end);
            if (pen > 0.0f) req->sampler.repeat_penalty = 1.0f + pen;
        }
        else if (strcmp(key, "seed") == 0) {
            req->sampler.seed = (uint64_t)jparse_int(&p, end);
        }
        else if (strcmp(key, "n") == 0 || strcmp(key, "stop") == 0 || strcmp(key, "logprobs") == 0 ||
                 strcmp(key, "user") == 0 || strcmp(key, "stream_options") == 0 || strcmp(key, "response_format") == 0) {
            if (jskip_value(&p, end)) { free(key); return -1; }
        }
        else if (strcmp(key, "messages") == 0) {
            if (jparse_messages(&p, end, &req->messages)) { free(key); return -1; }
        }
        else if (jskip_value(&p, end)) { free(key); return -1; }
        free(key);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// HTTP server plumbing
// ---------------------------------------------------------------------------

typedef struct {
    char    method[16];
    char    path[1024];
    int     content_length;  // -1 = absent
    int     chunked;
    char   *body;
    size_t  body_len;
} http_req_t;

#define MAX_BODY (16u * 1024u * 1024u)

static int send_all(int fd, const void *data, size_t len) {
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

static char *mem_find(const char *hay, size_t haylen, const char *needle, size_t needlelen) {
    if (haylen < needlelen || needlelen == 0) return NULL;
    for (size_t i = 0; i + needlelen <= haylen; i++) {
        if (memcmp(hay + i, needle, needlelen) == 0) return (char *)(hay + i);
    }
    return NULL;
}

typedef struct {
    int fd;
    char buf[4096];
    size_t pos;
    size_t len;
    int eof;
} rdr_t;

static int rdr_getc(rdr_t *r, int *out) {
    if (r->pos >= r->len) {
        ssize_t n = recv(r->fd, r->buf, sizeof(r->buf), 0);
        if (n <= 0) { r->eof = 1; return -1; }
        r->pos = 0;
        r->len = (size_t)n;
    }
    *out = (unsigned char)r->buf[r->pos++];
    return 0;
}

static int read_request(int fd, http_req_t *req) {
    buf_t buf; buf_init(&buf);
    const char *body_start = NULL;
    for (;;) {
        char *p = mem_find(buf.data, buf.len, "\r\n\r\n", 4);
        if (p) { body_start = p + 4; break; }
        if (buf.len > 65536) { buf_free(&buf); return -1; }
        buf_reserve(&buf, 4096);
        ssize_t r = recv(fd, buf.data + buf.len, 4096, 0);
        if (r <= 0) { buf_free(&buf); return -1; }
        buf.len += (size_t)r;
    }
    size_t header_len = (size_t)(body_start - buf.data);
    buf.data[header_len - 4] = '\0';   // NUL-terminate the header text

    // Request line
    char *line = buf.data;
    char *sp = strchr(line, ' ');
    if (!sp) { buf_free(&buf); return -1; }
    size_t mlen = (size_t)(sp - line);
    if (mlen >= sizeof(req->method)) mlen = sizeof(req->method) - 1;
    memcpy(req->method, line, mlen);
    req->method[mlen] = '\0';

    char *path = sp + 1;
    while (*path == ' ') path++;
    char *sp2 = strchr(path, ' ');
    if (!sp2) { buf_free(&buf); return -1; }
    size_t plen = (size_t)(sp2 - path);
    if (plen >= sizeof(req->path)) plen = sizeof(req->path) - 1;
    memcpy(req->path, path, plen);
    req->path[plen] = '\0';

    // Headers. The header text is NUL-terminated at header_len-4, which is the first
    // byte of the "\r\n\r\n" terminator. The last header line therefore has no trailing
    // '\n' inside the NUL-terminated region, so it is handled by the end-of-string branch.
    req->content_length = -1;
    req->chunked = 0;
    char *hdr = strchr(sp2, '\n');
    hdr = hdr ? hdr + 1 : NULL;
    while (hdr && *hdr) {
        char *nl = strchr(hdr, '\n');
        char *line_end;
        if (nl) { *nl = '\0'; line_end = nl; }
        else { line_end = hdr + strlen(hdr); }
        char *e = line_end;
        while (e > hdr && (e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = '\0';
        if (*hdr) {
            char *colon = strchr(hdr, ':');
            if (colon) {
                *colon = '\0';
                char *value = colon + 1;
                while (*value == ' ' || *value == '\t') value++;
                if (strcasecmp(hdr, "Content-Length") == 0) req->content_length = atoi(value);
                else if (strcasecmp(hdr, "Transfer-Encoding") == 0 && strcasestr(value, "chunked")) req->chunked = 1;
            }
        }
        hdr = nl ? nl + 1 : NULL;
    }

    size_t already = buf.len - header_len;

    if (req->chunked) {
        rdr_t r;
        r.fd = fd; r.pos = 0; r.len = 0; r.eof = 0;
        size_t seed = already;
        if (seed > sizeof(r.buf)) seed = sizeof(r.buf);
        if (seed > 0) { memcpy(r.buf, buf.data + header_len, seed); r.len = seed; }

        buf_t body; buf_init(&body);
        for (;;) {
            char szline[64]; int n = 0, c;
            while (n < 63 && rdr_getc(&r, &c) == 0 && c != '\n') {
                if (c != '\r') szline[n++] = (char)c;
            }
            if (r.eof) { buf_free(&body); buf_free(&buf); return -1; }
            szline[n] = '\0';
            char *semi = strchr(szline, ';');
            if (semi) *semi = '\0';
            unsigned long cs = strtoul(szline, NULL, 16);
            if (cs == 0) break;
            if (cs > MAX_BODY || body.len + cs > MAX_BODY) { buf_free(&body); buf_free(&buf); return -1; }
            buf_reserve(&body, (size_t)cs + 1);
            for (unsigned long i = 0; i < cs; i++) {
                if (rdr_getc(&r, &c) != 0) { buf_free(&body); buf_free(&buf); return -1; }
                body.data[body.len++] = (char)c;
            }
            rdr_getc(&r, &c);
            rdr_getc(&r, &c);
        }
        body.data[body.len] = '\0';
        req->body = body.data;
        req->body_len = body.len;
        buf_free(&buf);
        return 0;
    }

    if (req->content_length > 0) {
        size_t cl = (size_t)req->content_length;
        if (cl > MAX_BODY) { buf_free(&buf); return -1; }
        req->body = (char *)malloc(cl + 1);
        size_t have = already < cl ? already : cl;
        if (have) memcpy(req->body, buf.data + header_len, have);
        while (have < cl) {
            ssize_t r = recv(fd, req->body + have, cl - have, 0);
            if (r <= 0) { free(req->body); req->body = NULL; buf_free(&buf); return -1; }
            have += (size_t)r;
        }
        req->body[cl] = '\0';
        req->body_len = cl;
    } else {
        req->body = NULL;
        req->body_len = 0;
    }

    buf_free(&buf);
    return 0;
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
        "Connection: close\r\n"
        "\r\n", status, status_text, ctype, body_len);
    send_all(fd, hdr, (size_t)n);
    if (body_len) send_all(fd, body, body_len);
}

static void respond_json(int fd, int status, const char *status_text, const void *body, size_t body_len) {
    respond_raw(fd, status, status_text, "application/json", body, body_len);
}

static void respond_stream_start(int fd, const char *ctype) {
    char h[512];
    int n = snprintf(h, sizeof(h),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Connection: close\r\n"
        "\r\n", ctype);
    send_all(fd, h, (size_t)n);
}

static void handle_options(int fd) {
    const char *resp = "HTTP/1.1 204 No Content\r\n"
                       "Access-Control-Allow-Origin: *\r\n"
                       "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                       "Access-Control-Allow-Headers: *\r\n"
                       "Content-Length: 0\r\n"
                       "Connection: close\r\n\r\n";
    send_all(fd, resp, strlen(resp));
}

// Forward declaration (defined in the "JSON escaping" section below).
static void json_escape_append(buf_t *b, const char *s);

static void respond_json_error(int fd, int status, const char *msg) {
    buf_t b; buf_init(&b);
    buf_append_str(&b, "{\"error\":{\"message\":");
    json_escape_append(&b, msg);
    buf_append_str(&b, ",\"type\":\"invalid_request_error\"}}");
    respond_json(fd, status, status == 400 ? "Bad Request" : "Internal Server Error", b.data, b.len);
    buf_free(&b);
}

// ---------------------------------------------------------------------------
// JSON escaping (append a quoted, escaped string)
// ---------------------------------------------------------------------------

static void json_escape_append(buf_t *b, const char *s) {
    buf_append(b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '"':  buf_append(b, "\\\"", 2); break;
        case '\\': buf_append(b, "\\\\", 2); break;
        case '\n': buf_append(b, "\\n", 2); break;
        case '\r': buf_append(b, "\\r", 2); break;
        case '\t': buf_append(b, "\\t", 2); break;
        case '\b': buf_append(b, "\\b", 2); break;
        case '\f': buf_append(b, "\\f", 2); break;
        default:
            if (c < 0x20) { char t[8]; snprintf(t, sizeof(t), "\\u%04x", c); buf_append(b, t, 6); }
            else buf_append(b, (const char *)p, 1);
        }
    }
    buf_append(b, "\"", 1);
}

// ---------------------------------------------------------------------------
// Engine driving (always under the global mutex)
// ---------------------------------------------------------------------------

#define DEFAULT_SYSTEM "You are a helpful, respectful and honest AI assistant."

typedef struct {
    model_t       *model;
    tokenizer_t   *tok;
    engine_t      *eng;
    pthread_mutex_t lock;
    int             max_seq;
    int             max_gen;
    char           *model_name;
    const char     *system_prompt;   // may be NULL
} server_t;

static int is_stop_token(const tokenizer_t *tok, int32_t token_id) {
    if (!tok || token_id < 0 || token_id >= (int32_t)tok->vocab_size) return 1;
    if (token_id == tok->eos_id) return 1;
    const char *piece = tokenizer_decode(tok, token_id);
    if (!piece) return 1;
    if (strcmp(piece, "<|im_end|>") == 0 ||
        strcmp(piece, "<|endoftext|>") == 0 ||
        strcmp(piece, "<|im_start|>") == 0) {
        return 1;
    }
    return 0;
}

static void send_sse_chunk(int fd, const char *id, int64_t created, const char *model,
                           const char *object, int chat,
                           const char *content, const char *reasoning_content,
                           const char *finish_reason) {
    buf_t b; buf_init(&b);
    buf_append_str(&b, "data: {\"id\":");
    json_escape_append(&b, id);
    buf_append_str(&b, ",\"object\":");
    json_escape_append(&b, object);
    buf_appendf(&b, ",\"created\":%lld,\"model\":", (long long)created);
    json_escape_append(&b, model);
    buf_append_str(&b, ",\"choices\":[{\"index\":0");
    if (chat) {
        buf_append_str(&b, ",\"delta\":{");
        int has_field = 0;
        if (content) {
            buf_append_str(&b, "\"content\":");
            json_escape_append(&b, content);
            has_field = 1;
        }
        if (reasoning_content) {
            if (has_field) buf_append_str(&b, ",");
            buf_append_str(&b, "\"reasoning_content\":");
            json_escape_append(&b, reasoning_content);
        }
        buf_append_str(&b, "}");
    } else {
        buf_append_str(&b, ",\"text\":");
        json_escape_append(&b, content ? content : "");
    }
    if (finish_reason) { buf_append_str(&b, ",\"finish_reason\":"); json_escape_append(&b, finish_reason); }
    else { buf_append_str(&b, ",\"finish_reason\":null"); }
    buf_append_str(&b, "}]}\n\n");
    send_all(fd, b.data, b.len);
    buf_free(&b);
}

// Returns the number of bytes from start of `s` (len `len`) that form strictly complete UTF-8 characters.
// Any trailing incomplete multi-byte sequence is not counted.
static size_t valid_utf8_prefix_len(const unsigned char *s, size_t len) {
    if (len == 0) return 0;
    // Walk back to find the last leading byte
    size_t i = len;
    while (i > 0) {
        unsigned char c = s[i - 1];
        if ((c & 0x80) == 0) {
            // ASCII byte at end -> everything up to len is complete
            return len;
        }
        if ((c & 0xC0) == 0xC0) {
            // Found leading byte at index (i - 1)
            size_t lead_idx = i - 1;
            size_t seq_len = 1;
            if ((c & 0xE0) == 0xC0) seq_len = 2;
            else if ((c & 0xF0) == 0xE0) seq_len = 3;
            else if ((c & 0xF8) == 0xF0) seq_len = 4;

            size_t avail = len - lead_idx;
            if (avail >= seq_len) return len; // sequence complete
            else return lead_idx;            // incomplete sequence starts here
        }
        // Continuation byte (0x80..0xBF)
        i--;
        if (len - i > 4) break; // UTF-8 sequence cannot exceed 4 bytes
    }
    return len;
}

typedef struct {
    int n_prompt;
    int n_gen;
    int n_thought;
    double prompt_ms;
    double prompt_per_tok_ms;
    double prompt_tps;
    double gen_ms;
    double gen_per_tok_ms;
    double gen_tps;
} completion_stats_t;

static double get_time_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void append_usage_and_timings(buf_t *b, const completion_stats_t *st) {
    buf_appendf(b, ",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
                   "\"prompt_tokens_details\":{\"cached_tokens\":0},"
                   "\"completion_tokens_details\":{\"reasoning_tokens\":%d}}",
                st->n_prompt, st->n_gen, st->n_prompt + st->n_gen, st->n_thought);
    buf_appendf(b, ",\"timings\":{\"cache_n\":0,\"prompt_n\":%d,\"prompt_ms\":%.2f,"
                   "\"prompt_per_token_ms\":%.3f,\"prompt_per_second\":%.2f,"
                   "\"predicted_n\":%d,\"predicted_ms\":%.2f,"
                   "\"predicted_per_token_ms\":%.3f,\"predicted_per_second\":%.2f}",
                st->n_prompt, st->prompt_ms, st->prompt_per_tok_ms, st->prompt_tps,
                st->n_gen, st->gen_ms, st->gen_per_tok_ms, st->gen_tps);
}

// Prefill + autoregressive generation. Caller must hold server->lock.
static void generate_impl(server_t *s, const char *prompt, int max_gen,
                          const sampler_params_t *sampler_params,
                          int stream, int fd, const char *id, int64_t created,
                          const char *model, const char *object, int chat,
                          buf_t *thought_out, buf_t *reply, int *fin_out, completion_stats_t *stats_out) {
    int max_seq = s->max_seq;
    int32_t *tokens = (int32_t *)malloc((size_t)max_seq * sizeof(int32_t));
    int n_prompt = tokenizer_tokenize(s->tok, prompt, tokens, max_seq);
    if (n_prompt <= 0) { tokens[0] = s->tok->bos_id; n_prompt = 1; }

    engine_reset_context(s->eng);
    if (sampler_params) {
        engine_set_sampler(s->eng, sampler_params);
    } else {
        sampler_params_t def_s = sampler_params_default();
        engine_set_sampler(s->eng, &def_s);
    }

    double t_start = get_time_s();

    int pos = 0;
    int32_t curr = tokens[0];
    for (int i = 0; i < n_prompt; i++) {
        curr = engine_forward_token(s->eng, tokens[i], pos++);
    }

    double t_prefill = get_time_s();

    int n_gen = 0;
    int n_thought = 0;
    int stop_hit = 0;
    int in_think = 0;

    buf_t stream_thought_buf; buf_init(&stream_thought_buf);
    buf_t stream_content_buf; buf_init(&stream_content_buf);

    while (n_gen < max_gen && pos < max_seq) {
        if (is_stop_token(s->tok, curr)) { stop_hit = 1; break; }
        const char *piece = tokenizer_decode(s->tok, curr);
        if (piece && *piece) {
            if (chat) {
                if (!in_think && strstr(piece, "<think>") != NULL) {
                    in_think = 1;
                    n_thought++;
                    const char *p = strstr(piece, "<think>") + 7;
                    if (*p == '\n') p++;
                    if (*p) {
                        if (stream) buf_append_str(&stream_thought_buf, p);
                        if (thought_out) buf_append_str(thought_out, p);
                    }
                } else if (in_think && strstr(piece, "</think>") != NULL) {
                    // Flush any pending thought tokens before exiting think mode
                    if (stream && stream_thought_buf.len > 0) {
                        send_sse_chunk(fd, id, created, model, object, 1, NULL, stream_thought_buf.data, NULL);
                        stream_thought_buf.len = 0;
                    }
                    in_think = 0;
                    const char *p = strstr(piece, "</think>") + 8;
                    if (*p == '\n') p++;
                    if (*p) {
                        if (stream) buf_append_str(&stream_content_buf, p);
                        buf_append_str(reply, p);
                    }
                } else if (in_think) {
                    n_thought++;
                    if (stream) buf_append_str(&stream_thought_buf, piece);
                    if (thought_out) buf_append_str(thought_out, piece);
                } else {
                    if (stream) buf_append_str(&stream_content_buf, piece);
                    buf_append_str(reply, piece);
                }
            } else {
                if (stream) buf_append_str(&stream_content_buf, piece);
                buf_append_str(reply, piece);
            }

            // Flush complete UTF-8 sequences over SSE
            if (stream) {
                if (stream_thought_buf.len > 0) {
                    size_t vlen = valid_utf8_prefix_len((const unsigned char*)stream_thought_buf.data, stream_thought_buf.len);
                    if (vlen > 0) {
                        char save = stream_thought_buf.data[vlen];
                        stream_thought_buf.data[vlen] = '\0';
                        send_sse_chunk(fd, id, created, model, object, 1, NULL, stream_thought_buf.data, NULL);
                        stream_thought_buf.data[vlen] = save;
                        memmove(stream_thought_buf.data, stream_thought_buf.data + vlen, stream_thought_buf.len - vlen);
                        stream_thought_buf.len -= vlen;
                        if (stream_thought_buf.data) stream_thought_buf.data[stream_thought_buf.len] = '\0';
                    }
                }
                if (stream_content_buf.len > 0) {
                    size_t vlen = valid_utf8_prefix_len((const unsigned char*)stream_content_buf.data, stream_content_buf.len);
                    if (vlen > 0) {
                        char save = stream_content_buf.data[vlen];
                        stream_content_buf.data[vlen] = '\0';
                        send_sse_chunk(fd, id, created, model, object, chat ? 1 : 0, stream_content_buf.data, NULL, NULL);
                        stream_content_buf.data[vlen] = save;
                        memmove(stream_content_buf.data, stream_content_buf.data + vlen, stream_content_buf.len - vlen);
                        stream_content_buf.len -= vlen;
                        if (stream_content_buf.data) stream_content_buf.data[stream_content_buf.len] = '\0';
                    }
                }
            }
        }
        curr = engine_forward_token(s->eng, curr, pos++);
        n_gen++;
    }

    double t_end = get_time_s();

    completion_stats_t st;
    memset(&st, 0, sizeof(st));
    st.n_prompt = n_prompt;
    st.n_gen = n_gen;
    st.n_thought = n_thought;
    st.prompt_ms = (t_prefill - t_start) * 1000.0;
    if (st.prompt_ms < 0.001) st.prompt_ms = 0.001;
    st.prompt_per_tok_ms = st.prompt_ms / (double)n_prompt;
    st.prompt_tps = (double)n_prompt / ((t_prefill - t_start) > 0 ? (t_prefill - t_start) : 0.000001);

    st.gen_ms = (t_end - t_prefill) * 1000.0;
    if (st.gen_ms < 0.001) st.gen_ms = 0.001;
    st.gen_per_tok_ms = (n_gen > 0) ? (st.gen_ms / (double)n_gen) : 0.0;
    st.gen_tps = (n_gen > 0) ? ((double)n_gen / ((t_end - t_prefill) > 0 ? (t_end - t_prefill) : 0.000001)) : 0.0;

    if (stream) {
        // Flush any remaining trailing bytes
        if (stream_thought_buf.len > 0) {
            send_sse_chunk(fd, id, created, model, object, 1, NULL, stream_thought_buf.data, NULL);
        }
        if (stream_content_buf.len > 0) {
            send_sse_chunk(fd, id, created, model, object, chat ? 1 : 0, stream_content_buf.data, NULL, NULL);
        }
        send_sse_chunk(fd, id, created, model, object, chat, NULL, NULL, stop_hit ? "stop" : "length");

        // Send streaming usage and timings chunk (OpenAI & llama.cpp specification)
        buf_t stat_chunk; buf_init(&stat_chunk);
        buf_append_str(&stat_chunk, "data: {\"id\":");
        json_escape_append(&stat_chunk, id);
        buf_append_str(&stat_chunk, ",\"object\":");
        json_escape_append(&stat_chunk, object);
        buf_appendf(&stat_chunk, ",\"created\":%lld,\"model\":", (long long)created);
        json_escape_append(&stat_chunk, model);
        buf_append_str(&stat_chunk, ",\"choices\":[]");
        append_usage_and_timings(&stat_chunk, &st);
        buf_append_str(&stat_chunk, "}\n\n");
        send_all(fd, stat_chunk.data, stat_chunk.len);
        buf_free(&stat_chunk);

        send_all(fd, "data: [DONE]\n\n", 15);
    }

    buf_free(&stream_thought_buf);
    buf_free(&stream_content_buf);
    free(tokens);

    if (fin_out) *fin_out = stop_hit ? 0 : 1;
    if (stats_out) *stats_out = st;
}

static int resolve_max_gen(long req_max, int server_max) {
    int m = (req_max < 0) ? server_max : (int)req_max;
    if (m < 1) m = 1;
    if (m > server_max) m = server_max;
    return m;
}

static pthread_mutex_t g_id_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long g_req_counter = 0;

static char *make_id(const char *prefix) {
    pthread_mutex_lock(&g_id_lock);
    unsigned long long c = g_req_counter++;
    pthread_mutex_unlock(&g_id_lock);
    char *buf = (char *)malloc(64);
    snprintf(buf, 64, "%s%lld%llu", prefix, (long long)time(NULL), c);
    return buf;
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

static void handle_health(int fd) {
    respond_json(fd, 200, "OK", "{\"status\":\"ok\"}", 15);
}

static void handle_models(server_t *s, int fd) {
    buf_t b; buf_init(&b);
    int64_t created = (int64_t)time(NULL);
    buf_append_str(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
    json_escape_append(&b, s->model_name);
    buf_appendf(&b, ",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"tuff\","
                    "\"meta\":{\"vocab_type\":2,\"n_vocab\":%u,\"n_ctx_train\":%u,\"n_embd\":%u,"
                    "\"max_context_length\":%d,\"context_window\":%d,\"max_tokens\":%d}}]}",
                (long long)created,
                s->model->vocab_size,
                s->model->context_length ? s->model->context_length : (uint32_t)s->max_seq,
                (uint32_t)s->model->embedding_length,
                s->max_seq, s->max_seq, s->max_seq);
    respond_json(fd, 200, "OK", b.data, b.len);
    buf_free(&b);
}

static void handle_completion(server_t *s, int fd, http_req_t *req) {
    req_t r; req_init(&r);
    if (parse_request(req->body ? req->body : "", req->body_len, &r) != 0 || !r.prompt) {
        req_free(&r);
        respond_json_error(fd, 400, "Invalid request: expected JSON with a string \"prompt\" field");
        return;
    }
    const char *model = (r.model && r.model[0]) ? r.model : s->model_name;
    int max_gen = resolve_max_gen(r.max_tokens, s->max_gen);
    char *id = make_id("cmpl-");
    int64_t created = (int64_t)time(NULL);

    buf_t reply; buf_init(&reply);
    int fin = 0;
    completion_stats_t st;

    if (r.stream) respond_stream_start(fd, "text/event-stream");

    pthread_mutex_lock(&s->lock);
    generate_impl(s, r.prompt, max_gen, &r.sampler, r.stream, fd, id, created, model,
                  "text_completion", 0, NULL, &reply, &fin, &st);
    pthread_mutex_unlock(&s->lock);

    if (!r.stream) {
        buf_t out; buf_init(&out);
        buf_append_str(&out, "{\"id\":");
        json_escape_append(&out, id);
        buf_append_str(&out, ",\"object\":\"text_completion\",\"created\":");
        buf_appendf(&out, "%lld", (long long)created);
        buf_append_str(&out, ",\"model\":");
        json_escape_append(&out, model);
        buf_append_str(&out, ",\"choices\":[{\"text\":");
        json_escape_append(&out, reply.data ? reply.data : "");
        buf_append_str(&out, ",\"index\":0,\"finish_reason\":");
        json_escape_append(&out, fin ? "length" : "stop");
        buf_append_str(&out, ",\"logprobs\":null}]");
        append_usage_and_timings(&out, &st);
        buf_append_str(&out, "}");
        respond_json(fd, 200, "OK", out.data, out.len);
        buf_free(&out);
    }

    buf_free(&reply);
    free(id);
    req_free(&r);
}

static void handle_chat(server_t *s, int fd, http_req_t *req) {
    req_t r; req_init(&r);
    if (parse_request(req->body ? req->body : "", req->body_len, &r) != 0 || r.messages.n == 0) {
        req_free(&r);
        respond_json_error(fd, 400, "Invalid request: expected JSON with a non-empty \"messages\" array");
        return;
    }

    const char *reasoning_instruction = "";
    if (r.reasoning_effort) {
        if (strcmp(r.reasoning_effort, "none") == 0 || strcmp(r.reasoning_effort, "off") == 0 || strcmp(r.reasoning_effort, "0") == 0) {
            reasoning_instruction = "\nRespond directly to the user without any thinking process, reasoning, or <think> tags.";
        } else if (strcmp(r.reasoning_effort, "low") == 0) {
            reasoning_instruction = "\nKeep your reasoning process very brief and concise before answering.";
        } else if (strcmp(r.reasoning_effort, "high") == 0) {
            reasoning_instruction = "\nProvide a detailed, thorough step-by-step reasoning process before answering.";
        }
    }

    buf_t prompt; buf_init(&prompt);
    int has_system = 0;
    for (int i = 0; i < r.messages.n; i++) {
        if (strcmp(r.messages.items[i].role, "system") == 0) { has_system = 1; break; }
    }
    if (!has_system) {
        const char *sys = (s->system_prompt && s->system_prompt[0]) ? s->system_prompt : DEFAULT_SYSTEM;
        buf_appendf(&prompt, "<|im_start|>system\n%s%s<|im_end|>\n", sys, reasoning_instruction);
    }
    for (int i = 0; i < r.messages.n; i++) {
        const char *role = r.messages.items[i].role;
        const char *content = r.messages.items[i].content;
        if (strcmp(role, "system") == 0)      buf_appendf(&prompt, "<|im_start|>system\n%s%s<|im_end|>\n", content, reasoning_instruction);
        else if (strcmp(role, "user") == 0)   buf_appendf(&prompt, "<|im_start|>user\n%s<|im_end|>\n", content);
        else if (strcmp(role, "assistant") == 0) buf_appendf(&prompt, "<|im_start|>assistant\n%s<|im_end|>\n", content);
    }
    buf_append_str(&prompt, "<|im_start|>assistant\n");

    const char *model = (r.model && r.model[0]) ? r.model : s->model_name;
    int max_gen = resolve_max_gen(r.max_tokens, s->max_gen);
    char *id = make_id("chatcmpl-");
    int64_t created = (int64_t)time(NULL);

    buf_t reply; buf_init(&reply);
    buf_t thought; buf_init(&thought);
    int fin = 0;
    completion_stats_t st;

    if (r.stream) respond_stream_start(fd, "text/event-stream");

    pthread_mutex_lock(&s->lock);
    generate_impl(s, prompt.data, max_gen, &r.sampler, r.stream, fd, id, created, model,
                  r.stream ? "chat.completion.chunk" : "chat.completion",
                  1, &thought, &reply, &fin, &st);
    pthread_mutex_unlock(&s->lock);

    if (!r.stream) {
        buf_t out; buf_init(&out);
        buf_append_str(&out, "{\"id\":");
        json_escape_append(&out, id);
        buf_append_str(&out, ",\"object\":\"chat.completion\",\"created\":");
        buf_appendf(&out, "%lld", (long long)created);
        buf_append_str(&out, ",\"model\":");
        json_escape_append(&out, model);
        buf_append_str(&out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"");
        if (thought.len > 0) {
            buf_append_str(&out, ",\"reasoning_content\":");
            json_escape_append(&out, thought.data ? thought.data : "");
        }
        buf_append_str(&out, ",\"content\":");
        json_escape_append(&out, reply.data ? reply.data : "");
        buf_append_str(&out, "},\"finish_reason\":");
        json_escape_append(&out, fin ? "length" : "stop");
        buf_append_str(&out, "}]");
        append_usage_and_timings(&out, &st);
        buf_append_str(&out, "}");
        respond_json(fd, 200, "OK", out.data, out.len);
        buf_free(&out);
    }

    buf_free(&prompt);
    buf_free(&thought);
    buf_free(&reply);
    free(id);
    req_free(&r);
}

typedef struct {
    int fd;
    server_t *srv;
} conn_arg_t;

static void *handle_connection(void *arg) {
    conn_arg_t *ca = (conn_arg_t *)arg;
    int fd = ca->fd;
    server_t *s = ca->srv;
    free(ca);

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    http_req_t req;
    memset(&req, 0, sizeof(req));
    req.content_length = -1;

    if (read_request(fd, &req) != 0) {
        free(req.body);
        close(fd);
        return NULL;
    }

    if (strcmp(req.method, "OPTIONS") == 0) {
        handle_options(fd);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        handle_health(fd);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_models(s, fd);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/completions") == 0) {
        handle_completion(s, fd, &req);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/chat/completions") == 0) {
        handle_chat(s, fd, &req);
    } else {
        respond_json_error(fd, 404, "Not found");
    }

    free(req.body);
    close(fd);
    return NULL;
}

// ---------------------------------------------------------------------------
// Helpers: RSS / model name / usage
// ---------------------------------------------------------------------------

static double vmrss_mib(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1.0;
    char line[256];
    double vmrss = -1.0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmRSS: %lf kB", &vmrss) == 1) { vmrss /= 1024.0; break; }
    }
    fclose(f);
    return vmrss;
}

static char *model_name_from_path(const char *path) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    char *name = strdup(base);
    char *dot = strrchr(name, '.');
    if (dot && strcmp(dot, ".gguf") == 0) *dot = '\0';
    return name;
}

static int resolve_bind_addr(const char *host, int port, struct sockaddr_in *out) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo *res = NULL;
    if (getaddrinfo(host, portstr, &hints, &res) != 0) return -1;
    if (res) {
        memcpy(out, res->ai_addr, res->ai_addrlen);
        freeaddrinfo(res);
        return 0;
    }
    return -1;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --model <path_to_gguf> [options]\n"
        "\n"
        "OpenAI-compatible HTTP API server for Tuff.\n"
        "\n"
        "Options:\n"
        "  --model <path>             GGUF model file (required)\n"
        "  --port <n>                 listen port (default: 8080)\n"
        "  --host <addr>              bind address (default: 127.0.0.1)\n"
        "  -c, --ctx-size, --max-seq  max context length (default: 4096)\n"
        "  --fill-ram, --fill-vram    inflate context to maximize VRAM utilization\n"
        "  --max-gen <n>              max tokens per response, hard cap (default: 1024)\n"
        "  --system <prompt>          override the default chat system prompt\n"
        "\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  GET  /v1/models\n"
        "  POST /v1/completions        {\"prompt\":\"...\",\"max_tokens\":N,\"stream\":bool}\n"
        "  POST /v1/chat/completions   {\"messages\":[{\"role\":\"user\",\"content\":\"...\"}],...}\n"
        "\n"
        "Sampling is greedy-only; temperature / top_p are accepted and ignored.\n",
        prog);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *host = "127.0.0.1";
    int port = 8080;
    int max_seq = 131072;
    int max_gen = 2048;
    int fill_ram = 0;
    const char *system_prompt = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if ((strcmp(argv[i], "--max-seq") == 0 || strcmp(argv[i], "-c") == 0 ||
                    strcmp(argv[i], "--ctx-size") == 0 || strcmp(argv[i], "--ctx") == 0 ||
                    strcmp(argv[i], "--context-size") == 0) && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fill-ram") == 0 || strcmp(argv[i], "--fill-vram") == 0 ||
                   strcmp(argv[i], "--fill_ram") == 0 || strcmp(argv[i], "--fill_vram") == 0) {
            fill_ram = 1;
        } else if (strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc) {
            max_gen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model <path> is required\n");
        print_usage(argv[0]);
        return 2;
    }
    if (max_seq < 16) { fprintf(stderr, "Error: --max-seq too small\n"); return 2; }
    if (max_gen < 1)  { fprintf(stderr, "Error: --max-gen must be >= 1\n"); return 2; }

    signal(SIGPIPE, SIG_IGN);

    fprintf(stderr, "[tuff-server] TUFF_EXACT=%d (%s)\n", tuff_exact(),
            tuff_exact() ? "bit-exact reference kernels" : "fast paths permitted");
    fprintf(stderr, "[tuff-server] Loading GGUF model from: %s\n", model_path);

    model_t *model = model_load(model_path);
    if (!model) {
        fprintf(stderr, "[tuff-server] Failed to load model from %s\n", model_path);
        return 1;
    }

    {
        double rss_before = vmrss_mib();
        double t0 = (double)clock() / CLOCKS_PER_SEC;
        tuff_gguf_warmup_memory(model->gguf);
        double t1 = (double)clock() / CLOCKS_PER_SEC;
        double rss_after = vmrss_mib();
        fprintf(stderr, "[tuff-server] Page-warmup: %.2f s; VmRSS %.0f MiB -> %.0f MiB\n",
                t1 - t0, rss_before, rss_after);
    }

    gpu_prepare_ex(model, &max_seq, fill_ram);
    tokenizer_t *tok = tokenizer_init(model->gguf);

    server_t s;
    s.model = model;
    s.tok = tok;
    s.max_seq = max_seq;
    s.max_gen = max_gen;
    s.system_prompt = system_prompt;
    s.model_name = model_name_from_path(model_path);
    pthread_mutex_init(&s.lock, NULL);

    s.eng = engine_init(model, max_seq);
    if (!s.eng) {
        fprintf(stderr, "[tuff-server] engine_init failed\n");
        return 1;
    }

    fprintf(stderr, "[tuff-server] model=%s vocab=%u dim=%u layers=%u (attn=%u delta=%u)\n",
            s.model_name, model->vocab_size, model->embedding_length, model->block_count,
            model->n_attn_layers, model->n_delta_layers);
    fprintf(stderr, "[tuff-server] max_seq=%d max_gen=%d\n", max_seq, max_gen);
    fprintf(stderr, "[tuff-server] RSS=%.0f MiB  VRAM=%.1f MiB\n", vmrss_mib(), gpu_vram_used_mb());

    // Bind + listen
    struct sockaddr_in addr;
    if (resolve_bind_addr(host, port, &addr) != 0) {
        fprintf(stderr, "[tuff-server] Failed to resolve host %s\n", host);
        return 1;
    }
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) { perror("[tuff-server] socket"); return 1; }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[tuff-server] bind");
        close(lfd);
        return 1;
    }
    if (listen(lfd, 128) < 0) { perror("[tuff-server] listen"); close(lfd); return 1; }

    fprintf(stderr, "[tuff-server] listening on http://%s:%d  (model=%s, RSS=%.0f MiB, VRAM=%.1f MiB)\n",
            host, port, s.model_name, vmrss_mib(), gpu_vram_used_mb());
    fprintf(stderr, "[tuff-server] endpoints: /health /v1/models /v1/completions /v1/chat/completions\n");

    for (;;) {
        struct sockaddr_in ca;
        socklen_t clen = sizeof(ca);
        int cfd = accept(lfd, (struct sockaddr *)&ca, &clen);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("[tuff-server] accept");
            continue;
        }
        conn_arg_t *arg = (conn_arg_t *)malloc(sizeof(conn_arg_t));
        arg->fd = cfd;
        arg->srv = &s;
        pthread_t th;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&th, &attr, handle_connection, arg) != 0) {
            close(cfd);
            free(arg);
        }
        pthread_attr_destroy(&attr);
    }

    return 0;
}
