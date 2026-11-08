#include "q35_json.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

void q35_buf_init(q35_buf *b) { b->data = NULL; b->len = 0; b->cap = 0; }

void q35_buf_reserve(q35_buf *b, size_t extra) {
    size_t need = b->len + extra + 1;
    if (need <= b->cap) return;
    size_t ncap = b->cap ? b->cap : 64;
    while (ncap < need) ncap *= 2;
    char *nd = (char *)realloc(b->data, ncap);
    if (!nd) {
        fprintf(stderr, "[q35-json] FATAL: out of memory\n");
        exit(EXIT_FAILURE);
    }
    b->data = nd;
    b->cap = ncap;
}

void q35_buf_append(q35_buf *b, const char *s, size_t n) {
    if (n == 0) return;
    q35_buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

void q35_buf_append_str(q35_buf *b, const char *s) { q35_buf_append(b, s, strlen(s)); }

void q35_buf_appendf(q35_buf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) { va_end(ap2); return; }
    q35_buf_reserve(b, (size_t)need);
    vsnprintf(b->data + b->len, (size_t)need + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)need;
}

void q35_buf_free(q35_buf *b) {
    free(b->data);
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

typedef struct {
    const char *p;
    const char *end;
    int depth;
} jp_t;

static void jp_ws(jp_t *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void append_utf8(q35_buf *b, unsigned cp) {
    char t[4];
    int n = 0;
    if (cp < 0x80) t[n++] = (char)cp;
    else if (cp < 0x800) { t[n++] = (char)(0xC0 | (cp >> 6)); t[n++] = (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        t[n++] = (char)(0xE0 | (cp >> 12));
        t[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        t[n++] = (char)(0x80 | (cp & 0x3F));
    } else {
        t[n++] = (char)(0xF0 | (cp >> 18));
        t[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        t[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        t[n++] = (char)(0x80 | (cp & 0x3F));
    }
    q35_buf_append(b, t, (size_t)n);
}

static char *jp_string(jp_t *j) {
    jp_ws(j);
    if (j->p >= j->end || *j->p != '"') return NULL;
    j->p++;
    q35_buf b;
    q35_buf_init(&b);
    while (j->p < j->end && *j->p != '"') {
        unsigned char c = (unsigned char)*j->p;
        if (c < 0x20) { q35_buf_free(&b); return NULL; }
        if (c != '\\') {
            q35_buf_append(&b, (const char *)j->p, 1);
            j->p++;
            continue;
        }
        j->p++;
        if (j->p >= j->end) { q35_buf_free(&b); return NULL; }
        switch (*j->p) {
        case '"': q35_buf_append(&b, "\"", 1); j->p++; break;
        case '\\': q35_buf_append(&b, "\\", 1); j->p++; break;
        case '/': q35_buf_append(&b, "/", 1); j->p++; break;
        case 'b': q35_buf_append(&b, "\b", 1); j->p++; break;
        case 'f': q35_buf_append(&b, "\f", 1); j->p++; break;
        case 'n': q35_buf_append(&b, "\n", 1); j->p++; break;
        case 'r': q35_buf_append(&b, "\r", 1); j->p++; break;
        case 't': q35_buf_append(&b, "\t", 1); j->p++; break;
        case 'u': {
            j->p++;
            if (j->p + 4 > j->end) { q35_buf_free(&b); return NULL; }
            unsigned cp = 0;
            for (int i = 0; i < 4; i++) {
                int h = hexval(j->p[i]);
                if (h < 0) { q35_buf_free(&b); return NULL; }
                cp = (cp << 4) | (unsigned)h;
            }
            j->p += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && j->p + 6 <= j->end && j->p[0] == '\\' && j->p[1] == 'u') {
                unsigned lo = 0;
                int ok = 1;
                for (int i = 0; i < 4; i++) {
                    int h = hexval(j->p[2 + i]);
                    if (h < 0) { ok = 0; break; }
                    lo = (lo << 4) | (unsigned)h;
                }
                if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    j->p += 6;
                }
            }
            append_utf8(&b, cp);
            break;
        }
        default:
            q35_buf_free(&b);
            return NULL;
        }
    }
    if (j->p >= j->end || *j->p != '"') { q35_buf_free(&b); return NULL; }
    j->p++;
    if (!b.data) return strdup("");
    return b.data;
}

static qj_val *jnew(qj_type t) {
    qj_val *v = (qj_val *)calloc(1, sizeof(*v));
    v->type = t;
    return v;
}

static void jpush(qj_val *v, char *key, qj_val *item) {
    if (v->n >= v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = (qj_val **)realloc(v->items, sizeof(qj_val *) * (size_t)v->cap);
        if (v->type == QJ_OBJ) v->keys = (char **)realloc(v->keys, sizeof(char *) * (size_t)v->cap);
    }
    v->items[v->n] = item;
    if (v->type == QJ_OBJ) v->keys[v->n] = key;
    v->n++;
}

static qj_val *jp_value(jp_t *j);

static qj_val *jp_number(jp_t *j) {
    const char *s = j->p;
    if (j->p < j->end && (*j->p == '-' || *j->p == '+')) j->p++;
    int digits = 0;
    while (j->p < j->end && isdigit((unsigned char)*j->p)) { j->p++; digits++; }
    if (!digits) return NULL;
    int is_int = 1;
    if (j->p < j->end && *j->p == '.') {
        is_int = 0;
        j->p++;
        int fd = 0;
        while (j->p < j->end && isdigit((unsigned char)*j->p)) { j->p++; fd++; }
        if (!fd) return NULL;
    }
    if (j->p < j->end && (*j->p == 'e' || *j->p == 'E')) {
        is_int = 0;
        j->p++;
        if (j->p < j->end && (*j->p == '-' || *j->p == '+')) j->p++;
        int ed = 0;
        while (j->p < j->end && isdigit((unsigned char)*j->p)) { j->p++; ed++; }
        if (!ed) return NULL;
    }
    qj_val *v = jnew(QJ_NUM);
    size_t n = (size_t)(j->p - s);
    v->str = (char *)malloc(n + 1);
    memcpy(v->str, s, n);
    v->str[n] = '\0';
    v->num = strtod(v->str, NULL);
    v->is_int = is_int;
    return v;
}

static qj_val *jp_value(jp_t *j) {
    jp_ws(j);
    if (j->p >= j->end) return NULL;
    if (++j->depth > 64) return NULL;
    qj_val *v = NULL;
    char c = *j->p;
    if (c == '{') {
        j->p++;
        v = jnew(QJ_OBJ);
        jp_ws(j);
        if (j->p < j->end && *j->p == '}') { j->p++; j->depth--; return v; }
        for (;;) {
            jp_ws(j);
            char *key = jp_string(j);
            if (!key) { qj_free(v); return NULL; }
            jp_ws(j);
            if (j->p >= j->end || *j->p != ':') { free(key); qj_free(v); return NULL; }
            j->p++;
            qj_val *item = jp_value(j);
            if (!item) { free(key); qj_free(v); return NULL; }
            // Replace an existing key in place, like nlohmann's ordered map.
            int found = -1;
            for (int i = 0; i < v->n; i++) if (strcmp(v->keys[i], key) == 0) { found = i; break; }
            if (found >= 0) {
                free(key);
                qj_free(v->items[found]);
                v->items[found] = item;
            } else {
                jpush(v, key, item);
            }
            jp_ws(j);
            if (j->p >= j->end) { qj_free(v); return NULL; }
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == '}') { j->p++; break; }
            qj_free(v);
            return NULL;
        }
    } else if (c == '[') {
        j->p++;
        v = jnew(QJ_ARR);
        jp_ws(j);
        if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return v; }
        for (;;) {
            qj_val *item = jp_value(j);
            if (!item) { qj_free(v); return NULL; }
            jpush(v, NULL, item);
            jp_ws(j);
            if (j->p >= j->end) { qj_free(v); return NULL; }
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == ']') { j->p++; break; }
            qj_free(v);
            return NULL;
        }
    } else if (c == '"') {
        char *s = jp_string(j);
        if (!s) { j->depth--; return NULL; }
        v = jnew(QJ_STR);
        v->str = s;
    } else if (c == 't' && j->p + 4 <= j->end && memcmp(j->p, "true", 4) == 0) {
        j->p += 4;
        v = jnew(QJ_BOOL);
        v->boolean = 1;
    } else if (c == 'f' && j->p + 5 <= j->end && memcmp(j->p, "false", 5) == 0) {
        j->p += 5;
        v = jnew(QJ_BOOL);
    } else if (c == 'n' && j->p + 4 <= j->end && memcmp(j->p, "null", 4) == 0) {
        j->p += 4;
        v = jnew(QJ_NULL);
    } else if (c == '-' || c == '+' || isdigit((unsigned char)c)) {
        v = jp_number(j);
    }
    j->depth--;
    return v;
}

qj_val *qj_parse(const char *s, size_t len) {
    if (!s) return NULL;
    jp_t j = { s, s + len, 0 };
    qj_val *v = jp_value(&j);
    if (!v) return NULL;
    jp_ws(&j);
    if (j.p != j.end) { qj_free(v); return NULL; }
    return v;
}

void qj_free(qj_val *v) {
    if (!v) return;
    for (int i = 0; i < v->n; i++) {
        if (v->type == QJ_OBJ && v->keys) free(v->keys[i]);
        qj_free(v->items[i]);
    }
    free(v->items);
    free(v->keys);
    free(v->str);
    free(v);
}

qj_val *qj_get(const qj_val *obj, const char *key) {
    if (!obj || obj->type != QJ_OBJ || !key) return NULL;
    for (int i = 0; i < obj->n; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

const char *qj_str(const qj_val *v) { return (v && v->type == QJ_STR) ? v->str : NULL; }

double qj_num(const qj_val *v, double def) {
    if (!v) return def;
    if (v->type == QJ_NUM) return v->num;
    if (v->type == QJ_BOOL) return v->boolean ? 1.0 : 0.0;
    return def;
}

long qj_int(const qj_val *v, long def) {
    if (!v) return def;
    if (v->type == QJ_NUM) return (long)v->num;
    if (v->type == QJ_BOOL) return v->boolean ? 1 : 0;
    return def;
}

int qj_bool(const qj_val *v, int def) {
    if (!v) return def;
    if (v->type == QJ_BOOL) return v->boolean;
    if (v->type == QJ_NUM) return v->num != 0.0;
    if (v->type == QJ_STR) return v->str[0] != '\0';
    return def;
}

int qj_len(const qj_val *v) { return v ? v->n : 0; }

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

// Python repr / minja float_json: shortest round-trip decimal.
static void dump_float(q35_buf *out, double v) {
    if (isnan(v)) { q35_buf_append_str(out, "NaN"); return; }
    if (isinf(v)) { q35_buf_append_str(out, v < 0 ? "-Infinity" : "Infinity"); return; }
    char tmp[64];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof(tmp), "%.*g", prec, v);
        if (strtod(tmp, NULL) == v) break;
    }
    // C prints exponents with at least two digits ("1e+05"); Python also uses two for
    // |exp| < 100, so the common forms already agree.
    q35_buf_append_str(out, tmp);
}

static void dump_string(q35_buf *out, const char *s) {
    q35_buf_append(out, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
        case '"': q35_buf_append(out, "\\\"", 2); break;
        case '\\': q35_buf_append(out, "\\\\", 2); break;
        case '\b': q35_buf_append(out, "\\b", 2); break;
        case '\f': q35_buf_append(out, "\\f", 2); break;
        case '\n': q35_buf_append(out, "\\n", 2); break;
        case '\r': q35_buf_append(out, "\\r", 2); break;
        case '\t': q35_buf_append(out, "\\t", 2); break;
        default:
            if (*p < 0x20) {
                char t[8];
                snprintf(t, sizeof(t), "\\u%04x", *p);
                q35_buf_append_str(out, t);
            } else {
                q35_buf_append(out, (const char *)p, 1);
            }
        }
    }
    q35_buf_append(out, "\"", 1);
}

static void dump_json(q35_buf *out, const qj_val *v, int compact) {
    if (!v) { q35_buf_append_str(out, "null"); return; }
    switch (v->type) {
    case QJ_NULL: q35_buf_append_str(out, "null"); break;
    case QJ_BOOL: q35_buf_append_str(out, v->boolean ? "true" : "false"); break;
    case QJ_NUM:
        if (v->is_int && v->str) q35_buf_append_str(out, v->str);
        else dump_float(out, v->num);
        break;
    case QJ_STR: dump_string(out, v->str); break;
    case QJ_ARR:
        q35_buf_append(out, "[", 1);
        for (int i = 0; i < v->n; i++) {
            if (i) q35_buf_append(out, compact ? "," : ", ", compact ? 1 : 2);
            dump_json(out, v->items[i], compact);
        }
        q35_buf_append(out, "]", 1);
        break;
    case QJ_OBJ:
        q35_buf_append(out, "{", 1);
        for (int i = 0; i < v->n; i++) {
            if (i) q35_buf_append(out, compact ? "," : ", ", compact ? 1 : 2);
            dump_string(out, v->keys[i]);
            q35_buf_append(out, compact ? ":" : ": ", compact ? 1 : 2);
            dump_json(out, v->items[i], compact);
        }
        q35_buf_append(out, "}", 1);
        break;
    }
}

void qj_dump(q35_buf *out, const qj_val *v) { dump_json(out, v, 0); }
void qj_dump_compact(q35_buf *out, const qj_val *v) { dump_json(out, v, 1); }

char *qj_dump_alloc(const qj_val *v) {
    q35_buf b;
    q35_buf_init(&b);
    qj_dump(&b, v);
    return b.data ? b.data : strdup("");
}

// Jinja `value | string`: strings raw, bool Python-cased, numbers as JSON, containers as JSON.
void qj_dump_string(q35_buf *out, const qj_val *v) {
    if (!v) { q35_buf_append_str(out, "None"); return; }
    switch (v->type) {
    case QJ_STR: q35_buf_append_str(out, v->str); break;
    case QJ_BOOL: q35_buf_append_str(out, v->boolean ? "True" : "False"); break;
    case QJ_NULL: q35_buf_append_str(out, "None"); break;
    default: qj_dump(out, v); break;
    }
}
