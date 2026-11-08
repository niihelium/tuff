#ifndef Q35_JSON_H
#define Q35_JSON_H

// Small strict JSON reader/writer used by the qwen35 server side (backend B).
//
// The writer intentionally matches llama.cpp's minja `tojson` byte-for-byte for the
// constructs a chat template can emit: object insertion order, Python-style ", " / ": "
// separators, JSON string escapes and shortest-round-trip float formatting.  That is what
// makes the `tools` system block in the Qwen3.5 template identical to llama-server --jinja.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Growable byte buffer
// ---------------------------------------------------------------------------

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} q35_buf;

void q35_buf_init(q35_buf *b);
void q35_buf_free(q35_buf *b);
void q35_buf_reserve(q35_buf *b, size_t extra);
void q35_buf_append(q35_buf *b, const char *s, size_t n);
void q35_buf_append_str(q35_buf *b, const char *s);
void q35_buf_appendf(q35_buf *b, const char *fmt, ...);

// ---------------------------------------------------------------------------
// JSON values
// ---------------------------------------------------------------------------

typedef enum {
    QJ_NULL = 0,
    QJ_BOOL,
    QJ_NUM,
    QJ_STR,
    QJ_ARR,
    QJ_OBJ,
} qj_type;

typedef struct qj_val qj_val;

struct qj_val {
    qj_type type;
    int     boolean;
    double  num;
    int     is_int;      // number had no '.', 'e' or 'E'
    char   *str;         // decoded UTF-8 for QJ_STR, raw token for QJ_NUM
    qj_val **items;      // array items or object values
    char   **keys;       // object keys (parallel to items)
    int      n;
    int      cap;
};

// Parse one complete JSON value.  Returns NULL on malformed input.
qj_val *qj_parse(const char *s, size_t len);
void    qj_free(qj_val *v);

qj_val     *qj_get(const qj_val *obj, const char *key);
const char *qj_str(const qj_val *v);              // NULL unless QJ_STR
double      qj_num(const qj_val *v, double def);
long        qj_int(const qj_val *v, long def);
int         qj_bool(const qj_val *v, int def);
int         qj_len(const qj_val *v);

// Serialise like minja's `tojson` filter (insertion order, ", " and ": ").
void qj_dump(q35_buf *out, const qj_val *v);
// Compact form ("," and ":") used by llama.cpp for tool-call arguments.
void qj_dump_compact(q35_buf *out, const qj_val *v);
// Convenience: malloc'd serialisation.
char *qj_dump_alloc(const qj_val *v);

// Python `str()` equivalent for the template's `args_value | string` fallback.
void qj_dump_string(q35_buf *out, const qj_val *v);

#ifdef __cplusplus
}
#endif

#endif
