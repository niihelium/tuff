// G8: replay llama-server --jinja /apply-template + /tokenize fixtures through tuf's
// renderer and tokenizer.  Both the prompt bytes and the token ids must match exactly.
//
// usage: q35-template-render MODEL.gguf FIXTURES.json
//        q35-template-render MODEL.gguf -   (read one {"messages":...} body from stdin)

#include "q35_json.h"
#include "q35_template.h"
#include "gguf_parser.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_all(FILE *f) {
    q35_buf b;
    q35_buf_init(&b);
    char tmp[8192];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) q35_buf_append(&b, tmp, n);
    return b.data ? b.data : strdup("");
}

static char *render_body_json(const qj_val *body, char *err, size_t errlen) {
    const qj_val *messages = qj_get(body, "messages");
    if (!messages) { snprintf(err, errlen, "missing messages"); return NULL; }
    q35_msgs msgs;
    if (q35_messages_from_json(messages, &msgs, err, errlen) != 0) return NULL;
    int thinking = 0, add_gen = 1;
    q35_template_kwargs(qj_get(body, "chat_template_kwargs"), &thinking, &add_gen, 0, 1);
    if (qj_get(body, "add_generation_prompt"))
        add_gen = qj_bool(qj_get(body, "add_generation_prompt"), add_gen);
    const qj_val *tools = qj_get(body, "tools");
    char *p = q35_render_chat(&msgs, tools && tools->type == QJ_ARR ? tools : NULL,
                              add_gen, thinking, err, errlen);
    q35_messages_free(&msgs);
    return p;
}

static int cmp_tokens(const int32_t *a, int na, const qj_val *want) {
    if (na != want->n) return 0;
    for (int i = 0; i < na; i++) {
        if (want->items[i]->type != QJ_NUM) return 0;
        if ((long)want->items[i]->num != (long)a[i]) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf FIXTURES.json|-\n", argv[0]);
        return 2;
    }
    gguf_context_t *g = tuff_gguf_parse_file(argv[1]);
    if (!g) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    tokenizer_t *tok = tokenizer_init(g);
    if (!tok) { fprintf(stderr, "tokenizer init failed\n"); return 2; }

    char *text = strcmp(argv[2], "-") == 0 ? read_all(stdin) : NULL;
    if (text) {
        qj_val *body = qj_parse(text, strlen(text));
        char err[256] = {0};
        char *p = body ? render_body_json(body, err, sizeof(err)) : NULL;
        if (!p) { fprintf(stderr, "render failed: %s\n", err); return 1; }
        fputs(p, stdout);
        free(p);
        qj_free(body);
        free(text);
        return 0;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    char *doc = read_all(f);
    fclose(f);
    qj_val *arr = qj_parse(doc, strlen(doc));
    free(doc);
    if (!arr || arr->type != QJ_ARR) { fprintf(stderr, "fixtures are not an array\n"); return 2; }

    int fail = 0;
    for (int i = 0; i < arr->n; i++) {
        const qj_val *c = arr->items[i];
        const char *name = qj_str(qj_get(c, "name"));
        const qj_val *body = qj_get(c, "body");
        const char *want_prompt = qj_str(qj_get(c, "prompt"));
        const qj_val *want_tokens = qj_get(c, "tokens");
        const char *raw_text = qj_str(qj_get(c, "text"));
        if (raw_text) {
            int cap = (int)strlen(raw_text) * 2 + 64;
            int32_t *t = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
            int n = tokenizer_tokenize(tok, raw_text, t, cap);
            if (!want_tokens || !cmp_tokens(t, n, want_tokens)) {
                printf("FAIL %s tokens differ (got %d)\n", name, n);
                fail++;
            } else {
                printf("PASS %s (%d tokens)\n", name, n);
            }
            free(t);
            continue;
        }
        char err[256] = {0};
        char *p = body ? render_body_json(body, err, sizeof(err)) : NULL;
        if (!p) {
            printf("FAIL %s render: %s\n", name ? name : "?", err);
            fail++;
            continue;
        }
        int ok = want_prompt && strcmp(p, want_prompt) == 0;
        if (!ok) {
            size_t a = 0, b = 0;
            while (p[a] && want_prompt && want_prompt[a] && p[a] == want_prompt[a]) a++;
            b = a;
            printf("FAIL %s prompt differs at byte %zu (got %zu bytes, want %zu)\n",
                   name, a, strlen(p), want_prompt ? strlen(want_prompt) : 0);
            fprintf(stderr, "  got : ...%.80s...\n", p + (a > 40 ? a - 40 : 0));
            if (want_prompt)
                fprintf(stderr, "  want: ...%.80s...\n", want_prompt + (b > 40 ? b - 40 : 0));
            fail++;
        }
        int cap = (int)strlen(p) * 2 + 64;
        int32_t *t = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
        int n = tokenizer_tokenize(tok, p, t, cap);
        if (!want_tokens || !cmp_tokens(t, n, want_tokens)) {
            if (getenv("Q35_DUMP_TOKENS")) {
                printf("%s:", name);
                for (int k = 0; k < n; k++) printf(" %d", t[k]);
                printf("\n");
            }
            printf("FAIL %s tokens differ (got %d)\n", name, n);
            fail++;
        } else {
            if (getenv("Q35_DUMP_TOKENS")) {
                printf("%s:", name);
                for (int k = 0; k < n; k++) printf(" %d", t[k]);
                printf("\n");
            }
            printf("PASS %s (%d tokens)\n", name, n);
        }
        free(t);
        free(p);
    }
    int ncases = arr ? arr->n : 0;
    qj_free(arr);
    printf("G8: %s (%d cases, %d failures)\n", fail ? "FAIL" : "PASS", ncases, fail);
    tokenizer_free(tok);
    tuff_gguf_free(g);
    return fail ? 1 : 0;
}
