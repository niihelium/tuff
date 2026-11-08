// G10 helper: run tuf's Qwen3.5 output parser on a recorded raw string.
// usage: q35-parse-cli TOOLS.json|none RAW.txt THINKING EXTRACT_REASONING
// prints {"reasoning_content","content","tool_calls":[{"name","arguments"}]}

#include "q35_json.h"
#include "q35_toolparse.h"
#include "q35_template.h"

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

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s TOOLS.json|none RAW.txt THINKING EXTRACT\n", argv[0]);
        return 2;
    }
    qj_val *tools = NULL;
    if (strcmp(argv[1], "none") != 0) {
        FILE *tf = fopen(argv[1], "rb");
        if (!tf) { perror("tools"); return 2; }
        char *t = read_all(tf);
        fclose(tf);
        tools = qj_parse(t, strlen(t));
        free(t);
        if (!tools) { fprintf(stderr, "bad tools JSON\n"); return 2; }
    }
    FILE *rf = fopen(argv[2], "rb");
    if (!rf) { perror("raw"); return 2; }
    char *raw = read_all(rf);
    fclose(rf);
    q35_out o;
    q35_parse_output(raw, strlen(raw), tools, atoi(argv[3]), atoi(argv[4]), &o);
    q35_buf b;
    q35_buf_init(&b);
    q35_buf_append_str(&b, "{\"reasoning_content\":");
    qj_val rv;
    memset(&rv, 0, sizeof(rv));
    rv.type = QJ_STR;
    rv.str = o.reasoning;
    qj_dump(&b, &rv);
    q35_buf_append_str(&b, ",\"content\":");
    rv.str = o.content;
    qj_dump(&b, &rv);
    q35_buf_append_str(&b, ",\"tool_calls\":[");
    for (int i = 0; i < o.n_calls; i++) {
        if (i) q35_buf_append_str(&b, ",");
        q35_buf_append_str(&b, "{\"name\":");
        rv.str = o.calls[i].name;
        qj_dump(&b, &rv);
        q35_buf_append_str(&b, ",\"arguments\":");
        rv.str = o.calls[i].arguments;
        qj_dump(&b, &rv);
        q35_buf_append_str(&b, "}");
    }
    q35_buf_append_str(&b, "]}");
    fputs(b.data, stdout);
    fputc('\n', stdout);
    q35_buf_free(&b);
    q35_out_free(&o);
    free(raw);
    qj_free(tools);
    return 0;
}
