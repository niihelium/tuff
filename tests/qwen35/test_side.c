// Host-side tests for backend B: JSON writer, template renderer and tool/reasoning parser.
// Parser cases are ported from llama.cpp tests/test-chat.cpp (Qwen3.5) and
// tests/test-chat-peg-parser.cpp (qwen3-coder examples).

#include "q35_json.h"
#include "q35_template.h"
#include "q35_toolparse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void check_str(const char *name, const char *got, const char *want) {
    if (!got || strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL %s\n  got:  <<<%s>>>\n  want: <<<%s>>>\n", name,
                got ? got : "(null)", want);
        g_fail++;
    } else {
        printf("ok   %s\n", name);
    }
}

static void check_int(const char *name, long got, long want) {
    if (got != want) { fprintf(stderr, "FAIL %s: %ld != %ld\n", name, got, want); g_fail++; }
    else printf("ok   %s\n", name);
}

static void test_json(void) {
    const char *src = "{\"z\":1,\"a\":[true,false,null,2.5],\"s\":\"x\\ny\"}";
    qj_val *v = qj_parse(src, strlen(src));
    check_int("json parse", v != NULL, 1);
    q35_buf b; q35_buf_init(&b);
    qj_dump(&b, v);
    check_str("json order/format", b.data, "{\"z\": 1, \"a\": [true, false, null, 2.5], \"s\": \"x\\ny\"}");
    q35_buf_free(&b);
    qj_free(v);

    qj_val *bad = qj_parse("{\"a\":}", 6);
    check_int("json rejects malformed", bad == NULL, 1);
    qj_free(bad);
}

static qj_val *parse_lit(const char *s) { return qj_parse(s, strlen(s)); }

static void test_template(void) {
    char err[256];
    qj_val *msgs = parse_lit(
        "[{\"role\":\"system\",\"content\":\"sys\"},"
        "{\"role\":\"user\",\"content\":\"Hello\"}]");
    q35_msgs m;
    check_int("msgs parse", q35_messages_from_json(msgs, &m, err, sizeof(err)), 0);
    char *p = q35_render_chat(&m, NULL, 1, 0, err, sizeof(err));
    check_str("template sys+user nothink", p,
              "<|im_start|>system\nsys<|im_end|>\n"
              "<|im_start|>user\nHello<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    free(p);
    p = q35_render_chat(&m, NULL, 1, 1, err, sizeof(err));
    check_int("template thinking prompt", strstr(p, "<|im_start|>assistant\n<think>\n") != NULL, 1);
    free(p);
    q35_messages_free(&m);
    qj_free(msgs);

    // old assistant turn has no reasoning block; the latest one does.
    msgs = parse_lit(
        "[{\"role\":\"user\",\"content\":\"q1\"},"
        "{\"role\":\"assistant\",\"content\":\"a1\"},"
        "{\"role\":\"user\",\"content\":\"q2\"},"
        "{\"role\":\"assistant\",\"content\":\"a2\"}]");
    check_int("msgs parse 2", q35_messages_from_json(msgs, &m, err, sizeof(err)), 0);
    p = q35_render_chat(&m, NULL, 0, 0, err, sizeof(err));
    check_str("template old reasoning stripped", p,
              "<|im_start|>user\nq1<|im_end|>\n"
              "<|im_start|>assistant\na1<|im_end|>\n"
              "<|im_start|>user\nq2<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n");
    free(p);
    q35_messages_free(&m);
    qj_free(msgs);

    // reasoning_content on the last assistant message
    msgs = parse_lit(
        "[{\"role\":\"user\",\"content\":\"q\"},"
        "{\"role\":\"assistant\",\"content\":\"answer\",\"reasoning_content\":\"why\"}]");
    check_int("msgs parse 3", q35_messages_from_json(msgs, &m, err, sizeof(err)), 0);
    p = q35_render_chat(&m, NULL, 0, 0, err, sizeof(err));
    check_str("template reasoning_content", p,
              "<|im_start|>user\nq<|im_end|>\n"
              "<|im_start|>assistant\n<think>\nwhy\n</think>\n\nanswer<|im_end|>\n");
    free(p);
    q35_messages_free(&m);
    qj_free(msgs);

    // tool results group under a single user turn
    msgs = parse_lit(
        "[{\"role\":\"user\",\"content\":\"go\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":["
        "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":{\"x\":1}}}]},"
        "{\"role\":\"tool\",\"content\":\"r1\"},"
        "{\"role\":\"tool\",\"content\":\"r2\"}]");
    check_int("msgs parse 4", q35_messages_from_json(msgs, &m, err, sizeof(err)), 0);
    p = q35_render_chat(&m, NULL, 0, 0, err, sizeof(err));
    check_str("template tool call + grouped tool results", p,
              "<|im_start|>user\ngo<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=f>\n<parameter=x>\n1\n"
              "</parameter>\n</function>\n</tool_call><|im_end|>\n"
              "<|im_start|>user\n<tool_response>\nr1\n</tool_response>"
              "\n<tool_response>\nr2\n</tool_response><|im_end|>\n");
    free(p);
    q35_messages_free(&m);
    qj_free(msgs);

    // string arguments are parsed to objects before rendering
    msgs = parse_lit(
        "[{\"role\":\"user\",\"content\":\"go\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":["
        "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"s\\\":\\\"a b\\\"}\"}}]}]");
    check_int("msgs parse args string", q35_messages_from_json(msgs, &m, err, sizeof(err)), 0);
    p = q35_render_chat(&m, NULL, 0, 0, err, sizeof(err));
    check_int("template tool args string parsed", strstr(p, "<parameter=s>\na b\n</parameter>") != NULL, 1);
    free(p);
    q35_messages_free(&m);
    qj_free(msgs);
}

static const char *TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"special_function\","
    "\"description\":\"special\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"arg1\":{\"type\":\"integer\"}},\"required\":[\"arg1\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"python\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"filename\":{\"type\":\"string\"},\"oldString\":{\"type\":\"string\"},"
    "\"newString\":{\"type\":\"string\"}},\"required\":[\"filename\",\"oldString\",\"newString\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"empty_args\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{}}}}]";

static void run_parse(const char *name, const char *raw, qj_val *tools, int thinking,
                      int extract, const char *want_reason, const char *want_content,
                      int want_calls, const char *want0, const char *want1) {
    q35_out o;
    q35_parse_output(raw, strlen(raw), tools, thinking, extract, &o);
    char label[128];
    snprintf(label, sizeof(label), "%s reasoning", name);
    check_str(label, o.reasoning, want_reason);
    snprintf(label, sizeof(label), "%s content", name);
    check_str(label, o.content, want_content);
    snprintf(label, sizeof(label), "%s calls", name);
    check_int(label, o.n_calls, want_calls);
    if (want_calls > 0) {
        snprintf(label, sizeof(label), "%s args0", name);
        check_str(label, o.calls[0].arguments, want0);
    }
    if (want_calls > 1) {
        snprintf(label, sizeof(label), "%s args1", name);
        check_str(label, o.calls[1].arguments, want1);
    }
    q35_out_free(&o);
}

static void test_parser(void) {
    qj_val *tools = parse_lit(TOOLS);

    run_parse("reasoning", "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?", tools, 1, 1,
              "I'm\nthinking", "Hello, world!\nWhat's up?", 0, NULL, NULL);
    run_parse("reasoning none", "I'm\nthinking\n</think>\n\nHello, world!", tools, 1, 0,
              "", "I'm\nthinking\n</think>\n\nHello, world!", 0, NULL, NULL);

    run_parse("tool int",
              "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n"
              "</function>\n</tool_call>",
              tools, 0, 1, "", "", 1, "{\"arg1\":1}", NULL);

    run_parse("tool python",
              "<tool_call>\n<function=python>\n<parameter=code>\n"
              "def hello():\n    print(\"Hello, world!\")\n\nhello()\n"
              "</parameter>\n</function>\n</tool_call>",
              tools, 0, 1, "", "", 1,
              "{\"code\":\"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}", NULL);

    run_parse("tool edit",
              "<tool_call>\n<function=edit>\n<parameter=filename>\nfoo.c\n</parameter>\n"
              "<parameter=oldString>\n#iclunde\n</parameter>\n"
              "<parameter=newString>\n#include\n\n</parameter>\n"
              "</function>\n</tool_call>",
              tools, 0, 1, "", "", 1,
              "{\"filename\":\"foo.c\",\"oldString\":\"#iclunde\",\"newString\":\"#include\\n\"}",
              NULL);

    run_parse("tool empty args",
              "<tool_call>\n<function=empty_args>\n</function>\n</tool_call>",
              tools, 0, 1, "", "", 1, "{}", NULL);

    run_parse("parallel calls",
              "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n"
              "</function>\n</tool_call>\n"
              "<tool_call>\n<function=edit>\n<parameter=filename>\nf\n</parameter>\n"
              "<parameter=oldString>\na\n</parameter>\n<parameter=newString>\nb\n</parameter>\n"
              "</function>\n</tool_call>",
              tools, 0, 1, "", "", 2, "{\"arg1\":1}",
              "{\"filename\":\"f\",\"oldString\":\"a\",\"newString\":\"b\"}");

    run_parse("tool before think close",
              "Need to inspect the current directory.\n<tool_call>\n<function=special_function>\n"
              "<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>",
              tools, 1, 1, "Need to inspect the current directory.\n", "", 1, "{\"arg1\":1}", NULL);

    qj_free(tools);
}

int main(void) {
    test_json();
    test_template();
    test_parser();
    if (g_fail) { fprintf(stderr, "%d failure(s)\n", g_fail); return 1; }
    printf("all side tests passed\n");
    return 0;
}
