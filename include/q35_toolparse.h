#ifndef Q35_TOOLPARSE_H
#define Q35_TOOLPARSE_H

// Qwen3.5 output parser: <think> reasoning extraction and the XML tool-call format
// handled by llama.cpp's common/parsers/qwen3-coder.cpp.

#include "q35_json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *name;
    char *arguments;   // JSON object as a string
} q35_tc;

typedef struct {
    char   *reasoning; // owned, never NULL
    char   *content;   // owned, never NULL
    q35_tc *calls;     // owned
    int     n_calls;
} q35_out;

// `tools` is the request's OpenAI tools array (may be NULL).  `thinking_enabled` is the
// template's enable_thinking value; with it set, text before the first </think> (or
// <tool_call>) is reasoning even if the opening <think> was in the prompt.
void q35_parse_output(const char *raw, size_t len, const qj_val *tools,
                      int thinking_enabled, int extract_reasoning, q35_out *out);
void q35_out_free(q35_out *o);

// Parse one complete <tool_call> block starting at `p`.  Returns the number of bytes
// consumed or 0 when the block is incomplete/malformed.
size_t q35_parse_tool_block(const char *p, size_t len, const qj_val *tools,
                            q35_tc *out);

#ifdef __cplusplus
}
#endif

#endif
