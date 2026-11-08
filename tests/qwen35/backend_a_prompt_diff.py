#!/usr/bin/env python3
"""Backend-A prompt-renderer regression: compile the renderer from git HEAD and from the
current server.c on the same fixtures, and require byte-identical prompts.

The block extracted from handle_chat is self-contained (it only uses buf_t, chat_msgs_t and
system_prompt / reasoning_instruction).  This is the old-vs-new fixture test the brief asks
for; it fails if a later server.c edit changes backend A's prompt bytes.
"""
import subprocess, sys, tempfile, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = '    buf_t prompt; buf_init(&prompt);'
END = '    buf_append_str(&prompt, "<|im_start|>assistant\\n");'

def extract(src: str, label: str) -> str:
    a = src.index(START)
    b = src.index(END, a) + len(END)
    return src[a:b]

def cstr(s: str) -> str:
    out = '"'
    for ch in s:
        if ch == '"': out += '\\"'
        elif ch == '\\': out += '\\\\'
        elif ch == '\n': out += '\\n'
        elif ch == '\r': out += '\\r'
        elif ch == '\t': out += '\\t'
        else: out += ch
    return out + '"'

FIXTURES = [
    ("user", "" , [("user", "Hello")]),
    ("user_unicode", "", [("user", "你好 🌍 — test")]),
    ("system_user", "", [("system", "You are terse."), ("user", "Hi")]),
    ("system_override", "CUSTOM SYSTEM", [("user", "Hi")]),
    ("multi", "", [("user", "q1"), ("assistant", "a1"), ("user", "q2")]),
    ("assistant_only", "", [("assistant", "only")]),
    ("empty_contents", "", [("user", ""), ("assistant", "")]),
    ("code", "", [("user", "int main(){\n  return 0;\n}")]),
    ("toolish", "", [("tool", "result body"), ("user", "next")]),
]

REASONING = {
    "none": "\\nRespond directly to the user without any thinking process, reasoning, or <think> tags.",
    "low": "\\nKeep your reasoning process very brief and concise before answering.",
    "high": "\\nProvide a detailed, thorough step-by-step reasoning process before answering.",
    "": "",
}

def program(block: str) -> str:
    cases = []
    for name, sysp, msgs in FIXTURES:
        for rk, rv in REASONING.items():
            init = ", ".join(f"{{{cstr(r)}, {cstr(c)}}}" for r, c in msgs)
            cases.append((name + ("_" + rk if rk else ""), sysp, init, len(msgs), rv))
    parts = []
    parts.append(textwrap.dedent(r'''
        #include <stdarg.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        typedef struct { char *data; size_t len, cap; } buf_t;
        static void buf_init(buf_t *b) { b->data=NULL; b->len=0; b->cap=0; }
        static void buf_reserve(buf_t *b, size_t extra) {
            size_t need=b->len+extra+1; if (need<=b->cap) return;
            size_t ncap=b->cap?b->cap:64; while(ncap<need) ncap*=2;
            b->data=realloc(b->data,ncap); if(!b->data) exit(1); b->cap=ncap;
        }
        static void buf_append(buf_t *b, const char *s, size_t n) {
            if(!n) return; buf_reserve(b,n); memcpy(b->data+b->len,s,n); b->len+=n; b->data[b->len]=0;
        }
        static void buf_append_str(buf_t *b, const char *s) { buf_append(b,s,strlen(s)); }
        static void buf_appendf(buf_t *b, const char *fmt, ...) {
            va_list ap, ap2; va_start(ap,fmt); va_copy(ap2,ap);
            int need=vsnprintf(NULL,0,fmt,ap); va_end(ap);
            if(need<0){va_end(ap2);return;} buf_reserve(b,(size_t)need);
            vsnprintf(b->data+b->len,(size_t)need+1,fmt,ap2); va_end(ap2); b->len+=(size_t)need;
        }
        typedef struct { char *role; char *content; } chat_msg_t;
        typedef struct { chat_msg_t *items; int n; int cap; } chat_msgs_t;
        typedef struct { chat_msgs_t messages; } req_t;
        typedef struct { const char *system_prompt; } server_t;
        #define DEFAULT_SYSTEM "You are a helpful, respectful and honest AI assistant."
    '''))
    parts.append("static char *render(const char *system_prompt, const char *reasoning_instruction, "
                 "chat_msg_t *items, int n) {\n")
    parts.append("    server_t sst; server_t *s = &sst; s->system_prompt = system_prompt;\n")
    parts.append("    req_t r; r.messages.items = items; r.messages.n = n; r.messages.cap = n;\n")
    parts.append(block + "\n    return prompt.data ? prompt.data : strdup(\"\");\n}\n")
    parts.append("int main(void) {\n")
    for name, sysp, init, n, rv in cases:
        parts.append(f"    {{ chat_msg_t m[] = {{{init}}}; char *p = render({cstr(sysp)}, {cstr(rv)}, m, {n});\n")
        parts.append(f"      printf(\"{name} %zu \", strlen(p)); for (const unsigned char *q=(const unsigned char*)p; *q; q++) printf(\"%02x\", *q); printf(\"\\n\"); free(p); }}\n")
    parts.append("    return 0;\n}\n")
    return "".join(parts)

def main():
    head = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:src/server.c"], text=True)
    cur = (ROOT / "src/server.c").read_text()
    old_block = extract(head, "HEAD")
    new_block = extract(cur, "current")
    print(f"old block {len(old_block)} bytes, new block {len(new_block)} bytes, "
          f"textually_identical={old_block == new_block}")
    with tempfile.TemporaryDirectory() as td:
        outs = []
        for tag, block in (("old", old_block), ("new", new_block)):
            src = Path(td) / f"{tag}.c"
            exe = Path(td) / tag
            src.write_text(program(block))
            subprocess.check_call(["gcc", "-O2", "-o", str(exe), str(src)])
            outs.append(subprocess.check_output([str(exe)], text=True))
        if outs[0] != outs[1]:
            a, b = outs[0].splitlines(), outs[1].splitlines()
            for x, y in zip(a, b):
                if x != y:
                    print("DIFF", x.split()[0])
            print("backend-A prompt renderer: FAIL")
            return 1
    print(f"backend-A prompt renderer: PASS ({len(outs[0].splitlines())} fixtures, byte-identical)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
