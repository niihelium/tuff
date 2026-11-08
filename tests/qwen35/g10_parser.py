#!/usr/bin/env python3
"""G10: tuf parser vs llama.cpp's parser on recorded raw outputs.

The oracle is `q35-chatparse-ref`, a tiny C++ driver that links libllama-common and calls
the same common_chat_session parser llama-server uses (format/common_parsers/qwen3-coder).
This makes the comparison independent of sampling and of a running model.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = json.loads((HERE / "fixtures/parser_tools.json").read_text())

CASES = [
    ("tool_int", False,
     "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>"),
    ("tool_python", False,
     "<tool_call>\n<function=python>\n<parameter=code>\ndef hello():\n    print(\"Hello, world!\")\n\nhello()\n</parameter>\n</function>\n</tool_call>"),
    ("tool_edit", False,
     "<tool_call>\n<function=edit>\n<parameter=filename>\nfoo.c\n</parameter>\n<parameter=oldString>\n#iclunde\n</parameter>\n<parameter=newString>\n#include\n\n</parameter>\n</function>\n</tool_call>"),
    ("tool_empty", False,
     "<tool_call>\n<function=empty_args>\n</function>\n</tool_call>"),
    ("parallel", False,
     "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n"
     "<tool_call>\n<function=special_function_with_opt>\n<parameter=arg1>\n2\n</parameter>\n<parameter=arg2>\n3\n</parameter>\n</function>\n</tool_call>"),
    ("reasoning_tool", True,
     "I'm\nthinking\n</think>\n\n<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>"),
    ("reasoning_content", True,
     "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?"),
    ("tool_precedes_close", True,
     "Need to inspect the current directory.\n<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>"),
    ("no_tool_content", False, "Just a plain answer."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--oracle", default=str(ROOT / "build-srv/q35-chatparse-ref"))
    ap.add_argument("--cli", default=str(ROOT / "build-srv/q35-parse-cli"))
    a = ap.parse_args()
    cases = [{"name": n, "raw": raw, "thinking": th, "tools": TOOLS}
             for n, th, raw in CASES]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"model": a.model, "cases": cases}, f)
        path = f.name
    ref = json.loads(subprocess.check_output([a.oracle, path]))
    Path(path).unlink()
    fails = 0
    for (name, thinking, raw), r in zip(CASES, ref):
        with tempfile.NamedTemporaryFile("w", delete=False) as rf:
            rf.write(raw)
            raw_path = rf.name
        got = json.loads(subprocess.check_output(
            [a.cli, str(HERE / "fixtures/parser_tools.json"), raw_path,
             "1" if thinking else "0", "1"]))
        Path(raw_path).unlink()
        ok = (got["content"] == r["content"] and
              got["reasoning_content"] == r["reasoning_content"] and
              len(got["tool_calls"]) == len(r["tool_calls"]) and
              all(g["name"] == w["name"] and g["arguments"] == w["arguments"]
                  for g, w in zip(got["tool_calls"], r["tool_calls"])))
        if not ok:
            fails += 1
            print(f"FAIL {name}")
            print("  llama.cpp:", json.dumps(r)[:900])
            print("  tuf      :", json.dumps(got)[:900])
        else:
            print(f"PASS {name}: {len(r['tool_calls'])} call(s), "
                  f"reasoning {len(r['reasoning_content'])}B, content {len(r['content'])}B")
    print(f"G10: {'PASS' if fails == 0 else 'FAIL'} ({len(CASES)} cases, {fails} failures)")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
