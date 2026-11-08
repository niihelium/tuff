#!/usr/bin/env python3
"""Generate the G8 template fixtures from llama-server --jinja.

For every conversation this posts to llama-server's /apply-template and /tokenize and stores
`{name, body, prompt, tokens}`.  `q35-template-render` replays the same bodies through tuf's
renderer and tokenizer and requires both to be byte-identical.

usage: gen_g8_fixtures.py --port 8096 --out fixtures/g8_template.json
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_TOOLS = json.loads((HERE / "fixtures/agent_tools.json").read_text())
SIMPLE_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "days": {"type": "integer", "description": "Forecast length."},
                "metric": {"type": "boolean"},
                "options": {"type": "object", "properties": {"lang": {"type": "string"}}},
            },
            "required": ["city"],
        },
    },
}]
EDIT_TOOLS = [t for t in AGENT_TOOLS if t["function"]["name"] in ("edit_file", "write_file", "read_file")]


def msg(role, content=None, **kw):
    m = {"role": role}
    if content is not None:
        m["content"] = content
    m.update(kw)
    return m


def call(name, args, cid=None):
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid:
        c["id"] = cid
    return c


def cases():
    out = []

    def add(name, messages, tools=None, thinking=None, add_gen=None, tool_choice=None,
            reasoning_format=None):
        body = {"messages": messages}
        if tools is not None:
            body["tools"] = tools
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if add_gen is not None:
            body["add_generation_prompt"] = add_gen
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if reasoning_format is not None:
            body["reasoning_format"] = reasoning_format
        out.append({"name": name, "body": body})

    add("user_hello", [msg("user", "Hello")], thinking=False)
    add("user_unicode", [msg("user", "你好，世界。今天天气怎么样？")], thinking=False)
    add("user_emoji", [msg("user", "Summarize this: 🧪🚀🔥 — and keep the emoji.")], thinking=False)
    add("user_code", [msg("user", 'Fix this C:\n```c\nint main(){ printf("%d\\n", x); }\n```')],
        thinking=False)
    add("user_long", [msg("user", "word " * 900)], thinking=False)
    add("user_edges", [msg("user", "  leading and trailing  \n")], thinking=False)
    add("system_user", [msg("system", "You are terse."), msg("user", "Hi")], thinking=False)
    add("system_empty", [msg("system", "   "), msg("user", "Hi")], thinking=False)
    add("system_unicode", [msg("system", "Ты помощник. 回答は日本語で。"),
                           msg("user", "Explain RAID 5.")], thinking=False)
    add("content_parts", [msg("user", [{"type": "text", "text": "part one"},
                                       {"type": "text", "text": "part two"}])], thinking=False)

    add("multi_turn", [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")],
        thinking=False)
    add("multi_turn_old_think",
        [msg("user", "q1"),
         msg("assistant", "<think>\nold reasoning\n</think>\n\nanswer one"),
         msg("user", "q2")], thinking=False)
    add("last_assistant_think",
        [msg("user", "q1"), msg("assistant", "<think>\nthought\n</think>\n\nanswer")],
        thinking=False)
    add("assistant_reasoning_field",
        [msg("user", "q1"), msg("assistant", "answer", reasoning_content="because")],
        thinking=False)
    add("multi_turn_reasoning",
        [msg("user", "q1"), msg("assistant", "a1", reasoning_content="r1"),
         msg("user", "q2"), msg("assistant", "a2", reasoning_content="r2")], thinking=False)
    add("assistant_empty", [msg("user", "q1"), msg("assistant", ""), msg("user", "q2")],
        thinking=False)
    add("developer_role", [msg("developer", "dev instructions"), msg("user", "go")],
        thinking=False)
    add("eight_turns",
        [m for i in range(4) for m in (msg("user", f"question {i}"),
                                       msg("assistant", f"answer {i}"))][:-1] + [msg("user", "final")],
        thinking=False)

    add("tools_user", [msg("user", "List the files.")], tools=AGENT_TOOLS, thinking=False)
    add("tools_system_user", [msg("system", "You are tuff."), msg("user", "Read a file.")],
        tools=AGENT_TOOLS, thinking=False)
    add("tools_none_choice", [msg("user", "Just answer.")], tools=AGENT_TOOLS,
        thinking=False, tool_choice="none")
    add("tools_required", [msg("user", "Call something.")], tools=SIMPLE_TOOL,
        thinking=False, tool_choice="required")
    add("tools_assistant_call",
        [msg("user", "weather?"),
         msg("assistant", "Checking.", tool_calls=[call("get_weather", {"city": "Oslo"})]),
         msg("user", "and tomorrow?")], tools=SIMPLE_TOOL, thinking=False)
    add("tools_tool_result",
        [msg("user", "weather?"),
         msg("assistant", "", tool_calls=[call("get_weather", {"city": "Oslo"}, "call_1")]),
         msg("tool", "sunny, 12 C", tool_call_id="call_1"),
         msg("user", "thanks")], tools=SIMPLE_TOOL, thinking=False)
    add("tools_multi_results",
        [msg("user", "weather in two cities"),
         msg("assistant", "Checking both.", tool_calls=[
             call("get_weather", {"city": "Oslo"}, "c1"),
             call("get_weather", {"city": "Rome"}, "c2")]),
         msg("tool", "sunny", tool_call_id="c1"),
         msg("tool", "rain", tool_call_id="c2"),
         msg("user", "ok")], tools=SIMPLE_TOOL, thinking=False)
    add("tools_loop",
        [msg("user", "edit the file"),
         msg("assistant", "", tool_calls=[call("read_file", {"path": "a.txt"})]),
         msg("tool", "1: hello", tool_call_id="c1"),
         msg("user", "now edit"),
         msg("assistant", "", tool_calls=[
             call("edit_file", {"path": "a.txt", "old_string": "hello", "new_string": "hi"})]),
         msg("tool", "edited", tool_call_id="c2"),
         msg("user", "done?")], tools=EDIT_TOOLS, thinking=False)
    add("tools_reasoning_call",
        [msg("user", "weather?"),
         msg("assistant", "", reasoning_content="need a lookup",
             tool_calls=[call("get_weather", {"city": "Oslo", "days": 2})])],
        tools=SIMPLE_TOOL, thinking=False)
    add("tools_arg_types",
        [msg("user", "typed args"),
         msg("assistant", "", tool_calls=[
             call("get_weather", {"city": "Oslo", "days": 3, "metric": True,
                                  "options": {"lang": "no"}})])],
        tools=SIMPLE_TOOL, thinking=False)
    add("tools_empty_args",
        [msg("user", "list"),
         msg("assistant", "", tool_calls=[call("bash", {})]),
         msg("tool", "ok", tool_call_id="c")], tools=AGENT_TOOLS, thinking=False)
    add("tools_string_args",
        [msg("user", "edit"),
         msg("assistant", "", tool_calls=[
             call("edit_file", json.dumps({"path": "a b.txt", "old_string": "x\ny",
                                           "new_string": "z"}, ensure_ascii=False))])],
        tools=EDIT_TOOLS, thinking=False)

    add("think_on", [msg("user", "What is 2+2?")], thinking=True)
    add("think_off", [msg("user", "What is 2+2?")], thinking=False)
    add("think_on_multi", [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")],
        thinking=True)
    add("think_off_multi", [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")],
        thinking=False)
    add("think_on_tools", [msg("user", "weather?")], tools=SIMPLE_TOOL, thinking=True)
    add("think_off_tools", [msg("user", "weather?")], tools=SIMPLE_TOOL, thinking=False)
    add("think_on_reasoning", [msg("user", "q"), msg("assistant", "ans", reasoning_content="why")],
        thinking=True)
    add("think_off_reasoning", [msg("user", "q"), msg("assistant", "ans", reasoning_content="why")],
        thinking=False)
    add("think_off_inline_think",
        [msg("user", "q"), msg("assistant", "<think>\nkept inline?\n</think>\n\nans")],
        thinking=False)

    add("no_generation_prompt", [msg("user", "Hello")], thinking=False, add_gen=False)
    add("no_generation_prompt_tools", [msg("user", "Hello")], tools=AGENT_TOOLS,
        thinking=False, add_gen=False)
    add("tool_unicode", [msg("user", "weather?"),
                         msg("assistant", "", tool_calls=[
                             call("get_weather", {"city": "München"})]),
                         msg("tool", "sonnig ☀️ 21°C"), msg("user", "danke")],
        tools=SIMPLE_TOOL, thinking=False)
    add("special_tokens_in_content",
        [msg("user", "literal <|im_start|> and <|im_end|> markers")], thinking=False)
    add("tools_emoji", [msg("user", "🛠️ which tool?")], tools=AGENT_TOOLS, thinking=False)
    add("deepseek_reasoning_format", [msg("user", "Hi")], thinking=True,
        reasoning_format="deepseek")
    add("none_reasoning_format", [msg("user", "Hi")], thinking=True,
        reasoning_format="none")
    add("assistant_call_content", [msg("user", "go"),
                                   msg("assistant", "I will check.",
                                       tool_calls=[call("get_weather", {"city": "Oslo"})]),
                                   msg("user", "well?")], tools=SIMPLE_TOOL, thinking=False)
    return out


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--out", type=Path, default=HERE / "fixtures/g8_template.json")
    a = ap.parse_args()
    out = []
    for c in cases():
        body = dict(c["body"])
        try:
            r = post(a.port, "/apply-template", body)
        except Exception as e:
            print(f"FAILED {c['name']}: {e}")
            print(json.dumps(body)[:1200])
            raise
        prompt = r["prompt"]
        tok = post(a.port, "/tokenize", {"content": prompt})["tokens"]
        out.append({"name": c["name"], "body": body, "prompt": prompt, "tokens": tok})
        print(f"{c['name']}: prompt {len(prompt)} bytes, {len(tok)} tokens", flush=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {a.out} ({len(out)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
