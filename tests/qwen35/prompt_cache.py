#!/usr/bin/env python3
"""Backend B exact-prefix prompt cache: assert a multi-turn transcript prefills only the suffix.

The first turn is `[user]`; the second is `[user, assistant]`, which is exactly how an agent
resends its transcript.  The assistant turn after the last user query keeps the template's
think block, so the second prompt begins with the whole first sequence (prompt + generated
tokens) and only the `<|im_end|>` / next generation prompt are prefilled.
"""
import argparse, json, sys, urllib.request

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def chat(url, messages, max_tokens):
    return post(url + "/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}})

def turn(url, messages, max_tokens, label):
    r = chat(url, messages, max_tokens)
    u = r["usage"]
    p = u["prompt_tokens"]
    c = u["prompt_tokens_details"]["cached_tokens"]
    pre = r["timings"]["prompt_n"]
    content = r["choices"][0]["message"]["content"]
    print(f"{label}: prompt_tokens={p} cached={c} prompt_n={pre} content={content[:50]!r}")
    return p, c, pre, content, u["completion_tokens"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuf-url", default="http://127.0.0.1:8097")
    a = ap.parse_args()
    u = a.tuf_url
    fails = 0
    q1 = "List three primary colors, comma separated."
    p1, c1, pre1, a1, n1 = turn(u, [{"role": "user", "content": q1}], 48, "round1")
    p2, c2, pre2, a2, n2 = turn(u, [{"role": "user", "content": q1},
                                    {"role": "assistant", "content": a1}], 24, "round2")
    # The first prompt is 20 tokens; the generated answer follows it.  Reuse must cover at
    # least the first prompt and cannot exceed the second prompt.
    if c2 <= p1:
        print(f"FAIL round2 did not reuse the first prompt (cached={c2} <= {p1})")
        fails += 1
    elif c2 > p2:
        print("FAIL cached more than the prompt")
        fails += 1
    elif pre2 != p2 - c2:
        print(f"FAIL prompt_n {pre2} != total {p2} - cached {c2}")
        fails += 1
    elif pre2 > 24:
        print(f"FAIL round2 prefilled too much ({pre2} tokens; expected only the suffix)")
        fails += 1
    else:
        print(f"PASS round2: reused {c2} tokens, prefilled only {pre2}")

    # A different user prompt must reset the slot (no stale-state reuse).
    q3 = "What is the boiling point of water in Celsius?"
    p3, c3, pre3, a3, n3 = turn(u, [{"role": "user", "content": q3}], 16, "round3")
    if c3 != 0:
        print(f"FAIL divergent conversation reused {c3} tokens")
        fails += 1
    else:
        print("PASS divergent conversation: 0 cached tokens")

    # Its own next turn must reuse the second slot's sequence.
    p4, c4, pre4, a4, n4 = turn(u, [{"role": "user", "content": q3},
                                    {"role": "assistant", "content": a3}], 8, "round4")
    if c4 <= p3:
        print(f"FAIL round4 did not reuse the divergent turn (cached={c4} <= {p3})")
        fails += 1
    elif pre4 != p4 - c4 or pre4 > 16:
        print(f"FAIL round4 prefill accounting (prompt_n={pre4}, total={p4}, cached={c4})")
        fails += 1
    else:
        print(f"PASS round4: reused {c4} tokens, prefilled only {pre4}")

    print("prompt-cache:", "PASS" if fails == 0 else "FAIL")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
