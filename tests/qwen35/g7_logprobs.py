#!/usr/bin/env python3
"""G7: tuf vs llama-server logprobs on the same chat prompts (temperature 0).

llama-server is run with `-ngl 0` (CPU, exact same Q8_0 weights and fp32 accumulation
apart from its Q8_0 activation quantization).  The CUDA server quantizes activations to
Q8_1 and differs more; see the report.  We compare only positions where the greedy token
agrees and report the chosen-logprob delta distribution and the top-5 set match rate.
"""
import argparse, json, sys, urllib.request

PROMPTS = [
    "Hello",
    "Explain the difference between a process and a thread in two sentences.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of France, and why is it historically important?",
    "Summarize the plot of The Great Gatsby in three sentences.",
]

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8096")
    ap.add_argument("--tuf-url", default="http://127.0.0.1:8097")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    diffs = []
    set_ok = set_bad = 0
    positions = 0
    maxdiff = 0.0
    for pi, p in enumerate(PROMPTS):
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": a.tokens,
                "temperature": 0, "logprobs": True, "top_logprobs": a.top,
                "chat_template_kwargs": {"enable_thinking": False}}
        lr = post(a.llama_url + "/v1/chat/completions", body)
        tr = post(a.tuf_url + "/v1/chat/completions", body)
        le = (lr["choices"][0].get("logprobs") or {}).get("content") or []
        te = (tr["choices"][0].get("logprobs") or {}).get("content") or []
        agree = 0
        for i in range(min(len(le), len(te))):
            if le[i]["id"] != te[i]["id"]:
                break
            agree += 1
            positions += 1
            d = abs(le[i]["logprob"] - te[i]["logprob"])
            diffs.append(d)
            maxdiff = max(maxdiff, d)
            ls = {e["id"] for e in le[i]["top_logprobs"]}
            ts = {e["id"] for e in te[i]["top_logprobs"]}
            if ls == ts:
                set_ok += 1
            else:
                # near-tie: symmetric difference tokens sit within 0.10 of the 5th candidate
                lcut = sorted(e["logprob"] for e in le[i]["top_logprobs"])[-1]
                tcut = sorted(e["logprob"] for e in te[i]["top_logprobs"])[-1]
                lm = {e["id"]: e["logprob"] for e in le[i]["top_logprobs"]}
                tm = {e["id"]: e["logprob"] for e in te[i]["top_logprobs"]}
                near = all(abs(lm[t] - tcut) <= 0.10 for t in ls - ts) and \
                       all(abs(tm[t] - lcut) <= 0.10 for t in ts - ls)
                if near:
                    set_ok += 1
                else:
                    set_bad += 1
        print(f"prompt {pi}: {agree}/{a.tokens} positions with identical greedy token")
    diffs.sort()
    p50 = diffs[len(diffs) // 2] if diffs else 0.0
    p95 = diffs[int(len(diffs) * 0.95)] if diffs else 0.0
    over = sum(1 for d in diffs if d > 0.02)
    print(f"chosen logprob |delta|: n={len(diffs)} p50={p50:.5f} p95={p95:.5f} max={maxdiff:.5f} over_0.02={over}")
    print(f"top-{a.top} sets: exact-or-near-tie {set_ok}/{positions}, mismatched {set_bad}")
    ok = over == 0 and set_bad == 0 and positions > 0
    print(f"G7: {'PASS' if ok else 'FAIL'} ({positions} positions compared)")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
