#!/usr/bin/env python3
"""G12: 64 concurrent streaming clients, cancellations, disconnects; slot/leak check."""
import argparse, json, random, sys, threading, time
import httpx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CORPUS = json.loads((Path(__file__).resolve().parents[2] / "prompts_corpus.json").read_text())

def chat_body(prompt, max_tokens, stream=True):
    return {"model": "local", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": stream,
            "ignore_eos": False, "chat_template_kwargs": {"enable_thinking": False}}

def solo(url, prompt, max_tokens=32):
    with httpx.Client(timeout=120) as c:
        r = c.post(url + "/v1/chat/completions", json=chat_body(prompt, max_tokens, False))
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuf-url", default="http://127.0.0.1:8097")
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12)
    a = ap.parse_args()
    random.seed(a.seed)
    fixed = "Reply with exactly the word BANANA and nothing else."
    base = solo(a.tuf_url, fixed)
    print("solo baseline:", repr(base), flush=True)
    stop = threading.Event()
    stats = {"ok": 0, "cancelled": 0, "errors": 0, "mismatch": 0, "checked": 0}
    lock = threading.Lock()
    url = a.tuf_url + "/v1/chat/completions"

    def worker(wid):
        local = random.Random(a.seed * 1000 + wid)
        while not stop.is_set():
            if local.random() < 0.15:
                prompt, n = fixed, 32  # cross-talk / batch-invariance probe
            else:
                prompt = CORPUS[local.randrange(len(CORPUS))]["prompt"]
                n = local.choice([8, 16, 32, 64, 96])
            cancel = local.random() < 0.30
            try:
                with httpx.Client(timeout=120) as c:
                    if cancel:
                        # read one SSE event, then disconnect while the server still writes
                        try:
                            with c.stream("POST", url, json=chat_body(prompt, n, True)) as r:
                                r.raise_for_status()
                                it = r.iter_bytes()
                                next(it, None)
                        except Exception:
                            pass
                        with lock:
                            stats["cancelled"] += 1
                        continue
                    chunks = []
                    with c.stream("POST", url, json=chat_body(prompt, n, True)) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    obj = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    with lock: stats["errors"] += 1
                                    break
                                for ch in obj.get("choices", []):
                                    d = ch.get("delta") or {}
                                    if d.get("content"):
                                        chunks.append(d["content"])
                    text = "".join(chunks)
                    with lock:
                        stats["ok"] += 1
                    if prompt == fixed:
                        with lock:
                            stats["checked"] += 1
                            if text != base:
                                stats["mismatch"] += 1
                time.sleep(0.01)
            except Exception:
                with lock:
                    stats["errors"] += 1
                time.sleep(0.05)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(worker, i) for i in range(a.workers)]
        time.sleep(a.duration)
        stop.set()
        for f in futs:
            f.result(timeout=180)
    print("storm stats:", json.dumps(stats), "elapsed", round(time.time() - t0, 1), flush=True)

    # all slots must still be usable and identical to the solo run
    fails = 0
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: solo(a.tuf_url, fixed), range(4)))
    if any(o != base for o in outs):
        print("FAIL post-storm slot output differs", [repr(o) for o in outs])
        fails += 1
    else:
        print("PASS post-storm 4 concurrent slots match the solo run")
    if stats["errors"] > a.workers * 2:
        print("FAIL too many request errors:", stats["errors"])
        fails += 1
    if stats["mismatch"] > 0:
        print(f"FAIL {stats['mismatch']}/{stats['checked']} fixed-prompt streams differed from solo")
        fails += 1
    elif stats["checked"]:
        print(f"PASS {stats['checked']} fixed-prompt streams matched the solo run")
    if stats["ok"] == 0:
        print("FAIL no successful requests")
        fails += 1
    print("G12:", "PASS" if fails == 0 else "FAIL")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
