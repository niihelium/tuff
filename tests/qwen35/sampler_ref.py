#!/usr/bin/env python3
"""G6 host-sampler check: independent reference on fixed logits + chi-square + determinism."""
import json, math, subprocess, sys, os
BIN = os.environ.get("Q35_SAMPLER_TEST", "build-srv/q35-sampler-test")
LOGITS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.0, -1.0]

def run(*args):
    out = subprocess.check_output([BIN, *map(str, args)], text=True)
    return json.loads(out)

def reference_probs():
    # temperature 1, top_k 8, top_p 1, min_p 0 -> plain softmax
    m = max(LOGITS)
    e = [math.exp(x - m) for x in LOGITS]
    s = sum(e)
    return [x / s for x in e]

def main():
    fail = 0
    n = 20000
    h = run("hist", 7, n)
    p = reference_probs()
    obs = h["counts"]
    chi = sum((o - n * q) ** 2 / (n * q) for o, q in zip(obs, p))
    # chi2.ppf(0.999, df=7) = 24.322; scipy is optional.
    crit = 24.322
    try:
        from scipy.stats import chi2  # type: ignore
        crit = float(chi2.ppf(0.999, 7))
        pval = float(chi2.sf(chi, 7))
    except Exception:
        pval = float("nan")
    print(f"chi-square={chi:.3f} df=7 crit(0.999)={crit:.3f} p={pval:.4f} counts={obs}")
    if chi >= crit:
        print("FAIL chi-square")
        fail += 1
    else:
        print("PASS chi-square")

    g = run("greedy")["greedy"]
    # penalties: repeat 1.5, freq .5, presence 1, last_n 4, hist [0,0,1,2]
    expect = [4 / 1.5 - 2 * 0.5 - 1, 3 / 1.5 - 0.5 - 1, 2 / 1.5 - 0.5 - 1, 1.0]
    best = max(range(8), key=lambda i: (expect[i] if i < 4 else [0, -1, -2, -3][i - 4], -i))
    print(f"greedy={g} expected={best}")
    if g != best:
        print("FAIL greedy penalties")
        fail += 1
    else:
        print("PASS greedy penalties")

    a = run("det", 42)["seq"]
    b = run("det", 42)["seq"]
    c = run("det", 43)["seq"]
    print(f"same seed equal: {a == b}; different seed differs: {a != c}")
    if a != b:
        print("FAIL determinism")
        fail += 1
    else:
        print("PASS determinism")
    print("G6:", "PASS" if fail == 0 else "FAIL")
    return 1 if fail else 0

if __name__ == "__main__":
    sys.exit(main())
