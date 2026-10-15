#!/usr/bin/env python3
"""Checks for the LFM2.5 oracle gate (driven by tests/verify_lfm25_oracle.sh).

  exact      stage 1: the first N greedy tokens must be identical to the reference.
  teacher    stage 2: teacher-forced top-1 agreement. A step may disagree only where the
             reference itself is a near-tie (llama.cpp top1-top2 margin < --min-margin).
  tolerance  stage 3: logit error versus the FP64 reference must be no worse than
             llama.cpp's own (mean RMS ratio <= --max-ratio, worst step <= --max-step-ratio).

Why not a long exact-match run: two correct Q8_1-activation runtimes carry different rounding
noise (~0.1-0.2 logit RMS on LFM2.5-1.2B), and greedy decoding hits FP64 margins as small as
0.014 within 20 tokens. Past the first such near-tie the streams fork and every later token is
noise, so long exact parity measures luck, not correctness. See LFM25_ROADMAP.md (Gate design).
"""
import argparse
import sys

import numpy as np

MAGIC = b"CLBRLGT1"


def load_tokens(path):
    return [int(x) for x in open(path) if x.strip()]


def load_logits(path, vocab):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw[:8].tobytes() == MAGIC:  # tuff dump: magic + u32 n + u32 vocab
        n, v = np.frombuffer(raw[8:16].tobytes(), dtype=np.uint32)
        if v != vocab:
            sys.exit(f"{path}: vocab {v} != {vocab}")
        return raw[16:].view(np.float32).reshape(int(n), vocab)
    return raw.view(np.float32).reshape(-1, vocab)


def load_margins(path):
    rows = [line.split() for line in open(path) if line.strip()]
    return np.array([float(r[2]) for r in rows])


def cmd_exact(a):
    ref, got = load_tokens(a.ref)[:a.n], load_tokens(a.got)[:a.n]
    m = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
    ok = m == a.n and len(got) >= a.n
    print(f"  [1/3] exact prefix : {m}/{a.n} tokens identical -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"        first mismatch at #{m}: expected {ref[m] if m < len(ref) else 'EOF'}, "
              f"got {got[m] if m < len(got) else 'EOF'}")
    return 0 if ok else 1


def cmd_teacher(a):
    ref = load_tokens(a.ref)[:a.n]
    margins = load_margins(a.margins)[:a.n]
    lg = load_logits(a.tuff, a.vocab)[:a.n]
    n = min(len(ref), len(margins), len(lg))
    top1 = lg[:n].argmax(1)
    bad, ties = [], []
    for i in range(n):
        if top1[i] != ref[i]:
            (bad if margins[i] >= a.min_margin else ties).append(i)
    agree = n - len(bad) - len(ties)
    ok = not bad and n == a.n
    print(f"  [2/3] teacher-forced: {agree}/{n} top-1 agree; {len(ties)} near-tie miss(es) "
          f"(ref margin < {a.min_margin}), {len(bad)} hard miss(es) -> {'PASS' if ok else 'FAIL'}")
    for i in ties:
        print(f"        near-tie step {i}: ref {ref[i]} (margin {margins[i]:.3f}), tuff {top1[i]}")
    for i in bad:
        print(f"        HARD MISS step {i}: ref {ref[i]} (margin {margins[i]:.3f}), tuff {top1[i]}")
    return 0 if ok else 1


def cmd_tolerance(a):
    fp = load_logits(a.fp64, a.vocab)
    tu = load_logits(a.tuff, a.vocab)
    ll = load_logits(a.llama, a.vocab)
    n = min(a.n, len(fp), len(tu), len(ll))
    rms = lambda x: np.sqrt(((x[:n].astype(np.float64) - fp[:n]) ** 2).mean(1))
    r_tu, r_ll = rms(tu), rms(ll)
    mean_ratio = r_tu.mean() / r_ll.mean()
    step_ratio = (r_tu / r_ll).max()
    ok = mean_ratio <= a.max_ratio and step_ratio <= a.max_step_ratio and n == a.n
    print(f"  [3/3] FP64 tolerance: logit RMS vs FP64 over {n} steps: tuff {r_tu.mean():.4f}, "
          f"llama.cpp {r_ll.mean():.4f} (ratio {mean_ratio:.2f} <= {a.max_ratio}; "
          f"worst step {step_ratio:.2f} <= {a.max_step_ratio}) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("exact")
    p.add_argument("--ref", required=True)
    p.add_argument("--got", required=True)
    p.add_argument("-n", type=int, required=True)
    p.set_defaults(fn=cmd_exact)

    p = sub.add_parser("teacher")
    p.add_argument("--ref", required=True)
    p.add_argument("--margins", required=True)
    p.add_argument("--tuff", required=True, help="tuff --teacher-force --dump-logits output")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("-n", type=int, required=True)
    p.add_argument("--min-margin", type=float, default=0.5)
    p.set_defaults(fn=cmd_teacher)

    p = sub.add_parser("tolerance")
    p.add_argument("--fp64", required=True)
    p.add_argument("--tuff", required=True)
    p.add_argument("--llama", required=True)
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("-n", type=int, required=True)
    p.add_argument("--max-ratio", type=float, default=1.10)
    p.add_argument("--max-step-ratio", type=float, default=2.0)
    p.set_defaults(fn=cmd_tolerance)

    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
