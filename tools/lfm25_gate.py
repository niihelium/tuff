#!/usr/bin/env python3
"""Checks for the LFM2.5 oracle gate (driven by tests/verify_lfm25_oracle.sh).

  exact      stage 1: the first N greedy tokens must be identical to the reference, cut short at
             the first step whose reference margin < --min-margin (adaptive; 0 -> skipped).
  teacher    stage 2: teacher-forced top-1 agreement. A step may disagree only where the
             reference itself is a near-tie (llama.cpp top1-top2 margin < --min-margin), or,
             with --fp64, where ground truth does not prefer the reference token over tuff's by
             >= --min-margin (FP64 l[ref] - l[tuff]; negative when FP64 sides with tuff) -- for
             noisy references, e.g. the CPU-expert 24B one.
  tolerance  stage 3: logit error versus the FP64 reference must be no worse than
             llama.cpp's own. --stat mean (dense, default): mean RMS ratio <= --max-ratio and worst
             step <= --max-step-ratio. --stat robust (MoE): median ratio <= --max-ratio and p90
             ratio <= --max-p90-ratio; the worst step and mean are printed for information only.
             Rationale: near-tied router scores (4th vs 5th expert) make quantization noise flip
             which expert runs, giving heavy-tailed per-step error spikes in both runtimes; which
             runtime gets a flip on a given step is luck, so mean/worst-step are ill-posed there.

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
    n = a.n
    why = ""
    if a.margins:
        margins = load_margins(a.margins)
        cut = next((i for i, mg in enumerate(margins[:a.n]) if mg < a.min_margin), None)
        if cut is not None:
            n = cut
            why = f" (cut from {a.n}: reference near-tie at step {cut}, margin {margins[cut]:.3f} < {a.min_margin})"
    if n == 0:
        print(f"  [1/3] exact prefix : skipped (near-tie at step 0) -> PASS")
        return 0
    m = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
    ok = m >= n and len(got) >= n
    print(f"  [1/3] exact prefix : {min(m, n)}/{n} tokens identical{why} -> {'PASS' if ok else 'FAIL'}")
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
    fp = load_logits(a.fp64, a.vocab) if a.fp64 else None
    bad, ties, adj, notes = [], [], [], {}
    for i in range(n):
        if top1[i] != ref[i]:
            if margins[i] < a.min_margin:
                ties.append(i)
            elif fp is not None and i < len(fp):
                # Signed: how much ground truth (FP64) prefers the reference token over tuff's. A miss counts
                # against tuff only if FP64 prefers the reference by >= min_margin; if FP64 prefers tuff's token
                # (gap < 0) the reference itself was wrong there.
                gap = float(fp[i][ref[i]]) - float(fp[i][top1[i]])
                if gap < a.min_margin:
                    adj.append((i, gap))
                else:
                    bad.append(i)
                    notes[i] = f" (FP64 gap {gap:.3f} >= {a.min_margin})"
            else:
                bad.append(i)
                if fp is not None:
                    notes[i] = f" (not covered by FP64 ({len(fp)} steps) - raise N_FP)"
    agree = n - len(bad) - len(ties) - len(adj)
    ok = not bad and n == a.n
    extra = f", {len(adj)} FP64-adjudicated near-tie(s)" if adj else ""
    print(f"  [2/3] teacher-forced: {agree}/{n} top-1 agree; {len(ties)} near-tie miss(es) "
          f"(ref margin < {a.min_margin}){extra}, {len(bad)} hard miss(es) -> {'PASS' if ok else 'FAIL'}")
    for i in ties:
        print(f"        near-tie step {i}: ref {ref[i]} (margin {margins[i]:.3f}), tuff {top1[i]}")
    for i, gap in adj:
        print(f"        near-tie (FP64-adjudicated) step {i}: ref {ref[i]}, tuff {top1[i]} "
              f"(ref margin {margins[i]:.3f}, FP64 gap {gap:.3f})")
    for i in bad:
        print(f"        HARD MISS step {i}: ref {ref[i]} (margin {margins[i]:.3f}), tuff {top1[i]}{notes.get(i, '')}")
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
    if a.stat == "robust":
        med = np.median(r_tu) / np.median(r_ll)
        p90 = np.percentile(r_tu, 90) / np.percentile(r_ll, 90)
        ok = med <= a.max_ratio and p90 <= a.max_p90_ratio and n == a.n
        print(f"  [3/3] FP64 tolerance (robust, MoE routing flips): median ratio {med:.2f} <= {a.max_ratio}; "
              f"p90 ratio {p90:.2f} <= {a.max_p90_ratio} (info: mean {mean_ratio:.2f}, "
              f"worst step {step_ratio:.2f}) -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
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
    p.add_argument("--margins", help="reference margins file; the required prefix is cut at the first near-tie")
    p.add_argument("--min-margin", type=float, default=0.5)
    p.set_defaults(fn=cmd_exact)

    p = sub.add_parser("teacher")
    p.add_argument("--ref", required=True)
    p.add_argument("--margins", required=True)
    p.add_argument("--tuff", required=True, help="tuff --teacher-force --dump-logits output")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("-n", type=int, required=True)
    p.add_argument("--min-margin", type=float, default=0.5)
    p.add_argument("--fp64", help="FP64 logits (raw float32 [n x vocab]); adjudicates hard misses whose "
                   "FP64 preference for the reference token l[ref]-l[tuff] < --min-margin as near-ties "
                   "(negative = FP64 prefers tuff's token)")
    p.set_defaults(fn=cmd_teacher)

    p = sub.add_parser("tolerance")
    p.add_argument("--fp64", required=True)
    p.add_argument("--tuff", required=True)
    p.add_argument("--llama", required=True)
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("-n", type=int, required=True)
    p.add_argument("--max-ratio", type=float, default=1.10)
    p.add_argument("--max-step-ratio", type=float, default=2.0)
    p.add_argument("--stat", choices=["mean", "robust"], default="mean")
    p.add_argument("--max-p90-ratio", type=float, default=1.5)
    p.set_defaults(fn=cmd_tolerance)

    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
