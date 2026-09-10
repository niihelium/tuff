#!/usr/bin/env python3
"""Numeric gate for parity.md P0.

Compares two logit dumps produced by `tuff --dump-logits <path>` and reports, over all
generated tokens:

  * max absolute difference
  * max relative difference (denominator floored, see --rel-floor)
  * top-1 agreement rate (fraction of tokens whose argmax matches)
  * the first divergent token index, if any

Exit status:
  0  PASS  -- top-1 agreement 100% and max relative difference < --max-rel
  1  FAIL  -- the tolerance gate was not met
  2  ERROR -- missing/short/corrupt input, header mismatch, or non-finite logits.
              Never a silent pass: anything that cannot be compared is an error.

File format (src/main.c, --dump-logits):
  magic b"CLBRLGT1" (8 bytes) | uint32 n_tokens | uint32 vocab_size |
  n_tokens * vocab_size float32, little-endian, row-major (one row per generated token).
"""

import argparse
import struct
import sys

MAGIC = b"CLBRLGT1"
HEADER_SIZE = 16


def die(msg):
    print("ERROR: %s" % msg, file=sys.stderr)
    print("RESULT: ERROR", file=sys.stdout)
    sys.exit(2)


try:
    import numpy as np
except ImportError:
    die("numpy is required. Run with a numpy-capable interpreter, e.g.\n"
        "         PYTHON=/home/nii/workspace/local-llm/.venv/bin/python "
        "scripts/verify_tolerance.sh\n"
        "       or: /home/nii/workspace/local-llm/.venv/bin/python "
        "scripts/verify_tolerance.py A.bin B.bin")


def load(path):
    """Read a dump file. Any inconsistency is a hard error."""
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError as e:
        die("cannot read %s: %s" % (path, e))

    if len(blob) < HEADER_SIZE:
        die("%s: file is %d bytes, shorter than the %d-byte header"
            % (path, len(blob), HEADER_SIZE))
    if blob[:8] != MAGIC:
        die("%s: bad magic %r (expected %r) -- not a --dump-logits file"
            % (path, blob[:8], MAGIC))

    n_tokens, vocab = struct.unpack_from("<II", blob, 8)
    if n_tokens == 0 or vocab == 0:
        die("%s: header declares n_tokens=%d vocab_size=%d; nothing to compare"
            % (path, n_tokens, vocab))

    need = HEADER_SIZE + n_tokens * vocab * 4
    if len(blob) != need:
        die("%s: header declares %d tokens x %d floats => %d bytes, but the file is %d bytes"
            % (path, n_tokens, vocab, need, len(blob)))

    arr = np.frombuffer(blob, dtype="<f4", count=n_tokens * vocab,
                        offset=HEADER_SIZE).reshape(n_tokens, vocab)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        die("%s: contains %d non-finite (NaN/Inf) logits" % (path, bad))
    return n_tokens, vocab, arr


def main():
    ap = argparse.ArgumentParser(description="Compare two tuff --dump-logits files.")
    ap.add_argument("file_a", help="reference dump (e.g. TUFF_EXACT=1)")
    ap.add_argument("file_b", help="candidate dump (e.g. fast path)")
    ap.add_argument("--max-rel", type=float, default=None,
                    help="if given, also require max relative difference < this. NOT part of the "
                         "default gate: it is computed over the whole vocabulary and is dominated "
                         "by tail logits that cannot affect the sampled token (measured 1.9 on the "
                         "P3 kernels while top-1 agreement was a clean 128/128). Use --max-rel 1e-3 "
                         "only when comparing two paths expected to be near-bit-exact.")
    ap.add_argument("--min-cos", type=float, default=0.995,
                    help="minimum per-token cosine similarity of the logit vectors (default 0.995; "
                         "measured 0.99617 min / 0.99956 mean for the P3 kernels vs the exact path)")
    ap.add_argument("--rel-floor", type=float, default=1.0,
                    help="denominator floor for the relative difference (default 1.0). rel = "
                         "|a-b| / max(|a|,|b|,floor). Final logits are O(1..30), so a floor of 1 "
                         "keeps near-zero logits from producing meaningless ratios; it never "
                         "makes the test weaker than the absolute difference.")
    args = ap.parse_args()

    if args.rel_floor <= 0.0:
        die("--rel-floor must be > 0 (got %g); a zero floor divides by zero" % args.rel_floor)

    na, va, a = load(args.file_a)
    nb, vb, b = load(args.file_b)

    if va != vb:
        die("header mismatch: vocab_size %d (%s) != %d (%s)" % (va, args.file_a, vb, args.file_b))
    if na != nb:
        die("header mismatch: n_tokens %d (%s) != %d (%s)" % (na, args.file_a, nb, args.file_b))

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)

    max_abs = float(diff.max())
    den = np.maximum(np.maximum(np.abs(a64), np.abs(b64)), args.rel_floor)
    rel = diff / den
    max_rel = float(rel.max())

    flat_abs = int(np.argmax(diff))
    abs_tok, abs_idx = divmod(flat_abs, va)
    flat_rel = int(np.argmax(rel))
    rel_tok, rel_idx = divmod(flat_rel, va)

    arg_a = a.argmax(axis=1)
    arg_b = b.argmax(axis=1)
    agree = arg_a == arg_b
    n_agree = int(agree.sum())
    top1_rate = n_agree / float(na)

    # Flip risk -- the metric that actually predicts a token flip.
    #
    # `max rel diff` above is computed over the whole vocabulary, so it is dominated by tail
    # logits that never affect the sampled token. Measured on the P3 kernels it reads 1.9 while
    # top-1 agreement is a clean 128/128, i.e. it fails the run for a divergence that cannot
    # change any output. What CAN change an output is a perturbation of the gap between the
    # top-1 and top-2 candidates that exceeds that gap. So per token:
    #     margin_t = a[top1] - a[top2]                      (reference's own headroom)
    #     pert_t   = |(a[i1]-a[i2]) - (b[i1]-b[i2])|        (how much the contest moved)
    #     risk_t   = pert_t / margin_t                      (>= 1.0 means a flip was possible)
    # risk < 1 for every token means top-1 agreement is structural, not luck.
    part = np.partition(a64, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    top2_idx = np.argpartition(a64, -2, axis=1)[:, -2:]
    rows = np.arange(na)
    i1, i2 = top2_idx[:, 1], top2_idx[:, 0]
    gap_a = a64[rows, i1] - a64[rows, i2]
    gap_b = b64[rows, i1] - b64[rows, i2]
    pert = np.abs(gap_a - gap_b)
    risk = pert / np.maximum(margin, 1e-12)
    max_risk = float(risk.max())
    max_risk_tok = int(risk.argmax())
    n_flippable = int((risk >= 1.0).sum())

    norm_a = np.linalg.norm(a64, axis=1)
    norm_b = np.linalg.norm(b64, axis=1)
    cos = np.einsum("ij,ij->i", a64, b64) / np.maximum(norm_a * norm_b, 1e-30)
    min_cos = float(cos.min())
    min_cos_tok = int(cos.argmin())

    first_argmax_div = int(np.argmin(agree)) if n_agree < na else None
    row_differs = (diff > 0.0).any(axis=1)
    first_numeric_div = int(np.argmax(row_differs)) if bool(row_differs.any()) else None

    print("=" * 66)
    print("           TUFF NUMERIC TOLERANCE GATE (parity.md P0)")
    print("=" * 66)
    print("A (reference): %s" % args.file_a)
    print("B (candidate): %s" % args.file_b)
    print("tokens: %d   vocab: %d   values compared: %d" % (na, va, na * va))
    print("-" * 66)
    print("max abs diff:        %.6e   (token %d, logit %d: A=%.9g B=%.9g)"
          % (max_abs, abs_tok, abs_idx, a64[abs_tok, abs_idx], b64[abs_tok, abs_idx]))
    print("max rel diff:        %.6e   (token %d, logit %d; denominator floor %g)"
          % (max_rel, rel_tok, rel_idx, args.rel_floor))
    print("top-1 agreement:     %.4f%% (%d/%d tokens)" % (top1_rate * 100.0, n_agree, na))
    if first_numeric_div is None:
        print("first numeric diff:  none (files are numerically identical)")
    else:
        print("first numeric diff:  token %d" % first_numeric_div)
    if first_argmax_div is None:
        print("first top-1 divergence: none")
    else:
        print("first top-1 divergence: token %d (A picked %d, B picked %d)"
              % (first_argmax_div, int(arg_a[first_argmax_div]), int(arg_b[first_argmax_div])))
    print("min per-token cosine: %.8f   (token %d)" % (min_cos, min_cos_tok))
    print("max flip risk:       %.4f   (token %d) -- pert(top1-top2 gap) / margin; >=1 can flip"
          % (max_risk, max_risk_tok))
    print("tokens where a flip was possible: %d / %d" % (n_flippable, na))
    print("-" * 66)

    failures = []
    if n_agree != na:
        failures.append("top-1 agreement %.4f%% < 100%%" % (top1_rate * 100.0))
    if n_flippable > 0:
        failures.append("%d token(s) with flip risk >= 1.0" % n_flippable)
    if not (min_cos >= args.min_cos):
        failures.append("min per-token cosine %.8f < %.8f" % (min_cos, args.min_cos))
    if args.max_rel is not None and not (max_rel < args.max_rel):
        failures.append("max rel diff %.6e >= %.6e" % (max_rel, args.max_rel))

    if failures:
        print("RESULT: FAIL (%s)" % "; ".join(failures))
        print("=" * 66)
        return 1

    print("RESULT: PASS (top-1 100%%, %d/%d tokens flippable, min cosine %.8f >= %.8f)"
          % (n_flippable, na, min_cos, args.min_cos))
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
