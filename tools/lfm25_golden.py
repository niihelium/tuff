#!/usr/bin/env python3
"""Golden self-regression helper for the LFM2.5 port (driven by tests/verify_lfm25_golden.sh).

Pure stdlib (no numpy). Run as:  python -I tools/lfm25_golden.py <cmd> ...

  env-check   compare the golden's model identity + environment fingerprint with this machine/build
  check       compare one configuration's greedy tokens + teacher-forced logits file with the golden
  write       write a golden JSON from a (verified) configuration's outputs

A golden freezes the output of a VALIDATED build, bit for bit: the 128 greedy tokens, and the sha256
of every teacher-forced logit row (plus of the whole file). Any later change that is meant to be a pure
optimization/refactor must reproduce it byte-for-byte. It is environment-specific (GPU, driver, nvcc,
arch flags): on a different environment it is not comparable and must be regenerated after a full
oracle-gate pass. The full golden logits are too large for git; the golden JSON stores per-step top-1
ids and hashes, and a local cache (GOLDEN_CACHE, default /tmp/tuff_golden) of the golden logits is
filled on the first passing check and used for max |delta logit| diagnostics on a later failure.
"""
import argparse
import array
import datetime
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

MAGIC = b"CLBRLGT1"
HDR = 16
IDENT_BYTES = 64 << 20


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while True:
            n = 1 << 20 if left is None else min(1 << 20, left)
            if n == 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001 - fingerprint must never crash the gate, only be visibly "unknown"
        return f"unknown ({type(e).__name__})"


def cmake_var(bin_path, key):
    cache = os.path.join(os.path.dirname(os.path.abspath(bin_path)), "CMakeCache.txt")
    try:
        for line in open(cache):
            m = re.match(rf"^{re.escape(key)}:[A-Z]+=(.*)$", line.strip())
            if m:
                return m.group(1)
    except OSError:
        return "unknown (no CMakeCache.txt)"
    return "unset"


def environment(bin_path):
    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip().splitlines()
    gpu = gpu[0] if gpu else "unknown"
    nv = [ln.strip() for ln in run(["nvcc", "--version"]).splitlines() if "release" in ln]
    return {
        "gpu_and_driver": gpu,
        "nvcc": nv[0] if nv else "unknown",
        "cuda_architectures": cmake_var(bin_path, "CMAKE_CUDA_ARCHITECTURES"),
        "per_thread_stream": cmake_var(bin_path, "TUFF_PER_THREAD_STREAM"),
    }


def model_identity(path):
    return {"size": os.path.getsize(path), "sha256_first_64MiB": sha256_file(path, IDENT_BYTES)}


def read_tokens(path):
    return [int(x) for x in open(path) if x.strip()]


def read_logits(path):
    """-> (vocab, [row bytes ...]); validates magic/size."""
    raw = open(path, "rb").read()
    if len(raw) < HDR or raw[:8] != MAGIC:
        sys.exit(f"{path}: not a CLBRLGT1 logits file")
    n, vocab = struct.unpack("<II", raw[8:16])
    rb = vocab * 4
    if len(raw) != HDR + n * rb:
        sys.exit(f"{path}: size {len(raw)} != header-implied {HDR + n * rb}")
    return vocab, [raw[HDR + i * rb: HDR + (i + 1) * rb] for i in range(n)], raw


def floats(row):
    a = array.array("f")
    a.frombytes(row)
    return a


def top1(row):
    a = floats(row)
    return max(range(len(a)), key=a.__getitem__)


def cache_path(g):
    d = os.environ.get("GOLDEN_CACHE", "/tmp/tuff_golden")
    return os.path.join(d, f"lfm25_{g['mode']}.{g['logits_sha256'][:16]}.logits")


def cmd_env_check(a):
    g = json.load(open(a.golden))
    bad = []
    mi = model_identity(a.model)
    if mi != g["model"]:
        bad.append(f"model identity: have {mi}, golden {g['model']}")
    ref_sha = sha256_file(a.ref)
    if ref_sha != g["ref"]["sha256"]:
        bad.append(f"reference token file sha256: have {ref_sha[:16]}.., golden {g['ref']['sha256'][:16]}..")
    env = environment(a.tuff_bin)
    for k, v in env.items():
        if g["environment"].get(k) != v:
            bad.append(f"environment.{k}: have {v!r}, golden {g['environment'].get(k)!r}")
    if bad:
        print(f"[golden] {g['mode']}: FAIL - golden is NOT comparable on this model/environment:")
        for b in bad:
            print(f"    {b}")
        print("    The golden is environment-specific. It must be regenerated after a FULL oracle-gate pass:\n"
              f"      tests/verify_lfm25_golden.sh {g['mode']} --update --reason \"...\"")
        return 1
    return 0


def cmd_check(a):
    g = json.load(open(a.golden))
    toks = read_tokens(a.tokens)
    vocab, rows, raw = read_logits(a.logits)
    n = g["n"]
    fail = False
    # greedy tokens
    want_t = g["tokens"]
    first_tok = next((i for i in range(max(len(toks), len(want_t)))
                      if i >= len(toks) or i >= len(want_t) or toks[i] != want_t[i]), None)
    tok_same = sum(1 for x, y in zip(toks, want_t) if x == y)
    msgs = []
    if first_tok is not None or len(toks) != len(want_t):
        fail = True
        got = toks[first_tok] if first_tok is not None and first_tok < len(toks) else "EOF"
        want = want_t[first_tok] if first_tok is not None and first_tok < len(want_t) else "EOF"
        msgs.append(f"    first differing greedy token: #{first_tok}: got {got}, want {want}")
    # logits
    shas = [hashlib.sha256(r).hexdigest() for r in rows]
    same_rows = sum(1 for x, y in zip(shas, g["row_sha256"]) if x == y)
    bad_steps = [i for i in range(max(len(shas), n)) if i >= len(shas) or shas[i] != g["row_sha256"][i]]
    whole = hashlib.sha256(raw).hexdigest()
    if vocab != g["vocab"] or len(rows) != n:
        fail = True
        msgs.append(f"    logits shape {len(rows)} x {vocab}, golden {n} x {g['vocab']}")
    if bad_steps or whole != g["logits_sha256"]:
        fail = True
        if bad_steps:
            fs = bad_steps[0]
            line = f"    first differing logit step: {fs}; differing steps: {len(bad_steps)}/{n}"
            cp = cache_path(g)
            if fs < len(rows) and os.path.exists(cp):
                gold = open(cp, "rb").read()
                rb = vocab * 4
                gr = lambda i: gold[HDR + i * rb: HDR + (i + 1) * rb]  # noqa: E731
                ga, ha = floats(gr(fs)), floats(rows[fs])
                mx = max(abs(x - y) for x, y in zip(ga, ha))
                changed = sum(1 for i in bad_steps if i < len(rows) and top1(rows[i]) != g["top1"][i])
                line += f"; max |dlogit| at step {fs}: {mx:.6g}; steps with changed top-1: {changed}/{n}"
            else:
                changed = sum(1 for i in bad_steps if i < len(rows) and top1(rows[i]) != g["top1"][i])
                line += (f"; max |dlogit|: n/a (golden logits not cached at {cp}); "
                         f"steps with changed top-1: {changed}/{n}")
            msgs.append(line)
        else:
            msgs.append("    all rows identical but whole-file sha256 differs (header/trailing bytes)")
    verdict = "FAIL" if fail else "PASS"
    print(f"[golden] {g['mode']} {a.config}: tokens {tok_same}/{len(want_t)} identical, "
          f"logits {same_rows}/{n} steps byte-identical -> {verdict}")
    for m in msgs:
        print(m)
    if not fail:  # byte-identical to the golden: safe to (re)fill the local diagnostics cache
        cp = cache_path(g)
        if not os.path.exists(cp):
            try:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                with open(cp + ".tmp", "wb") as f:
                    f.write(raw)
                os.replace(cp + ".tmp", cp)
            except OSError:
                pass
    return 1 if fail else 0


def cmd_write(a):
    toks = read_tokens(a.tokens)
    vocab, rows, raw = read_logits(a.logits)
    if len(toks) != a.n or len(rows) != a.n:
        sys.exit(f"expected {a.n} tokens/rows, got {len(toks)}/{len(rows)}")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = run(["git", "rev-parse", "HEAD"], cwd=root).strip()
    status = run(["git", "status", "--porcelain"], cwd=root).splitlines()
    tracked_dirty = any(not ln.startswith("??") for ln in status)
    g = {
        "format": 1,
        "mode": a.mode,
        "model": model_identity(a.model),
        "prompt": a.prompt,
        "ctx": a.ctx,
        "n": a.n,
        "ref": {"path": a.ref, "sha256": sha256_file(a.ref)},
        "vocab": vocab,
        "tokens": toks,
        "top1": [top1(r) for r in rows],
        "logits_sha256": hashlib.sha256(raw).hexdigest(),
        "row_sha256": [hashlib.sha256(r).hexdigest() for r in rows],
        "environment": environment(a.tuff_bin),
        "provenance": {
            "git_commit": commit,
            "git_tracked_files_dirty": tracked_dirty,
            "git_status_porcelain": status[:40],
            "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reason": a.reason,
            "configs_verified_identical": a.configs.split(","),
        },
    }
    with open(a.out, "w") as f:
        json.dump(g, f, indent=1)
        f.write("\n")
    cp = cache_path(g)
    try:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(cp + ".tmp", cp)
    except OSError:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env-check")
    p.add_argument("--golden", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--tuff-bin", required=True)
    p.set_defaults(fn=cmd_env_check)

    p = sub.add_parser("check")
    p.add_argument("--golden", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--logits", required=True)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("write")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--tuff-bin", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("-n", type=int, required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--logits", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--configs", required=True)
    p.set_defaults(fn=cmd_write)

    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
