#!/usr/bin/env bash
# G1 numeric gate: kernel logits vs the independent fp32 GGUF reference, teacher-forced,
# 5 corpus prompts x N positions.  Argmax must agree at every position whose reference
# top-2 margin is >= 0.02, and max |dlogit| must be <= 0.05.
#
# usage: [Q35_BUILD=dir] [Q35_TEACHER_BATCH=N] tests/qwen35/g1.sh MODEL.gguf [N=256] [OUTDIR]
# G1_REUSE=1 reuses the llama/fp32-reference files already in OUTDIR (they do not depend on the kernel build).
# Q35_TEACHER_BATCH=N runs the kernel side on the batch path inside a batch of N slots.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="${1:?usage: g1.sh MODEL.gguf [N] [OUTDIR]}"
N="${2:-256}"
OUT="${3:-$ROOT/tests/qwen35/g1_out}"
NEAR=0.02
TOL=0.05
BUILD="${Q35_BUILD:-$ROOT/build}"   # binaries dir override (e.g. /tmp/claude-1000/q35build-sonnet)
TUF="$BUILD/qwen35-teacher"
LLAMA_REF="$BUILD/qwen35-llama-ref"
TOKENIZE="${LLAMA_TOKENIZE:-/home/nii/src/llama.cpp/build/bin/llama-tokenize}"
export LD_LIBRARY_PATH="/home/nii/src/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/home/nii/src/llama.cpp/gguf-py${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT"
uv run --with numpy --with pyyaml --no-project python3 - "$MODEL" "$N" "$OUT" "$TUF" "$LLAMA_REF" "$TOKENIZE" "$ROOT" "$NEAR" "$TOL" <<'PY'
import json, os, re, subprocess, sys
import numpy as np

model, N, out, tuf, llama, tokenize, root, near, tol = sys.argv[1:10]
N = int(N); near = float(near); tol = float(tol)
corpus = json.load(open(os.path.join(root, "prompts_corpus.json")))
V = 248320

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def tokenize_text(path):
    r = run([tokenize, "-m", model, "-f", path, "--ids", "--no-bos"])
    if r.returncode != 0:
        print("tokenize failed:", r.stderr[-500:]); sys.exit(2)
    m = re.search(r"\[([0-9,\s]*)\]", r.stdout)
    return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]

def chat(t):
    return ("<|im_start|>user\n" + t + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")

def write(p, toks):
    open(p, "w").write(" ".join(map(str, toks)))

fails = 0
for c in corpus:
    tag = c["id"]
    pf = os.path.join(out, f"{tag}.prompt.txt")
    open(pf, "w").write(chat(c["prompt"]))
    ptoks = tokenize_text(pf)
    ids = os.path.join(out, f"{tag}.ids.txt")
    write(ids, ptoks)
    lg = os.path.join(out, f"{tag}.llama.f32")
    tox = os.path.join(out, f"{tag}.llama.tokens.txt")
    reuse = os.environ.get("G1_REUSE") == "1"   # reuse llama/fp32-reference outputs already in OUTDIR
    if reuse and os.path.exists(tox) and os.path.getsize(tox) > 0:
        r = subprocess.CompletedProcess([], 0, "", "")
    else:
        r = run([llama, "--model", model, "--tokens", ids, "--logits-out", lg,
                 "--tokens-out", tox, "--generate", str(N)])
    if r.returncode != 0:
        print(f"FAIL {tag}: llama greedy failed: {r.stderr[-400:]}"); fails += 1; continue
    all_t = [int(x) for x in open(tox).read().split()]
    gen = all_t[-N:]
    seq = ptoks + gen
    sf = os.path.join(out, f"{tag}.seq.txt")
    write(sf, seq)

    rlog = os.path.join(out, f"{tag}.ref.f32")
    rmar = os.path.join(out, f"{tag}.ref.margins.txt")
    if reuse and os.path.exists(rlog) and os.path.exists(rmar) and os.path.getsize(rlog) > 0:
        r = subprocess.CompletedProcess([], 0, "", "")
    else:
        r = run([sys.executable, os.path.join(root, "tools", "qwen35_ref.py"),
                 "--model", model, "--tokens", sf, "--out", rlog, "--margin-out", rmar])
    if r.returncode != 0:
        print(f"FAIL {tag}: fp32 ref failed: {r.stderr[-500:]}"); fails += 1; continue

    klog = os.path.join(out, f"{tag}.kernel.f32")
    kmar = os.path.join(out, f"{tag}.kernel.margins.txt")
    r = run([tuf, "--model", model, "--tokens", sf, "--logits-out", klog,
             "--margins-out", kmar, "--chunk", "16"])
    if r.returncode != 0:
        print(f"FAIL {tag}: kernel failed: {r.stderr[-500:]}"); fails += 1; continue

    A = np.fromfile(klog, dtype=np.float32).reshape(-1, V)
    B = np.fromfile(rlog, dtype=np.float32).reshape(-1, V)
    n = min(len(A), len(B))
    marg = np.array([float(l.split()[2]) for l in open(rmar) if l.strip()])
    bad = 0; worst = 0.0
    for p in range(n):
        d = np.abs(A[p] - B[p]).max()
        worst = max(worst, float(d))
        if marg[p] >= near and int(A[p].argmax()) != int(B[p].argmax()):
            bad += 1
    d = np.abs(A[:n] - B[:n])
    p99 = float(np.percentile(d, 99))
    ok = bad == 0 and worst <= tol
    if not ok: fails += 1
    print(("PASS" if ok else "FAIL") + f" {tag:32s} n={n:4d} argmax_mismatch={bad} "
          f"max|dlogit|={worst:.5f} p99={p99:.5f} (tol {tol}, near {near})")

print(f"G1: {'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")
sys.exit(0 if fails == 0 else 1)
PY
