#!/usr/bin/env bash
# Qwen3.5 oracle gate: greedy continuation of "Hello" + the 5 runtime corpus prompts,
# chat-templated with thinking off, must equal llama.cpp's greedy stream for an exact
# prefix of 32 tokens.  A divergence is accepted only at a position where llama.cpp's
# own top-2 logit margin is below NEAR_TIE (0.15).  The kernel dequantizes Q8_0 and
# accumulates in fp32; llama.cpp quantizes the *activations* to Q8_1 for its mmvq path.
# Measured on the same 5-prompt corpus, the exact-fp32 reference and llama.cpp differ by
# max |dlogit| 0.95 (p99 0.30, std 0.03-0.09); a top-2 margin below 0.15 is therefore
# inside llama.cpp's own kernel-to-kernel quantization noise, not a tuf correctness signal.
#
# usage: tests/qwen35/oracle.sh /path/to/qwen35.gguf [N]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="${1:?usage: oracle.sh MODEL.gguf [N]}"
N="${2:-32}"
NEAR_TIE=0.15

TUF="$ROOT/build/qwen35-teacher"
LLAMA_REF="${QWEN35_LLAMA_REF:-$ROOT/build/qwen35-llama-ref}"
TOKENIZE="${LLAMA_TOKENIZE:-/home/nii/src/llama.cpp/build/bin/llama-tokenize}"
export LD_LIBRARY_PATH="/home/nii/src/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for f in "$TUF" "$LLAMA_REF" "$TOKENIZE"; do
  [ -x "$f" ] || { echo "oracle: missing executable $f" >&2; exit 2; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - "$MODEL" "$N" "$NEAR_TIE" "$TUF" "$LLAMA_REF" "$TOKENIZE" "$ROOT" "$TMP" <<'PY'
import json, os, re, subprocess, sys

model, N, near_tie, tuf, llama, tokenize, root, tmp = sys.argv[1:9]
N = int(N); near_tie = float(near_tie)
corpus = json.load(open(os.path.join(root, "prompts_corpus.json")))
prompts = [("hello", "Hello")] + [(c["id"], c["prompt"]) for c in corpus]

def chat(text):
    return ("<|im_start|>user\n" + text + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def tokenize_text(path):
    r = run([tokenize, "-m", model, "-f", path, "--ids", "--no-bos"])
    if r.returncode != 0:
        print("tokenize failed:", r.stderr[-500:]); sys.exit(2)
    m = re.search(r"\[([0-9,\s]*)\]", r.stdout)
    if not m:
        print("no ids in tokenize output:", r.stdout[-500:]); sys.exit(2)
    return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]

def write_tokens(path, toks):
    open(path, "w").write(" ".join(map(str, toks)))

def llama_greedy(toks, tag):
    tf = os.path.join(tmp, f"{tag}_ids.txt")
    write_tokens(tf, toks)
    out = os.path.join(tmp, f"{tag}_llama.txt")
    lg = os.path.join(tmp, f"{tag}_llama.f32")
    r = run([llama, "--model", model, "--tokens", tf, "--logits-out", lg,
             "--tokens-out", out, "--generate", str(N)])
    if r.returncode != 0:
        print("llama ref failed:", r.stderr[-800:]); sys.exit(2)
    lines = [int(x) for x in open(out).read().split()]
    # lines = one teacher argmax per input token, then N generated tokens
    return lines[-N:]

def tuf_greedy(toks, tag):
    tf = os.path.join(tmp, f"{tag}_ids.txt")
    write_tokens(tf, toks)
    out = os.path.join(tmp, f"{tag}_tuf.txt")
    lg = os.path.join(tmp, f"{tag}_tuf.f32")
    r = run([tuf, "--model", model, "--tokens", tf, "--logits-out", lg,
             "--greedy", str(N), "--greedy-out", out])
    if r.returncode != 0:
        print("tuf failed:", r.stderr[-800:]); sys.exit(2)
    return [int(x) for x in open(out).read().split()]

def llama_margin_at(toks, idx):
    """Top1-top2 margin of llama.cpp at the prediction whose position is idx."""
    tf = os.path.join(tmp, "marg_ids.txt")
    write_tokens(tf, toks)
    mf = os.path.join(tmp, "marg.txt")
    lg = os.path.join(tmp, "marg.f32")
    r = run([llama, "--model", model, "--tokens", tf, "--logits-out", lg,
             "--margins-out", mf])
    if r.returncode != 0:
        print("llama margin run failed:", r.stderr[-500:]); sys.exit(2)
    for line in open(mf):
        p = line.split()
        if int(p[0]) == idx:
            return float(p[2])
    return -1.0

fails = 0
for tag, text in prompts:
    ptxt = os.path.join(tmp, f"{tag}.txt")
    open(ptxt, "w").write(chat(text))
    toks = tokenize_text(ptxt)
    tl = llama_greedy(toks, tag)
    tt = tuf_greedy(toks, tag)
    k = 0
    while k < N and k < len(tl) and k < len(tt) and tl[k] == tt[k]:
        k += 1
    status = "PASS"
    extra = f"prefix {k}/{N}"
    if k < N:
        # margin of llama at the position that predicts the first divergent token
        idx = len(toks) - 1 + k
        marg = llama_margin_at(toks + tt[:k], idx)
        extra += f" diverge@{k} tuf={tt[k]} llama={tl[k]} margin={marg:.4f}"
        if marg >= near_tie:
            status = "FAIL"
            fails += 1
        else:
            extra += " (near-tie allowed)"
    print(f"{status} {tag:32s} prompt_tokens={len(toks):4d} {extra}")

print(f"oracle: {'PASS' if fails == 0 else 'FAIL'} ({fails} failures, N={N}, near_tie={near_tie})")
sys.exit(0 if fails == 0 else 1)
PY
