#!/usr/bin/env bash
# tests/verify_lfm25_oracle.sh - LFM2.5 correctness gate (three stages, all must pass).
#
# Usage: verify_lfm25_oracle.sh [dense|moe8|moe]   (dense = 1.2B, moe8 = 8B-A1B, moe = 24B-A2B)
#
#   [1/3] exact prefix    first N_EXACT (16) greedy tokens identical to the llama.cpp reference,
#                         cut short at the first reference step with margin < MIN_MARGIN
#   [2/3] teacher-forced  top-1 agreement on all N (128) reference steps; a miss is allowed only
#                         where the reference itself is a near-tie (llama.cpp margin < MIN_MARGIN),
#                         or where the FP64 logit gap between reference and tuff token is
#                         < MIN_MARGIN (FP64-adjudicated; covers the first N_FP steps)
#   [3/3] FP64 tolerance  over N_FP (32; moe 72) teacher-forced steps, tuff's logit RMS error vs an FP64
#                         numpy reference is no worse than llama.cpp's own. dense: mean ratio
#                         <= MAX_RATIO and worst step <= 2.0. MoE: robust statistic (router
#                         near-ties flip experts, heavy-tailed spikes): median ratio <= MAX_RATIO
#                         and p90 ratio <= MAX_P90_RATIO
#
# All thresholds live in tests/lfm25_gate_thresholds.txt (locked); an env override of any of them makes
# the run EXPLORATORY (exit 3, never a pass).
# The reference comes from tests/gen_lfm25_ref.sh (recipe recorded next to it). Rationale for
# not requiring a long exact-match run: see tools/lfm25_gate.py and LFM25_ROADMAP.md.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-dense}"
PROMPT="${PROMPT:-Hello}"

# Gate parameters are LOCKED in tests/lfm25_gate_thresholds.txt. An environment override that differs
# from the locked value is allowed for exploration, but the run is then marked EXPLORATORY (exit 3).
THR="tests/lfm25_gate_thresholds.txt"
[ -f "$THR" ] || { echo "Error: $THR missing" >&2; exit 1; }
THR_SHA="$(sha256sum "$THR" | cut -c1-12)"
locked() {
    local v
    v="$(sed -n "s/^$1=\([^#[:space:]]*\).*/\1/p" "$THR" | head -1)"
    [ -n "$v" ] || { echo "Error: $1 missing in $THR" >&2; exit 1; }
    echo "$v"
}
OVERRIDES=()
# param NAME LOCKED_VALUE: take env override if set, record it when it differs numerically/textually.
param() {
    local name="$1" lock="$2" cur
    cur="${!name:-$lock}"
    if [ "$cur" != "$lock" ] && ! awk -v a="$cur" -v b="$lock" 'BEGIN{exit !(a+0==b+0 && a ~ /^[0-9.]+$/)}'; then
        OVERRIDES+=("$name=$cur (locked $lock)")
    fi
    printf -v "$name" '%s' "$cur"
}
param N "$(locked N)"
param N_EXACT "$(locked N_EXACT)"
param MIN_MARGIN "$(locked MIN_MARGIN)"
param MAX_RATIO "$(locked MAX_RATIO)"
param MAX_STEP_RATIO "$(locked MAX_STEP_RATIO)"
param MAX_P90_RATIO "$(locked MAX_P90_RATIO)"
# Context for the tuff runs: prompt + N tokens fits easily. Without it tuff sizes the KV cache for
# the model's full 128k context (~3.1 GiB FP32 on 1.2B), so the gate could not share the GPU.
param CTX "$(locked CTX)"

REF_EXTRA=()
if [ "$MODE" = "dense" ]; then
    MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf}"
    REF="tests/ref_lfm25_dense_128.txt"
    STAT="$(locked STAT_DENSE)"
    param N_FP "$(locked N_FP_DENSE)"
elif [ "$MODE" = "moe8" ]; then
    MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf}"
    REF="tests/ref_lfm25_moe8_128.txt"
    STAT="$(locked STAT_MOE8)"
    param N_FP "$(locked N_FP_MOE8)"
elif [ "$MODE" = "moe" ]; then
    MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf}"
    REF="tests/ref_lfm25_moe_128.txt"
    STAT="$(locked STAT_MOE)"
    param N_FP "$(locked N_FP_MOE)"   # 72 steps so FP64 can adjudicate the noisy CPU-expert reference
    # Same recipe as tests/gen_lfm25_ref.sh moe: llama.cpp keeps the experts on its CPU backend.
    REF_EXTRA=(--cpu-experts)
else
    echo "Unknown mode: $MODE (choose 'dense', 'moe8' or 'moe')" >&2
    exit 2
fi
MARGINS="${REF%.txt}.margins.txt"

BIN="${TUFF_BIN:-./build/tuff}"
REF_BIN="${REF_BIN:-./build/lfm25-llama-ref}"

# Interpreter with numpy: $PYTHON, then the local-llm venv, then python3.
pick_python() {
    if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
    for cand in /home/nii/workspace/local-llm/.venv/bin/python python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy" >/dev/null 2>&1; then
            echo "$cand"; return
        fi
    done
    echo "python3"
}
PY="$(pick_python)"

for f in "$REF" "$MARGINS"; do
    [ -f "$f" ] || { echo "Error: $f missing; run tests/gen_lfm25_ref.sh $MODE" >&2; exit 1; }
done
[ -x "$BIN" ] || { echo "Error: $BIN not found; build tuff first." >&2; exit 1; }
[ -x "$REF_BIN" ] || { echo "Error: $REF_BIN not found; cmake --build build --target lfm25-llama-ref" >&2; exit 1; }

TMP_DIR="/tmp/tuff_oracle"
mkdir -p "$TMP_DIR"
GOT="$TMP_DIR/got_lfm25_${MODE}.txt"
ERR="$TMP_DIR/err_lfm25_${MODE}.txt"
TF_LOGITS="$TMP_DIR/tuff_tf_lfm25_${MODE}.bin"
TF_ERR="$TMP_DIR/tuff_tf_lfm25_${MODE}.err"
LL_LOGITS="$TMP_DIR/llama_tf_lfm25_${MODE}.bin"

echo "LFM2.5 oracle gate ($MODE): $MODEL_PATH"
echo "thresholds: $THR sha256=$THR_SHA"

# Run tuff: greedy prefix, then the teacher-forced pass with per-step logits.
"$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" -c "$CTX" -n "$N_EXACT" --temp 0 --dump-tokens > "$GOT" 2> "$ERR" \
    || { tail -20 "$ERR" >&2; exit 1; }
"$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 \
    --teacher-force "$REF" --dump-logits "$TF_LOGITS" > /dev/null 2> "$TF_ERR" || true
[ -s "$TF_LOGITS" ] || { tail -20 "$TF_ERR" >&2; echo "Error: tuff produced no teacher-forced logits" >&2; exit 1; }

# Both runtimes must see the same prompt tokens, or every comparison below is meaningless.
TUFF_PROMPT="$(sed -n 's/^\[Tuff\] Prompt tokens ([0-9]*): //p' "$ERR" | xargs)"
REF_PROMPT="$(sed -n 's/^prompt tokens ([0-9]*): *//p' "${REF%.txt}.recipe" 2>/dev/null | xargs)"
if [ -n "$REF_PROMPT" ] && [ "$TUFF_PROMPT" != "$REF_PROMPT" ]; then
    echo "Error: prompt tokenization differs: tuff [$TUFF_PROMPT] vs reference [$REF_PROMPT]" >&2
    exit 1
fi
VOCAB="$(od -An -tu4 -j12 -N4 "$TF_LOGITS" | xargs)"

# llama.cpp teacher-forced logits (seconds on GPU) and the FP64 reference (cached per model+ref).
"$REF_BIN" --model "$MODEL_PATH" --prompt "$PROMPT" -n "$N_FP" --teacher "$REF" \
    --logits-out "$LL_LOGITS" ${REF_EXTRA[@]+"${REF_EXTRA[@]}"} 2> /dev/null
KEY="$(stat -c '%s-%Y' "$MODEL_PATH")-$(sha1sum "$REF" | cut -c1-12)-$N_FP"
FP_LOGITS="$TMP_DIR/fp64_lfm25_${MODE}_${KEY}.bin"
if [ ! -s "$FP_LOGITS" ]; then
    echo "  (computing FP64 reference for $N_FP steps, ~1 min dense, ~3 min 8B MoE, 24B: see roadmap; cached afterwards)"
    "$PY" -I tools/lfm25_fp64_ref.py --model "$MODEL_PATH" --prompt-tokens "$TUFF_PROMPT" \
        --tokens "$REF" -n "$N_FP" --out "$FP_LOGITS.tmp" 2> "$TMP_DIR/fp64.err" \
        || { tail -20 "$TMP_DIR/fp64.err" >&2; exit 1; }
    mv "$FP_LOGITS.tmp" "$FP_LOGITS"
fi

echo "=========================================================="
rc=0
"$PY" tools/lfm25_gate.py exact --ref "$REF" --got "$GOT" -n "$N_EXACT" \
    --margins "$MARGINS" --min-margin "$MIN_MARGIN" || rc=1
"$PY" tools/lfm25_gate.py teacher --ref "$REF" --margins "$MARGINS" --tuff "$TF_LOGITS" \
    --vocab "$VOCAB" -n "$N" --min-margin "$MIN_MARGIN" --fp64 "$FP_LOGITS" || rc=1
"$PY" tools/lfm25_gate.py tolerance --fp64 "$FP_LOGITS" --tuff "$TF_LOGITS" --llama "$LL_LOGITS" \
    --vocab "$VOCAB" -n "$N_FP" --stat "$STAT" \
    --max-ratio "$MAX_RATIO" --max-step-ratio "$MAX_STEP_RATIO" --max-p90-ratio "$MAX_P90_RATIO" || rc=1
echo "=========================================================="
if [ "$rc" -eq 0 ]; then
    echo "  ORACLE PASS ($MODE)"
else
    echo "  ORACLE FAIL ($MODE)"
fi
echo "=========================================================="
if [ "${#OVERRIDES[@]}" -gt 0 ]; then
    list="$(printf '%s, ' "${OVERRIDES[@]}")"
    echo "EXPLORATORY (thresholds overridden: ${list%, }) — NOT A GATE RESULT"
    [ "$rc" -eq 0 ] && exit 3
fi
exit "$rc"
