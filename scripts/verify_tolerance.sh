#!/usr/bin/env bash
# P0 numeric gate (parity.md sec.3 item 2 / sec.4 P0).
#
# Runs tuff twice on the same prompt with the same greedy decode -- once with
# TUFF_EXACT=1 (bit-exact reference kernels) and once with the fast paths permitted --
# dumping the final logit vector of every generated token, then compares the two dumps with
# scripts/verify_tolerance.py (max abs/rel difference + top-1 agreement).
#
# This is the SECONDARY gate. The primary gate remains scripts/verify_vs_llama.sh: a 32/32
# token-identical greedy continuation against llama.cpp, which is the ground truth. This
# script deliberately does not touch it.
#
# Until any fast kernel actually exists, both runs execute the same code, so the expected
# result is 0.0 difference and 100% top-1 agreement. That zero is the proof the harness is
# wired up; anything else means the harness (or a kernel) is broken.
#
# Env overrides (same conventions as scripts/verify_vs_llama.sh):
#   MODEL_PATH  gguf to load
#   TUFF_BIN tuff binary (default ./build/tuff)
#   PROMPT      prompt text
#   N_TOKENS    tokens to generate per run (default 32; parity.md sec.3 asks for >=128 for
#               the full numeric gate -- run N_TOKENS=128 for that)
#   OUT_DIR     where to keep the two .bin dumps (default: a temp dir, removed on exit)
#   PYTHON      python interpreter with numpy
#   MAX_REL     relative-difference threshold (default 1e-3)
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
TUFF_BIN="${TUFF_BIN:-./build/tuff}"
PROMPT="${PROMPT:-Hello}"
N_TOKENS="${N_TOKENS:-32}"
MAX_REL="${MAX_REL:-1e-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo "        TUFF NUMERIC TOLERANCE GATE (exact vs fast)         "
echo "=================================================================="

if [ ! -x "$TUFF_BIN" ]; then
    echo "Error: tuff binary not found or not executable at $TUFF_BIN."
    echo "       Build it first, or set TUFF_BIN=<path> (e.g. ./build-p0/tuff)."
    exit 2
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: model not found at $MODEL_PATH (set MODEL_PATH=<path>)"
    exit 2
fi

# Pick an interpreter that has numpy: $PYTHON, then the local-llm venv, then python3.
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

if [ -n "${OUT_DIR:-}" ]; then
    mkdir -p "$OUT_DIR"
    KEEP_OUT=1
else
    OUT_DIR="$(mktemp -d)"
    KEEP_OUT=0
fi
EXACT_BIN_OUT="$OUT_DIR/logits_exact.bin"
FAST_BIN_OUT="$OUT_DIR/logits_fast.bin"
EXACT_LOG="$OUT_DIR/exact.stderr.log"
FAST_LOG="$OUT_DIR/fast.stderr.log"

cleanup() {
    if [ "$KEEP_OUT" -eq 0 ]; then rm -rf "$OUT_DIR"; fi
}
trap cleanup EXIT

echo "binary : $TUFF_BIN"
echo "model  : $MODEL_PATH"
echo "prompt : $PROMPT"
echo "tokens : $N_TOKENS"
echo "python : $PY"
echo "dumps  : $OUT_DIR"
echo "------------------------------------------------------------------"

run_dump() {
    # $1 = label, $2 = output path, $3 = log path, $4 = 1 for TUFF_EXACT=1, 0 for unset
    local label="$1" out="$2" log="$3" exact="$4"
    echo "[$label] running tuff --dump-logits (TUFF_EXACT=$( [ "$exact" -eq 1 ] && echo 1 || echo '<unset>' ))..."
    rm -f "$out"
    local rc=0
    if [ "$exact" -eq 1 ]; then
        TUFF_EXACT=1 "$TUFF_BIN" --model "$MODEL_PATH" --prompt "$PROMPT" \
            -n "$N_TOKENS" --dump-logits "$out" >/dev/null 2>"$log" || rc=$?
    else
        env -u TUFF_EXACT "$TUFF_BIN" --model "$MODEL_PATH" --prompt "$PROMPT" \
            -n "$N_TOKENS" --dump-logits "$out" >/dev/null 2>"$log" || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        echo "Error: $label run exited $rc. Last 30 lines of its stderr:"
        tail -n 30 "$log"
        exit 2
    fi
    if [ ! -s "$out" ]; then
        echo "Error: $label run produced no logit dump at $out. Last 30 lines of its stderr:"
        tail -n 30 "$log"
        exit 2
    fi
    echo "[$label] wrote $(stat -c %s "$out") bytes"
}

run_dump "1/3 exact" "$EXACT_BIN_OUT" "$EXACT_LOG" 1
run_dump "2/3 fast " "$FAST_BIN_OUT"  "$FAST_LOG"  0

echo "------------------------------------------------------------------"
echo "[3/3] comparing dumps..."
rc=0
"$PY" "$SCRIPT_DIR/verify_tolerance.py" "$EXACT_BIN_OUT" "$FAST_BIN_OUT" --max-rel "$MAX_REL" || rc=$?

if [ "$rc" -eq 0 ]; then
    echo "TOLERANCE GATE PASSED."
else
    echo "TOLERANCE GATE FAILED (exit $rc). Tuff stderr logs: $EXACT_LOG , $FAST_LOG"
    if [ "$KEEP_OUT" -eq 0 ]; then
        echo "  (set OUT_DIR=<dir> to keep the dumps and logs for inspection)"
    fi
fi
exit "$rc"
