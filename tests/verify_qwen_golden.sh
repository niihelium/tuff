#!/usr/bin/env bash
# tests/verify_qwen_golden.sh - Qwen3.5-35B-A3B golden self-regression gate (byte-identity).
#
# Usage: verify_qwen_golden.sh [plain]
#        verify_qwen_golden.sh --update --reason "<why the golden may change>"
#
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="${TUFF_BIN:-./build/tuff}"
GOLDEN="tests/golden/qwen35.json"
MODEL="/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
REF="tests/ref_qwen35_128.txt"
PROMPT="Hello"
CTX=1024
N=128

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x /home/nii/workspace/local-llm/.venv/bin/python ]; then PY=/home/nii/workspace/local-llm/.venv/bin/python
else PY=python3; fi

UPDATE=0
REASON=""
while [ $# -gt 0 ]; do
    case "$1" in
        plain) ;;
        --update) UPDATE=1 ;;
        --reason) shift; REASON="${1:-}" ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$UPDATE" -eq 1 ] && [ -z "$REASON" ]; then
    echo "Error: --update requires --reason \"<text>\"" >&2
    exit 2
fi

[ -x "$BIN" ] || { echo "Error: $BIN not found; build tuff first." >&2; exit 1; }
[ -f "$MODEL" ] || { echo "Error: $MODEL not found." >&2; exit 2; }
[ -f "$REF" ] || { echo "Error: $REF not found." >&2; exit 2; }

SCRATCH="$(mktemp -d /tmp/tuff_qwen_golden.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT

CONFIG="plain"
echo "=== Running Qwen Golden Gate (config: $CONFIG, fill_budget=0) ==="

# 1. Greedy tokens
env TUFF_FILL_BUDGET=0 "$BIN" --model "$MODEL" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 --dump-tokens \
    > "$SCRATCH/tokens" 2> "$SCRATCH/err1"

# 2. Teacher-forced logits
env TUFF_FILL_BUDGET=0 "$BIN" --model "$MODEL" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 \
    --teacher-force "$REF" --dump-logits "$SCRATCH/logits.bin" \
    > "$SCRATCH/tf.out" 2> "$SCRATCH/err2"

if [ "$UPDATE" -eq 1 ]; then
    "$PY" -I tools/lfm25_golden.py write \
        --out "$GOLDEN" \
        --mode qwen35 \
        --model "$MODEL" \
        --ref "$REF" \
        --tuff-bin "$BIN" \
        --prompt "$PROMPT" \
        --ctx "$CTX" \
        -n "$N" \
        --tokens "$SCRATCH/tokens" \
        --logits "$SCRATCH/logits.bin" \
        --reason "$REASON" \
        --configs "$CONFIG"
    echo "Golden updated at $GOLDEN: reason: $REASON"
    exit 0
fi

"$PY" -I tools/lfm25_golden.py check \
    --golden "$GOLDEN" \
    --config "$CONFIG" \
    --tokens "$SCRATCH/tokens" \
    --logits "$SCRATCH/logits.bin"
