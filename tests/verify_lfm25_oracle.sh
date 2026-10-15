#!/usr/bin/env bash
# tests/verify_lfm25_oracle.sh - Verify tuff-lfm25 against greedy oracle references
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-dense}" # "dense" or "moe"
N="${N:-128}"
PROMPT="${PROMPT:-Hello}"

if [ "$MODE" = "dense" ]; then
    MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf}"
    REF="tests/ref_lfm25_dense_128.txt"
elif [ "$MODE" = "moe" ]; then
    MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf}"
    REF="tests/ref_lfm25_moe_128.txt"
else
    echo "Unknown mode: $MODE (choose 'dense' or 'moe')" >&2
    exit 2
fi

BIN="${TUFF_LFM25_BIN:-./build/tuff-lfm25}"

if [ ! -f "$REF" ]; then
    echo "Error: reference file $REF does not exist" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "Error: binary $BIN not found or not executable. Compile tuff-lfm25 first." >&2
    exit 1
fi

TMP_DIR="/tmp/tuff_oracle"
mkdir -p "$TMP_DIR"
GOT="$TMP_DIR/got_lfm25_${MODE}.txt"
ERR="$TMP_DIR/err_lfm25_${MODE}.txt"

echo "Running $BIN on $MODEL_PATH for $N tokens..."
"$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" -n "$N" --temp 0 --dump-tokens > "$GOT" 2>"$ERR"

head -n "$N" "$REF" > "$TMP_DIR/ref_truncated.txt"

if diff -q "$TMP_DIR/ref_truncated.txt" "$GOT" >/dev/null 2>&1; then
    echo "=========================================================="
    echo "  ORACLE PASS: All $N tokens identical to ground truth!"
    echo "=========================================================="
    exit 0
else
    python3 - "$TMP_DIR/ref_truncated.txt" "$GOT" <<'EOF'
import sys
ref = [int(x.strip()) for x in open(sys.argv[1]) if x.strip()]
got = [int(x.strip()) for x in open(sys.argv[2]) if x.strip()]
m = 0
for a, b in zip(ref, got):
    if a != b:
        break
    m += 1
print("==========================================================")
print(f"  ORACLE FAIL: {m}/{len(ref)} leading tokens match")
print(f"  First mismatch at token #{m+1}: expected {ref[m] if m < len(ref) else 'EOF'}, got {got[m] if m < len(got) else 'EOF'}")
print(f"  Ref first 8: {ref[:8]}")
print(f"  Got first 8: {got[:8]}")
print("==========================================================")
EOF
    exit 1
fi
