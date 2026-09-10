#!/usr/bin/env bash
# Fast oracle check for the iteration loop: compares tuff --dump-tokens against the
# saved llama.cpp reference (scripts/ref_hello32.txt). The live server-based comparison is
# scripts/verify_vs_llama.sh; this is the same comparison with a cached reference so the loop
# is fast (no 20 GB llama-server load each time).
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
PROMPT="${PROMPT:-Hello}"
N="${N:-32}"
REF="scripts/ref_hello32.txt"
export TUFF_EXACT="${TUFF_EXACT:-1}"

if [ ! -f "$REF" ]; then
    echo "missing reference file $REF" >&2
    exit 2
fi

# PLAN.md sec.2 rule 7: the gate must run the same configuration as the benchmark. Pass the
# benchmark's flags through TUFF_ARGS, e.g. TUFF_ARGS="--n-gpu-expert-layers 10".
TUFF_ARGS="${TUFF_ARGS:-}"
[ -n "$TUFF_ARGS" ] && echo "oracle_check: extra args: $TUFF_ARGS" >&2

mkdir -p /tmp/opencode
# shellcheck disable=SC2086
"$PWD/build/tuff" --model "$MODEL_PATH" --prompt "$PROMPT" -n "$N" $TUFF_ARGS --dump-tokens > /tmp/opencode/got.txt 2>/tmp/opencode/got_err.txt
rc=$?
if [ $rc -ne 0 ]; then
    echo "tuff exited $rc" >&2
    tail -5 /tmp/opencode/got_err.txt >&2
    exit 1
fi

if diff -q "$REF" /tmp/opencode/got.txt >/dev/null; then
    echo "ORACLE PASS: all $N tokens identical"
    exit 0
else
    python3 - "$REF" /tmp/opencode/got.txt <<'EOF'
import sys
ref = [int(x) for x in open(sys.argv[1])]
got = [int(x) for x in open(sys.argv[2])]
m = 0
for a, b in zip(ref, got):
    if a != b:
        break
    m += 1
print(f"ORACLE FAIL: {m}/{len(ref)} leading tokens match")
print(f"  ref: {ref[:8]}")
print(f"  got: {got[:8]}")
EOF
    exit 1
fi
