#!/usr/bin/env bash
# tests/verify_lfm25_spec.sh [dense|moe8|moe|all] - losslessness check for n-gram speculative decoding (Task 5.2).
# For every model and prompt: greedy `--dump-tokens -n 256` without speculation, with `--spec-draft ngram`
# (adaptive k) and with `--spec-draft ngram` + LFM25_SPEC_K=7 forced; the token files must be byte-identical (cmp).
# The last prompt is a repetitive one so drafts, accepts and rollbacks happen for certain. Slow (one GPU process
# at a time, 3 runs per prompt per model). Prints PASS/FAIL per case with the acceptance stats; exit 1 on any difference.
# Env: TUFF_BIN (default ./build/tuff).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${TUFF_BIN:-./build/tuff}"
TARGET="${1:-all}"
case "$TARGET" in dense|moe8|moe|all) ;; *) echo "Usage: $0 [dense|moe8|moe|all]" >&2; exit 2 ;; esac
[ -x "$BIN" ] || { echo "Error: $BIN not found; build tuff first." >&2; exit 1; }

M_DENSE="/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
M_MOE8="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
M_MOE="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"
NAMES=(fibonacci io-async json-schema quicksort sqrt2-proof repeat)
PROMPTS=(
  "Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization."
  "Explain the difference between synchronous and asynchronous I/O in distributed operating systems."
  "Generate a JSON configuration schema for a high-availability PostgreSQL cluster with replication slots."
  "Write a quicksort algorithm in C with in-place Lomuto partitioning and explain its time complexity."
  "Prove that the square root of 2 is an irrational number using proof by contradiction."
  "Repeat this sequence exactly: 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8"
)

tmp="$(mktemp -d /tmp/tuff_spec.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
fail=0; ncase=0
stat() { sed -n 's/.*spec stats: steps=\([0-9]*\) drafted=\([0-9]*\) accepted=\([0-9]*\) acceptance=\([0-9.]*\)% tokens\/step=\([0-9.]*\).*/drafted=\2 acc=\4% tok\/step=\5/p' "$1" | tail -1; }
run_model() {  # <label> <model>
    local label="$1" model="$2" i n p res
    [ -f "$model" ] || { echo "Error: model $model not found" >&2; fail=1; return; }
    for i in "${!NAMES[@]}"; do
        n="${NAMES[$i]}"; p="${PROMPTS[$i]}"
        "$BIN" --model "$model" -c 1024 -n 256 --temp 0 --prompt "$p" --dump-tokens > "$tmp/base" 2> "$tmp/base.err" \
            || { echo "$label $n: baseline run failed"; tail -3 "$tmp/base.err"; fail=1; continue; }
        [ "$(wc -l < "$tmp/base")" -eq 256 ] || { echo "$label $n: baseline produced $(wc -l < "$tmp/base") tokens"; fail=1; continue; }
        "$BIN" --model "$model" -c 1024 -n 256 --temp 0 --prompt "$p" --dump-tokens --spec-draft ngram > "$tmp/spec" 2> "$tmp/spec.err" \
            || { echo "$label $n: spec run failed"; tail -3 "$tmp/spec.err"; fail=1; continue; }
        LFM25_SPEC_K=7 "$BIN" --model "$model" -c 1024 -n 256 --temp 0 --prompt "$p" --dump-tokens --spec-draft ngram --spec-k 7 > "$tmp/k7" 2> "$tmp/k7.err" \
            || { echo "$label $n: spec k=7 run failed"; tail -3 "$tmp/k7.err"; fail=1; continue; }
        ncase=$((ncase + 1))
        res="PASS"
        cmp -s "$tmp/base" "$tmp/spec" || res="FAIL(adaptive)"
        cmp -s "$tmp/base" "$tmp/k7" || { [ "$res" = PASS ] && res="FAIL(k=7)" || res="$res+FAIL(k=7)"; }
        [ "$res" = PASS ] || fail=1
        printf '%-6s %-12s %-18s adaptive: %-38s k=7: %s\n' "$label" "$n" "$res" "$(stat "$tmp/spec.err")" "$(stat "$tmp/k7.err")"
    done
}
[ "$TARGET" = all ] || [ "$TARGET" = dense ] && run_model dense "$M_DENSE"
[ "$TARGET" = all ] || [ "$TARGET" = moe8 ] && run_model moe8 "$M_MOE8"
[ "$TARGET" = all ] || [ "$TARGET" = moe ] && run_model moe "$M_MOE"
if [ "$fail" -eq 0 ] && [ "$ncase" -gt 0 ]; then echo "SPEC PASS ($ncase cases)"; else echo "SPEC FAIL"; fi
exit "$fail"
