#!/usr/bin/env bash
# scripts/lfm25_bench_moe.sh [model.gguf] - MoE expert-cache benchmark over four diverse prompts.
# Runs each prompt (-c 1024 -n 256 --temp 0 --runs 3, LFM25_CACHE_STATS=1) one at a time and prints a table:
# prompt | cache hit rate | upload GB/s (copy-stream time) | tok/s (mean of 3 runs).
# Extra env passes through (LFM25_PREFETCH, LFM25_CACHE_POLICY, LFM25_PINNED_EXPERTS, ...). TUFF_BIN overrides the binary.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${TUFF_BIN:-./build/tuff}"
MODEL="${1:-/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf}"
[ -x "$BIN" ] || { echo "Error: $BIN not found" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "Error: model $MODEL not found" >&2; exit 1; }

NAMES=(hello code history translate)
PROMPTS=(
  "Hello"
  "Write a Python function that parses a CSV file and computes per-column statistics, with error handling."
  "Explain the causes of the French Revolution and its consequences for European politics in the 19th century."
  "Translate to German and French: The quick brown fox jumps over the lazy dog near the river bank."
)

tmp="$(mktemp -d /tmp/lfm25_bench_moe.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
printf '%-10s %10s %12s %10s %s\n' prompt "hit rate" "upload GB/s" "tok/s" "notes"
for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"
    LFM25_CACHE_STATS=1 "$BIN" --model "$MODEL" -c 1024 -n 256 --temp 0 --runs 3 --prompt "${PROMPTS[$i]}" \
        > "$tmp/$n.out" 2> "$tmp/$n.err" || { echo "$n: tuff failed" >&2; tail -5 "$tmp/$n.err" >&2; continue; }
    # last stats line = last run; average hit rate / GB/s over the 3 runs
    hit="$(sed -n 's/.*hit rate \([0-9.]*\)%.*/\1/p' "$tmp/$n.err" | awk '{s+=$1;n++} END{if(n)printf "%.1f%%",s/n; else print "n/a"}')"
    gbs="$(sed -n 's/.*-> \([0-9.]*\) GB\/s.*/\1/p' "$tmp/$n.err" | grep -v '^$' | awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "n/a"}')"
    tps="$(sed -n 's/.*Pipeline Generation Speed: \([0-9.]*\) .*/\1/p' "$tmp/$n.out" "$tmp/$n.err" | head -1)"
    acc="$(sed -n 's/.*prediction accuracy \([0-9.]*%\).*/\1/p' "$tmp/$n.err" | tail -1)"
    printf '%-10s %10s %12s %10s %s\n' "$n" "$hit" "$gbs" "${tps:-n/a}" "${acc:+pred acc $acc}"
done
