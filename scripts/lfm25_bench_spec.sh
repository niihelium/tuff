#!/usr/bin/env bash
# scripts/lfm25_bench_spec.sh [model.gguf] - lossless n-gram speculation benchmark (Task 5.2).
# For each of 9 prompts (the 5 of bench_ngram_suite.sh + the 4 of lfm25_bench_moe.sh) runs the baseline and
# `--spec-draft ngram` back to back (-c 1024 -n 256 --temp 0 --runs 3, one process at a time) and prints:
# prompt | baseline tok/s | spec tok/s | ratio | acceptance | tokens/step.  The last line reports the worst ratio.
# Extra env passes through (LFM25_SPEC_K fixes k; LFM25_* cache knobs). TUFF_BIN overrides the binary,
# SPEC_ARGS adds tuff arguments to the speculative run only (e.g. "--spec-k 7").
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${TUFF_BIN:-./build/tuff}"
MODEL="${1:-/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf}"
[ -x "$BIN" ] || { echo "Error: $BIN not found" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "Error: model $MODEL not found" >&2; exit 1; }

NAMES=(fibonacci io-async json-schema quicksort sqrt2-proof hello py-csv history translate)
PROMPTS=(
  "Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization."
  "Explain the difference between synchronous and asynchronous I/O in distributed operating systems."
  "Generate a JSON configuration schema for a high-availability PostgreSQL cluster with replication slots."
  "Write a quicksort algorithm in C with in-place Lomuto partitioning and explain its time complexity."
  "Prove that the square root of 2 is an irrational number using proof by contradiction."
  "Hello"
  "Write a Python function that parses a CSV file and computes per-column statistics, with error handling."
  "Explain the causes of the French Revolution and its consequences for European politics in the 19th century."
  "Translate to German and French: The quick brown fox jumps over the lazy dog near the river bank."
)

tmp="$(mktemp -d /tmp/lfm25_bench_spec.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
echo "model: $(basename "$MODEL")   (-c 1024 -n 256 --temp 0 --runs 3; SPEC_ARGS='${SPEC_ARGS:-}' LFM25_SPEC_K='${LFM25_SPEC_K:-}')"
printf '%-12s %10s %10s %7s %11s %11s\n' prompt "base tok/s" "spec tok/s" ratio acceptance "tokens/step"
worst=""; worst_name=""
for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"
    "$BIN" --model "$MODEL" -c 1024 -n 256 --temp 0 --runs 3 --prompt "${PROMPTS[$i]}" > /dev/null 2> "$tmp/$n.base" \
        || { echo "$n: baseline failed" >&2; tail -5 "$tmp/$n.base" >&2; continue; }
    # shellcheck disable=SC2086
    "$BIN" --model "$MODEL" -c 1024 -n 256 --temp 0 --runs 3 --prompt "${PROMPTS[$i]}" --spec-draft ngram ${SPEC_ARGS:-} \
        > /dev/null 2> "$tmp/$n.spec" || { echo "$n: spec run failed" >&2; tail -5 "$tmp/$n.spec" >&2; continue; }
    b="$(sed -n 's/.*Pipeline Generation Speed: \([0-9.]*\) .*/\1/p' "$tmp/$n.base" | head -1)"
    s="$(sed -n 's/.*Pipeline Generation Speed: \([0-9.]*\) .*/\1/p' "$tmp/$n.spec" | head -1)"
    acc="$(sed -n 's/.*spec stats:.*acceptance=\([0-9.]*\)%.*/\1/p' "$tmp/$n.spec" | awk '{s+=$1;c++} END{if(c)printf "%.1f%%",s/c; else print "n/a"}')"
    tps="$(sed -n 's/.*spec stats:.*tokens\/step=\([0-9.]*\) .*/\1/p' "$tmp/$n.spec" | awk '{s+=$1;c++} END{if(c)printf "%.3f",s/c; else print "n/a"}')"
    ratio="$(awk -v a="${b:-0}" -v c="${s:-0}" 'BEGIN{if(a>0)printf "%.3f",c/a; else print "n/a"}')"
    printf '%-12s %10s %10s %7s %11s %11s\n' "$n" "${b:-n/a}" "${s:-n/a}" "$ratio" "$acc" "$tps"
    if [ "$ratio" != "n/a" ] && { [ -z "$worst" ] || awk -v r="$ratio" -v w="$worst" 'BEGIN{exit !(r<w)}'; }; then worst="$ratio"; worst_name="$n"; fi
done
echo "worst ratio: ${worst:-n/a} (${worst_name:-n/a})"
