#!/usr/bin/env bash
# scripts/lfm25_bench_cfg.sh "<label>" "<ENV=.. ENV=..>" [prompt-names...]
# Benchmarks ONE configuration (env string) on the 24B-A2B: -c 1024 --temp 0 --runs 3, n=128 and n=512 for each prompt
# (default: hello translate fibonacci history). One `tuff --jobs` process loads the model once and runs every
# (prompt, n) as a job. Prints one line per job:
#   <label> | <prompt> | n=<N> | tok/s <mean of 3 runs> | hit <cache hit rate %, mean of the runs> | cpuev <cpu-evaluated misses of the last run>
# Env: MODEL (default: the 24B Q4_K_M), TUFF_BIN (default ./build/tuff), BENCH_JOBS=0 -> one tuff process per job (old way).
# The config env string is applied to every job (LFM25_CACHE_STATS=1 is added); the caller's LFM25_* variables are unset.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${TUFF_BIN:-./build/tuff}"
M="${MODEL:-/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf}"
[ $# -ge 2 ] || { echo "usage: $0 \"<label>\" \"<ENV=.. ENV=..>\" [prompt-names...]" >&2; exit 2; }
label="$1"; envs="$2"; shift 2
[ -x "$BIN" ] || { echo "Error: $BIN not found" >&2; exit 1; }
[ -f "$M" ] || { echo "Error: model $M not found" >&2; exit 1; }
declare -A P
P[hello]="Hello"
P[translate]="Translate to German and French: The quick brown fox jumps over the lazy dog near the river bank."
P[fibonacci]="Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization."
P[history]="Explain the causes of the French Revolution and its consequences for European politics in the 19th century."
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(hello translate fibonacci history)
for n in "${names[@]}"; do [ -n "${P[$n]:-}" ] || { echo "unknown prompt name '$n' (hello translate fibonacci history)" >&2; exit 2; }; done

SCRUB=()
while IFS= read -r v; do case "$v" in LFM25_*) SCRUB+=(-u "$v") ;; esac; done < <(compgen -e)
q() { local s="${1//\\/\\\\}"; s="${s//\"/\\\"}"; printf '"%s"' "$s"; }

tmp="$(mktemp -d /tmp/lfm25_bench_cfg.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
Ns=(128 512)
jf="$tmp/jobs"; : > "$jf"
for n in "${names[@]}"; do
    for N in "${Ns[@]}"; do
        f="$tmp/$n.$N"
        if [ "${BENCH_JOBS:-1}" = 0 ]; then
            # shellcheck disable=SC2086
            env ${SCRUB[@]+"${SCRUB[@]}"} $envs LFM25_CACHE_STATS=1 "$BIN" --model "$M" -c 1024 -n "$N" --temp 0 --runs 3 --prompt "${P[$n]}" \
                > "$f.out" 2> "$f.err"
        else
            printf '%s LFM25_CACHE_STATS=1 -c 1024 -n %s --temp 0 --runs 3 --prompt %s --out %s --err %s --status %s\n' \
                "$envs" "$N" "$(q "${P[$n]}")" "$f.out" "$f.err" "$f.rc" >> "$jf"
        fi
    done
done
[ "${BENCH_JOBS:-1}" = 0 ] || env ${SCRUB[@]+"${SCRUB[@]}"} "$BIN" --model "$M" --jobs "$jf" > /dev/null 2> "$tmp/runner.err" || true
for n in "${names[@]}"; do
    for N in "${Ns[@]}"; do
        f="$tmp/$n.$N"
        tps=$(sed -n 's/.*Pipeline Generation Speed: \([0-9.]*\) .*/\1/p' "$f.err" | head -1)
        hit=$(sed -n 's/.*hit rate \([0-9.]*\)%.*/\1/p' "$f.err" | awk '{s+=$1;c++} END{if(c)printf "%.1f",s/c}')
        ev=$(sed -n 's/.*cpu-evaluated misses: \([0-9]*\) .*/\1/p' "$f.err" | tail -1)
        echo "$label | $n | n=$N | tok/s ${tps:-FAILED} | hit ${hit:-n/a}% | cpuev ${ev:-0}"
    done
done
