#!/usr/bin/env bash
# Marginal-gain A/B harness for H21/H22/H23 (iteration 50 hypotheses).
#
# Each hypothesis is an env flag, default OFF, so the baseline is the current build with no
# flag set. Every measurement is order-balanced (alternates which arm runs first) because the
# effects are <2% and thermal drift is ~0.6-0.8%/min.
#
# Usage:
#   ./scripts/marginal_ab.sh <TUFF_*_FLAG=1> [rounds]
#
# Example:
#   ./scripts/marginal_ab.sh TUFF_ROUTER_Q8=1 4
#
# Reports the paired ratio B/A per round and the mean +/- spread. Treat any |win| <2% as
# provisional until it survives >=4 order-balanced rounds.
set -uo pipefail
cd "$(dirname "$0")/.."

FLAG="${1:?usage: $0 <ENV_FLAG=1> [rounds]}"
ROUNDS="${2:-4}"
MODEL="${MODEL:-/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
BIN=./build/tuff

if ! ./scripts/check_power_state.sh >/dev/null 2>&1; then
    echo "ABORT: machine throttled — see ./scripts/check_power_state.sh" >&2
    [ "${FORCE:-0}" = "1" ] || exit 1
fi

run() {
    local env_flag="$1"
    local out
    if [ -n "$env_flag" ]; then
        out=$(env "$env_flag" "$BIN" --model "$MODEL" --prompt "Hello" -n 64 --warmup 32 --runs 3 2>&1)
    else
        out=$("$BIN" --model "$MODEL" --prompt "Hello" -n 64 --warmup 32 --runs 3 2>&1)
    fi
    echo "$out" | grep -oP 'Pipeline Generation Speed:\s*\K[\d.]+' | head -1
}

echo "marginal A/B: flag=$FLAG, rounds=$ROUNDS"
echo "protocol: -n 64 --warmup 32 --runs 3, alternating order"
echo

results=$(mktemp)
for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then
        a=$(run ""); b=$(run "$FLAG"); order="base,flag"
    else
        b=$(run "$FLAG"); a=$(run ""); order="flag,base"
    fi
    ratio=$(awk -v a="$a" -v b="$b" 'BEGIN{ if(a+0>0) printf "%.4f", b/a; else print "ERR" }')
    printf 'round %d (%s): base=%-8s flag=%-8s ratio=%s\n' "$r" "$order" "$a" "$b" "$ratio"
    echo "$ratio" >> "$results"
done

echo
awk '{n++; s+=$1; ss+=$1*$1} END {
    if (n<2) { print "not enough rounds"; exit }
    m=s/n; sd=sqrt((ss-n*m*m)/(n-1));
    printf "paired ratio (flag/base) over %d rounds: %.4f +/- %.4f  (%+.2f%% +/- %.2f%%)\n", n, m, sd, (m-1)*100, sd*100;
    if (m-2*sd <= 1.0 && m+2*sd >= 1.0) print "VERDICT: interval includes 1.0 — NOT established.";
    else if (m > 1.0) print "VERDICT: flag is faster (provisional).";
    else print "VERDICT: flag is SLOWER.";
}' "$results"
rm -f "$results"
