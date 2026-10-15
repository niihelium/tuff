#!/usr/bin/env bash
# Interleaved, order-balanced A/B between two tuff trees.
#
# WHY THIS EXISTS
# ---------------
# Measuring tree A, then tree B, and comparing the two means is not sound on a thermally
# throttling laptop: whichever is measured second is measured hotter. The mlxfast-challenge
# harness (docs/thermal-variance-investigation.md) quantifies this on Apple silicon and the
# shape of the problem is identical here:
#
#   - pairing in one session cancels common-mode host/hour variance, but it does NOT cancel
#     time-ordered drift — the candidate runs later, therefore hotter;
#   - they measure decode drift at 0.6-0.8 %/min (up to +20% over ~25 min) and within-host
#     decode CV of 14.4%, dominated by intermittent ~2x throttle SPIKES rather than smooth drift;
#   - their fix is a paired same-session measurement behind a thermal gate, plus an
#     ADJACENCY REORDER for the residual drift.
#
# Tuff's P4a result (23.03 -> 25.04 tok/s, +8.7%) was measured as two separate sessions
# minutes apart. A drift of that size can be manufactured by ordering alone, and the reported
# +/- spreads are within-run so they cannot detect it. This script re-measures properly.
#
# WHAT IT DOES
# ------------
# Runs R rounds. Each round runs both trees back to back, ALTERNATING which goes first
# (A,B then B,A then A,B ...) so a monotonic drift cancels in the round-mean instead of
# loading onto one tree. Reports the per-round paired ratio B/A and the spread across rounds,
# which is the number that actually carries an error bar.
#
# Also records GPU temperature and VRAM clock before every run, so drift and throttle spikes
# are visible in the output rather than silently folded into the result.
#
# Usage:  ./scripts/ab_interleaved.sh <treeA> <treeB> [rounds]
#   treeA/treeB: directories each containing build/tuff
# Env:  N (tokens, default 24)  WARMUP (default 16)  RUNS (default 3)
#       MODEL (gguf path -> --model; unset = tuff default)  TUFF_ARGS (extra args, e.g. "-c 1024")

set -uo pipefail
cd "$(dirname "$0")/.."

A="${1:?usage: $0 <treeA> <treeB> [rounds]}"
B="${2:?usage: $0 <treeA> <treeB> [rounds]}"
ROUNDS="${3:-4}"
N="${N:-24}"; WARMUP="${WARMUP:-16}"; RUNS="${RUNS:-3}"

for t in "$A" "$B"; do
    [ -x "$t/build/tuff" ] || { echo "missing $t/build/tuff" >&2; exit 2; }
done

if ! ./scripts/check_power_state.sh >/dev/null 2>&1; then
    echo "ABORT: machine throttled — see ./scripts/check_power_state.sh" >&2
    [ "${FORCE:-0}" = "1" ] || exit 1
fi

gpu_state() {
    nvidia-smi --query-gpu=temperature.gpu,clocks.mem,clocks.sm,power.draw \
               --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# One measurement. Echoes "tok/s<TAB>gpu_state_before".
measure() {
    local tree="$1" st log tok
    st=$(gpu_state)
    log=$(mktemp)
    ( cd "$tree" && ./build/tuff ${MODEL:+--model "$MODEL"} --prompt "${PROMPT:-Hello}" -n "$N" --warmup "$WARMUP" --runs "$RUNS" ${TUFF_ARGS:-} ) >"$log" 2>&1
    tok=$(awk '/Pipeline Generation Speed:/{print $4; exit}' "$log")
    rm -f "$log"
    printf '%s\t%s' "${tok:-ERR}" "$st"
}

echo "interleaved A/B — A=$A  B=$B"
echo "protocol: -n $N --warmup $WARMUP --runs $RUNS, $ROUNDS rounds, order alternates each round"
echo "gpu state = temp_C,mem_MHz,sm_MHz,power_W (sampled before each run)"
echo
printf '%-6s %-7s %-9s %-24s %-9s %-24s %-8s\n' round order A_tok/s A_gpu_before B_tok/s B_gpu_before ratio
results=$(mktemp)

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then
        order="A,B"
        IFS=$'\t' read -r a_tok a_st < <(measure "$A")
        IFS=$'\t' read -r b_tok b_st < <(measure "$B")
    else
        order="B,A"
        IFS=$'\t' read -r b_tok b_st < <(measure "$B")
        IFS=$'\t' read -r a_tok a_st < <(measure "$A")
    fi
    ratio=$(awk -v a="$a_tok" -v b="$b_tok" 'BEGIN{ if(a+0>0) printf "%.4f", b/a; else print "ERR" }')
    printf '%-6s %-7s %-9s %-24s %-9s %-24s %-8s\n' "$r" "$order" "$a_tok" "$a_st" "$b_tok" "$b_st" "$ratio"
    echo "$ratio" >> "$results"
done

echo
awk '{n++; s+=$1; ss+=$1*$1} END {
    if (n<2) { print "not enough rounds"; exit }
    m=s/n; sd=sqrt((ss-n*m*m)/(n-1));
    printf "paired ratio B/A over %d rounds: %.4f +/- %.4f  (%+.2f%% +/- %.2f)\n", n, m, sd, (m-1)*100, sd*100;
    lo=m-2*sd; hi=m+2*sd;
    printf "approx 95%% interval: [%.4f, %.4f] = [%+.2f%%, %+.2f%%]\n", lo, hi, (lo-1)*100, (hi-1)*100;
    if (lo <= 1.0 && hi >= 1.0) print "VERDICT: interval includes 1.0 -- the difference is NOT established.";
    else if (lo > 1.0) print "VERDICT: B is faster than A.";
    else print "VERDICT: B is SLOWER than A.";
}' "$results"
rm -f "$results"
