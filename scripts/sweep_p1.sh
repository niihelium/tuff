#!/usr/bin/env bash
# P1 tuning sweep — finish the fill-budget / worker-count search that iteration 17 started
# and never completed. Iteration 17 shipped TUFF_FILL_BUDGET=2 / TUFF_MOE_WORKERS=4 as
# defaults with an unverified in-progress note claiming budget 1 reaches ~25 tok/s. This
# script settles that with a real sweep.
#
# Refuses to run on a throttled machine — see scripts/check_power_state.sh and the
# measurement-validity notice in HONEST_BENCHMARK_REPORT.md. On battery this same binary
# measures 8.09 tok/s where it measures 21.90 on AC, so an unguarded sweep would "tune" the
# engine against a machine state that does not exist during real use.
set -uo pipefail
cd "$(dirname "$0")/.."

if ! ./scripts/check_power_state.sh; then
    echo
    echo "ABORT: machine is throttled; a sweep run now would produce non-admissible numbers."
    echo "Override only if you know what you are doing: FORCE=1 $0"
    [ "${FORCE:-0}" = "1" ] || exit 1
    echo "FORCE=1 set — continuing, but these numbers MUST NOT be recorded as results."
fi

N="${N:-16}"
WARMUP="${WARMUP:-8}"
RUNS="${RUNS:-3}"
BUDGETS="${BUDGETS:-0 1 2 3 4}"
WORKERS="${WORKERS:-2 3 4 6}"
OUT="${OUT:-/tmp/tuff_sweep_p1.tsv}"

printf 'budget\tworkers\ttok_s\tstddev\thit_rate\n' | tee "$OUT"

for b in $BUDGETS; do
  for w in $WORKERS; do
    log=$(mktemp)
    TUFF_PROFILE=1 TUFF_FILL_BUDGET="$b" TUFF_MOE_WORKERS="$w" \
        ./build/tuff -n "$N" --warmup "$WARMUP" --runs "$RUNS" >"$log" 2>&1

    # "Pipeline Generation Speed: 21.90 +/- 1.08 tok/s (...)"
    read -r tok sd < <(awk '/Pipeline Generation Speed:/{print $4, $6; exit}' "$log")
    # "Expert-cache hit rate: 67.0% (...)"
    hit=$(awk '/Expert-cache hit rate:/{print $4; exit}' "$log")

    printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$w" "${tok:-ERR}" "${sd:-ERR}" "${hit:-ERR}" | tee -a "$OUT"
    rm -f "$log"
  done
done

echo
echo "Sweep written to $OUT"
echo "Best:"
sort -k3 -g -r "$OUT" | grep -v '^budget' | head -3
