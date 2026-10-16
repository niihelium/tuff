#!/usr/bin/env bash
# tests/verify_lfm25_all.sh - every LFM2.5 gate in one go; exit non-zero if anything fails.
#
# Usage: verify_lfm25_all.sh [--controls] [--spec]
#
#   golden dense|dense26|moe8|moe   byte-identity vs the frozen goldens (graph on/off, expert cache)
#   oracle dense|dense26|moe8|moe   three-stage llama.cpp/FP64 gate with locked thresholds
#   router8/24, moeffn8/24     kernel unit tests vs llama.cpp fixtures
#   --controls                 also run the full negative-control suite (tests/lfm25_negative_controls.sh)
#   --spec                     also run the speculative-decoding losslessness check (tests/verify_lfm25_spec.sh; slow, ~6 min)
#
# Env: TUFF_BIN (default ./build/tuff; the unit-test binaries are taken from the same build dir), PYTHON.
# Builds nothing: run `cmake --build build -j16 --target tuff lfm25-llama-ref lfm25-router-test lfm25-moe-ffn-test` first.
set -uo pipefail

cd "$(dirname "$0")/.."

CONTROLS=0
SPEC=0
for a in "$@"; do
    case "$a" in
        --controls) CONTROLS=1 ;;
        --spec) SPEC=1 ;;
        *) echo "Usage: $0 [--controls] [--spec]" >&2; exit 2 ;;
    esac
done

export TUFF_BIN="${TUFF_BIN:-./build/tuff}"
BDIR="$(dirname "$TUFF_BIN")"
M8="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
M24="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"
LOG="$(mktemp -d /tmp/tuff_all.XXXXXX)"
trap 'rm -rf "$LOG"' EXIT

NAMES=(); RES=(); SECS=()
step() {  # <name> <cmd...>
    local name="$1"; shift
    local t0=$SECONDS rc
    echo "=== $name ==="
    "$@" > "$LOG/out" 2>&1; rc=$?
    if [ $rc -ne 0 ]; then tail -25 "$LOG/out"
    elif [ "$name" = "negative controls" ]; then cat "$LOG/out"   # show every [negctl] verdict, not just the last
    else tail -3 "$LOG/out"; fi
    NAMES+=("$name"); SECS+=($((SECONDS - t0)))
    if [ $rc -eq 0 ]; then RES+=("PASS"); else RES+=("FAIL (exit $rc)"); fi
}

for m in dense dense26 moe8 moe; do step "golden $m" ./tests/verify_lfm25_golden.sh "$m"; done
for m in dense dense26 moe8 moe; do step "oracle $m" ./tests/verify_lfm25_oracle.sh "$m"; done
step "router8"   "$BDIR/lfm25-router-test"  "$M8"  tests/fixtures/lfm25_8b_router_cases.bin
step "router24"  "$BDIR/lfm25-router-test"  "$M24" tests/fixtures/lfm25_24b_router_cases.bin
step "moeffn8"   "$BDIR/lfm25-moe-ffn-test" "$M8"  tests/fixtures/lfm25_8b_moe_ffn_cases.bin
step "moeffn24"  "$BDIR/lfm25-moe-ffn-test" "$M24" tests/fixtures/lfm25_24b_moe_ffn_cases.bin
[ "$SPEC" -eq 1 ] && step "spec lossless" ./tests/verify_lfm25_spec.sh
[ "$CONTROLS" -eq 1 ] && step "negative controls" ./tests/lfm25_negative_controls.sh

echo
echo "================ LFM2.5 summary ================"
fail=0
for i in "${!NAMES[@]}"; do
    printf '  %-20s %-14s %4ss\n' "${NAMES[$i]}" "${RES[$i]}" "${SECS[$i]}"
    [ "${RES[$i]}" = "PASS" ] || fail=1
done
echo "================================================"
if [ "$fail" -eq 0 ]; then echo "ALL LFM2.5 GATES PASS"; else echo "LFM2.5 GATES FAIL"; fi
exit "$fail"
