#!/usr/bin/env bash
# tests/verify_lfm25_golden.sh - LFM2.5 golden self-regression gate (byte-identity).
#
# Usage: verify_lfm25_golden.sh dense|dense26|moe8|moe|all
#        verify_lfm25_golden.sh <mode> --update --reason "<why the golden may change>"
#
# The oracle gate (verify_lfm25_oracle.sh) compares against llama.cpp/FP64 with tolerances, so it cannot
# see bit-level regressions (a reversed warp-reduction order passes it). This gate freezes tuff's own
# validated output - the 128 greedy tokens and the sha256 of every teacher-forced logit row - in
# tests/golden/lfm25_<mode>.json and requires every later build to reproduce it BYTE-FOR-BYTE, in every
# configuration: LFM25_CUDA_GRAPH=1 and =0 (all modes), plus the forced expert cache for moe8.
# Optimizations and refactors must pass this unchanged; only an intentional numerics change may
# regenerate a golden, and only via --update (which first requires a full oracle-gate pass).
#
# A golden is environment-specific (model file, GPU/driver, nvcc, CUDA arch flags): on a different
# environment the check FAILS with a message instead of skipping.
#
# Env: TUFF_BIN (default ./build/tuff), PYTHON, GOLDEN_CACHE (local cache of golden logits, only used
#      for max |delta logit| diagnostics; default /tmp/tuff_golden).
set -uo pipefail

cd "$(dirname "$0")/.."

BIN="${TUFF_BIN:-./build/tuff}"
THR="tests/lfm25_gate_thresholds.txt"
PROMPT="Hello"

locked() {
    local v
    v="$(sed -n "s/^$1=\\([^#[:space:]]*\\).*/\\1/p" "$THR" | head -1)"
    [ -n "$v" ] || { echo "Error: $1 missing in $THR" >&2; exit 1; }
    echo "$v"
}
[ -f "$THR" ] || { echo "Error: $THR missing" >&2; exit 1; }
N="$(locked N)"
CTX="$(locked CTX)"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x /home/nii/workspace/local-llm/.venv/bin/python ]; then PY=/home/nii/workspace/local-llm/.venv/bin/python
else PY=python3; fi
GOLD_PY=("$PY" -I tools/lfm25_golden.py)

usage() { sed -n '2,5p' "$0" >&2; exit 2; }

TARGET=""
UPDATE=0
REASON=""
while [ $# -gt 0 ]; do
    case "$1" in
        dense|dense26|moe8|moe|all) TARGET="$1" ;;
        --update) UPDATE=1 ;;
        --reason) shift; REASON="${1:-}" ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
    shift
done
[ -n "$TARGET" ] || usage
if [ "$UPDATE" -eq 1 ] && [ -z "$REASON" ]; then
    echo "Error: --update requires --reason \"<text>\" (why is it legitimate for the golden to change?)" >&2
    exit 2
fi
if [ "$UPDATE" -eq 0 ] && [ -n "$REASON" ]; then
    echo "Error: --reason only makes sense with --update" >&2
    exit 2
fi
[ -x "$BIN" ] || { echo "Error: $BIN not found; build tuff first." >&2; exit 1; }

SCRATCH="$(mktemp -d /tmp/tuff_golden_run.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT

# Sets MODEL, REF, CONFIGS ("name|ENV=..;ENV=..|extra tuff args" entries, the last field optional and appended to BOTH
# tuff runs; the first entry is the one stored as the golden).
mode_setup() {
    case "$1" in
        dense)
            MODEL="/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
            REF="tests/ref_lfm25_dense_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0"
                     "tf-batch4||--tf-batch 4" "tf-batch8-rollback||--tf-batch 8 --tf-rollback") ;;
        dense26)
            MODEL="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-2.6B/LFM2.5-2.6B-Q4_K_M.gguf"
            REF="tests/ref_lfm25_dense26_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0") ;;
        moe8)
            MODEL="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
            REF="tests/ref_lfm25_moe8_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0"
                     "expert-cache|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=600"
                     "expert-cache-tiny|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300"
                     "expert-cache-tiny-alt|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300;LFM25_PREFETCH=1;LFM25_PINNED_EXPERTS=0;LFM25_CACHE_POLICY=lru"
                     "tf-batch4||--tf-batch 4"
                     "tf-batch4-rollback-cache|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300|--tf-batch 4 --tf-rollback") ;;
        moe)
            MODEL="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"
            REF="tests/ref_lfm25_moe_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0" "prefetch|LFM25_PREFETCH=1"
                     "tf-batch4||--tf-batch 4") ;;
    esac
}

# split_entry <entry>: sets name, envs, xargs.
split_entry() {
    local entry="$1" rest
    name="${entry%%|*}"; rest="${entry#*|}"
    envs="${rest%%|*}"
    if [[ "$rest" == *"|"* ]]; then xargs_="${rest#*|}"; else xargs_=""; fi
}

# run_config <dir> <env-string> [extra tuff args]: writes <dir>/tokens and <dir>/logits. Returns 1 (with a reason on
# stderr) if tuff did not produce a complete output; tuff exits non-zero on the teacher-forced pass
# by design (strict 128/128 check), so success is judged by the file, as in verify_lfm25_oracle.sh.
run_config() {
    local dir="$1" envs="${2//;/ }" xa="${3:-}"
    mkdir -p "$dir"
    # shellcheck disable=SC2086
    env $envs "$BIN" --model "$MODEL" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 --dump-tokens $xa \
        > "$dir/tokens" 2> "$dir/err1" || { tail -5 "$dir/err1" >&2; echo "greedy run failed" >&2; return 1; }
    [ "$(wc -l < "$dir/tokens")" -eq "$N" ] || { echo "greedy run produced $(wc -l < "$dir/tokens") tokens, want $N" >&2; return 1; }
    # shellcheck disable=SC2086
    env $envs "$BIN" --model "$MODEL" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 \
        --teacher-force "$REF" --dump-logits "$dir/logits" $xa > /dev/null 2> "$dir/err2" || true
    [ -s "$dir/logits" ] || { tail -5 "$dir/err2" >&2; echo "teacher-forced run produced no logits" >&2; return 1; }
    local vocab want got
    vocab="$(od -An -tu4 -j12 -N4 "$dir/logits" | xargs)"
    want=$((16 + N * vocab * 4)); got="$(stat -c %s "$dir/logits")"
    [ "$got" -eq "$want" ] || { echo "logits file is $got bytes, want $want" >&2; return 1; }
    return 0
}

check_mode() {
    local mode="$1" golden="tests/golden/lfm25_$1.json" rc=0 entry name envs xargs_
    mode_setup "$mode"
    if [ ! -f "$golden" ]; then
        echo "[golden] $mode: FAIL - $golden missing (create it with: $0 $mode --update --reason \"...\")"
        echo "GOLDEN FAIL ($mode)"; return 1
    fi
    if ! "${GOLD_PY[@]}" env-check --golden "$golden" --model "$MODEL" --ref "$REF" --tuff-bin "$BIN"; then
        echo "GOLDEN FAIL ($mode)"; return 1
    fi
    for entry in "${CONFIGS[@]}"; do
        split_entry "$entry"
        rm -rf "$SCRATCH/$mode-$name"
        if ! run_config "$SCRATCH/$mode-$name" "$envs" "$xargs_"; then
            echo "[golden] $mode $name: tuff run incomplete -> FAIL"; rc=1; continue
        fi
        "${GOLD_PY[@]}" check --golden "$golden" --config "$name" \
            --tokens "$SCRATCH/$mode-$name/tokens" --logits "$SCRATCH/$mode-$name/logits" || rc=1
        rm -f "$SCRATCH/$mode-$name/logits"
    done
    if [ "$rc" -eq 0 ]; then echo "GOLDEN PASS ($mode)"; else echo "GOLDEN FAIL ($mode)"; fi
    return "$rc"
}

update_mode() {
    local mode="$1" golden="tests/golden/lfm25_$1.json" entry name envs xargs_ first="" names=""
    mode_setup "$mode"
    echo "[golden] $mode: update requested (reason: $REASON)"
    echo "[golden] $mode: step 1/3 - full oracle gate must pass first"
    if ! TUFF_BIN="$BIN" PYTHON="${PYTHON:-}" ./tests/verify_lfm25_oracle.sh "$mode"; then
        echo "[golden] $mode: REFUSED - oracle gate did not pass (exit != 0); a golden is never recorded from a failing build"
        return 1
    fi
    echo "[golden] $mode: step 2/3 - all configurations must agree byte-for-byte"
    for entry in "${CONFIGS[@]}"; do
        split_entry "$entry"
        rm -rf "$SCRATCH/$mode-$name"
        run_config "$SCRATCH/$mode-$name" "$envs" "$xargs_" || { echo "[golden] $mode $name: tuff run incomplete - REFUSED"; return 1; }
        names="${names:+$names,}$name"
        if [ -z "$first" ]; then
            first="$name"
        else
            cmp -s "$SCRATCH/$mode-$first/tokens" "$SCRATCH/$mode-$name/tokens" \
                && cmp -s "$SCRATCH/$mode-$first/logits" "$SCRATCH/$mode-$name/logits" \
                || { echo "[golden] $mode: REFUSED - $name differs from $first (tokens or logits); fix the nondeterminism/divergence first"; return 1; }
            echo "[golden] $mode: $name == $first (tokens and logits byte-identical)"
        fi
    done
    echo "[golden] $mode: step 3/3 - writing $golden"
    mkdir -p tests/golden
    "${GOLD_PY[@]}" write --out "$golden" --mode "$mode" --model "$MODEL" --ref "$REF" --tuff-bin "$BIN" \
        --prompt "$PROMPT" --ctx "$CTX" -n "$N" --tokens "$SCRATCH/$mode-$first/tokens" \
        --logits "$SCRATCH/$mode-$first/logits" --reason "$REASON" --configs "$names" || return 1
    echo "GOLDEN UPDATED ($mode) — this is not a pass"
    return 0
}

modes=("$TARGET")
[ "$TARGET" = "all" ] && modes=(dense moe8 moe)
overall=0
for m in "${modes[@]}"; do
    if [ "$UPDATE" -eq 1 ]; then update_mode "$m" || overall=1
    else check_mode "$m" || overall=1; fi
done
exit "$overall"
