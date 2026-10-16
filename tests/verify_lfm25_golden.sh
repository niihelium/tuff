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
# Which tuff run each feature config exercises (every config does BOTH a greedy `--dump-tokens` run and a
# teacher-forced `--dump-logits` run; an arg only matters for the run that implements it):
#   --tf-batch N [--tf-rollback]   teacher-forced run only (batched verify forward / conv-state rollback)
#   --spec-draft ngram [--spec-k]  greedy run only (speculative decoding is a greedy-decode feature)
#   LFM25_CUDA_GRAPH=0/1           both runs; on 24B (always the expert-cache path) graph=1 and graph=0 are the
#                                  same ungraphed cache path, the pair only guards against env handling changes
#   LFM25_FORCE_EXPERT_CACHE=1 ..  both runs (8B forced through the cache; 24B always uses it)
# Environment hygiene: every LFM25_* variable of the caller is unset for the tuff runs (a warning lists them) and
# only the config's own env is applied. Effective path: tuff prints "[LFM25 CACHE] N slots" iff the expert cache is
# active, so each run's stderr is required to show it exactly when the config says so. There is no log line for
# "CUDA graph enabled" (graph_enabled = LFM25_CUDA_GRAPH!=0 && !expert cache), so "graphed" is established from the
# scrubbed env + the cache line, not observed directly.
# Exit: 0 pass, 1 golden mismatch / incomplete run, 2 setup error (missing model/ref/golden-env, bad usage).
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

# Caller LFM25_* variables must not change what a config runs.
SCRUB=(); SCRUBBED=()
while IFS= read -r v; do
    case "$v" in LFM25_*) SCRUB+=(-u "$v"); SCRUBBED+=("$v") ;; esac
done < <(compgen -e)
if [ "${#SCRUBBED[@]}" -gt 0 ]; then
    echo "WARNING: scrubbed caller environment for tuff runs (unset): ${SCRUBBED[*]}" >&2
fi

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

# Sets MODEL, REF, IS_MOE (0/1), CACHE_ALWAYS (1 = this model always runs the expert cache), CONFIGS ("name|ENV=..;ENV=..|extra tuff args" entries, the last field optional and appended to BOTH
# tuff runs; the first entry is the one stored as the golden).
mode_setup() {
    IS_MOE=0; CACHE_ALWAYS=0
    case "$1" in
        dense)
            MODEL="/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
            REF="tests/ref_lfm25_dense_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0"
                     "tf-batch4||--tf-batch 4" "tf-batch8-rollback||--tf-batch 8 --tf-rollback"
                     "spec-ngram||--spec-draft ngram" "spec-ngram-k7||--spec-draft ngram --spec-k 7") ;;
        dense26)
            MODEL="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-2.6B/LFM2.5-2.6B-Q4_K_M.gguf"
            REF="tests/ref_lfm25_dense26_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0") ;;
        moe8)
            IS_MOE=1
            MODEL="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
            REF="tests/ref_lfm25_moe8_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0"
                     "expert-cache|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=600"
                     "expert-cache-tiny|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300"
                     "expert-cache-tiny-alt|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300;LFM25_PREFETCH=1;LFM25_PINNED_EXPERTS=0;LFM25_CACHE_POLICY=lru"
                     "tf-batch4||--tf-batch 4"
                     "tf-batch4-rollback-cache|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300|--tf-batch 4 --tf-rollback"
                     "spec-ngram||--spec-draft ngram"
                     "spec-ngram-cache|LFM25_FORCE_EXPERT_CACHE=1;LFM25_EXPERT_CACHE_MB=300|--spec-draft ngram") ;;
        moe)
            IS_MOE=1; CACHE_ALWAYS=1
            MODEL="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"
            REF="tests/ref_lfm25_moe_128.txt"
            CONFIGS=("graph=1|LFM25_CUDA_GRAPH=1" "graph=0|LFM25_CUDA_GRAPH=0" "prefetch|LFM25_PREFETCH=1"
                     "tf-batch4||--tf-batch 4" "spec-ngram||--spec-draft ngram") ;;
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
# stderr) if tuff did not produce a complete output. Greedy run: exit 0 and N tokens. Teacher-forced run: exit 0 or 1
# (1 = tuff's own strict 128/128 check) AND the "Teacher-Forced Verification Result" banner AND a logits file of the
# exact size. Both runs must show the expert-cache line iff the config is supposed to use the cache.
run_config() {
    local dir="$1" envs="${2//;/ }" xa="${3:-}" rc cache_want=0
    mkdir -p "$dir"
    if [ "$IS_MOE" -eq 1 ] && { [ "$CACHE_ALWAYS" -eq 1 ] || [[ " $envs " == *" LFM25_FORCE_EXPERT_CACHE=1 "* ]]; }; then cache_want=1; fi
    # shellcheck disable=SC2086
    env ${SCRUB[@]+"${SCRUB[@]}"} $envs "$BIN" --model "$MODEL" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 --dump-tokens $xa \
        > "$dir/tokens" 2> "$dir/err1" || { tail -5 "$dir/err1" >&2; echo "greedy run failed" >&2; return 1; }
    [ "$(wc -l < "$dir/tokens")" -eq "$N" ] || { echo "greedy run produced $(wc -l < "$dir/tokens") tokens, want $N" >&2; return 1; }
    # shellcheck disable=SC2086
    env ${SCRUB[@]+"${SCRUB[@]}"} $envs "$BIN" --model "$MODEL" --prompt "$PROMPT" -c "$CTX" -n "$N" --temp 0 \
        --teacher-force "$REF" --dump-logits "$dir/logits" $xa > /dev/null 2> "$dir/err2" && rc=0 || rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then tail -5 "$dir/err2" >&2; echo "teacher-forced run exited $rc" >&2; return 1; fi
    grep -q "Teacher-Forced Verification Result" "$dir/err2" || { tail -5 "$dir/err2" >&2; echo "teacher-forced run printed no result banner" >&2; return 1; }
    [ -s "$dir/logits" ] || { tail -5 "$dir/err2" >&2; echo "teacher-forced run produced no logits" >&2; return 1; }
    local vocab want got e has
    vocab="$(od -An -tu4 -j12 -N4 "$dir/logits" | xargs)"
    want=$((16 + N * vocab * 4)); got="$(stat -c %s "$dir/logits")"
    [ "$got" -eq "$want" ] || { echo "logits file is $got bytes, want $want" >&2; return 1; }
    for e in err1 err2; do
        has=0; grep -q '^\[LFM25 CACHE\] [0-9]* slots' "$dir/$e" && has=1
        [ "$has" -eq "$cache_want" ] || { echo "effective path mismatch ($e): expert cache expected=$cache_want seen=$has" >&2; return 1; }
    done
    return 0
}

check_mode() {
    local mode="$1" golden="tests/golden/lfm25_$1.json" rc=0 entry name envs xargs_ f
    mode_setup "$mode"
    for f in "$MODEL" "$REF"; do
        [ -f "$f" ] || { echo "[golden] $mode: setup error - $f missing"; echo "GOLDEN ERROR ($mode)"; return 2; }
    done
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
    else check_mode "$m"; r=$?; [ "$r" -gt "$overall" ] && overall=$r; fi
done
exit "$overall"
