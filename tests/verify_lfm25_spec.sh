#!/usr/bin/env bash
# tests/verify_lfm25_spec.sh [dense|moe8|moe|all] - losslessness check for n-gram speculative decoding (Task 5.2).
# For every model and prompt: greedy `--dump-tokens -n 256` without speculation, with `--spec-draft ngram`
# (adaptive k) and with `--spec-draft ngram` + LFM25_SPEC_K=7 forced; the token files must be byte-identical (cmp).
# The last prompt is a repetitive one so drafts, accepts and rollbacks happen for certain. Slow (one GPU process
# at a time, 3 runs per prompt per model). Prints PASS/FAIL per case with the acceptance stats; exit 1 on any difference.
# Speculation must actually happen, or byte-identity proves nothing: every spec run must print its
# `[Tuff] spec stats:` line; on the repeat prompt the adaptive and the k=7 run must have drafted>0; and per model at
# least one k=7 run must have a rejected draft (accepted < drafted), i.e. the rollback path ran. Otherwise SPEC FAIL.
# Exit: 0 pass, 1 SPEC FAIL (difference / no speculation), 2 SPEC ERROR (setup: missing model, tuff run failed).
# Env: SPEC_JOBS (default 1: one `tuff --jobs` process per model; 0: one process per run), TUFF_BIN (default ./build/tuff). Caller LFM25_* variables are unset for the tuff runs (warning printed).
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

SCRUB=(); SCRUBBED=()
while IFS= read -r v; do
    case "$v" in LFM25_*) SCRUB+=(-u "$v"); SCRUBBED+=("$v") ;; esac
done < <(compgen -e)
if [ "${#SCRUBBED[@]}" -gt 0 ]; then
    echo "WARNING: scrubbed caller environment for tuff runs (unset): ${SCRUBBED[*]}" >&2
fi
T() { env ${SCRUB[@]+"${SCRUB[@]}"} "$@"; }

tmp="$(mktemp -d /tmp/tuff_spec.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
fail=0; err=0; ncase=0
sfield() {  # <file> <drafted|accepted>: value from the last spec stats line, empty if the line is missing
    sed -n "s/.*spec stats: steps=[0-9]* drafted=\\([0-9]*\\) accepted=\\([0-9]*\\).*/\\1 \\2/p" "$1" | tail -1 | { read -r d a; [ "$2" = drafted ] && echo "$d" || echo "$a"; }
}
stat() { sed -n 's/.*spec stats: steps=\([0-9]*\) drafted=\([0-9]*\) accepted=\([0-9]*\) acceptance=\([0-9.]*\)% tokens\/step=\([0-9.]*\).*/drafted=\2 acc=\4% tok\/step=\5/p' "$1" | tail -1; }
q() { local s="${1//\\/\\\\}"; s="${s//\"/\\\"}"; printf '"%s"' "$s"; }   # quote for a tuff jobs line
# exec_model <label> <model>: runs the 3 variants of every prompt; files $tmp/<label>.<name>.{base,spec,k7}, .err, .rc.
# SPEC_JOBS=1 (default): ONE tuff --jobs process per model (the model is loaded once); SPEC_JOBS=0: one process per run.
exec_model() {
    local label="$1" model="$2" i n p v jf="$tmp/$1.jobs"
    : > "$jf"
    for i in "${!NAMES[@]}"; do
        n="${NAMES[$i]}"; p="${PROMPTS[$i]}"
        for v in base spec k7; do
            local f="$tmp/$label.$n.$v" xa="" ev=""
            [ "$v" != base ] && xa="--spec-draft ngram"
            [ "$v" = k7 ] && { xa="--spec-draft ngram --spec-k 7"; ev="LFM25_SPEC_K=7"; }
            if [ "${SPEC_JOBS:-1}" = 0 ]; then
                # shellcheck disable=SC2086
                T $ev "$BIN" --model "$model" -c 1024 -n 256 --temp 0 --prompt "$p" --dump-tokens $xa > "$f" 2> "$f.err"; echo $? > "$f.rc"
            else
                printf '%s -c 1024 -n 256 --temp 0 --prompt %s --dump-tokens %s --out %s --err %s --status %s\n' \
                    "$ev" "$(q "$p")" "$xa" "$f" "$f.err" "$f.rc" >> "$jf"
            fi
        done
    done
    [ "${SPEC_JOBS:-1}" = 0 ] || T "$BIN" --model "$model" --jobs "$jf" > /dev/null 2> "$tmp/$label.runner.err" || true
}
run_model() {  # <label> <model>
    local label="$1" model="$2" i n p res why d_ad a_ad d_k7 a_k7 rejected=0 b
    [ -f "$model" ] || { echo "Error: model $model not found" >&2; err=1; return; }
    exec_model "$label" "$model"
    for i in "${!NAMES[@]}"; do
        n="${NAMES[$i]}"; p="${PROMPTS[$i]}"; b="$tmp/$label.$n"
        [ "$(cat "$b.base.rc" 2>/dev/null)" = 0 ] || { echo "$label $n: baseline run failed"; tail -3 "$b.base.err"; err=1; continue; }
        [ "$(wc -l < "$b.base")" -eq 256 ] || { echo "$label $n: baseline produced $(wc -l < "$b.base") tokens"; err=1; continue; }
        [ "$(cat "$b.spec.rc" 2>/dev/null)" = 0 ] || { echo "$label $n: spec run failed"; tail -3 "$b.spec.err"; err=1; continue; }
        [ "$(cat "$b.k7.rc" 2>/dev/null)" = 0 ] || { echo "$label $n: spec k=7 run failed"; tail -3 "$b.k7.err"; err=1; continue; }
        ncase=$((ncase + 1))
        res="PASS"
        cmp -s "$b.base" "$b.spec" || res="FAIL(adaptive)"
        cmp -s "$b.base" "$b.k7" || { [ "$res" = PASS ] && res="FAIL(k=7)" || res="$res+FAIL(k=7)"; }
        # speculation must really have happened
        d_ad="$(sfield "$b.spec.err" drafted)"; a_ad="$(sfield "$b.spec.err" accepted)"
        d_k7="$(sfield "$b.k7.err" drafted)"; a_k7="$(sfield "$b.k7.err" accepted)"
        why=""
        if [ -z "$d_ad" ] || [ -z "$d_k7" ]; then why="no spec stats line"
        elif [ "$n" = repeat ] && { [ "$d_ad" -le 0 ] || [ "$d_k7" -le 0 ]; }; then why="drafted=0 on the repeat prompt (adaptive $d_ad, k=7 $d_k7)"
        fi
        [ -z "$why" ] || { [ "$res" = PASS ] && res="FAIL(no-spec)" || res="$res+FAIL(no-spec)"; echo "$label $n: $why"; }
        [ -n "$d_k7" ] && [ "$a_k7" -lt "$d_k7" ] && rejected=1
        [ "$res" = PASS ] || fail=1
        printf '%-6s %-12s %-18s adaptive: %-38s k=7: %s\n' "$label" "$n" "$res" "$(stat "$b.spec.err")" "$(stat "$b.k7.err")"
    done
    if [ "$rejected" -eq 0 ]; then echo "$label: no rejected draft in any k=7 run (accepted == drafted everywhere) - rollback path never exercised"; fail=1; fi
}
[ "$TARGET" = all ] || [ "$TARGET" = dense ] && run_model dense "$M_DENSE"
[ "$TARGET" = all ] || [ "$TARGET" = moe8 ] && run_model moe8 "$M_MOE8"
[ "$TARGET" = all ] || [ "$TARGET" = moe ] && run_model moe "$M_MOE"
if [ "$err" -ne 0 ]; then echo "SPEC ERROR (setup)"; exit 2; fi
if [ "$fail" -eq 0 ] && [ "$ncase" -gt 0 ]; then echo "SPEC PASS ($ncase cases)"; exit 0; fi
echo "SPEC FAIL"; exit 1
