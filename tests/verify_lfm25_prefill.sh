#!/usr/bin/env bash
# tests/verify_lfm25_prefill.sh [dense|moe8|all] - batched prompt prefill (Task 7.1) vs token-by-token prefill.
# The goldens use a one-token prompt and cannot see prefill bugs. This runs a fixed multi-chunk prompt (> 32 tokens, odd
# tail: the length is checked to be neither a multiple of 8 nor below 33) with `--prefill-batch 1` and `--prefill-batch 8`,
# greedy, `--dump-tokens` and `--dump-logits` (the logit rows include the one that selected the first generated token),
# and requires cmp-identical tokens and logits. The batch-8 run must report (TUFF_TIMING) that it really prefilled in
# batches of 8, so a silently disabled batched path cannot pass.
# Exit: 0 PREFILL PASS, 1 PREFILL FAIL, 2 PREFILL ERROR (setup). Env: TUFF_BIN (default ./build/tuff).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${TUFF_BIN:-./build/tuff}"
TARGET="${1:-all}"
case "$TARGET" in dense|moe8|all) ;; *) echo "Usage: $0 [dense|moe8|all]" >&2; exit 2 ;; esac
[ -x "$BIN" ] || { echo "Error: $BIN not found; build tuff first." >&2; exit 2; }
M_DENSE="/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
M_MOE8="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
PROMPT="Explain in a few sentences why the sky appears blue during the day but turns red at sunset, and mention the role of scattering. Then give two short examples."

SCRUB=(); SCRUBBED=()
while IFS= read -r v; do
    case "$v" in LFM25_*) SCRUB+=(-u "$v"); SCRUBBED+=("$v") ;; esac
done < <(compgen -e)
[ "${#SCRUBBED[@]}" -eq 0 ] || echo "WARNING: scrubbed caller environment for tuff runs (unset): ${SCRUBBED[*]}" >&2
T() { env ${SCRUB[@]+"${SCRUB[@]}"} "$@"; }
tmp="$(mktemp -d /tmp/tuff_prefill.XXXXXX)"; trap 'rm -rf "$tmp"' EXIT
fail=0; err=0; ncase=0

run_model() {  # <label> <model>
    local label="$1" model="$2" b f n why=""
    [ -f "$model" ] || { echo "Error: model $model not found" >&2; err=1; return; }
    for b in 1 8; do
        f="$tmp/$label.b$b"
        TUFF_TIMING=1 T "$BIN" --model "$model" -c 1024 -n 16 --temp 0 --prompt "$PROMPT" --prefill-batch $b --dump-tokens > "$f.tok" 2> "$f.tokerr"; echo $? > "$f.tokrc"
        TUFF_TIMING=1 T "$BIN" --model "$model" -c 1024 -n 16 --temp 0 --prompt "$PROMPT" --prefill-batch $b --dump-logits "$f.lg" > "$f.lgo" 2> "$f.lgerr"; echo $? > "$f.lgrc"
        for x in tok lg; do
            [ "$(cat "$f.${x}rc")" = 0 ] || { echo "$label: --prefill-batch $b ($x) run failed"; tail -3 "$f.${x}err"; err=1; return; }
        done
    done
    n="$(sed -n 's/.*Prompt tokens (\([0-9]*\)).*/\1/p' "$tmp/$label.b8.tokerr" | head -1)"
    if [ -z "$n" ] || [ "$n" -lt 33 ] || [ $((n % 8)) -eq 0 ]; then echo "$label: prompt has $n tokens; need >= 33 and not a multiple of 8"; err=1; return; fi
    [ "$(wc -l < "$tmp/$label.b8.tok")" -eq 16 ] && [ -s "$tmp/$label.b8.lg" ] || { echo "$label: missing output"; err=1; return; }
    grep -q "prefill: $n tokens, batch 8," "$tmp/$label.b8.tokerr" && grep -q "prefill: $n tokens, batch 8," "$tmp/$label.b8.lgerr" || why="batch-8 run did not prefill in batches of 8"
    grep -q "prefill: $n tokens, batch 1," "$tmp/$label.b1.tokerr" || why="batch-1 run did not prefill token by token"
    cmp -s "$tmp/$label.b1.tok" "$tmp/$label.b8.tok" || why="$why tokens differ"
    cmp -s "$tmp/$label.b1.lg" "$tmp/$label.b8.lg" || why="$why logits differ"
    ncase=$((ncase + 1))
    if [ -z "$why" ]; then echo "$label: $n-token prompt, --prefill-batch 1 vs 8: tokens and logits identical (PASS)"
    else echo "$label: $n-token prompt:$why (FAIL)"; fail=1; fi
}
[ "$TARGET" = all ] || [ "$TARGET" = dense ] && run_model dense "$M_DENSE"
[ "$TARGET" = all ] || [ "$TARGET" = moe8 ] && run_model moe8 "$M_MOE8"
if [ "$err" -ne 0 ]; then echo "PREFILL ERROR (setup)"; exit 2; fi
if [ "$fail" -eq 0 ] && [ "$ncase" -gt 0 ]; then echo "PREFILL PASS ($ncase models)"; exit 0; fi
echo "PREFILL FAIL"; exit 1
