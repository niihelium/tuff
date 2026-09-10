#!/usr/bin/env bash
# M1 oracle: compares tuff's greedy-decoded TOKEN ID sequence against llama.cpp's,
# for the same prompt, same context, same (temp=0, i.e. argmax) sampling.
#
# Earlier versions of this script diffed llama-cli's stdout against tuff's ENTIRE
# stdout, which included the loading banner, the prompt-token dump, and the speed report —
# so it could never pass even with a correct engine. This version compares only the
# generated token IDs:
#   - tuff:  `--dump-tokens` mode, one generated token ID per line on stdout, all
#                  diagnostics on stderr (see src/main.c).
#   - llama.cpp:   llama-server's /completion endpoint with `"return_tokens": true`, which
#                  returns the exact sampled token IDs (not just detokenized text) in the
#                  JSON response's "tokens" array (tools/server/server-task.cpp).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/nii/workspace/llama.cpp/build/bin/llama-server}"
TUFF_BIN="${TUFF_BIN:-./build/tuff}"
PROMPT="${PROMPT:-Hello}"
N_TOKENS="${N_TOKENS:-32}"
PORT="${PORT:-8091}"

echo "=================================================================="
echo "          TUFF ORACLE VERIFICATION VS LLAMA.CPP            "
echo "=================================================================="

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "Error: llama-server not found or executable at $LLAMA_SERVER"
    exit 1
fi
if [ ! -x "$TUFF_BIN" ]; then
    echo "Error: tuff binary not found at $TUFF_BIN. Build it first."
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required to parse the llama-server JSON response"
    exit 1
fi

SERVER_LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$SERVER_LOG"
}
trap cleanup EXIT

echo "[1/3] Starting llama-server (CPU-only, reference)..."
CUDA_VISIBLE_DEVICES="" "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -ngl 0 \
    -c 4096 \
    -t 8 \
    --port "$PORT" \
    --no-webui \
    --seed 1 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "      waiting for server to become healthy..."
READY=0
for _ in $(seq 1 120); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q "200"; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: llama-server exited early. Log:"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done
if [ "$READY" -ne 1 ]; then
    echo "Error: llama-server did not become healthy in time. Log:"
    cat "$SERVER_LOG"
    exit 1
fi

echo "[2/3] Requesting reference token IDs from llama-server (temperature=0, greedy)..."
RESPONSE=$(curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True}))' "$PROMPT" "$N_TOKENS")")

LLAMA_IDS=$(python3 -c '
import json, sys
resp = json.loads(sys.argv[1])
if "tokens" not in resp:
    sys.stderr.write("Unexpected server response (no \"tokens\" field): %s\n" % sys.argv[1][:500])
    sys.exit(1)
for t in resp["tokens"]:
    print(t)
' "$RESPONSE")

echo "[3/3] Running tuff --dump-tokens..."
TUFF_IDS=$("$TUFF_BIN" --model "$MODEL_PATH" --prompt "$PROMPT" -n "$N_TOKENS" --dump-tokens 2>/dev/null)

echo "------------------------------------------------------------------"
echo "LLAMA.CPP TOKEN IDS ($(echo "$LLAMA_IDS" | wc -l) tokens):"
echo "$LLAMA_IDS" | tr '\n' ' '
echo
echo "------------------------------------------------------------------"
echo "TUFF TOKEN IDS ($(echo "$TUFF_IDS" | wc -l) tokens):"
echo "$TUFF_IDS" | tr '\n' ' '
echo
echo "------------------------------------------------------------------"

if [ "$LLAMA_IDS" = "$TUFF_IDS" ]; then
    echo "ORACLE PASSED: token IDs are 100% identical."
    exit 0
else
    echo "ORACLE FAILED: token ID sequences differ."
    exit 1
fi
