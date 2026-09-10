#!/usr/bin/env bash
# Start Tuff OpenAI-compatible HTTP API server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODEL="/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-${MAX_SEQ:-}}"
FILL_RAM="${FILL_RAM:-0}"

# Auto-mount pohjola NVMe partition if model is not accessible
if [ ! -f "$MODEL_PATH" ] && [ -b "/dev/nvme0n1p1" ]; then
    echo "[run.sh] Mounting /dev/nvme0n1p1..."
    udisksctl mount -b /dev/nvme0n1p1 >/dev/null 2>&1 || true
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[run.sh] Error: Model file not found at: $MODEL_PATH" >&2
    echo "[run.sh] Please provide MODEL_PATH or ensure the drive is mounted." >&2
    exit 1
fi

# Build if binary is missing
if [ ! -f "$SCRIPT_DIR/build/tuff-server" ]; then
    echo "[run.sh] Building tuff-server..."
    cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build "$SCRIPT_DIR/build" --target tuff-server -j8
fi

EXTRA_ARGS=()
if [ -n "$CTX_SIZE" ]; then
    EXTRA_ARGS+=("-c" "$CTX_SIZE")
fi
if [ "$FILL_RAM" = "1" ] || [ "$FILL_RAM" = "true" ] || [ "$FILL_RAM" = "yes" ]; then
    EXTRA_ARGS+=("--fill-ram")
fi

echo "[run.sh] Starting tuff-server on http://${HOST}:${PORT}"
echo "[run.sh] Model: $MODEL_PATH"

exec "$SCRIPT_DIR/build/tuff-server" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@"
