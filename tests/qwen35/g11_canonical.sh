#!/usr/bin/env bash
# G11 (canonical gsm8k): start the requested engine and run 50 items, greedy, thinking off.
# usage: g11_canonical.sh tuf|llama MODEL.gguf MODEL_NAME OUTDIR [EXTRA_ARGS...]
set -euo pipefail
ENGINE=$1; MODEL=$(realpath "$2"); NAME=$3; OUT=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
if [ "$ENGINE" = tuf ]; then
  "$ROOT/build-srv/tuff-server" --model "$MODEL" --port 8097 -np 4 -c 65536 > "$OUT/server.log" 2>&1 &
else
  /home/nii/.local/bin/llama-server -m "$MODEL" --port 8096 --host 127.0.0.1 -ngl 99 -c 65536 -np 4 --kv-unified --jinja > "$OUT/server.log" 2>&1 &
fi
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT
PORT=8097; [ "$ENGINE" = llama ] && PORT=8096
for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
cd "$ROOT/../sandbox"
UV_CACHE_DIR=/tmp/uvc-dsh uv run --extra evals python evals/canonical.py gsm8k \
  --endpoint "http://127.0.0.1:$PORT/v1" --name "$NAME" --limit 50 --concurrency 4 \
  --out-dir "$OUT" "$@"
