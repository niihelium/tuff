#!/usr/bin/env bash
# G11 (agent): run the real headless agent against tuff-server and check the file it wrote.
# usage: g11_agent.sh MODEL.gguf MODEL_NAME WORKDIR [TASK_PROMPT]
set -euo pipefail
MODEL=$(realpath "$1"); NAME=$2; WORK=$3
TASK=${4:-"Create a file named hello.txt in the workspace whose content is exactly: hello from tuff"}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK/out"
rm -f "$WORK/hello.txt"
"$ROOT/build-srv/tuff-server" --model "$MODEL" --port 8097 -np 4 -c 65536 \
    --reasoning-format deepseek > "$WORK/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8097/health >/dev/null && break; sleep 1; done
cd "$ROOT/../agent"
set +e
PROFILE=env LLM_BASE_URL=http://127.0.0.1:8097/v1 LLM_MODEL="$NAME" LLM_CONTEXT_WINDOW=16384 \
WORKDIR="$WORK" OUTPUT_DIR="$WORK/out" TASK_PROMPT="$TASK" TOOLSET=workspace \
CONTEXT_MODE=normal MAX_TOOL_ROUNDS=10 GUARD="${GUARD:-fix}" \
LLM_EXTRA_BODY='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
bun run src/run.ts > "$WORK/agent.log" 2>&1
RC=$?
set -e
echo "agent exit=$RC"
echo "--- file ---"; cat "$WORK/hello.txt" 2>/dev/null || echo "(missing)"
echo "--- report ---"; cat "$WORK/out/report.json" 2>/dev/null || true
echo
python3 - "$WORK" <<'PY'
import json, sys, pathlib
w = pathlib.Path(sys.argv[1])
tr = json.loads((w / "out/trace.json").read_text()) if (w / "out/trace.json").exists() else {}
s = json.dumps(tr)
ok_file = (w / "hello.txt").exists() and (w / "hello.txt").read_text().strip() == "hello from tuff"
ok_tool = '"tool_calls"' in s or '"tool_call"' in s or "write_file" in s
print(f"G11 agent: file={'PASS' if ok_file else 'FAIL'} streaming_tool_calls={'PASS' if ok_tool else 'FAIL'}")
sys.exit(0 if ok_file and ok_tool else 1)
PY
