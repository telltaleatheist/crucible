#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen3.5-9b}"
SERVER_PID=""
ROOT=""

cleanup() {
  status=$?
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then
    if [ "$status" -ne 0 ] && [ -d "$ROOT/home/logs" ]; then
      KEPT="${TMPDIR:-/tmp}/crucible-measure-failed.$$"
      mkdir -p "$KEPT"
      cp -r "$ROOT/home/logs" "$KEPT/logs"
      echo "measure: kept the engine log at $KEPT/logs (exit $status)" >&2
    fi
    rm -rf "$ROOT"
  fi
}
trap cleanup EXIT

REAL_HOME="${CRUCIBLE_HOME:-$HOME/.crucible}"
[ -d "$REAL_HOME/envs/llm" ] || { echo "no llm env at $REAL_HOME/envs/llm" >&2; exit 2; }

BACKEND="$(python3 -c 'from crucible.backend import detect_backend; print(detect_backend().kind)')"
[ -d "$REAL_HOME/models/$MODEL/$BACKEND" ] || {
  echo "$MODEL is not pulled for $BACKEND" >&2; exit 2; }

CONTEXT="$(python3 - "$MODEL" <<'PY'
import sys
from crucible.manifests import load_manifest
print(load_manifest(sys.argv[1]).context_default)
PY
)"

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/crucible-measure.XXXXXX")"
export CRUCIBLE_HOME="$ROOT/home"
WORK="$ROOT/work"; mkdir -p "$CRUCIBLE_HOME" "$WORK"
ln -s "$REAL_HOME/envs" "$CRUCIBLE_HOME/envs"
ln -s "$REAL_HOME/models" "$CRUCIBLE_HOME/models"
PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"

card() {
  python3 - "$BACKEND" <<'PY'
import sys
from crucible.accelerator import read_state
state = read_state(sys.argv[1], 0)
print(state.used_bytes)
PY
}

engine_rss() {
  local total=0 rss
  for pid in $(pgrep -f "models/$MODEL/$BACKEND" || true); do
    rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')"
    [ -n "$rss" ] && total=$((total + rss * 1024))
  done
  echo "$total"
}

echo "measuring $MODEL on $BACKEND, context $CONTEXT"
BEFORE="$(card)"
echo "  card before:  $((BEFORE / 1024 / 1024)) MiB in use"

crucible init --enable-llm --host 127.0.0.1 --port "$PORT" --name "crucible@measure" >/dev/null
TOKEN="$(crucible token --show)"
BASE="http://127.0.0.1:$PORT/v1"
crucible serve --host 127.0.0.1 --port "$PORT" --log-level warning >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
  curl -fsS --max-time 2 "$BASE/ping" -o /dev/null 2>/dev/null && break
  sleep 0.25
done

AUTH=(-H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1" -H 'Content-Type: application/json')

JOB="$(curl -sS "${AUTH[@]}" -d "{\"type\":\"load-model\",\"model\":\"$MODEL\",\"params\":{\"timeout_s\":1800}}" \
  "$BASE/jobs" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
curl -sS -N --max-time 2000 "${AUTH[@]}" "$BASE/jobs/$JOB/events" >"$WORK/load.sse"
grep -q 'event: done' "$WORK/load.sse" || { echo "the load failed:"; tail -5 "$WORK/load.sse"; exit 1; }

sleep 5
REST="$(card)"; REST_RSS="$(engine_rss)"
echo "  at rest:      card $((REST / 1024 / 1024)) MiB (+$(( (REST - BEFORE) / 1024 / 1024 )) MiB), engine rss $((REST_RSS / 1024 / 1024)) MiB"

"$REAL_HOME/envs/llm/bin/python" - \
  "$MODEL" "$CONTEXT" "$REAL_HOME/models/$MODEL/$BACKEND" "$WORK/big.json" <<'PY'
import json, sys
from transformers import AutoTokenizer

model, context, weights, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tokenizer = AutoTokenizer.from_pretrained(weights)
lead = "Here is a list. Reply with only the word OK.\n"
template_and_answer_tokens = 96
budget = context - template_and_answer_tokens
filler = " ".join(f"item{index}" for index in range(budget))
content = tokenizer.decode(tokenizer(lead + filler)["input_ids"][:budget])
total = len(tokenizer(content)["input_ids"])
print(f"  prompt sized to {total} tokens of a {context}-token context", flush=True)
json.dump(
    {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0,
    },
    open(out, "w"),
)
PY
echo "  running a completion that fills $CONTEXT tokens of context..."
curl -sS --max-time 1800 "${AUTH[@]}" --data-binary "@$WORK/big.json" \
  "$BASE/openai/chat/completions" >"$WORK/big.out.json" || true
python3 - "$WORK/big.out.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
if "usage" in body:
    print("  prompt tokens:", body["usage"]["prompt_tokens"])
else:
    print("  the engine refused the long prompt:", json.dumps(body)[:300])
PY

LOADED="$(card)"; LOADED_RSS="$(engine_rss)"
echo "  under context: card $((LOADED / 1024 / 1024)) MiB (+$(( (LOADED - BEFORE) / 1024 / 1024 )) MiB), engine rss $((LOADED_RSS / 1024 / 1024)) MiB"

MEASURED=$((LOADED - BEFORE))
echo
echo "memory_bytes_estimate = $MEASURED   # $((MEASURED / 1024 / 1024)) MiB, $(python3 -c "print(f'{$MEASURED/1e9:.2f}')") GB"
echo "  (engine rss was $((LOADED_RSS / 1024 / 1024)) MiB — a floor, not the figure)"

curl -sS "${AUTH[@]}" -d "{\"type\":\"unload-model\",\"model\":\"$MODEL\"}" "$BASE/jobs" \
  | python3 -c 'import json,sys; print("  unload job", json.load(sys.stdin)["job_id"])'
sleep 8
AFTER="$(card)"
echo "  card after:   $((AFTER / 1024 / 1024)) MiB in use (started at $((BEFORE / 1024 / 1024)) MiB)"
