#!/usr/bin/env bash
# Live keeper for the `llm` job type: a real server, a real engine, a real model.
#
# Unlike scripts/keeper-live.sh this one needs the host to be ready: the llm env
# installed and the model pulled. It does not install either — that is minutes of
# downloading and it is not a keeper's job — but it refuses BY NAME if they are
# missing rather than skipping.
#
#   ./scripts/keeper-llm-live.sh                 # qwen3.5-9b
#   CRUCIBLE_LLM_MODEL=qwen3.5-27b ./scripts/keeper-llm-live.sh
#
# What it proves, end to end:
#   1. the card is idle before anything starts (and the run is refused if not)
#   2. load-model streams `warming` and ends `done {resident}`
#   3. a non-streamed chat comes back through the proxy
#   4. a streamed chat keeps its SSE framing and its [DONE]
#   5. a wrong model name is 409 model_not_resident naming the resident one
#   6. unload-model frees the engine
#   7. the accelerator is back where it started
#
# Exits 0 only if every check passed. Trust the exit code.

set -euo pipefail

MODEL="${CRUCIBLE_LLM_MODEL:-qwen3.5-9b}"
LOAD_TIMEOUT="${CRUCIBLE_LOAD_TIMEOUT:-1200}"

PASSED=0
FAILED=0
SERVER_PID=""
ROOT=""

log()  { printf '  %s\n' "$*"; }
ok()   { PASSED=$((PASSED + 1)); printf 'ok    %s\n' "$*"; }
bad()  { FAILED=$((FAILED + 1)); printf 'FAIL  %s\n' "$*" >&2; }

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # SIGTERM only. The server's own shutdown stops any resident engine the same
    # way — a SIGKILL here would leave a CUDA process wedged in WSL2.
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then
    rm -rf "$ROOT"
  fi
}
trap cleanup EXIT

command -v crucible >/dev/null || { echo "keeper: no crucible on PATH" >&2; exit 2; }
command -v curl >/dev/null     || { echo "keeper: no curl on PATH" >&2; exit 2; }
command -v python3 >/dev/null  || { echo "keeper: no python3 on PATH" >&2; exit 2; }

# --------------------------------------------------------------- host readiness
#
# The keeper runs against the operator's real ~/.crucible (the env and the model
# weights live there and are gigabytes); only the server's config, jobs and
# uploads go somewhere throwaway. So: read the real home first, then point
# CRUCIBLE_HOME at a copy of it that shares the envs and models by symlink.

REAL_HOME="${CRUCIBLE_HOME:-$HOME/.crucible}"
[ -d "$REAL_HOME/envs/llm" ] || {
  echo "keeper: no llm env at $REAL_HOME/envs/llm — run \`crucible install llm\`" >&2
  exit 2
}

BACKEND="$(python3 - <<'PY'
from crucible.backend import detect_backend
print(detect_backend().kind)
PY
)"
[ -d "$REAL_HOME/models/$MODEL/$BACKEND" ] || {
  echo "keeper: $MODEL is not pulled for $BACKEND (no $REAL_HOME/models/$MODEL/$BACKEND)" >&2
  echo "keeper: run \`crucible models pull $MODEL\`" >&2
  exit 2
}

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/crucible-llm-keeper.XXXXXX")"
export CRUCIBLE_HOME="$ROOT/home"
WORK="$ROOT/work"
mkdir -p "$CRUCIBLE_HOME" "$WORK"
ln -s "$REAL_HOME/envs" "$CRUCIBLE_HOME/envs"
ln -s "$REAL_HOME/models" "$CRUCIBLE_HOME/models"

PORT="$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"

echo "keeper: model=$MODEL backend=$BACKEND home=$CRUCIBLE_HOME port=$PORT"

# ------------------------------------------------------------- 1. the card first

accelerator_used() {
  python3 - "$BACKEND" <<'PY'
import sys
from crucible.accelerator import read_state
state = read_state(sys.argv[1], 0)
print(f"{state.used_bytes} {state.free_bytes} {state.total_bytes}")
PY
}

read -r USED_BEFORE FREE_BEFORE TOTAL <<<"$(accelerator_used)"
log "accelerator before: $((USED_BEFORE / 1024 / 1024)) MiB used, $((FREE_BEFORE / 1024 / 1024)) MiB free of $((TOTAL / 1024 / 1024)) MiB"

# ------------------------------------------------------------------ init + serve

crucible init --enable-echo --enable-llm --host 127.0.0.1 --port "$PORT" \
  --name "crucible@llm-keeper" >"$WORK/init.txt"
TOKEN="$(crucible token --show)"
[ -n "$TOKEN" ] || { echo "keeper: token is empty" >&2; exit 2; }

crucible doctor --json >"$WORK/doctor.json"
if python3 - "$WORK/doctor.json" "$MODEL" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
env = report["llm_env"]
assert env is not None and env["installed"], f"llm env not installed: {env}"
if report["problems"]:
    raise SystemExit("doctor problems: " + "; ".join(report["problems"]))
print(env["detail"])
PY
then
  ok "doctor reports the llm env ready"
else
  bad "doctor is unhealthy: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["problems"])' "$WORK/doctor.json")"
fi

BASE="http://127.0.0.1:$PORT/v1"
crucible serve --host 127.0.0.1 --port "$PORT" --log-level warning >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!

UP=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "$BASE/ping" -o "$WORK/ping.json" 2>/dev/null; then UP=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "keeper: the server exited before answering:" >&2; cat "$WORK/serve.log" >&2; exit 2
  fi
  sleep 0.25
done
[ "$UP" = "1" ] || { echo "keeper: server never answered $BASE/ping" >&2; cat "$WORK/serve.log" >&2; exit 2; }

AUTH=(-H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1")
JSON=(-H 'Content-Type: application/json')

# --------------------------------------------------------------- 2. GET /models

CODE="$(curl -sS -o "$WORK/models.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/models")"
if [ "$CODE" = "200" ] && python3 - "$WORK/models.json" "$MODEL" <<'PY'
import json, sys
rows = {row["id"]: row for row in json.load(open(sys.argv[1]))}
row = rows[sys.argv[2]]
assert row["backend_supported"], row
assert row["installed"], row
assert row["loadable"], row
assert row["resident"] is False, row
assert row["memory_bytes_estimate"] > 0, row
PY
then
  ok "GET /models says $MODEL is installed and loadable"
else
  bad "GET /models returned $CODE: $(cat "$WORK/models.json")"
fi

# ------------------------------------- 3. the proxy refuses before anything loads

CODE="$(curl -sS -o "$WORK/409a.json" -w '%{http_code}' "${AUTH[@]}" "${JSON[@]}" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
  "$BASE/openai/chat/completions")"
if [ "$CODE" = "409" ] && grep -q 'model_not_resident' "$WORK/409a.json" && grep -q 'no model is' "$WORK/409a.json"; then
  ok "chat before any load is 409 model_not_resident (never loads implicitly)"
else
  bad "chat before load returned $CODE: $(cat "$WORK/409a.json")"
fi

# ------------------------------------------------------------------- 4. the load

CODE="$(curl -sS -o "$WORK/load.json" -w '%{http_code}' "${AUTH[@]}" "${JSON[@]}" \
  -d "{\"type\":\"load-model\",\"model\":\"$MODEL\",\"params\":{\"timeout_s\":$LOAD_TIMEOUT}}" \
  "$BASE/jobs")"
if [ "$CODE" != "202" ]; then
  bad "POST load-model returned $CODE: $(cat "$WORK/load.json")"
  echo "keeper: $PASSED passed, $FAILED failed"; exit 1
fi
JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$WORK/load.json")"
log "load job $JOB_ID — streaming its events (this takes minutes)"

LOAD_START="$(date +%s)"
curl -sS -N --max-time "$((LOAD_TIMEOUT + 120))" "${AUTH[@]}" "$BASE/jobs/$JOB_ID/events" >"$WORK/load.sse"
LOAD_SECONDS=$(( $(date +%s) - LOAD_START ))

if python3 - "$WORK/load.sse" "$MODEL" <<'PY'
import json, sys
kinds, data = [], []
for line in open(sys.argv[1]):
    if line.startswith("event:"):
        kinds.append(line.split(":", 1)[1].strip())
    elif line.startswith("data:"):
        data.append(json.loads(line.split(":", 1)[1].strip()))
assert kinds[0] == "queued", kinds
if kinds[-1] == "failed":
    raise SystemExit("the load failed: " + json.dumps(data[-1]))
assert kinds[-1] == "done", kinds
assert kinds.count("warming") >= 2, f"only {kinds.count('warming')} warming events"
assert data[-1].get("resident") == sys.argv[2], data[-1]
for message in [d["message"] for k, d in zip(kinds, data) if k == "warming"][:6]:
    print("    warming:", message)
PY
then
  ok "load-model streamed warming and ended done {resident: $MODEL} in ${LOAD_SECONDS}s"
else
  bad "load-model event stream: $(grep '^event:' "$WORK/load.sse" | tr '\n' ' ')"
  echo "keeper: $PASSED passed, $FAILED failed"; exit 1
fi

read -r USED_LOADED FREE_LOADED _ <<<"$(accelerator_used)"
log "accelerator loaded: $((USED_LOADED / 1024 / 1024)) MiB used (+$(( (USED_LOADED - USED_BEFORE) / 1024 / 1024 )) MiB)"

CODE="$(curl -sS -o "$WORK/health.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/health")"
if [ "$CODE" = "200" ] && python3 - "$WORK/health.json" "$MODEL" <<'PY'
import json, sys
health = json.load(open(sys.argv[1]))
assert health["resident_models"] == [sys.argv[2]], health
PY
then
  ok "GET /health reports resident_models [$MODEL]"
else
  bad "GET /health returned $CODE: $(cat "$WORK/health.json")"
fi

# --------------------------------------------------- 5. a non-streamed chat

cat >"$WORK/chat.json" <<JSON
{"model": "$MODEL",
 "messages": [{"role": "user", "content": "Reply with exactly: Crucible is running."}],
 "temperature": 0, "max_tokens": 32}
JSON
CODE="$(curl -sS -o "$WORK/chat.out.json" -w '%{http_code}' --max-time 600 \
  "${AUTH[@]}" "${JSON[@]}" --data-binary "@$WORK/chat.json" "$BASE/openai/chat/completions")"
if [ "$CODE" = "200" ] && python3 - "$WORK/chat.out.json" "$MODEL" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["choices"][0]["message"]["content"].strip(), body
assert body["usage"]["completion_tokens"] > 0, body
print("    content:", repr(body["choices"][0]["message"]["content"]))
print("    usage:", body["usage"])
PY
then
  ok "non-streamed chat completion through the proxy"
else
  bad "non-streamed chat returned $CODE: $(cat "$WORK/chat.out.json")"
fi

# ------------------------------------------------------- 6. a streamed chat

cat >"$WORK/chatstream.json" <<JSON
{"model": "$MODEL",
 "messages": [{"role": "user", "content": "Count from one to five."}],
 "temperature": 0, "max_tokens": 48, "stream": true}
JSON
curl -sS -N --max-time 600 "${AUTH[@]}" "${JSON[@]}" \
  --data-binary "@$WORK/chatstream.json" "$BASE/openai/chat/completions" >"$WORK/chat.sse"
if python3 - "$WORK/chat.sse" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read()
frames = [line for line in raw.split("\n") if line.startswith("data: ")]
assert frames, "no SSE frames came back"
assert frames[-1].strip() == "data: [DONE]", frames[-1]
pieces = []
for frame in frames[:-1]:
    chunk = json.loads(frame[len("data: "):])
    pieces.append(chunk["choices"][0]["delta"].get("content", ""))
assert len(pieces) > 1, f"only {len(pieces)} deltas — that is not a stream"
assert "".join(pieces).strip(), "the stream carried no text"
print("    deltas:", len(pieces))
print("    content:", repr("".join(pieces)))
PY
then
  ok "streamed chat completion, SSE framing intact, terminated by [DONE]"
else
  bad "streamed chat: $(head -c 400 "$WORK/chat.sse")"
fi

# ------------------------------------ 7. a wrong model name is 409, naming this one

CODE="$(curl -sS -o "$WORK/409b.json" -w '%{http_code}' "${AUTH[@]}" "${JSON[@]}" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}' \
  "$BASE/openai/chat/completions")"
if [ "$CODE" = "409" ] && python3 - "$WORK/409b.json" "$MODEL" <<'PY'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["code"] == "model_not_resident", error
assert error["details"] == {"requested": "gpt-4", "resident": sys.argv[2]}, error
print("    body:", json.dumps(error))
PY
then
  ok "a wrong model name is 409 model_not_resident naming $MODEL"
else
  bad "wrong model name returned $CODE: $(cat "$WORK/409b.json")"
fi

# ----------------------------------------------------------------- 8. the unload

CODE="$(curl -sS -o "$WORK/unload.json" -w '%{http_code}' "${AUTH[@]}" "${JSON[@]}" \
  -d "{\"type\":\"unload-model\",\"model\":\"$MODEL\"}" "$BASE/jobs")"
if [ "$CODE" != "202" ]; then
  bad "POST unload-model returned $CODE: $(cat "$WORK/unload.json")"
else
  UNLOAD_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$WORK/unload.json")"
  curl -sS -N --max-time 300 "${AUTH[@]}" "$BASE/jobs/$UNLOAD_ID/events" >"$WORK/unload.sse"
  if python3 - "$WORK/unload.sse" <<'PY'
import json, sys
kinds, data = [], []
for line in open(sys.argv[1]):
    if line.startswith("event:"):
        kinds.append(line.split(":", 1)[1].strip())
    elif line.startswith("data:"):
        data.append(json.loads(line.split(":", 1)[1].strip()))
if kinds[-1] == "failed":
    raise SystemExit("the unload failed: " + json.dumps(data[-1]))
assert kinds[-1] == "done", kinds
assert data[-1].get("resident") is None, data[-1]
PY
  then
    ok "unload-model ended done with nothing resident"
  else
    bad "unload-model event stream: $(grep '^event:' "$WORK/unload.sse" | tr '\n' ' ')"
  fi
fi

CODE="$(curl -sS -o "$WORK/health2.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/health")"
if [ "$CODE" = "200" ] && grep -q '"resident_models":\[\]' "$WORK/health2.json"; then
  ok "GET /health reports nothing resident"
else
  bad "GET /health after unload: $(cat "$WORK/health2.json")"
fi

# ----------------------------------------------- 9. the accelerator came back

SETTLED=0
for _ in $(seq 1 60); do
  read -r USED_AFTER FREE_AFTER _ <<<"$(accelerator_used)"
  # Back to within a GiB of where it started.
  if [ "$USED_AFTER" -le $((USED_BEFORE + 1073741824)) ]; then SETTLED=1; break; fi
  sleep 2
done
log "accelerator after: $((USED_AFTER / 1024 / 1024)) MiB used (started at $((USED_BEFORE / 1024 / 1024)) MiB)"
if [ "$SETTLED" = "1" ]; then
  ok "the accelerator is back to its idle figure"
else
  bad "the accelerator did not come back: $((USED_AFTER / 1024 / 1024)) MiB used vs $((USED_BEFORE / 1024 / 1024)) MiB before"
fi

echo "keeper: $PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
