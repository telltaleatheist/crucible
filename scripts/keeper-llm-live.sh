#!/usr/bin/env bash
# Live keeper for the `llm` job type: a real server, a real engine, a real model.
#
# LOCAL MODE (default). Starts a throwaway server on a free port against the
# operator's real ~/.crucible, because the env and the weights live there and are
# gigabytes. It does not install either — that is minutes of downloading and not a
# keeper's job — but it refuses BY NAME if they are missing rather than skipping.
#
#   ./scripts/keeper-llm-live.sh
#   CRUCIBLE_LLM_MODEL=qwen3.8-27b ./scripts/keeper-llm-live.sh
#
# REMOTE MODE. Set both CRUCIBLE_URL and CRUCIBLE_TOKEN and the keeper drives a
# server already running somewhere else, starting nothing of its own:
#
#   export CRUCIBLE_URL=http://owens-mac-studio.hs.owenmorgan.com:7100
#   export CRUCIBLE_TOKEN=...
#   ./scripts/keeper-llm-live.sh
#
# That is the shape that matters for the apps: a Windows client driving a GPU on
# another machine over the tailnet. In remote mode the keeper needs nothing but
# curl and a python3 — no crucible on PATH, no env, no weights, no sight of the
# card. The accelerator checks become what a client can honestly see: /v1/health
# and /v1/models. Setting one of the two variables and not the other is a
# refusal, never a silent fall back to local mode.
#
# What it proves, end to end:
#   1. the server is reachable and says what it is
#   2. the model is installed and loadable there, pinned to a revision, and
#      /info's llm capability carries exactly the same rows as /models
#   3. a chat before any load is 409 model_not_resident — never an implicit load
#   4. load-model streams `warming` and ends `done {resident}`
#   5. a non-streamed chat comes back through the proxy
#   6. a streamed chat keeps its SSE framing and its [DONE]
#   7. a wrong model name is 409 model_not_resident naming the resident one
#   8. unload-model frees the engine and nothing is resident afterwards
#   9. (local mode only) the accelerator is back where it started
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

command -v curl >/dev/null || { echo "keeper: no curl on PATH" >&2; exit 2; }

# python3 is spelled `python` on a Windows Git Bash, where `python3` is a Store
# stub that refuses to run. Find a real one and name it; never guess silently.
PY="${CRUCIBLE_PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)' >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  done
fi
[ -n "$PY" ] || { echo "keeper: no python 3.9+ on PATH (set CRUCIBLE_PYTHON)" >&2; exit 2; }

URL="${CRUCIBLE_URL:-}"
TOKEN="${CRUCIBLE_TOKEN:-}"
if [ -n "$URL" ] && [ -z "$TOKEN" ]; then
  echo "keeper: CRUCIBLE_URL is set but CRUCIBLE_TOKEN is not" >&2; exit 2
fi
if [ -z "$URL" ] && [ -n "$TOKEN" ]; then
  echo "keeper: CRUCIBLE_TOKEN is set but CRUCIBLE_URL is not" >&2; exit 2
fi
REMOTE=0
[ -n "$URL" ] && REMOTE=1

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/crucible-llm-keeper.XXXXXX")"
WORK="$ROOT/work"
mkdir -p "$WORK"

# ------------------------------------------------------------------ the server

if [ "$REMOTE" = "1" ]; then
  BASE="${URL%/}/v1"
  echo "keeper: model=$MODEL driving REMOTE $BASE (python $PY)"
else
  command -v crucible >/dev/null || { echo "keeper: no crucible on PATH" >&2; exit 2; }

  REAL_HOME="${CRUCIBLE_HOME:-$HOME/.crucible}"
  [ -d "$REAL_HOME/envs/llm" ] || {
    echo "keeper: no llm env at $REAL_HOME/envs/llm — run \`crucible install llm\`" >&2
    exit 2
  }
  BACKEND="$("$PY" -c 'from crucible.backend import detect_backend; print(detect_backend().kind)')"
  [ -d "$REAL_HOME/models/$MODEL/$BACKEND" ] || {
    echo "keeper: $MODEL is not pulled for $BACKEND (no $REAL_HOME/models/$MODEL/$BACKEND)" >&2
    echo "keeper: run \`crucible models pull $MODEL\`" >&2
    exit 2
  }

  export CRUCIBLE_HOME="$ROOT/home"
  mkdir -p "$CRUCIBLE_HOME"
  ln -s "$REAL_HOME/envs" "$CRUCIBLE_HOME/envs"
  ln -s "$REAL_HOME/models" "$CRUCIBLE_HOME/models"

  PORT="$("$PY" -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"
  echo "keeper: model=$MODEL backend=$BACKEND home=$CRUCIBLE_HOME port=$PORT"

  accelerator_used() {
    "$PY" - "$BACKEND" <<'PYCODE'
import sys
from crucible.accelerator import read_state
state = read_state(sys.argv[1], 0)
print(f"{state.used_bytes} {state.free_bytes} {state.total_bytes}")
PYCODE
  }
  read -r USED_BEFORE FREE_BEFORE TOTAL <<<"$(accelerator_used)"
  log "accelerator before: $((USED_BEFORE / 1024 / 1024)) MiB used, $((FREE_BEFORE / 1024 / 1024)) MiB free of $((TOTAL / 1024 / 1024)) MiB"

  crucible init --enable-echo --enable-llm --host 127.0.0.1 --port "$PORT" \
    --name "crucible@llm-keeper" >"$WORK/init.txt"
  TOKEN="$(crucible token --show)"
  [ -n "$TOKEN" ] || { echo "keeper: token is empty" >&2; exit 2; }

  crucible doctor --json >"$WORK/doctor.json"
  if "$PY" - "$WORK/doctor.json" <<'PYCODE'
import json, sys
report = json.load(open(sys.argv[1]))
env = report["llm_env"]
assert env is not None and env["installed"], f"llm env not installed: {env}"
if report["problems"]:
    raise SystemExit("doctor problems: " + "; ".join(report["problems"]))
print(env["detail"])
PYCODE
  then
    ok "doctor reports the llm env ready"
  else
    bad "doctor is unhealthy: $(cat "$WORK/doctor.json")"
  fi

  BASE="http://127.0.0.1:$PORT/v1"
  crucible serve --host 127.0.0.1 --port "$PORT" --log-level warning >"$WORK/serve.log" 2>&1 &
  SERVER_PID=$!
fi

# ------------------------------------------------------------- 1. it answers

UP=0
for _ in $(seq 1 160); do
  if curl -fsS --max-time 4 "$BASE/ping" -o "$WORK/ping.json" 2>/dev/null; then UP=1; break; fi
  if [ "$REMOTE" = "0" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "keeper: the server exited before answering:" >&2; cat "$WORK/serve.log" >&2; exit 2
  fi
  sleep 0.25
done
[ "$UP" = "1" ] || {
  echo "keeper: server never answered $BASE/ping" >&2
  [ "$REMOTE" = "0" ] && cat "$WORK/serve.log" >&2
  exit 2
}

AUTH=(-H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1")
JSON=(-H 'Content-Type: application/json')

CODE="$(curl -sS -o "$WORK/info.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/info")"
if [ "$CODE" = "200" ] && "$PY" - "$WORK/info.json" <<'PYCODE'
import json, sys
info = json.load(open(sys.argv[1]))
host = info["host"]
assert host["backend"] in ("cuda-linux", "mlx-darwin"), info
print(f"    {info['server']['name']} — {host['backend']} on "
      f"{host['platform']}/{host['arch']}, {host['gpu']['name']}, "
      f"{host['gpu']['vram_bytes'] / 1024 ** 3:.1f} GiB")
PYCODE
then
  ok "GET /info — the server says what it is"
else
  bad "GET /info returned $CODE: $(cat "$WORK/info.json")"
fi

# --------------------------------------------------------------- 2. GET /models

CODE="$(curl -sS -o "$WORK/models.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/models")"
if [ "$CODE" = "200" ] && "$PY" - "$WORK/models.json" "$MODEL" <<'PYCODE'
import json, sys
payload = json.load(open(sys.argv[1]))
assert isinstance(payload, list), f"GET /v1/models must be a bare array, got {type(payload).__name__}"
rows = {row["id"]: row for row in payload}
row = rows[sys.argv[2]]
assert row["backend_supported"], row
assert row["installed"], row
assert row["loadable"], row
assert row["resident"] is False, row
assert row["memory_bytes_estimate"] > 0, row
# The pin this host would serve, and the same row shape /info carries.
revision = row["revision"]
assert isinstance(revision, str) and len(revision) == 40, row
print("    revision:", revision)
PYCODE
then
  ok "GET /models is a bare array and says $MODEL is installed and loadable"
else
  bad "GET /models returned $CODE: $(cat "$WORK/models.json")"
fi

# The llm capability in /info is the same rows from the same producer, so a
# client that has called /info never asks twice and never reconciles two
# descriptions of one model.
if "$PY" - "$WORK/info.json" "$WORK/models.json" <<'PYCODE'
import json, sys
info = json.load(open(sys.argv[1]))
models = json.load(open(sys.argv[2]))
capabilities = {entry["job_type"]: entry for entry in info["capabilities"]}
assert "llm" in capabilities, sorted(capabilities)
assert capabilities["llm"]["models"] == models, "the llm capability rows are not /models' rows"
PYCODE
then
  ok "/info's llm capability carries exactly the /models rows"
else
  bad "/info's llm capability differs from /models"
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
JOB_ID="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$WORK/load.json")"
log "load job $JOB_ID — streaming its events (this takes minutes)"

LOAD_START="$(date +%s)"
curl -sS -N --max-time "$((LOAD_TIMEOUT + 120))" "${AUTH[@]}" "$BASE/jobs/$JOB_ID/events" >"$WORK/load.sse"
LOAD_SECONDS=$(( $(date +%s) - LOAD_START ))

if "$PY" - "$WORK/load.sse" "$MODEL" <<'PYCODE'
import json, sys
kinds, data = [], []
for line in open(sys.argv[1], encoding="utf-8"):
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
PYCODE
then
  ok "load-model streamed warming and ended done {resident: $MODEL} in ${LOAD_SECONDS}s"
else
  bad "load-model event stream: $(grep '^event:' "$WORK/load.sse" | tr '\n' ' ')"
  echo "keeper: $PASSED passed, $FAILED failed"; exit 1
fi

if [ "$REMOTE" = "0" ]; then
  read -r USED_LOADED FREE_LOADED _ <<<"$(accelerator_used)"
  log "accelerator loaded: $((USED_LOADED / 1024 / 1024)) MiB used (+$(( (USED_LOADED - USED_BEFORE) / 1024 / 1024 )) MiB)"
fi

CODE="$(curl -sS -o "$WORK/health.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/health")"
if [ "$CODE" = "200" ] && "$PY" - "$WORK/health.json" "$MODEL" <<'PYCODE'
import json, sys
health = json.load(open(sys.argv[1]))
assert health["resident_models"] == [sys.argv[2]], health
PYCODE
then
  ok "GET /health reports resident_models [$MODEL]"
else
  bad "GET /health returned $CODE: $(cat "$WORK/health.json")"
fi

# --------------------------------------------------- 5. a non-streamed chat
#
# Qwen3.5 is a reasoning model: it emits a `reasoning` field first and only then
# `content`, so a small token ceiling finishes inside the reasoning and comes back
# with no `content` at all. The fix is to say so rather than to buy the answer with
# a generous budget: `chat_template_kwargs: {"enable_thinking": false}` is read per
# request by mlx-lm and honoured by vLLM, and the SDK sends it as `thinking: false`.
# The proxy forwards it verbatim, which is the other half of what this checks.

cat >"$WORK/chat.json" <<JSON
{"model": "$MODEL",
 "messages": [{"role": "user", "content": "Reply with exactly this and nothing else: Crucible is running."}],
 "temperature": 0, "max_tokens": 64,
 "chat_template_kwargs": {"enable_thinking": false}}
JSON
CODE="$(curl -sS -o "$WORK/chat.out.json" -w '%{http_code}' --max-time 900 \
  "${AUTH[@]}" "${JSON[@]}" --data-binary "@$WORK/chat.json" "$BASE/openai/chat/completions")"
if [ "$CODE" = "200" ] && "$PY" - "$WORK/chat.out.json" <<'PYCODE'
import json, sys
body = json.load(open(sys.argv[1]))
choice = body["choices"][0]
assert choice["finish_reason"] == "stop", choice["finish_reason"]
content = choice["message"].get("content") or ""
assert content.strip(), f"no content in {list(choice['message'])}"
assert body["usage"]["completion_tokens"] > 0, body
reasoning = choice["message"].get("reasoning") or ""
print("    content:", repr(content))
if reasoning:
    print(f"    reasoning: {len(reasoning)} chars (Qwen3.5 thinks first)")
print("    usage:", body["usage"])
PYCODE
then
  ok "non-streamed chat completion through the proxy"
else
  bad "non-streamed chat returned $CODE: $(cat "$WORK/chat.out.json")"
fi

# ------------------------------------------------------- 6. a streamed chat

cat >"$WORK/chatstream.json" <<JSON
{"model": "$MODEL",
 "messages": [{"role": "user", "content": "Count from one to five, digits only, separated by spaces."}],
 "temperature": 0, "max_tokens": 64, "stream": true,
 "chat_template_kwargs": {"enable_thinking": false}}
JSON
curl -sS -N --max-time 900 "${AUTH[@]}" "${JSON[@]}" \
  --data-binary "@$WORK/chatstream.json" "$BASE/openai/chat/completions" >"$WORK/chat.sse"
if "$PY" - "$WORK/chat.sse" <<'PYCODE'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read()
frames = [line for line in raw.split("\n") if line.startswith("data: ")]
assert frames, "no SSE frames came back"
assert frames[-1].strip() == "data: [DONE]", frames[-1]
content, reasoning = [], []
for frame in frames[:-1]:
    delta = json.loads(frame[len("data: "):])["choices"][0]["delta"]
    content.append(delta.get("content") or "")
    reasoning.append(delta.get("reasoning") or "")
assert len(frames) > 2, f"only {len(frames)} frames — that is not a stream"
assert "".join(content).strip(), "the stream carried no content"
print("    frames:", len(frames), "(including [DONE])")
if "".join(reasoning).strip():
    print(f"    reasoning: {len(''.join(reasoning))} chars streamed before the answer")
print("    content:", repr("".join(content)))
PYCODE
then
  ok "streamed chat completion, SSE framing intact, terminated by [DONE]"
else
  bad "streamed chat: $(head -c 400 "$WORK/chat.sse")"
fi

# ------------------------------------ 7. a wrong model name is 409, naming this one

CODE="$(curl -sS -o "$WORK/409b.json" -w '%{http_code}' "${AUTH[@]}" "${JSON[@]}" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}' \
  "$BASE/openai/chat/completions")"
if [ "$CODE" = "409" ] && "$PY" - "$WORK/409b.json" "$MODEL" <<'PYCODE'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["code"] == "model_not_resident", error
assert error["details"] == {"requested": "gpt-4", "resident": sys.argv[2]}, error
print("    body:", json.dumps(error))
PYCODE
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
  UNLOAD_ID="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$WORK/unload.json")"
  curl -sS -N --max-time 600 "${AUTH[@]}" "$BASE/jobs/$UNLOAD_ID/events" >"$WORK/unload.sse"
  if "$PY" - "$WORK/unload.sse" <<'PYCODE'
import json, sys
kinds, data = [], []
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith("event:"):
        kinds.append(line.split(":", 1)[1].strip())
    elif line.startswith("data:"):
        data.append(json.loads(line.split(":", 1)[1].strip()))
if kinds[-1] == "failed":
    raise SystemExit("the unload failed: " + json.dumps(data[-1]))
assert kinds[-1] == "done", kinds
assert data[-1].get("resident") is None, data[-1]
PYCODE
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

CODE="$(curl -sS -o "$WORK/openaimodels.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/openai/models")"
if [ "$CODE" = "200" ] && grep -q '"data":\[\]' "$WORK/openaimodels.json"; then
  ok "GET /openai/models is an empty list once nothing is resident"
else
  bad "GET /openai/models after unload returned $CODE: $(cat "$WORK/openaimodels.json")"
fi

# ----------------------------------------------- 9. the accelerator came back

if [ "$REMOTE" = "1" ]; then
  log "remote mode: the card belongs to the server, and a client cannot see it."
  log "             /health and /openai/models above are what a client can check."
else
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
fi

echo "keeper: $PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
