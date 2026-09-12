#!/usr/bin/env bash
# Live keeper: start a real crucible server on a free port with a throwaway home,
# then drive ping / info / auth refusals / an echo job over HTTP with curl.
#
# Exits 0 only if every check passed. Trust the exit code.
#
#   ./scripts/keeper-live.sh
#
# Requires: the `crucible` console script on PATH (pip install -e .), curl, python3.

set -euo pipefail

PASSED=0
FAILED=0
SERVER_PID=""
ROOT=""

log()  { printf '  %s\n' "$*"; }
ok()   { PASSED=$((PASSED + 1)); printf 'ok    %s\n' "$*"; }
bad()  { FAILED=$((FAILED + 1)); printf 'FAIL  %s\n' "$*" >&2; }

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi
  if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then
    rm -rf "$ROOT"
  fi
}
trap cleanup EXIT

command -v crucible >/dev/null || { echo "keeper: no crucible on PATH" >&2; exit 2; }
command -v curl >/dev/null     || { echo "keeper: no curl on PATH" >&2; exit 2; }
command -v python3 >/dev/null  || { echo "keeper: no python3 on PATH" >&2; exit 2; }

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/crucible-keeper.XXXXXX")"
export CRUCIBLE_HOME="$ROOT/home"
WORK="$ROOT/work"
mkdir -p "$WORK"

PORT="$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"

echo "keeper: home=$CRUCIBLE_HOME port=$PORT"

# ---------------------------------------------------------------- init + serve

crucible init --enable-echo --host 127.0.0.1 --port "$PORT" --name "crucible@keeper" >"$WORK/init.txt"
log "$(head -n 2 "$WORK/init.txt" | tr '\n' ' ')"
TOKEN="$(crucible token --show)"
[ -n "$TOKEN" ] || { echo "keeper: token is empty" >&2; exit 2; }

crucible doctor --json >"$WORK/doctor.json"
python3 - "$WORK/doctor.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if not report["healthy"]:
    raise SystemExit("doctor is unhealthy: " + "; ".join(report["problems"]))
PY
ok "doctor reports healthy"

BASE="http://127.0.0.1:$PORT/v1"
crucible serve --host 127.0.0.1 --port "$PORT" --log-level warning >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!

UP=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "$BASE/ping" -o "$WORK/ping.json" 2>/dev/null; then
    UP=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "keeper: the server exited before answering:" >&2
    cat "$WORK/serve.log" >&2
    exit 2
  fi
  sleep 0.25
done
[ "$UP" = "1" ] || { echo "keeper: server never answered on $BASE/ping" >&2; cat "$WORK/serve.log" >&2; exit 2; }

# ------------------------------------------------------------------- checks

AUTH=(-H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1")

# 1. ping, unauthenticated
if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("crucible") is True and d.get("api_version")==1 else 1)' "$WORK/ping.json"; then
  ok "GET /ping without auth"
else
  bad "GET /ping did not identify as crucible api 1: $(cat "$WORK/ping.json")"
fi

# 2. info, authenticated
CODE="$(curl -sS -o "$WORK/info.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/info")"
if [ "$CODE" = "200" ] && python3 - "$WORK/info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
assert info["server"]["api_version"] == 1, info
assert info["host"]["backend"] in ("cuda-linux", "mlx-darwin"), info
assert info["host"]["gpu"]["name"], info
assert any(c["job_type"] == "echo" for c in info["capabilities"]), info
PY
then
  ok "GET /info ($(python3 -c 'import json,sys; i=json.load(open(sys.argv[1])); print(i["host"]["backend"], "-", i["host"]["gpu"]["name"])' "$WORK/info.json"))"
else
  bad "GET /info returned $CODE: $(cat "$WORK/info.json")"
fi

# 3. wrong token is 401
CODE="$(curl -sS -o "$WORK/401.json" -w '%{http_code}' -H "Authorization: Bearer wrong" -H "X-Crucible-Api: 1" "$BASE/info")"
if [ "$CODE" = "401" ] && grep -q '"unauthorized"' "$WORK/401.json"; then
  ok "wrong token is 401 unauthorized"
else
  bad "wrong token returned $CODE: $(cat "$WORK/401.json")"
fi

# 4. missing api version header is 426
CODE="$(curl -sS -o "$WORK/426.json" -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$BASE/info")"
if [ "$CODE" = "426" ] && grep -q 'api_version' "$WORK/426.json"; then
  ok "missing X-Crucible-Api header is 426"
else
  bad "missing api version returned $CODE: $(cat "$WORK/426.json")"
fi

# 5. wrong api version is 426 naming both
CODE="$(curl -sS -o "$WORK/426b.json" -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 99" "$BASE/info")"
if [ "$CODE" = "426" ] && python3 - "$WORK/426b.json" <<'PY'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["code"] == "api_version_mismatch", error
assert error["details"] == {"server_api_version": 1, "client_api_version": 99}, error
PY
then
  ok "api version 99 is 426 naming both versions"
else
  bad "api version 99 returned $CODE: $(cat "$WORK/426b.json")"
fi

# 6. unknown job type is refused by name
CODE="$(curl -sS -o "$WORK/badtype.json" -w '%{http_code}' "${AUTH[@]}" -H 'Content-Type: application/json' -d '{"type":"summon"}' "$BASE/jobs")"
if [ "$CODE" = "400" ] && grep -q 'unknown_job_type' "$WORK/badtype.json"; then
  ok "unknown job type is 400 unknown_job_type"
else
  bad "unknown job type returned $CODE: $(cat "$WORK/badtype.json")"
fi

# 7. echo, end to end
head -c 65536 /dev/urandom >"$WORK/payload.bin"
python3 - "$WORK/payload.bin" "$WORK/job.json" <<'PY'
import base64, json, sys
data = open(sys.argv[1], "rb").read()
body = {
    "type": "echo",
    "params": {"delay_ms": 40},
    "inputs": {"payload.bin": {"inline_base64": base64.b64encode(data).decode("ascii")}},
}
json.dump(body, open(sys.argv[2], "w"))
PY

CODE="$(curl -sS -o "$WORK/created.json" -w '%{http_code}' "${AUTH[@]}" -H 'Content-Type: application/json' --data-binary "@$WORK/job.json" "$BASE/jobs")"
if [ "$CODE" != "202" ]; then
  bad "POST /jobs returned $CODE: $(cat "$WORK/created.json")"
else
  ok "POST /jobs accepted (202)"
  JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$WORK/created.json")"

  curl -sS -N --max-time 60 "${AUTH[@]}" "$BASE/jobs/$JOB_ID/events" >"$WORK/events.sse"
  if python3 - "$WORK/events.sse" <<'PY'
import sys
kinds = [line.split(":", 1)[1].strip()
         for line in open(sys.argv[1])
         if line.startswith("event:")]
assert kinds, "no events in the stream"
assert kinds[0] == "queued", kinds
assert "artifact" in kinds, kinds
assert kinds[-1] == "done", kinds
assert kinds.index("artifact") < kinds.index("done"), kinds
PY
  then
    ok "SSE stream: queued ... artifact ... done"
  else
    bad "SSE stream out of order: $(grep '^event:' "$WORK/events.sse" | tr '\n' ' ')"
  fi

  CODE="$(curl -sS -o "$WORK/echoed.bin" -w '%{http_code}' "${AUTH[@]}" "$BASE/jobs/$JOB_ID/artifacts/payload.bin")"
  if [ "$CODE" = "200" ] && python3 - "$WORK/payload.bin" "$WORK/echoed.bin" <<'PY'
import hashlib, sys
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
a, b = digest(sys.argv[1]), digest(sys.argv[2])
assert a == b, f"{a} != {b}"
PY
  then
    ok "artifact bytes are identical to the input"
  else
    bad "artifact fetch returned $CODE or the bytes differ"
  fi

  CODE="$(curl -sS -o "$WORK/prov.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/jobs/$JOB_ID/artifacts/payload.bin.provenance.json")"
  if [ "$CODE" = "200" ] && python3 - "$WORK/prov.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for key in ("server", "backend", "job_type", "model", "params", "started", "finished"):
    assert key in p, (key, p)
assert p["job_type"] == "echo", p
assert p["started"] and p["finished"], p
PY
  then
    ok "provenance sidecar is present and complete"
  else
    bad "provenance fetch returned $CODE: $(cat "$WORK/prov.json")"
  fi
fi

# 8. health
CODE="$(curl -sS -o "$WORK/health.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/health")"
if [ "$CODE" = "200" ] && grep -q '"queue_depth"' "$WORK/health.json"; then
  ok "GET /health"
else
  bad "GET /health returned $CODE: $(cat "$WORK/health.json")"
fi

echo "keeper: $PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
