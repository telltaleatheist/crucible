#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="$REPO/sdk/ts"
SERVER_PID=""
ROOT=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then
    rm -rf "$ROOT"
  fi
}
trap cleanup EXIT

command -v crucible >/dev/null || { echo "e2e: no crucible on PATH (pip install -e .)" >&2; exit 2; }
command -v node >/dev/null     || { echo "e2e: no node on PATH" >&2; exit 2; }
command -v npm >/dev/null      || { echo "e2e: no npm on PATH" >&2; exit 2; }
command -v python3 >/dev/null  || { echo "e2e: no python3 on PATH" >&2; exit 2; }

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 20 ] || { echo "e2e: node $NODE_MAJOR is too old; the SDK needs 20+" >&2; exit 2; }

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/crucible-e2e.XXXXXX")"
export CRUCIBLE_HOME="$ROOT/home"

PORT="$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"

echo "e2e: home=$CRUCIBLE_HOME port=$PORT node=$(node --version)"

crucible init --enable-echo --host 127.0.0.1 --port "$PORT" --name "crucible@e2e" >"$ROOT/init.txt"
TOKEN="$(crucible token --show)"
[ -n "$TOKEN" ] || { echo "e2e: the minted token is empty" >&2; exit 2; }

crucible serve --host 127.0.0.1 --port "$PORT" --log-level warning >"$ROOT/serve.log" 2>&1 &
SERVER_PID=$!

BASE="http://127.0.0.1:$PORT"
UP=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "$BASE/v1/ping" -o "$ROOT/ping.json" 2>/dev/null; then
    UP=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "e2e: the server exited before answering:" >&2
    cat "$ROOT/serve.log" >&2
    exit 2
  fi
  sleep 0.25
done
[ "$UP" = "1" ] || { echo "e2e: the server never answered $BASE/v1/ping" >&2; cat "$ROOT/serve.log" >&2; exit 2; }

cd "$SDK"
npm ci --no-audit --no-fund
npm run build
npm run build:test

export CRUCIBLE_URL="$BASE"
export CRUCIBLE_TOKEN="$TOKEN"

set +e
node --test build/test/e2e.test.js
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
  echo "e2e: the suite failed ($STATUS); the server said:" >&2
  cat "$ROOT/serve.log" >&2
fi
exit "$STATUS"
