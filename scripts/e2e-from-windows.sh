#!/usr/bin/env bash
# End-to-end on Windows: the server in WSL2, the client native.
#
# This is the arrangement BookForge will actually ship on the PC — Crucible has
# no Windows code path at all (DESIGN.md section 2), so the Electron main
# process talks over HTTP to a Linux server one hop away. Running the suite this
# way proves the seam, not just the code.
#
#   ./scripts/e2e-from-windows.sh          # from Git Bash, in the repo root
#
# The Windows side needs: node 20+, npm, curl, wsl.exe.
# The WSL side needs: the crucible checkout at $WSL_REPO on the branch under
# test, with `pip install -e .` done in the conda env at $WSL_ENV.
#
# The WSL server is stopped with SIGTERM. Nothing here ever sends SIGKILL into
# the guest: a hard-killed GPU process wedges the whole distro.

set -euo pipefail

DISTRO="${CRUCIBLE_WSL_DISTRO:-Ubuntu}"
WSL_REPO="${CRUCIBLE_WSL_REPO:-/home/telltale/crucible}"
WSL_ENV="${CRUCIBLE_WSL_ENV:-/home/telltale/anaconda3/envs/crucible}"
CRUCIBLE_BIN="$WSL_ENV/bin/crucible"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="$REPO/sdk/ts"
WSL_HOME=""

# `--exec` matters: without it wsl.exe hands the string to the login shell on
# the Windows side first, which pre-expands $vars before Linux ever sees them.
wsl_run() {
  wsl.exe -d "$DISTRO" --exec bash -c "$1"
}

cleanup() {
  if [ -n "$WSL_HOME" ]; then
    wsl_run "
      if [ -f '$WSL_HOME/pid' ]; then
        pid=\$(cat '$WSL_HOME/pid')
        if kill -0 \$pid 2>/dev/null; then
          kill -TERM \$pid
          for _ in \$(seq 1 40); do
            kill -0 \$pid 2>/dev/null || break
            sleep 0.25
          done
        fi
      fi
      rm -rf '$WSL_HOME'
    " >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

command -v node >/dev/null || { echo "e2e: no node on PATH" >&2; exit 2; }
command -v npm >/dev/null  || { echo "e2e: no npm on PATH" >&2; exit 2; }
command -v curl >/dev/null || { echo "e2e: no curl on PATH" >&2; exit 2; }
command -v wsl.exe >/dev/null || { echo "e2e: no wsl.exe on PATH" >&2; exit 2; }

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 20 ] || { echo "e2e: node $NODE_MAJOR is too old; the SDK needs 20+" >&2; exit 2; }

wsl_run "test -x '$CRUCIBLE_BIN'" \
  || { echo "e2e: $CRUCIBLE_BIN is missing in $DISTRO (pip install -e . in that env)" >&2; exit 2; }

# What the WSL checkout is actually on, so the report cannot claim the wrong tree.
WSL_HEAD="$(wsl_run "git -C '$WSL_REPO' rev-parse --short HEAD" | tr -d '\r')"
WSL_BRANCH="$(wsl_run "git -C '$WSL_REPO' rev-parse --abbrev-ref HEAD" | tr -d '\r')"
echo "e2e: wsl server from $WSL_REPO @ $WSL_BRANCH ($WSL_HEAD)"

WSL_HOME="$(wsl_run "mktemp -d /tmp/crucible-e2e.XXXXXX" | tr -d '\r')"
[ -n "$WSL_HOME" ] || { echo "e2e: could not make a temp home in $DISTRO" >&2; exit 2; }

PORT="$(wsl_run "python3 -c \"
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
\"" | tr -d '\r')"
[ -n "$PORT" ] || { echo "e2e: could not pick a free port in $DISTRO" >&2; exit 2; }

echo "e2e: wsl home=$WSL_HOME port=$PORT windows node=$(node --version)"

wsl_run "CRUCIBLE_HOME='$WSL_HOME' '$CRUCIBLE_BIN' init --enable-echo --host 127.0.0.1 --port $PORT --name 'crucible@wsl-e2e'" >/dev/null
TOKEN="$(wsl_run "CRUCIBLE_HOME='$WSL_HOME' '$CRUCIBLE_BIN' token --show" | tr -d '\r')"
[ -n "$TOKEN" ] || { echo "e2e: the minted token is empty" >&2; exit 2; }

# nohup so the server outlives this wsl.exe invocation; the pid file is how the
# next invocation finds it to send SIGTERM.
wsl_run "cd '$WSL_HOME' && CRUCIBLE_HOME='$WSL_HOME' nohup '$CRUCIBLE_BIN' serve --host 127.0.0.1 --port $PORT --log-level warning >'$WSL_HOME/serve.log' 2>&1 & echo \$! > '$WSL_HOME/pid'"

BASE="http://127.0.0.1:$PORT"
UP=0
for _ in $(seq 1 160); do
  if curl -fsS --max-time 2 "$BASE/v1/ping" -o /dev/null 2>/dev/null; then
    UP=1
    break
  fi
  sleep 0.25
done
if [ "$UP" != "1" ]; then
  echo "e2e: the WSL server never answered $BASE/v1/ping from Windows" >&2
  wsl_run "cat '$WSL_HOME/serve.log' 2>/dev/null" >&2 || true
  exit 2
fi
echo "e2e: windows reached the WSL server at $BASE"

# ------------------------------------------------------------------- the suite

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
  echo "e2e: the suite failed ($STATUS); the WSL server said:" >&2
  wsl_run "cat '$WSL_HOME/serve.log' 2>/dev/null" >&2 || true
fi
exit "$STATUS"
