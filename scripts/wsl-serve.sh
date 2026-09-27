#!/usr/bin/env bash
set -euo pipefail

command="${1:-}"
home="${2:-}"

[ -n "$command" ] || { echo "wsl-serve: no command (start|stop|log)" >&2; exit 2; }
[ -n "$home" ]    || { echo "wsl-serve: no home directory" >&2; exit 2; }

case "$command" in
  start)
    crucible="${3:-}"
    [ -x "$crucible" ] || { echo "wsl-serve: $crucible is not executable" >&2; exit 2; }
    mkdir -p "$home"

    port="$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"

    export CRUCIBLE_HOME="$home"
    "$crucible" init --enable-echo --host 127.0.0.1 --port "$port" \
      --name "crucible@wsl-e2e" >"$home/init.log"

    setsid bash -c '
      echo $$ >"$1/pid"
      exec "$2" serve --host 127.0.0.1 --port "$3" --log-level warning
    ' wsl-serve "$home" "$crucible" "$port" </dev/null >"$home/serve.log" 2>&1 &
    disown

    for _ in $(seq 1 40); do
      [ -s "$home/pid" ] && break
      sleep 0.25
    done
    [ -s "$home/pid" ] || { echo "wsl-serve: the server never recorded a pid" >&2; exit 2; }

    echo "port=$port"
    echo "token=$("$crucible" token --show)"
    ;;

  stop)
    if [ -s "$home/pid" ]; then
      pid="$(cat "$home/pid")"
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid"
        for _ in $(seq 1 60); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.25
        done
        if kill -0 "$pid" 2>/dev/null; then
          echo "wsl-serve: pid $pid ignored SIGTERM; leaving it and $home alone" >&2
          exit 1
        fi
      fi
    fi
    rm -rf "$home"
    ;;

  log)
    cat "$home/serve.log" 2>/dev/null || true
    ;;

  *)
    echo "wsl-serve: unknown command $command (start|stop|log)" >&2
    exit 2
    ;;
esac
