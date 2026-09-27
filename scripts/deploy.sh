#!/usr/bin/env bash
set -euo pipefail

REPO_SLUG="telltaleatheist/crucible"

FLEET="pc mac"

DISTRO="Ubuntu"

fail() { echo "deploy: $*" >&2; exit 1; }

release=""
only=""
assume_yes=0
force=0
interrupt=0

usage() {
  cat <<'USAGE'
Put a release on every machine that runs one, and prove it landed.

  ./scripts/deploy.sh                              what each machine runs today
  ./scripts/deploy.sh --release 0.6.8              install that release everywhere
  ./scripts/deploy.sh --release 0.6.8 --only pc    only the named machines (pc, mac)
  ./scripts/deploy.sh --release 0.6.8 --yes        do not ask before restarting services
  ./scripts/deploy.sh --release 0.6.8 --interrupt  restart a BUSY server, or one whose busy state is unknown
  ./scripts/deploy.sh --release 0.6.8 --force      install even where the record names it, or names a NEWER release
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --release) [ $# -ge 2 ] || fail "--release needs a version"; release="$2"; shift 2 ;;
    --only)    [ $# -ge 2 ] || fail "--only needs a comma-separated list"; only="$2"; shift 2 ;;
    --yes|-y)  assume_yes=1; shift ;;
    --force)   force=1; shift ;;
    --interrupt) interrupt=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument $1" ;;
  esac
done

if [ -n "$release" ]; then
  case "$release" in
    v*) fail "name the version without the leading v: --release ${release#v}" ;;
  esac
  echo "$release" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
    || fail "--release wants a x.y.z version, got '$release'"
fi

for name in $(echo "$only" | tr ',' ' '); do
  case " $FLEET " in *" $name "*) continue ;; esac
  case "$name" in
    wsl)     fail "there is no 'wsl' machine any more: the host drives the guest (docs/internals/scripts.md, \"Deploying\"), so the PC is one entry. Use --only pc" ;;
    windows) fail "'windows' and 'wsl' are one machine now, called pc. Use --only pc" ;;
    *)       fail "--only names '$name', which is not one of: $FLEET" ;;
  esac
done

selected() {
  [ -z "$only" ] && return 0
  case ",$only," in *,"$1",*) return 0 ;; esac
  return 1
}

if selected pc; then
  command -v wsl.exe >/dev/null 2>&1 \
    || fail "pc is read and installed from the Windows PC itself (it needs Windows' wsl command on PATH), and this host has none. Run this on the PC, or pass --only mac"
  [ -n "${LOCALAPPDATA:-}" ] \
    || fail "pc is read and installed from the Windows PC itself (it needs LOCALAPPDATA, where the host keeps installation.json), and this host has none. Run this on the PC, or pass --only mac"
fi

parse_release() {
  python -c '
import json, sys
try:
    print(json.load(sys.stdin)["release"])
except Exception:
    pass
'
}

record_guest() {
  local out
  out="$(wsl.exe -d "$DISTRO" --exec bash -c 'cat "$HOME/.crucible/installation.json" 2>/dev/null' </dev/null 2>/dev/null | parse_release || true)"
  if [ -z "$out" ]; then
    wsl.exe -d "$DISTRO" --exec bash -c 'exit 0' </dev/null >/dev/null 2>&1 || { echo unreachable; return; }
    echo none; return
  fi
  echo "$out"
}

record_host() {
  local out
  out="$(cat "$LOCALAPPDATA/crucible/installation.json" 2>/dev/null | parse_release || true)"
  [ -n "$out" ] && { echo "$out"; return; }
  echo none
}

read_pc() {
  local host guest
  host="$(record_host)"
  guest="$(record_guest)"
  [ "$host" = "$guest" ] && { echo "$host"; return; }
  echo "host:$host guest:$guest"
}

unreachable() {
  case "$1" in unreachable|*:unreachable|*:unreachable\ *) return 0 ;; esac
  return 1
}

newer_than_release() {
  local reading="$1" want="$2"
  python - "$reading" "$want" <<'PY'
import re, sys
reading, want = sys.argv[1], sys.argv[2]
number = lambda text: tuple(int(part) for part in text.split("."))
seen = [number(found) for found in re.findall(r"\b\d+\.\d+\.\d+\b", reading)]
raise SystemExit(0 if any(found > number(want) for found in seen) else 1)
PY
}

read_mac() {
  local out
  out="$(ssh -n -o ConnectTimeout=8 -o BatchMode=yes mac 'cat "$HOME/.crucible/installation.json" 2>/dev/null' 2>/dev/null | parse_release || true)"
  if [ -z "$out" ]; then
    ssh -n -o ConnectTimeout=8 -o BatchMode=yes mac true >/dev/null 2>&1 || { echo unreachable; return; }
    echo none; return
  fi
  echo "$out"
}

install_sh_url() { echo "https://github.com/$REPO_SLUG/releases/download/v$1/install.sh"; }
install_ps1_url() { echo "https://github.com/$REPO_SLUG/releases/download/v$1/install.ps1"; }

shquote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

remote_install_payload() {
  printf 'set -e; f=$(mktemp); curl -fsSL --retry 3 -o "$f" %s; sh -n "$f"; sh "$f" --release %s; rm -f "$f"' \
    "'$1'" "'$2'"
}

install_mac() {
  ssh -n -o ConnectTimeout=15 mac \
    "\"\$SHELL\" -lc $(shquote "$(remote_install_payload "$(install_sh_url "$1")" "$1")")"
}

install_pc() {
  local script="${TMPDIR:-/tmp}/crucible-install-$1.ps1" status=0
  curl -fsSL -o "$script" "$(install_ps1_url "$1")"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$script" 2>/dev/null || echo "$script")" -Release "$1" || status=$?
  rm -f "$script" || echo "deploy: could not remove $script" >&2
  return "$status"
}

AWAIT_ATTEMPTS="${DEPLOY_AWAIT_ATTEMPTS:-150}"

await_release() {
  local machine="$1" want="$2" seen=""
  local attempt=0
  while [ "$attempt" -lt "$AWAIT_ATTEMPTS" ]; do
    seen="$("read_$machine")"
    [ "$seen" = "$want" ] && { echo "$seen"; return; }
    attempt=$(( attempt + 1 ))
    sleep 2
  done
  echo "$seen"
}

echo "deploy: what each machine runs"
declare -A BEFORE
any_behind=0
downgrades=""
for machine in $FLEET; do
  selected "$machine" || continue
  BEFORE[$machine]="$("read_$machine")"
  note=""
  if [ -n "$release" ]; then
    if [ "${BEFORE[$machine]}" = "$release" ]; then
      if [ "$force" = "1" ]; then note="  (already $release, reinstalling anyway)"; any_behind=1
      else note="  (already $release)"; fi
    elif unreachable "${BEFORE[$machine]}"; then
      note="  (CANNOT BE ASKED — will not be touched)"
    elif newer_than_release "${BEFORE[$machine]}" "$release"; then
      if [ "$force" = "1" ]; then note="  -> $release  (a DOWNGRADE, because --force)"; any_behind=1
      else note="  (NEWER than $release — refused without --force)"; downgrades="$downgrades $machine"; fi
    else
      note="  -> $release"; any_behind=1
    fi
  fi
  printf '  %-8s %s%s\n' "$machine" "${BEFORE[$machine]}" "$note"
done

if [ -z "$release" ]; then
  echo "deploy: pass --release <x.y.z> to install one"
  exit 0
fi

if [ -n "$downgrades" ]; then
  fail "$release is older than what runs on:$downgrades. A deploy never goes backwards by accident; re-run with --force to downgrade on purpose"
fi

if [ "$any_behind" = "0" ]; then
  echo "deploy: every selected machine already runs $release"
  exit 0
fi

if [ "$assume_yes" != "1" ]; then
  echo "deploy: this restarts the WSL engine, the Windows tray host and the Mac agent."
  printf 'deploy: install %s on the machines marked above? [y/N] ' "$release"
  read -r answer || answer=""
  case "$answer" in y|Y|yes|YES) ;; *) echo "deploy: nothing was changed"; exit 1 ;; esac
fi

work="$(mktemp -d)"
trap 'rm -rf "$work" 2>/dev/null || echo "deploy: could not remove $work" >&2' EXIT

busy_probe() {
  cat <<'PROBE'
home="${CRUCIBLE_HOME:-$HOME/.crucible}"
config="$home/config.toml"
[ -f "$config" ] || { echo "unknown(no $config to read the token and port from)"; exit 0; }
tok=$(sed -n 's/^token *= *"\(.*\)"/\1/p' "$config" 2>/dev/null | head -1)
[ -n "$tok" ] || { echo "unknown(no token in $config to ask with)"; exit 0; }
port=$(sed -n 's/^port *= *\([0-9][0-9]*\).*/\1/p' "$config" 2>/dev/null | head -1)
[ -n "$port" ] || port=7100
host=$(sed -n 's/^host *= *"\(.*\)"/\1/p' "$config" 2>/dev/null | head -1)
case "$host" in ""|0.0.0.0) host=127.0.0.1 ;; "::") host="[::1]" ;; *:*) host="[$host]" ;; esac
py="$home/server/bin/python"
[ -x "$py" ] || py="$(command -v python3 || true)"
[ -n "$py" ] || { echo "unknown(no python at $home/server/bin/python to read the answer with)"; exit 0; }
status=0
body=$(curl -sS -m 8 -H "Authorization: Bearer $tok" -H "X-Crucible-Api: 1" \
  "http://$host:$port/v1/activity" 2>/dev/null) || status=$?
if [ "$status" = "7" ]; then echo "idle(nothing listens on $host:$port)"; exit 0; fi
[ "$status" = "0" ] || { echo "unknown(curl exited $status asking $host:$port/v1/activity)"; exit 0; }
[ -n "$body" ] || { echo "unknown($host:$port/v1/activity answered nothing)"; exit 0; }
printf '%s' "$body" | "$py" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as exc:
    print("unknown(/v1/activity answered something that is not JSON: %s)" % exc); raise SystemExit(0)
if not isinstance(d, dict) or "running" not in d:
    print("unknown(/v1/activity answered without a running list: %s)" % str(d)[:120]); raise SystemExit(0)
busy = []
for row in d.get("running") or []:
    busy.append("job %s (%s) %d%% done" % (row.get("job_id"), row.get("type"),
                                           round((row.get("progress") or 0) * 100)))
queued = len(d.get("queued") or [])
if queued:
    busy.append("%d queued" % queued)
if d.get("streaming"):
    busy.append("a streaming session")
chat = (d.get("chat") or {}).get("in_flight") or 0
if chat:
    busy.append("%d chat completion(s) in flight" % chat)
lease = d.get("lease")
if lease:
    busy.append("a lease held by %s for %s until %s" % (
        lease.get("client") or "an unnamed client", lease.get("act"),
        lease.get("expires_at")))
print(("busy(" + "; ".join(busy) + ")") if busy else "idle")
' 
PROBE
}

busy_pc() {
  local said
  said="$(wsl.exe -d "$DISTRO" --exec bash -c "$(busy_probe)" </dev/null 2>/dev/null)" \
    || { echo "unknown(the probe would not run inside $DISTRO)"; return; }
  [ -n "$said" ] && echo "$said" || echo "unknown(the probe in $DISTRO printed nothing)"
}

busy_mac() {
  local said
  said="$(ssh -n -o ConnectTimeout=8 -o BatchMode=yes mac "$(busy_probe)" 2>/dev/null)" \
    || { echo "unknown(ssh mac would not run the probe)"; return; }
  [ -n "$said" ] && echo "$said" || echo "unknown(the probe on mac printed nothing)"
}

deploy_one() {
  local machine="$1" want="$2" started after
  started="$(date +%s)"
  if "install_$machine" "$want"; then
    after="$(await_release "$machine" "$want")"
    if [ "$after" = "$want" ]; then
      echo "now runs $after"
    else
      echo "still reports $after a minute after installing $want"
      echo "reports:$after" > "$work/$machine.why"
    fi
  else
    echo "installer failed" > "$work/$machine.why"
  fi
  echo $(( $(date +%s) - started )) > "$work/$machine.seconds"
}

failed=""
running=""
for machine in $FLEET; do
  selected "$machine" || continue
  if [ "${BEFORE[$machine]}" = "$release" ] && [ "$force" != "1" ]; then
    continue
  fi
  if unreachable "${BEFORE[$machine]}"; then
    failed="$failed $machine(unreachable: ${BEFORE[$machine]})"; continue
  fi

  if [ "$interrupt" != "1" ]; then
    state="$("busy_$machine")"
    case "$state" in
      busy*)
        echo
        echo "deploy: $machine is WORKING and was not touched: ${state#busy}"
        echo "deploy:   it would have been restarted mid-job, which loses"
        echo "deploy:   whatever the job had not yet written to disk."
        echo "deploy:   Wait for it, or re-run with --interrupt to take it anyway."
        failed="$failed $machine(busy)"
        continue ;;
      idle|idle\(*)
        ;;
      *)
        echo
        echo "deploy: $machine was not touched: whether it is working could not be learned: ${state#unknown}"
        echo "deploy:   a restart of a server that might be mid-job loses whatever"
        echo "deploy:   the job had not yet written to disk, so not knowing is a refusal."
        echo "deploy:   Fix what the probe names, or re-run with --interrupt to take it anyway."
        failed="$failed $machine(busy state unknown)"
        continue ;;
    esac
  fi

  running="$running $machine"
  echo
  echo "deploy: $machine  ${BEFORE[$machine]} -> $release"
  deploy_one "$machine" "$release" 2>&1 | sed "s/^/$machine: /" &
done

wait

echo
for machine in $running; do
  if [ ! -f "$work/$machine.seconds" ]; then
    failed="$failed $machine(no result: its subshell left no record)"
    continue
  fi
  echo "deploy: timing $machine $(cat "$work/$machine.seconds")"
  if [ -s "$work/$machine.why" ]; then
    failed="$failed $machine($(cat "$work/$machine.why"))"
  fi
done

echo
if [ -n "$failed" ]; then
  echo "deploy: NOT everywhere —$failed" >&2
  exit 1
fi
echo "deploy: $release is on every selected machine"
echo "deploy: the candidate is installed, so it can now be promoted:"
echo "  $(python "$(dirname "${BASH_SOURCE[0]}")/promote_release.py" --tag "v$release" --print-command)"
