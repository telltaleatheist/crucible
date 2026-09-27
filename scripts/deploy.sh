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
  ./scripts/deploy.sh --release 0.6.8 --interrupt  restart a BUSY server too
  ./scripts/deploy.sh --release 0.6.8 --force      install even where the record names it
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

selected() {
  [ -z "$only" ] && return 0
  case ",$only," in *,"$1",*) return 0 ;; esac
  return 1
}

for name in $(echo "$only" | tr ',' ' '); do
  case " $FLEET " in *" $name "*) continue ;; esac
  case "$name" in
    wsl)     fail "there is no 'wsl' machine any more: the host drives the guest (PHASE15-HOST.md 4.3), so the PC is one entry. Use --only pc" ;;
    windows) fail "'windows' and 'wsl' are one machine now, called pc. Use --only pc" ;;
    *)       fail "--only names '$name', which is not one of: $FLEET" ;;
  esac
done

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

await_release() {
  local machine="$1" want="$2" seen=""
  local attempt=0
  while [ "$attempt" -lt 150 ]; do
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
for machine in $FLEET; do
  selected "$machine" || continue
  BEFORE[$machine]="$("read_$machine")"
  note=""
  if [ -n "$release" ]; then
    case "${BEFORE[$machine]}" in
      "$release")   if [ "$force" = "1" ]; then note="  (already $release, reinstalling anyway)"; any_behind=1
                    else note="  (already $release)"; fi ;;
      unreachable)  note="  (CANNOT BE ASKED — will not be touched)" ;;
      *)            note="  -> $release"; any_behind=1 ;;
    esac
  fi
  printf '  %-8s %s%s\n' "$machine" "${BEFORE[$machine]}" "$note"
done

if [ -z "$release" ]; then
  echo "deploy: pass --release <x.y.z> to install one"
  exit 0
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
tok=$(sed -n 's/^token *= *"\(.*\)"/\1/p' "$HOME/.crucible/config.toml" 2>/dev/null | head -1)
[ -n "$tok" ] || { echo "idle(no token to ask with)"; exit 0; }
body=$(curl -sS -m 8 -H "Authorization: Bearer $tok" -H "X-Crucible-Api: 1" \
  http://127.0.0.1:7100/v1/activity 2>/dev/null) || { echo "idle(not answering)"; exit 0; }
[ -n "$body" ] || { echo "idle(not answering)"; exit 0; }
printf '%s' "$body" | "$HOME/.crucible/server/bin/python" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("idle(activity unreadable)"); raise SystemExit(0)
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
  wsl.exe -d "$DISTRO" --exec bash -c "$(busy_probe)" </dev/null 2>/dev/null     || echo "idle(could not ask)"
}

busy_mac() {
  ssh -n -o ConnectTimeout=8 -o BatchMode=yes mac "$(busy_probe)" 2>/dev/null     || echo "idle(could not ask)"
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
  case "${BEFORE[$machine]}" in
    "$release") [ "$force" = "1" ] || continue ;;
    unreachable)
      failed="$failed $machine(unreachable)"; continue ;;
  esac

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
echo "  python scripts/promote_release.py --tag v$release --publish --confirmed-install-smoke"
