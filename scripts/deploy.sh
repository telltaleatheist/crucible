#!/usr/bin/env bash
# Put a release on every machine that runs one, and prove it landed.
#
#   ./scripts/deploy.sh                    # what each machine runs today
#   ./scripts/deploy.sh --release 0.6.8    # install that release everywhere
#   ./scripts/deploy.sh --release 0.6.8 --only wsl,mac
#   ./scripts/deploy.sh --release 0.6.8 --yes     # do not ask first
#
# THREE MACHINES RUN CRUCIBLE and until now each was upgraded by hand, in its
# own shell, with its own spelling of the same command. That is why they drift:
# on 2026-09-16 the WSL engine was 0.6.3, the Mac 0.6.3 and the Windows host
# 0.6.5, while the newest release was 0.6.7. Nothing was broken. Three manual
# steps had simply been done a different number of times.
#
# WHAT A MACHINE RUNS IS READ FROM THE MACHINE, never assumed: every install
# writes `<CRUCIBLE_HOME>/installation.json` with the release it unpacked, and
# that one file has the same meaning on Linux, macOS and Windows. It is read
# BEFORE (to decide whether there is anything to do) and AFTER (to prove the
# install did what it said). An install whose after-value is not the release
# asked for is a FAILURE here, however cheerful its own output was.
#
# THE INSTALLER COMES FROM THE RELEASE BEING INSTALLED, not from
# `releases/latest/download/`. The documented one-liner deliberately uses
# `latest` for a person starting from nothing, and `latest` only moves when
# `promote_release.py` runs — which is exactly how a candidate that nobody has
# installed yet becomes impossible to install with its own script. Naming the
# tag sidesteps the whole question: v0.6.8's install.sh installs v0.6.8.
#
# Every installer here is idempotent and re-runnable after a failure; that is
# their own contract, not an assumption of this script.
#
# THIS RESTARTS SERVICES. The WSL engine, the Windows tray host and the Mac
# launchd agent all go down and come back. It asks before it does that unless
# --yes is passed.

set -euo pipefail

REPO_SLUG="telltaleatheist/crucible"

# THE FLEET. These three are a fact about this deployment, not about Crucible —
# a different operator has a different list, and there is nothing secret here:
# `mac` is an ssh alias from ~/.ssh/config, and `Ubuntu` is the WSL distro name.
#
# `windows` is the HOST, not an engine: since PHASE15-HOST.md 4.4 install.ps1
# installs `crucible host` and stops, and the host owns the WSL sequence from
# there. So `wsl` and `windows` are two installs on one physical machine, and
# both are listed because both have their own installation.json and drift
# independently — which is precisely what happened (0.6.3 beside 0.6.5).
FLEET="wsl windows mac"

fail() { echo "deploy: $*" >&2; exit 1; }

release=""
only=""
assume_yes=0

while [ $# -gt 0 ]; do
  case "$1" in
    --release) [ $# -ge 2 ] || fail "--release needs a version"; release="$2"; shift 2 ;;
    --only)    [ $# -ge 2 ] || fail "--only needs a comma-separated list"; only="$2"; shift 2 ;;
    --yes|-y)  assume_yes=1; shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# ------------------------------------------------------- what each machine runs
#
# Each `read_*` prints the installed release, or the word `none` when the
# machine has no installation.json, or `unreachable` when it cannot be asked.
# The three are told apart because they mean different things: `none` is a
# machine to install onto, `unreachable` is a machine whose state is UNKNOWN
# and which is therefore never reported as up to date.

# Printed by every reader, so the parsing lives in one place. Reads the file on
# stdin and prints its `release`, and prints nothing at all if it cannot.
parse_release() {
  python -c '
import json, sys
try:
    print(json.load(sys.stdin)["release"])
except Exception:
    pass
'
}

read_wsl() {
  local out
  out="$(wsl.exe -d Ubuntu --exec bash -c 'cat "$HOME/.crucible/installation.json" 2>/dev/null' </dev/null 2>/dev/null | parse_release || true)"
  if [ -z "$out" ]; then
    wsl.exe -d Ubuntu --exec bash -c 'exit 0' </dev/null >/dev/null 2>&1 || { echo unreachable; return; }
    echo none; return
  fi
  echo "$out"
}

read_windows() {
  local out
  out="$(cat "$LOCALAPPDATA/crucible/installation.json" 2>/dev/null | parse_release || true)"
  [ -n "$out" ] && { echo "$out"; return; }
  echo none
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

# --------------------------------------------------------------- the installers

install_sh_url() { echo "https://github.com/$REPO_SLUG/releases/download/v$1/install.sh"; }
install_ps1_url() { echo "https://github.com/$REPO_SLUG/releases/download/v$1/install.ps1"; }

install_wsl() {
  # --exec, so wsl.exe hands the string to bash instead of letting the Windows
  # side pre-expand a `$` in it first.
  wsl.exe -d Ubuntu --exec bash -lc \
    "set -e; curl -fsSL '$(install_sh_url "$1")' | sh -s -- --release '$1'"
}

install_mac() {
  # THE ACCOUNT'S OWN LOGIN SHELL, asked rather than named. A bare
  # `ssh host cmd` runs a NON-login shell, and macOS hands that
  # PATH=/usr/bin:/bin:/usr/sbin:/sbin - four directories, no Homebrew.
  # The 0.6.8 deploy refused the Mac for a missing zstd on 2026-09-17
  # while the machine had one at /opt/homebrew/bin/zstd the whole time.
  #
  # `bash -lc` does NOT fix it and was the first thing tried: bash reads
  # ~/.bash_profile, this account is zsh, and its Homebrew line lives in
  # ~/.zprofile. Naming a shell here guesses at something the machine
  # already knows, so $SHELL is expanded REMOTELY and answers for itself.
  ssh -n -o ConnectTimeout=15 mac \
    "\"\$SHELL\" -lc \"set -e; curl -fsSL '$(install_sh_url "$1")' | sh -s -- --release '$1'\""
}

install_windows() {
  # `irm | iex` cannot take a parameter, so the script is fetched to a file
  # first — the same reason its own header gives for -Uninstall.
  local script="${TMPDIR:-/tmp}/crucible-install-$1.ps1"
  curl -fsSL -o "$script" "$(install_ps1_url "$1")"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$script" 2>/dev/null || echo "$script")" -Release "$1"
  rm -f "$script"
}

# THE RECORD IS NOT ALWAYS WRITTEN BY THE INSTALLER. `installation.json` is
# published by `crucible local publish` when the runtime STARTS, so on Windows
# especially — where install.ps1 unpacks the host pack, launches the host and
# returns — the file can still hold the old release for a moment after the
# installer has exited successfully. Reading it once, immediately, is reading a
# race. This waits for the value to become the one asked for, and gives up with
# whatever it last saw so the caller reports the truth rather than a timeout.
await_release() {
  local machine="$1" want="$2" seen=""
  local attempt=0
  while [ "$attempt" -lt 20 ]; do
    seen="$("read_$machine")"
    [ "$seen" = "$want" ] && { echo "$seen"; return; }
    attempt=$(( attempt + 1 ))
    sleep 3
  done
  echo "$seen"
}

# ------------------------------------------------------------------- the report

echo "deploy: what each machine runs"
declare -A BEFORE
any_behind=0
for machine in $FLEET; do
  selected "$machine" || continue
  BEFORE[$machine]="$("read_$machine")"
  note=""
  if [ -n "$release" ]; then
    case "${BEFORE[$machine]}" in
      "$release")   note="  (already $release)" ;;
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
  # `|| answer=""` because `set -e` would otherwise make a closed stdin an
  # unexplained exit 1 instead of the refusal it actually is. That is not
  # hypothetical: every reader above used to eat the answer (ssh reads stdin
  # unless told not to), and the prompt exited silently with nothing typed.
  read -r answer || answer=""
  case "$answer" in y|Y|yes|YES) ;; *) echo "deploy: nothing was changed"; exit 1 ;; esac
fi

# ------------------------------------------------------------------- the work

failed=""
for machine in $FLEET; do
  selected "$machine" || continue
  case "${BEFORE[$machine]}" in
    "$release") continue ;;
    unreachable)
      # NOT a skip. A machine that could not be asked is a machine whose state
      # is unknown, and the summary must say so rather than imply it is fine.
      failed="$failed $machine(unreachable)"; continue ;;
  esac

  echo
  echo "deploy: $machine  ${BEFORE[$machine]} -> $release"
  if "install_$machine" "$release"; then
    after="$(await_release "$machine" "$release")"
    if [ "$after" = "$release" ]; then
      echo "deploy: $machine now runs $after"
    else
      echo "deploy: $machine still reports $after a minute after installing $release" >&2
      failed="$failed $machine(reports:$after)"
    fi
  else
    failed="$failed $machine(installer failed)"
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
