#!/usr/bin/env bash
# Put a release on every machine that runs one, and prove it landed.
#
#   ./scripts/deploy.sh                    # what each machine runs today
#   ./scripts/deploy.sh --release 0.6.8    # install that release everywhere
#   ./scripts/deploy.sh --release 0.6.8 --only pc
#   ./scripts/deploy.sh --release 0.6.8 --yes     # do not ask first
#
# TWO MACHINES RUN CRUCIBLE — the PC and the Mac — and until this existed each
# was upgraded by hand, in its own shell, with its own spelling of the same
# command. That is why they drift: on 2026-09-16 the WSL engine was 0.6.3, the
# Mac 0.6.3 and the Windows host 0.6.5, while the newest release was 0.6.7.
# Nothing was broken. Three manual steps had simply been done a different
# number of times.
#
# WHAT A MACHINE RUNS IS READ FROM THE MACHINE, never assumed: every install
# writes `<CRUCIBLE_HOME>/installation.json` with the release it unpacked, and
# that one file has the same meaning on Linux, macOS and Windows. It is read
# BEFORE (to decide whether there is anything to do) and AFTER (to prove the
# install did what it said). An install whose after-value is not the release
# asked for is a FAILURE here, however cheerful its own output was.
#
# THE PC HAS TWO OF THOSE RECORDS AND ONE INSTALL. The host's, on Windows, and
# the engine's, inside the distro — and only install.ps1 is run here, because
# the host is what drives the guest (PHASE15-HOST.md 4.3/4.4). Both records are
# read, and the PC is upgraded only when both name the release.
#
# **KNOWN GAP, 2026-09-18, and this script does not paper over it.** The host
# does NOT carry the guest forward on its own after install.ps1 restarts it.
# `crucible/host/app.py:1175` hands the install sequence to the door and
# nothing in `main()` ever calls it, so it runs only on a `POST /install`; and
# on a machine the guest already owns — every upgrade — `app.py:1221` takes the
# `Owner.WSL_UNIT` branch and calls `walk._complete()` (`app.py:1231`), which
# `installer.py:404` implements as "emit `done` describing the engine that is
# already there". `_guest_install`, the one place `install.sh --release` runs
# inside the distro, is `installer.py:607` and is reached only from `run()`
# (`installer.py:381`). So a deploy today upgrades the host and leaves the
# guest where it was; this script WAITS for the guest record and then reports
# the PC by name with both halves in the line. Poking the door from here would
# be a second driver of the guest, which is the thing that was just removed.
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
# THE MACHINES ARE INSTALLED AT THE SAME TIME, each in its own subshell, with
# every line named for the machine it came from. They share nothing — one is a
# box and the other is at the end of an ssh — so a queue only ever added their
# times together, and carried up to a minute of `await_release` polling behind
# each one. One machine failing does not stop the other, and the summary names
# it. See "the work" below.
#
# THIS RESTARTS SERVICES. The Windows tray host, the WSL engine it owns and the
# Mac launchd agent all go down and come back — now at once rather than in turn.
# It asks before it does that unless --yes is passed, and it asks ONCE, before
# the fan-out: the subshells share one stdin, so a prompt inside one of them
# would be answered for the others by whichever read first.

set -euo pipefail

REPO_SLUG="telltaleatheist/crucible"

# THE FLEET. Two machines, which is a fact about this deployment and not about
# Crucible — a different operator has a different list, and there is nothing
# secret here: `mac` is an ssh alias from ~/.ssh/config, and `Ubuntu` is the WSL
# distro name.
#
# `pc` USED TO BE TWO ENTRIES, `wsl` and `windows`, and that was the bug. Owen,
# 2026-09-18: *"windows is the driver; the thing moving wsl forward. use the
# established, installed, functional system to drive the new one."* Since
# PHASE15-HOST.md 4.4 install.ps1 installs `crucible host` and STOPS, and 4.3
# puts the whole WSL sequence behind the host's own door — "ONE implementation
# of the sequence, the host's; the bootstrap is its client". A `wsl` entry here
# that curled install.sh into the guest was a SECOND driver of that guest,
# racing the one the design names.
#
# So the PC is one entry and one install: install.ps1 at the tag. Its VERDICT
# still reads both records, because the machine has two — the host's in
# %LOCALAPPDATA% and the guest's in the distro — and the PC is done only when
# both name the release. That is what caught the drift this script was written
# for (0.6.3 beside 0.6.5) and it catches it from one entry just as well.
#
# `mac` is at the end of an ssh, has no host, and shares nothing with the PC,
# so the two run at the same time.
FLEET="pc mac"

# The distro whose record is the PC's second half. `install.ps1` does not take
# it; the host knows its own.
DISTRO="Ubuntu"

fail() { echo "deploy: $*" >&2; exit 1; }

release=""
only=""
assume_yes=0
# A machine whose record already NAMES the release is skipped, because the
# record is the whole point of reading it. But a record is written partway
# through an install, so a run that DIED after writing it leaves a machine
# that claims the version and never finished: on 2026-09-17 a truncated
# `curl | sh` stopped after local-register, and the retry then skipped the
# machine as already done. --force installs anyway.
force=0

while [ $# -gt 0 ]; do
  case "$1" in
    --release) [ $# -ge 2 ] || fail "--release needs a version"; release="$2"; shift 2 ;;
    --only)    [ $# -ge 2 ] || fail "--only needs a comma-separated list"; only="$2"; shift 2 ;;
    --yes|-y)  assume_yes=1; shift ;;
    --force)   force=1; shift ;;
    -h|--help) sed -n '2,63p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# EVERY NAME IN --only IS A MACHINE, checked here rather than silently matching
# nothing. `wsl` and `windows` were two of the three names this script took
# until 2026-09-18 and both are now one machine, so they are refused BY NAME
# with what replaced them — a typo that installs nothing and reports success is
# the failure this whole script exists to end.
for name in $(echo "$only" | tr ',' ' '); do
  case " $FLEET " in *" $name "*) continue ;; esac
  case "$name" in
    wsl)     fail "there is no 'wsl' machine any more: the host drives the guest (PHASE15-HOST.md 4.3), so the PC is one entry. Use --only pc" ;;
    windows) fail "'windows' and 'wsl' are one machine now, called pc. Use --only pc" ;;
    *)       fail "--only names '$name', which is not one of: $FLEET" ;;
  esac
done

# ------------------------------------------------------- what each machine runs
#
# Each `read_<machine>` prints the installed release, or the word `none` when
# the machine has no installation.json, or `unreachable` when it cannot be
# asked. The three are told apart because they mean different things: `none` is
# a machine to install onto, `unreachable` is a machine whose state is UNKNOWN
# and which is therefore never reported as up to date.
#
# A machine's reader may consult more than one record — the PC has two — and
# whatever it prints is compared to the release as a single string, so a
# machine that is only half upgraded prints something that cannot equal it.

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

# THE PC'S TWO RECORDS. The host's, written on Windows, and the guest's,
# written inside the distro. They are separate files because they are separate
# installs of separate things; they are read together because one machine is
# not upgraded until both of them say so.
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

# ONE ANSWER FROM TWO RECORDS, and the answer only collapses to a bare version
# when they AGREE. Anything else prints `host:<a> guest:<b>`, which can never
# equal the release asked for — so `await_release` keeps waiting, the summary
# reports the machine by name, and the halves it is stuck between are in the
# line. A PC whose guest never moves reads `pc(reports:host:0.6.9 guest:0.6.8)`,
# which says what is wrong without anybody having to go and look.
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

# --------------------------------------------------------------- the installers

install_sh_url() { echo "https://github.com/$REPO_SLUG/releases/download/v$1/install.sh"; }
install_ps1_url() { echo "https://github.com/$REPO_SLUG/releases/download/v$1/install.ps1"; }

# FETCH TO A FILE, THEN RUN IT - never `curl | sh`.
#
# A pipe throws curl's exit status away: the remote `sh` reads whatever
# arrived and its own status is all `set -e` can see, so a transfer cut in
# half is an installer that runs half. Measured 2026-09-17 installing 0.6.11
# into WSL: `sh: 352: Syntax error: Unterminated quoted string`, from a
# published install.sh that is byte-identical to the repo's and passes
# `sh -n`. It had already written installation.json with the new version, so
# the machine then LOOKED upgraded while linger, capability-write and
# local-start had never run.
#
# Written to a file, curl's failure is the command's failure. `sh -n` after
# it is the second half: a truncation that still parses would otherwise run.
# install_pc has fetched to a file all along, for its own reason.
# The format string is SINGLE-quoted so `$(mktemp)` and `$f` reach the far
# machine as text. Double-quoted, bash ran mktemp HERE and expanded $f to
# nothing, and the payload came out as `curl -o ""` - caught by printing it
# before trusting it, which is the only reason this note is not a defect.
# ONE MORE SHELL PARSE ON THE MAC THAN THERE WAS IN WSL. `ssh mac '"$SHELL"
# -lc <payload>'` is a STRING the remote shell parses before $SHELL ever sees
# it, so a payload containing double quotes (and the one below must, for $f)
# ends that string early. Measured while building this against the WSL caller
# that no longer exists: it took the payload and the Mac answered `no such
# file or directory`.
shquote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

remote_install_payload() {
  printf 'set -e; f=$(mktemp); curl -fsSL --retry 3 -o "$f" %s; sh -n "$f"; sh "$f" --release %s; rm -f "$f"' \
    "'$1'" "'$2'"
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
    "\"\$SHELL\" -lc $(shquote "$(remote_install_payload "$(install_sh_url "$1")" "$1")")"
}

# THE PC'S ONE INSTALL. install.ps1 unpacks the host pack, writes the Startup
# shortcut, starts `crucible host` and stops (PHASE15-HOST.md 4.4); the guest is
# the host's to carry from there (4.3). Nothing here touches the distro — a
# second driver of it is exactly what was removed on 2026-09-18.
install_pc() {
  # `irm | iex` cannot take a parameter, so the script is fetched to a file
  # first — the same reason its own header gives for -Uninstall.
  local script="${TMPDIR:-/tmp}/crucible-install-$1.ps1" status=0
  curl -fsSL -o "$script" "$(install_ps1_url "$1")"
  # THE INSTALLER'S STATUS, NOT THE CLEANUP'S. `rm -f` was the last command in
  # this function, so it WAS this function's exit status — and `set -e` is
  # disabled inside a function called as an `if` condition, which is how this
  # is called. A refused install.ps1 therefore returned 0, and the only thing
  # that noticed was the after-check a minute of polling later, reporting a
  # machine that "still reports" the old version rather than one whose
  # installer said no. Caught by tests/test_deploy_parallel.py, 2026-09-18.
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$script" 2>/dev/null || echo "$script")" -Release "$1" || status=$?
  # A cleanup failure is not an install failure. It is not silent either.
  rm -f "$script" || echo "deploy: could not remove $script" >&2
  return "$status"
}

# THE RECORD IS NOT ALWAYS WRITTEN BY THE INSTALLER. `installation.json` is
# published by `crucible local publish` when the runtime STARTS, so on Windows
# especially — where install.ps1 unpacks the host pack, launches the host and
# returns — the file can still hold the old release for a moment after the
# installer has exited successfully. Reading it once, immediately, is reading a
# race. This waits for the value to become the one asked for, and gives up with
# whatever it last saw so the caller reports the truth rather than a timeout.
#
# TWO SECONDS, THIRTY TIMES — the same sixty-second ceiling that three seconds
# twenty times was, asked often enough that a record published one second after
# the installer returns is not read back two seconds later. With the fleet
# installing at once this poll is the tail of the whole run rather than a third
# of it, so its granularity is the last thing standing between an install
# finishing and deploy saying so.
# THE CEILING IS THE PC's, SIZED FROM THE HOST's OWN CONSTANTS, not a feeling.
# On the PC the record this waits for is the GUEST's, and the guest is carried
# by the host AFTER install.ps1 has returned: a presence settle of up to
# PRESENCE_SETTLE_CEILING_SECONDS (195 s when a recovery recipe runs first;
# 7 s measured on 1.0.3), then install.sh in the guest (23-33 s measured on the
# Mac for the same script), then the engine restart and publish (~30 s). The
# 60 s this had until 1.0.4 held 12 s of margin on the happy path and could
# not report the recovery path at all, and a low ceiling costs a WRONG verdict
# on a correct install, while a high one costs nothing on a machine that
# succeeded because this returns the instant the record matches. 258 s
# rounded up: 150 polls of 2 s.
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
  # `|| answer=""` because `set -e` would otherwise make a closed stdin an
  # unexplained exit 1 instead of the refusal it actually is. That is not
  # hypothetical: every reader above used to eat the answer (ssh reads stdin
  # unless told not to), and the prompt exited silently with nothing typed.
  read -r answer || answer=""
  case "$answer" in y|Y|yes|YES) ;; *) echo "deploy: nothing was changed"; exit 1 ;; esac
fi

# ------------------------------------------------------------------- the work
#
# BOTH AT ONCE. The two machines share nothing — the PC is one box and `mac` is
# at the end of an ssh — so installing them one after another only ever added
# their times together, and each carried up to sixty seconds of `await_release`
# polling behind it. Measured 2026-09-18: a release whose actual installing was
# about ninety seconds spent several minutes of wall-clock in that queue.
#
# Each machine therefore gets its own subshell, and its verdict comes back
# through a FILE rather than through an exit status: a background job's status
# says that A job failed and cannot say which machine it was, and every line of
# the summary below names a machine.
#
# EVERY LINE IS PREFIXED with the machine it came from, because three installers
# writing to one terminal at the same moment is unreadable otherwise — and that
# includes the installers' own progress lines, which are most of what they
# print. stderr is merged into stdout for the same reason: two streams through
# two prefixers interleave by buffer rather than by line, and half a line with
# somebody else's name on it is worse than a line on the wrong stream. The
# failure summary at the end is still stderr.
#
# `sed` prefixes at newlines, so a progress bar that repaints with a bare
# carriage return would arrive as one enormous unprefixed line. None of these
# three installers is run with a terminal on either side — `wsl.exe --exec`,
# `ssh -n` and `powershell.exe -File` — which is the condition under which pip
# and curl print progress as whole lines instead of repainting.

work="$(mktemp -d)"
# A cleanup failure is not a deploy failure. It is not silent either.
trap 'rm -rf "$work" 2>/dev/null || echo "deploy: could not remove $work" >&2' EXIT

# One machine, whole: install it, wait for its record, and leave two files —
# `<machine>.seconds` always, `<machine>.why` only when something went wrong.
# An absent `.why` beside a present `.seconds` is the success signal, and an
# absent `.seconds` means this function did not finish, which is reported.
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
      # NOT a skip. A machine that could not be asked is a machine whose state
      # is unknown, and the summary must say so rather than imply it is fine.
      failed="$failed $machine(unreachable)"; continue ;;
  esac

  running="$running $machine"
  echo
  echo "deploy: $machine  ${BEFORE[$machine]} -> $release"
  deploy_one "$machine" "$release" 2>&1 | sed "s/^/$machine: /" &
done

# No arguments: every machine, however long the slowest takes. The verdicts are
# read out of $work below rather than from this status, which cannot name one.
wait

echo
for machine in $running; do
  if [ ! -f "$work/$machine.seconds" ]; then
    # Its subshell was killed, or died before it could record anything. That is
    # a machine in an unknown state, which is never reported as done.
    failed="$failed $machine(no result: its subshell left no record)"
    continue
  fi
  # READ BY `ship.sh`, which folds these into its own table. deploy.sh is what
  # knows: it forked the installs and it joined them, and a caller timing the
  # whole call would only ever learn the slowest machine.
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
