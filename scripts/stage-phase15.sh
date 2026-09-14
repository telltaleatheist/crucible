#!/usr/bin/env bash
#
# PHASE15-HOST.md section 8, the STAGING half: S1 and S2.
#
# Everything the test run needs that costs network and disk but NOT the card.
# Owen holds the card for a fine-tune until he says otherwise, so this script
# is written so that nothing in it can touch one:
#
#   * S2 never starts a server and never loads a model. It builds the Windows
#     host pack, unpacks it, initialises a TEMPORARY `CRUCIBLE_HOME` and pulls
#     three subjects into it. A pull is network and disk.
#   * S1 restarts the WSL service so it runs the branch HEAD and reads
#     `/v1/info` back. A restart is not a job.
#
# WHAT IT WILL NOT DO, by construction and not by care:
#
#   * touch port 7100. The staged Windows home is initialised on **7101**,
#     which is the host's door port and is free while no host runs; 7100 is
#     the live WSL server's and stays its.
#   * write the Startup shortcut, add a portproxy, or install anything
#     system-wide. Everything lands under C:\tmp\phase15-testrun\.
#   * submit a job anywhere.
#
# Usage, from Git Bash on the PC:
#
#     scripts/stage-phase15.sh s2          # the Windows side (the long pole)
#     scripts/stage-phase15.sh s1          # the WSL server, after the commits
#     scripts/stage-phase15.sh both
#     scripts/stage-phase15.sh s2 --dry-run
#
set -u -o pipefail

STAGE="${1:-both}"
DRY_RUN=0
for argument in "$@"; do
  if [ "$argument" = "--dry-run" ]; then DRY_RUN=1; fi
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="C:/tmp/phase15-testrun"
RUN_ROOT_UNIX="/c/tmp/phase15-testrun"
PACKS_DIR="${RUN_ROOT}/packs"
HOST_DIR="${RUN_ROOT}/host"
STAGED_HOME="${RUN_ROOT}/home"
STAGED_PORT=7101

WSL_DISTRO="Ubuntu"
WSL_PYTHON="/home/telltale/anaconda3/envs/crucible/bin/python"
WSL_PIP="/home/telltale/anaconda3/envs/crucible/bin/pip"
WSL_CHECKOUT="/home/telltale/crucible"
WSL_REMOTE="pc"

# The three subjects S2 pulls, in the order the button needs them.
SUBJECTS=(
  "engine llama-cpp"
  "model dots-ocr"
  "model qwen3.5-9b"
)

say() { printf '\n=== %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }

# Every command goes through this, so --dry-run prints exactly what would run
# and nothing has a second code path.
run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD RUN: %s\n' "$*"
    return 0
  fi
  "$@"
}

# The same, for a shell string (a pipeline, or a wsl.exe -c body).
run_shell() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD RUN: %s\n' "$1"
    return 0
  fi
  bash -c "$1"
}

fail() {
  printf '\nSTAGE FAILED: %s\n' "$*" >&2
  exit 1
}

seconds_since() {
  # Wall clock, printed per step, because the report wants figures and
  # "it took a while" is not one.
  printf '%s' "$(( $(date +%s) - $1 ))"
}

# --------------------------------------------------------------------- S2
#
# The Windows side, staged. Nothing here is installed for the machine: the
# pack is unpacked into a directory under C:\tmp and the server home is a
# second one, so removing C:\tmp\phase15-testrun removes the whole of it.

stage_s2() {
  say "S2 — the Windows side, staged under ${RUN_ROOT}"
  local started
  started="$(date +%s)"

  run mkdir -p "${RUN_ROOT_UNIX}" "${RUN_ROOT_UNIX}/packs" "${RUN_ROOT_UNIX}/host"

  say "S2.1 build the host pack (crucible envpack build host)"
  local t
  t="$(date +%s)"
  # WINDOWS' OWN tar FIRST, and `envpack` refuses by name without it: Git for
  # Windows ships GNU tar 1.32, which has no libzstd and would shell out to a
  # `zstd.exe` this machine does not have — at the END of the build, after the
  # interpreter download and the pip run. `C:\Windows\System32\tar.exe` is
  # bsdtar 3.8.1 with libzstd built in. Prepended for this command only.
  run_shell "cd '${REPO}' && PATH=/c/Windows/System32:\$PATH python -m crucible envpack build host --out '${PACKS_DIR}'" \
    || fail "S2.1 crucible envpack build host"
  note "S2.1 took $(seconds_since "$t")s"

  say "S2.2 unpack it into ${HOST_DIR}"
  t="$(date +%s)"
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD RUN: python -m crucible envpack install host --out %s --target %s\n' \
      "${PACKS_DIR}" "${HOST_DIR}"
  else
    # The pack is a multi-part .tar.zst beside an envpacks.json. Unpacked with
    # python's own tarfile through zstandard, which is what `envpack` already
    # depends on — no 7-zip, no tar on PATH, nothing this machine might not
    # have.
    python - "$PACKS_DIR" "$HOST_DIR" <<'PYEOF' || fail "S2.2 unpack"
import json
import pathlib
import sys
import tarfile

import zstandard

packs = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
manifest = json.loads((packs / "envpacks.json").read_text(encoding="utf-8"))
entries = [entry for entry in manifest["packs"] if entry["name"] == "host"]
if not entries:
    raise SystemExit(f"no host pack in {packs / 'envpacks.json'}")
entry = entries[0]
parts = [packs / name for name in entry["parts"]]
for part in parts:
    if not part.is_file():
        raise SystemExit(f"{part} is not there")
target.mkdir(parents=True, exist_ok=True)
print(f"unpacking {len(parts)} part(s) into {target}")


class Joined:
    """The parts, read end to end as one stream."""

    def __init__(self, paths):
        self._paths = list(paths)
        self._handle = None
        self._index = 0

    def read(self, size=-1):
        out = b""
        while True:
            if self._handle is None:
                if self._index >= len(self._paths):
                    return out
                self._handle = self._paths[self._index].open("rb")
                self._index += 1
            block = self._handle.read(size if size > 0 else -1)
            if block:
                out += block
                if size > 0 and len(out) >= size:
                    return out
                continue
            self._handle.close()
            self._handle = None
            if size <= 0:
                continue


stream = zstandard.ZstdDecompressor().stream_reader(Joined(parts))
with tarfile.open(fileobj=stream, mode="r|") as bundle:
    bundle.extractall(target)
print("unpacked")
PYEOF
  fi
  note "S2.2 took $(seconds_since "$t")s"

  local console="${HOST_DIR}/host/crucible.cmd"
  if [ "$DRY_RUN" = "0" ] && [ ! -f "${RUN_ROOT_UNIX}/host/host/crucible.cmd" ]; then
    # The pack unpacks a `host/` directory; if a future pack changes that,
    # this says so rather than running a path that is not there.
    console="$(find "${RUN_ROOT_UNIX}/host" -name crucible.cmd -print -quit)"
    [ -n "${console}" ] || fail "S2.2 no crucible.cmd under ${HOST_DIR}"
    console="$(cygpath -w "${console}" 2>/dev/null || printf '%s' "${console}")"
  fi
  note "console: ${console}"

  say "S2.3 initialise a temporary home on port ${STAGED_PORT}"
  t="$(date +%s)"
  run_shell "CRUCIBLE_HOME='${STAGED_HOME}' '${console}' init --backend llama-windows --port ${STAGED_PORT} --enable-echo --enable-llm --force" \
    || fail "S2.3 crucible init"
  note "S2.3 took $(seconds_since "$t")s"

  say "S2.4 pull the engine and the two GGUF subjects (network and disk only)"
  for subject in "${SUBJECTS[@]}"; do
    # shellcheck disable=SC2086
    set -- $subject
    local kind="$1" id="$2"
    t="$(date +%s)"
    note "pulling ${kind} ${id}"
    if [ "${kind}" = "engine" ]; then
      run_shell "CRUCIBLE_HOME='${STAGED_HOME}' '${console}' install llm --verbose" \
        || fail "S2.4 install llm (the engine subject)"
    else
      run_shell "CRUCIBLE_HOME='${STAGED_HOME}' '${console}' models pull ${id}" \
        || fail "S2.4 models pull ${id}"
    fi
    note "${kind} ${id} took $(seconds_since "$t")s"
  done

  say "S2.5 what is on disk"
  run_shell "CRUCIBLE_HOME='${STAGED_HOME}' '${console}' doctor || true"
  if [ "$DRY_RUN" = "0" ]; then
    du -sh "${RUN_ROOT_UNIX}/home" 2>/dev/null || true
    find "${RUN_ROOT_UNIX}/home" -type f -size +10M -printf '%10s  %p\n' 2>/dev/null | sort -rn || true
  fi
  note "S2 took $(seconds_since "$started")s in total"
}

# --------------------------------------------------------------------- S1
#
# The live WSL clone, fast-forwarded to this branch's HEAD and restarted.
# `pc` is the remote it already has for this checkout.

stage_s1() {
  say "S1 — the WSL server runs this branch's HEAD"
  local started
  started="$(date +%s)"
  local head
  head="$(git -C "${REPO}" rev-parse HEAD)"
  local branch
  branch="$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
  note "HEAD ${head} on ${branch}"

  say "S1.1 fetch and fast-forward ${WSL_CHECKOUT}"
  run wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
    "cd ${WSL_CHECKOUT} && git fetch ${WSL_REMOTE} && git checkout ${branch} 2>/dev/null || git checkout -b ${branch} ${WSL_REMOTE}/${branch}" \
    || fail "S1.1 git checkout in the guest"
  run wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
    "cd ${WSL_CHECKOUT} && git merge --ff-only ${head} && git rev-parse HEAD" \
    || fail "S1.1 git merge --ff-only ${head}"

  say "S1.2 pip install -e . in the crucible env"
  run wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
    "cd ${WSL_CHECKOUT} && ${WSL_PIP} install -e . --no-deps --quiet && ${WSL_PYTHON} -c 'import crucible; print(crucible.VERSION)'" \
    || fail "S1.2 pip install -e ."

  say "S1.3 restart the service"
  # The user bus first, because that is what the unit is; the root recipe is
  # the one found on 2026-09-14 for a session with no bus, named rather than
  # tried silently (PHASE15-HOST.md 4.1).
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD RUN: systemctl --user restart crucible (in %s), else systemctl restart user@1000 as root\n' "${WSL_DISTRO}"
  else
    if ! wsl.exe -d "${WSL_DISTRO}" --exec bash -lc "systemctl --user restart crucible"; then
      note "the user bus did not take it; trying the root recipe (systemctl restart user@1000)"
      wsl.exe -d "${WSL_DISTRO}" -u root --exec bash -lc "systemctl restart user@1000" \
        || fail "S1.3 neither recipe restarted the unit"
      wsl.exe -d "${WSL_DISTRO}" --exec bash -lc "systemctl --user restart crucible" \
        || fail "S1.3 systemctl --user restart crucible after the bus came back"
    fi
  fi

  say "S1.4 /v1/info answers the new build"
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD RUN: curl -s 127.0.0.1:7100/v1/info (with the bearer from config.toml)\n'
  else
    local token
    token="$(wsl.exe -d "${WSL_DISTRO}" --exec bash -lc "grep -m1 '^token' ~/.crucible/config.toml | cut -d'\"' -f2" | tr -d '\r')"
    [ -n "${token}" ] || fail "S1.4 no token in the guest's config.toml"
    local tries=0
    while [ "${tries}" -lt 30 ]; do
      if wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
        "curl -sS -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' http://127.0.0.1:7100/v1/info" >/dev/null 2>&1; then
        break
      fi
      tries=$(( tries + 1 ))
      sleep 2
    done
    wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
      "curl -sS -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' http://127.0.0.1:7100/v1/info" \
      || fail "S1.4 /v1/info did not answer"
    printf '\n'
    wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
      "curl -sS -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' http://127.0.0.1:7100/v1/capability" \
      || fail "S1.4 /v1/capability did not answer"
    printf '\n'
  fi
  note "S1 took $(seconds_since "$started")s"
}

case "${STAGE}" in
  s1) stage_s1 ;;
  s2) stage_s2 ;;
  both) stage_s2; stage_s1 ;;
  *) fail "usage: stage-phase15.sh [s1|s2|both] [--dry-run]" ;;
esac

say "staging done"
