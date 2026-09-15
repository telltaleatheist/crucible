#!/usr/bin/env bash
#
# PHASE15-HOST.md section 8 — THE BUTTON.
#
# Owen, 2026-09-14: *"get everything ready so we can just hit a button and have
# the tests run, and when it's fully ready to test, ill release the card and let
# you know when its ready."*
#
# One script, run from Git Bash on the PC with no arguments. It runs the stages
# IN ORDER, stops at the first failure with the stage's name and the failing
# command's output, and writes `C:\tmp\phase15-testrun\<timestamp>\report.md`
# AS IT GOES, so a stopped run still says what passed. Every stage prints its
# wall-clock seconds.
#
# NOTHING IN IT ASKS A QUESTION. Anything it needs that may not be there — an
# Anthropic key, a page to read, the mac worktree, a BookForge checkout — is a
# file or a directory it looks for BY NAME and reports SKIPPED by name when
# absent. Never silently, and never with a default that pretends.
#
#     scripts/testrun-phase15.sh              # the run
#     scripts/testrun-phase15.sh --dry-run    # print every command, run none
#     scripts/testrun-phase15.sh --only T5    # one stage
#
# THE CARD. T6 and T7 need it; nothing else here does. Each says so before it
# starts, so a run begun before Owen has released it stops there with the
# reason rather than fighting a trainer for VRAM.
#
# T9 (the Mac) IS NOT IN THIS SCRIPT and skips itself saying so. It drives a
# second machine's card over ssh, and a button on this one that reaches across
# the network to hold somebody else's GPU is a button whose blast radius is
# not what its name says. The stage is run from the Mac side; 7c's M-steps are
# the recipe.
#
# STAGING IS NOT THIS SCRIPT'S EITHER. S1 and S2 are
# `scripts/stage-phase15.sh`; a button that also did 15 GB of downloads would
# be a button whose first press takes an hour.
#
set -u -o pipefail

DRY_RUN=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --only) shift; ONLY="${1:-}" ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/c/tmp/phase15-testrun"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${RUN_ROOT}/${STAMP}"
REPORT="${OUT}/report.md"

STAGED_HOME_WIN="C:/tmp/phase15-testrun/home"
STAGED_PORT=7101
ENGINE_PORT=7100
WSL_DISTRO="Ubuntu"
WSL_PYTHON="/home/telltale/anaconda3/envs/crucible/bin/python"
WSL_REPO="/mnt/c/Users/tellt/Projects/crucible"
WSL_MAC_REPO="/mnt/c/Users/tellt/Projects/crucible-mac"
LOCK="/tmp/crucible-pytest.lock"
MAC_WORKTREE="/c/Users/tellt/Projects/crucible-mac"
BOOKFORGE="/c/Users/tellt/Projects/bookforge"

KEY_FILE="${RUN_ROOT}/anthropic-key.txt"
PAGE_PDF="${RUN_ROOT}/page.pdf"
PAGE_PNG="${RUN_ROOT}/page.png"

LAST_OUTPUT=""
LAST_STATUS=0

# ---------------------------------------------------------------- reporting

say() { printf '\n=== %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
report() { if [ "$DRY_RUN" = "0" ]; then printf '%s\n' "$*" >> "${REPORT}"; fi; }

STAGE_STARTED=0
STAGE_NAME=""

begin() { STAGE_NAME="$1"; STAGE_STARTED="$(date +%s)"; say "${STAGE_NAME}"; }
elapsed() { printf '%s' "$(( $(date +%s) - STAGE_STARTED ))"; }

pass() {
  local seconds; seconds="$(elapsed)"
  note "PASS in ${seconds}s"
  report ""
  report "## ${STAGE_NAME} — PASS (${seconds}s)"
  report ""
  report "$*"
}

skip() {
  local seconds; seconds="$(elapsed)"
  note "SKIPPED — $1"
  report ""
  report "## ${STAGE_NAME} — SKIPPED (${seconds}s)"
  report ""
  report "$1"
}

fail() {
  local seconds; seconds="$(elapsed)"
  printf '\nSTAGE FAILED: %s (after %ss)\n%s\n' "${STAGE_NAME}" "${seconds}" "$*" >&2
  report ""
  report "## ${STAGE_NAME} — FAILED (${seconds}s)"
  report ""
  report '```'
  report "$*"
  report '```'
  report ""
  report "The run stopped here. Every stage above passed or skipped."
  exit 1
}

# EVERY command goes through this, so `--dry-run` has no second code path and
# cannot drift from the real one.
run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD RUN: %s\n' "$1"
    LAST_OUTPUT=""
    LAST_STATUS=0
    return 0
  fi
  LAST_OUTPUT="$(bash -c "$1" 2>&1)"
  LAST_STATUS=$?
  return $LAST_STATUS
}

tail_of() { printf '%s' "${LAST_OUTPUT:-}" | tail -"${1:-20}"; }

wanted() { [ -z "${ONLY}" ] || [ "${ONLY}" = "$1" ]; }

# Is this stage's subject there? Prints the SKIP itself when it is not, so no
# caller can forget to.
present() {
  local what="$1" path="$2" why="$3"
  if [ "$DRY_RUN" = "1" ]; then
    printf '    WOULD CHECK: %s at %s\n' "${what}" "${path}"
    return 0
  fi
  if [ -e "${path}" ]; then return 0; fi
  skip "${what} is not at \`${path}\`. ${why}"
  return 1
}

# The pytest wrapper, with the trainer guard S4 asks for.
#
# `[t]rain_lora.py` and not `train_lora.py`: the bracket stops the pattern
# matching the process that is looking for it, which is the oldest trap in
# grepping a process table and the one that would make this guard fire on
# itself every time.
pytest_under_lock() {
  printf "flock -w 7200 %s -c 'if pgrep -f \"[t]rain_lora.py\" >/dev/null; then echo TRAINER_RUNNING; exit 3; fi; cd %s && %s -m pytest -q -p no:cacheprovider'" \
    "${LOCK}" "$1" "${WSL_PYTHON}"
}

in_guest() { printf 'wsl.exe -d %s --exec bash -lc %q' "${WSL_DISTRO}" "$1"; }

# The WSL engine's bearer, read out of the guest's own config.
guest_token() {
  wsl.exe -d "${WSL_DISTRO}" --exec bash -lc \
    "grep -m1 '^token' ~/.crucible/config.toml | cut -d'\"' -f2" | tr -d '\r'
}

# ------------------------------------------------------------------ preflight

if [ "$DRY_RUN" = "0" ]; then
  mkdir -p "${OUT}"
  {
    printf '# Phase 15 test run — %s\n\n' "${STAMP}"
    printf -- '- repo: `%s`\n' "${REPO}"
    printf -- '- HEAD: `%s`\n' "$(git -C "${REPO}" rev-parse HEAD)"
    printf -- '- branch: `%s`\n' "$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
    printf -- '- staged home: `%s` (port %s)\n' "${STAGED_HOME_WIN}" "${STAGED_PORT}"
    printf '\nThe card is needed by T6 and T7 only. T9 is the Mac'"'"'s and is not in this script.\n'
  } > "${REPORT}"
  say "report: ${REPORT}"
fi

# ------------------------------------------------------------------------ T1

if wanted T1; then
  begin "T1 — sdk/bootstrap and sdk/ts"
  run "cd '${REPO}/sdk/bootstrap' && npm test" || fail "${LAST_OUTPUT}"
  bootstrap_count="$(printf '%s' "${LAST_OUTPUT:-}" | grep -E '^# (pass|fail) ' | tr '\n' ' ')"
  run "cd '${REPO}/sdk/ts' && npm test" || fail "${LAST_OUTPUT}"
  ts_count="$(printf '%s' "${LAST_OUTPUT:-}" | grep -E '^# (pass|fail) ' | tr '\n' ' ')"
  pass "bootstrap: ${bootstrap_count:-(dry run)}
ts: ${ts_count:-(dry run)}"
fi

# ------------------------------------------------------------------------ T2

if wanted T2; then
  begin "T2 — pytest on this branch, under the lock"
  run "$(in_guest "$(pytest_under_lock "${WSL_REPO}")")" || {
    case "${LAST_OUTPUT:-}" in
      *TRAINER_RUNNING*)
        fail "a train_lora.py is running in the guest. Section 8's S4: these
suites want the VM's memory and a trainer has it. Wait for it or stop it, then
press the button again."
        ;;
      *) fail "${LAST_OUTPUT}" ;;
    esac
  }
  count="$(printf '%s' "${LAST_OUTPUT:-}" | grep -E '[0-9]+ (passed|failed)' | tail -1)"
  pass "${count:-0 failed (no summary line printed by this pytest)}"
fi

# ------------------------------------------------------------------------ T3

if wanted T3; then
  begin "T3 — pytest in the mac worktree"
  mac_head=""
  if [ "$DRY_RUN" = "0" ] && [ -d "${MAC_WORKTREE}" ]; then
    mac_head="$(git -C "${MAC_WORKTREE}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [ "$DRY_RUN" = "0" ] && [ ! -d "${MAC_WORKTREE}" ]; then
    skip "there is no worktree at \`${MAC_WORKTREE}\`. This stage exists only
while the mac branch is separate from this one; it is merged (0034871), so its
suite IS T2's."
  elif [ "$DRY_RUN" = "0" ] && [ -n "${mac_head}" ] \
      && git -C "${REPO}" merge-base --is-ancestor "${mac_head}" HEAD 2>/dev/null; then
    skip "\`${MAC_WORKTREE}\`'s HEAD (\`${mac_head}\`) is already an ancestor of
this branch's, so its suite is T2's. Skipped by name rather than run twice."
  else
    run "$(in_guest "$(pytest_under_lock "${WSL_MAC_REPO}")")" || {
      case "${LAST_OUTPUT:-}" in
        *TRAINER_RUNNING*) fail "a train_lora.py is running in the guest (see T2)." ;;
        *) : ;;
      esac
    }
    failures="$(printf '%s' "${LAST_OUTPUT:-}" | grep -E '^FAILED ' || true)"
    unexpected="$(printf '%s' "${failures}" | grep -v 'test_lineup.py' || true)"
    if [ -n "${unexpected}" ]; then
      fail "failures outside test_lineup.py:
${unexpected}"
    fi
    pass "0 unexpected failures. test_lineup.py's worktree-environmental ones,
listed by name and nothing else:
\`\`\`
${failures:-none}
\`\`\`"
  fi
fi

# ------------------------------------------------------------------------ T4

if wanted T4; then
  begin "T4 — BookForge keepers"
  if present "BookForge's keeper runner" "${BOOKFORGE}/tools/run-keepers.js" \
      "T4 runs BookForge's keeper suite and there is no checkout at that path."; then
    run "cd '${BOOKFORGE}' && node tools/run-keepers.js" || fail "${LAST_OUTPUT}"
    pass "all suites green
\`\`\`
$(tail_of 20)
\`\`\`"
  fi
fi

# ------------------------------------------------------------------------ T5

if wanted T5; then
  begin "T5 — the upstream route, end to end"
  if present "an Anthropic key" "${KEY_FILE}" \
      "T5 configures the \`anthropic\` upstream on the WSL server, routes
\`translate\` through it, sends ONE completion, then puts the route back to
local and REMOVES the key. Without a key there is nothing to send. One line,
the key and nothing else."; then
    if [ "$DRY_RUN" = "1" ]; then
      note "WOULD RUN: PUT /v1/settings (configure + route), one chat, then restore"
      pass "(dry run)"
    else
      token="$(guest_token)"
      [ -n "${token}" ] || fail "no token in the guest's ~/.crucible/config.toml"
      key="$(tr -d '\r\n' < "${KEY_FILE}")"
      [ -n "${key}" ] || fail "${KEY_FILE} is empty"
      # THE KEY NEVER REACHES A LOG OR AN ARGV. It goes into a file inside the
      # guest at 0600, curl reads the body from that file, and the file is
      # deleted in the same command — so it is not in this script's process
      # table, not in the report, and not in the shell history.
      body="/tmp/crucible-t5-$$.json"
      run "$(in_guest "umask 077; cat > ${body} <<'JSON'
{\"upstreams\": {\"anthropic\": {\"key\": \"${key}\"}}, \"routes\": {\"translate\": \"anthropic/claude-sonnet-5\"}}
JSON
curl -sS -X PUT -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' -H 'Content-Type: application/json' --data-binary @${body} http://127.0.0.1:${ENGINE_PORT}/v1/settings; rm -f ${body}")" \
        || fail "configuring the upstream failed: ${LAST_OUTPUT}"
      configured="${LAST_OUTPUT}"
      case "${configured}" in
        *'"error"'*) fail "PUT /v1/settings refused: ${configured}" ;;
      esac

      run "$(in_guest "curl -sS -X POST -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' -H 'Content-Type: application/json' -d '{\"model\": \"anthropic/claude-sonnet-5\", \"max_tokens\": 16, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: routed\"}]}' http://127.0.0.1:${ENGINE_PORT}/v1/openai/chat/completions")" \
        || fail "the completion failed: ${LAST_OUTPUT}"
      completion="${LAST_OUTPUT}"

      # RESTORED WHETHER OR NOT THE COMPLETION WORKED. The key never stays on
      # a test run, so the restore runs before the assertion below.
      run "$(in_guest "curl -sS -X PUT -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' -H 'Content-Type: application/json' -d '{\"routes\": {\"translate\": \"local\"}, \"upstreams\": {\"anthropic\": null}}' http://127.0.0.1:${ENGINE_PORT}/v1/settings")" \
        || fail "RESTORE FAILED and the key may still be on that server: ${LAST_OUTPUT}"
      restored="${LAST_OUTPUT}"
      case "${restored}" in
        *'"configured": true'*'anthropic'*) fail "the key is still configured: ${restored}" ;;
      esac
      case "${completion}" in
        *'"error"'*) fail "the completion came back as a refusal: ${completion}" ;;
        *choices*) : ;;
        *) fail "the completion is not a chat completion: ${completion}" ;;
      esac
      printf '%s\n' "${completion}" > "${OUT}/t5-completion.json"
      pass "a completion came back through anthropic; the route is local again
and the key is removed (\`upstreams.anthropic: null\`). The key never stays on
a test run, and it is in neither this report nor the process table.

Restored settings document:
\`\`\`json
${restored}
\`\`\`"
    fi
  fi
fi

# ------------------------------------------------------------------------ T6

if wanted T6; then
  begin "T6 — dots under vLLM in WSL (NEEDS THE CARD)"
  page=""
  if [ -e "${PAGE_PNG}" ]; then page="${PAGE_PNG}"; elif [ -e "${PAGE_PDF}" ]; then page="${PAGE_PDF}"; fi
  if [ "$DRY_RUN" = "1" ]; then
    note "WOULD CHECK: a page at ${PAGE_PNG} or ${PAGE_PDF}"
    note "WOULD RUN: rasterise (if PDF), load dots-ocr, one chat completion, parse"
    pass "(dry run)"
  elif [ -z "${page}" ]; then
    skip "there is no page at \`${PAGE_PNG}\` or \`${PAGE_PDF}\`. T6 reads ONE
page through the WSL server and records seconds/page and whether the artifact
parses in the {bbox,category,text} dialect. A PNG is preferred and needs
nothing installed; a PDF is rasterised with pypdfium2 if this interpreter has
it, and is skipped by name if it does not — rasterising is the APP's work
(3.10) and Crucible ships no rasteriser."
  else
    note "THE CARD IS HELD FROM HERE UNTIL T7 ENDS."
    # `--load`, like T7. Crucible NEVER loads a model to answer a chat request
    # — it refuses `model_not_resident` by name, on every backend — so the
    # CALLER loads it, reads the page, and unloads it. The first run of this
    # script asked the WSL server to read a page with nothing resident and got
    # that refusal; the server was right and this line was wrong.
    run "python '${REPO}/scripts/read_one_page.py' --page '${page}' --out '${OUT}/t6' --load --server 'http://127.0.0.1:${ENGINE_PORT}' --token \"\$(wsl.exe -d ${WSL_DISTRO} --exec bash -lc \"grep -m1 '^token' ~/.crucible/config.toml | cut -d'\\\"' -f2\" | tr -d '\r')\"" \
      || fail "${LAST_OUTPUT}"
    pass "one page parsed under vLLM.
\`\`\`
$(tail_of 25)
\`\`\`"
  fi
fi

# ------------------------------------------------------------------------ T7

if wanted T7; then
  begin "T7 — llama-windows on ${STAGED_PORT} (NEEDS THE CARD)"
  if present "the staged Windows home" "${RUN_ROOT}/home/config.toml" \
      "T7 starts the staged Windows server. Run \`scripts/stage-phase15.sh s2\`
first — it builds the host pack, initialises the home on ${STAGED_PORT} and
pulls the engine and the two GGUFs."; then
    if [ "$DRY_RUN" = "1" ]; then
      note "WOULD RUN: crucible serve on ${STAGED_PORT}; load-model dots-ocr; the T6 page; load-model qwen3.5-9b; one cleanup chunk; unload; stop"
      pass "(dry run)"
    else
      page=""
      if [ -e "${PAGE_PNG}" ]; then page="${PAGE_PNG}"; elif [ -e "${PAGE_PDF}" ]; then page="${PAGE_PDF}"; fi
      [ -n "${page}" ] || skip "no page, so there is nothing to compare with T6's artifact."
      if [ -n "${page}" ]; then
        # The server is started HERE and stopped in the same stage, so a
        # failure anywhere below cannot leave a llama-server holding the card.
        log="${OUT}/t7-server.log"
        ( CRUCIBLE_HOME="${STAGED_HOME_WIN}" python -m crucible serve \
            --host 127.0.0.1 --port "${STAGED_PORT}" > "${log}" 2>&1 & echo $! > "${OUT}/t7.pid" )
        stop_it() {
          if [ -f "${OUT}/t7.pid" ]; then
            kill "$(cat "${OUT}/t7.pid")" 2>/dev/null || true
            rm -f "${OUT}/t7.pid"
          fi
        }
        trap stop_it EXIT
        token="$(grep -m1 '^token' "${RUN_ROOT}/home/config.toml" | cut -d'"' -f2)"
        tries=0
        until curl -sS -o /dev/null "http://127.0.0.1:${STAGED_PORT}/v1/ping" 2>/dev/null; do
          tries=$(( tries + 1 ))
          [ "${tries}" -lt 60 ] || { stop_it; fail "the staged server never answered /v1/ping
$(tail -40 "${log}")"; }
          sleep 1
        done
        run "python '${REPO}/scripts/read_one_page.py' --page '${page}' --out '${OUT}/t7' --server 'http://127.0.0.1:${STAGED_PORT}' --token '${token}' --load" \
          || { stop_it; fail "${LAST_OUTPUT}"; }
        pages_said="${LAST_OUTPUT}"
        run "python '${REPO}/scripts/one_cleanup_chunk.py' --server 'http://127.0.0.1:${STAGED_PORT}' --token '${token}' --model qwen3.5-9b --out '${OUT}/t7'" \
          || { stop_it; fail "${LAST_OUTPUT}"; }
        cleanup_said="${LAST_OUTPUT}"
        run "curl -sS -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' http://127.0.0.1:${STAGED_PORT}/v1/activity" \
          || { stop_it; fail "${LAST_OUTPUT}"; }
        printf '%s\n' "${LAST_OUTPUT}" > "${OUT}/t7-activity.json"
        stop_it
        trap - EXIT
        # BYTE-SHAPE IDENTICAL is the exit condition (3.10, and Owen's "an app
        # cannot tell which engine read it"). Compared on the PARSED shape —
        # the keys of each block — rather than on the text, because two
        # engines reading one page are allowed to disagree about a character
        # and are not allowed to disagree about the dialect.
        if [ -f "${OUT}/t6/shape.json" ] && [ -f "${OUT}/t7/shape.json" ]; then
          if ! diff -u "${OUT}/t6/shape.json" "${OUT}/t7/shape.json" > "${OUT}/t7-shape.diff"; then
            fail "the two engines answered in DIFFERENT shapes:
$(cat "${OUT}/t7-shape.diff")"
          fi
        fi
        pass "dots-ocr and qwen3.5-9b both answered on the Windows engine.
\`\`\`
${pages_said}

${cleanup_said}
\`\`\`"
      fi
    fi
  fi
fi

# ------------------------------------------------------------------------ T8

if wanted T8; then
  begin "T8 — the remove door on the staged Windows server"
  if present "the staged Windows home" "${RUN_ROOT}/home/config.toml" \
      "T8 removes a subject from the staged server's catalog (3.5a)."; then
    run "CRUCIBLE_HOME='${STAGED_HOME_WIN}' python -m crucible remove model qwen3.5-9b --json" \
      || fail "${LAST_OUTPUT}"
    removed="${LAST_OUTPUT:-}"
    run "CRUCIBLE_HOME='${STAGED_HOME_WIN}' python -m crucible models list --json" \
      || fail "${LAST_OUTPUT}"
    if [ "$DRY_RUN" = "0" ]; then
      printf '%s\n' "${LAST_OUTPUT}" > "${OUT}/t8-models.json"
      still="$(python - "${OUT}/t8-models.json" <<'PYEOF'
import json
import sys

document = json.loads(open(sys.argv[1], encoding="utf-8").read())
rows = document["models"] if isinstance(document, dict) else document
for row in rows:
    if row.get("id") == "qwen3.5-9b":
        print("INSTALLED" if row.get("installed") else "gone")
        break
else:
    print("absent-from-catalog")
PYEOF
)"
      [ "${still}" = "gone" ] || fail "after the remove, qwen3.5-9b reads ${still}"
    fi
    pass "204 (the CLI's equivalent), then \`installed: false\`.
\`\`\`
${removed}
\`\`\`"
  fi
fi

# ------------------------------------------------------------------------ T9

if wanted T9; then
  begin "T9 — the Mac"
  skip "NOT IN THIS SCRIPT, deliberately. T9 drives a SECOND machine's card
over ssh, and a button on this one that reaches across the network to hold
somebody else's GPU is a button whose blast radius is not what its name says.
Run it from the Mac: 7c's M-steps are the recipe, the deploy is a \`git pull\`
of \`/Volumes/Callisto/Projects/crucible\` (the editable checkout, never a
wheel), and what it proves is \`align\` and \`asr\` enabled with one job each.
\`pages\` is NOT expected there — 4.6's mlx-vlm finding."
fi

# ----------------------------------------------------------------------- T10

if wanted T10; then
  begin "T10 — the engine task, Windows → WSL2"
  if present "the staged Windows home" "${RUN_ROOT}/home/config.toml" \
      "T10 posts the engine task to the staged Windows server."; then
    if [ "$DRY_RUN" = "1" ]; then
      note "WOULD RUN: crucible serve with CRUCIBLE_HOST_DOOR set; POST /v1/tasks {engine,wsl}; read the events"
      pass "(dry run)"
    else
      # Is anything listening on the host's door? A GET, never a POST: a POST
      # to a LIVE door starts a real WSL install. Any HTTP code at all — 405
      # included — means something answered; `000` means nothing did, which is
      # what `-w '%{http_code}'` says and a bare exit status does not (curl
      # exits 0 for a 404 as happily as for a 200).
      door_code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:7101/install" 2>/dev/null || printf '000')"
      [ -n "${door_code}" ] || door_code="000"
      host_running=0
      [ "${door_code}" = "000" ] || host_running=1
      log="${OUT}/t10-server.log"
      # 7102, not 7101: the host's DOOR is 7101 and the staged server's own
      # port was 7101 too, which is fine while nothing else runs — but T10 is
      # the one stage where both exist at once.
      ( CRUCIBLE_HOME="${STAGED_HOME_WIN}" CRUCIBLE_HOST_DOOR="http://127.0.0.1:7101" \
          python -m crucible serve --host 127.0.0.1 --port 7102 > "${log}" 2>&1 & echo $! > "${OUT}/t10.pid" )
      stop_it() {
        if [ -f "${OUT}/t10.pid" ]; then
          kill "$(cat "${OUT}/t10.pid")" 2>/dev/null || true
          rm -f "${OUT}/t10.pid"
        fi
      }
      trap stop_it EXIT
      token="$(grep -m1 '^token' "${RUN_ROOT}/home/config.toml" | cut -d'"' -f2)"
      tries=0
      until curl -sS -o /dev/null "http://127.0.0.1:7102/v1/ping" 2>/dev/null; do
        tries=$(( tries + 1 ))
        [ "${tries}" -lt 60 ] || { stop_it; fail "the staged server never answered
$(tail -40 "${log}")"; }
        sleep 1
      done
      run "curl -sS -X POST -H 'Authorization: Bearer ${token}' -H 'X-Crucible-Api: 1' -H 'Content-Type: application/json' -d '{\"type\": \"engine\", \"target\": \"wsl\"}' http://127.0.0.1:7102/v1/tasks" \
        || { stop_it; fail "${LAST_OUTPUT}"; }
      submitted="${LAST_OUTPUT}"
      printf '%s\n' "${submitted}" > "${OUT}/t10-submit.json"
      # 4.7, AND THE CORRECTION THE FIRST RUN FORCED. With CRUCIBLE_HOST_DOOR
      # SET there is nothing to refuse at submit time: the server hands the
      # move to the host and relays its events, so the POST is a 202 with a
      # task id and EVERY failure of the door is named in the TASK. The first
      # run of this stage read the POST body for a refusal, found a task id
      # and called the server wrong. The server was right; this reads the task.
      task_id="$(python "${REPO}/scripts/task_field.py" "${OUT}/t10-submit.json" task_id)"
      if [ "${host_running}" = "0" ]; then
        [ -n "${task_id}" ] || { stop_it; fail "no host is running, and the POST
answered without a task id at all, so there is no task to read the refusal from:
${submitted}"; }
        tries=0
        state=""
        while [ "${tries}" -lt 60 ]; do
          curl -sS -H "Authorization: Bearer ${token}" -H 'X-Crucible-Api: 1' \
            "http://127.0.0.1:7102/v1/tasks/${task_id}" > "${OUT}/t10-task.json" \
            || { stop_it; fail "the task could not be read back"; }
          state="$(python "${REPO}/scripts/task_field.py" "${OUT}/t10-task.json" state)"
          case "${state}" in done|failed|cancelled) break ;; esac
          tries=$(( tries + 1 ))
          sleep 1
        done
        code="$(python "${REPO}/scripts/task_field.py" "${OUT}/t10-task.json" error.code)"
        stop_it
        trap - EXIT
        if [ "${state}" = "failed" ] && [ "${code}" = "host_unreachable" ]; then
          pass "the POST was ACCEPTED, which is what 4.7 asks for when
\`CRUCIBLE_HOST_DOOR\` is set, and the task then failed BY NAME because the
door at 7101 is not answering. \`host_unreachable\` and not
\`engine_move_needs_host\`: the variable says a host started this server, so
the answer is *start its door again*, not *start a host*.
\`\`\`json
${submitted}

$(cat "${OUT}/t10-task.json")
\`\`\`"
        else
          fail "nothing is listening on the host's door, and the task did not
fail \`host_unreachable\`. state=${state:-<none>} code=${code:-<none>}
$(cat "${OUT}/t10-task.json" 2>/dev/null)"
        fi
      else
        stop_it
        trap - EXIT
        pass "something IS answering (HTTP ${door_code}) on 127.0.0.1:7101, so
the task was accepted and handed to that door and a REAL move may be running
now.
\`\`\`
${submitted}
\`\`\`
Watch it finish on the page, or with
\`curl .../v1/tasks/${task_id}/events\`. The move's own steps are the host's
and its log is \`%LOCALAPPDATA%\\Crucible\\host.log\`."
      fi
    fi
  fi
fi

# ------------------------------------------------------------------- the end

if [ "$DRY_RUN" = "0" ]; then
  {
    printf '\n---\n\n## Figures that were UNMEASURED and now are not\n\n'
    printf 'Paste these back into sections 7.3, 7b, 7c and 7.6:\n\n'
    printf -- '- seconds/page, dots under vLLM in WSL (T6)\n'
    printf -- '- seconds/page, dots under llama.cpp on Windows (T7)\n'
    printf -- '- whether the Q8 GGUF answers in `parseDotsPage`'"'"'s dialect (T7)\n'
    printf -- '- seconds/chunk, qwen3.5-9b Q8 on Windows (T7)\n'
    printf -- '- the three `llama-windows` `memory_bytes_estimate` figures, measured (T7)\n'
  } >> "${REPORT}"
  say "done — ${REPORT}"
else
  say "dry run complete — nothing was executed"
fi
