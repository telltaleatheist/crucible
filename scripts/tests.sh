#!/usr/bin/env bash
# Run the tests that name what changed.
#
#   ./scripts/tests.sh              # same as --changed
#   ./scripts/tests.sh --changed    # only the tests that name the change
#   ./scripts/tests.sh --all        # everything
#   ./scripts/tests.sh --list       # what --changed WOULD run, and why
#
# Owen, 2026-09-18 (PHASE20-CODE-NOT-ENVIRONMENTS.md 7): *"If we change
# crucible's handshake logic, we don't need to re-run the GPU test. We can test
# the handshake logic we just built and assume the GPU works since it did last
# time we changed anything. If it breaks, we can debug from there."*
#
# THE WHOLE SELECTION RULE:
#
#   tests/test_x.py changed     -> run tests/test_x.py
#   any other .py, .ts or .sh   -> run every test file that names it: by its
#                                  repo path, by its module, or by its basename
#   a file whose only change is -> nothing. A version literal is not a
#     a version literal            behaviour change, and the files that carry
#                                  one change on every single release
#   anything else               -> nothing
#
# THE LAST LINE USED TO SAY "run everything", and above it sat a list of files
# whose change fanned out to the whole suite: conftest, the fakes, app.py,
# config.py, and anything under .github/. The reasoning was that not knowing
# what a file affects is a reason to run MORE, and it was right while a release
# leaned on this. Nothing leans on it now — `ship.sh` runs no tests at all
# (PHASE20 7) and this is a person's tool for the branch they are on. A
# selector that answers "all 101 files, ten minutes" to a one-line change is a
# selector nobody runs, which is how the suite came to be on the deploy path in
# the first place. So the direction of every uncertainty here is now FEWER
# tests, and `--all` is the answer when something is wrong or we're debugging.
#
# The mapping is textual, not semantic: "a test file that mentions this module"
# over-selects (a comment counts) and can under-select (a module reached only
# through a re-export is not named). Both are said out loud — an unnamed file
# is reported by name rather than quietly widening — and neither is a proof.
# This is a way to iterate quickly. `--all` is the proof.
#
# NOTHING HERE CAN SELECT A LIVE KEEPER. `scripts/keeper-live.sh`,
# `keeper-tts-live.sh` and `keeper-llm-live.sh` are how this repo marks a check
# that needs a real model on a real card; they are shell scripts, not pytest
# files, and every candidate below is looked for only inside `tests/test_*.py`.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "tests: $*" >&2; exit 1; }

# ------------------------------------------------------------ where pytest runs
#
# NOT HERE, when "here" is Windows. The suite drives `crucible`, and the CLI
# refuses to run on Windows by design, so the Python tests run inside WSL2 —
# which `scripts/e2e-from-windows.sh` already arranges, and whose two settings
# (`CRUCIBLE_WSL_DISTRO`, `CRUCIBLE_WSL_ENV`) are reused here rather than
# invented again. wsl.exe translates the current drvfs directory itself, so the
# relative test paths chosen below mean the same thing on both sides.
#
# AND IT REFUSES WHILE A FINE-TUNE IS RUNNING. The WSL guest has 13 GB and one
# card, and a pytest run started beside `train_lora.py` takes them from it. This
# was a rule somebody had to remember; it is now a refusal. `flock` is the same
# rule against a second copy of this script.
DISTRO="${CRUCIBLE_WSL_DISTRO:-Ubuntu}"
WSL_ENV="${CRUCIBLE_WSL_ENV:-/home/telltale/anaconda3/envs/crucible}"

if [ -n "${CRUCIBLE_PYTEST_PYTHON:-}" ]; then
  pytest_cmd() { "$CRUCIBLE_PYTEST_PYTHON" -m pytest -q "$@"; }
  WHERE="$CRUCIBLE_PYTEST_PYTHON"
else
  case "${OSTYPE:-}" in
    msys*|cygwin*)
      command -v wsl.exe >/dev/null || fail "no wsl.exe on PATH, and the Python suite does not run on Windows"
      pytest_cmd() {
        wsl.exe -d "$DISTRO" --exec bash -c "
          if pgrep -f '[t]rain_lora' >/dev/null; then
            echo 'tests: a fine-tune is running in $DISTRO — it owns the card and the RAM. Refusing.' >&2
            exit 2
          fi
          exec flock -w 1800 /tmp/crucible-pytest.lock '$WSL_ENV/bin/python' -m pytest -q $*"
      }
      WHERE="$DISTRO:$WSL_ENV"
      ;;
    *)
      pytest_cmd() { python -m pytest -q "$@"; }
      WHERE="python"
      ;;
  esac
fi

#: True when the ONLY thing that changed in a file is its version literal.
#:
#: TWO FILES CHANGE ON EVERY SINGLE RELEASE: `crucible/__init__.py` holds the
#: package surface AND the version, and `pyproject.toml` holds the pytest
#: configuration AND the version. Both are named by tests — this repo's own
#: release-tooling tests name `crucible/__init__.py` by path — so without this
#: rule every release would select those tests for a one-line change that is
#: not a behaviour change at all.
version_line_only() {
  local diff
  # `$base` to the WORKING TREE, which is where an uncommitted bump lives.
  diff="$(git diff -U0 "$base" -- "$1")"
  [ -n "$diff" ] || return 1
  # Every +/- line that is not a diff header must be a version literal, and
  # both are anchored at column 0 -- so a dependency pin changing inside a
  # pyproject table is not one of them and is an ordinary change.
  ! echo "$diff" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | grep -qvE '^[+-](VERSION|version) = "'
}

mode="--changed"
case "${1:-}" in
  ""|--changed) mode="--changed" ;;
  --all) mode="--all" ;;
  --list) mode="--list" ;;
  -h|--help) sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) fail "unknown argument $1 (try --changed, --all or --list)" ;;
esac

run_all() {
  # `--list` is a question, not an instruction, whichever branch reaches here.
  [ "$mode" = "--list" ] && { echo "tests: would run the whole suite"; exit 0; }
  echo "tests: the whole suite ($WHERE)"
  pytest_cmd
  exit $?
}

[ "$mode" = "--all" ] && run_all

# ------------------------------------------------------------- what changed
#
# Since the last tag, PLUS whatever is in the working tree — a release is cut
# from a commit, but this is run before the commit exists.

# THE TAGS ARE NOT HERE UNLESS THEY ARE FETCHED, and they usually are not.
# `release.sh` creates each tag SERVER-SIDE, through `gh release create --target
# <sha>`, so nothing ever writes it into this checkout. Measured 2026-09-17: the
# repo had tags up to v0.6.0 while the remote had v0.6.7, so `git describe`
# answered with a tag seven releases old and "what changed since the last
# release" was seven releases of changes -- which is not wrong so much as
# useless, and silently so, because it just runs everything and looks careful.
#
# A base that cannot be trusted is the one thing left that still widens, and it
# widens because there is nothing to narrow AGAINST — not because a file was
# not understood.
if ! git fetch --tags --quiet origin 2>/dev/null; then
  echo "tests: could not fetch tags, so 'since the last release' cannot be trusted"
  run_all
fi

base="$(git describe --tags --abbrev=0 2>/dev/null || true)"
if [ -z "$base" ]; then
  echo "tests: no tag to compare against"
  run_all
fi

changed="$( { git diff --name-only "$base"..HEAD; git status --porcelain | cut -c4-; } | sort -u )"
if [ -z "$changed" ]; then
  echo "tests: nothing has changed since $base"
  exit 0
fi

# ------------------------------------------------------- what they select

selected=""
reasons=""
unnamed=""

#: One reason per selected file: the FIRST that chose it. Two changed files can
#: reach the same test, and printing both reasons beside one file reads as two
#: selections of it.
add() {
  case " $selected " in *" $1 "*) return 0 ;; esac
  selected="$selected $1"
  reasons="$reasons$1|$2
"
}

for path in $changed; do
  [ -e "$path" ] || continue        # deleted files select nothing of their own
  case "$path" in
    tests/test_*.py)
      add "$path" "it is the test that changed"
      continue
      ;;
    # ONLY CODE SELECTS TESTS. A workflow, a doc, an `envs/*.txt` recipe, a
    # module.json, a licence: nothing under tests/ runs any of them, and the
    # one that used to widen -- .github/* -- widened on the grounds that CI
    # configuration can change how anything runs, which is true of the run CI
    # does and nothing to do with the run a person is about to do here.
    *.py|*.ts|*.sh) ;;
    *) continue ;;
  esac

  version_line_only "$path" && continue

  # THREE NAMES ARE TRIED, most specific first, and the first that finds
  # anything wins:
  #
  #   the file's own repo-relative path  — how a test names a script or a
  #                                        data file it reads
  #   its module stem, for crucible/*.py — how a test names an import
  #   its basename                       — how a test names a file it does not
  #                                        reach through the repo root
  #
  # AND A NAME THAT MATCHES ALMOST EVERYTHING IS NOT A MATCH. `scripts/
  # tests.sh` has the stem "tests", which appears in every test file in the
  # repo, so it selected all of them while printing a confident per-file
  # reason for each. A selector that cannot tell "this really is everything"
  # from "my search term was useless" is worse than no selector, because it
  # looks like it worked. Over the threshold, the name is discarded and the
  # next one is tried.
  total="$(ls tests/test_*.py | wc -l)"
  base_name="$(basename "$path")"
  stem="${base_name%.*}"
  candidates="$path"
  case "$path" in crucible/*.py) candidates="$candidates $stem" ;; esac
  candidates="$candidates $base_name"
  hits=""
  matched=""
  for name in $candidates; do
    found="$(grep -rlw -- "$name" tests/ --include='test_*.py' 2>/dev/null || true)"
    [ -z "$found" ] && continue
    count="$(echo "$found" | wc -l)"
    if [ "$count" -gt $(( total * 3 / 5 )) ]; then
      echo "tests: '$name' matches $count of $total test files, which narrows nothing — ignoring it"
      continue
    fi
    hits="$found"; matched="$name"; break
  done
  if [ -z "$hits" ]; then
    # SAID, NOT WIDENED. A file no test names is either untested or reached
    # only through a re-export, and this cannot tell those apart — but under
    # PHASE20 7 neither is a reason to run the other hundred files. It is a
    # reason to name the file, so the person reading the list can decide.
    unnamed="$unnamed $path"
  else
    for hit in $hits; do add "$hit" "names $matched"; done
  fi
done

echo "tests: since $base, these files changed:"
echo "$changed" | sed 's/^/  /'
[ -n "$unnamed" ] && echo "tests: no test in tests/ names:$unnamed"

if [ -z "$(echo "$selected" | tr -d ' ')" ]; then
  echo "tests: nothing that changed is named by any test — run --all if that is a surprise"
  exit 0
fi

echo "tests: selecting"
printf '%s' "$reasons" | while IFS='|' read -r path reason; do
  [ -n "$path" ] || continue
  printf '  %s  <- %s\n' "$path" "$reason"
done
echo

if [ "$mode" = "--list" ]; then
  echo "tests: would run:$selected"
  exit 0
fi

echo "tests: $(echo $selected | wc -w) file(s) of the $(ls tests/test_*.py | wc -l) in tests/ ($WHERE)"
# shellcheck disable=SC2086
pytest_cmd $selected
status=$?
if [ "$status" != "0" ]; then
  echo "tests: FAILED. Run ./scripts/tests.sh --all before deciding this is unrelated." >&2
fi
exit "$status"
