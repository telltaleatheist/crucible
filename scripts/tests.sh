#!/usr/bin/env bash
# Run the tests that can say something about what changed.
#
#   ./scripts/tests.sh              # same as --changed
#   ./scripts/tests.sh --changed    # only what the diff since the last tag reaches
#   ./scripts/tests.sh --all        # everything
#   ./scripts/tests.sh --list       # what --changed WOULD run, and why
#
# Owen, 2026-09-17: *"We don't have to run a billion tests every time we cut a
# release. Only test the things that changed."* The full suite is 2112 tests and
# takes ten minutes, nearly all of it spent waiting rather than computing, and a
# patch release usually touches one module.
#
# HOW A CHANGED FILE IS TURNED INTO TESTS, and the one rule that matters:
#
#   tests/test_x.py changed   -> run tests/test_x.py
#   any other file changed    -> run every test file that names it: by path, by
#                                module stem, or failing those by its directory
#   anything WIDE changed     -> run everything (the list is below)
#   nothing names it usefully -> run everything, and say which file caused it
#
# THE LAST LINE IS THE IMPORTANT ONE. A selector that silently skips what it does
# not understand is a selector that gets quieter as the codebase grows, which is
# the exact moment it should get louder. Not knowing what a file affects is a
# reason to run everything, never a reason to run less — so the default direction
# of every uncertainty here is MORE tests.
#
# And the mapping is textual, not semantic: "a test file that mentions this
# module" over-selects (a comment counts) and can under-select (a module reached
# only through a re-export is not named). The under-selection is the real risk,
# which is why WIDE exists and why --all is what a release ultimately runs when
# anything structural moved. This is a way to iterate quickly, not a proof.

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

#: Files whose change can affect anything, so they select the whole suite.
#: `conftest.py` and the fakes are every test's environment; `app.py` and
#: `config.py` are reached from almost everywhere. The last line is the
#: exception that makes the rest work -- see `version_line_only`.
is_wide() {
  case "$1" in
    tests/conftest.py|tests/live_server.py|tests/fake_*.py) return 0 ;;
    crucible/app.py|crucible/config.py) return 0 ;;
    # CI configuration can change how anything runs. The release SCRIPTS
    # cannot -- nothing under test imports them -- so they go through the
    # ordinary rule, which finds tests/test_release_tooling.py by name.
    .github/*) return 0 ;;
    crucible/__init__.py|pyproject.toml) version_line_only "$1" && return 1 || return 0 ;;
  esac
  return 1
}

#: True when the ONLY thing that changed in a file is its version literal.
#:
#: TWO FILES CHANGE ON EVERY SINGLE RELEASE and both would otherwise be wide:
#: `crucible/__init__.py` holds the package surface AND the version, and
#: `pyproject.toml` holds the pytest configuration AND the version. Treating
#: either as wide means every release runs the whole suite no matter what the
#: release actually contains -- a selector switched off precisely when it is
#: being asked to work. A one-line version change is not a behaviour change, and
#: the tests that care about the version find it by name like any other module.
version_line_only() {
  local diff
  # `$base` to the WORKING TREE, which is where an uncommitted bump lives.
  diff="$(git diff -U0 "$base" -- "$1")"
  [ -n "$diff" ] || return 1
  # Every +/- line that is not a diff header must be a version literal, and
  # both are anchored at column 0 -- so a dependency pin changing inside a
  # pyproject table is not one of them and still widens.
  ! echo "$diff" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | grep -qvE '^[+-](VERSION|version) = "'
}

#: Files that cannot affect the Python tests at all. Every one of these is here
#: because it has no importer in `tests/` -- documentation, the vendored SDK's
#: own test suites (which `release.sh` builds and runs separately), and licences.
is_irrelevant() {
  case "$1" in
    docs/*|README.md|LICENSE|*.md) return 0 ;;
    sdk/*) return 0 ;;
    modules/*.module.json) return 0 ;;
  esac
  return 1
}

mode="--changed"
case "${1:-}" in
  ""|--changed) mode="--changed" ;;
  --all) mode="--all" ;;
  --list) mode="--list" ;;
  -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) fail "unknown argument $1 (try --changed, --all or --list)" ;;
esac

run_all() {
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
everything=""

add() { selected="$selected $1"; reasons="$reasons
  $1  <- $2"; }

for path in $changed; do
  [ -e "$path" ] || continue        # deleted files select nothing of their own
  if is_irrelevant "$path"; then
    continue
  fi
  if is_wide "$path"; then
    everything="$everything $path"
    continue
  fi
  case "$path" in
    tests/test_*.py)
      add "$path" "it is the test that changed"
      ;;
    *)
      # THREE NAMES ARE TRIED, most specific first, and the first that finds
      # anything wins:
      #
      #   the file's own repo-relative path  — how a test names a script or a
      #                                        data file it reads
      #   its module stem, for crucible/*.py — how a test names an import
      #   its directory                      — how a test names a family of
      #                                        data files it does not name one
      #                                        by one (crucible/voices/*.toml)
      #
      # AND A NAME THAT MATCHES ALMOST EVERYTHING IS NOT A MATCH. `scripts/
      # tests.sh` has the stem "tests", which appears in every test file in the
      # repo, so it selected all of them while printing a confident per-file
      # reason for each. A selector that cannot tell "this really is everything"
      # from "my search term was useless" is worse than no selector, because it
      # looks like it worked. Over the threshold, the name is discarded and the
      # next one is tried; if none survives, the answer is the whole suite.
      total="$(ls tests/test_*.py | wc -l)"
      stem="$(basename "$path")"
      stem="${stem%.*}"
      parent="$(basename "$(dirname "$path")")"
      candidates="$path"
      case "$path" in crucible/*.py) candidates="$candidates $stem" ;; esac
      # THE DIRECTORY IS ONLY A USEFUL NAME INSIDE THE PACKAGE, where it names a
      # family of data files a test reads together (crucible/voices/*.toml is
      # what tests mean by "voices"). Outside it, a directory is just a place:
      # "scripts" appears in twenty-three test files and says nothing about any
      # of them.
      case "$path" in crucible/*/*) candidates="$candidates $parent" ;; esac
      hits=""
      matched=""
      for name in $candidates; do
        found="$(grep -rlw -- "$name" tests/ --include='test_*.py' 2>/dev/null || true)"
        [ -z "$found" ] && continue
        count="$(echo "$found" | wc -l)"
        if [ "$count" -gt $(( total * 3 / 5 )) ]; then
          echo "tests: '"'"'$name'"'"' matches $count of $total test files, which narrows nothing — ignoring it"
          continue
        fi
        hits="$found"; matched="$name"; break
      done
      if [ -z "$hits" ]; then
        # A file no test names — or names only uselessly — is either untested or
        # reached indirectly, and this cannot tell those apart. Both are reasons
        # to run everything.
        everything="$everything $path(no test names it)"
      else
        for hit in $hits; do add "$hit" "names $matched"; done
      fi
      ;;
  esac
done

if [ -n "$everything" ]; then
  echo "tests: these changes reach further than this can narrow:$everything"
  [ "$mode" = "--list" ] && { echo "tests: would run the whole suite"; exit 0; }
  run_all
fi

selected="$(echo $selected | tr ' ' '\n' | sort -u | tr '\n' ' ')"
if [ -z "$(echo "$selected" | tr -d ' ')" ]; then
  echo "tests: nothing changed that any test covers (docs and the SDK only)"
  exit 0
fi

echo "tests: since $base, these files changed:"
echo "$changed" | sed 's/^/  /'
echo "tests: selecting$reasons"
echo

if [ "$mode" = "--list" ]; then
  echo "tests: would run: $selected"
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
