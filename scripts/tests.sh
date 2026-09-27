#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "tests: $*" >&2; exit 1; }

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

version_line_only() {
  local diff
  diff="$(git diff -U0 "$base" -- "$1")"
  [ -n "$diff" ] || return 1
  ! echo "$diff" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | grep -qvE '^[+-](VERSION|version) = "'
}

usage() {
  cat <<'USAGE'
Run the tests that name what changed.

  ./scripts/tests.sh              same as --changed
  ./scripts/tests.sh --changed    only the tests that name the change
  ./scripts/tests.sh --all        everything
  ./scripts/tests.sh --list       what --changed WOULD run, and why
USAGE
}

mode="--changed"
case "${1:-}" in
  ""|--changed) mode="--changed" ;;
  --all) mode="--all" ;;
  --list) mode="--list" ;;
  -h|--help) usage; exit 0 ;;
  *) fail "unknown argument $1 (try --changed, --all or --list)" ;;
esac

run_all() {
  [ "$mode" = "--list" ] && { echo "tests: would run the whole suite"; exit 0; }
  echo "tests: the whole suite ($WHERE)"
  pytest_cmd
  exit $?
}

[ "$mode" = "--all" ] && run_all

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

selected=""
reasons=""
unnamed=""

add() {
  case " $selected " in *" $1 "*) return 0 ;; esac
  selected="$selected $1"
  reasons="$reasons$1|$2
"
}

for path in $changed; do
  [ -e "$path" ] || continue
  case "$path" in
    tests/test_*.py)
      add "$path" "it is the test that changed"
      continue
      ;;
    *.py|*.ts|*.sh) ;;
    *) continue ;;
  esac

  version_line_only "$path" && continue

  total="$(ls tests/test_*.py | wc -l)"
  base_name="$(basename "$path")"
  stem="${base_name%.*}"
  candidates="$path"
  case "$path" in crucible/*.py) candidates="$candidates $stem" ;; esac
  candidates="$candidates $base_name"
  hits=""
  matched=""
  for name in $candidates; do
    found="$(grep -rlw --include='test_*.py' -- "$name" tests/ 2>/dev/null || true)"
    [ -z "$found" ] && continue
    count="$(echo "$found" | wc -l)"
    if [ "$count" -gt $(( total * 3 / 5 )) ]; then
      echo "tests: '$name' matches $count of $total test files, which narrows nothing — ignoring it"
      continue
    fi
    hits="$found"; matched="$name"; break
  done
  if [ -z "$hits" ]; then
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
pytest_cmd $selected
status=$?
if [ "$status" != "0" ]; then
  echo "tests: FAILED. Run ./scripts/tests.sh --all before deciding this is unrelated." >&2
fi
exit "$status"
