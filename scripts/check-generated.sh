#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

PYTHON="${PYTHON:-python}"

GENERATORS="gen-api-docs.py:docs/API.md gen-modules.py:modules/*.module.json gen-foundry-lineup.py:foundry-lineup.json"

usage() {
  cat <<'USAGE'
Every file a script generates is what that script would write today.

  ./scripts/check-generated.sh         check docs/API.md, modules/*.module.json, foundry-lineup.json
  ./scripts/check-generated.sh --fix   regenerate all three (then commit what changed)

Runs with $PYTHON (default: python), which must import this checkout's crucible.
USAGE
}

fix=0
case "${1:-}" in
  "") ;;
  --fix) fix=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "check-generated: unknown argument $1 (try --fix or --help)" >&2; exit 2 ;;
esac

stale=""
for entry in $GENERATORS; do
  script="${entry%%:*}"
  output="${entry#*:}"
  if [ "$fix" = "1" ]; then
    "$PYTHON" "scripts/$script" || { echo "check-generated: $PYTHON scripts/$script failed" >&2; exit 1; }
    continue
  fi
  if "$PYTHON" "scripts/$script" --check; then
    echo "check-generated: $output is current"
  else
    echo "check-generated: $output is STALE" >&2
    stale="$stale $script"
  fi
done

if [ -n "$stale" ]; then
  echo "check-generated: stale:$stale" >&2
  echo "check-generated: regenerate with ./scripts/check-generated.sh --fix, then commit what it wrote" >&2
  exit 1
fi
[ "$fix" = "1" ] && echo "check-generated: regenerated; commit what changed (git status shows it)"
exit 0
