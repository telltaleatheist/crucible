#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "ship: $*" >&2; exit 1; }

SHIP_STARTED="$(date +%s)"
PHASE_ROWS=""
phase_name=""
phase_started=0

phase_close() {
  [ -n "$phase_name" ] || return 0
  PHASE_ROWS="$PHASE_ROWS$phase_name|$(( $(date +%s) - phase_started ))
"
  phase_name=""
}

step() {
  phase_close
  phase_name="$*"
  phase_started="$(date +%s)"
  echo
  echo "=== $* ==="
}

hms() {
  if [ "$1" -ge 60 ]; then printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 ));
  else printf '%ds' "$1"; fi
}

where_it_went() {
  local status="$1"
  if [ "$status" -ne 0 ] && [ -n "$phase_name" ]; then
    phase_name="$phase_name — FAILED"
  fi
  phase_close
  local what="v$version"
  [ -n "$version" ] || what="this run"
  echo
  echo "ship: where $what went ($(hms $(( $(date +%s) - SHIP_STARTED ))))"
  printf '%s' "$PHASE_ROWS" | while IFS='|' read -r name seconds; do
    [ -n "$name" ] || continue
    printf '  %-38s %8s\n' "$name" "$(hms "$seconds")"
  done
}

level=""
dry_run=0
no_bump=0
do_deploy=0
version=""

usage() {
  cat <<'USAGE'
One command from "the code is finished" to "the release is cut".

  ./scripts/ship.sh patch              0.6.7 -> 0.6.8, and cut it
  ./scripts/ship.sh minor
  ./scripts/ship.sh 0.7.0
  ./scripts/ship.sh patch --dry-run    everything except creating anything
  ./scripts/ship.sh --no-bump          cut the version the tree already names
  ./scripts/ship.sh patch --deploy     and put it on the machines afterwards

Steps: clean tree, bump, commit and push, the cut, the machines (--deploy),
the promote command (printed, not run), and a table of where the time went.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --no-bump) no_bump=1; shift ;;
    --deploy)  do_deploy=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) fail "unknown argument $1" ;;
    *) [ -z "$level" ] || fail "name one version bump, not two ($level and $1)"; level="$1"; shift ;;
  esac
done

if [ "$no_bump" = "1" ]; then
  [ -z "$level" ] || fail "--no-bump and a version ($level) are two different instructions"
else
  [ -n "$level" ] || fail "say what to bump: major, minor, patch, an explicit x.y.z, or --no-bump"
fi

trap 'where_it_went "$?"' EXIT

step "the tree"
[ -z "$(git status --porcelain)" ] \
  || fail "the working tree is dirty. A release is cut from a commit, so commit or stash first"
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "main" ] || fail "on $branch, not main"
git fetch --quiet origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || fail "HEAD and origin/main differ; push or pull first"
echo "ship: main is clean and matches origin at $(git rev-parse --short HEAD)"

if [ "$no_bump" = "1" ]; then
  version="$(python scripts/bump.py --check)" \
    || fail "the version places are not one version (bump.py said why, above); run 'python scripts/bump.py --align', commit, and ship again"
  step "the version bump: none, $version unchanged (--no-bump)"
else
  step "the version bump"
  python scripts/bump.py "$level"
  version="$(python scripts/bump.py --check)" \
    || fail "the bump left the version places disagreeing (bump.py said why, above); 'git checkout .' undoes it"
fi

if [ "$dry_run" = "1" ]; then
  echo
  echo "ship: --dry-run, so v$version was not committed, pushed or tagged."
  echo "ship: the bump is in the working tree; 'git checkout .' undoes it, and"
  echo "ship: './scripts/ship.sh --no-bump' carries on from here."
  git --no-pager diff --stat
  exit 0
fi

step "the commit and the push"
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "Cut $version"
fi
git push origin main

step "the build and the cut of v$version"
./scripts/release.sh

if [ "$do_deploy" = "1" ]; then
  step "the machines"
  deploy_log="$(mktemp)"
  ./scripts/deploy.sh --release "$version" --yes | tee "$deploy_log"
  phase_close
  while read -r machine seconds; do
    [ -n "$machine" ] || continue
    PHASE_ROWS="$PHASE_ROWS  the machines: $machine|$seconds
"
  done <<EOF
$(sed -n 's/^deploy: timing \([A-Za-z0-9_-]*\) \([0-9][0-9]*\)$/\1 \2/p' "$deploy_log")
EOF
  rm -f "$deploy_log" || echo "ship: could not remove $deploy_log" >&2
fi

echo
echo "ship: v$version is cut, built and downloadable."
if [ "$do_deploy" != "1" ]; then
  echo "ship: put it on the machines:"
  echo "  ./scripts/deploy.sh --release $version"
fi
echo "ship: then promote it — this is the step that makes releases/latest serve $version,"
echo "ship: and the flag attests that you installed it, so only you can pass it:"
echo "  $(python scripts/promote_release.py --tag "v$version" --print-command)"
echo "ship: and repin the apps:"
echo "  node tools/adopt-crucible-release.mjs $version     # in bookforge, and in foundry"
