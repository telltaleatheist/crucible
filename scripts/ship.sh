#!/usr/bin/env bash
# One command from "the code is finished" to "the candidate is cut and its packs are building".
#
#   ./scripts/ship.sh patch              # 0.6.7 -> 0.6.8, cut it, watch the packs
#   ./scripts/ship.sh minor
#   ./scripts/ship.sh 0.7.0
#   ./scripts/ship.sh patch --dry-run    # everything except creating anything
#   ./scripts/ship.sh --no-bump          # cut the version the tree already names
#   ./scripts/ship.sh patch --deploy     # and put it on the three machines afterwards
#
# RELEASING WAS NINE STEPS IN FOUR PLACES, and the reason it was slow had almost
# nothing to do with the work: seven version literals edited by hand, two
# generators that had to be remembered, a commit, a push, a build, a dispatch, a
# workflow watched in a browser tab, three machines upgraded in three shells, and
# a promotion that was forgotten six times in a row. Every one of those steps was
# correct. There were just too many of them to do reliably at 5am.
#
# So this is the order, and each step's refusal is its own:
#
#   1. the tree is clean and pushed          (a release must be reproducible)
#   2. bump.py writes the seven and regenerates      (scripts/bump.py)
#   3. the tests that can say no             (scripts/tests.sh --changed)
#   4. commit and push
#   5. release.sh                            (tag, six assets, dispatch envpacks)
#   6. watch the pack build                  (gh run watch)
#   7. deploy.sh                             (optional; three machines)
#   8. the promote command, printed          (see below — it is not run here)
#
# THERE IS NO SEPARATE DRY-RUN GATE BEFORE THE COMMIT, and there cannot be:
# `release.sh --dry-run` refuses a dirty tree and an unpushed HEAD, which is
# exactly what a just-bumped working tree is. The first version of this ran it
# at step 4 and it refused every time. It is not needed either — release.sh
# checks the tree, the branch, the remote, all seven versions, the generated
# files and the absence of the tag, and BUILDS all six assets, before it creates
# anything. Nothing is published by a run that fails; the cost of a failure
# after step 4 is a bump commit on main with no release beside it, which
# `./scripts/ship.sh --no-bump` picks straight back up.
#
# PROMOTION IS NOT AUTOMATED AND MUST NOT BE. `promote_release.py --publish`
# requires `--confirmed-install-smoke`, which is an ATTESTATION that a person
# installed the candidate and it worked. Release metadata cannot prove that, and
# a script that passed the flag on its own would be a script that lies. What this
# does instead is get the candidate installed (step 8) so the attestation is
# true, and then print the command.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "ship: $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

level=""
dry_run=0
no_bump=0
do_deploy=0
test_mode="--changed"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --no-bump) no_bump=1; shift ;;
    --deploy)  do_deploy=1; shift ;;
    --test)    [ $# -ge 2 ] || fail "--test needs all, changed or none"; test_mode="--$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) fail "unknown argument $1" ;;
    *) [ -z "$level" ] || fail "name one version bump, not two ($level and $1)"; level="$1"; shift ;;
  esac
done

if [ "$no_bump" = "1" ]; then
  [ -z "$level" ] || fail "--no-bump and a version ($level) are two different instructions"
else
  [ -n "$level" ] || fail "say what to bump: major, minor, patch, an explicit x.y.z, or --no-bump"
fi

# ------------------------------------------------------------- 1. the tree

step "the tree"
[ -z "$(git status --porcelain)" ] \
  || fail "the working tree is dirty. A release is cut from a commit, so commit or stash first"
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "main" ] || fail "on $branch, not main"
git fetch --quiet origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || fail "HEAD and origin/main differ; push or pull first"
echo "ship: main is clean and matches origin at $(git rev-parse --short HEAD)"

# ------------------------------------------------------------- 2. the version

if [ "$no_bump" = "1" ]; then
  version="$(sed -n 's/^VERSION = "\(.*\)"$/\1/p' crucible/__init__.py)"
  step "the version: $version, unchanged (--no-bump)"
else
  step "the version"
  python scripts/bump.py "$level"
  version="$(sed -n 's/^VERSION = "\(.*\)"$/\1/p' crucible/__init__.py)"
  [ -n "$version" ] || fail "could not read the new version back from crucible/__init__.py"
fi

# ------------------------------------------------------------- 3. the tests

step "the tests"
case "$test_mode" in
  --none) echo "ship: skipped by --test none" ;;
  *) ./scripts/tests.sh "$test_mode" || fail "the tests said no; nothing has been pushed or tagged" ;;
esac

if [ "$dry_run" = "1" ]; then
  echo
  echo "ship: --dry-run, so v$version was not committed, pushed or tagged."
  echo "ship: the bump is in the working tree; 'git checkout .' undoes it, and"
  echo "ship: './scripts/ship.sh --no-bump' carries on from here."
  git --no-pager diff --stat
  exit 0
fi

# ------------------------------------------------------- 4. commit and push

if [ -n "$(git status --porcelain)" ]; then
  step "committing $version"
  git add -A
  git commit -m "Cut $version"
fi
git push origin main

# --------------------------------------------------------------- 5. the cut

step "cutting v$version"
# The newest envpacks run BEFORE the dispatch. Step 7 waits for one that is not
# this -- otherwise `gh run watch` is handed the PREVIOUS release's run, which is
# already complete and already green, and reports success in under a second
# without a single pack having been built.
previous_run="$(gh run list --workflow envpacks.yml --limit 1 --json databaseId \
                  --jq '.[0].databaseId' 2>/dev/null || true)"
./scripts/release.sh

# ---------------------------------------------------------- 6. the pack build
#
# `release.sh` dispatches `envpacks.yml` and returns. The run takes a moment to
# appear, so this waits for one dispatched AT OR AFTER the cut rather than
# grabbing the newest, which would otherwise be the PREVIOUS release's run.

step "the environment packs"
run_id=""
for _ in $(seq 1 45); do
  newest="$(gh run list --workflow envpacks.yml --limit 1 --json databaseId \
              --jq '.[0].databaseId' 2>/dev/null || true)"
  if [ -n "$newest" ] && [ "$newest" != "$previous_run" ]; then
    run_id="$newest"; break
  fi
  sleep 4
done
if [ -z "$run_id" ]; then
  echo "ship: no NEW envpacks run appeared within three minutes of the dispatch." >&2
  echo "ship: the tag and its six assets exist; the packs do not. Check and re-dispatch:" >&2
  echo "  gh run list --workflow envpacks.yml" >&2
  echo "  gh workflow run envpacks.yml -f tag=v$version" >&2
  exit 1
else
  echo "ship: watching envpacks run $run_id"
  echo "ship: only the packs whose recipe changed are built; the rest are carried"
  gh run watch "$run_id" --exit-status || fail "the pack build failed. The tag and its six assets exist; re-dispatch with: gh workflow run envpacks.yml -f tag=v$version"
fi

# ------------------------------------------------------------- 7. the machines

if [ "$do_deploy" = "1" ]; then
  step "the machines"
  ./scripts/deploy.sh --release "$version"
fi

# ------------------------------------------------------------- 8. what is left

echo
echo "ship: v$version is cut, built and downloadable."
if [ "$do_deploy" != "1" ]; then
  echo "ship: put it on the three machines:"
  echo "  ./scripts/deploy.sh --release $version"
fi
echo "ship: then promote it — this is the step that makes releases/latest serve $version,"
echo "ship: and the flag attests that you installed it, so only you can pass it:"
echo "  python scripts/promote_release.py --tag v$version --publish --confirmed-install-smoke"
echo "ship: and repin the apps:"
echo "  node tools/adopt-crucible-release.mjs $version     # in bookforge, and in foundry"
