#!/usr/bin/env bash
# One command from "the code is finished" to "the release is cut".
#
#   ./scripts/ship.sh patch              # 0.6.7 -> 0.6.8, and cut it
#   ./scripts/ship.sh minor
#   ./scripts/ship.sh 0.7.0
#   ./scripts/ship.sh patch --dry-run    # everything except creating anything
#   ./scripts/ship.sh --no-bump          # cut the version the tree already names
#   ./scripts/ship.sh patch --deploy     # and put it on the machines afterwards
#
# RELEASING WAS NINE STEPS IN FOUR PLACES, and the reason it was slow had almost
# nothing to do with the work: seven version literals edited by hand, two
# generators that had to be remembered, a commit, a push, a build, a workflow
# watched in a browser tab, the machines upgraded by hand in their own shells,
# and a promotion that was forgotten six times in a row. Every one of those
# steps was correct. There were just too many of them to do reliably at 5am.
#
# So this is the order, and each step's refusal is its own:
#
#   1. the tree is clean and pushed          (a release must be reproducible)
#   2. bump.py writes the seven and regenerates      (scripts/bump.py)
#   3. commit and push
#   4. the cut                               (scripts/release.sh)
#   5. the machines, at the same time        (scripts/deploy.sh; --deploy only)
#   6. the promote command, printed          (scripts/promote_release.py — not run)
#   7. where it went                         (a row per step, and per machine)
#
# THERE IS NO TEST STEP, and there is no flag that adds one back. Owen,
# 2026-09-18 (PHASE20-CODE-NOT-ENVIRONMENTS.md 7): *"Normal deploy does not need
# 25 minutes worth of tests. We should run one or two focused tests on the area
# of code we changed before we reach the deploy stage. By the time we reach
# deploy, we should know it's going to work already."* The tests that can say
# anything about a change ran on the branch, focused on what changed, before the
# merge — the repo's changed-file selector is what does that, and it is a
# person's tool for their branch, not a stage in this. A flag that defaulted to
# running none of them would have been the same suite one word away from being
# back on the deploy path, which is the thing being removed, so the step is gone
# rather than quiet. NEITHER THE SELECTOR NOR THE SUITE IS NAMED ANYWHERE BELOW,
# deliberately: a release script that knows how to run tests is a release script
# somebody will make run them again.
#
# NOTHING IS WAITED FOR EITHER. This used to watch the workflow that built the
# environment packs, because a release was environments as well as code. Under
# PHASE20 a release is a wheel and its installers; `release.sh` dispatches no
# CI, so there is no run to wait for and no browser tab to watch it in.
#
# THERE IS NO SEPARATE DRY-RUN GATE BEFORE THE COMMIT, and there cannot be:
# `release.sh --dry-run` refuses a dirty tree and an unpushed HEAD, which is
# exactly what a just-bumped working tree is. The first version of this ran it
# at step 3 and it refused every time. It is not needed either — release.sh
# checks the tree, the branch, the remote, all seven versions, the generated
# files and the absence of the tag, and BUILDS every asset, before it creates
# anything. Nothing is published by a run that fails; the cost of a failure
# after step 3 is a bump commit on main with no release beside it, which
# `./scripts/ship.sh --no-bump` picks straight back up.
#
# PROMOTION IS NOT AUTOMATED AND MUST NOT BE. `promote_release.py --publish`
# requires `--confirmed-install-smoke`, which is an ATTESTATION that a person
# installed the candidate and it worked. Release metadata cannot prove that, and
# a script that passed the flag on its own would be a script that lies. What this
# does instead is get the candidate installed (step 5) so the attestation is
# true, and then print the command.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "ship: $*" >&2; exit 1; }

# ------------------------------------------------------------------ the clock
#
# Owen, 2026-09-18: *"why does it take so long? are we fully repackaging it
# every time?"* — and nobody could answer from a run's own output, because
# nothing here recorded a single number. A release that is measured is a release
# whose next "why was that slow" is read off the log instead of guessed at.
#
# `step` is the banner AND the clock. A phase is the gap between two banners, so
# a step added later cannot be a step nobody timed — there is no second call to
# forget. The rows are accumulated as `name|seconds` lines and rendered once at
# the end; a run that dies partway prints no table, which is correct, because
# the failure is the thing to read then.
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

# Seconds as a person reads them. Under a minute stays seconds, because "0m07s"
# is a number pretending to be precise about the wrong thing.
hms() {
  if [ "$1" -ge 60 ]; then printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 ));
  else printf '%ds' "$1"; fi
}

# WHERE THE RELEASE WENT. Printed last, from the rows every `step` left behind.
where_it_went() {
  phase_close
  echo
  echo "ship: where v$version went ($(hms $(( $(date +%s) - SHIP_STARTED ))))"
  printf '%s' "$PHASE_ROWS" | while IFS='|' read -r name seconds; do
    [ -n "$name" ] || continue
    printf '  %-38s %8s\n' "$name" "$(hms "$seconds")"
  done
}

level=""
dry_run=0
no_bump=0
do_deploy=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --no-bump) no_bump=1; shift ;;
    --deploy)  do_deploy=1; shift ;;
    -h|--help) sed -n '2,62p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
  step "the version bump: none, $version unchanged (--no-bump)"
else
  step "the version bump"
  python scripts/bump.py "$level"
  version="$(sed -n 's/^VERSION = "\(.*\)"$/\1/p' crucible/__init__.py)"
  [ -n "$version" ] || fail "could not read the new version back from crucible/__init__.py"
fi

if [ "$dry_run" = "1" ]; then
  echo
  echo "ship: --dry-run, so v$version was not committed, pushed or tagged."
  echo "ship: the bump is in the working tree; 'git checkout .' undoes it, and"
  echo "ship: './scripts/ship.sh --no-bump' carries on from here."
  git --no-pager diff --stat
  where_it_went
  exit 0
fi

# ------------------------------------------------------- 3. commit and push

step "the commit and the push"
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "Cut $version"
fi
git push origin main

# --------------------------------------------------------------- 4. the cut

step "the build and the cut of v$version"
./scripts/release.sh

# ------------------------------------------------------------- 5. the machines

if [ "$do_deploy" = "1" ]; then
  step "the machines"
  # `--yes` BECAUSE NOBODY IS AT THE KEYBOARD. deploy.sh's "this restarts the
  # Windows tray host, the WSL engine it owns and the Mac agent, y/N" is the right
  # question when a person typed `deploy.sh`; reached through here it is a
  # prompt with no stdin behind it, and the cutover agent watched a release sit
  # at it (`cutover-progress.log`, 02:16:32). The person answered it when they
  # passed --deploy.
  #
  # `tee` so the operator watches it live AND the per-machine numbers can be
  # read back afterwards. deploy.sh is what knows them: it forked the three
  # installs and it joined them, so timing the call from here would only ever
  # learn the slowest machine. stderr is left alone, so its failure summary
  # still lands on stderr.
  deploy_log="$(mktemp)"
  ./scripts/deploy.sh --release "$version" --yes | tee "$deploy_log"
  # Closed HERE so the fleet's total is the row above its machines rather than
  # below them: the point of the three rows is that they do not add up to it.
  phase_close
  # A here-doc and not a pipe: a pipe would put the loop in a subshell and the
  # rows would be appended to a copy of PHASE_ROWS that dies with it.
  while read -r machine seconds; do
    [ -n "$machine" ] || continue
    PHASE_ROWS="$PHASE_ROWS  the machines: $machine|$seconds
"
  done <<EOF
$(sed -n 's/^deploy: timing \([A-Za-z0-9_-]*\) \([0-9][0-9]*\)$/\1 \2/p' "$deploy_log")
EOF
  # A cleanup failure is not a release failure, and it is not silent either.
  rm -f "$deploy_log" || echo "ship: could not remove $deploy_log" >&2
fi

# ------------------------------------------------------------- 6. what is left

echo
echo "ship: v$version is cut, built and downloadable."
if [ "$do_deploy" != "1" ]; then
  echo "ship: put it on the machines:"
  echo "  ./scripts/deploy.sh --release $version"
fi
echo "ship: then promote it — this is the step that makes releases/latest serve $version,"
echo "ship: and the flag attests that you installed it, so only you can pass it:"
echo "  python scripts/promote_release.py --tag v$version --publish --confirmed-install-smoke"
echo "ship: and repin the apps:"
echo "  node tools/adopt-crucible-release.mjs $version     # in bookforge, and in foundry"

where_it_went
