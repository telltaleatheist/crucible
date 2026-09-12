#!/usr/bin/env bash
# Cut one release: the Python sdist, the Python wheel, and the SDK tarball,
# under a single tag `v<version>`.
#
#   ./scripts/release.sh                 # release main
#   ./scripts/release.sh --dry-run       # build and check, create nothing
#   ./scripts/release.sh --branch <name> # release a branch (see below)
#
# One version, one tag, one release, three assets:
#
#   crucible-<ver>.tar.gz          the server sdist
#   crucible-<ver>-py3-none-any.whl  the server wheel
#   crucible-client-<ver>.tgz      the TypeScript SDK, installable by URL
#
# The version is read from three places and every one of them must agree:
# crucible/__init__.py, sdk/ts/package.json, and sdk/ts/src/version.ts (which
# the SDK reports in its User-Agent). A mismatch is a refusal, not a warning.
#
# If the tag already exists — locally, on the remote, or as a release — this
# refuses. Re-cutting a version is how two different sets of bytes end up with
# one name.
#
# Runs on Git Bash (Windows), Linux and macOS. Requires: git, gh (logged in),
# node 20+, npm, python with the `build` module.

set -euo pipefail

RELEASE_BRANCH="main"
REPO_SLUG="telltaleatheist/crucible"

branch_override=""
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --branch)
      # The one legitimate use: the first release of a feature branch the lead
      # is about to merge, so the tarball URL exists before the merge. The
      # branch is named in the release notes, so nobody has to guess later.
      [ $# -ge 2 ] || { echo "release: --branch needs a branch name" >&2; exit 2; }
      branch_override="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "release: unknown argument $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "release: $*" >&2; exit 1; }

for tool in git gh node npm python; do
  command -v "$tool" >/dev/null || fail "no $tool on PATH"
done
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 20 ] || fail "node $NODE_MAJOR is too old; the SDK needs 20+"
python -c 'import build' 2>/dev/null || fail "python has no \`build\` module (pip install build)"
gh auth status >/dev/null 2>&1 || fail "gh is not logged in (gh auth login)"

# ------------------------------------------------------------------ the tree

WANTED_BRANCH="${branch_override:-$RELEASE_BRANCH}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "$WANTED_BRANCH" ] \
  || fail "on branch $BRANCH, not $WANTED_BRANCH (releases come from $RELEASE_BRANCH; pass --branch to release another)"

[ -z "$(git status --porcelain)" ] \
  || fail "the working tree is dirty; a release must be reproducible from the tag"

HEAD_SHA="$(git rev-parse HEAD)"
git fetch --quiet origin "$BRANCH"
REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"
[ "$HEAD_SHA" = "$REMOTE_SHA" ] \
  || fail "HEAD ($(git rev-parse --short HEAD)) is not origin/$BRANCH ($(git rev-parse --short "origin/$BRANCH")); push first"

# ----------------------------------------------------------------- the version

PY_VERSION="$(sed -n 's/^VERSION = "\(.*\)"$/\1/p' crucible/__init__.py)"
SDK_VERSION="$(node -p "require('./sdk/ts/package.json').version")"
UA_VERSION="$(sed -n "s/^export const SDK_VERSION = '\(.*\)';$/\1/p" sdk/ts/src/version.ts)"

[ -n "$PY_VERSION" ]  || fail "could not read VERSION from crucible/__init__.py"
[ -n "$SDK_VERSION" ] || fail "could not read version from sdk/ts/package.json"
[ -n "$UA_VERSION" ]  || fail "could not read SDK_VERSION from sdk/ts/src/version.ts"

[ "$PY_VERSION" = "$SDK_VERSION" ] \
  || fail "crucible/__init__.py says $PY_VERSION but sdk/ts/package.json says $SDK_VERSION; one release, one version"
[ "$PY_VERSION" = "$UA_VERSION" ] \
  || fail "crucible/__init__.py says $PY_VERSION but sdk/ts/src/version.ts says $UA_VERSION; the SDK would report the wrong version in User-Agent"

VERSION="$PY_VERSION"
TAG="v$VERSION"
echo "release: $TAG from $BRANCH ($(git rev-parse --short HEAD))"

# ------------------------------------------------------------- refuse a re-cut

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  fail "tag $TAG already exists locally; a version is cut once"
fi
if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  fail "tag $TAG already exists on origin; a version is cut once"
fi
if gh release view "$TAG" --repo "$REPO_SLUG" >/dev/null 2>&1; then
  fail "release $TAG already exists; a version is cut once"
fi

# -------------------------------------------------------------------- building

OUT="$REPO/dist"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "release: building the python sdist and wheel"
python -m build --outdir "$OUT" >"$OUT/python-build.log" 2>&1 \
  || { cat "$OUT/python-build.log" >&2; fail "python -m build failed"; }

echo "release: building the sdk"
( cd sdk/ts && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null )
( cd sdk/ts && npm pack --silent --pack-destination "$OUT" >/dev/null )

SDIST="$OUT/crucible-$VERSION.tar.gz"
WHEEL="$OUT/crucible-$VERSION-py3-none-any.whl"
TGZ="$OUT/crucible-client-$VERSION.tgz"
for asset in "$SDIST" "$WHEEL" "$TGZ"; do
  [ -f "$asset" ] || fail "expected asset $asset was not built"
done
echo "release: built"
for asset in "$SDIST" "$WHEEL" "$TGZ"; do
  echo "  $(basename "$asset")"
done

if [ "$dry_run" = "1" ]; then
  echo "release: --dry-run, so $TAG was not created"
  exit 0
fi

# ------------------------------------------------------------------- the release

NOTES_HEADER="Server \`crucible\` $VERSION and TypeScript client \`@crucible/client\` $VERSION."
if [ -n "$branch_override" ]; then
  NOTES_HEADER="$NOTES_HEADER

Cut from branch \`$branch_override\` at \`$(git rev-parse --short HEAD)\`, before it was merged to \`$RELEASE_BRANCH\`."
fi
NOTES_HEADER="$NOTES_HEADER

Install the client:

\`\`\`
npm install https://github.com/$REPO_SLUG/releases/download/$TAG/crucible-client-$VERSION.tgz
\`\`\`"

gh release create "$TAG" \
  --repo "$REPO_SLUG" \
  --target "$HEAD_SHA" \
  --title "$TAG" \
  --generate-notes \
  --notes "$NOTES_HEADER" \
  "$SDIST" "$WHEEL" "$TGZ"

echo "release: $TAG created"
gh release view "$TAG" --repo "$REPO_SLUG" --json tagName,url,assets \
  --jq '.tagName + "  " + .url, (.assets[] | "  asset: " + .name + " (" + (.size|tostring) + " bytes)")'
