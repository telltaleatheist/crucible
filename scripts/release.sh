#!/usr/bin/env bash
set -euo pipefail

RELEASE_BRANCH="main"
REPO_SLUG="telltaleatheist/crucible"

branch_override=""
dry_run=0

usage() {
  cat <<'USAGE'
Cut one release under a single tag v<version>: the sdist, the wheel and its
.sha256, the client and bootstrap tarballs, install.sh and install.ps1.

  ./scripts/release.sh                 release main
  ./scripts/release.sh --dry-run       build and check, create nothing
  ./scripts/release.sh --branch <name> release a branch about to merge
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --branch)
      [ $# -ge 2 ] || { echo "release: --branch needs a branch name" >&2; exit 2; }
      branch_override="$2"; shift 2 ;;
    -h|--help)
      usage
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

PY_VERSION="$(sed -n 's/^VERSION = "\(.*\)"$/\1/p' crucible/__init__.py)"
SDK_VERSION="$(node -p "require('./sdk/ts/package.json').version")"
UA_VERSION="$(sed -n "s/^export const SDK_VERSION = '\(.*\)';$/\1/p" sdk/ts/src/version.ts)"
TOML_VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml)"
BOOT_VERSION="$(node -p "require('./sdk/bootstrap/package.json').version")"
BOOT_PEER="$(node -p "require('./sdk/bootstrap/package.json').peerDependencies['@crucible/client']")"
BOOT_LITERAL="$(sed -n "s/^export const BOOTSTRAP_VERSION = '\(.*\)';$/\1/p" sdk/bootstrap/src/version.ts)"

[ -n "$PY_VERSION" ]   || fail "could not read VERSION from crucible/__init__.py"
[ -n "$SDK_VERSION" ]  || fail "could not read version from sdk/ts/package.json"
[ -n "$UA_VERSION" ]   || fail "could not read SDK_VERSION from sdk/ts/src/version.ts"
[ -n "$TOML_VERSION" ] || fail "could not read version from pyproject.toml"
[ -n "$BOOT_VERSION" ] || fail "could not read version from sdk/bootstrap/package.json"
[ -n "$BOOT_PEER" ] && [ "$BOOT_PEER" != "undefined" ] \
  || fail "could not read the @crucible/client peer pin from sdk/bootstrap/package.json"
[ -n "$BOOT_LITERAL" ] || fail "could not read BOOTSTRAP_VERSION from sdk/bootstrap/src/version.ts"

[ "$PY_VERSION" = "$TOML_VERSION" ] \
  || fail "crucible/__init__.py says $PY_VERSION but pyproject.toml says $TOML_VERSION; the wheel would carry the wrong version (this is what nearly shipped 0.1.0 bytes as v0.2.0)"

[ "$PY_VERSION" = "$SDK_VERSION" ] \
  || fail "crucible/__init__.py says $PY_VERSION but sdk/ts/package.json says $SDK_VERSION; one release, one version"
[ "$PY_VERSION" = "$UA_VERSION" ] \
  || fail "crucible/__init__.py says $PY_VERSION but sdk/ts/src/version.ts says $UA_VERSION; the SDK would report the wrong version in User-Agent"
[ "$PY_VERSION" = "$BOOT_VERSION" ] \
  || fail "crucible/__init__.py says $PY_VERSION but sdk/bootstrap/package.json says $BOOT_VERSION; the bootstrapper ships at the server's version"
[ "$PY_VERSION" = "$BOOT_PEER" ] \
  || fail "sdk/bootstrap/package.json pins @crucible/client $BOOT_PEER, not $PY_VERSION; the bootstrapper must peer-depend on the client cut beside it"
[ "$PY_VERSION" = "$BOOT_LITERAL" ] \
  || fail "crucible/__init__.py says $PY_VERSION but sdk/bootstrap/src/version.ts says $BOOT_LITERAL; the bootstrapper would name itself wrongly"

VERSION="$PY_VERSION"
TAG="v$VERSION"
echo "release: $TAG from $BRANCH ($(git rev-parse --short HEAD))"

echo "release: the generated app modules match the manifests"
python scripts/gen-modules.py --check >/dev/null   || fail "modules/*.module.json are stale; run 'python scripts/gen-modules.py' and commit them (this is what shipped stale in v0.6.3)"

echo "release: the API reference matches the app"
python scripts/gen-api-docs.py --check >/dev/null   || fail "docs/API.md is stale; run 'python scripts/gen-api-docs.py' and commit it"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  fail "tag $TAG already exists locally; a version is cut once"
fi
if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  fail "tag $TAG already exists on origin; a version is cut once"
fi
if gh release view "$TAG" --repo "$REPO_SLUG" >/dev/null 2>&1; then
  fail "release $TAG already exists; a version is cut once"
fi

OUT="$REPO/dist"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "release: building the python sdist and wheel"
python -m build --outdir "$OUT" >"$OUT/python-build.log" 2>&1 \
  || { cat "$OUT/python-build.log" >&2; fail "python -m build failed"; }

echo "release: building the sdk"
( cd sdk/ts && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null )
( cd sdk/ts && npm pack --silent --pack-destination "$OUT" >/dev/null )

echo "release: building the bootstrap"
( cd sdk/bootstrap && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null )
( cd sdk/bootstrap && npm pack --silent --pack-destination "$OUT" >/dev/null )

echo "release: the generated installers match bootstrap's step list"
( cd sdk/bootstrap && npm run gen:install -- --check >/dev/null ) \
  || fail "sdk/bootstrap/scripts/install.sh|.ps1 are stale; run 'npm run gen:install' in sdk/bootstrap and commit them"
INSTALL_SH="$REPO/sdk/bootstrap/scripts/install.sh"
INSTALL_PS1="$REPO/sdk/bootstrap/scripts/install.ps1"
for asset in "$INSTALL_SH" "$INSTALL_PS1"; do
  [ -f "$asset" ] || fail "expected asset $asset was not generated"
done

SDIST="$OUT/crucible-$VERSION.tar.gz"
WHEEL="$OUT/crucible-$VERSION-py3-none-any.whl"
TGZ="$OUT/crucible-client-$VERSION.tgz"
BOOT="$OUT/crucible-bootstrap-$VERSION.tgz"
for asset in "$SDIST" "$WHEEL" "$TGZ" "$BOOT"; do
  [ -f "$asset" ] || fail "expected asset $asset was not built"
done

WHEEL_SHA="$WHEEL.sha256"
if command -v sha256sum >/dev/null 2>&1; then
  ( cd "$OUT" && sha256sum "$(basename "$WHEEL")" > "$(basename "$WHEEL_SHA")" )
else
  ( cd "$OUT" && shasum -a 256 "$(basename "$WHEEL")" > "$(basename "$WHEEL_SHA")" )
fi
[ -s "$WHEEL_SHA" ] || fail "could not write $WHEEL_SHA"

echo "release: built"
for asset in "$SDIST" "$WHEEL" "$WHEEL_SHA" "$TGZ" "$BOOT"; do
  echo "  $(basename "$asset")"
done

if [ "$dry_run" = "1" ]; then
  echo "release: --dry-run, so $TAG was not created"
  exit 0
fi

NOTES_HEADER="Server \`crucible\` $VERSION, TypeScript client \`@crucible/client\` $VERSION, and app-side bootstrapper \`@crucible/bootstrap\` $VERSION."
if [ -n "$branch_override" ]; then
  NOTES_HEADER="$NOTES_HEADER

Cut from branch \`$branch_override\` at \`$(git rev-parse --short HEAD)\`, before it was merged to \`$RELEASE_BRANCH\`."
fi
NOTES_HEADER="$NOTES_HEADER

Install the client, and the bootstrapper beside it (it peer-depends on the client at this exact version):

\`\`\`
npm install https://github.com/$REPO_SLUG/releases/download/$TAG/crucible-client-$VERSION.tgz
npm install https://github.com/$REPO_SLUG/releases/download/$TAG/crucible-bootstrap-$VERSION.tgz
\`\`\`"

NOTES_HEADER="$NOTES_HEADER

### Environments

This release carries CODE. \`crucible install <type>\` builds a job env from its
recipe (\`crucible/envs/<type>/<recipe>.txt\`) with pip, from PyPI and the
pinned indexes; \`install.sh\` and \`install.ps1\` download the pinned CPython from
python-build-standalone and pip this release's wheel into it; the Windows host
imports Ubuntu's own WSL image from cloud-images.ubuntu.com. None of those bytes
are ours, so none of them are here."

gh release create "$TAG" \
  --repo "$REPO_SLUG" \
  --target "$HEAD_SHA" \
  --title "$TAG" \
  --prerelease --latest=false \
  --generate-notes \
  --notes "$NOTES_HEADER" \
  "$SDIST" "$WHEEL" "$WHEEL_SHA" "$TGZ" "$BOOT" \
  "$INSTALL_SH" "$INSTALL_PS1"

echo "release: $TAG candidate created (prerelease, not latest)"
echo "release: test a fresh install, then run python scripts/promote_release.py --tag $TAG --publish --confirmed-install-smoke"
gh release view "$TAG" --repo "$REPO_SLUG" --json tagName,url,assets \
  --jq '.tagName + "  " + .url, (.assets[] | "  asset: " + .name + " (" + (.size|tostring) + " bytes)")'
