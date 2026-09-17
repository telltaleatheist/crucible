#!/usr/bin/env bash
# Cut one release: the Python sdist, the Python wheel, the SDK tarball and the
# bootstrap tarball, under a single tag `v<version>`.
#
#   ./scripts/release.sh                 # release main
#   ./scripts/release.sh --dry-run       # build and check, create nothing
#   ./scripts/release.sh --branch <name> # release a branch (see below)
#
# One version, one tag, one release, six assets:
#
#   crucible-<ver>.tar.gz          the server sdist
#   crucible-<ver>-py3-none-any.whl  the server wheel
#   crucible-client-<ver>.tgz      the TypeScript SDK, installable by URL
#   crucible-bootstrap-<ver>.tgz   the app-side installer/ensurer (PHASE5-APPS.md 6.0),
#                                  peer-depending on the client at this exact version
#   install.sh                     the standalone installer for Linux/WSL and macOS
#   install.ps1                    the same for Windows: the HOST pack, and stop
#
# The two installers are GENERATED from bootstrap's own step list
# (PHASE14-ENVPACKS.md 4a), so an app-driven install and a hand install cannot
# differ. A stale one refuses the cut. Since PHASE15-HOST.md 4.4, `install.ps1`
# no longer walks the WSL states itself: it installs `crucible host` and stops,
# and the host owns the sequence from there — for the page's engine switch
# (4.7), for an app's `install()` and for a hand install alike.
#
# THE WORKFLOW BELOW UPLOADS AN ELEVENTH PACK. `.github/workflows/envpacks.yml`
# gained a `windows-latest` job for `crucible-env-host-llama-windows-<ver>`
# (4.4), which is what `install.ps1` downloads. A release without it is one
# where the first thing anybody runs on Windows refuses `pack_not_published`,
# which is why the manifest job waits for it.
#
# The ENVIRONMENT PACKS are not built here. `.github/workflows/envpacks.yml`
# is dispatched with this tag afterwards and uploads them beside the four,
# because a pack is built on the backend it targets and this script runs on
# one machine
# (PHASE14-ENVPACKS.md section 3.3). What this script does about them is refuse
# to cut a tag when that workflow is absent, and name the packs the tag will
# attempt in the notes.
#
# The version is read from seven places and every one of them must agree:
# crucible/__init__.py, pyproject.toml, sdk/ts/package.json, sdk/ts/src/version.ts (which
# the SDK reports in its User-Agent), sdk/bootstrap/package.json, its @crucible/client
# peer pin, and sdk/bootstrap/src/version.ts. A mismatch is a refusal, not a warning:
# a bootstrapper is never paired with a server nobody tested it against.
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
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

# The packs are listed in the notes below and they are generated from
# `crucible/envpack.py`, which is the one owner of what packs exist. A release
# machine that cannot import the package would get notes that silently omit
# them, which is worse than a refusal here.
python -c 'import crucible.envpack' 2>/dev/null \
  || fail "crucible is not importable in this python (pip install -e .); the notes list the packs this tag carries and are generated from crucible/envpack.py"

# -------------------------------------------------- the packs have a builder
#
# THE PACKS ARE PUBLISHED BY AN EXPLICIT DISPATCH, not by this tag.
# `.github/workflows/envpacks.yml` takes the tag as a `workflow_dispatch`
# input and NO trigger of its own — deliberately, as 0.6.2 recorded: "tag
# creation cannot launch another builder that overwrites verified candidate
# assets." The dispatch is this script's, after the release exists, so there
# is exactly one caller. A patch release rebuilds only the CORE runtime packs
# (they carry this code) and REUSES the unchanged inference archives under
# their original filenames, which `scripts/release_packs.py` plans and stages.
#
# Since 0.6.0 `crucible install <type>` DOWNLOADS a pack by default and
# refuses `pack_not_published` when the release has none, so a release whose
# packs were never built is one where every fresh machine's first install
# fails by name — the worst kind of working release. THIS SCRIPT DISPATCHES
# THEM: `gh workflow run envpacks.yml -f tag=$TAG` runs once the release
# exists, at the bottom of this file. The check here is the earlier half —
# that the workflow is in the tree at all — so a cut refuses up front rather
# than creating a release and then failing to dispatch anything.
ENVPACKS_WORKFLOW=".github/workflows/envpacks.yml"
[ -f "$ENVPACKS_WORKFLOW" ] \
  || fail "$ENVPACKS_WORKFLOW is not in this tree, so the tag would build no environment packs and \`crucible install\` would refuse every job type \`pack_not_published\` (PHASE14-ENVPACKS.md section 3.3)"

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

# THE APP MODULES carry this version too — `modules/<app>.module.json` names it
# as `<version>+<content hash>`, and apps vendor those bytes verbatim. They are
# GENERATED, so a bump that does not regenerate them ships a release whose
# module files name the release before it. That is exactly what v0.6.3 shipped:
# the seven version places above all agreed and nothing looked at `modules/`.
echo "release: the generated app modules match the manifests"
python scripts/gen-modules.py --check >/dev/null   || fail "modules/*.module.json are stale; run 'python scripts/gen-modules.py' and commit them (this is what shipped stale in v0.6.3)"

# THE API REFERENCE IS GENERATED TOO, from the FastAPI app this release ships.
# A field added to a request model and not regenerated ships a reference that
# does not mention it, which is the same failure as a stale module manifest and
# harder to notice, because nothing downstream breaks — a reader is just wrong.
echo "release: the API reference matches the app"
python scripts/gen-api-docs.py --check >/dev/null   || fail "docs/API.md is stale; run 'python scripts/gen-api-docs.py' and commit it"

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

# The bootstrap's dev dependency on the client is `file:../ts`, which `npm ci`
# links in place — so the client must already be built, and it is, just above.
echo "release: building the bootstrap"
( cd sdk/bootstrap && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null )
( cd sdk/bootstrap && npm pack --silent --pack-destination "$OUT" >/dev/null )

# THE TWO STANDALONE INSTALLERS (PHASE14-ENVPACKS.md 4a) are GENERATED from
# bootstrap's own step list, so that an app-driven install and a hand install
# cannot differ. A committed script that no longer matches that list is two
# answers to "how is Crucible installed", which is the shape
# docs/ARCHITECTURE.md section 1 is about - so a stale one refuses the cut
# rather than shipping beside a bootstrapper it disagrees with.
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
echo "release: built"
for asset in "$SDIST" "$WHEEL" "$TGZ" "$BOOT"; do
  echo "  $(basename "$asset")"
done

if [ "$dry_run" = "1" ]; then
  echo "release: --dry-run, so $TAG was not created"
  exit 0
fi

# ------------------------------------------------------------------- the release

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

# --------------------------------------------------- what the packs will be
#
# NAMED IN THE NOTES, from `crucible envpack list`'s own source, so a reader of
# the release page can tell a pack that was never meant to exist from one whose
# CI job failed. The manifest (`envpacks.json`) is uploaded LAST by
# `envpacks.yml`, so its presence is the release's own statement of what
# actually built; this list is what was ATTEMPTED.
PACK_LIST="$(python -c '
from crucible import envpack
for name, backend in envpack.every_pack():
    print(f"- `{name}` / {backend}")
')"
NOTES_HEADER="$NOTES_HEADER

### Environment packs

\`crucible install <type>\` downloads a pack from this release and unpacks it; it
builds nothing. This release script explicitly dispatches
\`.github/workflows/envpacks.yml\` for this tag and attempts:

$PACK_LIST

Each is \`crucible-env-<name>-<backend>-$VERSION.tar.zst\`, split into
\`.part00\`… under 1900 MiB, with \`envpacks.json\` naming the sha256 of the
reassembled whole. A pack missing from \`envpacks.json\` is one whose job did not
finish; \`crucible install\` refuses it \`pack_not_published\` rather than building
it quietly, and \`crucible envpack build <name>\` is the way to make it by hand."

gh release create "$TAG" \
  --repo "$REPO_SLUG" \
  --target "$HEAD_SHA" \
  --title "$TAG" \
  --prerelease --latest=false \
  --generate-notes \
  --notes "$NOTES_HEADER" \
  "$SDIST" "$WHEEL" "$TGZ" "$BOOT" \
  "$INSTALL_SH" "$INSTALL_PS1"

echo "release: $TAG candidate created (prerelease, not latest)"
gh workflow run envpacks.yml --repo "$REPO_SLUG" -f tag="$TAG"
echo "release: wait for every pack and rootfs, test fresh native installs, then run python scripts/promote_release.py --tag $TAG --publish --confirmed-install-smoke"
gh release view "$TAG" --repo "$REPO_SLUG" --json tagName,url,assets \
  --jq '.tagName + "  " + .url, (.assets[] | "  asset: " + .name + " (" + (.size|tostring) + " bytes)")'
