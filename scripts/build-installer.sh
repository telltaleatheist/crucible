#!/usr/bin/env bash
set -euo pipefail

NSIS_VERSION="3.13"
NSIS_URL="https://downloads.sourceforge.net/project/nsis/NSIS%203/$NSIS_VERSION/nsis-$NSIS_VERSION.zip"
NSIS_SHA="ba63dffc4410ee89193e1cb5a41989991bd77c61068da17e3156d136b7b0b3d8"
NSCURL_VERSION="26.9.28.324"
NSCURL_URL="https://github.com/negrutiu/nsis-nscurl/releases/download/v$NSCURL_VERSION/NScurl.zip"
NSCURL_SHA="44e787707ffc1285e4d0dc62932e866205f57aeb1c55d3a700ec6f3c28f16e23"
REPO_SLUG="telltaleatheist/crucible"

usage() {
  cat <<'USAGE'
Build the Windows setup, dist/crucible-setup-<version>.exe, with a pinned portable
NSIS and the pinned NScurl plugin fetched into a build cache.

  ./scripts/build-installer.sh                          this checkout's version and dist/ wheel
  ./scripts/build-installer.sh --wheel <path>           pin the setup to this wheel's sha256
  ./scripts/build-installer.sh --version <v> --out <dir>

The setup downloads the release's wheel at install time, so --wheel must be the exact
wheel the release uploads. The cache is $CRUCIBLE_BUILD_CACHE, else build/installer-cache.
USAGE
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail() { echo "build-installer: $*" >&2; exit 1; }

version=""
wheel=""
out="$REPO/dist"
while [ $# -gt 0 ]; do
  case "$1" in
    --version) [ $# -ge 2 ] || fail "--version needs a version"; version="$2"; shift 2 ;;
    --wheel) [ $# -ge 2 ] || fail "--wheel needs a path"; wheel="$2"; shift 2 ;;
    --out) [ $# -ge 2 ] || fail "--out needs a directory"; out="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument $1 (try --help)" ;;
  esac
done

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *) fail "the Windows setup is built with Windows NSIS; run this on the Windows PC in Git Bash" ;;
esac

[ -n "$version" ] || version="$(python scripts/bump.py --check)" \
  || fail "the version places disagree (bump.py said why, above); run 'python scripts/bump.py --align' first"
[ -n "$wheel" ] || wheel="$out/crucible-$version-py3-none-any.whl"
[ -f "$wheel" ] || fail "no wheel at $wheel; build it with: python -m build --wheel --outdir $out"
[ "$(basename "$wheel")" = "crucible-$version-py3-none-any.whl" ] \
  || fail "$wheel is not the $version wheel the release uploads (crucible-$version-py3-none-any.whl)"

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

cache="${CRUCIBLE_BUILD_CACHE:-$REPO/build/installer-cache}"
mkdir -p "$cache"

fetch_pinned() {
  local url="$1" target="$2" pinned="$3" got
  if [ -f "$target" ] && [ "$(sha_of "$target")" = "$pinned" ]; then
    return 0
  fi
  rm -f "$target"
  echo "build-installer: fetching $url"
  curl -fL --retry 3 --retry-delay 2 -sS -o "$target.partial" "$url" || fail "could not download $url; check the network and run this again"
  got="$(sha_of "$target.partial")"
  if [ "$got" != "$pinned" ]; then
    rm -f "$target.partial"
    fail "$url hashes $got, but this script pins $pinned; the download was deleted. Run this again; if it repeats, the upstream file changed and the pin in scripts/build-installer.sh must be reviewed"
  fi
  mv "$target.partial" "$target"
}

fetch_pinned "$NSIS_URL" "$cache/nsis-$NSIS_VERSION.zip" "$NSIS_SHA"
fetch_pinned "$NSCURL_URL" "$cache/NScurl-$NSCURL_VERSION.zip" "$NSCURL_SHA"

nsis="$cache/nsis-$NSIS_VERSION"
if [ ! -x "$nsis/makensis.exe" ]; then
  rm -rf "$nsis"
  unzip -q "$cache/nsis-$NSIS_VERSION.zip" -d "$cache" || fail "could not unpack $cache/nsis-$NSIS_VERSION.zip"
fi
plugins="$cache/nscurl-$NSCURL_VERSION"
if [ ! -f "$plugins/x86-unicode/NScurl.dll" ]; then
  rm -rf "$plugins" "$cache/nscurl-unpack"
  unzip -q "$cache/NScurl-$NSCURL_VERSION.zip" -d "$cache/nscurl-unpack" || fail "could not unpack the NScurl plugin"
  mv "$cache/nscurl-unpack/Plugins" "$plugins"
  rm -rf "$cache/nscurl-unpack"
fi

ps1="$REPO/sdk/bootstrap/scripts/install.ps1"
pin() { sed -n "s/^\$$1 = '\(.*\)'\$/\1/p" "$ps1" | head -n 1; }
py_url="$(pin PyUrl)"; py_sha="$(pin PySha)"; py_asset="$(pin PyAsset)"; py_version="$(pin PyVersion)"
for value in "$py_url" "$py_sha" "$py_asset" "$py_version"; do
  [ -n "$value" ] || fail "could not read the pinned interpreter out of $ps1; run 'npm run gen:install' in sdk/bootstrap"
done

wheel_name="crucible-$version-py3-none-any.whl"
wheel_sha="$(sha_of "$wheel")"
setup="$out/crucible-setup-$version.exe"
mkdir -p "$out"

winpath() { cygpath -w "$1"; }

echo "build-installer: $setup (wheel $wheel_sha)"
"$nsis/makensis.exe" -V2 -NOCD \
  "-DVERSION=$version" \
  "-DOUTFILE=$(winpath "$setup")" \
  "-DPLUGINS=$(winpath "$plugins/x86-unicode")" \
  "-DICON=$(winpath "$REPO/crucible/desktop_app/assets/crucible.ico")" \
  "-DWELCOME=$(winpath "$REPO/installer/windows/welcome.bmp")" \
  "-DINSTALL_PS1=$(winpath "$ps1")" \
  "-DPY_URL=$py_url" "-DPY_SHA=$py_sha" "-DPY_ASSET=$py_asset" "-DPY_VERSION=$py_version" \
  "-DWHEEL_URL=https://github.com/$REPO_SLUG/releases/download/v$version/$wheel_name" \
  "-DWHEEL_NAME=$wheel_name" "-DWHEEL_SHA=$wheel_sha" \
  "$(winpath "$REPO/installer/windows/crucible.nsi")" \
  || fail "makensis refused installer/windows/crucible.nsi (its message is above)"
[ -s "$setup" ] || fail "makensis finished but wrote no $setup"
echo "build-installer: built $setup"
