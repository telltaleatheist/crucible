#!/bin/sh
# GENERATED FILE — do not edit.
# Written by sdk/bootstrap/scripts/gen-install-scripts.ts from src/steps.ts
# and src/wsl-states.ts, so a hand install and an app-driven install cannot
# differ (PHASE14-ENVPACKS.md 4a). Regenerate: npm run gen:install
#
# Install a Crucible on this machine (Linux x86_64, macOS arm64, or inside a
# WSL2 distro). Downloads the server pack from the release, initialises it,
# installs the service, and prints the line that pairs an app with it.
#
#   curl -fsSL https://github.com/telltaleatheist/crucible/releases/latest/download/install.sh | sh
#
# CRUCIBLE_RELEASE=<version> picks a release other than the one this script
# was cut with. Everything here is idempotent: run it again after a failure.

set -eu

RELEASE="${CRUCIBLE_RELEASE:-0.6.0}"

say() { printf 'crucible: %s\n' "$*"; }
die() { printf 'crucible: %s\n' "$*" >&2; exit 1; }

# --- backend -------------------------------------------------------------
# Two backends and no third. Windows is never one: on Windows this script
# runs INSIDE the WSL2 distro that install.ps1 imported.
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)  BACKEND=cuda-linux; SHA_TOOL="sha256sum";     MECHANISM=systemd ;;
  Darwin/arm64)  BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256"; MECHANISM=launchd ;;
  *) die "unsupported_platform: $(uname -s)/$(uname -m) is not a Crucible backend (cuda-linux on Linux x86_64, mlx-darwin on Apple Silicon)" ;;
esac
say "release $RELEASE, backend $BACKEND"

# --- host-facts ----------------------------------------------------------
# read this host: CRUCIBLE_HOME, the user, free disk, the tools a pack needs
say "host-facts"
crucible_probe() {
  h="${CRUCIBLE_HOME:-$HOME/.crucible}"; echo "home=$h"; echo "user=$(id -un)"; d="$h"; while [ ! -d "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done; echo "free_kib=$(df -Pk "$d" | awk 'NR==2 {print $4}')"; for t in curl tar zstd; do command -v "$t" >/dev/null 2>&1 || echo "missing=$t"; done; c="$h/server/bin/crucible"; if test -x "$c"; then echo "crucible=$c"; echo "version=$("$c" --version 2>&1 | head -1)"; fi; s="$h/server/.pack"; if test -f "$s"; then cat "$s"; fi; exit 0
}
probe_out="$(crucible_probe)"
CRUCIBLE_HOME="$(printf '%s\n' "$probe_out" | sed -n 's/^home=//p')"
GUEST_USER="$(printf '%s\n' "$probe_out" | sed -n 's/^user=//p')"
free_kib="$(printf '%s\n' "$probe_out" | sed -n 's/^free_kib=//p')"
stamp_sha="$(printf '%s\n' "$probe_out" | sed -n 's/^sha256=//p')"
missing="$(printf '%s\n' "$probe_out" | sed -n 's/^missing=//p' | tr '\n' ' ')"
if [ -n "$missing" ]; then die "guest_missing_tool: this machine has no $missing; a pack is fetched with curl and unpacked with tar --zstd"; fi

# --- server-pack ---------------------------------------------------------
# download, verify and unpack the server pack — the interpreter comes WITH it
say "server-pack"
dest="$CRUCIBLE_HOME/server"
partial="$dest.partial"
downloads="$CRUCIBLE_HOME/downloads"
manifest_url="https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/envpacks.json"
manifest="$(curl -fsSL --retry 3 "$manifest_url")" || die "pack_manifest_unreadable: could not fetch $manifest_url"
pack="$(printf '%s' "$manifest" | awk -v RS='}' -v b="$BACKEND" '$0 ~ /"name"[[:space:]]*:[[:space:]]*"server"/ && $0 ~ ("\"backend\"[[:space:]]*:[[:space:]]*\"" b "\"")' | tr -d '\n')"
[ -n "$pack" ] || die "pack_not_published: the $RELEASE release publishes no server pack for $BACKEND"
want_sha="$(printf '%s' "$pack" | sed -n 's/.*"sha256"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p')"
unpacked="$(printf '%s' "$pack" | sed -n 's/.*"unpacked_bytes"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p')"
archive_bytes="$(printf '%s' "$pack" | sed -n 's/.*"bytes"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p')"
parts="$(printf '%s' "$pack" | sed -n 's/.*"parts"[[:space:]]*:[[:space:]]*\[\([^]]*\)\].*/\1/p' | tr -d ' "' | tr ',' ' ')"
[ -n "$want_sha" ] && [ -n "$parts" ] && [ -n "$unpacked" ] && [ -n "$archive_bytes" ] || die "pack_manifest_unreadable: $manifest_url does not describe the server pack"
if [ "$stamp_sha" = "$want_sha" ] && [ -x "$dest/bin/crucible" ]; then
  say "server-pack: already installed ($want_sha)"
else
  n=0; for part in $parts; do n=$(( n + 1 )); done
  need_kib=$(( (unpacked + archive_bytes + archive_bytes / n) / 1024 ))
  [ "$free_kib" -ge "$need_kib" ] || die "pack_disk: the server pack needs $(( need_kib / 1048576 )) GiB free and there is $(( free_kib / 1048576 )) GiB"
  archive="$downloads/$(printf '%s' "$parts" | awk '{print $1}' | sed 's/\.part[0-9]*$//')"
  rm -f "$archive"; mkdir -p "$downloads"
  for part in $parts; do
    say "server-pack: $part"
    curl -fL --retry 3 --retry-delay 2 --continue-at - --create-dirs -o "$downloads/$part" "https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/$part" || die "pack_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/$part"
    cat "$downloads/$part" >> "$archive" && rm -f "$downloads/$part"
  done
  got_sha="$($SHA_TOOL "$archive" | awk '{print $1}')"
  if [ "$got_sha" != "$want_sha" ]; then rm -f "$archive"; die "pack_sha_mismatch: $archive hashes $got_sha, the manifest says $want_sha"; fi
  rm -rf "$partial" && mkdir -p "$partial"
  tar --zstd -xf "$archive" -C "$partial" || die "pack_unpack_failed: tar would not open $archive"
  "$partial/bin/crucible" --version >/dev/null || die "pack_unpack_failed: $partial/bin/crucible would not run"
  rm -rf "$dest" && mv "$partial" "$dest"
  printf 'sha256=%s\nrelease=%s\n' "$want_sha" "$RELEASE" > "$dest/.pack"
  rm -f "$archive"
fi
CRUCIBLE="$dest/bin/crucible"

# --- init ----------------------------------------------------------------
# write config.toml with a token this side minted
say "init"
if [ -f "$CRUCIBLE_HOME/config.toml" ]; then
  say "init: $CRUCIBLE_HOME/config.toml exists; its token is kept"
else
  TOKEN="$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=')"
  "$CRUCIBLE" 'init' '--token' "$TOKEN" || die "step_failed: init"
fi

# --- service-install -----------------------------------------------------
# write the systemd unit (or the launchd plist) and start it
say "service-install"
"$CRUCIBLE" 'service' 'install' || die "step_failed: service-install"

# --- linger --------------------------------------------------------------
# make the service survive a logout
say "linger"
if [ "$MECHANISM" = systemd ]; then
  if loginctl show-user "$GUEST_USER" -p Linger 2>/dev/null | grep -q 'Linger=yes'; then
    say "linger: already on for $GUEST_USER"
  elif [ "$(id -u)" = 0 ] && loginctl enable-linger "$GUEST_USER"; then
    say "linger: granted to $GUEST_USER"
  elif sudo -n loginctl enable-linger "$GUEST_USER" 2>/dev/null; then
    say "linger: granted to $GUEST_USER with sudo"
  else
    say "linger: NOT granted. The Crucible service will stop when you log out."
    say "linger: run this once, by hand:  sudo loginctl enable-linger $GUEST_USER"
  fi
fi

# --- capability-write ----------------------------------------------------
# record what this card can hold
say "capability-write"
"$CRUCIBLE" 'capability' '--write' || die "step_failed: capability-write"

# --- done ----------------------------------------------------------------
say "installed. Pair an app with the line below."
"$CRUCIBLE" token --url
