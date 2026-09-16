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
# On a rented Linux box with a GPU, where the server is reached over the
# network and the bearer token is the lock:
#
#   curl -fsSL https://github.com/telltaleatheist/crucible/releases/latest/download/install.sh \
#     | sh -s -- --token "$CRUCIBLE_TOKEN" --host 0.0.0.0 --install llm
#
# And to take it off again, keeping the weights:
#
#   curl -fsSL https://github.com/telltaleatheist/crucible/releases/latest/download/install.sh | sh -s -- --uninstall
#
# CRUCIBLE_RELEASE=<version> picks a release other than the one this script
# was cut with. Everything here is idempotent: run it again after a failure.

set -eu

RELEASE="${CRUCIBLE_RELEASE:-0.6.5}"

say() { printf 'crucible: %s\n' "$*"; }
die() { printf 'crucible: %s\n' "$*" >&2; exit 1; }

# --- arguments -----------------------------------------------------------
# The flags a person types. An app never reaches this file: it calls
# `install()`, which walks the SAME step list (PHASE14 4a).
usage() {
  cat <<'USAGE'
crucible install.sh — install or remove a Crucible on this machine.

Install:
  --token <t>          use this bearer token instead of minting one
  --host <addr>        bind address for the server (default 127.0.0.1;
                       a rented box is reached over the network, so it
                       wants 0.0.0.0 — the bearer token is the lock)
  --port <n>           bind port (default 7100)
  --install <type>     also install this job type from its pack. Repeatable.
                       tts names its engine: --install tts=higgs-v3
  --from-source <ref>  build the server from a git ref instead of the
                       published pack (a branch, a tag or a sha)
  --release <version>  install this release rather than the built-in one
  --min-free-gib <n>   refuse unless this much disk is free for the weights

Remove:
  --uninstall          undo the install, in the inverse order
  --purge-weights      with --uninstall: delete the weights too
  --dry-run            with --uninstall: print every step and touch nothing

USAGE
}

UNINSTALL=0
PURGE_WEIGHTS=0
DRY_RUN=0
TOKEN=""
BIND=""
JOB_TYPES=""
FROM_SOURCE=""
MIN_FREE_GIB=""
need() { [ "$1" -ge 2 ] || die "flag_needs_value: $2 takes a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall) UNINSTALL=1 ;;
    --purge-weights) PURGE_WEIGHTS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --token) need $# "--token"; shift; TOKEN="$1" ;;
    --host) need $# "--host"; shift; BIND="$BIND --host $1" ;;
    --port) need $# "--port"; shift; BIND="$BIND --port $1" ;;
    --install) need $# "--install"; shift; JOB_TYPES="$JOB_TYPES $1" ;;
    --from-source) need $# "--from-source"; shift; FROM_SOURCE="$1" ;;
    --release) need $# "--release"; shift; RELEASE="$1" ;;
    --min-free-gib) need $# "--min-free-gib"; shift; MIN_FREE_GIB="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown_flag: $1 is not a flag this installer takes; run with --help" ;;
  esac
  shift
done
if [ "$UNINSTALL" = 0 ] && [ "$PURGE_WEIGHTS" = 1 ]; then
  die "flag_needs_uninstall: --purge-weights deletes weights and only means something with --uninstall"
fi

# --- backend -------------------------------------------------------------
# Two backends and no third. Windows is never one: on Windows this script
# runs INSIDE the WSL2 distro that install.ps1 imported.
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)  BACKEND=cuda-linux; SHA_TOOL="sha256sum";     MECHANISM=systemd ;;
  Darwin/arm64)  BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256"; MECHANISM=launchd ;;
  *) die "unsupported_platform: $(uname -s)/$(uname -m) is not a Crucible backend (cuda-linux on Linux x86_64, mlx-darwin on Apple Silicon)" ;;
esac
say "release $RELEASE, backend $BACKEND"

# --- uninstall -----------------------------------------------------------
# The inverse, and then this script exits: `crucible uninstall` does the
# nine steps inside CRUCIBLE_HOME and this removes the pack it unpacked.
if [ "$UNINSTALL" = 1 ]; then
  say "uninstall"
  CRUCIBLE_HOME="${CRUCIBLE_HOME:-$HOME/.crucible}"
  CRUCIBLE="$CRUCIBLE_HOME/server/bin/crucible"
  if [ ! -x "$CRUCIBLE" ]; then
    die "not_installed: there is no $CRUCIBLE on this machine, so there is no Crucible here for this script to remove. \$CRUCIBLE_HOME names where one would be"
  fi
  UNINSTALL_FLAGS=""
  if [ "$PURGE_WEIGHTS" = 1 ]; then UNINSTALL_FLAGS="$UNINSTALL_FLAGS --purge-weights"; fi
  if [ "$DRY_RUN" = 1 ]; then UNINSTALL_FLAGS="$UNINSTALL_FLAGS --dry-run"; fi
  "$CRUCIBLE" 'uninstall' $UNINSTALL_FLAGS || die "step_failed: uninstall"
  # The pack, which the verb deliberately leaves: it is the interpreter
  # that just ran, and this script is what unpacked it.
  say "server-pack"
  if [ "$DRY_RUN" = 1 ]; then
    say "server-pack: would remove $CRUCIBLE_HOME/server and $CRUCIBLE_HOME/downloads"
    say "home: would remove $CRUCIBLE_HOME if it were then empty"
  else
    rm -rf "$CRUCIBLE_HOME/server" "$CRUCIBLE_HOME/server.partial" "$CRUCIBLE_HOME/downloads"
    say "server-pack: removed $CRUCIBLE_HOME/server"
    if rmdir "$CRUCIBLE_HOME" 2>/dev/null; then
      say "home: removed $CRUCIBLE_HOME"
    else
      say "home: KEPT $CRUCIBLE_HOME — it still holds $(ls -A "$CRUCIBLE_HOME" | tr '\n' ' ')"
      say "home: weights are kept unless --purge-weights; nothing else here was Crucible's to delete"
    fi
  fi
  say "uninstalled."
  exit 0
fi

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

# --- prerequisites -------------------------------------------------------
# Named, and never guessed around. A missing one is a refusal here rather
# than a job type that refuses its first request a week later.
say "prerequisites"
if [ "$BACKEND" = cuda-linux ]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "no_nvidia_smi: there is no nvidia-smi on PATH. cuda-linux runs vLLM, SGLang and torch on an NVIDIA card; a box without the driver is not this backend"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')"
  [ -n "$driver" ] || die "no_nvidia_driver: nvidia-smi is on PATH and named no driver. On a rented GPU box that usually means the image has the CUDA userland and not the kernel module"
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:].')"
  [ -n "$cap" ] || die "no_cuda_arch: this driver would not report a compute capability, so nothing here can say whether the engines will build for this card"
  [ "$cap" -ge 70 ] || die "cuda_arch_too_old: this card reports compute capability $cap (7.0 is the floor: vLLM and SGLang ship no kernels below it, and torch's wheels drop it too). Rent a card at 7.0 or newer"
  card="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  say "prerequisites: $card, driver $driver, compute capability $cap"
fi
case " $JOB_TYPES " in
  *" tts"*|*" asr"*|*" rvc"*|*" align"*|*" denoise"*)
    command -v ffmpeg >/dev/null 2>&1 || die "no_ffmpeg: --install named a job type that decodes audio and there is no ffmpeg on PATH. Install it first; crucible would otherwise install cleanly and refuse its first job" ;;
  *)
    if command -v ffmpeg >/dev/null 2>&1; then
      say "prerequisites: ffmpeg present"
    else
      say "prerequisites: NO ffmpeg on PATH. Nothing asked for today needs it; tts, asr, align, rvc and denoise will refuse until it is there"
    fi ;;
esac
if [ -n "$MIN_FREE_GIB" ]; then
  have_gib=$(( free_kib / 1048576 ))
  [ "$have_gib" -ge "$MIN_FREE_GIB" ] || die "disk_too_small: $CRUCIBLE_HOME has ${have_gib} GiB free and --min-free-gib asked for $MIN_FREE_GIB"
fi
say "prerequisites: $(( free_kib / 1048576 )) GiB free at $CRUCIBLE_HOME. Weights are pulled later and priced then — a 9B model is ~18 GiB, a Higgs voice ~8.5 GiB"

# --- server-pack ---------------------------------------------------------
# download, verify and unpack the server pack — the interpreter comes WITH it
say "server-pack"
if [ -n "$FROM_SOURCE" ]; then
  say "server-pack: --from-source $FROM_SOURCE, building instead of downloading"
  command -v git >/dev/null 2>&1 || die "guest_missing_tool: --from-source needs git"
  command -v python3 >/dev/null 2>&1 || die "guest_missing_tool: --from-source needs a python3 on this machine to build the venv with. The published pack brings its own interpreter; a source build cannot"
  dest="$CRUCIBLE_HOME/server"
  src="$CRUCIBLE_HOME/src"
  rm -rf "$src"
  git clone --filter=blob:none "https://github.com/telltaleatheist/crucible" "$src" || die "from_source_clone_failed: https://github.com/telltaleatheist/crucible"
  git -C "$src" checkout --detach "$FROM_SOURCE" || die "from_source_ref_unknown: the checkout has no ref called $FROM_SOURCE"
  partial="$dest.partial"
  rm -rf "$partial"
  python3 -m venv "$partial" || die "from_source_venv_failed: python3 -m venv would not make $partial"
  "$partial/bin/python" -m pip install --upgrade pip setuptools wheel || die "from_source_install_failed: pip would not update itself in $partial"
  "$partial/bin/python" -m pip install "$src" || die "from_source_install_failed: pip would not install $src into $partial"
  if [ "$(uname -s)" = Darwin ]; then "$partial/bin/python" -m pip install pystray pillow || die "from_source_install_failed: desktop packages could not be installed"; fi
  "$partial/bin/python" -c 'from pathlib import Path; import sys; from crucible.envpack import relocate_console_scripts; relocate_console_scripts(Path(sys.argv[1]))' "$partial" || die "from_source_install_failed: console scripts could not be relocated"
  "$partial/bin/crucible" --version >/dev/null || die "from_source_install_failed: $partial/bin/crucible would not run"
  activate_crucible_pack() {
    _crucible_dest="$dest"; _crucible_partial="$partial"; _crucible_previous="$_crucible_dest.previous"
    if [ -e "$_crucible_previous" ]; then echo "upgrade_recovery_required: $_crucible_previous was preserved from an interrupted upgrade" >&2; return 1; fi
    if [ -e "$_crucible_dest" ]; then
      "$_crucible_partial/bin/crucible" local shutdown || return 1
      mv "$_crucible_dest" "$_crucible_previous" || return 1
    fi
    if mv "$_crucible_partial" "$_crucible_dest" && "$_crucible_dest/bin/crucible" --version; then
      rm -rf "$_crucible_previous" || return 1
    else
      if [ -e "$_crucible_dest" ] && [ ! -e "$_crucible_partial" ]; then mv "$_crucible_dest" "$_crucible_partial" || return 1; fi
      if [ -e "$_crucible_previous" ]; then mv "$_crucible_previous" "$_crucible_dest" || return 1; fi
      echo "upgrade_activation_failed: the previous runtime was preserved" >&2; return 1
    fi
  }
  activate_crucible_pack || die "from_source_install_failed: runtime activation failed; previous runtime preserved"
  printf 'sha256=%s\nrelease=%s\n' "from-source" "$(git -C "$src" rev-parse HEAD)" > "$dest/.pack"
  CRUCIBLE="$dest/bin/crucible"
  say "server-pack: built $("$CRUCIBLE" --version) from $(git -C "$src" rev-parse --short HEAD)"
else
  dest="$CRUCIBLE_HOME/server"
  partial="$dest.partial"
  downloads="$CRUCIBLE_HOME/downloads"
  manifest_url="https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/envpacks.json"
  manifest="$(curl -fsSL --retry 3 "$manifest_url")" || die "pack_manifest_unreadable: could not fetch $manifest_url"
  pack="$(printf '%s' "$manifest" | awk -v RS='}' -v b="$BACKEND" '$0 ~ /"name"[[:space:]]*:[[:space:]]*"server"/ && $0 ~ ("\"backend\"[[:space:]]*:[[:space:]]*\"" b "\"")' | tr -d '\r\n')"
  [ -n "$pack" ] || die "pack_not_published: the $RELEASE release publishes no server pack for $BACKEND"
  want_sha="$(printf '%s' "$pack" | sed -n 's/.*"sha256"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p')"
  unpacked="$(printf '%s' "$pack" | sed -n 's/.*"unpacked_bytes"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p')"
  archive_bytes="$(printf '%s' "$pack" | sed -n 's/.*"bytes"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p')"
  parts="$(printf '%s' "$pack" | sed -n 's/.*"parts"[[:space:]]*:[[:space:]]*\[\([^]]*\)\].*/\1/p' | tr -d '[:space:]"' | tr ',' ' ')"
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
    activate_crucible_pack() {
    _crucible_dest="$dest"; _crucible_partial="$partial"; _crucible_previous="$_crucible_dest.previous"
    if [ -e "$_crucible_previous" ]; then echo "upgrade_recovery_required: $_crucible_previous was preserved from an interrupted upgrade" >&2; return 1; fi
    if [ -e "$_crucible_dest" ]; then
      "$_crucible_partial/bin/crucible" local shutdown || return 1
      mv "$_crucible_dest" "$_crucible_previous" || return 1
    fi
    if mv "$_crucible_partial" "$_crucible_dest" && "$_crucible_dest/bin/crucible" --version; then
      rm -rf "$_crucible_previous" || return 1
    else
      if [ -e "$_crucible_dest" ] && [ ! -e "$_crucible_partial" ]; then mv "$_crucible_dest" "$_crucible_partial" || return 1; fi
      if [ -e "$_crucible_previous" ]; then mv "$_crucible_previous" "$_crucible_dest" || return 1; fi
      echo "upgrade_activation_failed: the previous runtime was preserved" >&2; return 1
    fi
  }
  activate_crucible_pack || die "pack_activation_failed: the previous runtime was preserved"
    printf 'sha256=%s\nrelease=%s\n' "$want_sha" "$RELEASE" > "$dest/.pack"
    rm -f "$archive"
  fi
  CRUCIBLE="$dest/bin/crucible"
fi

# --- init ----------------------------------------------------------------
# write config.toml with a token this side minted
say "init"
if [ -f "$CRUCIBLE_HOME/config.toml" ]; then
  say "init: $CRUCIBLE_HOME/config.toml exists; its token is kept"
else
  if [ -z "${TOKEN:-}" ]; then
    TOKEN="$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=')"
  fi
  "$CRUCIBLE" 'init' '--token' "$TOKEN" $BIND || die "step_failed: init"
fi

# --- install-job-types ---------------------------------------------------
# `--install <type>`, from the published packs. Empty on a bare run, which
# is 4a: a Crucible that serves nothing until somebody asks.
if [ -n "$JOB_TYPES" ]; then
  for entry in $JOB_TYPES; do
    case "$entry" in
      *=*) type="${entry%%=*}"; engine="${entry#*=}" ;;
      *)   type="$entry"; engine="" ;;
    esac
    say "install-$type"
    if [ -n "$engine" ]; then
      "$CRUCIBLE" 'install' "$type" '--narrator-engine' "$engine" || die "step_failed: install-$type"
    else
      "$CRUCIBLE" 'install' "$type" || die "step_failed: install-$type"
    fi
  done
fi

# --- service-install -----------------------------------------------------
# write the systemd unit (or the launchd plist) and start it
say "service-install"
"$CRUCIBLE" 'service' 'install' || die "step_failed: service-install"

# --- local-register ------------------------------------------------------
# publish and configure the local Crucible register
say "local-register"
"$CRUCIBLE" 'local' 'register' || die "step_failed: local-register"

# --- local-install-cli ---------------------------------------------------
# publish and configure the local Crucible install-cli
say "local-install-cli"
"$CRUCIBLE" 'local' 'install-cli' || die "step_failed: local-install-cli"

# --- local-install-desktop -----------------------------------------------
# publish and configure the local Crucible install-desktop
say "local-install-desktop"
"$CRUCIBLE" 'local' 'install-desktop' || die "step_failed: local-install-desktop"

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

# --- local-start ---------------------------------------------------------
# wait for the paired engine to answer with authenticated identity
say "local-start"
"$CRUCIBLE" 'local' 'start' '--json' || die "step_failed: local-start"

# --- done ----------------------------------------------------------------
say "installed. Pair an app with the line below."
"$CRUCIBLE" token --url
