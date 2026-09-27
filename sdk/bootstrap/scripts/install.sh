#!/bin/sh

set -eu

RELEASE="${CRUCIBLE_RELEASE:-}"

say() { printf 'crucible: %s\n' "$*"; }
die() { printf 'crucible: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
crucible install.sh — install or remove a Crucible on this machine.

Install:
  --token <t>          use this bearer token instead of minting one
  --host <addr>        bind address for the server (default 127.0.0.1;
                       a rented box is reached over the network, so it
                       wants 0.0.0.0 — the bearer token is the lock)
  --port <n>           bind port (default 7100)
  --install <type>     also install this job type, from its recipe. Repeatable.
                       tts names its engine: --install tts=higgs-v3
  --from-source <ref>  install the server from a git ref instead of the
                       release's wheel (a branch, a tag or a sha)
  --release <version>  install this exact release rather than the channel's
                       latest. The one override; it still downloads from that
                       release, so it is a pin and not an offline install.
  --rollback-to <ver>  an operator rollback: install this EXACT older release
                       over a newer one already on this disk. Must name the
                       same version as --release; there is no other way down.
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
ROLLBACK_TO=""
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
    --rollback-to) need $# "--rollback-to"; shift; ROLLBACK_TO="$1" ;;
    --min-free-gib) need $# "--min-free-gib"; shift; MIN_FREE_GIB="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown_flag: $1 is not a flag this installer takes; run with --help" ;;
  esac
  shift
done
if [ "$UNINSTALL" = 0 ] && [ "$PURGE_WEIGHTS" = 1 ]; then
  die "flag_needs_uninstall: --purge-weights deletes weights and only means something with --uninstall"
fi
if [ "$UNINSTALL" = 1 ] && [ -n "$ROLLBACK_TO" ]; then
  die "flag_needs_install: --rollback-to names a release to INSTALL and means nothing with --uninstall"
fi

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)  BACKEND=cuda-linux; SHA_TOOL="sha256sum";     MECHANISM=systemd ;;
  Darwin/arm64)  BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256"; MECHANISM=launchd ;;
  *) die "unsupported_platform: $(uname -s)/$(uname -m) is not a Crucible backend (cuda-linux on Linux x86_64, mlx-darwin on Apple Silicon)" ;;
esac
newest_release() {
  curl -fsSL --retry 3 -H "Accept: application/vnd.github+json" "https://api.github.com/repos/telltaleatheist/crucible/releases/latest" |
    grep -o '"tag_name": *"[^"]*"' | head -n 1 | cut -d'"' -f4
}
if [ "$UNINSTALL" = 1 ]; then
  say "backend $BACKEND, uninstalling"
else
  if [ -z "$RELEASE" ]; then
    tag="$(newest_release)" || tag=""
    RELEASE="${tag#v}"
  fi
  [ -n "$RELEASE" ] || die "release_channel_unreadable: could not read the release channel at https://api.github.com/repos/telltaleatheist/crucible/releases/latest -- name a release with --release <version> or CRUCIBLE_RELEASE=<version>"
  [ -z "$ROLLBACK_TO" ] || [ "$ROLLBACK_TO" = "$RELEASE" ] || die "rollback_version_mismatch: --rollback-to names $ROLLBACK_TO and the release being installed is $RELEASE; a rollback names the exact Crucible you want back"
  say "release $RELEASE, backend $BACKEND"
fi

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
  say "server"
  if [ "$DRY_RUN" = 1 ]; then
    say "server: would remove $CRUCIBLE_HOME/server and $CRUCIBLE_HOME/downloads"
    say "home: would remove $CRUCIBLE_HOME if it were then empty"
  else
    rm -rf "$CRUCIBLE_HOME/server" "$CRUCIBLE_HOME/server.partial" "$CRUCIBLE_HOME/downloads"
    say "server: removed $CRUCIBLE_HOME/server"
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

say "host-facts"
crucible_probe() {
  h="${CRUCIBLE_HOME:-$HOME/.crucible}"; echo "home=$h"; echo "user=$(id -un)"; d="$h"; while [ ! -d "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done; echo "free_kib=$(df -Pk "$d" | awk 'NR==2 {print $4}')"; for t in curl tar; do command -v "$t" >/dev/null 2>&1 || echo "missing=$t"; done; c="$h/server/bin/crucible"; if test -x "$c"; then echo "crucible=$c"; echo "version=$("$c" --version 2>&1 | head -1)"; fi; s="$h/server/.crucible"; if test -f "$s"; then cat "$s"; fi; exit 0
}
probe_out="$(crucible_probe)"
CRUCIBLE_HOME="$(printf '%s\n' "$probe_out" | sed -n 's/^home=//p')"
GUEST_USER="$(printf '%s\n' "$probe_out" | sed -n 's/^user=//p')"
free_kib="$(printf '%s\n' "$probe_out" | sed -n 's/^free_kib=//p')"
stamp_python_sha="$(printf '%s\n' "$probe_out" | sed -n 's/^python_sha256=//p')"
stamp_release="$(printf '%s\n' "$probe_out" | sed -n 's/^release=//p')"
missing="$(printf '%s\n' "$probe_out" | sed -n 's/^missing=//p' | tr '\n' ' ')"
if [ -n "$missing" ]; then die "guest_missing_tool: this machine has no $missing; the interpreter is fetched with curl and unpacked with tar"; fi

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
if [ "$BACKEND" = cuda-linux ]; then
  say "prerequisites: ffmpeg is Crucible's own; the first job type that decodes audio places it in $CRUCIBLE_HOME/tools/bin"
else
  case " $JOB_TYPES " in
    *" tts"*|*" asr"*|*" rvc"*|*" align"*|*" denoise"*)
      command -v ffmpeg >/dev/null 2>&1 || die "no_ffmpeg: --install named a job type that decodes audio and this Mac has no ffmpeg on PATH. Install Homebrew's first (brew install ffmpeg); crucible would otherwise install cleanly and refuse its first job" ;;
    *)
      if command -v ffmpeg >/dev/null 2>&1; then
        say "prerequisites: ffmpeg present"
      else
        say "prerequisites: NO ffmpeg on PATH. Nothing asked for today needs it; on this Mac tts, asr, align, rvc and denoise will refuse until Homebrew's is installed (brew install ffmpeg)"
      fi ;;
  esac
fi
if [ -n "$MIN_FREE_GIB" ]; then
  have_gib=$(( free_kib / 1048576 ))
  [ "$have_gib" -ge "$MIN_FREE_GIB" ] || die "disk_too_small: $CRUCIBLE_HOME has ${have_gib} GiB free and --min-free-gib asked for $MIN_FREE_GIB"
fi
if [ -n "${WSL_DISTRO_NAME:-}" ]; then
  say "prerequisites: WSL's virtual disk can grow to $(( free_kib / 1048576 )) GiB; the real limit is the free space on the Windows drive it lives on"
else
  say "prerequisites: $(( free_kib / 1048576 )) GiB free at $CRUCIBLE_HOME. Weights are pulled later and priced then — a 9B model is ~18 GiB, a Higgs voice ~8.5 GiB"
fi

say "server"
dest="$CRUCIBLE_HOME/server"
partial="$dest.partial"
downloads="$CRUCIBLE_HOME/downloads"
wheel=""
case "$BACKEND" in
  cuda-linux) py_asset='cpython-3.11.16+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz'; py_sha='faa0758583a63f14c5eee516af82738403b59c13edda6fc0a21d953febd89eed'; py_version='3.11.16'; py_url='https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz' ;;
  mlx-darwin) py_asset='cpython-3.11.16+20260901-aarch64-apple-darwin-install_only.tar.gz'; py_sha='50424fa409e8ae84b82a3052522f64695b47dff2158b70bb7358e0ebd6c085c9'; py_version='3.11.16'; py_url='https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16+20260901-aarch64-apple-darwin-install_only.tar.gz' ;;
  *) die "unsupported_platform: no interpreter is pinned for $BACKEND" ;;
esac
crucible_older() {
  awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".");
    for (i = 1; i <= 3; i++) { if ((x[i]+0) < (y[i]+0)) exit 0; if ((x[i]+0) > (y[i]+0)) exit 1 } exit 1 }'
}
if [ -n "$stamp_release" ] && crucible_older "$RELEASE" "$stamp_release"; then
  [ "$ROLLBACK_TO" = "$RELEASE" ] || die "install_would_downgrade: $dest is the $stamp_release release and this would install $RELEASE over it. Nothing was downloaded. An operator who means to go back names the version: --rollback-to $RELEASE"
fi
quiesced=0
crucible_quiesce() {
  if [ ! -x "$dest/bin/crucible" ]; then quiesced=1; return 0; fi
  if [ -n "$wheel" ] && [ -f "$downloads/$wheel" ]; then
    stage="$downloads/stage"
    rm -rf "$stage"
    if "$dest/bin/python3" -m pip install --quiet --no-deps --no-input --target "$stage" "$downloads/$wheel" >/dev/null 2>&1 \
      && PYTHONPATH="$stage" "$dest/bin/python3" -m crucible.cli local shutdown; then
      rm -rf "$stage"; quiesced=1; say "server: the running server was stopped by $RELEASE's own code"; return 0
    fi
    rm -rf "$stage"
    say "server: $RELEASE's code could not stop the running server; asking the installed release to stop it"
  fi
  "$dest/bin/crucible" local shutdown || return 1
  quiesced=1
}
crucible_quiesce_after() {
  if [ "$quiesced" = 1 ]; then return 0; fi
  say "server: stopping the old server with the release just installed"
  "$dest/bin/crucible" local shutdown || die "upgrade_stop_failed: the Crucible server that was already running could not be stopped, by the old release or by $RELEASE, so the new one cannot take its place yet. $RELEASE is installed and takes over when the old server next stops: when this machine restarts, or on Windows when Crucible next starts"
  quiesced=1
}
if [ -z "$FROM_SOURCE" ]; then
  wheel="crucible-$RELEASE-py3-none-any.whl"
  say "server: $wheel"
  mkdir -p "$downloads"; rm -f "$downloads/$wheel"
  curl -fL --retry 3 --retry-delay 2 --create-dirs -o "$downloads/$wheel" "https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/$wheel" || die "runtime_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/$wheel"
  want_sha="$(curl -fsSL --retry 3 "https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/crucible-$RELEASE-py3-none-any.whl.sha256" | awk '{print $1}')" || die "runtime_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/crucible-$RELEASE-py3-none-any.whl.sha256"
  case "$want_sha" in *[!0-9a-f]*|"") die "runtime_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$RELEASE/crucible-$RELEASE-py3-none-any.whl.sha256 is not a sha256" ;; esac
  got_sha="$($SHA_TOOL "$downloads/$wheel" | awk '{print $1}')"
  if [ "$got_sha" != "$want_sha" ]; then rm -f "$downloads/$wheel"; die "runtime_sha_mismatch: $wheel hashes $got_sha, the release says $want_sha. The download was deleted"; fi
fi
if [ -n "$stamp_release" ] && [ "$stamp_python_sha" = "$py_sha" ] && [ -x "$dest/bin/python3" ]; then
  say "server: python $py_version is already at $dest"
else
  say "server: python $py_version from python-build-standalone"
  mkdir -p "$downloads"; rm -f "$downloads/$py_asset"
  total="$(curl -fsSLI -m 20 "$py_url" | tr -d '\r' | awk 'tolower($1) == "content-length:" { print $2 }' | tail -n 1)"
  case "$total" in ''|*[!0-9]*) total=null ;; esac
  curl -fL --retry 3 --retry-delay 2 --create-dirs -o "$downloads/$py_asset" "$py_url" &
  fetch_pid=$!
  while kill -0 "$fetch_pid" 2>/dev/null; do
    got=0
    if [ -f "$downloads/$py_asset" ]; then got="$(wc -c < "$downloads/$py_asset" | tr -d ' ')"; fi
    printf 'crucible-progress {"bytes_done": %s, "bytes_total": %s, "file": "%s"}\n' "$got" "$total" "$py_asset"
    sleep 1
  done
  wait "$fetch_pid" || die "runtime_download_failed: $py_url"
  got_sha="$($SHA_TOOL "$downloads/$py_asset" | awk '{print $1}')"
  if [ "$got_sha" != "$py_sha" ]; then rm -f "$downloads/$py_asset"; die "runtime_sha_mismatch: $py_asset hashes $got_sha and this installer pins $py_sha. The download was deleted"; fi
  rm -rf "$partial" && mkdir -p "$partial"
  tar -xzf "$downloads/$py_asset" -C "$partial" || die "runtime_unpack_failed: tar would not open $downloads/$py_asset"
  [ -x "$partial/python/bin/python3" ] || die "runtime_unpack_failed: $py_asset unpacked without a python/bin/python3"
  activate_crucible_runtime() {
  _crucible_dest="$dest"; _crucible_partial="$partial/python"; _crucible_previous="$_crucible_dest.previous"
  if [ -e "$_crucible_previous" ]; then echo "upgrade_recovery_required: $_crucible_previous was preserved from an interrupted upgrade" >&2; return 1; fi
  if [ -e "$_crucible_dest" ]; then
    crucible_quiesce || return 1
    mv "$_crucible_dest" "$_crucible_previous" || return 1
  fi
  if mv "$_crucible_partial" "$_crucible_dest" && "$_crucible_dest/bin/python3" --version; then
    rm -rf "$_crucible_previous" || return 1
  else
    if [ -e "$_crucible_dest" ] && [ ! -e "$_crucible_partial" ]; then mv "$_crucible_dest" "$_crucible_partial" || return 1; fi
    if [ -e "$_crucible_previous" ]; then mv "$_crucible_previous" "$_crucible_dest" || return 1; fi
    echo "upgrade_activation_failed: the previous runtime was preserved" >&2; return 1
  fi
}
activate_crucible_runtime || die "runtime_unpack_failed: the previous runtime was preserved"
  rm -rf "$partial" "$downloads/$py_asset"
fi
if [ -n "$FROM_SOURCE" ]; then
  say "server: --from-source $FROM_SOURCE, installing from a checkout instead of the wheel"
  command -v git >/dev/null 2>&1 || die "guest_missing_tool: --from-source needs git"
  src="$CRUCIBLE_HOME/src"
  rm -rf "$src"
  git clone --filter=blob:none "https://github.com/telltaleatheist/crucible" "$src" || die "from_source_clone_failed: https://github.com/telltaleatheist/crucible"
  git -C "$src" checkout --detach "$FROM_SOURCE" || die "from_source_ref_unknown: the checkout has no ref called $FROM_SOURCE"
  crucible_quiesce || say "server: the running server did not stop; installing the checkout and stopping it with that"
  "$dest/bin/python3" -m pip install --upgrade --no-input "$src" || die "from_source_install_failed: pip would not install $src into $dest"
  if [ "$(uname -s)" = Darwin ]; then "$dest/bin/python3" -m pip install pystray pillow || die "from_source_install_failed: desktop packages could not be installed"; fi
  printf 'python_sha256=%s\npython_version=%s\nrelease=%s\n' "$py_sha" "$py_version" "$(git -C "$src" rev-parse HEAD)" > "$dest/.crucible"
  CRUCIBLE="$dest/bin/crucible"
  crucible_quiesce_after
  say "server: installed $("$CRUCIBLE" --version) from $(git -C "$src" rev-parse --short HEAD)"
else
  crucible_quiesce || say "server: the running server did not stop; installing $RELEASE and stopping it with that"
  "$dest/bin/python3" -m pip install --upgrade --no-input "$downloads/$wheel" || die "runtime_install_failed: pip would not install $wheel"
  if [ "$(uname -s)" = Darwin ]; then "$dest/bin/python3" -m pip install pystray pillow || die "runtime_install_failed: the desktop packages would not install"; fi
  rm -f "$downloads/$wheel"
  printf 'python_sha256=%s\npython_version=%s\nrelease=%s\n' "$py_sha" "$py_version" "$RELEASE" > "$dest/.crucible"
  CRUCIBLE="$dest/bin/crucible"
  "$CRUCIBLE" --version >/dev/null || die "runtime_install_failed: $CRUCIBLE would not run"
  crucible_quiesce_after
fi

say "init"
if [ -f "$CRUCIBLE_HOME/config.toml" ]; then
  say "init: $CRUCIBLE_HOME/config.toml exists; its token is kept"
else
  if [ -z "${TOKEN:-}" ]; then
    TOKEN="$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=')"
  fi
  "$CRUCIBLE" 'init' '--token' "$TOKEN" $BIND || die "step_failed: init"
fi

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

say "env-patch-llm"
"$CRUCIBLE" 'env' 'patch' 'llm' || die "step_failed: env-patch-llm"

say "service-install"
"$CRUCIBLE" 'service' 'install' || die "step_failed: service-install"

say "local-register"
"$CRUCIBLE" 'local' 'register' || die "step_failed: local-register"

say "local-install-cli"
"$CRUCIBLE" 'local' 'install-cli' || die "step_failed: local-install-cli"

say "local-install-desktop"
"$CRUCIBLE" 'local' 'install-desktop' || die "step_failed: local-install-desktop"

say "linger"
if [ "$MECHANISM" = systemd ] && ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
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

say "capability-write"
"$CRUCIBLE" 'capability' '--write' || die "step_failed: capability-write"

say "local-start"
"$CRUCIBLE" 'local' 'start' '--json' || die "step_failed: local-start"

say "installed. Pair an app with the line below."
"$CRUCIBLE" token --url
