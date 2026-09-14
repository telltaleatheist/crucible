#!/usr/bin/env bash
# Build `crucible-rootfs-<version>.tar.zst` — the WSL2 image Crucible owns.
#
# PHASE14-ENVPACKS.md 4b defines what is in it and WHY, and this script is the
# one thing that produces it. Run on ubuntu-latest in CI, on the tag, beside
# the env-pack jobs; the two assets it writes go on the release:
#
#     crucible-rootfs-<version>.tar.zst          the image
#     crucible-rootfs-<version>.tar.zst.sha256   one line: the digest
#
#     ./build-rootfs.sh 0.6.0 [outdir]
#
# It cannot be built on Windows, which is why it is a release asset rather than
# something bootstrap makes: `wsl --import` needs a Linux filesystem tarball
# with Linux ownership and permissions intact.
#
# WHAT IS IN IT, and nothing else:
#   - Ubuntu 24.04, minimal
#   - systemd + dbus, because `[boot] systemd=true` needs an init to start
#   - curl, ca-certificates, tar, zstd: exactly what the pack install uses
#   - a user `crucible`, no password, passwordless sudo, and `[user]
#     default=crucible` so `wsl -d crucible` lands as that user with no
#     first-run dialog for an app to answer
#   - `/etc/wsl.conf` beginning with the marker `# crucible-rootfs`, which is
#     how bootstrap tells OUR distro from one wearing the same name
#
# What is NOT in it: Python (the server pack carries its own), CUDA (WSL
# exposes the Windows driver), and every convenience a person's own distro has.
set -euo pipefail

VERSION="${1:?usage: build-rootfs.sh <version> [outdir]}"
OUT="${2:-dist}"
IMAGE="crucible-rootfs:${VERSION}"
ASSET="crucible-rootfs-${VERSION}.tar.zst"

command -v docker >/dev/null || { echo "build-rootfs: docker is required" >&2; exit 1; }
command -v zstd >/dev/null || { echo "build-rootfs: zstd is required" >&2; exit 1; }

mkdir -p "$OUT"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The marker line is FIRST, and `distro.ts` WSL_CONF_TEXT is the same text.
cat > "$work/wsl.conf" <<'CONF'
# crucible-rootfs
[boot]
systemd=true
[user]
default=crucible
CONF

cat > "$work/Dockerfile" <<'DOCKER'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      systemd systemd-sysv dbus \
      curl ca-certificates tar zstd sudo \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /var/log/* /tmp/*
RUN useradd --create-home --shell /bin/bash crucible \
 && passwd --delete crucible \
 && printf 'crucible ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/crucible \
 && chmod 0440 /etc/sudoers.d/crucible
COPY wsl.conf /etc/wsl.conf
DOCKER

echo "build-rootfs: building $IMAGE"
docker build --platform linux/amd64 -t "$IMAGE" "$work"

echo "build-rootfs: exporting the filesystem"
container="$(docker create --platform linux/amd64 "$IMAGE" /bin/true)"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true; rm -rf "$work"' EXIT
docker export "$container" | zstd -19 -T0 -o "$OUT/$ASSET"

# The digest asset install.ps1 and distro.ts check the download against. One
# line, the digest first, so `awk '{print $1}'` and PowerShell's split agree.
( cd "$OUT" && sha256sum "$ASSET" > "$ASSET.sha256" )

echo "build-rootfs: wrote $OUT/$ASSET"
cat "$OUT/$ASSET.sha256"

# A rootfs that cannot be imported is not an asset. CI cannot run `wsl
# --import`, so the check is the one thing that can be checked here: the
# archive opens, and the two files that make it ours are in it.
echo "build-rootfs: verifying the archive"
zstd -dc "$OUT/$ASSET" | tar -tf - etc/wsl.conf home/crucible >/dev/null
zstd -dc "$OUT/$ASSET" | tar -xO -f - etc/wsl.conf | grep -q '^# crucible-rootfs' \
  || { echo "build-rootfs: /etc/wsl.conf has no marker — bootstrap would re-import this forever" >&2; exit 1; }
echo "build-rootfs: ok"
