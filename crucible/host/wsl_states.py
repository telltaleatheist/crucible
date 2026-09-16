# GENERATED FILE — do not edit.
# Written by sdk/bootstrap/scripts/gen-install-scripts.ts from src/steps.ts
# and src/wsl-states.ts, so a hand install and an app-driven install cannot
# differ (PHASE14-ENVPACKS.md 4a). Regenerate: npm run gen:install
#
# The WSL state table of PHASE14-ENVPACKS.md 4c, as DATA, for `crucible host`
# (PHASE15-HOST.md 4.3). The ORDER is the order they are tried: deepest cause
# first, so "virtualization is off in the firmware" is never reported as "WSL
# is not installed".
#
# `sentence`, `action_text`, `action_url` and `probe_argv` carry `{said}` /
# `{app_distro}` / `{guest_user}` / `{release}` / `{required}` / `{free}`
# where the TypeScript interpolated something the caller measures. The
# PREDICATES are not here: they are code, and they live in
# `crucible/host/wslstate.py`, one per code, tied to this file by a test.

from __future__ import annotations

from dataclasses import dataclass

#: The distro Crucible owns. One name, and its owner is sdk/bootstrap/src/distro.ts.
CRUCIBLE_DISTRO = "crucible"
ROOTFS_ASSET_TEMPLATE = "crucible-rootfs-{version}.tar.zst"
RELEASE_REPOSITORY = "telltaleatheist/crucible"

#: The line /etc/wsl.conf carries in the Crucible rootfs and nowhere else.
WSL_CONF_MARKER = "# crucible-rootfs"

#: /etc/wsl.conf, exactly as the rootfs ships it and as the repair writes it.
WSL_CONF_TEXT = "# crucible-rootfs\n[boot]\nsystemd=true\n[user]\ndefault=crucible\n"


@dataclass(frozen=True)
class WslStateDef:
    """One row of 4c. `optional` rows are only probed when the caller asks."""

    code: str
    probe: str
    probe_argv: tuple[str, ...]
    sentence: str
    action_kind: str
    action_argv: tuple[str, ...]
    action_text: str
    action_url: str
    optional: bool


WSL_STATES: tuple[WslStateDef, ...] = (
    WslStateDef(
        code="virtualization_disabled",
        probe="wsl-status",
        probe_argv=("wsl.exe", "--status", ),
        sentence="Windows cannot start a virtual machine: {said}",
        action_kind="instruct",
        action_argv=(),
        action_text="Virtualization is turned off in this machine's firmware. Restart, open the BIOS/UEFI setup (usually Del or F2 during boot), and enable Intel VT-x (Intel) or SVM Mode (AMD). Then run Enable WSL again.",
        action_url="",
        optional=False,
    ),
    WslStateDef(
        code="wsl_missing",
        probe="wsl-status",
        probe_argv=("wsl.exe", "--status", ),
        sentence="This machine has no wsl.exe: the Windows Subsystem for Linux has never been enabled.",
        action_kind="run-elevated",
        action_argv=("wsl.exe", "--install", "--no-distribution", ),
        action_text="",
        action_url="",
        optional=False,
    ),
    WslStateDef(
        code="wsl1_only",
        probe="wsl-status",
        probe_argv=("wsl.exe", "--status", ),
        sentence="WSL is set to version 1, which has no GPU. Crucible needs WSL2.",
        action_kind="run",
        action_argv=("wsl.exe", "--set-default-version", "2", ),
        action_text="",
        action_url="",
        optional=False,
    ),
    WslStateDef(
        code="no_crucible_distro",
        probe="wsl-list",
        probe_argv=("wsl.exe", "-l", "-v", ),
        sentence="Crucible has no Linux of its own on this machine yet (the \"crucible\" distribution). Installing one takes a download and touches nothing you already have.",
        action_kind="run",
        action_argv=("wsl.exe", "--import", "crucible", "<install dir>", "<rootfs>", "--version", "2", ),
        action_text="",
        action_url="",
        optional=False,
    ),
    WslStateDef(
        code="distro_not_systemd",
        probe="wsl-conf",
        probe_argv=("wsl.exe", "-d", "crucible", "-u", "root", "--exec", "bash", "-c", "test -f /etc/wsl.conf && cat /etc/wsl.conf || true", ),
        sentence="The \"crucible\" distribution is not running systemd, so the Crucible service cannot start in it. It is ours: this is repaired without asking.",
        action_kind="run",
        action_argv=("wsl.exe", "--terminate", "crucible", ),
        action_text="",
        action_url="",
        optional=False,
    ),
    WslStateDef(
        code="foreign_distro_not_systemd",
        probe="app-distro-conf",
        probe_argv=("wsl.exe", "-d", "{app_distro}", "-u", "root", "--exec", "bash", "-c", "test -f /etc/wsl.conf && cat /etc/wsl.conf || true", ),
        sentence="\"{app_distro}\" is the distribution this app was told to use, and it is not running systemd. Crucible will not write to a distribution you chose without being asked.",
        action_kind="instruct",
        action_argv=(),
        action_text="Add [boot] systemd=true to /etc/wsl.conf in \"{app_distro}\" and run wsl --terminate {app_distro}, or let Crucible import its own distribution instead.",
        action_url="",
        optional=True,
    ),
    WslStateDef(
        code="guest_no_network",
        probe="guest-network",
        probe_argv=("wsl.exe", "-d", "crucible", "--exec", "curl", "-fsS", "-m", "20", "-o", "/dev/null", "https://github.com/telltaleatheist/crucible/releases/download/v{release}/envpacks.json", ),
        sentence="The \"crucible\" distribution cannot reach https://github.com/telltaleatheist/crucible/releases/download/v{release}/envpacks.json ({said}). A VPN or a proxy on this machine usually explains it; there is nothing to install until it can.",
        action_kind="link",
        action_argv=(),
        action_text="",
        action_url="https://github.com/telltaleatheist/crucible/releases/download/v{release}/envpacks.json",
        optional=True,
    ),
    WslStateDef(
        code="pack_disk",
        probe="guest-disk",
        probe_argv=("wsl.exe", "-d", "crucible", "--exec", "bash", "-c", "df -Pk \"$HOME\" | awk 'NR==2 {print $4}'", ),
        sentence="This install needs {required} free and the \"crucible\" distribution has {free}. Nothing has been downloaded.",
        action_kind="instruct",
        action_argv=(),
        action_text="Free some space on the drive WSL keeps its disk on, then try again.",
        action_url="",
        optional=True,
    ),
    WslStateDef(
        code="linger_unreadable",
        probe="guest-root",
        probe_argv=("wsl.exe", "-d", "crucible", "-u", "root", "--exec", "id", "-u", ),
        sentence="The \"crucible\" distribution will not let Crucible in as root ({said}), so it cannot make the server survive a logout.",
        action_kind="instruct",
        action_argv=(),
        action_text="Run this yourself inside the distribution: sudo loginctl enable-linger {guest_user}",
        action_url="",
        optional=False,
    ),
    WslStateDef(
        code="wsl_ready",
        probe="wsl-list",
        probe_argv=("wsl.exe", "-l", "-v", ),
        sentence="WSL2 is ready and the \"crucible\" distribution is there.",
        action_kind="instruct",
        action_argv=(),
        action_text="Nothing to do.",
        action_url="",
        optional=False,
    ),
)

#: Every code, in table order. `crucible/host/wslstate.py` must have a
#: predicate for each, and no others.
WSL_STATE_CODES: tuple[str, ...] = tuple(state.code for state in WSL_STATES)
