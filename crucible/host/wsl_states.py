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
# `{app_distro}` / `{release}` / `{required}` / `{free}`
# where the TypeScript interpolated something the caller measures. The
# PREDICATES are not here: they are code, and they live in
# `crucible/host/wslstate.py`, one per code, tied to this file by a test.

from __future__ import annotations

from dataclasses import dataclass

#: The distro Crucible owns. One name, and its owner is sdk/bootstrap/src/distro.ts.
CRUCIBLE_DISTRO = "crucible"
RELEASE_REPOSITORY = "telltaleatheist/crucible"

#: CANONICAL'S OWN WSL IMAGE, and the sums file beside it. PHASE20 section 2:
#: the release carries no rootfs of ours any more, and we store no digest of
#: theirs -- the sums file in the same directory is the digest's one owner.
UBUNTU_WSL_SERIES = "24.04"
UBUNTU_WSL_ROOTFS = "ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz"
UBUNTU_WSL_ROOTFS_URL = "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz"
UBUNTU_WSL_SUMS_URL = "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/SHA256SUMS"

#: PHASE19 2.2's record of what happened to the engine move, in the host
#: home. Read by `crucible/host/outcome.py`, which is its writer and parser,
#: and by `install.ps1` for its closing sentence (2.7).
WSL_OUTCOME_NAME = "wsl-outcome.json"

#: The line /etc/wsl.conf carries in a Crucible distro and nowhere else.
WSL_CONF_MARKER = "# crucible-rootfs"

#: /etc/wsl.conf, exactly as the import writes it and as the repair rewrites it.
WSL_CONF_TEXT = "# crucible-rootfs\n[boot]\nsystemd=true\n[user]\ndefault=crucible\n"

#: What an imported Canonical image needs before anything can be installed
#: into it: the crucible user, passwordless sudo, and the wsl.conf above.
#: One root script, and its owner is distro.ts -- see `finishImportScript`.
FINISH_IMPORT_SCRIPT = "id -u crucible >/dev/null 2>&1 || useradd --create-home --shell /bin/bash crucible\npasswd --delete crucible >/dev/null\nprintf 'crucible ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/crucible\nchmod 0440 /etc/sudoers.d/crucible\ncat > /etc/wsl.conf <<'EOF'\n# crucible-rootfs\n[boot]\nsystemd=true\n[user]\ndefault=crucible\nEOF"


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
    #: PHASE19 2.1: can the tray carry a machine past this state with
    #: nobody in front of it? True for the rows whose action is something
    #: we run, and for `wsl_ready`, which needs nothing run at all. False
    #: is the tray writing `cannot` and stopping.
    automatic: bool
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
        automatic=False,
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
        automatic=True,
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
        automatic=True,
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
        automatic=True,
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
        automatic=True,
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
        automatic=False,
        optional=True,
    ),
    WslStateDef(
        code="guest_no_network",
        probe="guest-network",
        probe_argv=("wsl.exe", "-d", "crucible", "--exec", "bash", "-c", "set -e; for u in {indexes}; do curl -fsSL -I -m 20 -o /dev/null \"$u\" || { echo \"$u could not be reached\" >&2; exit 1; }; done", ),
        sentence="The \"crucible\" distribution cannot reach one of the places this install downloads from: {said}. A VPN or a proxy on this machine usually explains it; there is nothing to install until it can.",
        action_kind="link",
        action_argv=(),
        action_text="",
        action_url="https://github.com/telltaleatheist/crucible/releases/download/v{release}/crucible-{release}-py3-none-any.whl",
        automatic=False,
        optional=True,
    ),
    WslStateDef(
        code="guest_no_disk",
        probe="guest-disk",
        probe_argv=("wsl.exe", "-d", "crucible", "--exec", "bash", "-c", "df -Pk \"$HOME\" | awk 'NR==2 {print $4}'", ),
        sentence="This install needs {required} free and the \"crucible\" distribution has {free}. Nothing has been downloaded.",
        action_kind="instruct",
        action_argv=(),
        action_text="Free some space on the drive WSL keeps its disk on, then try again.",
        action_url="",
        automatic=False,
        optional=True,
    ),
    WslStateDef(
        code="guest_root_unreachable",
        probe="guest-root",
        probe_argv=("wsl.exe", "-d", "crucible", "-u", "root", "--exec", "id", "-u", ),
        sentence="The \"crucible\" distribution will not let Crucible in as root ({said}). The server is installed as a system service, which needs root to write, so there is nothing to install until it does.",
        action_kind="instruct",
        action_argv=(),
        action_text="Enable the root account in \"crucible\", or let Crucible import its own distribution, which grants root through wsl.exe with no password.",
        action_url="",
        automatic=False,
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
        automatic=True,
        optional=False,
    ),
)

#: Every code, in table order. `crucible/host/wslstate.py` must have a
#: predicate for each, and no others.
WSL_STATE_CODES: tuple[str, ...] = tuple(state.code for state in WSL_STATES)
