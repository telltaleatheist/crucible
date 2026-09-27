from __future__ import annotations

from dataclasses import dataclass

CRUCIBLE_DISTRO = "crucible"
RELEASE_REPOSITORY = "telltaleatheist/crucible"

UBUNTU_WSL_SERIES = "24.04"
UBUNTU_WSL_ROOTFS = "ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz"
UBUNTU_WSL_ROOTFS_URL = "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz"
UBUNTU_WSL_SUMS_URL = "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/SHA256SUMS"

WSL_OUTCOME_NAME = "wsl-outcome.json"

WSL_CONF_MARKER = "# crucible-rootfs"

WSL_CONF_TEXT = "# crucible-rootfs\n[boot]\nsystemd=true\n[user]\ndefault=crucible\n"

FINISH_IMPORT_SCRIPT = "id -u crucible >/dev/null 2>&1 || useradd --create-home --shell /bin/bash crucible\npasswd --delete crucible >/dev/null\nprintf 'crucible ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/crucible\nchmod 0440 /etc/sudoers.d/crucible\nmkdir -p /etc/cloud && touch /etc/cloud/cloud-init.disabled\nmkdir -p /usr/lib/binfmt.d && printf ':WSLInterop:M::MZ::/init:PF\\n' > /usr/lib/binfmt.d/WSLInterop.conf\ncat > /etc/wsl.conf <<'EOF'\n# crucible-rootfs\n[boot]\nsystemd=true\n[user]\ndefault=crucible\nEOF"


@dataclass(frozen=True)
class WslStateDef:
    code: str
    probe: str
    probe_argv: tuple[str, ...]
    sentence: str
    action_kind: str
    action_argv: tuple[str, ...]
    action_text: str
    action_url: str
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

WSL_STATE_CODES: tuple[str, ...] = tuple(state.code for state in WSL_STATES)
