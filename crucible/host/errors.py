"""Every refusal the host makes, by the name PHASE15-HOST.md section 4 gives it.

Same rule as `crucible/errors.py`: there is no error here that means "something
went wrong". A `HostError` carries a CODE a person can search for and an app can
switch on, and the door (4.3) puts that same code on the wire verbatim — which
is why the codes are listed in one place rather than spelled at each raise.
"""

from __future__ import annotations

from ..errors import CrucibleError

#: Every code this package raises, with what it means. The door's `error` event
#: carries one of these or one of the 4c state codes (`crucible/host/wsl_states.py`),
#: and `sdk/bootstrap/src/hostdoor.ts` is the client that reads them.
HOST_ERROR_CODES: dict[str, str] = {
    "host_door_unavailable": "The local controller could not bind its control port; its owned child was stopped.",
    "host_windows_only": (
        "`crucible orchestrator` is a Windows verb. On Linux and macOS the server runs "
        "on the machine and its service manager already supervises it (4.4: "
        "there is no host on the Mac)."
    ),
    "host_no_localappdata": (
        "LOCALAPPDATA is not set, so there is no per-user directory to put the "
        "host's files in. It is read from the environment and never assembled "
        "from a username."
    ),
    "host_no_token": (
        "the host has no config yet, so the install door has no bearer to check "
        "against. True only before the first host-mode `crucible init`."
    ),
    "host_unauthorized": "the install door was called without the engine token.",
    "host_install_running": (
        "a second POST /install arrived while one was in flight. There is one "
        "install on a machine and the second caller waits."
    ),
    "host_already_running": (
        "another `crucible orchestrator` already holds this machine's tray. Two trays "
        "would boot the same distro twice and watch each other's recoveries."
    ),
    "host_no_pack": (
        "this machine has no host runtime installed, so there is no "
        "`crucible.cmd` to start a host-mode server with."
    ),
    "guest_ahead_of_host": (
        "the WSL guest is a NEWER Crucible than this host. One release per "
        "machine, and the host is what moves the guest forward — never "
        "backwards, so it was left exactly as it is and this is the report."
    ),
    "guest_release_unreadable": (
        "the WSL guest's installation.json names a release this build cannot "
        "order against its own, so nothing could say whether it is behind. It "
        "was left alone rather than carried."
    ),
    "config_from_unreadable": "`--config-from` was given a file that is not TOML.",
    "config_from_no_token": (
        "`--config-from` was given a document with no `auth.token`. The point of "
        "the flag is that the token survives the move to the guest; a file "
        "without one carries nothing."
    ),
    "config_from_no_reserve": (
        "`--config-from` was given a document with no `[accelerator]` "
        "desktop_allowance_bytes and desktop_allowance_basis to carry."
    ),
    "pairing_acl_failed": (
        "the pairing file's ACL could not be set to this user only. The file is "
        "DELETED rather than left readable by everybody with a token in it."
    ),
    "orchestrator_distro_invalid": (
        "`[orchestrator] distro` in the Windows config names something that is "
        "not a distribution name. A person who wrote it meant to grant "
        "something, so it is refused rather than ignored (PHASE17 2.5)."
    ),
    "orchestrator_wsl_invalid": (
        "`[orchestrator] wsl` in the Windows config holds something other than "
        '"never". It is the one way to keep a machine native on purpose '
        "(PHASE19 1), so a value nobody defined is refused rather than ignored."
    ),
    # The sentences a PERSON reads for the three restart codes are
    # `installer.REBOOT_SENTENCE`, `REBOOT_STILL_OWED_SENTENCE` and
    # `REBOOT_AGAIN_SENTENCE`. All three name "Update and restart", because WSL
    # is installed by Windows' servicing step and a restart that skips or
    # postpones waiting updates installs nothing (#14, Owen's ruling,
    # 2026-09-26), and all three say someone must sign in afterwards (#8).
    "wsl_reboot_required": (
        "WSL was enabled and Windows must restart to install it. The person is "
        'told to choose "Update and restart", because Windows installs WSL in '
        "the same servicing step as its waiting updates. The move stops here "
        "and the tray resumes it once somebody signs in after the restart "
        "(PHASE19 2.3)."
    ),
    "wsl_reboot_still_owed": (
        "Windows restarted and servicing has still not installed WSL, which is "
        "normal when updates were waiting. Within the restart budget "
        '(`installer.RESTART_BUDGET`) this is reboot-pending: one more "Update '
        'and restart", a sign-in, and the tray goes on (#19).'
    ),
    "wsl_reboot_again": (
        "the restart budget is spent and WSL is still not installed, usually "
        "because the restarts skipped or postponed waiting updates. The person "
        'is told to install the updates and choose "Update and restart"; the '
        "tray re-checks at every start and goes on when WSL is live (PHASE19 "
        "2.4, #14, #19)."
    ),
    "wsl_outcome_invalid": (
        "`wsl-outcome.json` is present and is not the document PHASE19 2.2 "
        "describes. It is what the tray decides from, so it is refused rather "
        "than read as 'nothing has happened here yet'."
    ),
    "wsl_state_unknown": (
        "the 4c table answered a code this build has no predicate for, which "
        "means the generated table and the predicates have drifted."
    ),
}


class HostError(CrucibleError):
    """A host refusal. `code` is a key of `HOST_ERROR_CODES` or a 4c state code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
