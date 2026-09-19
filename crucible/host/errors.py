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
        "`crucible host` is a Windows verb. On Linux and macOS the server runs "
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
        "another `crucible host` already holds this machine's tray. Two trays "
        "would boot the same distro twice and watch each other's recoveries."
    ),
    "host_no_pack": (
        "this machine has no unpacked host pack, so there is no `crucible.cmd` "
        "to start a host-mode server with."
    ),
    "config_from_unreadable": "`--config-from` was given a file that is not TOML.",
    "config_from_no_token": (
        "`--config-from` was given a document with no `auth.token`. The point of "
        "the flag is that the token survives the move to the guest; a file "
        "without one carries nothing."
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
    "orchestrator_recipe_not_ours": (
        "a recovery recipe that restarts everything uid 1000 owns was asked "
        "for in a distribution Crucible did not import. CONSENT widens "
        "watching, claiming and the unit restart; it never widens this "
        "(PHASE15-HOST.md 4.1a, PHASE17 2.5)."
    ),
    "portproxy_self_loop": (
        "a portproxy row was asked for whose LISTEN SET contains the address it "
        "forwards TO, so it would accept its own connection and dial itself. "
        "`0.0.0.0` is the case that bit: it includes `127.0.0.1`, which is where "
        "every Crucible forward points. Measured on Owen's PC 2026-09-17, 15.5k "
        "of 16.4k ephemeral ports in TIME_WAIT."
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
