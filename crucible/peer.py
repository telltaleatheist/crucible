"""The orchestrator/engine relation. PHASE17-ORCHESTRATOR.md sections 2 and 3.

Two halves of one handshake live in here, deliberately in one file:

* the **engine's** side — {@link PeerState}, the claim an engine records about
  itself and answers `/v1/info` and `/v1/peer` with;
* the **orchestrator's** side — {@link claim_engine}, {@link release_engine}
  and {@link read_peer}, three functions that make the calls.

One file because the two are one contract, and a refusal name that lives
beside the code that raises it but not beside the code that reads it is a name
that drifts. Nothing in here imports FastAPI, httpx, or anything else the
server stack drags in: the orchestrator half runs inside the Windows tray,
which must start in well under a second at login and must not hold an event
loop (`crucible/host/door.py` states the same rule for the same reason). So
the client calls are `urllib.request`.

WHY A CLAIM IS NOT A PERMISSION
--------------------------------
An engine never checks its claim before doing anything. There is nothing an
orchestrator asks an engine to do that an app may not also ask — the
orchestrator's powers are all on the OTHER side of the relation, over
`wsl.exe`, a child process and a systemd unit. So a claim is a STATEMENT OF
FACT by the one process that knows it, recorded so that `/v1/info` can answer
"who manages this", and nothing else. Reading it as an authorisation would
invite exactly the mistake this system keeps finding: a second gate on a door
that already has one.

WHY IT IS NOT PERSISTED
------------------------
`managed_by` dies with the process. A claim written to `config.toml` would
outlive the orchestrator that made it — uninstall the tray, reboot, and the
engine still names a door that will never answer again — which is
`docs/ARCHITECTURE.md`'s one shape: a fact with two owners and nothing
comparing them. The relation is RE-ASSERTED instead: the orchestrator claims
at presence-detection and again on every down-to-up edge of its watch, so an
engine that restarted unmanaged is claimed again within one 15-second tick.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .errors import ApiError
from .jobs.base import utcnow

#: The two roles. A property of a PROCESS, never of an install: on a Windows
#: machine with no WSL, ONE install runs both as two processes.
ROLE_ENGINE = "engine"
ROLE_ORCHESTRATOR = "orchestrator"
ROLES: tuple[str, ...] = (ROLE_ENGINE, ROLE_ORCHESTRATOR)

#: The orchestrator's backend kind — ON THE WIRE AND NOWHERE ELSE.
#:
#: `detect_backend()` never returns it, `crucible init --backend` never accepts
#: it, and it is deliberately absent from `crucible/backend.py`'s
#: `BACKEND_KINDS`, which is the list of backends a SERVER can be configured
#: as. It is a literal in the orchestrator's own `/v1/info` because a client
#: reading `host.backend` must get an answer that is true, and "what
#: accelerator does the thing that plays nothing have" has exactly one honest
#: answer.
BACKEND_ORCHESTRATOR = "orchestrator"

#: How an orchestrator HOLDS its engine (PHASE15 4.1a), on the wire.
#:
#: `child` and not 4.1a's internal `host-child`: on the wire the word "host" is
#: the thing this phase renames, and the orchestrator is the only possible
#: parent, so the qualifier says nothing the field does not.
OWNER_WSL_UNIT = "wsl-unit"
OWNER_CHILD = "child"
OWNER_FOUND = "found"
OWNERS: tuple[str, ...] = (OWNER_WSL_UNIT, OWNER_CHILD, OWNER_FOUND)

#: The relation's three refusals. They are `unauthorized` and
#: `api_version_mismatch` with the relation's name on them, and that is
#: deliberate: the caller is not an app, it is an orchestrator that has just
#: booted an engine and is telling it so. Told `unauthorized`, an orchestrator
#: cannot tell "the token I copied out of the guest's pairing file is stale"
#: from "some app's token is wrong" — one is its own bug and the other is not
#: its business.
PEER_TOKEN_MISMATCH = "peer_token_mismatch"
PEER_VERSION_INCOMPATIBLE = "peer_version_incompatible"
PEER_ALREADY_MANAGED = "peer_already_managed"

#: How long an orchestrator waits on a claim. It is one small POST to loopback;
#: a tray must not block its startup on a server that is wedged.
CLAIM_TIMEOUT_SECONDS = 10.0

CLAIM_PATH = "/v1/peer/claim"
PEER_PATH = "/v1/peer"


def normalise_url(url: str) -> str:
    """One spelling per orchestrator, so a re-claim is recognised as one.

    `http://127.0.0.1:7101/` and `http://127.0.0.1:7101` are the same door, and
    an engine that thought otherwise would answer `peer_already_managed` to the
    very orchestrator that holds the claim — on its own watch tick, forever.
    Only the trailing slash is touched: everything else about a URL is the
    caller's to spell, and lowercasing a host or dropping a default port would
    be this module inventing an opinion about addresses.
    """
    return url.strip().rstrip("/")


@dataclass(frozen=True)
class Orchestrator:
    """Who claimed this engine. Three strings and no more.

    `version` is the orchestrator's SOFTWARE version, for an operator reading
    a page and for a log line. The API version is not in here: it travels in
    `X-Crucible-Api-Version` on the claim, where every other call already
    carries it, and a second copy in the body would be a fact with two owners.
    """

    name: str
    url: str
    version: str

    @staticmethod
    def from_body(raw: Any) -> "Orchestrator":
        """Read the `orchestrator` block, or refuse by name.

        Every field is required. There is no default name and no default url —
        an orchestrator that could not say who it was would be recorded as
        something no operator could find and no release could match.
        """
        if not isinstance(raw, dict):
            raise ApiError(
                400,
                "invalid_request",
                "the claim body is {\"orchestrator\": {\"name\", \"url\", "
                "\"version\"}}; there is no `orchestrator` object in this one",
            )
        values: dict[str, str] = {}
        for key in ("name", "url", "version"):
            value = raw.get(key)
            if not isinstance(value, str) or value.strip() == "":
                raise ApiError(
                    400,
                    "invalid_request",
                    f"the claim's `orchestrator.{key}` must be a non-empty "
                    "string. An orchestrator that cannot say who it is would "
                    "be recorded as something no operator could find",
                )
            values[key] = value.strip()
        return Orchestrator(
            name=values["name"],
            url=normalise_url(values["url"]),
            version=values["version"],
        )

    def to_dict(self) -> dict[str, str]:
        """The block a claim sends. `managed_by` on the wire is NARROWER."""
        return {"name": self.name, "url": self.url, "version": self.version}


@dataclass(frozen=True)
class Claim:
    """A live claim: who, and when they said so."""

    orchestrator: Orchestrator
    claimed: str

    def managed_by(self) -> dict[str, str]:
        """What `/v1/info` and `/v1/peer` publish: the NAME and the URL only.

        Not the version. `managed_by` answers "who manages this and where do I
        reach them"; the orchestrator's own version is a fact about the
        orchestrator, which is what `GET /v1/info` on ITS door is for. Two
        copies of a version string in two documents is two things to keep in
        step across an upgrade that changes exactly one of them.
        """
        return {"name": self.orchestrator.name, "url": self.orchestrator.url}


class PeerState:
    """THE ENGINE'S HALF. One claim or none, in memory, for this process.

    Not thread-safe by a lock and not needing one: every mutation happens on
    the event loop, from a route handler, and the routes that touch it are two
    short synchronous functions. A lock here would be ceremony around an
    assignment.
    """

    def __init__(self) -> None:
        self._claim: Claim | None = None

    @property
    def claim_held(self) -> Claim | None:
        return self._claim

    def managed_by(self) -> dict[str, str] | None:
        """`/v1/info`'s field. `None` is a complete, correct answer.

        An engine nobody claims is a whole Crucible: the Mac, the droplet, and
        any `crucible serve` run by hand.
        """
        return None if self._claim is None else self._claim.managed_by()

    def claim(self, orchestrator: Orchestrator, *, force: bool = False) -> Claim:
        """Record the claim, or refuse `peer_already_managed`.

        **The same url re-claiming is success**, and it must be: an
        orchestrator re-claims on every down-to-up edge of its watch, and an
        engine that restarted has forgotten (section 2.3). Same url, new claim
        time, 200.
        """
        held = self._claim
        if held is not None and held.orchestrator.url != orchestrator.url and not force:
            raise ApiError(
                409,
                PEER_ALREADY_MANAGED,
                f"this engine is already managed by {held.orchestrator.name} at "
                f"{held.orchestrator.url}, and {orchestrator.url} is a different "
                "orchestrator. Two orchestrators claiming one engine is a "
                "machine misconfigured — two trays booting one distro, or a "
                "second install nobody remembers — and the first answer is a "
                "refusal naming the other one rather than a silent steal. A "
                "person looking at the operator page can see both and send "
                "`force`",
                {
                    "managed_by": held.managed_by(),
                    "claimed": held.claimed,
                    "claimant": orchestrator.to_dict(),
                },
            )
        self._claim = Claim(orchestrator=orchestrator, claimed=utcnow())
        return self._claim

    def release(self, orchestrator: Orchestrator | None, *, force: bool = False) -> None:
        """Drop the claim. A release is a release.

        Nothing claimed is not a refusal: "there is no claim" is the state the
        caller asked for, so there is no `peer_not_managed` and never was.
        Somebody ELSE's claim is refused, because releasing one by accident is
        how an engine ends up unmanaged with a tray still watching it.
        """
        held = self._claim
        if held is None:
            return
        if (
            orchestrator is not None
            and held.orchestrator.url != orchestrator.url
            and not force
        ):
            raise ApiError(
                409,
                PEER_ALREADY_MANAGED,
                f"this engine is managed by {held.orchestrator.name} at "
                f"{held.orchestrator.url}; {orchestrator.url} does not hold that "
                "claim and does not release it. Send `force` to take it away, "
                "which is a person's act and not an orchestrator's",
                {"managed_by": held.managed_by(), "claimant": orchestrator.to_dict()},
            )
        self._claim = None

    def document(self, uptime_s: float) -> dict[str, Any]:
        """`GET /v1/peer` — the relation's own read (section 2.4).

        `uptime_s` is the engine's MONOTONIC uptime, which is what tells an
        orchestrator that an engine answering again is a NEW process rather
        than the one it claimed — the signal that a re-claim is owed. A wall
        clock would make that signal lie across an NTP correction.
        """
        return {
            "role": ROLE_ENGINE,
            "managed_by": self.managed_by(),
            "uptime_s": uptime_s,
        }


# --------------------------------------------------------------------------
# THE ORCHESTRATOR'S HALF — three calls, `urllib` only.
# --------------------------------------------------------------------------


class PeerCallFailed(Exception):
    """A claim, release or read that did not happen, with what was said.

    NOT an `ApiError`: this is the orchestrator's side, and there is no HTTP
    response being composed here. The orchestrator's answer to a failed claim
    is a line in its log and a retry on the next watch tick — never a crash,
    because an engine that will not be claimed is still an engine, and a tray
    that died trying to tell it so would take the watch with it.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _call(
    url: str,
    token: str,
    method: str,
    body: dict[str, Any] | None,
    *,
    api_version: int,
    timeout_s: float,
) -> dict[str, Any]:
    """One authenticated call to an engine's peer door, or `PeerCallFailed`."""
    from . import API_HEADER

    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            API_HEADER: str(api_version),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as answer:
            raw = answer.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise PeerCallFailed(_refusal_code(detail), f"HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise PeerCallFailed(
            "peer_unreachable",
            f"{url} did not answer: {type(exc).__name__}: {exc}",
        ) from None
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise PeerCallFailed("peer_unreadable", f"{url} answered non-JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise PeerCallFailed("peer_unreadable", f"{url} answered a {type(parsed).__name__}")
    return parsed


def _refusal_code(body: str) -> str:
    """The engine's OWN code out of its refusal body, or `peer_unreadable`.

    The same rule `crucible/tasks.py`'s `_host_refusal_code` follows in the
    other direction: a refusal carries a name, and inventing a second one here
    would give one fact two names depending on which side read it. A body this
    side cannot parse becomes `peer_unreadable`, which is the honest answer —
    something refused, and it did not say what in a shape this understands.
    """
    try:
        error = json.loads(body).get("error")
        code = error.get("code") if isinstance(error, dict) else None
    except (json.JSONDecodeError, AttributeError):
        return "peer_unreadable"
    return str(code) if isinstance(code, str) and code else "peer_unreadable"


def claim_engine(
    engine_url: str,
    token: str,
    orchestrator: Orchestrator,
    *,
    api_version: int,
    force: bool = False,
    timeout_s: float = CLAIM_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """`POST /v1/peer/claim` — tell an engine who manages it.

    `force` defaults to False and **no orchestrator sends True on any code
    path** (section 2.1). It is a person's act, through the page, because two
    orchestrators on one engine is a machine misconfigured and the right first
    answer is the refusal that names the other one.
    """
    body: dict[str, Any] = {"orchestrator": orchestrator.to_dict()}
    if force:
        body["force"] = True
    return _call(
        f"{normalise_url(engine_url)}{CLAIM_PATH}",
        token,
        "POST",
        body,
        api_version=api_version,
        timeout_s=timeout_s,
    )


def release_engine(
    engine_url: str,
    token: str,
    orchestrator: Orchestrator,
    *,
    api_version: int,
    timeout_s: float = CLAIM_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """`DELETE /v1/peer/claim` — the orchestrator's Quit.

    A tray that exits leaving `managed_by` pointing at a door that no longer
    answers is PHASE15 3.6's "a file that exists and disagrees is worse than
    none", one layer up.
    """
    return _call(
        f"{normalise_url(engine_url)}{CLAIM_PATH}",
        token,
        "DELETE",
        {"orchestrator": orchestrator.to_dict()},
        api_version=api_version,
        timeout_s=timeout_s,
    )


def read_peer(
    engine_url: str,
    token: str,
    *,
    api_version: int,
    timeout_s: float = CLAIM_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """`GET /v1/peer` — `{role, managed_by, uptime_s}` (section 2.4)."""
    return _call(
        f"{normalise_url(engine_url)}{PEER_PATH}",
        token,
        "GET",
        None,
        api_version=api_version,
        timeout_s=timeout_s,
    )


__all__ = [
    "BACKEND_ORCHESTRATOR",
    "CLAIM_PATH",
    "Claim",
    "Orchestrator",
    "OWNERS",
    "OWNER_CHILD",
    "OWNER_FOUND",
    "OWNER_WSL_UNIT",
    "PEER_ALREADY_MANAGED",
    "PEER_PATH",
    "PEER_TOKEN_MISMATCH",
    "PEER_VERSION_INCOMPATIBLE",
    "PeerCallFailed",
    "PeerState",
    "ROLES",
    "ROLE_ENGINE",
    "ROLE_ORCHESTRATOR",
    "claim_engine",
    "normalise_url",
    "read_peer",
    "release_engine",
]
