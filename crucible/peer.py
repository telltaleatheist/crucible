from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .errors import ApiError
from .jobs.base import utcnow

ROLE_ENGINE = "engine"
ROLE_ORCHESTRATOR = "orchestrator"
ROLES: tuple[str, ...] = (ROLE_ENGINE, ROLE_ORCHESTRATOR)

BACKEND_ORCHESTRATOR = "orchestrator"

OWNER_WSL_UNIT = "wsl-unit"
OWNER_CHILD = "child"
OWNER_FOUND = "found"
OWNERS: tuple[str, ...] = (OWNER_WSL_UNIT, OWNER_CHILD, OWNER_FOUND)

PEER_TOKEN_MISMATCH = "peer_token_mismatch"
PEER_VERSION_INCOMPATIBLE = "peer_version_incompatible"
PEER_ALREADY_MANAGED = "peer_already_managed"

CLAIM_TIMEOUT_SECONDS = 10.0

CLAIM_PATH = "/v1/peer/claim"
PEER_PATH = "/v1/peer"


def normalise_url(url: str) -> str:
    return url.strip().rstrip("/")


@dataclass(frozen=True)
class Orchestrator:
    name: str
    url: str
    version: str

    @staticmethod
    def from_body(raw: Any) -> "Orchestrator":
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
        return {"name": self.name, "url": self.url, "version": self.version}


@dataclass(frozen=True)
class Claim:
    orchestrator: Orchestrator
    claimed: str

    def managed_by(self) -> dict[str, str]:
        return {"name": self.orchestrator.name, "url": self.orchestrator.url}


class PeerState:
    def __init__(self) -> None:
        self._claim: Claim | None = None

    def managed_by(self) -> dict[str, str] | None:
        return None if self._claim is None else self._claim.managed_by()

    def claim(self, orchestrator: Orchestrator, *, force: bool = False) -> Claim:
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
        return {
            "role": ROLE_ENGINE,
            "managed_by": self.managed_by(),
            "uptime_s": uptime_s,
        }


class PeerCallFailed(Exception):
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
    return _call(
        f"{normalise_url(engine_url)}{PEER_PATH}",
        token,
        "GET",
        None,
        api_version=api_version,
        timeout_s=timeout_s,
    )


def read_info(
    engine_url: str,
    token: str,
    *,
    api_version: int,
    timeout_s: float = CLAIM_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return _call(
        f"{normalise_url(engine_url)}/v1/info",
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
    "read_info",
    "read_peer",
    "release_engine",
]
