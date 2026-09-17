"""Device pairing: how an app gets this engine's token without a person typing it.

OPEN BY DEFAULT (Owen, 2026-09-17): *"i dont think we need the approve
authentication. ollama allows anybody to connect if they can reach it. make that
the case with crucible servers as well."* So a request is APPROVED the moment it
is made, and connecting is: type the address, wait two seconds, connected.

WHAT THAT MEANS, SAID PLAINLY. Reaching the port is the whole of the
authorisation. The bearer token still exists and every other door still demands
it, but it is no longer a secret this module withholds — it is handed to whoever
asks. It is an identifier, not a lock. That is deliberate and it is Ollama's
posture, which is the posture that was asked for.

The one way this differs from Ollama, recorded because it is not obvious: an
Ollama nobody guards leaks compute. A Crucible nobody guards also leaks the
ability to SPEND a configured `[upstreams.*]` account — `GET /v1/settings` never
returns a key (`upstreams.settings_entry`), so keys cannot be stolen through this,
but a route to a cloud model can be called and billed. A machine with no upstream
configured has nothing here that Ollama does not.

THE MECHANISM IS KEPT, NOT DELETED. `[auth] open_pairing = false` restores the
approval step exactly as it was, and everything that served it — the short code,
the operator list, the decision door — still works. Deleting it would have made
"open" the only thing this can ever be, and a default is a thing you can change.

Unchanged either way: the app polls with a separate high-entropy secret, never
with the short code; requests are bounded, expire after five minutes, and
disappear on restart. Those are flood guards, not authentication, and an open
door still wants them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import secrets
import threading
import time
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from .errors import ApiError

TTL = 300
INTERVAL = 2
MAX_REQUESTS = 32


class StartPairing(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    client_name: str = Field(min_length=1, max_length=80, pattern=r"^[^\x00-\x1f\x7f]+$")


class PollPairing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    device_code: str = Field(min_length=32, max_length=128)


class DecidePairing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    user_code: str = Field(min_length=9, max_length=9, pattern=r"^[A-Z0-9]{4}-[A-Z0-9]{4}$")
    allow: bool


@dataclass
class Entry:
    id: str
    secret_hash: str
    user_code: str
    client_name: str
    address: str
    created: float
    expires: float
    state: str = "pending"
    last_poll: float = float("-inf")


class PairingRequests:
    """The pending table. `open_pairing` decides what a new request starts as.

    REQUIRED, with no default. The ruled default lives in exactly one place --
    `config._open_pairing`, where an absent `[auth] open_pairing` reads as True --
    and a second statement of it here would be a second owner of one fact. That
    is `docs/ARCHITECTURE.md`'s one shape, and it was not hypothetical: with a
    default here, flipping it changed nothing observable, because `api.py` passes
    the config's value over the top. A default nothing reads is a default that
    lies to the next person who edits it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 *, open_pairing: bool):
        self.clock = clock
        self.open_pairing = open_pairing
        self.entries: dict[str, Entry] = {}
        self.lock = threading.Lock()

    def _expire(self, now: float) -> None:
        self.entries = {key: entry for key, entry in self.entries.items() if entry.expires > now}

    def start(self, client_name: str, address: str) -> dict:
        with self.lock:
            now = self.clock()
            self._expire(now)
            if len(self.entries) >= MAX_REQUESTS or any(
                entry.address == address and now - entry.created < 5 for entry in self.entries.values()
            ):
                raise ApiError(429, "pairing_busy", "Too many connection requests; wait before trying again")
            secret = secrets.token_urlsafe(32)
            alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
            raw = "".join(secrets.choice(alphabet) for _ in range(8))
            entry = Entry(secrets.token_urlsafe(18), hashlib.sha256(secret.encode()).hexdigest(),
                          raw[:4] + "-" + raw[4:], client_name.strip(), address, now, now + TTL,
                          state="pending" if self.open_pairing is False else "approved")
            self.entries[entry.id] = entry
            # `approval_required` is ADDITIVE and it is the point: a client that
            # reads it shows a short code only when a code is going to be needed,
            # instead of displaying one nobody will ever be asked to approve. A
            # client that ignores it still works — it polls, and the answer is
            # already `approved`.
            return {"id": entry.id, "device_code": secret, "user_code": entry.user_code,
                    "expires_in": TTL, "interval": INTERVAL,
                    "approval_required": self.open_pairing is False}

    def pending(self) -> list[dict]:
        with self.lock:
            now = self.clock()
            self._expire(now)
            return [{"id": entry.id, "user_code": entry.user_code, "client_name": entry.client_name,
                     "address": entry.address, "expires_in": max(0, int(entry.expires - now))}
                    for entry in self.entries.values() if entry.state == "pending"]

    def decide(self, id: str, user_code: str, allow: bool) -> dict:
        with self.lock:
            self._expire(self.clock())
            entry = self.entries.get(id)
            if entry is None:
                raise ApiError(404, "pairing_expired", "This connection request has expired")
            if not secrets.compare_digest(entry.user_code, user_code):
                raise ApiError(409, "pairing_code_mismatch", "The displayed connection code changed; refresh before approving")
            if entry.state != "pending":
                raise ApiError(409, "pairing_decided", "This connection request already has a decision")
            entry.state = "approved" if allow else "denied"
            return {"status": entry.state}

    def poll(self, id: str, device_code: str) -> str:
        with self.lock:
            now = self.clock()
            self._expire(now)
            entry = self.entries.get(id)
            if entry is None:
                return "expired"
            if not secrets.compare_digest(entry.secret_hash, hashlib.sha256(device_code.encode()).hexdigest()):
                raise ApiError(403, "pairing_invalid", "This connection request cannot be read with that device credential")
            if now - entry.last_poll < INTERVAL:
                raise ApiError(429, "pairing_slow_down", "Wait two seconds between connection checks")
            entry.last_poll = now
            # Approved responses may be retried with the same device secret until
            # expiry, so a lost HTTP response does not strand an approved app.
            return entry.state
