"""Short-lived device pairing; finding an engine never discloses its credential.

Only an authenticated operator can approve the code displayed by a requesting
app. The app polls with a separate high-entropy secret, never with the short code.
Pending requests are bounded, expire after five minutes and disappear on restart.
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
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
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
                          raw[:4] + "-" + raw[4:], client_name.strip(), address, now, now + TTL)
            self.entries[entry.id] = entry
            return {"id": entry.id, "device_code": secret, "user_code": entry.user_code,
                    "expires_in": TTL, "interval": INTERVAL}

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
