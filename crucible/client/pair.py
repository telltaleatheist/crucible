from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
from typing import Any, Callable

from ..protocol import API_VERSION, DEFAULT_PORT
from . import transport
from .errors import ClientRefusal, error_in

PAIR_TIMEOUT_SECONDS = 10.0

PAIR_DEFAULT_PORT = DEFAULT_PORT

PAIRING_VERSION = 1

CLIENT_NAME_LIMIT = 80

POLL_FLOOR_SECONDS = 2.0

REQUEST_LIFETIME_SECONDS = 300.0

SLOW_DOWN = "pairing_slow_down"


def pair_origin(address: str) -> str:
    raw = address.strip()
    if not raw or any(ch in raw for ch in " @?#"):
        raise ClientRefusal(
            "pair_bad_address: give the other computer's address, like "
            "192.168.68.88 or kylies-pc, with no token or path in it"
        )
    if "://" not in raw:
        if raw.count(":") > 1 and not raw.startswith("["):
            raw = f"[{raw}]"
        raw = "http://" + raw
    parts = urllib.parse.urlsplit(raw)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.path not in ("", "/") or port == -1):
        raise ClientRefusal(
            f"pair_bad_address: {address!r} is not an address this can dial; "
            "give an IP address or a computer name, e.g. 192.168.68.88"
        )
    if port is None:
        port = 443 if parts.scheme == "https" else PAIR_DEFAULT_PORT
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}:{port}"


def pair_call(origin: str, path: str, body: dict[str, Any] | None = None,
              token: str | None = None) -> dict[str, Any]:
    try:
        with transport.open_url(
            origin,
            "GET" if body is None else "POST",
            path,
            token=token,
            body=None if body is None else transport.json_bytes(body),
            content_type=None if body is None else transport.JSON,
            timeout=PAIR_TIMEOUT_SECONDS,
        ) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        _body, error = error_in(exc.read())
        if error is not None:
            raise ClientRefusal(f"{error.code}: {error.message}") from None
        raise ClientRefusal(
            f"pair_not_crucible: {origin} answered HTTP {exc.code}, not as a Crucible"
        ) from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ClientRefusal(
            f"pair_unreachable: nothing answered at {origin} from this computer "
            f"({getattr(exc, 'reason', exc)}). On that computer, Crucible has to be "
            "running and shared with the network: on Windows that is "
            "`crucible lan enable` there, which also says if its network is "
            "marked Public. Check the address, then try again"
        ) from None
    except ValueError:
        raise ClientRefusal(f"pair_not_crucible: {origin} did not answer with JSON") from None
    if not isinstance(value, dict):
        raise ClientRefusal(f"pair_not_crucible: {origin} did not answer as a Crucible")
    return value


def _crucible_name(origin: str) -> str:
    ping = pair_call(origin, "/v1/ping")
    name = ping.get("name")
    if ping.get("crucible") is not True or not isinstance(name, str) or not name:
        raise ClientRefusal(f"pair_not_crucible: {origin} is not a Crucible")
    if ping.get("api_version") != API_VERSION:
        raise ClientRefusal(
            f"api_version_mismatch: {name} speaks API version "
            f"{ping.get('api_version')} and this computer speaks {API_VERSION}. "
            "Update whichever of the two is older"
        )
    if ping.get("pairing_version") != PAIRING_VERSION:
        raise ClientRefusal(
            f"pairing_unavailable: {name} is too old to connect by address; "
            "update Crucible on that computer"
        )
    return name


def _start(origin: str, name: str, client_name: str) -> dict[str, Any]:
    start = pair_call(
        origin, "/v1/pairing/start", {"client_name": client_name[:CLIENT_NAME_LIMIT]}
    )
    if start.get("name") != name or not isinstance(start.get("id"), str) \
            or not isinstance(start.get("device_code"), str):
        raise ClientRefusal(
            f"pair_not_crucible: {origin} returned an incompatible pairing request"
        )
    return start


def _poll(origin: str, start: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return pair_call(origin, "/v1/pairing/poll",
                         {"id": start["id"], "device_code": start["device_code"]})
    except ClientRefusal as exc:
        if str(exc).startswith(SLOW_DOWN):
            return None
        raise


def _approved_token(origin: str, name: str, answer: dict[str, Any]) -> str:
    token = answer.get("token")
    if answer.get("name") != name or not isinstance(token, str) or not token:
        raise ClientRefusal(f"pair_not_crucible: {origin} returned an incompatible approval")
    pair_call(origin, "/v1/info", token=token)
    return token


def pair(address: str, *, client_name: str, sleep: Callable[[float], None] | None = None,
         notify: Callable[[str], None] | None = None) -> tuple[str, str, str]:
    sleep = time.sleep if sleep is None else sleep
    origin = pair_origin(address)
    name = _crucible_name(origin)
    start = _start(origin, name, client_name)
    if start.get("approval_required") is not False and notify is not None:
        notify(
            f"{name} asks for approval: on that computer, open Crucible's console "
            f"and approve the code {start.get('user_code')}. Waiting..."
        )
    interval = max(POLL_FLOOR_SECONDS, float(start.get("interval") or 2))
    remaining = float(start.get("expires_in") or REQUEST_LIFETIME_SECONDS)
    first = True
    while remaining > 0:
        if not first:
            sleep(interval)
            remaining -= interval
        first = False
        answer = _poll(origin, start)
        if answer is None:
            continue
        status = answer.get("status")
        if status == "approved":
            return name, origin, _approved_token(origin, name, answer)
        if status == "denied":
            raise ClientRefusal(f"pair_denied: {name} turned this computer's request down")
        if status == "expired":
            break
    raise ClientRefusal(
        f"pair_expired: nobody approved the request on {name} in time; run this again"
    )
