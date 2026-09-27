from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import outcome
from .errors import HostError
from .paths import LOG_NAME, door_url

DOOR = door_url()
INSTALL_URL = DOOR + "/install"
EVENTS_URL = DOOR + "/install/events"

STREAM_READ_TIMEOUT_SECONDS = 30 * 60.0

CONTROLLER_START_SECONDS = 90.0

Say = Callable[[str], None]


def _sign_out(home: Path) -> str:
    return (
        f"Its log is {Path(home) / LOG_NAME}. Sign out of Windows and sign back "
        "in, then try again."
    )


def bearer(home: Path) -> str | None:
    from ..pairing import parse_pairing_line

    try:
        return parse_pairing_line((home / "pairing").read_text(encoding="utf-8").strip()).token
    except (OSError, ValueError):
        pass
    from .app import read_token

    return read_token(home)


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _controller_up() -> bool:
    try:
        with _opener().open(DOOR + "/v1/ping", timeout=3) as response:
            answered = json.load(response)
    except (OSError, ValueError):
        return False
    return isinstance(answered, dict) and answered.get("role") == "orchestrator"


def _ensure_controller(home: Path, say: Say) -> None:
    if _controller_up():
        return
    say("Starting Crucible's controller first.")
    from ..local import _spawn_controller

    try:
        _spawn_controller(home)
    except OSError as exc:
        raise HostError(
            "host_door_unavailable",
            f"Crucible's background app could not be started ({exc}), so nothing "
            f"could be tried again. {_sign_out(home)} Then Try again works.",
        ) from exc
    deadline = time.monotonic() + CONTROLLER_START_SECONDS
    while not _controller_up():
        if time.monotonic() >= deadline:
            raise HostError(
                "host_door_unavailable",
                "Crucible's background app was started but did not answer within "
                f"{CONTROLLER_START_SECONDS:.0f} s, so nothing could be tried "
                f"again. {_sign_out(home)} Then Try again works.",
            )
        time.sleep(0.5)


def _describe(envelope: dict[str, object], say: Say) -> None:
    event, data = envelope.get("event"), envelope.get("data")
    if not isinstance(data, dict):
        return
    if event == "step":
        from .installer import STEP_WORDS

        name = str(data.get("name"))
        say(f"Step {data.get('index')} of {data.get('total')}: {STEP_WORDS.get(name, name)}")
    elif event == "line":
        text = data.get("text")
        if isinstance(text, str) and text.startswith("wsl: "):
            say(text[len("wsl: "):])
    elif event == "failed":
        say(f"Setting up the Linux engine stopped ({data.get('code')}).")


def _follow(response: object, say: Say) -> None:
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
        except ValueError:
            continue
        if isinstance(envelope, dict):
            _describe(envelope, say)


def try_again(home: Path, say: Say = print) -> outcome.Outcome | None:
    _ensure_controller(home, say)
    deadline = time.monotonic() + CONTROLLER_START_SECONDS
    token = bearer(home)
    while token is None and time.monotonic() < deadline:
        time.sleep(0.5)
        token = bearer(home)
    if token is None:
        raise HostError(
            "host_no_token",
            "Crucible's controller is running but has not set up its engine yet, "
            "so there is nothing to try again with. Wait a minute and try again.",
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Crucible-Api": "1",
    }
    request = urllib.request.Request(
        INSTALL_URL, data=json.dumps({"target": "wsl"}).encode(), headers=headers, method="POST"
    )
    say("Trying again to set up the Linux engine.")
    try:
        with _opener().open(request, timeout=STREAM_READ_TIMEOUT_SECONDS) as response:
            _follow(response, say)
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise HostError("host_door_unavailable", f"Crucible's background app refused the request (HTTP {exc.code}). {_sign_out(home)}") from exc
        say("Crucible was already setting it up; following that.")
        attach = urllib.request.Request(EVENTS_URL, headers=headers)
        try:
            with _opener().open(attach, timeout=STREAM_READ_TIMEOUT_SECONDS) as response:
                _follow(response, say)
        except urllib.error.HTTPError as again:
            if again.code != 404:
                raise HostError("host_door_unavailable", f"Crucible's background app refused the request (HTTP {again.code}). {_sign_out(home)}") from again
    return outcome.read(home)
