"""Try again, for a person with no app: `crucible orchestrator --try-again`.

FRESH-INSTALL #16 (kylies-pc, 2026-09-26). PHASE19 2.5 made Try again a button
in the apps, and kylies-pc had no app. The one retry left was
`POST http://127.0.0.1:7101/install` with `{"target": "wsl"}` and the engine's
bearer, which nobody would ever find. This is that same call, made by the
product: the SAME door, the SAME move and the same claim as an app's button,
so a person and an app cannot get two different retries. The desktop tray's
Try again item (`crucible/desktop.py`) calls it too.

It follows the move to its end and says how it ended, from `wsl-outcome.json`
(2.2), because "it started" is not an answer a person can act on.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import outcome
from .errors import HostError

#: The orchestrator's door (PHASE17 3.2). `/install` is 2.5's Try again.
DOOR = "http://127.0.0.1:7101"
INSTALL_URL = DOOR + "/install"
EVENTS_URL = DOOR + "/install/events"

#: How long one read of the move's stream may be silent. The distro import is
#: 28 s with no line (FRESH-INSTALL #26); a guest pip line can take minutes.
STREAM_READ_TIMEOUT_SECONDS = 30 * 60.0

#: How long to wait for a controller this call had to start.
CONTROLLER_START_SECONDS = 90.0

Say = Callable[[str], None]


def bearer(home: Path) -> str | None:
    """The engine's token, as the door checks it, from this machine's own files.

    The pairing file first (the host's copy of whichever engine is live, 3.6),
    then the host's own config (a native engine that has not written one yet).
    """
    from ..pairing import parse_pairing_line

    try:
        return parse_pairing_line((home / "pairing").read_text(encoding="utf-8").strip()).token
    except (OSError, ValueError):
        pass
    from .app import read_token

    return read_token(home)


def _opener() -> urllib.request.OpenerDirector:
    # Local door; the invoking shell's proxy must not route it (local.py's rule).
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _controller_up() -> bool:
    try:
        with _opener().open(DOOR + "/v1/ping", timeout=3) as response:
            answered = json.load(response)
    except (OSError, ValueError):
        return False
    return isinstance(answered, dict) and answered.get("role") == "orchestrator"


def _ensure_controller(home: Path, say: Say) -> None:
    """Start the controller when it is not running, as the tray does."""
    if _controller_up():
        return
    say("Starting Crucible's controller first.")
    from ..local import _spawn_controller

    _spawn_controller(home)
    deadline = time.monotonic() + CONTROLLER_START_SECONDS
    while not _controller_up():
        if time.monotonic() >= deadline:
            raise HostError(
                "host_door_unavailable",
                "Crucible's controller did not start, so nothing could be tried "
                "again. Restart Windows, sign in, and Crucible starts by itself.",
            )
        time.sleep(0.5)


def _describe(envelope: dict[str, object], say: Say) -> None:
    """One plain line per step and per WSL fact; pip's own lines stay out."""
    event, data = envelope.get("event"), envelope.get("data")
    if not isinstance(data, dict):
        return
    if event == "step":
        say(f"Step {data.get('index')} of {data.get('total')}: {data.get('name')}")
    elif event == "line":
        text = data.get("text")
        if isinstance(text, str) and text.startswith("wsl: "):
            say(text[len("wsl: "):])
    elif event == "failed":
        say(f"Stopped: {data.get('code')}")


def _follow(response: object, say: Say) -> None:
    for raw in response:  # type: ignore[attr-defined]
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
    """Run the move once more through the door, follow it, return how it ended.

    A move already in flight (the tray's own, at start) answers 409
    `host_install_running`, and this then ATTACHES to it (2.6) instead of
    failing: the person asked for the move, and there it is.
    """
    _ensure_controller(home, say)
    # A controller that has just started writes its config and pairing within
    # seconds (`local.act`'s fresh-install wait); the door needs that token.
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
            raise HostError("host_door_unavailable", f"the controller refused: HTTP {exc.code}") from exc
        say("Crucible was already setting it up; following that.")
        attach = urllib.request.Request(EVENTS_URL, headers=headers)
        try:
            with _opener().open(attach, timeout=STREAM_READ_TIMEOUT_SECONDS) as response:
                _follow(response, say)
        except urllib.error.HTTPError as again:
            if again.code != 404:  # 404: it ended between the two calls
                raise HostError("host_door_unavailable", f"the controller refused: HTTP {again.code}") from again
    return outcome.read(home)
