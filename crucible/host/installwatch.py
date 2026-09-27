"""What `install.ps1` prints after it starts the tray.

FRESH INSTALL ON KYLIES-PC, 2026-09-26, snags #6, #7 and #28. The script used to
wait ten seconds for `wsl-outcome.json` and print one of two sentences:

* #6: the move reached `wsl_reboot_required` twelve seconds AFTER the script had
  said "the app you installed from will show its progress" and exited 0. The one
  place that said "restart" was `host.log`, found by digging.
* #7: a PowerShell install has no app. From a console, the console IS the app.
* #28: the 1.0.46 installer's last line was 1.0.45's failure sentence, read back
  out of the file and printed as if it were this run's.

So on a console this WAITS for the move to reach an ending — done, a restart
owed, cannot, failed, declined — and prints each step on the way and the ending
in plain words, the restart included. It reads what the tray already publishes
(PHASE19 2.6: `GET /install`, `GET /install/events` on the door, and the outcome
file of 2.2) and decides nothing about the move itself.

An outcome counts as THIS run's only if it was written after the install
started (`--since`). An older one is printed with its time and release, as
history, and never as the current state.

When an APP ran the script (`--brief`), the app watches the move itself through
the same door (PHASE19 2.6), so this keeps 2.7's short ending: a few seconds for
a verdict, then the general sentence.

Everything printed goes through `log.plain`: ASCII, because the reader is
Windows PowerShell 5.1 in whatever code page the machine has (#12).
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

from . import outcome
from .errors import HostError
from .installer import STEP_WORDS, TRY_AGAIN_HINT
from .log import plain
from .paths import door_url

#: 2.7's wait when an app is the reader: long enough for a machine that cannot
#: to say so (`wsl --status` answers at once), short enough that the app is not
#: kept from attaching to the move.
BRIEF_SECONDS = 10.0

#: How long a console waits for the tray to DECIDE: its first presence pass
#: (up to `presence.WATCH_SECONDS`, 15 s), the native engine coming up, then
#: the decision. Two minutes is several times that. Past it, nothing is running
#: and nothing new was recorded, so the tray has decided to do nothing and the
#: console says what the machine last recorded.
DECISION_SECONDS = 120.0

#: When the door cannot be asked (no pairing yet, a refused token) the console
#: cannot see a move running, only its ending in the outcome file. It waits this
#: long for that ending, saying so every `HEARTBEAT_SECONDS`.
BLIND_SECONDS = 45 * 60.0
HEARTBEAT_SECONDS = 120.0

POLL_SECONDS = 1.0

#: A stream that has been quiet this long is asked again from the top. The
#: longest quiet step is the import (about 30 s); this is far past it.
STREAM_READ_SECONDS = 300.0

#: A line of the move longer than this is cut from the FRONT: the end of a
#: failing tool's output is where it says why (`RunResult.said`).
LINE_LIMIT = 300

#: An ending is never cut short: it is the instruction. This bounds only a
#: sentence that carries a tool's whole output.
ENDING_LIMIT = 1500

#: Endings are wrapped to this, so a console breaks them between words.
WIDTH = 78

#: The general sentence of 2.7, for an app reader. Kept word for word.
APP_SENTENCE = (
    "It is setting up its Linux engine now; the app you installed from will "
    "show its progress."
)

CONSOLE_START = (
    "Setting up the Linux engine. Each step is shown here. You can close this "
    "window at any time: the setup carries on by itself in the background."
)

DONE_SENTENCE = (
    "Done. Crucible's Linux engine is running on this PC. There is nothing "
    "else to do."
)

ALREADY_SENTENCE = (
    "Crucible's Linux engine was already set up on this PC. If it needs this "
    "release, Crucible moves it up by itself in the next few minutes. There is "
    "nothing else to do."
)

FOUND_SENTENCE = (
    "An engine that Crucible did not install is already running on this PC, "
    "so Crucible uses that one and sets nothing up. There is nothing else to do."
)

DECLINED_SENTENCE = (
    "This PC is set to keep the Windows engine ([orchestrator] wsl = \"never\" "
    "in config.toml), so Crucible does not set up a Linux engine. Crucible is "
    "ready and there is nothing else to do."
)

UNDECIDED_SENTENCE = (
    "Crucible has not started setting up its Linux engine yet. It does that by "
    "itself in the background, and the Crucible icon by the clock shows how it "
    "is going. There is nothing you need to do now."
)

TRAY_GONE_SENTENCE = (
    "Crucible's background app (the icon by the clock) is not answering, so "
    "nothing can carry on the setup right now. Sign out of Windows and sign "
    "back in: it starts again at sign-in and carries on by itself."
)

#: The first line of every ending that needs a person to restart Windows.
RESTART_BANNER = 'ACTION NEEDED: restart Windows with "Update and restart".'

#: The codes whose ending is a restart the person performs (#14).
RESTART_CODES = outcome.REBOOT_CODES | {"wsl_reboot_again"}


def one_line(text: object, limit: int = LINE_LIMIT) -> str:
    """A line of the move as a console shows it: ASCII, one line, bounded.

    wsl.exe's own text arrives multi-line, sometimes with a NUL between every
    character (#10, #17, #26). The console gets one line of it.
    """
    flat = " ".join(plain(str(text)).split())
    return flat if len(flat) <= limit else "..." + flat[-limit:]


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _local(value: str) -> str:
    """`at`, in the reader's own clock, to the minute."""
    parsed = _parse_time(value)
    return value if parsed is None else parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def recorded(home: Path) -> outcome.Outcome | None:
    """The outcome file, or None. An unreadable one is not this run's ending."""
    try:
        return outcome.read(home)
    except HostError:
        return None


def is_fresh(record: outcome.Outcome | None, since: datetime) -> bool:
    """Was this written by the install that started at `since` (#28)?"""
    if record is None:
        return False
    at = _parse_time(record.at)
    return at is not None and at >= since.replace(microsecond=0)


def ending(record: outcome.Outcome) -> list[str]:
    """What an ending means to the person at the console, in order."""
    sentence = one_line(record.sentence, ENDING_LIMIT) if record.sentence else ""
    if record.state == outcome.DONE:
        return [DONE_SENTENCE]
    if record.state == outcome.DECLINED:
        return [DECLINED_SENTENCE]
    if record.state == outcome.REBOOT_PENDING or record.code in RESTART_CODES:
        # The sentence IS the instruction (`installer.REBOOT_SENTENCE`,
        # `REBOOT_AGAIN_SENTENCE`); the banner makes it impossible to miss.
        return [RESTART_BANNER, sentence]
    if record.state == outcome.CANNOT:
        return [
            "Crucible cannot set up its Linux engine on this PC yet:",
            sentence,
            "The Windows engine keeps working meanwhile. Once that is fixed, "
            + TRY_AGAIN_HINT + ".",
        ]
    # `failed`: retried by the tray at its next start, once (PHASE19 2.2).
    if record.attempts < outcome.FAILED_ATTEMPT_CEILING:
        after = (
            "Crucible tries once more by itself the next time someone signs in to "
            "this PC. "
            "To try now instead, " + TRY_AGAIN_HINT + "."
        )
    else:
        after = (
            f"It has stopped {record.attempts} times in a row, so it will not "
            "try again by itself. " + TRY_AGAIN_HINT[0].upper() + TRY_AGAIN_HINT[1:]
            + " to try again."
        )
    return ["Setting up the Linux engine stopped:", sentence, after]


class Console:
    """Prints the move's events as a person reads them."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        #: The last tenth printed per file, so a download is ten lines, not
        #: three hundred.
        self._tenths: dict[str, int] = {}

    def say(self, text: str) -> None:
        if text:
            print(plain(text), file=self._out, flush=True)

    def event(self, envelope: dict[str, object]) -> None:
        kind = envelope.get("event")
        data = envelope.get("data")
        if not isinstance(data, dict):
            return
        if kind == "step":
            name = str(data.get("name", ""))
            self.say(
                f"Step {data.get('index', '?')} of {data.get('total', '?')}: "
                f"{STEP_WORDS.get(name, name)}"
            )
        elif kind == "line":
            text = one_line(data.get("text", ""))
            if text:
                self.say("  " + text)
        elif kind == "progress":
            self._progress(data)
        # `state`, `failed` and `done` are the ending's to say, from the
        # outcome file, once: printing them here would say it twice.

    def _progress(self, data: dict[str, object]) -> None:
        done, total, name = data.get("bytes_done"), data.get("bytes_total"), str(data.get("file", ""))
        if not isinstance(done, int) or not isinstance(total, int) or total <= 0:
            return
        tenth = min(10, done * 10 // total)
        if tenth <= self._tenths.get(name, -1):
            return
        self._tenths[name] = tenth
        self.say(f"  downloading {name}: {tenth * 10}% of {total / 1024 ** 2:.0f} MB")

    def paragraph(self, text: str) -> None:
        """A sentence a person must read, wrapped between words."""
        if text:
            self.say(textwrap.fill(plain(text), WIDTH, break_on_hyphens=False))

    def ending(self, record: outcome.Outcome) -> None:
        self.say("-" * WIDTH)
        for text in ending(record):
            self.paragraph(text)


def _token(home: Path) -> str | None:
    """The engine's bearer, which is what the door takes (door.py)."""
    from ..local import LocalError, connection

    try:
        return connection(home)[2]
    except (LocalError, OSError, ValueError):
        return None


def _open(path: str, token: str | None, timeout: float):
    headers = {} if token is None else {"Authorization": f"Bearer {token}", "X-Crucible-Api": "1"}
    request = urllib.request.Request(door_url(path), headers=headers)
    # Loopback: never through the caller's proxy (`crucible/local.py`).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def door_alive() -> bool:
    """Does the tray's door answer at all? `/v1/ping` takes no bearer (door.py)."""
    try:
        with _open("/v1/ping", None, 3.0) as response:
            return json.load(response).get("crucible") is True
    except (OSError, ValueError, AttributeError, urllib.error.URLError, http.client.HTTPException):
        return False


def door_status(token: str | None) -> dict[str, object] | None:
    """`GET /install` (PHASE19 2.6), or None when the door cannot be asked."""
    if token is None:
        return None
    try:
        with _open("/install", token, 5.0) as response:
            value = json.load(response)
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return None
    return value if isinstance(value, dict) else None


def follow(token: str, console: Console, seen: int) -> int:
    """`GET /install/events` until the move ends. Returns the last id printed."""
    try:
        with _open("/install/events", token, STREAM_READ_SECONDS) as response:
            for raw in response:
                try:
                    envelope = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(envelope, dict):
                    continue
                number = envelope.get("id")
                if isinstance(number, int):
                    # A re-attach replays the ring; what was printed is skipped.
                    if number <= seen:
                        continue
                    seen = number
                console.event(envelope)
    except (OSError, socket.timeout, urllib.error.URLError, http.client.HTTPException):
        # A 404 (the move ended between the two asks), a hang-up, a quiet
        # stream: the caller asks `GET /install` again.
        pass
    return seen


def watch(
    home: Path,
    since: datetime,
    *,
    brief: bool,
    out: TextIO = sys.stdout,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    console = Console(out)
    started = clock()
    if brief:
        while clock() - started < BRIEF_SECONDS:
            record = recorded(home)
            if is_fresh(record, since):
                assert record is not None
                # The outcome's OWN words for anything but `done` (2.7).
                console.say(
                    DONE_SENTENCE if record.state == outcome.DONE or not record.sentence
                    else one_line(record.sentence, ENDING_LIMIT)
                )
                return 0
            sleep(0.5)
        console.say(APP_SENTENCE)
        return 0

    console.paragraph(CONSOLE_START)
    token = _token(home)
    seen = 0
    heartbeat = started
    while True:
        status = door_status(token)
        if token is None or status is None:
            token = _token(home)  # the pairing appears once the engine has started
        if status is not None and status.get("running") is True and token is not None:
            seen = follow(token, console, seen)
            sleep(POLL_SECONDS)
            continue
        record = recorded(home)
        if is_fresh(record, since):
            assert record is not None
            console.ending(record)
            return 0
        presence = status.get("presence") if status is not None else None
        owner = presence.get("owner") if isinstance(presence, dict) else None
        if owner == "wsl-unit":
            console.paragraph(ALREADY_SENTENCE)
            return 0
        if owner == "found":
            console.paragraph(FOUND_SENTENCE)
            return 0
        waited = clock() - started
        if status is None and waited >= DECISION_SECONDS and not door_alive():
            console.paragraph(TRAY_GONE_SENTENCE)
            return 0
        if status is None and waited < BLIND_SECONDS:
            # Nothing can say whether a move is running, so only its ending in
            # the file can end the wait.
            if clock() - heartbeat >= HEARTBEAT_SECONDS:
                heartbeat = clock()
                console.say(f"  still working ({waited / 60:.0f} min so far)")
        elif waited >= DECISION_SECONDS:
            if record is None:
                console.paragraph(UNDECIDED_SENTENCE)
                return 0
            # #28: history, said as history.
            console.paragraph(
                f"Nothing new has happened yet. The last time Crucible set up its "
                f"Linux engine here ({_local(record.at)}, Crucible {record.release}), "
                "it ended like this:"
            )
            console.ending(record)
            return 0
        sleep(POLL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible.host.installwatch",
        description="Print the engine move's progress and ending (install.ps1's last step).",
    )
    parser.add_argument("--home", required=True, help="the host home, CRUCIBLE_HOME")
    parser.add_argument("--since", required=True, help="ISO-8601 UTC: when this install started the tray")
    parser.add_argument("--brief", action="store_true", help="an app is reading: wait briefly, then hand over")
    args = parser.parse_args(argv)
    since = _parse_time(args.since)
    if since is None:
        parser.error(f"--since {args.since!r} is not an ISO-8601 time")
    try:
        return watch(Path(args.home), since, brief=args.brief)
    except KeyboardInterrupt:
        # Closing the window is allowed; the tray carries on without it.
        return 0


if __name__ == "__main__":
    sys.exit(main())
