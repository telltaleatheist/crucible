from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import textwrap
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, TextIO

from .. import controller_client, local
from ..platform.paths import LOG_NAME, door_url
from . import outcome
from .errors import HostError
from .installer import STEP_WORDS, TRY_AGAIN_HINT
from .log import plain

BRIEF_SECONDS = 10.0

DECISION_SECONDS = 120.0

BLIND_SECONDS = 45 * 60.0
HEARTBEAT_SECONDS = 120.0

POLL_SECONDS = 1.0

STREAM_READ_SECONDS = 300.0

LINE_LIMIT = 300

ENDING_LIMIT = 1500

WIDTH = 78

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

STARTING_TRAY_SENTENCE = (
    "Crucible's controller (the background process behind the icon by the "
    "clock) is not answering, so this window is starting it."
)

TRAY_GONE_SENTENCE = (
    "Crucible's controller (the background process behind the icon by the "
    "clock) is not answering and could not be started from here, so nothing can carry on the setup right "
    "now. Sign out of Windows and sign back in: it starts again at sign-in "
    "and carries on by itself. Its log is:"
)

CONTROLLER_START_SECONDS = controller_client.START_SECONDS


def start_controller(home: Path) -> bool:
    try:
        local._spawn_controller(home)
    except (OSError, AttributeError, ValueError):
        return False
    return True

RESTART_BANNER = 'ACTION NEEDED: restart Windows with "Update and restart".'

RESTART_CODES = outcome.RESTART_BANNER_CODES


def one_line(text: object, limit: int = LINE_LIMIT) -> str:
    flat = " ".join(plain(str(text)).split())
    return flat if len(flat) <= limit else "..." + flat[-limit:]


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _local(value: str) -> str:
    parsed = _parse_time(value)
    return value if parsed is None else parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def recorded(home: Path) -> outcome.Outcome | None:
    try:
        return outcome.read(home)
    except HostError:
        return None


def is_fresh(record: outcome.Outcome | None, since: datetime) -> bool:
    if record is None:
        return False
    at = _parse_time(record.at)
    return at is not None and at >= since.replace(microsecond=0)


def ending(record: outcome.Outcome) -> list[str]:
    sentence = one_line(record.sentence, ENDING_LIMIT) if record.sentence else ""
    if record.state == outcome.DONE:
        return [DONE_SENTENCE]
    if record.state == outcome.DECLINED:
        return [DECLINED_SENTENCE]
    if record.state == outcome.REBOOT_PENDING or record.code in RESTART_CODES:
        return [RESTART_BANNER, sentence]
    if record.state == outcome.CANNOT:
        return [
            "Crucible cannot set up its Linux engine on this PC yet:",
            sentence,
            "The Windows engine keeps working meanwhile. Once that is fixed, "
            + TRY_AGAIN_HINT + ".",
        ]
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
    def __init__(self, out: TextIO) -> None:
        self._out = out
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
        if text:
            self.say(textwrap.fill(plain(text), WIDTH, break_on_hyphens=False))

    def ending(self, record: outcome.Outcome) -> None:
        self.say("-" * WIDTH)
        for text in ending(record):
            self.paragraph(text)

    def tray_gone(self, home: Path) -> None:
        self.paragraph(TRAY_GONE_SENTENCE)
        self.say(f"  {Path(home) / LOG_NAME}")


def _token(home: Path) -> str | None:
    return controller_client.bearer(home)


def _open(path: str, token: str | None, timeout: float):
    return controller_client.open_url(door_url(path), token=token, timeout=timeout)


def door_alive() -> bool:
    return controller_client.is_up()


def door_status(token: str | None) -> dict[str, object] | None:
    if token is None:
        return None
    try:
        with _open("/install", token, 5.0) as response:
            value = json.load(response)
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return None
    return value if isinstance(value, dict) else None


def _envelopes(response) -> Iterator[dict]:
    for raw in response:
        try:
            envelope = json.loads(raw)
        except ValueError:
            continue
        if isinstance(envelope, dict):
            yield envelope


def _print_new(response, console: Console, seen: int) -> int:
    for envelope in _envelopes(response):
        number = envelope.get("id")
        if isinstance(number, int):
            if number <= seen:
                continue
            seen = number
        console.event(envelope)
    return seen


def follow(token: str, console: Console, seen: int) -> int:
    try:
        with _open("/install/events", token, STREAM_READ_SECONDS) as response:
            seen = _print_new(response, console, seen)
    except (OSError, socket.timeout, urllib.error.URLError, http.client.HTTPException):
        pass
    return seen


def _watch_briefly(
    home: Path, since: datetime, console: Console,
    clock: Callable[[], float], sleep: Callable[[float], None],
) -> int:
    started = clock()
    while clock() - started < BRIEF_SECONDS:
        record = recorded(home)
        if is_fresh(record, since):
            assert record is not None
            console.say(
                DONE_SENTENCE if record.state == outcome.DONE or not record.sentence
                else one_line(record.sentence, ENDING_LIMIT)
            )
            return 0
        sleep(0.5)
    console.say(APP_SENTENCE)
    return 0


class _Watch:
    def __init__(
        self, home: Path, since: datetime, console: Console, *,
        clock: Callable[[], float], alive: Callable[[], bool], start: Callable[[Path], bool],
    ) -> None:
        self._home = home
        self._since = since
        self._console = console
        self._clock = clock
        self._alive = alive
        self._start = start
        self._started = clock()
        self._heartbeat = self._started
        self._started_controller_at: float | None = None
        self._token = _token(home)
        self._seen = 0

    def tick(self) -> int | None:
        status = door_status(self._token)
        if self._token is None or status is None:
            self._token = _token(self._home)
        if status is not None and status.get("running") is True and self._token is not None:
            self._seen = follow(self._token, self._console, self._seen)
            return None
        record = recorded(self._home)
        if is_fresh(record, self._since):
            assert record is not None
            self._console.ending(record)
            return 0
        if self._settled_by_owner(status):
            return 0
        waited = self._clock() - self._started
        if status is None and waited >= DECISION_SECONDS and not self._alive():
            return self._start_controller_once()
        return self._wait_or_report(status is None, record, waited)

    def _settled_by_owner(self, status: dict[str, object] | None) -> bool:
        presence = status.get("presence") if status is not None else None
        owner = presence.get("owner") if isinstance(presence, dict) else None
        sentence = {"wsl-unit": ALREADY_SENTENCE, "found": FOUND_SENTENCE}.get(str(owner))
        if sentence is None:
            return False
        self._console.paragraph(sentence)
        return True

    def _start_controller_once(self) -> int | None:
        if self._started_controller_at is None:
            self._started_controller_at = self._clock()
            self._console.paragraph(STARTING_TRAY_SENTENCE)
            if self._start(self._home):
                return None
        elif self._clock() - self._started_controller_at < CONTROLLER_START_SECONDS:
            return None
        self._console.tray_gone(self._home)
        return 0

    def _wait_or_report(self, blind: bool, record: outcome.Outcome | None, waited: float) -> int | None:
        if blind and waited < BLIND_SECONDS:
            if self._clock() - self._heartbeat >= HEARTBEAT_SECONDS:
                self._heartbeat = self._clock()
                self._console.say(f"  still working ({waited / 60:.0f} min so far)")
            return None
        if waited < DECISION_SECONDS:
            return None
        if record is None:
            self._console.paragraph(UNDECIDED_SENTENCE)
            return 0
        self._console.paragraph(
            f"Nothing new has happened yet. The last time Crucible set up its "
            f"Linux engine here ({_local(record.at)}, Crucible {record.release}), "
            "it ended like this:"
        )
        self._console.ending(record)
        return 0


def watch(
    home: Path,
    since: datetime,
    *,
    brief: bool,
    out: TextIO = sys.stdout,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    alive: Callable[[], bool] = door_alive,
    start: Callable[[Path], bool] = start_controller,
) -> int:
    console = Console(out)
    if brief:
        return _watch_briefly(home, since, console, clock, sleep)
    console.paragraph(CONSOLE_START)
    watching = _Watch(home, since, console, clock=clock, alive=alive, start=start)
    while True:
        ended = watching.tick()
        if ended is not None:
            return ended
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
        return 0


if __name__ == "__main__":
    sys.exit(main())
