from __future__ import annotations

import asyncio
import base64
import binascii
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .engines import EngineError, NarratorEngine
from .jobs.base import utcnow
from .errors import ApiError, JobCancelled, JobError
from .narratorengines import HIGGS_V3
from .narratorvoices import take_sampling
from .residency import KIND_TTS, Residency, describe_resident
from .voices import VoiceManifest

__all__ = [
    "GRACE_SECONDS",
    "STREAM_BATCH_WIDTH",
    "StreamManager",
    "StreamSession",
    "batch_width_for",
]

GRACE_SECONDS = 15.0

WATCHDOG_POLL_SECONDS = 0.25

CLOSE_JOIN_SECONDS = 30.0

STREAM_SILENCE_TIMEOUT_SECONDS = 600.0

DURATION_TOLERANCE_SECONDS = 0.05

STREAM_BATCH_WIDTH = {HIGGS_V3: 1}

BATCH_COALESCE_SECONDS = 0.025


def batch_width_for(narrator_engine: str) -> int:
    width = STREAM_BATCH_WIDTH.get(narrator_engine)
    if width is None:
        raise ApiError(
            500,
            "unknown_narrator_engine",
            f"no measured streaming batch width for narrator engine "
            f"{narrator_engine!r}; this build knows "
            f"{sorted(STREAM_BATCH_WIDTH)}",
        )
    return width


PENDING = "pending"
RUNNING = "running"
FINISHED = "finished"


@dataclass
class _Row:
    id: str
    text: str
    take: int
    sampling: dict[str, Any] | None
    slot: int
    state: str = PENDING
    seq: int = 0
    seconds: float = 0.0
    cancel_requested: bool = False
    retired: bool = False
    capped: bool | None = None
    gap_sec: float | None = None
    failure: str | None = None

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class _Event:
    id: int
    event: str
    data: dict[str, Any]
    at: float


class StreamSession:
    def __init__(
        self,
        *,
        session_id: str,
        voice: str,
        language: str,
        fingerprint: str,
        sample_rate: int,
        backend: str,
        narrator_engine: str,
        client: str | None,
        engine: NarratorEngine,
        residency: Residency,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.id = session_id
        self.voice = voice
        self.client = client
        self.opened_at = utcnow()
        self.language = language
        self.fingerprint = fingerprint
        self.sample_rate = sample_rate
        self.backend = backend
        self.narrator_engine = narrator_engine
        self.batch_width = batch_width_for(narrator_engine)

        self._engine = engine
        self._residency = residency
        self._loop = loop

        self._state = threading.Lock()
        self._rows: dict[str, _Row] = {}
        self._pending: deque[str] = deque()
        self._inflight: dict[int, str] = {}
        self._next_slot = 0

        self._wake = threading.Event()
        self._closing: str | None = None
        self._closed = threading.Event()

        self._events: deque[_Event] = deque()
        self._next_event_id = 0
        self._floor = 0
        self._attached: list["_Reader"] = []

        self._ever_attached = False
        self._detached_at: float | None = time.monotonic()

        self._worker = threading.Thread(
            target=self._run, name=f"crucible-tts-stream-{session_id}", daemon=True
        )


    def _emit(self, event: str, data: dict[str, Any]) -> None:
        self._loop.call_soon_threadsafe(self._append, event, data)

    def _append(self, event: str, data: dict[str, Any]) -> None:
        self._next_event_id += 1
        self._events.append(
            _Event(id=self._next_event_id, event=event, data=data, at=time.monotonic())
        )
        if self._floor == 0:
            self._floor = self._next_event_id
        for reader in list(self._attached):
            reader.waiter.set()
        self._prune()

    def _prune(self) -> None:
        if not self._events:
            return
        horizon = time.monotonic() - GRACE_SECONDS
        cursors = [reader.delivered for reader in self._attached]
        floor_cursor = min(cursors) if cursors else self._next_event_id
        while self._events:
            oldest = self._events[0]
            if oldest.at >= horizon or oldest.id > floor_cursor:
                break
            self._events.popleft()
            self._floor = oldest.id + 1


    def progress_report(self) -> dict[str, Any]:
        with self._state:
            rows = list(self._rows.values())
        finished = sum(1 for row in rows if row.state == FINISHED)
        return {
            "said": len(rows),
            "finished": finished,
            "in_flight": len(rows) - finished,
            "seconds": round(sum(row.seconds for row in rows), 3),
            "chars": sum(row.chars for row in rows),
        }

    def check_replayable(self, delivered: int) -> None:
        if delivered < self._floor - 1:
            raise ApiError(
                409,
                "replay_unavailable",
                f"session {self.id} can no longer replay from event {delivered}: "
                f"the oldest frame it still holds is {self._floor}. A stream is "
                f"replayable for {GRACE_SECONDS:.0f}s after the last reader "
                "leaves",
                {"session_id": self.id, "oldest": self._floor},
            )

    def attach(self, delivered: int) -> "_Reader":
        reader = _Reader(waiter=asyncio.Event(), delivered=delivered)
        self._attached.append(reader)
        self._ever_attached = True
        self._detached_at = None
        return reader

    def detach(self, reader: "_Reader") -> None:
        if reader in self._attached:
            self._attached.remove(reader)
        if not self._attached:
            self._detached_at = time.monotonic()

    @property
    def ever_attached(self) -> bool:
        return self._ever_attached

    def grace_expired(self, now: float) -> bool:
        detached = self._detached_at
        return detached is not None and now - detached > GRACE_SECONDS

    def frames_after(self, delivered: int) -> list[_Event]:
        return [event for event in self._events if event.id > delivered]


    def start(self) -> None:
        self._worker.start()

    def say(
        self, row_id: str, text: str, take: int, sampling: dict[str, Any] | None
    ) -> str:
        if not self._ever_attached:
            raise ApiError(
                409,
                "stream_not_attached",
                f"session {self.id} has never had an event stream attached, so "
                "there is nowhere for this row's audio to go. Open "
                f"GET /v1/tts/stream/{self.id}/events first",
                {"session_id": self.id},
            )
        if take > 0 and not self._engine.announces_item_take():
            raise ApiError(
                409,
                "sampling_not_wired",
                f"take {take} resolves to sampling {sampling}, and the narrator "
                f"serving session {self.id} did not announce `itemTake` on "
                f"its ready line — it has no per-item rung channel, so this "
                f"row would be rendered at take 0, in take 0's seed lane, and "
                f"reported as take {take}. Re-resolve the tts env's narrator "
                f"pin to a bookforge commit carrying "
                f"narrator/engine/item_sampling.py. Take 0 says fine.",
                {"session_id": self.id, "id": row_id, "take": take},
            )
        with self._state:
            if self._closing is not None:
                raise ApiError(
                    409,
                    "stream_closing",
                    f"session {self.id} is closing ({self._closing}) and will "
                    "not take new rows",
                    {"session_id": self.id},
                )
            if row_id in self._rows:
                raise ApiError(
                    400,
                    "duplicate_row_id",
                    f"session {self.id} already has a row {row_id!r}. The id is "
                    "the only thing that says which row a frame is about, so two "
                    "rows sharing one would be two streams of audio under one "
                    "name",
                    {"session_id": self.id, "id": row_id},
                )
            row = _Row(
                id=row_id, text=text, take=take, sampling=sampling,
                slot=self._next_slot,
            )
            self._next_slot += 1
            self._rows[row_id] = row
            self._pending.append(row_id)
        self._wake.set()
        return row_id

    def cancel(self, row_id: str) -> str:
        with self._state:
            row = self._rows.get(row_id)
            if row is None:
                raise ApiError(
                    404,
                    "unknown_row",
                    f"session {self.id} has no row {row_id!r}",
                    {"session_id": self.id, "id": row_id},
                )
            if row.state == FINISHED or row.retired:
                return "already_finished"
            row.cancel_requested = True
            outcome = "dropped" if row.state == PENDING else "aborting_batch"
        self._wake.set()
        return outcome

    def cancel_all(self) -> int:
        with self._state:
            live = [
                row for row in self._rows.values()
                if not row.retired and row.state != FINISHED
            ]
            for row in live:
                row.cancel_requested = True
        self._wake.set()
        return len(live)

    def begin_close(self, reason: str) -> None:
        with self._state:
            if self._closing is None:
                self._closing = reason
            for row in self._rows.values():
                if not row.retired:
                    row.cancel_requested = True
        self._wake.set()

    def join(self, timeout: float) -> bool:
        self._closed.wait(timeout)
        return self._closed.is_set()


    def _run(self) -> None:
        self._emit(
            "ready",
            {
                "voice": self.voice,
                "fingerprint": self.fingerprint,
                "sample_rate": self.sample_rate,
                "backend": self.backend,
            },
        )
        reason = "the session was closed"
        try:
            while True:
                batch = self._take_batch()
                if batch is None:
                    with self._state:
                        closing = self._closing
                    if closing is not None:
                        reason = closing
                        break
                    self._wake.wait(WATCHDOG_POLL_SECONDS)
                    self._wake.clear()
                    continue
                self._dispatch(batch)
        except EngineError as exc:
            reason = f"narrator failed: {exc}"
            self._emit("error", {"id": None, "code": "engine_failed", "message": str(exc)})
        except Exception as exc:
            reason = f"the session failed: {type(exc).__name__}: {exc}"
            self._emit(
                "error",
                {"id": None, "code": "stream_failed", "message": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            self._retire_everything_left(reason)
            self._emit("closed", {"reason": reason})
            try:
                self._residency.release(f"tts stream {self.id}")
            except JobError:
                pass
            self._closed.set()

    def _take_batch(self) -> list[_Row] | None:
        batch = self._claim_pending(self.batch_width)
        if not batch:
            return None
        deadline = time.monotonic() + BATCH_COALESCE_SECONDS
        while len(batch) < self.batch_width and time.monotonic() < deadline:
            more = self._claim_pending(self.batch_width - len(batch))
            if more:
                batch.extend(more)
            else:
                time.sleep(0.002)
        return batch

    def _claim_pending(self, room: int) -> list[_Row]:
        batch: list[_Row] = []
        with self._state:
            while self._pending and len(batch) < room:
                row = self._rows[self._pending.popleft()]
                if row.cancel_requested:
                    row.state = FINISHED
                    self._retire(row, cancelled=True)
                    continue
                row.state = RUNNING
                self._inflight[row.slot] = row.id
                batch.append(row)
        return batch

    def _dispatch(self, batch: list[_Row]) -> None:
        request = {
            "action": "generate_batch",
            "language": self.language,
            "items": [
                {"i": row.slot, "text": row.text, "stream": True,
                 "take": row.take}
                | ({} if row.sampling is None else {"sampling": row.sampling})
                for row in batch
            ],
        }
        cancelled_mid_batch = False
        try:
            for message in self._engine.converse(
                request,
                terminal=frozenset({"batch_done"}),
                silence_timeout=STREAM_SILENCE_TIMEOUT_SECONDS,
                cancelled=self._anything_cancelled,
            ):
                self._on_message(message)
        except JobCancelled:
            cancelled_mid_batch = True
        finally:
            self._abort_for_cancel(batch, cancelled_mid_batch)

    def _anything_cancelled(self) -> bool:
        with self._state:
            return any(
                self._rows[row_id].cancel_requested
                for row_id in self._inflight.values()
            )

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == "batch_chunk":
            self._on_chunk(message)
        elif kind == "batch_item":
            self._on_item(message)
        elif kind in ("batch_done", "stopped", "status"):
            return
        else:
            raise EngineError(
                f"narrator sent a {kind!r} message during a streamed "
                "generate_batch; this door knows batch_chunk, batch_item, "
                "batch_done, stopped and status"
            )

    def _row_for(self, message: dict[str, Any]) -> _Row | None:
        slot = message.get("i")
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise EngineError(
                f"narrator sent a {message['type']} whose `i` is {slot!r}, which "
                "is not a row slot. Rows retire out of order, so `i` is the only "
                "thing that says which row a reply is about"
            )
        with self._state:
            row_id = self._inflight.get(slot)
        if row_id is None:
            raise EngineError(
                f"narrator sent a {message['type']} for slot {slot}, which this "
                "session has no row in flight for"
            )
        row = self._rows[row_id]
        return None if row.retired else row

    def _on_chunk(self, message: dict[str, Any]) -> None:
        row = self._row_for(message)
        if row is None:
            return
        try:
            pcm, seconds = self._pcm_of(message, row)
        except _RowFailure as failure:
            self._fail(row, "narrator_protocol", str(failure))
            return
        with self._state:
            seq = row.seq
            row.seq += 1
            row.seconds += seconds
        self._emit(
            "audio",
            {
                "id": row.id,
                "seq": seq,
                "pcm_base64": base64.b64encode(pcm).decode("ascii"),
                "seconds": seconds,
            },
        )

    def _on_item(self, message: dict[str, Any]) -> None:
        row = self._row_for(message)
        if row is None:
            return
        failure = message.get("message")
        if isinstance(failure, str):
            row.state = FINISHED
            row.failure = failure
            return
        if message.get("cancelled") is True:
            row.state = FINISHED
            row.failure = "cancelled"
            return

        reported = message.get("duration")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            if abs(row.seconds - float(reported)) > DURATION_TOLERANCE_SECONDS:
                row.state = FINISHED
                self._fail(
                    row,
                    "narrator_protocol",
                    f"narrator reported {float(reported):.3f}s for row {row.id} "
                    f"but this session delivered {row.seconds:.3f}s of audio. A "
                    "reply that describes audio other than the audio attached to "
                    "it is not a measurement",
                )
                return
        capped = message.get("capped")
        row.capped = capped if isinstance(capped, bool) else None
        gap = message.get("gapSec")
        if not isinstance(gap, (int, float)) or isinstance(gap, bool) or gap < 0:
            row.state = FINISHED
            self._fail(
                row,
                "narrator_protocol",
                f"narrator retired row {row.id} with gapSec {gap!r}, which is not "
                "a gap in seconds. The player realizes the silence between rows on "
                "this door, so narrator has to state the one it classified; a "
                "narrator without the field is older than this server",
            )
            return
        row.gap_sec = float(gap)
        row.state = FINISHED
        self._retire(row, cancelled=False)

    def _abort_for_cancel(self, batch: list[_Row], cancelled_mid_batch: bool) -> None:
        with self._state:
            inflight = list(self._inflight.items())
            self._inflight.clear()
        for slot, row_id in inflight:
            row = self._rows[row_id]
            if row.retired:
                continue
            failure = row.failure
            if failure is None:
                self._fail(
                    row,
                    "narrator_protocol",
                    f"narrator ended the batch with row {row.id} (slot {slot}) "
                    "unanswered. One answer per row is its own guarantee, so a "
                    "short batch is not a short answer",
                )
                continue
            if row.cancel_requested:
                self._retire(row, cancelled=True)
                continue
            if not cancelled_mid_batch:
                self._fail(row, "row_failed", failure)
                continue
            row.state = PENDING
            row.failure = None
            self._emit(
                "restart",
                {
                    "id": row.id,
                    "from_seq": row.seq,
                    "reason": (
                        "the batch this row was in was aborted to cancel another "
                        "row; narrator has no per-row cancel, so the audio "
                        f"already sent for {row.id} is void and it starts again"
                    ),
                },
            )
            with self._state:
                row.seconds = 0.0
                self._pending.appendleft(row.id)
            self._wake.set()

    def _retire_everything_left(self, reason: str) -> None:
        with self._state:
            open_rows = [row for row in self._rows.values() if not row.retired]
            self._pending.clear()
            self._inflight.clear()
        for row in open_rows:
            row.state = FINISHED
            self._retire(row, cancelled=True, note=reason)


    def _retire(self, row: _Row, *, cancelled: bool, note: str | None = None) -> None:
        if row.retired:
            return
        row.retired = True
        row.state = FINISHED
        seconds = row.seconds
        data: dict[str, Any] = {
            "id": row.id,
            "seconds": seconds,
            "chars": row.chars,
            "chars_per_sec": (row.chars / seconds) if seconds > 0 else None,
            "capped": row.capped,
            "cancelled": cancelled,
            "gap_sec": row.gap_sec,
        }
        if note is not None:
            data["note"] = note
        self._emit("done", data)

    def _fail(self, row: _Row, code: str, message: str) -> None:
        if row.retired:
            return
        row.retired = True
        row.state = FINISHED
        self._emit("error", {"id": row.id, "code": code, "message": message})

    def _pcm_of(self, message: dict[str, Any], row: _Row) -> tuple[bytes, float]:
        if message.get("format") != "pcm16":
            raise _RowFailure(
                f"narrator sent format {message.get('format')!r}, not 'pcm16'; "
                "Crucible streams signed 16-bit little-endian mono and will not "
                "guess at another layout"
            )
        reported_rate = message.get("sampleRate")
        if reported_rate != self.sample_rate:
            raise _RowFailure(
                f"narrator streamed this chunk at {reported_rate!r} Hz while "
                f"{self.voice} was loaded at {self.sample_rate}. Crucible "
                "refuses rather than resamples"
            )
        payload = message.get("data")
        if not isinstance(payload, str):
            raise _RowFailure(f"narrator sent no base64 audio for row {row.id}")
        try:
            pcm = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _RowFailure(
                f"narrator's audio for row {row.id} is not base64: {exc}"
            ) from None
        if not pcm or len(pcm) % 2:
            raise _RowFailure(
                f"narrator sent {len(pcm)} bytes for row {row.id}, which is not a "
                "whole number of 16-bit samples"
            )
        measured = len(pcm) / 2 / self.sample_rate
        reported = message.get("duration")
        if not isinstance(reported, (int, float)) or isinstance(reported, bool):
            raise _RowFailure(
                f"narrator reported duration {reported!r} for a chunk of row "
                f"{row.id}, which is not a duration"
            )
        if abs(measured - float(reported)) > DURATION_TOLERANCE_SECONDS:
            raise _RowFailure(
                f"narrator reported {float(reported):.3f}s for a chunk of row "
                f"{row.id} but sent {measured:.3f}s of audio"
            )
        return pcm, measured


class _RowFailure(Exception):
    ...


@dataclass
class _Reader:
    waiter: asyncio.Event
    delivered: int


class StreamManager:
    def __init__(
        self, residency: Residency, on_closed: Callable[[str], Any] | None = None
    ) -> None:
        self._residency = residency
        self._on_closed = on_closed
        self._lock = threading.Lock()
        self._session: StreamSession | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_watchdog = threading.Event()

    def when_closed(self, on_closed: Callable[[str], Any]) -> None:
        self._on_closed = on_closed

    @property
    def session(self) -> StreamSession | None:
        return self._session

    def get(self, session_id: str) -> StreamSession:
        session = self._session
        if session is None or session.id != session_id:
            raise ApiError(
                404,
                "unknown_session",
                f"there is no streaming session {session_id!r} on this server"
                + ("" if session is None else f"; the open one is {session.id!r}"),
                {"session_id": session_id},
            )
        return session

    def open(
        self,
        *,
        voice: str,
        language: str,
        manifest: VoiceManifest,
        client: str | None,
        loop: asyncio.AbstractEventLoop,
    ) -> StreamSession:
        residency = self._residency
        with self._lock:
            existing = self._session
            if existing is not None:
                raise ApiError(
                    409,
                    "stream_session_open",
                    f"session {existing.id!r} is already streaming {existing.voice!r} "
                    "on this server, and a session holds the resident voice's "
                    "whole attention. Close it first",
                    {"session_id": existing.id, "voice": existing.voice},
                )
            try:
                residency.refuse_if_stopping(f"stream {voice!r}")
            except JobError as exc:
                raise ApiError(409, exc.code, exc.message) from None
            resident = residency.resident_voice
            if resident is None or resident.voice_id != voice:
                raise ApiError(
                    409,
                    "voice_not_resident",
                    f"{voice!r} is not resident on this server; "
                    + describe_resident(residency, KIND_TTS, "no voice is")
                    + ". The streaming door never loads a voice — post a "
                    "load-voice job first",
                    {"requested": voice, "resident": residency.resident_id},
                )
            engine = residency.voice_engine
            if engine is None:
                raise ApiError(
                    500,
                    "voice_not_resident",
                    f"{voice!r} is recorded as resident but there is no narrator "
                    "process serving it",
                )
            session_id = uuid.uuid4().hex
            holder = f"tts stream {session_id}"
            session = StreamSession(
                session_id=session_id,
                voice=voice,
                language=language,
                fingerprint=resident.fingerprint,
                sample_rate=resident.sample_rate,
                backend=resident.backend,
                narrator_engine=manifest.narrator_engine,
                client=client,
                engine=engine,
                residency=residency,
                loop=loop,
            )
            try:
                residency.claim(holder, may_mutate=False)
            except JobError as exc:
                raise ApiError(409, exc.code, exc.message) from None
            self._session = session
            self._ensure_watchdog()
        session.start()
        return session

    def close(self, session: StreamSession, reason: str) -> bool:
        session.begin_close(reason)
        finished = session.join(CLOSE_JOIN_SECONDS)
        if not finished:
            return False
        with self._lock:
            if self._session is session:
                self._session = None
        if self._on_closed is not None:
            self._on_closed(f"streaming session {session.id} closed")
        return True

    def shutdown(self) -> None:
        self._stop_watchdog.set()
        session = self._session
        if session is not None:
            self.close(session, "the server is shutting down")
        watchdog = self._watchdog
        if watchdog is not None:
            watchdog.join(timeout=WATCHDOG_POLL_SECONDS * 8)
            self._watchdog = None


    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._stop_watchdog.clear()
        self._watchdog = threading.Thread(
            target=self._watch, name="crucible-tts-stream-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watch(self) -> None:
        while not self._stop_watchdog.wait(WATCHDOG_POLL_SECONDS):
            session = self._session
            if session is None:
                return
            if session.grace_expired(time.monotonic()):
                self.close(
                    session,
                    f"no event stream reattached within {GRACE_SECONDS:.0f}s of "
                    "the last one dropping",
                )


def require_streamable(manifest: VoiceManifest, backend_kind: str) -> None:
    if not manifest.supports(backend_kind):
        raise ApiError(
            400,
            "backend_unsupported",
            f"voice {manifest.id!r} has no {backend_kind} block",
        )


def require_sayable(
    manifest: VoiceManifest, take: int
) -> dict[str, Any] | None:
    return take_sampling(manifest, take)
