from __future__ import annotations

import asyncio
import base64
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .. import ttsstream
from ..clock import utcnow
from ..engines import EngineError, NarratorEngine
from ..errors import ApiError, JobCancelled, JobError
from ..residency import Residency
from . import decode
from .log import EventLog, Frame, Reader
from .validate import batch_width_for

WATCHDOG_POLL_SECONDS = 0.25

STREAM_SILENCE_TIMEOUT_SECONDS = 600.0

BATCH_COALESCE_SECONDS = 0.025

COALESCE_NAP_SECONDS = 0.002

PENDING = "pending"
RUNNING = "running"
FINISHED = "finished"


@dataclass
class Row:
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

    @property
    def live(self) -> bool:
        return not self.retired and self.state != FINISHED


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
        self._log = EventLog(session_id, loop)

        self._state = threading.Lock()
        self._rows: dict[str, Row] = {}
        self._pending: deque[str] = deque()
        self._inflight: dict[int, str] = {}
        self._next_slot = 0

        self._wake = threading.Event()
        self._closing: str | None = None
        self._closed = threading.Event()

        self._worker = threading.Thread(
            target=self._run, name=f"crucible-tts-stream-{session_id}", daemon=True
        )

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        self._log.emit(event, data)

    def check_replayable(self, delivered: int) -> None:
        self._log.check_replayable(delivered)

    def attach(self, delivered: int) -> Reader:
        return self._log.attach(delivered)

    def detach(self, reader: Reader) -> None:
        self._log.detach(reader)

    def grace_expired(self, now: float) -> bool:
        return self._log.grace_expired(now)

    def frames_after(self, delivered: int) -> list[Frame]:
        return self._log.frames_after(delivered)

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

    def start(self) -> None:
        self._worker.start()

    def say(
        self, row_id: str, text: str, take: int, sampling: dict[str, Any] | None
    ) -> str:
        self._refuse_unreachable_row(row_id, take, sampling)
        with self._state:
            self._refuse_row_while_locked(row_id)
            row = Row(
                id=row_id, text=text, take=take, sampling=sampling,
                slot=self._next_slot,
            )
            self._next_slot += 1
            self._rows[row_id] = row
            self._pending.append(row_id)
        self._wake.set()
        return row_id

    def _refuse_unreachable_row(
        self, row_id: str, take: int, sampling: dict[str, Any] | None
    ) -> None:
        if not self._log.ever_attached:
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

    def _refuse_row_while_locked(self, row_id: str) -> None:
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
            live = [row for row in self._rows.values() if row.live]
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
            reason = self._serve_until_closed()
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

    def _serve_until_closed(self) -> str:
        while True:
            batch = self._take_batch()
            if batch is not None:
                self._dispatch(batch)
                continue
            with self._state:
                closing = self._closing
            if closing is not None:
                return closing
            self._wake.wait(WATCHDOG_POLL_SECONDS)
            self._wake.clear()

    def _take_batch(self) -> list[Row] | None:
        batch = self._claim_pending(self.batch_width)
        if not batch:
            return None
        deadline = time.monotonic() + ttsstream.BATCH_COALESCE_SECONDS
        while len(batch) < self.batch_width and time.monotonic() < deadline:
            more = self._claim_pending(self.batch_width - len(batch))
            if more:
                batch.extend(more)
            else:
                time.sleep(COALESCE_NAP_SECONDS)
        return batch

    def _claim_pending(self, room: int) -> list[Row]:
        batch: list[Row] = []
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

    def _dispatch(self, batch: list[Row]) -> None:
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
                terminal=frozenset({decode.TERMINAL}),
                silence_timeout=STREAM_SILENCE_TIMEOUT_SECONDS,
                cancelled=self._anything_cancelled,
            ):
                self._on_message(message)
        except JobCancelled:
            cancelled_mid_batch = True
        finally:
            self._abort_for_cancel(cancelled_mid_batch)

    def _anything_cancelled(self) -> bool:
        with self._state:
            return any(
                self._rows[row_id].cancel_requested
                for row_id in self._inflight.values()
            )

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == decode.CHUNK:
            self._on_chunk(message)
        elif kind == decode.ITEM:
            self._on_item(message)
        elif kind not in decode.IGNORED:
            raise decode.refuse_unknown_kind(kind)

    def _row_for(self, message: dict[str, Any]) -> Row | None:
        slot = decode.slot_of(message)
        with self._state:
            row_id = self._inflight.get(slot)
        if row_id is None:
            raise decode.unknown_slot(message, slot)
        row = self._rows[row_id]
        return None if row.retired else row

    def _on_chunk(self, message: dict[str, Any]) -> None:
        row = self._row_for(message)
        if row is None:
            return
        try:
            pcm, seconds = decode.pcm_of(
                message, row.id, sample_rate=self.sample_rate, voice=self.voice
            )
        except decode.RowFailure as failure:
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
        end = decode.item_end(message, row.id, row.seconds)
        row.state = FINISHED
        if end.failure is not None:
            row.failure = end.failure
            return
        row.capped = end.capped
        if end.protocol_error is not None:
            self._fail(row, "narrator_protocol", end.protocol_error)
            return
        row.gap_sec = end.gap_sec
        self._retire(row, cancelled=False)

    def _abort_for_cancel(self, cancelled_mid_batch: bool) -> None:
        with self._state:
            inflight = list(self._inflight.items())
            self._inflight.clear()
        for slot, row_id in inflight:
            row = self._rows[row_id]
            if row.retired:
                continue
            if row.failure is None:
                self._fail(
                    row,
                    "narrator_protocol",
                    f"narrator ended the batch with row {row.id} (slot {slot}) "
                    "unanswered. One answer per row is its own guarantee, so a "
                    "short batch is not a short answer",
                )
            elif row.cancel_requested:
                self._retire(row, cancelled=True)
            elif not cancelled_mid_batch:
                self._fail(row, "row_failed", row.failure)
            else:
                self._restart(row)

    def _restart(self, row: Row) -> None:
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

    def _retire(self, row: Row, *, cancelled: bool, note: str | None = None) -> None:
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

    def _fail(self, row: Row, code: str, message: str) -> None:
        if row.retired:
            return
        row.retired = True
        row.state = FINISHED
        self._emit("error", {"id": row.id, "code": code, "message": message})
