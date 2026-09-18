"""The streaming door — a session, an event stream, and posts. PHASE3-TTS.md section 7.

The Listen path, the in-app Play button, and the browser extension Owen uses
every Sunday. Its requirements are not the render door's with a smaller buffer;
they are different in kind: sub-sentence audio emitted *while a row is still
generating*, rows retired out of order within a batch, and a cancel that aborts
work in flight.

This file is everything section 7 said the engine seam does **not** give: the
session, the id allocation, the replay buffer, the grace window, the per-row
cancel, and the batching. `crucible/engines/narrator.py` gives the process
lifetime, `send()` for an out-of-band op while a request is in flight, and
`converse()` yielding `batch_chunk` lines in arrival order as they land. Those
three are the whole of what is borrowed.

Why this is not a WebSocket
---------------------------
Node 20 has no global `WebSocket` — it is behind `--experimental-websocket`
there and only ordinary in 22 — and Electron 33, which is what BookForge ships,
bundles Node 20.18 while the SDK runs in the **main** process. Checked on this
machine rather than assumed. So the door is SSE plus three posts, which is
`fetch` and `ReadableStream` and nothing else on either side. The gain is not
only the dependency: `Last-Event-ID` already works on this server's streams, so
a Listen connection that drops in a tunnel reattaches mid-sentence instead of
losing the row.

Two threads and one wire
------------------------
A session owns exactly one worker thread. It is the only thing in the process
that converses with narrator for the session's lifetime, because narrator has
one stdin and one stdout: `Residency.claimed()` is what stops the render door
and a `load-voice` from doing it at the same time, and a session holds that
claim from the moment it opens until the moment it closes.

Everything the worker has to say goes onto the event log through
`loop.call_soon_threadsafe`, exactly as `JobContext` marshals a job's progress —
so ordering is the order the worker emitted, and the SSE generators on the event
loop read a list that is never half-written.

The frames
----------
    ready   {voice, fingerprint, sample_rate, backend}
    audio   {id, seq, pcm_base64, seconds}
    restart {id, from_seq, reason}
    done    {id, seconds, chars, chars_per_sec, capped, cancelled}
    error   {id?, code, message}
    closed  {reason}

`restart` is this file's addition to section 7's list, and the reason is under
`_abort_for_cancel` below: it is the frame a client needs to know that the audio
it already holds for a row is void. Without it, resubmitting a row after a
per-row cancel would deliver the row's first seconds twice and no frame would
say so.

THIS DOOR IS UNGUARDED, AND THAT IS A DECISION
----------------------------------------------
There is no `guard` on `done` and there is not going to be one. The render door
carries the engine's retake verdict on every `chunk` event (Owen's ruling of
2026-09-13, `docs/PHASE6-REMOTE-RENDER.md` section 3); this door carries none,
because **nothing guards a streamed row in the first place**.

Owen ruled on the asymmetry directly, the same day: *"streaming can stay
unguarded. it needs speed over all else. i believe it's been unguarded this whole
time. if it becomes a problem we can add a guard later."* He is right that it has
always been so — narrator's serve world has never guarded a Higgs row — so this
is the status quo affirmed, not a regression accepted. **Nothing is owed here.** A
guard on this door would be a new decision, weighed against latency, not the
discharge of a debt.

The price, named here because it belongs next to the ruling: `_send_batch` below
sets `"stream": True` on **every row of every batch**, and narrator's
`serve/worker.py` routes any batch containing a streamed row past the guarded arm
entirely (`generate_batch`'s `if any(it.get('stream') is True ...)` test, which
comes before the `render_many` capability check). So it is not only the row
someone is listening to: the whole batch, read-ahead included, renders unguarded.
That is the shape of what was chosen — a retake doubles the latency a listener is
already sitting on, and this is the door where that cost is not payable.

What must NOT happen is a half-path toward it. A `guard` field on an unguarded
door would be worse than no field: it would read as "the engine looked at this
and was happy" on audio nothing looked at.
"""

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
from .narratorvoices import take_sampling
from .residency import KIND_TTS, Residency, describe_resident
from .voices import VoiceError, VoiceManifest

__all__ = [
    "GRACE_SECONDS",
    "STREAM_BATCH_WIDTH",
    "StreamManager",
    "StreamSession",
    "batch_width_for",
]

#: How long a session survives with nobody reading its event stream. A reconnect
#: carrying `Last-Event-ID` inside this window reattaches to the same session and
#: is replayed what it missed; when it closes, the session closes and every row
#: still in flight is cancelled. Work nobody is waiting for is time stolen from
#: the next job — the same rule as the `llm` proxy, for the same reason.
GRACE_SECONDS = 15.0

#: How often the watchdog looks for a session whose grace window has run out.
#: Far below the window, far above what asking costs.
WATCHDOG_POLL_SECONDS = 0.25

#: How long `close()` waits for the worker thread to put the wire down before it
#: gives up and leaves the session registered as still closing. narrator honours
#: a cancel by aborting what is in flight, so this is a wedge detector rather
#: than a budget; a session that exceeds it blocks the next one **by name**
#: rather than silently letting two conversations onto one pipe.
CLOSE_JOIN_SECONDS = 30.0

#: How long narrator may go without saying anything at all during a session. A
#: SILENCE timeout reset by every line, the render door's own number and for the
#: same reason: ten minutes is Owen's standing ceiling for a single Higgs chunk.
STREAM_SILENCE_TIMEOUT_SECONDS = 600.0

#: How far narrator's reported duration may sit from the duration of the PCM it
#: actually sent. One Higgs frame is 40 ms at 25 fps, so more than this is not
#: rounding. The render door's constant, applied to sub-sentence chunks too: the
#: server measures, and a reply describing audio other than the audio attached to
#: it is not a measurement.
DURATION_TOLERANCE_SECONDS = 0.05

#: How many rows the streaming door hands narrator in one `generate_batch`.
#:
#: There is no batching parameter on the wire — how many rows the engine runs at
#: once is engine tuning and belongs to the server — so these are the server's
#: numbers, and they are BookForge's measured ones rather than invented ones.
#: BookForge's worker pool coalesces a 25 ms window into
#: `min(STREAM_RAMP_WIDTH = 8, streamBatchCeiling())`, and the ceiling is
#: **`HIGGS_STREAM_BATCH_WIDTH = 1`** for Higgs, which was measured worthless
#: above one at 2.0x realtime (CLIENT-SURFACES.md section 3.3). So 1 is the
#: width actually run in production today, and 1 is the number here.
#:
#: A TABLE OF ONE, keyed by engine on purpose. Owen's ruling of 2026-09-14
#: removed `orpheus`, whose measured width was 8 (see
#: `voices.NARRATOR_ENGINE_SAMPLING`). A second engine adds a row, and
#: `batch_width_for` refuses one that has not — the width must be MEASURED, and
#: there is deliberately no default to fall back on.
#:
#: The width is also the **cost of a per-row cancel** — see `_abort_for_cancel`.
STREAM_BATCH_WIDTH = {"higgs-v3": 1}

#: How long a batch waits for more rows before it dispatches what it has.
#:
#: Found by testing rather than designed in: rows arrive one HTTP post at a
#: time, and a worker that dispatched the instant the first one landed put every
#: row in a batch of its own. The width above would then have been a number that
#: never happened, and with it the whole cost of a per-row cancel would have
#: looked free right up until the day it was not. BookForge has the same window
#: for the same reason and measured it at 25 ms (its worker pool's
#: `flushBatch()`, CLIENT-SURFACES.md section 3.3), so this is that number
#: rather than a new one.
#:
#: It costs nothing on `higgs-v3`, where the width is 1 and there is nothing to
#: coalesce: the wait is skipped entirely rather than added to every sentence's
#: latency.
BATCH_COALESCE_SECONDS = 0.025


def batch_width_for(narrator_engine: str) -> int:
    """`STREAM_BATCH_WIDTH` for this engine, or a refusal naming it.

    No default. A narrator engine nobody has measured a width for is an engine
    whose batching is unknown, and a guess is wrong in both directions: too low
    quietly halves throughput, too high quietly multiplies a cancel's cost.
    """
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


# ------------------------------------------------------------------- the rows


PENDING = "pending"
RUNNING = "running"
FINISHED = "finished"


@dataclass
class _Row:
    """One `say`, from the post that created it to the `done` that retires it."""

    #: The client's id for this row, and the only id on the wire. Rows retire out
    #: of order, so this is what every frame is keyed by.
    id: str
    text: str
    take: int
    #: This row's rung, in narrator's per-item spelling, or None at take 0 —
    #: which means "the loaded voice's own sampling" and is sent as no key at
    #: all. Resolved from the voice's ladder when the `say` was accepted, so a
    #: row that is resubmitted after a cancel (`restart`) goes back out under
    #: the numbers it was asked for rather than the voice's default.
    sampling: dict[str, Any] | None
    #: narrator's `i`. The client's ids are strings and narrator's batch keys are
    #: positions, so the session allocates one per row and never reuses it — a
    #: reused slot would land a resubmitted row's audio under the id of the row
    #: that used the slot before it.
    slot: int
    state: str = PENDING
    #: The next `seq` this row will emit. It never restarts, not even across a
    #: resubmission: see `restart` below, which says which seq the void ends at.
    seq: int = 0
    #: Seconds of audio actually delivered to the client for this row, measured
    #: from the bytes rather than read off narrator's reply.
    seconds: float = 0.0
    #: True once the client has asked for this row specifically to stop.
    cancel_requested: bool = False
    #: True once a frame has been emitted that ends this row, so a late reply for
    #: it is ignored rather than reported twice.
    retired: bool = False
    #: What narrator said about the frame cap on the retiring row, when it says
    #: anything at all. `None` means "narrator did not say" and is NEVER to be
    #: read as `false` (PHASE3-TTS.md section 6).
    capped: bool | None = None
    #: The `message` narrator retired this row with, if it retired it without
    #: audio. Held rather than acted on until the batch ends, because whether it
    #: means "this row failed" or "this row was collateral" is only knowable once
    #: it is known whether this session sent a cancel during the batch.
    failure: str | None = None

    @property
    def chars(self) -> int:
        """The server's own count of the text it sent, never a number off the wire."""
        return len(self.text)


# --------------------------------------------------------------- the session


@dataclass
class _Event:
    id: int
    event: str
    data: dict[str, Any]
    at: float


class StreamSession:
    """One open streaming session: its rows, its event log, and its worker.

    Created and closed by `StreamManager`, which is the thing that enforces one
    at a time. Everything public here is either called from the event loop (the
    API routes and the SSE generators) or from the worker thread, and the
    docstrings say which.
    """

    def __init__(
        self,
        *,
        session_id: str,
        voice: str,
        language: str,
        fingerprint: str,
        sample_rate: int,
        backend: str,
        max_chars: int,
        narrator_engine: str,
        client: str | None,
        engine: NarratorEngine,
        residency: Residency,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.id = session_id
        self.voice = voice
        #: The User-Agent that opened this session, recorded for the same reason
        #: `Job.client` is and read the same way: **null means it did not say**,
        #: never a name this server invented. A bench that guessed would be
        #: confidently wrong about who is on the card (PHASE7-LANES.md section 5).
        self.client = client
        #: Wall clock, in `Job.started`'s own format, because a bench puts the two
        #: side by side and a second time format would be a second thing to parse.
        self.opened_at = utcnow()
        self.language = language
        self.fingerprint = fingerprint
        self.sample_rate = sample_rate
        self.backend = backend
        self.max_chars = max_chars
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

        #: Set whenever there is something for the worker to do, or to notice.
        self._wake = threading.Event()
        self._closing: str | None = None
        self._closed = threading.Event()

        #: The event log. Bounded by the grace window rather than by a count —
        #: see `_prune`.
        self._events: deque[_Event] = deque()
        self._next_event_id = 0
        #: The oldest id still replayable. A `Last-Event-ID` below it is refused
        #: rather than skipped past: skipping would hand the client a hole in the
        #: audio and nothing would say so.
        self._floor = 0
        #: Every attached event stream. Each carries its own waiter, so two
        #: streams never steal each other's wakeup, and its own cursor, which is
        #: what `_prune` will not drop a frame below.
        self._attached: list["_Reader"] = []

        self._ever_attached = False
        self._detached_at: float | None = time.monotonic()

        self._worker = threading.Thread(
            target=self._run, name=f"crucible-tts-stream-{session_id}", daemon=True
        )

    # ------------------------------------------------------ the event log

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        """Append one frame. **Worker thread**, marshalled onto the loop."""
        self._loop.call_soon_threadsafe(self._append, event, data)

    def _append(self, event: str, data: dict[str, Any]) -> None:
        """**Event loop only** — the same rule `JobStore.append_event` states."""
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
        """Drop frames nobody can still want. **Event loop only.**

        A frame is droppable when it is older than the grace window **and** every
        attached reader has already been handed it. Both halves matter: the time
        bound is what makes a reattach inside the window able to replay, and the
        cursor bound is what stops a slow socket losing audio it has not read.

        This is why the log has no size limit. Audio is 48 KB/s of PCM and 64
        KB/s once base64'd, so a count would either be a number of seconds
        written as a number of frames — wrong the moment `CHUNK_MS` changes — or
        it would be big enough to hold a whole book in memory. The window is the
        real bound, and it is the one the contract already states.
        """
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

    # ------------------------------------------------------- readers, SSE

    def progress_report(self) -> dict[str, Any]:
        """What a bench can honestly say about this session. **Any thread.**

        THERE IS NO PERCENTAGE HERE, AND THERE NEVER CAN BE. A render job knows
        its own denominator: the client posted every chunk up front, so
        `fraction` is a real quantity and `Job.progress` is a real answer. A
        streaming session's rows arrive **one `say` at a time, indefinitely**, on
        a reader's whim or a browser extension's — there is no total, so any
        percentage would be a percentage of the work that has happened to arrive
        so far, which is a number that goes DOWN when more work arrives.

        Owen, 2026-09-13, on the three BookForge surfaces that stream rather than
        queue — the streaming page, the correct-sentences/re-roll page and the
        browser extension: *"those places are independent of a queue but claim a
        server while they run... that means crucible wont always have a percent
        complete to hand back."*

        So the wire says `progress: null` for a session and counts instead. The
        key is PRESENT and null, which is this server's one rule for "did not
        say" (see `sdk/ts/src/shape.ts`); an absent key would mean "this build
        does not speak the field", which is a different piece of news. A bench
        reading null draws a spinner and the counts, not a bar at 0%.
        """
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
        """Can this session still start a stream after event `delivered`?

        Asked **before** the StreamingResponse is built, so the answer can be an
        HTTP refusal with a body rather than an empty 200 that ends immediately.
        Crucible will not skip the gap: audio with a hole in it and nothing on
        the wire saying so is the exact failure this door exists to avoid.
        """
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
        """Open one event stream. **Event loop only.**

        `delivered` is `Last-Event-ID`: everything after it is replayed and then
        the stream follows live. A reattach inside the grace window lands here,
        and it is the one behaviour a WebSocket could not have given for free.
        """
        reader = _Reader(waiter=asyncio.Event(), delivered=delivered)
        self._attached.append(reader)
        self._ever_attached = True
        self._detached_at = None
        return reader

    def detach(self, reader: "_Reader") -> None:
        """Close one event stream. **Event loop only.** Starts the grace window."""
        if reader in self._attached:
            self._attached.remove(reader)
        if not self._attached:
            self._detached_at = time.monotonic()

    @property
    def ever_attached(self) -> bool:
        return self._ever_attached

    def grace_expired(self, now: float) -> bool:
        """Has nobody been reading for longer than the window? **Any thread.**"""
        detached = self._detached_at
        return detached is not None and now - detached > GRACE_SECONDS

    def frames_after(self, delivered: int) -> list[_Event]:
        """Everything with an id above `delivered`. **Event loop only.**"""
        return [event for event in self._events if event.id > delivered]

    # ---------------------------------------------------------- the ops

    def start(self) -> None:
        self._worker.start()

    def say(
        self, row_id: str, text: str, take: int, sampling: dict[str, Any] | None
    ) -> str:
        """Accept one row. **Event loop only.** Returns the row's id.

        `sampling` is the take's rung, already resolved by `require_sayable`
        against this voice's ladder — never a client's numbers, which do not
        exist on this wire.

        202 and the row's id, not the audio: a client that wants the audio reads
        the stream. A client that never opened one is refused here by name rather
        than generating into nothing.
        """
        if not self._ever_attached:
            raise ApiError(
                409,
                "stream_not_attached",
                f"session {self.id} has never had an event stream attached, so "
                "there is nowhere for this row's audio to go. Open "
                f"GET /v1/tts/stream/{self.id}/events first",
                {"session_id": self.id},
            )
        # THE SAME RUNG CHECK THE RENDER DOOR MAKES, for the same reason and
        # against the same narrator. `require_sayable` resolved this take
        # against the ladder, but resolving a rung is not delivering it: a
        # narrator built before `narrator/engine/item_sampling.py` reads an
        # item's `voice` and nothing else, so the rung is dropped in silence and
        # the row comes back at take 0 under take N's name. Measured on the
        # render door 2026-09-15 (two takes, byte-identical audio); this door
        # sends the rung through the very same `generate_batch` items, so it had
        # the very same hole. Refused per ROW, which is this door's grain, and
        # only above take 0 — take 0 sends the lane every narrator draws in and
        # no `sampling` key, and every narrator renders that correctly.
        #
        # ON THE TAKE, NOT ON THE SAMPLING — the render door's own correction
        # of 2026-09-15: a rung may move the seed and declare no numbers at
        # all, and such a rung would sail past a sampling-shaped gate.
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
        """Stop one row. **Event loop only.** Returns what it cost.

        See `_abort_for_cancel` for what "in flight" costs and why.
        """
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
                # Not an error: a cancel that arrives as the row retires is the
                # ordinary race on a live connection, and the answer is what
                # happened rather than a refusal.
                return "already_finished"
            row.cancel_requested = True
            outcome = "dropped" if row.state == PENDING else "aborting_batch"
        self._wake.set()
        return outcome

    def cancel_all(self) -> int:
        """Stop every row. **Event loop only.** Returns how many were still live."""
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
        """Ask the worker to put the wire down. **Any thread.**"""
        with self._state:
            if self._closing is None:
                self._closing = reason
            for row in self._rows.values():
                if not row.retired:
                    row.cancel_requested = True
        self._wake.set()

    def join(self, timeout: float) -> bool:
        """Wait for the worker to release the wire. **Never the event loop.**"""
        self._closed.wait(timeout)
        return self._closed.is_set()

    # -------------------------------------------------------- the worker

    def _run(self) -> None:
        """The session's whole conversation with narrator. **Worker thread.**"""
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
            # narrator died, or wrote something that is not a protocol message.
            # The session cannot continue on a wire that is gone, and saying so
            # by name is the whole of what is left to do.
            reason = f"narrator failed: {exc}"
            self._emit("error", {"id": None, "code": "engine_failed", "message": str(exc)})
        except Exception as exc:  # pragma: no cover — a bug here, surfaced not swallowed
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
            except JobError:  # pragma: no cover — released twice is a bug, not a state
                pass
            self._closed.set()

    def _take_batch(self) -> list[_Row] | None:
        """Up to `batch_width` rows to dispatch, after the coalescing window.

        The window is skipped when the width is 1, which is every voice this
        build ships: on `higgs-v3` there is nothing to coalesce and waiting
        25 ms would be 25 ms added to the first syllable of every sentence.
        """
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
        """Take up to `room` pending rows, dropping the cancelled ones.

        A row cancelled **before** narrator was told about it never starts, which
        is the exact half of per-row cancel: it costs nothing and takes nothing
        with it.
        """
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
        """One `generate_batch`, streamed, and every line it produces.

        `stream: true` on every item is the difference between this door and the
        render door, which sends no `stream` flag anywhere and takes narrator's
        pre-existing whole-row path byte for byte.

        **A batch here may MIX rungs**, unlike the render door's, where one
        `take` governs the whole job: rows arrive one `say` at a time and each
        carries its own. narrator's per-item channel is per item precisely for
        this; the MLX arm splits its slab by sampling group — and, since the
        seed lane landed, by TAKE as well, because one `mx.random.seed` serves
        a whole slab — rather than rendering anyone at another row's numbers or
        in another row's lane.

        `take` rides on every item including 0, and `sampling` only when the
        rung declares numbers; the render door's own item comment says why the
        two are spelled differently.
        """
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
            # `converse()` raises this once the terminal message has been seen
            # after it sent narrator a cancel, so the batch is over and every
            # reply is already in hand. This is the expected end of a cancelled
            # batch, not a failure.
            cancelled_mid_batch = True
        finally:
            self._abort_for_cancel(batch, cancelled_mid_batch)

    def _anything_cancelled(self) -> bool:
        """Does any in-flight row want to stop? **Worker thread, via converse.**"""
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
            # `stopped` is narrator's answer to the cancel this session sent, and
            # `batch_done` ends the iterator on its own. Neither says anything a
            # client needs.
            return
        else:
            raise EngineError(
                f"narrator sent a {kind!r} message during a streamed "
                "generate_batch; this door knows batch_chunk, batch_item, "
                "batch_done, stopped and status"
            )

    def _row_for(self, message: dict[str, Any]) -> _Row | None:
        """The row a reply is about, keyed by narrator's `i`. **Worker thread.**

        A reply for a slot nobody asked for is a protocol error rather than an
        extra frame — the same rule the render door states about `batch_item`,
        and for the same reason: `i` is the only identifier that crosses this
        wire in both directions.
        """
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
        """One sub-sentence chunk, measured and passed on. **Worker thread.**"""
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
                # Re-encoded rather than forwarded. narrator's own base64 is what
                # arrived, but `_pcm_of` has already decoded it to measure the
                # duration, and publishing the bytes that were measured is the
                # only way the `seconds` on this frame describes the audio on it.
                "pcm_base64": base64.b64encode(pcm).decode("ascii"),
                "seconds": seconds,
            },
        )

    def _on_item(self, message: dict[str, Any]) -> None:
        """A row retiring. **Worker thread.**

        narrator reports a per-row failure as a `message` and no `data`; a
        streamed row that finished carries `cancelled` and the duration it
        emitted. Both shapes end the row here, and which of the three endings it
        gets is decided in `_abort_for_cancel` for the rows a cancel touched.
        """
        row = self._row_for(message)
        if row is None:
            return
        failure = message.get("message")
        if isinstance(failure, str):
            # Left in flight deliberately: `_abort_for_cancel` decides whether
            # this is a row killed by this session's own cancel (resubmit) or a
            # row that failed on its own (report and retire).
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
        row.state = FINISHED
        self._retire(row, cancelled=False)

    def _abort_for_cancel(self, batch: list[_Row], cancelled_mid_batch: bool) -> None:
        """Decide what a per-row cancel did to every row in the aborted batch.

        **This is the honest implementation of the op narrator cannot do, and
        what follows is what it costs.**

        The contract says `{"op": "cancel", "id": "r12"}`. narrator has no such
        op: its `cancel` aborts **everything in flight**. Redefining the op to
        mean "cancel all" would be a lie on the wire, and reporting a row as
        cancelled while its neighbours silently died with it would be worse. So:

        - A row not yet handed to narrator is dropped before it starts
          (`_take_batch`). Exact, free, and the common case when a reader stops
          a queued paragraph.
        - A row **in flight** is stopped by aborting the batch it is in, and the
          survivors of that batch — rows nobody cancelled, which narrator killed
          along with it — are **resubmitted**.

        The cost is therefore the batch's width:

        - **higgs-v3: exact, and free.** `HIGGS_STREAM_BATCH_WIDTH = 1`
          (CLIENT-SURFACES.md section 3.3, measured worthless above one at 2.0x
          realtime), so the in-flight row IS the batch. Nothing survives to be
          resubmitted, and this branch never fires. It is the only engine in
          this build (`voices.NARRATOR_ENGINE_SAMPLING`), so the branch is
          written for the engine after it rather than exercised today.
        - **A width of N: up to N-1 other rows lose their progress.** The ramp
          dispatches N, so cancelling one in flight throws away whatever the
          others had generated and generates them again from the start.
          Measured in rows rather than in seconds, because what a row had done
          when the axe fell is not knowable from here.

        A resubmitted row keeps its id and its `seq` counter and gets a
        **`restart {id, from_seq}`** frame first: everything it has already sent
        under that id with a lower seq is void. Without that frame a client would
        concatenate the row's first seconds twice and nothing on the wire would
        say so. `seq` never restarts, so "ids strictly increase within a row"
        stays true across the restart, and `from_seq` is where the good audio
        begins.

        The alternative shapes were considered and are worse. Failing the
        survivors by name is honest but makes one reader's cancel destroy another
        paragraph the client still wants. Making the op mean `cancel_all` is the
        lie. Queueing the cancel until the batch ends is not a cancel.
        """
        with self._state:
            inflight = list(self._inflight.items())
            self._inflight.clear()
        for slot, row_id in inflight:
            row = self._rows[row_id]
            if row.retired:
                continue
            failure = row.failure
            if failure is None:
                # In flight when the iterator ended and narrator never retired
                # it. One answer per row is narrator's own guarantee, so this is
                # a short batch and a protocol failure rather than a partial
                # answer.
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
                # Nothing this session sent killed it, so this is the row's own
                # failure and it is reported as one. That is also what makes a
                # resubmission self-limiting: a row that genuinely fails comes
                # back to a batch with no cancel in it and is reported here.
                self._fail(row, "row_failed", failure)
                continue
            # A survivor. It lost its batch to somebody else's cancel.
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
        """Close every row that is still open. **Worker thread, on the way out.**"""
        with self._state:
            open_rows = [row for row in self._rows.values() if not row.retired]
            self._pending.clear()
            self._inflight.clear()
        for row in open_rows:
            row.state = FINISHED
            self._retire(row, cancelled=True, note=reason)

    # ------------------------------------------------------ retiring a row

    def _retire(self, row: _Row, *, cancelled: bool, note: str | None = None) -> None:
        """Emit this row's `done`. **Worker thread.**

        `seconds` is what this session measured in the PCM it delivered, and
        `chars` is the server's own count of the text it sent — never a number
        read back off a reply, for the reason `crucible/workers.py` gives about
        positional results.
        """
        if row.retired:
            return
        row.retired = True
        row.state = FINISHED
        seconds = row.seconds
        data: dict[str, Any] = {
            "id": row.id,
            "seconds": seconds,
            "chars": row.chars,
            # A cancelled row that emitted nothing has no rate, and 0.0 would
            # read as an infinitely slow narrator to a pace guard.
            "chars_per_sec": (row.chars / seconds) if seconds > 0 else None,
            "capped": row.capped,
            "cancelled": cancelled,
        }
        if note is not None:
            data["note"] = note
        self._emit("done", data)

    def _fail(self, row: _Row, code: str, message: str) -> None:
        """This row produced nothing usable. **Worker thread.**"""
        if row.retired:
            return
        row.retired = True
        row.state = FINISHED
        self._emit("error", {"id": row.id, "code": code, "message": message})

    def _pcm_of(self, message: dict[str, Any], row: _Row) -> tuple[bytes, float]:
        """This chunk's audio and the duration measured in it. **Worker thread.**"""
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
    """This chunk is not usable. Its row's neighbours still are."""


@dataclass
class _Reader:
    """One attached event stream: its own wakeup, and its own cursor.

    A waiter each, so two streams never steal each other's wakeup — the rule
    `JobStore.subscribe` already states. A cursor each because `_prune` will not
    drop a frame below the slowest of them.
    """

    waiter: asyncio.Event
    delivered: int


# --------------------------------------------------------------- the manager


class StreamManager:
    """One streaming session at a time, for one server instance.

    A session holds the resident voice's attention — literally, through
    `Residency.claimed()` — so a second is refused by name rather than queued.
    Two readers listening to one voice at once is not a feature anybody asked
    for, and pretending to offer it would mean two conversations on narrator's
    one pipe.
    """

    def __init__(
        self, residency: Residency, on_closed: Callable[[str], Any] | None = None
    ) -> None:
        self._residency = residency
        #: What to do once a session has really put the wire down — Owen's
        #: unload ruling asks here, because a closed session is one of the four
        #: ways the card stops being held (`crucible/settle.py`). Called on
        #: whichever thread closed the session, which is never the event loop,
        #: so it may block on an engine that takes its time stopping.
        self._on_closed = on_closed
        self._lock = threading.Lock()
        self._session: StreamSession | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_watchdog = threading.Event()

    def when_closed(self, on_closed: Callable[[str], Any]) -> None:
        """Say what happens after a session lets the card go.

        Set after construction because the settlement needs this manager (a
        session's claim is one of the facts it reads) and this manager needs the
        settlement — one of the two has to be wired second, and it is this one.
        """
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
        """Open the session, or refuse by name. **Event loop only.**"""
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
            resident = residency.resident_voice
            if resident is None or resident.voice_id != voice:
                # The streaming door never loads, exactly as chat never loads:
                # a connection is fine-grained and unattended, and two clients
                # alternating would thrash the card. Only the render job loads,
                # and PHASE3-TTS.md section 6 says why it is the exception.
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
            if engine is None:  # unreachable: a resident voice publishes its engine
                raise ApiError(
                    500,
                    "voice_not_resident",
                    f"{voice!r} is recorded as resident but there is no narrator "
                    "process serving it",
                )
            session_id = uuid.uuid4().hex
            holder = f"tts stream {session_id}"
            # BUILT BEFORE THE CARD IS CLAIMED, and that order is the fix rather
            # than an arrangement. The only thing that ever releases a stream
            # claim is the worker thread `start()` begins — a claim has no
            # expiry and no watchdog — so while the claim came first, every
            # refusal between the two lines held the card for the life of the
            # process, and every later load, unload and render was answered
            # `engine_in_use` naming a session that was never opened.
            #
            # Validating before claiming rather than unwinding after it, because
            # the constructor's one refusal is derivable from what is already in
            # hand: `batch_width_for(narrator_engine)`, which has deliberately
            # no default width. Moving the whole construction up states that
            # once rather than checking half of it twice, and leaves the claim
            # as the last thing here that can fail.
            session = StreamSession(
                session_id=session_id,
                voice=voice,
                language=language,
                fingerprint=resident.fingerprint,
                sample_rate=resident.sample_rate,
                backend=resident.backend,
                max_chars=resident.max_chars,
                narrator_engine=manifest.narrator_engine,
                client=client,
                engine=engine,
                residency=residency,
                loop=loop,
            )
            try:
                # `may_mutate=False`: the streaming door never loads a voice,
                # so no thread is exempted from the guard and a `load-voice`
                # arriving from anywhere is refused while this session is open.
                residency.claim(holder, may_mutate=False)
            except JobError as exc:
                raise ApiError(409, exc.code, exc.message) from None
            self._session = session
            self._ensure_watchdog()
        session.start()
        return session

    def close(self, session: StreamSession, reason: str) -> bool:
        """Close one session and wait for the wire. **Never the event loop.**

        Returns whether the worker actually put the wire down. It is a bool
        rather than a raise because a `DELETE` that reports "closing" is still a
        true answer; what must not happen is a new session opening onto a pipe
        the old worker is still reading, and that is prevented by leaving this
        one registered until it is really finished.
        """
        session.begin_close(reason)
        finished = session.join(CLOSE_JOIN_SECONDS)
        if not finished:
            # The worker still has the wire, so the claim is still held and the
            # card is still spoken for. Asking for a settlement now would be
            # asking a question whose answer is "no" by construction.
            return False
        with self._lock:
            if self._session is session:
                self._session = None
        if self._on_closed is not None:
            self._on_closed(f"streaming session {session.id} closed")
        return True

    def shutdown(self) -> None:
        """Close whatever is open. Called when the server exits."""
        self._stop_watchdog.set()
        session = self._session
        if session is not None:
            self.close(session, "the server is shutting down")
        watchdog = self._watchdog
        if watchdog is not None:
            watchdog.join(timeout=WATCHDOG_POLL_SECONDS * 8)
            self._watchdog = None

    # ------------------------------------------------------- the watchdog

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._stop_watchdog.clear()
        self._watchdog = threading.Thread(
            target=self._watch, name="crucible-tts-stream-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watch(self) -> None:
        """Close a session whose grace window has run out.

        The window is the reason this door is SSE. A dropped stream does not
        cancel immediately: it starts a 15-second window in which a reconnect
        carrying `Last-Event-ID` reattaches to the same session and is replayed
        what it missed, which is the difference between a tunnel costing a
        reconnect and costing a sentence. When it closes, the session closes and
        every row still in flight is cancelled, because work nobody is waiting
        for is time stolen from the next job.
        """
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


# --------------------------------------------------------------- refusals


def require_streamable(manifest: VoiceManifest, backend_kind: str) -> None:
    """What the streaming door refuses about the voice itself.

    **A zero-shot voice is no longer refused here** (2026-09-14). It was, as
    `voice_kind_unsupported`, because narrator's load message carried no
    reference clips and Crucible had no way to condition one. The load door
    now carries the clip (`params.reference`, section 5), and this door NEVER
    LOADS — it refuses `voice_not_resident` for anything that is not already
    on the card — so a `zeroshot` session can only ever attach to a voice that
    was loaded with its clip. There is nothing left here to refuse: refusing
    the kind anyway would make the load unusable by the one client that most
    wants it, the browser extension, whose whole path is this door.
    """
    if not manifest.supports(backend_kind):  # pragma: no cover — it is resident
        raise ApiError(
            400,
            "backend_unsupported",
            f"voice {manifest.id!r} has no {backend_kind} block",
        )
    # A VOICE WHOSE TAKE-0 SAMPLING DEVIATES IS NO LONGER REFUSED HERE. It was,
    # as `sampling_not_wired`, on the grounds that narrator's only sampling
    # channel was `register_voice_caps` and its vocabulary was the older
    # engine's. That stopped being true twice: take 0's sampling reaches
    # narrator through the voices document Crucible writes at every load
    # (section 4), and a rung above 0 reaches it per item (`require_sayable`
    # below).
    #
    # `sampling_not_wired` still exists, one door along, saying something else:
    # not "the contract has no channel" but "the narrator on this wire has
    # none". `say` asks the live engine (2026-09-15), because the tts env pins
    # narrator by commit and a pin may predate the channel. This function is
    # about the MANIFEST and has no engine, which is why the check is not here.


def require_sayable(
    manifest: VoiceManifest, take: int
) -> dict[str, Any] | None:
    """What a `say` may ask for, and the rung it resolves to.

    The render door's rules, one row at a time. Returns the take's sampling in
    narrator's per-item spelling, or `None` at take 0 — which is the loaded
    voice's own sampling and is what sending no key means.
    """
    try:
        manifest.take(take)
    except VoiceError as exc:
        # Never clamped to the last rung: a silent clamp is a retake ladder that
        # stops climbing without telling anyone.
        raise ApiError(
            400,
            "unknown_take",
            str(exc),
            {"voice": manifest.id, "take": take, "takes": len(manifest.takes)},
        ) from None
    return take_sampling(manifest, take)
