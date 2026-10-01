from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

from .. import ttsstream
from ..cardkinds import KIND_TTS
from ..engines import NarratorEngine
from ..errors import ApiError, JobError
from ..residency import Residency, describe_resident
from ..voices import VoiceManifest
from .session import WATCHDOG_POLL_SECONDS, StreamSession

CLOSE_JOIN_SECONDS = 30.0


class StreamManager:
    """The one TTS stream session. It opens only inside a queue session
    (crucible/queuesessions.py), which is what holds the card for it; the residency claim
    it takes is the narrator's single conversation (one stdin, one stdout), not a hold
    on the card, so nothing here settles the card when a stream closes."""

    def __init__(self, residency: Residency) -> None:
        self._residency = residency
        self._lock = threading.Lock()
        self._session: StreamSession | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_watchdog = threading.Event()

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
            self._refuse_a_second_session()
            resident, engine = self._resident_narrator(voice)
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

    def _refuse_a_second_session(self) -> None:
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

    def _resident_narrator(self, voice: str) -> tuple[Any, NarratorEngine]:
        residency = self._residency
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
                + ". The stream opens on the resident voice; its queue session "
                "loads it first (a load-voice job in the session)",
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
        return resident, engine

    def close(
        self, session: StreamSession, reason: str, cause: dict[str, Any] | None = None
    ) -> bool:
        session.begin_close(reason, cause)
        finished = session.join(CLOSE_JOIN_SECONDS)
        if not finished:
            return False
        with self._lock:
            if self._session is session:
                self._session = None
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
                    f"no event stream reattached within {ttsstream.GRACE_SECONDS:.0f}s of "
                    "the last one dropping",
                )
