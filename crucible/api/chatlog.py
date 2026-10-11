"""One stderr line per chat completion the chat door relays, streamed or not, as the
decide, embed and rerank doors print one per call: which model and how it was chosen,
for whom, how long it waited and ran, its tokens, why it stopped, and what bounded it.
Never what was said: no message, no answer, no schema."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .. import verbmodel
from ..structured import constrained_fields

HOW = {
    verbmodel.MODEL: "named",
    verbmodel.CEILING: "ceiling",
    verbmodel.SETTINGS: "server pick, settings",
    verbmodel.AUTO: "server pick, auto",
}


def _constraint(body: dict[str, Any]) -> str:
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and isinstance(response_format.get("type"), str):
        return f"response_format {response_format['type']}"
    fields = constrained_fields(body)
    return ", ".join(sorted(fields)) if fields else "none"


@dataclass
class ChatLog:
    model: str
    source: str
    client: str | None
    arrived: float
    max_tokens: Any
    constraint: str
    stream: bool
    started: float | None = None
    status: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    _buffer: bytes = field(default=b"", repr=False)

    @classmethod
    def of(
        cls, body: dict[str, Any], model: str, source: str, client: str | None, arrived: float
    ) -> "ChatLog":
        return cls(
            model=model, source=source, client=client, arrived=arrived,
            max_tokens=body.get("max_tokens", body.get("max_completion_tokens")),
            constraint=_constraint(body), stream=body.get("stream") is True,
        )

    def sent(self) -> None:
        self.started = time.monotonic()

    def _read(self, document: Any) -> None:
        if not isinstance(document, dict):
            return
        choices = document.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
            if isinstance(reason, str):
                self.finish_reason = reason
        usage = document.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                self.cached_tokens = details.get("cached_tokens", self.cached_tokens)

    def completion(self, status: int, content: bytes) -> None:
        """A whole answer (not streamed)."""
        self.status = status
        if status != 200:
            return
        try:
            self._read(json.loads(content))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

    def chunk(self, status: int, chunk: bytes) -> None:
        """One piece of a streamed answer, read for its frames' finish_reason and usage."""
        self.status = status
        if status != 200:
            return
        self._buffer += chunk
        while b"\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    continue
                try:
                    self._read(json.loads(data))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    def line(self, outcome: str | None = None) -> str:
        now = time.monotonic()
        started = self.started if self.started is not None else now
        tokens = (
            "tokens unreported"
            if self.prompt_tokens is None and self.completion_tokens is None
            else f"{self.prompt_tokens} in"
            + ("" if self.cached_tokens is None else f" ({self.cached_tokens} cached)")
            + f", {self.completion_tokens} out"
        )
        ended = outcome if outcome is not None else (
            f"finish {self.finish_reason}" if self.status == 200 else f"engine status {self.status}"
        )
        return (
            f"crucible: chat on {self.model!r} ({HOW.get(self.source, self.source)}) for "
            f"{self.client or 'an unnamed client'}: waited {(started - self.arrived) * 1000:.0f} "
            f"ms, answered in {(now - started) * 1000:.0f} ms; {tokens}; {ended}; max_tokens "
            f"{self.max_tokens if self.max_tokens is not None else 'unset'}; "
            f"{self.constraint}{'; streamed' if self.stream else ''}"
        )

    def report(self, outcome: str | None = None) -> None:
        print(self.line(outcome), file=sys.stderr)


__all__ = ["ChatLog"]
