"""What the whisper run and the Qwen3-ASR run build the same way.

`transcript.json`'s shape, the `decoding` progress event, and the one way a
worker failure becomes this job's failure. Both runs import it; neither
imports the other's module for it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ... import workers
from ...errors import JobError
from ..base import JobContext
from . import speechonly


@contextmanager
def worker_failed(*also: type[BaseException]) -> Iterator[None]:
    """A `WorkerError` (and any of `also`) inside becomes `JobError("worker_failed")`."""
    try:
        yield
    except (workers.WorkerError, *also) as exc:
        raise JobError("worker_failed", str(exc)) from None


def progress_decoding(ctx: JobContext, message: str, processed_s: float = 0.0) -> None:
    """A `decoding` event. It drives no fraction: none of the transcript exists
    yet. Every stage line carries the same three numbers, so a consumer reads
    one shape; a `total_s` of 0 is "the container has not been probed"."""
    ctx.progress(
        0.0,
        message,
        stage="decoding",
        processed_s=processed_s,
        total_s=0.0,
        cues=0,
    )


def transcript_document(
    *,
    model: str,
    spec: Any,
    engine: dict[str, Any],
    language: Any,
    language_probability: Any,
    language_requested: Any,
    vad_filter: bool,
    word_timestamps: bool,
    initial_prompt: str | None,
    prompt: dict[str, Any],
    duration_s: float,
    layout: dict[str, Any],
    speech: dict[str, Any] | None,
    timeline: "speechonly.Timeline | None",
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """`transcript.json`, in its one key order for every engine.

    `engine` is what an engine adds after the weights' identity (Qwen: its
    engine, dtype and aligner), `prompt` what it adds after `initial_prompt`
    (Qwen: `context`), and `layout` how the audio was cut (whisper's windows,
    Qwen's pieces).
    """
    return {
        "model": model,
        "revision": spec.revision,
        "hf_repo": spec.hf_repo,
        **engine,
        "language": language,
        "language_probability": language_probability,
        "language_requested": language_requested,
        "vad_filter": vad_filter,
        "word_timestamps": word_timestamps,
        "initial_prompt": initial_prompt,
        **prompt,
        "duration_s": duration_s,
        **layout,
        **speechonly.report(speech, timeline),
        "segments": segments,
    }
