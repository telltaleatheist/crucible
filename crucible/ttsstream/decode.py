from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

from ..engines import EngineError

DURATION_TOLERANCE_SECONDS = 0.05

CHUNK = "batch_chunk"
ITEM = "batch_item"
TERMINAL = "batch_done"
IGNORED = frozenset({TERMINAL, "stopped", "status"})


class RowFailure(Exception):
    ...


@dataclass(frozen=True)
class ItemEnd:
    failure: str | None = None
    protocol_error: str | None = None
    capped: bool | None = None
    gap_sec: float | None = None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def refuse_unknown_kind(kind: str) -> EngineError:
    return EngineError(
        f"narrator sent a {kind!r} message during a streamed "
        "generate_batch; this door knows batch_chunk, batch_item, "
        "batch_done, stopped and status"
    )


def slot_of(message: dict[str, Any]) -> int:
    slot = message.get("i")
    if not isinstance(slot, int) or isinstance(slot, bool):
        raise EngineError(
            f"narrator sent a {message['type']} whose `i` is {slot!r}, which "
            "is not a row slot. Rows retire out of order, so `i` is the only "
            "thing that says which row a reply is about"
        )
    return slot


def unknown_slot(message: dict[str, Any], slot: int) -> EngineError:
    return EngineError(
        f"narrator sent a {message['type']} for slot {slot}, which this "
        "session has no row in flight for"
    )


def pcm_of(
    message: dict[str, Any], row_id: str, *, sample_rate: int, voice: str
) -> tuple[bytes, float]:
    if message.get("format") != "pcm16":
        raise RowFailure(
            f"narrator sent format {message.get('format')!r}, not 'pcm16'; "
            "Crucible streams signed 16-bit little-endian mono and will not "
            "guess at another layout"
        )
    reported_rate = message.get("sampleRate")
    if reported_rate != sample_rate:
        raise RowFailure(
            f"narrator streamed this chunk at {reported_rate!r} Hz while "
            f"{voice} was loaded at {sample_rate}. Crucible "
            "refuses rather than resamples"
        )
    payload = message.get("data")
    if not isinstance(payload, str):
        raise RowFailure(f"narrator sent no base64 audio for row {row_id}")
    try:
        pcm = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RowFailure(
            f"narrator's audio for row {row_id} is not base64: {exc}"
        ) from None
    if not pcm or len(pcm) % 2:
        raise RowFailure(
            f"narrator sent {len(pcm)} bytes for row {row_id}, which is not a "
            "whole number of 16-bit samples"
        )
    measured = len(pcm) / 2 / sample_rate
    reported = message.get("duration")
    if not is_number(reported):
        raise RowFailure(
            f"narrator reported duration {reported!r} for a chunk of row "
            f"{row_id}, which is not a duration"
        )
    if abs(measured - float(reported)) > DURATION_TOLERANCE_SECONDS:
        raise RowFailure(
            f"narrator reported {float(reported):.3f}s for a chunk of row "
            f"{row_id} but sent {measured:.3f}s of audio"
        )
    return pcm, measured


def item_end(message: dict[str, Any], row_id: str, delivered_seconds: float) -> ItemEnd:
    failure = message.get("message")
    if isinstance(failure, str):
        return ItemEnd(failure=failure)
    if message.get("cancelled") is True:
        return ItemEnd(failure="cancelled")
    reported = message.get("duration")
    mismatch = is_number(reported) and (
        abs(delivered_seconds - float(reported)) > DURATION_TOLERANCE_SECONDS
    )
    if mismatch:
        return ItemEnd(
            protocol_error=(
                f"narrator reported {float(reported):.3f}s for row {row_id} "
                f"but this session delivered {delivered_seconds:.3f}s of audio. A "
                "reply that describes audio other than the audio attached to "
                "it is not a measurement"
            )
        )
    capped = message.get("capped")
    capped = capped if isinstance(capped, bool) else None
    gap = message.get("gapSec")
    if not is_number(gap) or gap < 0:
        return ItemEnd(
            capped=capped,
            protocol_error=(
                f"narrator retired row {row_id} with gapSec {gap!r}, which is not "
                "a gap in seconds. The player realizes the silence between rows on "
                "this door, so narrator has to state the one it classified; a "
                "narrator without the field is older than this server"
            ),
        )
    return ItemEnd(capped=capped, gap_sec=float(gap))
