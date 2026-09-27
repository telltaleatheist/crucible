from __future__ import annotations

from typing import Any

from .. import ttsstream
from ..errors import ApiError
from ..narratorengines import HIGGS_V3
from ..narratorvoices import take_sampling
from ..voices import VoiceManifest

STREAM_BATCH_WIDTH = {HIGGS_V3: 1}


def batch_width_for(narrator_engine: str) -> int:
    widths = ttsstream.STREAM_BATCH_WIDTH
    width = widths.get(narrator_engine)
    if width is None:
        raise ApiError(
            500,
            "unknown_narrator_engine",
            f"no measured streaming batch width for narrator engine "
            f"{narrator_engine!r}; this build knows "
            f"{sorted(widths)}",
        )
    return width


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
