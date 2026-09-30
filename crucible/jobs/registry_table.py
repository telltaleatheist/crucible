from __future__ import annotations

from ..jobtypes import JOB_TYPE_SPECS
from . import align, asr, audio, denoise, echo, image, llm, rvc, segment, tts
from .alignlongform import jobtype as alignlongform
from .binding import JobTypeBinding

REGISTRY_TABLE: tuple[JobTypeBinding, ...] = (
    *echo.JOB_TYPES,
    *llm.JOB_TYPES,
    *tts.JOB_TYPES,
    *asr.JOB_TYPES,
    *align.JOB_TYPES,
    *alignlongform.JOB_TYPES,
    *rvc.JOB_TYPES,
    *denoise.JOB_TYPES,
    *image.JOB_TYPES,
    *audio.JOB_TYPES,
    *segment.JOB_TYPES,
)

if tuple(binding.spec for binding in REGISTRY_TABLE) != JOB_TYPE_SPECS:
    raise TypeError(
        "crucible/jobs/registry_table.py binds "
        f"{[binding.spec.name for binding in REGISTRY_TABLE]} and "
        "crucible/jobtypes.py declares "
        f"{[spec.name for spec in JOB_TYPE_SPECS]}; every declared job type needs "
        "exactly one factory, in the declared order"
    )

__all__ = ["REGISTRY_TABLE"]
