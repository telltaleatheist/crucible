from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...errors import ApiError, JobError
from ...segmentmodels import KIND_WORDS, SegmentBackendSpec, SegmentManifest

MAX_POINTS = 64

PROMPT_PARAMS: tuple[str, ...] = ("points", "box")


class SegmentPoint(BaseModel):
    """One click, in input pixels: `label` 1 keeps what is under it, 0 leaves it out."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(
        ge=0, description="Pixels from the input's left edge, under its width."
    )
    y: float = Field(
        ge=0, description="Pixels from the input's top edge, under its height."
    )
    label: Literal[0, 1] = Field(
        description="1 keeps what is under the point, 0 leaves it out."
    )


class SegmentParams(BaseModel):
    """`params` for a segment job. Unknown keys are refused; which model takes `points` and `box` is its class."""

    model_config = ConfigDict(extra="forbid")

    points: list[SegmentPoint] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_POINTS,
        description="`sam2.1-hiera-large` only: 1 to 64 clicks in the input's "
        "pixels, at least one label 1 unless there is a box. `birefnet` refuses it.",
    )
    box: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="`sam2.1-hiera-large` only: [x0, y0, x1, y1] in the input's "
        "pixels, top-left first, x1 > x0 and y1 > y0. `birefnet` refuses it.",
    )

    @field_validator("box")
    @classmethod
    def corners_in_order(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        x0, y0, x1, y1 = value
        if min(value) < 0:
            raise ValueError("box corners are pixels in the input picture and cannot be negative")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"box is [x0, y0, x1, y1] with x1 > x0 and y1 > y0 (the top-left corner "
                f"first); got {value}"
            )
        return value

    @model_validator(mode="after")
    def something_to_keep(self) -> "SegmentParams":
        if self.box is None and self.points is not None and not any(p.label == 1 for p in self.points):
            raise ValueError(
                "every point is label 0 (leave out) and there is no box, so nothing says "
                "what to select; add a label-1 point on the object, or a box around it"
            )
        return self


def _refusal(code: str, message: str, manifest: SegmentManifest, spec: SegmentBackendSpec, **extra: Any) -> ApiError:
    return ApiError(
        400,
        code,
        message,
        {"model": manifest.id, "kind": manifest.kind, "backend": spec.backend, **extra},
    )


def takes(manifest: SegmentManifest) -> list[str]:
    return [*(PROMPT_PARAMS if manifest.prompted else ())]


def _other_model(manifest: SegmentManifest, others: dict[str, SegmentManifest]) -> str:
    wanted = not manifest.prompted
    offered = sorted(m.id for m in others.values() if m.prompted == wanted)
    return " or ".join(offered) if offered else "a model of the other class"


def refuse_what_the_model_cannot_take(
    params: SegmentParams,
    manifest: SegmentManifest,
    spec: SegmentBackendSpec,
    catalog: dict[str, SegmentManifest],
) -> None:
    sent = [name for name in PROMPT_PARAMS if getattr(params, name) is not None]
    if not manifest.prompted and sent:
        param = sent[0]
        raise _refusal(
            "segment_param_unsupported",
            f"{manifest.id} does not take {param!r}: it makes "
            f"{KIND_WORDS[manifest.kind]}, from the picture alone. Drop {param!r}, or "
            f"send the job to {_other_model(manifest, catalog)} to select what you point at",
            manifest,
            spec,
            param=param,
            takes=takes(manifest),
        )
    if manifest.prompted and not sent:
        raise _refusal(
            "segment_param_missing",
            f"{manifest.id} makes {KIND_WORDS[manifest.kind]}, and this job points at "
            "nothing: send 'points' ([{\"x\": …, \"y\": …, \"label\": 1}], input pixels) "
            f"and/or 'box' ([x0, y0, x1, y1]). To cut out the main subject with no "
            f"pointing, send the job to {_other_model(manifest, catalog)}",
            manifest,
            spec,
            missing=list(PROMPT_PARAMS),
        )


def refuse_outside(params: SegmentParams, width: int, height: int, input_name: str) -> None:
    """A point or box past the picture's edge is refused by name before the model loads."""
    outside: list[str] = []
    for index, point in enumerate(params.points or ()):
        if point.x >= width or point.y >= height:
            outside.append(f"points[{index}] ({point.x:g}, {point.y:g})")
    if params.box is not None:
        x0, y0, x1, y1 = params.box
        if x0 >= width or y0 >= height or x1 > width or y1 > height:
            outside.append(f"box {params.box}")
    if outside:
        raise JobError(
            "segment_prompt_outside_picture",
            f"{', '.join(outside)} lies outside {input_name}, which is {width}x{height} "
            f"pixels: x runs 0 to {width - 1} and y 0 to {height - 1} from the top-left "
            "corner (a box may end on the edge). EXIF orientation is not applied: "
            "send coordinates in the picture's stored pixels",
        )


__all__ = [
    "MAX_POINTS",
    "PROMPT_PARAMS",
    "SegmentParams",
    "SegmentPoint",
    "refuse_outside",
    "refuse_what_the_model_cannot_take",
    "takes",
]
