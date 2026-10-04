"""A video job's params, and the arm's refusals by name.

Every refusal here happens before a worker starts: at submit for what the
params alone decide, and in `run` (before the session) for the one thing
that needs the inputs, the image-to-video token ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...errors import ApiError
from ...videomodels import (
    FRAME_STRIDE,
    IMAGE_TO_VIDEO,
    LATENT_SCALE,
    TEXT_TO_VIDEO,
    VideoBackendSpec,
    frames_for,
    video_tokens,
)

MAX_SEED = 2**32 - 1



class VideoParams(BaseModel):
    """`params` for a video job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        description="The shot, the motion, the light and the sound, in one "
        "paragraph (docs/VIDEO.md); not blank."
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Refused by name (`video_param_unsupported`): the distilled "
        "checkpoint runs without guidance and would never read it.",
    )
    width: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Frame width in pixels, sent with `height` or not at all (null: "
        "the model's default, 1280x704). A multiple of 32 (64 on a Mac), sides 256 "
        "to 1280, at most 901,120 pixels.",
    )
    height: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Frame height in pixels, sent with `width` or not at all; the "
        "same rules as `width`.",
    )
    duration_s: float | None = Field(
        default=None,
        gt=0.0,
        le=600.0,
        description="Seconds of clip, rounded to the model's 8k+1 frame grid; null "
        "is the model's default (5). Not with `num_frames`.",
    )
    num_frames: int | None = Field(
        default=None,
        ge=1,
        le=10_000,
        description="The exact frame count instead of `duration_s`; must be 8k+1 "
        "(49, 97, 121, …).",
    )
    fps: int | None = Field(
        default=None,
        ge=1,
        le=240,
        description="Frames per second, 24 or 25 for ltx-2.5-distilled; null is the "
        "model's default (24).",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SEED,
        description="0 to 4294967295; null lets the server choose one and report it. "
        "A seed reproduces a clip on the machine that made it.",
    )
    steps: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Must equal the model's fixed step count (8 for the distilled "
        "checkpoint) or be left out; any other value is refused.",
    )
    audio: bool = Field(
        default=True,
        description="false makes a silent clip: the sound is generated with the "
        "picture but not decoded.",
    )

    @field_validator("prompt")
    @classmethod
    def says_something(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError(
                "the prompt is empty; describe the shot, the motion, the light and "
                "the sound in one paragraph"
            )
        return value

    @model_validator(mode="after")
    def one_length(self) -> "VideoParams":
        if self.duration_s is not None and self.num_frames is not None:
            raise ValueError(
                "send duration_s or num_frames, not both; they are the same length "
                "said two ways"
            )
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "send width and height together, or neither for the model's default "
                "size; one side alone does not say the shape of the frame"
            )
        return self


@dataclass(frozen=True)
class Settled:
    mode: str
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    seed: int
    video_tokens: int

    @property
    def duration_s(self) -> float:
        return round(self.num_frames / self.fps, 3)


def _refusal(code: str, message: str, model: str, spec: VideoBackendSpec, **extra: Any) -> ApiError:
    return ApiError(400, code, message, {"model": model, "backend": spec.backend, **extra})


def _refuse_untaken(params: VideoParams, model: str, spec: VideoBackendSpec) -> None:
    if params.negative_prompt is not None:
        why = spec.why_not("negative_prompt") or "this arm has no negative prompt"
        raise _refusal(
            "video_param_unsupported",
            f"{model} on {spec.backend} does not take 'negative_prompt': {why}",
            model, spec, param="negative_prompt",
        )
    if params.steps is not None and params.steps != spec.steps:
        raise _refusal(
            "video_param_unsupported",
            f"{model} on {spec.backend} runs exactly {spec.steps} steps: it is the "
            f"distilled checkpoint, sampled on its own fixed {spec.steps}-sigma "
            f"schedule, and any other count would hand it a schedule it was not "
            f"trained on. Send steps {spec.steps} or leave steps out",
            model, spec, param="steps", steps=spec.steps,
        )
    if params.fps is not None and params.fps not in spec.fps:
        raise _refusal(
            "video_param_unsupported",
            f"{model} on {spec.backend} makes clips at {' or '.join(map(str, spec.fps))} "
            f"frames a second, not {params.fps}; other rates are untried here. "
            f"Send fps {spec.default_fps} or leave it out",
            model, spec, param="fps", fps=list(spec.fps),
        )


def _size(params: VideoParams, spec: VideoBackendSpec) -> tuple[int, int]:
    if params.width is None or params.height is None:
        return spec.default_width, spec.default_height
    return params.width, params.height


def _refuse_size(width: int, height: int, model: str, spec: VideoBackendSpec) -> None:
    details = {"width": width, "height": height}
    multiple = spec.size_multiple
    if width % multiple or height % multiple:
        why = f"its VAE works in {LATENT_SCALE}-pixel blocks"
        if multiple != LATENT_SCALE:
            why += f", and on {spec.backend} its first pass runs at half the size"
        raise _refusal(
            "video_size_not_supported",
            f"{model} on {spec.backend} needs width and height in multiples of "
            f"{multiple} ({why}); {width}x{height} is not. Round each "
            f"side to a multiple of {multiple}, for example 1280x704 for 16:9",
            model, spec, size_multiple=multiple, **details,
        )
    if min(width, height) < spec.min_side:
        raise _refusal(
            "video_size_not_supported",
            f"{width}x{height} has a side under {spec.min_side}; {model} makes no "
            f"frame smaller than {spec.min_side} pixels on a side",
            model, spec, min_side=spec.min_side, **details,
        )
    pixels = width * height
    if max(width, height) > spec.max_side or pixels > spec.max_pixels:
        raise _refusal(
            "video_too_large",
            f"{width}x{height} is {pixels:,} pixels a frame; {model} on {spec.backend} "
            f"makes at most {spec.max_pixels:,} with no side over {spec.max_side} "
            f"(1280x704 either way up), because its memory ({spec.memory_basis}: "
            f"{spec.memory_note}) was sized at that limit. Ask for a smaller frame",
            model, spec, max_pixels=spec.max_pixels, max_side=spec.max_side, **details,
        )


def _frames(params: VideoParams, fps: int, model: str, spec: VideoBackendSpec) -> int:
    if params.num_frames is not None:
        frames = params.num_frames
        if frames < FRAME_STRIDE + 1 or (frames - 1) % FRAME_STRIDE:
            below = max(FRAME_STRIDE + 1, (frames - 1) // FRAME_STRIDE * FRAME_STRIDE + 1)
            raise _refusal(
                "video_frames_not_supported",
                f"num_frames {frames} is not on {model}'s frame grid: its VAE packs "
                f"{FRAME_STRIDE} frames per latent after the first, so a clip is "
                f"{FRAME_STRIDE}k+1 frames. Send {below} or {below + FRAME_STRIDE}, "
                "or send duration_s and let the job pick",
                model, spec, num_frames=frames, frame_stride=FRAME_STRIDE,
            )
        return frames
    duration = params.duration_s if params.duration_s is not None else spec.default_duration_s
    return frames_for(duration, fps)


def _refuse_length(frames: int, fps: int, model: str, spec: VideoBackendSpec) -> None:
    if frames > spec.max_frames:
        raise _refusal(
            "video_too_long",
            f"{frames} frames is {frames / fps:.2f} s at {fps} fps; {model} on "
            f"{spec.backend} makes at most {spec.max_frames} frames "
            f"({spec.max_frames / fps:.2f} s at {fps} fps) because its memory "
            f"({spec.memory_basis}) was sized at that length. Ask for a shorter clip",
            model, spec, num_frames=frames, max_frames=spec.max_frames,
        )


def refuse_tokens(settled: Settled, model: str, spec: VideoBackendSpec) -> None:
    ceiling = spec.token_ceiling(settled.mode)
    if settled.video_tokens <= ceiling:
        return
    per_frame = (settled.width // LATENT_SCALE) * (settled.height // LATENT_SCALE)
    most_latents = ceiling // per_frame
    most_frames = max(0, (most_latents - 1) * FRAME_STRIDE + 1) if most_latents else 0
    longest = (
        f"At {settled.width}x{settled.height} the longest {settled.mode} clip is "
        f"{most_frames} frames ({most_frames / settled.fps:.2f} s at {settled.fps} fps)"
        if most_frames >= FRAME_STRIDE + 1
        else f"{settled.width}x{settled.height} is too large a frame for {settled.mode}"
    )
    why = (
        " (image-to-video carries a timestep per token, so it holds more on the "
        "card per frame than text-to-video)"
        if settled.mode == IMAGE_TO_VIDEO
        else ""
    )
    raise _refusal(
        "video_too_large",
        f"{settled.width}x{settled.height} x {settled.num_frames} frames is "
        f"{settled.video_tokens:,} video tokens; {model} on {spec.backend} denoises at "
        f"most {ceiling:,} for {settled.mode}{why}, the size its memory "
        f"({spec.memory_basis}) was sized at. {longest}; ask for fewer frames or a "
        "smaller frame",
        model, spec, video_tokens=settled.video_tokens, max_video_tokens=ceiling,
        mode=settled.mode,
    )


def settle(
    params: VideoParams, model: str, spec: VideoBackendSpec, seed: int, mode: str = TEXT_TO_VIDEO
) -> Settled:
    _refuse_untaken(params, model, spec)
    if mode == IMAGE_TO_VIDEO and not spec.image_to_video:
        raise _refusal(
            "image_to_video_unsupported",
            f"{model} on {spec.backend} does not start from an input image; send the "
            "job without an input for text-to-video",
            model, spec,
        )
    width, height = _size(params, spec)
    _refuse_size(width, height, model, spec)
    fps = params.fps if params.fps is not None else spec.default_fps
    frames = _frames(params, fps, model, spec)
    _refuse_length(frames, fps, model, spec)
    settled = Settled(
        mode=mode,
        width=width,
        height=height,
        num_frames=frames,
        fps=fps,
        steps=spec.steps,
        seed=seed,
        video_tokens=video_tokens(width, height, frames),
    )
    refuse_tokens(settled, model, spec)
    return settled


__all__ = [
    "IMAGE_TO_VIDEO",
    "MAX_SEED",
    "Settled",
    "TEXT_TO_VIDEO",
    "VideoParams",
    "refuse_tokens",
    "settle",
]
