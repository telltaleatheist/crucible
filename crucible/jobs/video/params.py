"""A video job's params, and the arm's refusals by name.

Every refusal here happens before a worker starts: at submit for what the
params alone decide, and in `run` (before the session) for what needs the
inputs: the image-to-video token ceiling, whether the arm takes a soundtrack,
and a clip's length against the soundtrack it is made to. A lip-dub model's clip is
its reference video's length and rate, so it is settled from the reference's
measured facts (`settle_lip_dub`); at submit only its params are checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...errors import ApiError
from ...videomodels import (
    AUDIO_TO_VIDEO,
    FRAME_STRIDE,
    IMAGE_AUDIO_TO_VIDEO,
    IMAGE_TO_VIDEO,
    LATENT_SCALE,
    LIP_DUB_MODE,
    PICTURE_MODES,
    SOUNDTRACK_MODES,
    TEXT_TO_VIDEO,
    VideoBackendSpec,
    frames_for,
    video_tokens,
)

MAX_SEED = 2**32 - 1

# The IC-LoRA reference's conditioning strength when a lip-dub job sends none: the value
# Lightricks' recipe and the port's CLI run at (all of the reference kept outside the lips).
LIP_DUB_REFERENCE_STRENGTH = 1.0

# How far a default lip-dub frame's shape may be from its reference's before a smaller
# frame of a nearer shape is preferred.
REFERENCE_SHAPE_TOLERANCE = 0.05


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
        "is the model's default (5), or with a soundtrack input the soundtrack's own "
        "length rounded down onto the grid. Not with `num_frames`.",
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
        "picture but not decoded. Refused with a soundtrack input, which is the "
        "clip's sound, and by a lip-dub model, whose sound is the new line.",
    )
    reference_strength: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Lip-dub models only (ltx-2.3-dubit): how strongly the reference "
        "video holds the picture outside the mouth, above 0 and at most 1; null is 1.0, "
        "Lightricks' recipe. Refused by every other model.",
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
    """`fps` is a whole number for a clip the model times itself; a lip-dub clip has its
    reference's rate, which can be fractional (29.97 is 30000/1001), so it also carries
    that rate exactly as `frame_rate` for the mux."""

    mode: str
    width: int
    height: int
    num_frames: int
    fps: int | float
    steps: int
    seed: int
    video_tokens: int
    soundtrack_s: float | None = None
    frame_rate: str | None = None
    reference_strength: float | None = None

    @property
    def duration_s(self) -> float:
        return round(self.num_frames / self.fps, 3)


def _refusal(code: str, message: str, model: str, spec: VideoBackendSpec, **extra: Any) -> ApiError:
    return ApiError(400, code, message, {"model": model, "backend": spec.backend, **extra})


def _refuse_untaken(params: VideoParams, model: str, spec: VideoBackendSpec) -> None:
    if params.reference_strength is not None and not spec.lip_dub:
        raise _refusal(
            "video_param_unsupported",
            f"{model} does not take 'reference_strength': it is the strength of a lip-dub "
            "model's reference video, and this model makes a clip from a prompt. Leave it "
            "out, or send the job to a lip-dub model (ltx-2.3-dubit, on a Mac)",
            model, spec, param="reference_strength",
        )
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


def _frames(
    params: VideoParams, fps: int, model: str, spec: VideoBackendSpec, soundtrack_s: float | None
) -> int:
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
    if params.duration_s is None and soundtrack_s is not None:
        return longest_covered(soundtrack_s, fps)
    duration = params.duration_s if params.duration_s is not None else spec.default_duration_s
    return frames_for(duration, fps)


def longest_covered(soundtrack_s: float, fps: int) -> int:
    """The longest clip on the 8k+1 grid that a soundtrack of soundtrack_s seconds lasts
    through, rounded down so the clip never runs past its sound. Below one grid step (the
    shortest clip is 9 frames) it is 1, which no clip is."""
    return max(0, int((soundtrack_s * fps - 1) // FRAME_STRIDE)) * FRAME_STRIDE + 1


def _refuse_length(
    frames: int, fps: int, model: str, spec: VideoBackendSpec, from_soundtrack: float | None
) -> None:
    if frames > spec.max_frames:
        where = (
            f". The length came from the soundtrack ({from_soundtrack:.2f} s); send "
            "duration_s or num_frames for a clip of its first part"
            if from_soundtrack is not None
            else ""
        )
        raise _refusal(
            "video_too_long",
            f"{frames} frames is {frames / fps:.2f} s at {fps} fps; {model} on "
            f"{spec.backend} makes at most {spec.max_frames} frames "
            f"({spec.max_frames / fps:.2f} s at {fps} fps) because its memory "
            f"({spec.memory_basis}) was sized at that length. Ask for a shorter clip{where}",
            model, spec, num_frames=frames, max_frames=spec.max_frames,
        )


def refuse_mode(params: VideoParams, mode: str, model: str, spec: VideoBackendSpec) -> None:
    if mode in PICTURE_MODES and not spec.image_to_video:
        raise _refusal(
            "image_to_video_unsupported",
            f"{model} on {spec.backend} does not start from an input image; send the "
            "job without an input for text-to-video",
            model, spec,
        )
    if mode not in SOUNDTRACK_MODES:
        return
    if not spec.audio_to_video:
        raise _refusal(
            "audio_to_video_unsupported",
            f"{model} on {spec.backend} does not make a picture to a supplied soundtrack: "
            f"its engine ({spec.engine}) generates the sound with the picture and has no "
            "way to hold one fixed. Send the job without the audio input, or to a Mac "
            "(mlx-darwin), whose arm does",
            model, spec,
        )
    if not params.audio:
        raise _refusal(
            "video_param_unsupported",
            "audio false asks for a silent clip, but this job carries a soundtrack, which "
            "is the clip's sound. Leave audio out to keep the soundtrack, or send the job "
            "without the audio input for a silent clip",
            model, spec, param="audio",
        )


def _refuse_soundtrack(
    frames: int, fps: int, soundtrack_s: float, model: str, spec: VideoBackendSpec
) -> None:
    shortest = FRAME_STRIDE + 1
    if frames >= shortest and frames / fps <= soundtrack_s:
        return
    longest = longest_covered(soundtrack_s, fps)
    never = "a supplied soundtrack is the clip's whole sound and is never padded or looped"
    if longest < shortest:
        message = (
            f"the soundtrack is {soundtrack_s:.2f} s, shorter than the shortest clip, "
            f"{shortest} frames ({shortest / fps:.2f} s at {fps} fps); {never}. Send a "
            "longer soundtrack"
        )
    else:
        message = (
            f"the soundtrack is {soundtrack_s:.2f} s, and {frames} frames at {fps} fps is "
            f"{frames / fps:.2f} s; {never}, so the longest clip it covers is {longest} "
            f"frames ({longest / fps:.2f} s at {fps} fps). Send num_frames {longest} or "
            "less, or leave the length out and the soundtrack sets it"
        )
    raise _refusal(
        "video_soundtrack_too_short", message,
        model, spec, soundtrack_s=soundtrack_s, num_frames=frames, longest_frames=longest,
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
        " (a start picture adds a timestep per token, so it holds more on the "
        "card per frame than text-to-video)"
        if settled.mode in PICTURE_MODES
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
    params: VideoParams,
    model: str,
    spec: VideoBackendSpec,
    seed: int,
    mode: str = TEXT_TO_VIDEO,
    soundtrack_s: float | None = None,
) -> Settled:
    """The clip a job makes. `soundtrack_s` is the supplied soundtrack's measured length,
    given exactly when `mode` is one of the soundtrack modes."""
    if (mode in SOUNDTRACK_MODES) != (soundtrack_s is not None):
        raise ValueError(
            f"settle() was given mode {mode!r} with soundtrack_s {soundtrack_s!r}; the "
            "soundtrack modes carry its length and the others none"
        )
    _refuse_untaken(params, model, spec)
    refuse_mode(params, mode, model, spec)
    width, height = _size(params, spec)
    _refuse_size(width, height, model, spec)
    fps = params.fps if params.fps is not None else spec.default_fps
    frames = _frames(params, fps, model, spec, soundtrack_s)
    derived = params.num_frames is None and params.duration_s is None
    _refuse_length(frames, fps, model, spec, soundtrack_s if derived else None)
    if soundtrack_s is not None:
        _refuse_soundtrack(frames, fps, soundtrack_s, model, spec)
    settled = Settled(
        mode=mode,
        width=width,
        height=height,
        num_frames=frames,
        fps=fps,
        steps=spec.steps,
        seed=seed,
        video_tokens=video_tokens(width, height, frames),
        soundtrack_s=soundtrack_s,
    )
    refuse_tokens(settled, model, spec)
    return settled


def snap_frames(frames: int) -> int:
    """A reference's frame count rounded down onto the 8k+1 grid: the frames a lip-dub clip
    keeps (ltx_pipelines_mlx/lipdub.py, _snap_frames_to_8k1). Below 9 there is no clip."""
    return max(0, (frames - 1) // FRAME_STRIDE) * FRAME_STRIDE + 1


def refuse_lip_dub_params(params: VideoParams, model: str, spec: VideoBackendSpec) -> None:
    """What a lip-dub job's params alone decide, at submit: the length, rate and sound are
    the reference's, so asking for them is refused rather than ignored."""
    _refuse_untaken(params, model, spec)
    reference = "the reference video"
    asked = {
        "duration_s": (params.duration_s, f"the clip is as long as {reference}"),
        "num_frames": (params.num_frames, f"the clip is as long as {reference}"),
        "fps": (params.fps, f"the clip keeps {reference}'s frame rate"),
    }
    for name, (value, why) in asked.items():
        if value is not None:
            raise _refusal(
                "video_param_unsupported",
                f"{model} does not take '{name}': {why} (its frame count rounded down "
                f"onto the {FRAME_STRIDE}k+1 grid). Trim the reference to the part you "
                f"want and leave {name} out",
                model, spec, param=name,
            )
    if not params.audio:
        raise _refusal(
            "video_param_unsupported",
            f"audio false asks for a silent clip, but {model}'s clip is the new line spoken; "
            "its sound is the point of it. Leave audio out",
            model, spec, param="audio",
        )
    if params.width is not None and params.height is not None:
        _refuse_size(params.width, params.height, model, spec)


def reference_size(width: int, height: int, spec: VideoBackendSpec) -> tuple[int, int]:
    """The default frame for a reference of width x height: on the block's grid and within
    its pixel and side ceilings, the largest frame whose shape is within 5% of the
    reference's, or failing any, the nearest shape (16:9 is 768x448 at the measured
    ceiling, 9:16 is 448x768, 4:3 is 576x448)."""
    aspect = width / height
    sides = range(spec.min_side, spec.max_side + 1, spec.size_multiple)
    frames = [
        (across, down, abs((across / down) / aspect - 1.0))
        for across in sides
        for down in sides
        if across * down <= spec.max_pixels
    ]
    close = [frame for frame in frames if frame[2] <= REFERENCE_SHAPE_TOLERANCE]
    if close:
        across, down, _ = max(close, key=lambda f: (f[0] * f[1], -f[2]))
    else:
        across, down, _ = min(frames, key=lambda f: (f[2], -f[0] * f[1]))
    return across, down


def settle_lip_dub(
    params: VideoParams,
    model: str,
    spec: VideoBackendSpec,
    seed: int,
    *,
    frames: int,
    frame_rate: str,
    width: int,
    height: int,
) -> Settled:
    """The clip a lip-dub job makes from a reference of `frames` frames at `frame_rate`
    (ffprobe's exact rate, 30000/1001 for 29.97) and width x height. The clip keeps every
    frame of the reference it can (rounded down onto the grid), and a reference longer than
    the block admits is refused, never cut."""
    refuse_lip_dub_params(params, model, spec)
    fps = float(Fraction(frame_rate))
    if params.width is not None and params.height is not None:
        size = (params.width, params.height)
    else:
        size = reference_size(width, height, spec)
    _refuse_size(*size, model, spec)
    kept = snap_frames(frames)
    if kept < FRAME_STRIDE + 1:
        raise _refusal(
            "video_reference_too_short",
            f"the reference video is {frames} frames ({frames / fps:.2f} s at {fps:.3g} fps); "
            f"a clip is at least {FRAME_STRIDE + 1} frames. Send a longer reference",
            model, spec, reference_frames=frames,
        )
    if kept > spec.max_frames:
        longest = spec.max_frames / fps
        raise _refusal(
            "video_reference_too_long",
            f"the reference video is {frames} frames ({frames / fps:.2f} s at {fps:.3g} fps), "
            f"so the clip would be {kept} frames; {model} on {spec.backend} makes at most "
            f"{spec.max_frames} frames ({longest:.2f} s at this reference's {fps:.3g} fps) "
            f"because its memory ({spec.memory_basis}) was sized at that length. The clip "
            f"is never cut short: trim the reference to {longest:.2f} s or less and send "
            "the part you want dubbed",
            model, spec, reference_frames=frames, num_frames=kept,
            max_frames=spec.max_frames, max_seconds=round(longest, 3),
        )
    settled = Settled(
        mode=LIP_DUB_MODE,
        width=size[0],
        height=size[1],
        num_frames=kept,
        fps=fps,
        steps=spec.steps,
        seed=seed,
        video_tokens=video_tokens(size[0], size[1], kept),
        frame_rate=frame_rate,
        reference_strength=(
            params.reference_strength
            if params.reference_strength is not None
            else LIP_DUB_REFERENCE_STRENGTH
        ),
    )
    refuse_tokens(settled, model, spec)
    return settled


__all__ = [
    "AUDIO_TO_VIDEO",
    "IMAGE_AUDIO_TO_VIDEO",
    "IMAGE_TO_VIDEO",
    "LIP_DUB_REFERENCE_STRENGTH",
    "MAX_SEED",
    "Settled",
    "TEXT_TO_VIDEO",
    "VideoParams",
    "longest_covered",
    "reference_size",
    "refuse_lip_dub_params",
    "refuse_mode",
    "refuse_tokens",
    "settle",
    "settle_lip_dub",
    "snap_frames",
]
