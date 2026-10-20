"""Video model manifests: crucible/video/<id>.toml, one file per model.

A video manifest is an image manifest's shape with what a clip adds: a frame
ceiling, the frame rates the model was run at, a token budget per mode (the
number that sizes the denoising stage's memory), the fixed step count of a
distilled schedule, a declared memory figure per stage, whether the arm makes a
picture to a supplied soundtrack (`audio_to_video`), and companion files
pulled from a second repo and verified by sha256 (the same `Companion` an
audio manifest declares; the PC's arm has one, the Mac's none). The Mac's arm
adds its MLX cache limit and the steps of its full-size refining pass.

A model's `pipeline` (in [model]) says what it makes. `distilled`, the default, makes a
clip from a prompt and optionally a start picture or a soundtrack. `lip-dub` re-makes a
supplied reference video so its speaker says a new line: its length and frame rate come
from the reference, so its blocks declare no fps or default length, and its two
companions are the IC-LoRA fused into the transformer and the text encoder the pack does
not carry. docs/internals/video.md says why each number is what it is.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audiomodels import Companion, CompanionFile
from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .manifests import MODELS_PULL_COMMAND
from .tomltable import (
    HF_REPO_PATTERN,
    MODEL_ID_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    check_table,
)

VIDEO_DIR_ENV = "CRUCIBLE_VIDEO_DIR"

LTX = "ltx"

LTX_2_MLX = "ltx-2-mlx"

VIDEO_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset({LTX}),
    MLX_DARWIN: frozenset({LTX_2_MLX}),
}

ENGINE_DEVICE: dict[tuple[str, str], str] = {
    (LTX, CUDA_LINUX): "cuda",
    (LTX_2_MLX, MLX_DARWIN): "metal",
}

ENGINE_TRANSFORMER_COMPANION: dict[str, bool] = {
    LTX: True,
    LTX_2_MLX: False,
}

DISTILLED = "distilled"

LIP_DUB = "lip-dub"

PIPELINES: tuple[str, ...] = (DISTILLED, LIP_DUB)

# Which pipelines each engine's worker runs: ltx2mlx_worker.py drives the port's
# DistilledPipeline and its LipDubPipeline; ltx_worker.py (diffusers) only the distilled one.
ENGINE_PIPELINES: dict[str, frozenset[str]] = {
    LTX: frozenset({DISTILLED}),
    LTX_2_MLX: frozenset({DISTILLED, LIP_DUB}),
}

# A lip-dub block's companions, in this order: the lip-dub IC-LoRA (one .safetensors) and
# the Gemma text encoder, which the LTX-2.3 MLX pack does not carry.
IC_LORA = "ic-lora"

TEXT_ENCODER = "text-encoder"

LIP_DUB_COMPANIONS: tuple[str, ...] = (IC_LORA, TEXT_ENCODER)

# What only the distilled pipeline declares: a lip-dub clip's rate and length are the
# reference video's.
CLIP_TIMING: dict[str, Any] = {
    "fps": list,
    "default_fps": int,
    "default_duration_s": (int, float),
}

# Which engines' workers hold a supplied soundtrack fixed while they denoise the picture:
# ltx2mlx_worker.py does (the port's frozen audio state); ltx_worker.py (diffusers) does not.
ENGINE_AUDIO_TO_VIDEO: dict[str, bool] = {
    LTX: False,
    LTX_2_MLX: True,
}

MEMORY_BASES = frozenset({"measured", "declared"})

DTYPES = frozenset({"bfloat16", "float16", "float32"})

FRAME_STRIDE = 8

LATENT_SCALE = 32

TEXT_TO_VIDEO = "text-to-video"

IMAGE_TO_VIDEO = "image-to-video"

AUDIO_TO_VIDEO = "audio-to-video"

IMAGE_AUDIO_TO_VIDEO = "image-audio-to-video"

# A lip-dub model's one mode: a reference video re-made so its speaker says a new line.
LIP_DUB_MODE = "lip-dub"

MODES: tuple[str, ...] = (
    TEXT_TO_VIDEO, IMAGE_TO_VIDEO, AUDIO_TO_VIDEO, IMAGE_AUDIO_TO_VIDEO, LIP_DUB_MODE,
)

PICTURE_MODES = frozenset({IMAGE_TO_VIDEO, IMAGE_AUDIO_TO_VIDEO})

SOUNDTRACK_MODES = frozenset({AUDIO_TO_VIDEO, IMAGE_AUDIO_TO_VIDEO})

STAGES: tuple[str, ...] = (
    "encoding",
    "audio_encoding",
    "connecting",
    "conditioning",
    "denoising",
    "refining",
    "decoding",
    "audio_decoding",
)

OPTIONAL_PARAMS: tuple[str, ...] = ("negative_prompt",)

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "display": str,
    "licence": str,
    "licence_url": str,
    "commercial_use": str,
}
_MODEL_OPTIONAL: dict[str, type] = {
    "pipeline": str,
}
_BACKEND_REQUIRED: dict[str, Any] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "gated": bool,
    "dtype": str,
    "memory_bytes_estimate": int,
    "memory_basis": str,
    "memory_note": str,
    "files": list,
    "size_multiple": int,
    "min_side": int,
    "max_side": int,
    "max_pixels": int,
    "default_width": int,
    "default_height": int,
    "max_frames": int,
    "max_video_tokens": int,
    "steps": int,
    "audio_sample_rate": int,
    "stage_memory_bytes": dict,
}
_BACKEND_OPTIONAL: dict[str, Any] = {
    "max_video_tokens_image_to_video": int,
    "not_taken": dict,
    "companions": list,
    "mlx_cache_limit_bytes": int,
    "refine_steps": int,
    "tiled_max_frames": int,
    "tiled_max_video_tokens": int,
    "audio_to_video": bool,
    **CLIP_TIMING,
}
_COMPANION_REQUIRED: dict[str, Any] = {
    "name": str,
    "hf_repo": str,
    "revision": str,
    "files": list,
}
_COMPANION_FILE_REQUIRED: dict[str, Any] = {
    "source": str,
    "target": str,
    "sha256": str,
    "bytes": int,
}


class VideoManifestError(CrucibleError):
    ...


def frames_for(duration_s: float, fps: int) -> int:
    """The frame count a duration asks for, on the VAE's causal 8k+1 grid."""
    stride = max(1, round((duration_s * fps - 1) / FRAME_STRIDE))
    return stride * FRAME_STRIDE + 1


def latent_frames(num_frames: int) -> int:
    return (num_frames - 1) // FRAME_STRIDE + 1


def video_tokens(width: int, height: int, num_frames: int) -> int:
    """Latent cells the transformer denoises at full size: the VAE packs 32x32 pixels
    and 8 frames (after the first) into one, whatever grid a backend's sizes keep to."""
    return latent_frames(num_frames) * (width // LATENT_SCALE) * (height // LATENT_SCALE)


@dataclass(frozen=True)
class VideoBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    gated: bool
    dtype: str
    memory_bytes_estimate: int
    memory_basis: str
    memory_note: str
    files: tuple[str, ...]
    size_multiple: int
    min_side: int
    max_side: int
    max_pixels: int
    fps: tuple[int, ...]
    default_fps: int | None
    default_width: int
    default_height: int
    default_duration_s: float | None
    max_frames: int
    max_video_tokens: int
    steps: int
    audio_sample_rate: int
    stage_memory_bytes: tuple[tuple[str, int], ...]
    companions: tuple[Companion, ...]
    max_video_tokens_image_to_video: int | None = None
    not_taken: tuple[tuple[str, str], ...] = ()
    mlx_cache_limit_bytes: int | None = None
    refine_steps: int | None = None
    tiled_max_frames: int | None = None
    tiled_max_video_tokens: int | None = None
    audio_to_video: bool = False
    pipeline: str = DISTILLED

    @property
    def device(self) -> str:
        return ENGINE_DEVICE[(self.engine, self.backend)]

    @property
    def lip_dub(self) -> bool:
        return self.pipeline == LIP_DUB

    def companion(self, name: str) -> Companion:
        for found in self.companions:
            if found.name == name:
                return found
        raise VideoManifestError(
            f"the {self.backend} block declares no {name!r} companion; it has "
            f"{[c.name for c in self.companions]}"
        )

    @property
    def image_to_video(self) -> bool:
        return self.max_video_tokens_image_to_video is not None

    @property
    def transformer_companion(self) -> Companion | None:
        if not ENGINE_TRANSFORMER_COMPANION[self.engine]:
            return None
        return self.companions[0]

    def transformer_path(self, model_dir: Path) -> Path | None:
        companion = self.transformer_companion
        if companion is None:
            return None
        return model_dir / companion.name / companion.files[0].target

    def token_ceiling(self, mode: str) -> int:
        """A start picture adds a timestep per video token, so both picture modes have the
        image-to-video ceiling; a soundtrack alone adds per-token timesteps only to the
        audio's few tokens, so audio-to-video keeps the text-to-video one."""
        if mode in PICTURE_MODES and self.max_video_tokens_image_to_video is not None:
            return self.max_video_tokens_image_to_video
        return self.max_video_tokens

    def video_tokens(self, width: int, height: int, num_frames: int) -> int:
        return video_tokens(width, height, num_frames)

    def why_not(self, param: str) -> str | None:
        return dict(self.not_taken).get(param)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "gated": self.gated,
            "dtype": self.dtype,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory_basis": self.memory_basis,
            "memory_note": self.memory_note,
            "size_multiple": self.size_multiple,
            "min_side": self.min_side,
            "max_side": self.max_side,
            "max_pixels": self.max_pixels,
            "fps": list(self.fps),
            "default_fps": self.default_fps,
            "default_width": self.default_width,
            "default_height": self.default_height,
            "default_duration_s": self.default_duration_s,
            "max_frames": self.max_frames,
            "max_video_tokens": self.max_video_tokens,
            "max_video_tokens_image_to_video": self.max_video_tokens_image_to_video,
            "image_to_video": self.image_to_video,
            "audio_to_video": self.audio_to_video,
            "steps": self.steps,
            "audio_sample_rate": self.audio_sample_rate,
            "stage_memory_bytes": dict(self.stage_memory_bytes),
            "companions": [companion.to_dict() for companion in self.companions],
            "mlx_cache_limit_bytes": self.mlx_cache_limit_bytes,
            "refine_steps": self.refine_steps,
            "tiled_max_frames": self.tiled_max_frames,
            "tiled_max_video_tokens": self.tiled_max_video_tokens,
            "pipeline": self.pipeline,
        }


@dataclass(frozen=True)
class VideoManifest:
    weights_family = "models"

    id: str
    family: str
    display: str
    licence: str
    licence_url: str
    commercial_use: str
    backends: dict[str, VideoBackendSpec]
    path: Path
    pipeline: str = DISTILLED

    @property
    def lip_dub(self) -> bool:
        return self.pipeline == LIP_DUB

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[VideoManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> VideoBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise VideoManifestError(
                f"video model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "display": self.display,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "commercial_use": self.commercial_use,
            "pipeline": self.pipeline,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def video_manifests_dir() -> Path:
    override = os.environ.get(VIDEO_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise VideoManifestError(f"{VIDEO_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "video"
    if not path.is_dir():
        raise VideoManifestError(
            f"no video manifests at {path}; they are package data and this "
            f"install has lost them, or ${VIDEO_DIR_ENV} must point at them"
        )
    return path


def _check_document(document: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise VideoManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a video manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    for table in ("model", "backends"):
        if not isinstance(document.get(table), dict):
            raise VideoManifestError(f"{path.name}: missing the [{table}] table")
    if not document["backends"]:
        raise VideoManifestError(
            f"{path.name}: no backend blocks; a video model nothing can run is "
            "not a video model"
        )


def _parse_model(model: dict[str, Any], path: Path, expected_id: str) -> str:
    check_table(
        f"{path.name} [model]", model, _MODEL_REQUIRED, _MODEL_OPTIONAL, error=VideoManifestError
    )
    if model.get("pipeline", DISTILLED) not in PIPELINES:
        raise VideoManifestError(
            f"{path.name}: model.pipeline {model['pipeline']!r} is not one of {list(PIPELINES)}"
        )
    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise VideoManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise VideoManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    return model_id


def _check_pin(where: str, hf_repo: Any, revision: Any) -> None:
    if not HF_REPO_PATTERN.match(hf_repo):
        raise VideoManifestError(
            f"{where}: hf_repo {hf_repo!r} is not an <owner>/<name> HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(revision):
        raise VideoManifestError(
            f"{where}: revision {revision!r} must be a full 40-character commit "
            "sha, so a pull is reproducible; branch names are not pins"
        )


def _check_backend(where: str, kind: str) -> None:
    if kind not in VIDEO_BACKEND_ENGINES:
        raise VideoManifestError(
            f"{where}: {kind!r} is not a video backend; the video backends are "
            f"{sorted(VIDEO_BACKEND_ENGINES)}. Windows is not "
            "(docs/internals/video.md, \"Backends\")"
        )


def _check_engine(where: str, kind: str, block: dict[str, Any], pipeline: str) -> None:
    if block["engine"] not in VIDEO_BACKEND_ENGINES[kind]:
        raise VideoManifestError(
            f"{where}: engine {block['engine']!r} does not run on {kind}; that "
            f"backend's video engines are {sorted(VIDEO_BACKEND_ENGINES[kind])}"
        )
    if pipeline not in ENGINE_PIPELINES[block["engine"]]:
        raise VideoManifestError(
            f"{where}: engine {block['engine']!r} has no {pipeline} pipeline; it runs "
            f"{sorted(ENGINE_PIPELINES[block['engine']])}"
        )
    _check_pin(where, block["hf_repo"], block["revision"])
    if block["dtype"] not in DTYPES:
        raise VideoManifestError(
            f"{where}: dtype {block['dtype']!r} is not one of {sorted(DTYPES)}"
        )


def _check_mlx(where: str, kind: str, block: dict[str, Any]) -> None:
    stated = "mlx_cache_limit_bytes" in block
    if kind == MLX_DARWIN and not stated:
        raise VideoManifestError(
            f"{where}: mlx_cache_limit_bytes is required on {MLX_DARWIN}; without "
            "it MLX keeps every freed buffer and the process grows past the "
            "declared stage memory"
        )
    if kind != MLX_DARWIN and stated:
        raise VideoManifestError(
            f"{where}: mlx_cache_limit_bytes is an MLX setting and {kind} does not "
            "run MLX"
        )
    if stated and block["mlx_cache_limit_bytes"] <= 0:
        raise VideoManifestError(f"{where}: mlx_cache_limit_bytes must be positive")
    if "refine_steps" in block and block["refine_steps"] <= 0:
        raise VideoManifestError(f"{where}: refine_steps must be positive")


def _stage_memory(where: str, block: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    table = block["stage_memory_bytes"]
    unknown = sorted(set(table) - set(STAGES))
    if unknown or not table:
        raise VideoManifestError(
            f"{where}: stage_memory_bytes names {unknown or 'nothing'}; the "
            f"stages are {list(STAGES)}"
        )
    for stage, value in table.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise VideoManifestError(
                f"{where}: stage_memory_bytes.{stage} must be a positive integer"
            )
    largest = max(table.values())
    if block["memory_bytes_estimate"] != largest:
        raise VideoManifestError(
            f"{where}: memory_bytes_estimate is {block['memory_bytes_estimate']} but "
            f"the largest stage ({max(table, key=table.get)}) declares {largest}; "
            "one stage is on the card at a time, so the model needs its largest "
            "stage and no more"
        )
    return tuple((stage, table[stage]) for stage in STAGES if stage in table)


def _check_memory(where: str, block: dict[str, Any]) -> None:
    if block["memory_bytes_estimate"] <= 0:
        raise VideoManifestError(f"{where}: memory_bytes_estimate must be positive")
    if block["memory_basis"] not in MEMORY_BASES:
        raise VideoManifestError(
            f"{where}: memory_basis {block['memory_basis']!r} is not one of "
            f"{sorted(MEMORY_BASES)}"
        )
    if not block["memory_note"].strip():
        raise VideoManifestError(
            f"{where}: memory_note is empty; it says where the number came from "
            "and at what size of clip it holds"
        )


def _check_sizes(where: str, block: dict[str, Any]) -> None:
    multiple = block["size_multiple"]
    if multiple <= 0 or multiple % LATENT_SCALE:
        raise VideoManifestError(
            f"{where}: size_multiple {multiple} must be a positive multiple of the "
            f"VAE's {LATENT_SCALE}-pixel cell"
        )
    for key in ("min_side", "max_side", "default_width", "default_height"):
        if block[key] <= 0 or block[key] % multiple:
            raise VideoManifestError(
                f"{where}: {key} {block[key]} must be a positive multiple of "
                f"size_multiple {multiple}"
            )
    if block["min_side"] > block["max_side"]:
        raise VideoManifestError(f"{where}: min_side is above max_side")
    if block["max_pixels"] < block["min_side"] ** 2:
        raise VideoManifestError(
            f"{where}: max_pixels {block['max_pixels']} admits no clip at all"
        )
    default_pixels = block["default_width"] * block["default_height"]
    if (
        max(block["default_width"], block["default_height"]) > block["max_side"]
        or default_pixels > block["max_pixels"]
    ):
        raise VideoManifestError(
            f"{where}: the default size {block['default_width']}x"
            f"{block['default_height']} is past the block's own ceiling"
        )


def _check_timing(where: str, block: dict[str, Any], pipeline: str) -> None:
    """The distilled pipeline declares its rates and default length; a lip-dub clip takes
    both from its reference, so its block declaring either would be a number nothing reads."""
    stated = sorted(key for key in CLIP_TIMING if key in block)
    if pipeline == LIP_DUB and stated:
        raise VideoManifestError(
            f"{where}: a lip-dub block declares no {stated}; the clip's frame rate and "
            "length are the reference video's"
        )
    absent = sorted(key for key in CLIP_TIMING if key not in block)
    if pipeline == DISTILLED and absent:
        raise VideoManifestError(f"{where}: missing required key(s) {absent}")
    if pipeline == LIP_DUB:
        for key in ("max_video_tokens_image_to_video", "audio_to_video",
                    "tiled_max_frames", "tiled_max_video_tokens"):
            if key in block:
                raise VideoManifestError(
                    f"{where}: {key} is a distilled-pipeline limit; a lip-dub block "
                    "takes no start picture or soundtrack and its passes are not tiled"
                )
        if "refine_steps" not in block:
            raise VideoManifestError(
                f"{where}: a lip-dub block declares refine_steps, its full-size pass's steps"
            )


def _check_frames(where: str, block: dict[str, Any]) -> None:
    frames = block["max_frames"]
    if frames < FRAME_STRIDE + 1 or (frames - 1) % FRAME_STRIDE:
        raise VideoManifestError(
            f"{where}: max_frames {frames} is not on the VAE's frame grid "
            f"({FRAME_STRIDE}k+1, at least {FRAME_STRIDE + 1})"
        )
    if block["steps"] <= 0 or block["audio_sample_rate"] <= 0:
        raise VideoManifestError(f"{where}: steps and audio_sample_rate must be positive")


def _check_time(where: str, block: dict[str, Any]) -> tuple[int, ...]:
    if "fps" not in block:
        return ()
    rates = block["fps"]
    if not rates or not all(
        isinstance(rate, int) and not isinstance(rate, bool) and rate > 0 for rate in rates
    ):
        raise VideoManifestError(f"{where}: fps must be a non-empty list of positive integers")
    if block["default_fps"] not in rates:
        raise VideoManifestError(
            f"{where}: default_fps {block['default_fps']} is not one of fps {rates}"
        )
    frames = block["max_frames"]
    if block["default_duration_s"] <= 0:
        raise VideoManifestError(f"{where}: default_duration_s must be positive")
    if frames_for(float(block["default_duration_s"]), block["default_fps"]) > frames:
        raise VideoManifestError(
            f"{where}: default_duration_s {block['default_duration_s']} at "
            f"{block['default_fps']} fps is past max_frames {frames}"
        )
    return tuple(sorted(set(rates)))


def _check_tokens(where: str, block: dict[str, Any]) -> None:
    ceilings = [block["max_video_tokens"]]
    if "max_video_tokens_image_to_video" in block:
        ceilings.append(block["max_video_tokens_image_to_video"])
    smallest = latent_frames(FRAME_STRIDE + 1) * (block["min_side"] // LATENT_SCALE) ** 2
    for ceiling in ceilings:
        if ceiling < smallest:
            raise VideoManifestError(
                f"{where}: a token ceiling of {ceiling} admits no clip at all "
                f"(the smallest clip is {smallest} tokens)"
            )
    # A lip-dub clip is as long as its reference, so its default size must hold the
    # longest reference the block admits.
    default_frames = (
        frames_for(float(block["default_duration_s"]), block["default_fps"])
        if "default_duration_s" in block
        else block["max_frames"]
    )
    default_tokens = video_tokens(block["default_width"], block["default_height"], default_frames)
    if default_tokens > block["max_video_tokens"]:
        raise VideoManifestError(
            f"{where}: the default clip is {default_tokens} tokens, past "
            f"max_video_tokens {block['max_video_tokens']}"
        )


def _check_tiled(where: str, kind: str, block: dict[str, Any]) -> None:
    named = [key for key in ("tiled_max_frames", "tiled_max_video_tokens") if key in block]
    if not named:
        return
    if len(named) != 2:
        raise VideoManifestError(
            f"{where}: tiled_max_frames and tiled_max_video_tokens come as a pair; "
            f"only {named[0]} is declared"
        )
    if block["engine"] != "ltx-2-mlx":
        raise VideoManifestError(
            f"{where}: tiled limits apply only where the full-size pass is tiled "
            "(the ltx-2-mlx engine)"
        )
    if block["tiled_max_frames"] < block["max_frames"] or (
        block["tiled_max_video_tokens"] < block["max_video_tokens"]
    ):
        raise VideoManifestError(
            f"{where}: tiled limits must be at least the untiled ones"
        )
    if (block["tiled_max_frames"] - 1) % FRAME_STRIDE:
        raise VideoManifestError(
            f"{where}: tiled_max_frames must be {FRAME_STRIDE}k+1"
        )


def _check_audio_to_video(where: str, block: dict[str, Any]) -> None:
    if block.get("audio_to_video") is True and not ENGINE_AUDIO_TO_VIDEO[block["engine"]]:
        raise VideoManifestError(
            f"{where}: audio_to_video is declared but engine {block['engine']!r} has no "
            "audio-to-video path (its worker would generate the sound instead of holding "
            "the supplied one); the engines that have one are "
            f"{sorted(engine for engine, has in ENGINE_AUDIO_TO_VIDEO.items() if has)}"
        )


def _strings(where: str, key: str, values: list[Any]) -> tuple[str, ...]:
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise VideoManifestError(f"{where}: {key} must be a non-empty list of strings")
    return tuple(values)


def _not_taken(where: str, block: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    table = block.get("not_taken", {})
    for param, why in table.items():
        if param not in OPTIONAL_PARAMS or not isinstance(why, str) or not why.strip():
            raise VideoManifestError(
                f"{where}: not_taken.{param} must name one of {list(OPTIONAL_PARAMS)} "
                "with the reason as a string"
            )
    return tuple(sorted(table.items()))


def _companion_file(where: str, entry: Any) -> CompanionFile:
    if not isinstance(entry, dict):
        raise VideoManifestError(f"{where}: each companion file is a table")
    check_table(where, entry, _COMPANION_FILE_REQUIRED, error=VideoManifestError)
    if not SHA256_PATTERN.match(entry["sha256"]):
        raise VideoManifestError(f"{where}: sha256 must be 64 lower-case hex digits")
    return CompanionFile(**entry)


def _companion(where: str, entry: Any) -> Companion:
    if not isinstance(entry, dict):
        raise VideoManifestError(f"{where}: each companion is a table")
    check_table(where, entry, _COMPANION_REQUIRED, error=VideoManifestError)
    _check_pin(where, entry["hf_repo"], entry["revision"])
    if not MODEL_ID_PATTERN.match(entry["name"]):
        raise VideoManifestError(f"{where}: companion name {entry['name']!r} is not a plain name")
    files = tuple(
        _companion_file(f"{where} files[{index}]", item)
        for index, item in enumerate(entry["files"])
    )
    if not files:
        raise VideoManifestError(f"{where}: a companion with no files pulls nothing")
    return Companion(entry["name"], entry["hf_repo"], entry["revision"], files)


def _lip_dub_companions(where: str, companions: tuple[Companion, ...]) -> tuple[Companion, ...]:
    names = tuple(companion.name for companion in companions)
    if names != LIP_DUB_COMPANIONS:
        raise VideoManifestError(
            f"{where}: a lip-dub block declares exactly the companions "
            f"{list(LIP_DUB_COMPANIONS)}, in that order (the IC-LoRA, then the text "
            f"encoder); it declares {list(names)}"
        )
    lora = companions[0].files
    if len(lora) != 1 or not lora[0].target.endswith(".safetensors"):
        raise VideoManifestError(
            f"{where}: the {IC_LORA} companion is one .safetensors file, the LoRA the "
            "worker fuses into the transformer"
        )
    return companions


def _companions(where: str, block: dict[str, Any], pipeline: str) -> tuple[Companion, ...]:
    companions = tuple(
        _companion(f"{where} companions[{index}]", entry)
        for index, entry in enumerate(block.get("companions", []))
    )
    if pipeline == LIP_DUB:
        return _lip_dub_companions(where, companions)
    if not ENGINE_TRANSFORMER_COMPANION[block["engine"]]:
        if companions:
            raise VideoManifestError(
                f"{where}: engine {block['engine']!r} reads its transformer from "
                f"{block['hf_repo']} itself; a companion would be pulled and never read"
            )
        return ()
    if len(companions) != 1 or len(companions[0].files) != 1:
        raise VideoManifestError(
            f"{where}: a video block declares exactly one companion with exactly "
            "one file, the quantized transformer the denoising stage loads"
        )
    if not companions[0].files[0].target.endswith(".gguf"):
        raise VideoManifestError(
            f"{where}: the transformer companion's file must be a .gguf; that is "
            "the one format the worker streams onto the card"
        )
    return companions


def _parse_backend(path: Path, kind: str, block: Any, pipeline: str) -> VideoBackendSpec:
    where = f"{path.name} [backends.{kind}]"
    _check_backend(where, kind)
    if not isinstance(block, dict):
        raise VideoManifestError(f"{where}: must be a table")
    check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=VideoManifestError)
    _check_engine(where, kind, block, pipeline)
    _check_timing(where, block, pipeline)
    _check_mlx(where, kind, block)
    _check_memory(where, block)
    stages = _stage_memory(where, block)
    _check_sizes(where, block)
    _check_frames(where, block)
    rates = _check_time(where, block)
    _check_tokens(where, block)
    _check_tiled(where, kind, block)
    _check_audio_to_video(where, block)
    return VideoBackendSpec(
        backend=kind,
        engine=block["engine"],
        hf_repo=block["hf_repo"],
        revision=block["revision"],
        gated=block["gated"],
        dtype=block["dtype"],
        memory_bytes_estimate=block["memory_bytes_estimate"],
        memory_basis=block["memory_basis"],
        memory_note=block["memory_note"],
        files=_strings(where, "files", block["files"]),
        size_multiple=block["size_multiple"],
        min_side=block["min_side"],
        max_side=block["max_side"],
        max_pixels=block["max_pixels"],
        fps=rates,
        default_fps=block.get("default_fps"),
        default_width=block["default_width"],
        default_height=block["default_height"],
        default_duration_s=(
            float(block["default_duration_s"]) if "default_duration_s" in block else None
        ),
        max_frames=block["max_frames"],
        max_video_tokens=block["max_video_tokens"],
        steps=block["steps"],
        audio_sample_rate=block["audio_sample_rate"],
        stage_memory_bytes=stages,
        companions=_companions(where, block, pipeline),
        max_video_tokens_image_to_video=block.get("max_video_tokens_image_to_video"),
        not_taken=_not_taken(where, block),
        mlx_cache_limit_bytes=block.get("mlx_cache_limit_bytes"),
        refine_steps=block.get("refine_steps"),
        tiled_max_frames=block.get("tiled_max_frames"),
        tiled_max_video_tokens=block.get("tiled_max_video_tokens"),
        audio_to_video=block.get("audio_to_video", False),
        pipeline=pipeline,
    )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> VideoManifest:
    _check_document(document, path)
    model = document["model"]
    model_id = _parse_model(model, path, expected_id)
    pipeline = model.get("pipeline", DISTILLED)
    backends = {
        kind: _parse_backend(path, kind, block, pipeline)
        for kind, block in document["backends"].items()
    }
    return VideoManifest(
        id=model_id,
        family=model["family"],
        display=model["display"],
        licence=model["licence"],
        licence_url=model["licence_url"],
        commercial_use=model["commercial_use"],
        backends=backends,
        path=path,
        pipeline=pipeline,
    )


def parse_video_manifest(text: str, path: Path, expected_id: str) -> VideoManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VideoManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_video_manifest(model_id: str, directory: Path | None = None) -> VideoManifest:
    root = directory if directory is not None else video_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise VideoManifestError(
            f"no video manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VideoManifestError(f"could not read {path}: {exc}") from exc
    return parse_video_manifest(text, path, model_id)


def load_all_video_manifests(directory: Path | None = None) -> dict[str, VideoManifest]:
    root = directory if directory is not None else video_manifests_dir()
    return {
        path.stem: load_video_manifest(path.stem, root)
        for path in sorted(root.glob("*.toml"), key=lambda p: p.stem)
    }


def engines_on(backend_kind: str) -> tuple[str, ...]:
    return tuple(sorted(VIDEO_BACKEND_ENGINES.get(backend_kind, ())))


def load_clip_manifests(directory: Path | None = None) -> dict[str, VideoManifest]:
    """The models that make a clip from a prompt (the `video` verb's candidates); a lip-dub
    model needs a reference video, which a verb's caller has not sent."""
    return {
        model_id: manifest
        for model_id, manifest in load_all_video_manifests(directory).items()
        if not manifest.lip_dub
    }


__all__ = [
    "AUDIO_TO_VIDEO",
    "DISTILLED",
    "FRAME_STRIDE",
    "IC_LORA",
    "IMAGE_AUDIO_TO_VIDEO",
    "IMAGE_TO_VIDEO",
    "LATENT_SCALE",
    "LIP_DUB",
    "LIP_DUB_COMPANIONS",
    "LIP_DUB_MODE",
    "LTX",
    "LTX_2_MLX",
    "MODES",
    "PICTURE_MODES",
    "SOUNDTRACK_MODES",
    "STAGES",
    "TEXT_ENCODER",
    "TEXT_TO_VIDEO",
    "VIDEO_BACKEND_ENGINES",
    "VIDEO_DIR_ENV",
    "VideoBackendSpec",
    "VideoManifest",
    "VideoManifestError",
    "engines_on",
    "frames_for",
    "latent_frames",
    "load_all_video_manifests",
    "load_clip_manifests",
    "load_video_manifest",
    "parse_video_manifest",
    "video_manifests_dir",
    "video_tokens",
]
