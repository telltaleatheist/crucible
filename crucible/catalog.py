from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import audioweights, denoisemodels, llamacpp, rvcbase, videoweights, weights
from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .audiomodels import load_all_audio_manifests
from .backend import Backend
from .cardkinds import (
    KIND_ALIGN,
    KIND_AUDIO,
    KIND_DENOISE,
    KIND_IMAGE,
    KIND_LLM,
    KIND_SEGMENT,
    KIND_TTS,
    KIND_VIDEO,
)
from .clock import utcnow
from .config import Config
from .errors import ApiError, CrucibleError
from .imagemodels import load_all_image_manifests
from .manifests import BACKEND_ENGINES, ModelManifest, load_all_manifests
from .residency import Residency
from .rvcmodels import load_all_rvc_manifests
from .segmentmodels import load_all_segment_manifests
from .videomodels import load_all_video_manifests
from .voicecatalog import (
    declared_voice_backends,
    declared_voice_ids,
    load_all_voices,
    moves_to,
    pull_target,
    refresh_unresolved,
)

KINDS: tuple[str, ...] = ("model", "voice", "rvc", "rvc-base", "denoise", "engine")

RVC_BASE_ID = "base"

_RESIDENT_KIND_FOR_JOB_TYPE: dict[str, str] = {
    "llm": KIND_LLM,
    "align": KIND_ALIGN,
    "tts": KIND_TTS,
    "denoise": KIND_DENOISE,
    "image": KIND_IMAGE,
    "audio": KIND_AUDIO,
    "segment": KIND_SEGMENT,
    "video": KIND_VIDEO,
}


@dataclass(frozen=True)
class Subject:

    kind: str
    id: str
    name: str | None
    job_type: str
    expected_bytes: int | None
    source: str
    pull_command: str
    installed: Callable[[], weights.InstalledWeights | None]
    pull: Callable[..., weights.InstalledWeights]
    remove: Callable[[], Path]
    shares_weights_of: str | None = None
    missing_files: Callable[[], list[str]] | None = None
    moves_to: Callable[[], str | None] | None = None

    def would_move(self) -> str | None:
        return None if self.moves_to is None else self.moves_to()


def subjects(config: Config, backend: Backend) -> list[Subject]:
    return [
        *_model_subjects(config, backend),
        *_audio_subjects(config, backend),
        *_video_subjects(config, backend),
        *_voice_subjects(config, backend),
        *_rvc_subjects(config, backend),
        _rvc_base_subject(config),
        *_engine_subjects(config, backend),
        *_denoise_subjects(config, backend),
    ]


def _model_subjects(config: Config, backend: Backend) -> list[Subject]:
    found: list[Subject] = []
    for job_type, loaded in (
        ("llm", load_all_manifests()),
        ("asr", load_all_asr_manifests()),
        ("align", load_all_align_manifests()),
        ("image", load_all_image_manifests()),
        ("segment", load_all_segment_manifests()),
    ):
        for manifest in loaded.values():
            if manifest.supports(backend.kind):
                found.append(_model_subject(config, backend, job_type, manifest))
    return found


def _audio_subjects(config: Config, backend: Backend) -> list[Subject]:
    return [
        _audio_subject(config, manifest, manifest.spec(backend.kind))
        for manifest in load_all_audio_manifests().values()
        if manifest.supports(backend.kind)
    ]


def _audio_subject(config: Config, manifest: Any, spec: Any) -> Subject:
    return Subject(
        kind="model",
        id=manifest.id,
        name=manifest.display,
        job_type="audio",
        expected_bytes=None,
        source=f"hf:{spec.hf_repo}",
        pull_command=manifest.pull_command,
        installed=lambda: audioweights.installed(config, manifest, spec),
        pull=lambda **kwargs: audioweights.pull(config, manifest, spec, **kwargs),
        remove=lambda: audioweights.remove(config, manifest, spec),
    )


def _video_subjects(config: Config, backend: Backend) -> list[Subject]:
    return [
        _video_subject(config, manifest, manifest.spec(backend.kind))
        for manifest in load_all_video_manifests().values()
        if manifest.supports(backend.kind)
    ]


def _video_subject(config: Config, manifest: Any, spec: Any) -> Subject:
    return Subject(
        kind="model",
        id=manifest.id,
        name=manifest.display,
        job_type="video",
        expected_bytes=None,
        source=f"hf:{spec.hf_repo}",
        pull_command=manifest.pull_command,
        installed=lambda: videoweights.installed(config, manifest, spec),
        pull=lambda **kwargs: videoweights.pull(config, manifest, spec, **kwargs),
        remove=lambda: videoweights.remove(config, manifest, spec),
    )


def _model_subject(
    config: Config, backend: Backend, job_type: str, manifest: Any
) -> Subject:
    spec = manifest.spec(backend.kind)
    base_id = getattr(manifest, "weights_of", None)
    return Subject(
        kind="model",
        id=manifest.id,
        name=getattr(manifest, "display", None),
        job_type=job_type,
        expected_bytes=(
            None
            if base_id is None or manifest.extra_files(backend.kind)
            else 0
        ),
        source=f"hf:{spec.hf_repo}",
        pull_command=manifest.pull_command,
        installed=_installed_weights(config, manifest, spec),
        pull=_pull_weights(config, manifest, spec),
        remove=_remove_weights(config, manifest, spec),
        shares_weights_of=base_id,
        missing_files=(
            None
            if base_id is None
            else _missing_extras(config, manifest, backend.kind)
        ),
    )


def _voice_subjects(config: Config, backend: Backend) -> list[Subject]:
    found: list[Subject] = []
    for voice in load_all_voices().values():
        if not voice.supports(backend.kind):
            continue
        spec = voice.spec(backend.kind)
        if spec.source == weights.LOCAL:
            continue
        found.append(
            Subject(
                kind="voice",
                id=voice.id,
                name=voice.display,
                job_type="tts",
                expected_bytes=None,
                source=f"hf:{spec.hf_repo}",
                pull_command=voice.pull_command,
                installed=_installed_weights(config, voice, spec),
                pull=_pull_voice(config, voice, backend.kind),
                remove=_remove_weights(config, voice, spec),
                moves_to=_voice_moves_to(config, voice),
            )
        )
    return found


def _voice_moves_to(config: Config, voice: Any) -> Callable[[], str | None]:
    return lambda: moves_to(config.home, voice)


def _pull_voice(
    config: Config, voice: Any, backend_kind: str
) -> Callable[..., weights.InstalledWeights]:
    def pull(**kwargs: Any) -> weights.InstalledWeights:
        target = pull_target(config.home, voice)
        if not target.supports(backend_kind):
            raise weights.WeightsError(
                f"voice {voice.id!r} moved to a revision whose crucible-voice.toml "
                f"has no {backend_kind} arm (it declares {sorted(target.backends)}); "
                "this machine stays on the revision it has. Publish a revision with "
                "that arm and run `crucible voices check-updates`"
            )
        return weights.pull(config, target, target.spec(backend_kind), **kwargs)

    return pull


def _rvc_subjects(config: Config, backend: Backend) -> list[Subject]:
    found: list[Subject] = []
    for model in load_all_rvc_manifests().values():
        if not model.supports(backend.kind):
            continue
        spec = model.spec(backend.kind)
        found.append(
            Subject(
                kind="rvc",
                id=model.id,
                name=model.display,
                job_type="rvc",
                expected_bytes=spec.archive_bytes,
                source=f"hf:{spec.hf_repo}",
                pull_command=model.pull_command,
                installed=_installed_weights(config, model, spec),
                pull=_pull_archive(config, model, spec),
                remove=_remove_weights(config, model, spec),
            )
        )
    return found


def _rvc_base_subject(config: Config) -> Subject:
    assets = rvcbase.load_rvc_base()
    return Subject(
        kind="rvc-base",
        id=RVC_BASE_ID,
        name=f"{assets.id}'s base assets",
        job_type="rvc",
        expected_bytes=assets.total_bytes,
        source=f"hf:{assets.hf_repo}",
        pull_command=rvcbase.PULL_COMMAND,
        installed=lambda: rvcbase.installed(config, assets),
        pull=lambda **kwargs: rvcbase.pull(config, assets, **kwargs),
        remove=lambda: weights.remove_files(
            rvcbase.base_root(config), assets.targets
        ),
    )


def _engine_subjects(config: Config, backend: Backend) -> list[Subject]:
    if backend.kind != llamacpp_backend():
        return []
    build = llamacpp.build_for(backend.gpu.vendor)
    return [
        Subject(
            kind=llamacpp.ENGINE_KIND,
            id=llamacpp.LLAMA_CPP_ID,
            name=f"llama.cpp {llamacpp.LLAMA_CPP_RELEASE} ({build})",
            job_type="llm",
            expected_bytes=llamacpp.expected_bytes(build),
            source=f"github:ggml-org/llama.cpp@{llamacpp.LLAMA_CPP_RELEASE}",
            pull_command=llamacpp.INSTALL_COMMAND,
            installed=_installed_engine(config, build),
            pull=_pull_engine(config, build),
            remove=lambda: llamacpp.remove(config),
        )
    ]


def _denoise_subjects(config: Config, backend: Backend) -> list[Subject]:
    found: list[Subject] = []
    for separator in denoisemodels.load_all_denoise_manifests().values():
        if not separator.supports(backend.kind):
            continue
        spec = separator.spec(backend.kind)
        found.append(
            Subject(
                kind="denoise",
                id=separator.id,
                name=separator.display,
                job_type="denoise",
                expected_bytes=spec.total_bytes,
                source=f"hf:{spec.hf_repo}",
                pull_command=separator.pull_command,
                installed=_installed_denoise(config, separator, spec),
                pull=_pull_denoise(config, separator, spec),
                remove=_remove_denoise(config, separator),
            )
        )
    return found


def _installed_weights(
    config: Config, manifest: Any, spec: Any
) -> Callable[[], weights.InstalledWeights | None]:
    return lambda: weights.installed(config, manifest, spec)


def _pull_weights(
    config: Config, manifest: Any, spec: Any
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: weights.pull(config, manifest, spec, **kwargs)


def _missing_extras(
    config: Config, manifest: Any, backend_kind: str
) -> Callable[[], list[str]]:
    def missing() -> list[str]:
        directory = weights.subject_dir(config, manifest, backend_kind)
        return [
            name
            for name in manifest.extra_files(backend_kind)
            if not (directory / name).is_file()
        ]

    return missing


def _pull_archive(
    config: Config, manifest: Any, spec: Any
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: weights.pull_archive(config, manifest, spec, **kwargs)


def llamacpp_backend() -> str:
    from .backend import LLAMA_WINDOWS

    return LLAMA_WINDOWS


def _installed_engine(
    config: Config, build: str
) -> Callable[[], weights.InstalledWeights | None]:
    return lambda: llamacpp.installed(config, build)


def _pull_engine(
    config: Config, build: str
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: llamacpp.pull(config, build, **kwargs)


def _remove_weights(
    config: Config, manifest: Any, spec: Any
) -> Callable[[], Path]:
    return lambda: weights.remove(config, manifest, spec)


def _remove_denoise(config: Config, manifest: Any) -> Callable[[], Path]:
    return lambda: weights.remove_files(
        denoisemodels.denoise_models_root(config.home),
        (manifest.model_filename, manifest.config_filename),
        stamp_name=denoisemodels.stamp_name(manifest),
    )


def _installed_denoise(
    config: Config, manifest: Any, spec: Any
) -> Callable[[], weights.InstalledWeights | None]:
    return lambda: denoisemodels.installed(config.home, manifest, spec)


def _pull_denoise(
    config: Config, manifest: Any, spec: Any
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: denoisemodels.pull(config, manifest, spec, **kwargs)


def ids_reading(subject: Subject) -> frozenset[str]:
    if subject.kind != "model":
        return frozenset({subject.id})
    return frozenset(
        {subject.id}
        | {
            manifest.id
            for manifest in load_all_manifests().values()
            if manifest.weights_of == subject.id
        }
    )


def declared_ids() -> dict[str, list[str]]:
    return {
        "model": sorted(
            {
                *load_all_manifests(),
                *load_all_asr_manifests(),
                *load_all_align_manifests(),
                *load_all_image_manifests(),
                *load_all_audio_manifests(),
                *load_all_segment_manifests(),
                *load_all_video_manifests(),
            }
        ),
        "voice": sorted(declared_voice_ids()),
        "rvc": sorted(load_all_rvc_manifests()),
        "rvc-base": [RVC_BASE_ID],
        "denoise": sorted(denoisemodels.load_all_denoise_manifests()),
        llamacpp.ENGINE_KIND: [llamacpp.LLAMA_CPP_ID],
    }


def _backends_in(
    subject_id: str, *loads: Callable[[], dict[str, Any]]
) -> list[str]:
    found: set[str] = set()
    for load in loads:
        manifest = load().get(subject_id)
        if manifest is not None:
            found.update(manifest.backends)
    return sorted(found)


def backends_declaring(kind: str, subject_id: str) -> list[str]:
    if kind == "model":
        return _backends_in(
            subject_id,
            load_all_manifests,
            load_all_asr_manifests,
            load_all_align_manifests,
            load_all_image_manifests,
            load_all_audio_manifests,
            load_all_segment_manifests,
            load_all_video_manifests,
        )
    if kind == "voice":
        return declared_voice_backends(subject_id)
    if kind == "rvc":
        return _backends_in(subject_id, load_all_rvc_manifests)
    if kind == "denoise":
        return _backends_in(subject_id, denoisemodels.load_all_denoise_manifests)
    return sorted(BACKEND_ENGINES)


def _stranded(config: Config) -> list[weights.StrandedWeights]:
    aliases = {
        manifest.id
        for manifest in (*load_all_manifests().values(), *load_all_asr_manifests().values())
        if getattr(manifest, "weights_of", None) is not None
    }
    return weights.stranded(
        config,
        ModelManifest.weights_family,
        tuple(BACKEND_ENGINES),
        lambda subject_id: (
            () if subject_id in aliases else backends_declaring("model", subject_id)
        ),
    )


def stranded_weights(config: Config) -> list[dict[str, Any]]:
    return [entry.to_dict() for entry in _stranded(config)]


def _stranded_subject(
    config: Config, backend: Backend, subject_id: str
) -> Subject | None:
    # Weights this build declares nothing for on this backend: a model retired from
    # the catalog (qwen3.5-4b-bside-4bit, 1.0.124) leaves its folder in the store,
    # and the store, not the catalog, is what says it is there. It is removable and
    # nothing else: there is no manifest to pull it again or to load it from.
    for entry in _stranded(config):
        if entry.subject_id == subject_id and entry.backend == backend.kind:
            return _stranded_model(config, entry)
    return None


def _stranded_model(config: Config, entry: weights.StrandedWeights) -> Subject:
    def no_pull(**_kwargs: Any) -> weights.InstalledWeights:
        raise weights.WeightsError(
            f"model {entry.subject_id!r} is not in this build's catalog; its "
            f"weights at {entry.path} can only be removed"
        )

    return Subject(
        kind="model",
        id=entry.subject_id,
        name=None,
        job_type="stranded",
        expected_bytes=None,
        source=f"stranded:{entry.path}",
        pull_command="",
        installed=lambda: weights.InstalledWeights(
            path=entry.path,
            hf_repo=None,
            revision=None,
            bytes=entry.bytes,
            pulled=None,
            source="stranded",
        ),
        pull=no_pull,
        remove=lambda: weights.remove_stranded(config, entry),
    )


def find(
    config: Config, backend: Backend, kind: str, subject_id: str
) -> Subject | None:
    for subject in subjects(config, backend):
        if subject.kind == kind and subject.id == subject_id:
            return subject
    return None


def find_resolving(
    config: Config, backend: Backend, kind: str, subject_id: str
) -> Subject | None:
    found = find(config, backend, kind, subject_id)
    if found is not None or kind != "voice":
        return found
    if refresh_unresolved(config.home, subject_id) is None:
        return None
    return find(config, backend, kind, subject_id)


LISTING = "`crucible api catalog` (GET /v1/catalog)"


class RemoveRefused(CrucibleError):

    def __init__(
        self, status_code: int, code: str, message: str, details: dict[str, Any]
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details


@dataclass(frozen=True)
class Removed:
    subject: Subject
    found: weights.InstalledWeights
    path: Path


def locate_installed(
    config: Config, backend: Backend, kind: str, subject_id: str
) -> tuple[Subject, weights.InstalledWeights]:
    named = {"kind": kind, "id": subject_id}
    if kind not in KINDS:
        raise RemoveRefused(
            404,
            "subject_unknown",
            f"{kind!r} is not a subject kind; they are {list(KINDS)}",
            named,
        )
    subject = find(config, backend, kind, subject_id)
    if subject is None and kind == "model":
        subject = _stranded_subject(config, backend, subject_id)
    if subject is None:
        raise RemoveRefused(
            404,
            "subject_unknown",
            f"this server has no {kind} called {subject_id!r} for "
            f"{backend.kind}, and no weights under that id are left in its store. "
            f"{LISTING} lists every subject it can hold",
            named,
        )
    found = subject.installed()
    if found is None:
        raise RemoveRefused(
            409,
            "subject_not_installed",
            f"{kind} {subject_id!r} is not installed on this server, so "
            "there is nothing to remove. Refused rather than answered 204: "
            "a caller told 'done' about a subject that was never there "
            "would believe a migration had deleted something",
            named,
        )
    return subject, found


def remove_subject(
    config: Config,
    backend: Backend,
    kind: str,
    subject_id: str,
    holder: Callable[[Subject], dict[str, Any] | None],
) -> Removed:
    subject, found = locate_installed(config, backend, kind, subject_id)
    who = holder(subject)
    if who is not None:
        raise RemoveRefused(
            409,
            "subject_in_use",
            f"{kind} {subject_id!r} cannot be removed: {who['who']}. "
            "Deleting the files under a running engine would leave it "
            "serving a model that is no longer on the disk",
            who,
        )
    return Removed(subject=subject, found=found, path=subject.remove())


def rows(config: Config, backend: Backend, residency: Residency) -> list[dict[str, Any]]:
    resident = residency.resident

    def is_resident(subject: Subject) -> bool:
        wanted = _RESIDENT_KIND_FOR_JOB_TYPE.get(subject.job_type)
        if wanted is None or resident is None:
            return False
        return resident.kind == wanted and resident.id == subject.id

    try:
        built: list[dict[str, Any]] = []
        for subject in subjects(config, backend):
            found = subject.installed()
            built.append(
                {
                    "kind": subject.kind,
                    "id": subject.id,
                    "name": subject.name,
                    "job_type": subject.job_type,
                    "installed": found is not None,
                    "installed_bytes": None if found is None else found.bytes,
                    "expected_bytes": subject.expected_bytes,
                    "shares_weights_of": subject.shares_weights_of,
                    "missing_files": (
                        None
                        if subject.missing_files is None
                        else subject.missing_files()
                    ),
                    # Retired with the manifests' `[local] minimum_for` (2026-10-09):
                    # always empty, and still sent because SDKs before then demand it.
                    "floors": [],
                    "license": None,
                    "source": subject.source,
                    "resident": is_resident(subject),
                }
            )
    except ApiError:
        raise
    except CrucibleError as exc:
        raise ApiError(
            503,
            "catalog_unreadable",
            f"this server cannot read its own catalog: {type(exc).__name__}: "
            f"{exc}. Nothing is listed rather than some of it, because a catalog "
            "missing a row is indistinguishable from a build that does not ship "
            "it",
        ) from None
    return built


class Removals:

    LIMIT = 20

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def record(
        self,
        *,
        kind: str,
        subject_id: str,
        bytes_freed: int | None,
        act: str | None,
        client: str | None,
    ) -> None:
        row = {
            "at": utcnow(),
            "kind": kind,
            "id": subject_id,
            "bytes_freed": bytes_freed,
            "act": act,
            "client": client,
        }
        with self._lock:
            self._rows.append(row)
            del self._rows[: -self.LIMIT]

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._rows))


__all__ = [
    "KINDS",
    "LISTING",
    "Removals",
    "RemoveRefused",
    "Removed",
    "RVC_BASE_ID",
    "Subject",
    "declared_ids",
    "find",
    "ids_reading",
    "locate_installed",
    "remove_subject",
    "rows",
    "subjects",
]
