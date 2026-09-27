from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import denoisemodels, lineup, llamacpp, rvcbase, weights
from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import Backend
from .config import Config
from .errors import ApiError, CrucibleError
from .manifests import BACKEND_ENGINES, ModelManifest, load_all_manifests
from .jobs.base import utcnow
from .residency import KIND_ALIGN, KIND_DENOISE, KIND_LLM, KIND_TTS, Residency
from .rvcmodels import load_all_rvc_manifests
from .voices import load_all_voices

KINDS: tuple[str, ...] = ("model", "voice", "rvc", "rvc-base", "denoise", "engine")

RVC_BASE_ID = "base"

_RESIDENT_KIND_FOR_JOB_TYPE: dict[str, str] = {
    "llm": KIND_LLM,
    "align": KIND_ALIGN,
    "tts": KIND_TTS,
    "denoise": KIND_DENOISE,
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


def subjects(config: Config, backend: Backend) -> list[Subject]:
    found: list[Subject] = []

    for job_type, loaded in (
        ("llm", load_all_manifests()),
        ("asr", load_all_asr_manifests()),
        ("align", load_all_align_manifests()),
    ):
        for manifest in loaded.values():
            if not manifest.supports(backend.kind):
                continue
            spec = manifest.spec(backend.kind)
            base_id = getattr(manifest, "weights_of", None)
            found.append(
                Subject(
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
                    pull_command=f"crucible models pull {manifest.id}",
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
            )

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
                pull_command=f"crucible voices pull {voice.id}",
                installed=_installed_weights(config, voice, spec),
                pull=_pull_weights(config, voice, spec),
                remove=_remove_weights(config, voice, spec),
            )
        )

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
                pull_command=f"crucible rvc pull {model.id}",
                installed=_installed_weights(config, model, spec),
                pull=_pull_archive(config, model, spec),
                remove=_remove_weights(config, model, spec),
            )
        )

    assets = rvcbase.load_rvc_base()
    found.append(
        Subject(
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
    )

    if backend.kind == llamacpp_backend():
        build = llamacpp.build_for(backend.gpu.vendor)
        found.append(
            Subject(
                kind=llamacpp.ENGINE_KIND,
                id=llamacpp.LLAMA_CPP_ID,
                name=f"llama.cpp {llamacpp.LLAMA_CPP_RELEASE} ({build})",
                job_type="llm",
                expected_bytes=llamacpp.expected_bytes(build),
                source=f"github:ggml-org/llama.cpp@{llamacpp.LLAMA_CPP_RELEASE}",
                pull_command="crucible install llm",
                installed=_installed_engine(config, build),
                pull=_pull_engine(config, build),
                remove=lambda: llamacpp.remove(config),
            )
        )

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
                pull_command=f"{denoisemodels.PULL_COMMAND} {separator.id}",
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
            }
        ),
        "voice": sorted(_declared_voice_ids()),
        "rvc": sorted(load_all_rvc_manifests()),
        "rvc-base": [RVC_BASE_ID],
        "denoise": sorted(denoisemodels.load_all_denoise_manifests()),
        llamacpp.ENGINE_KIND: [llamacpp.LLAMA_CPP_ID],
    }


def _declared_voice_ids() -> set[str]:
    from .voicerepo import _parse_pins, packaged_pins_path
    from .voices import _engine_voices, _voices_in, voices_dir

    pins_path = packaged_pins_path()
    pinned = _parse_pins(pins_path.read_text(encoding="utf-8"), pins_path) if pins_path.is_file() else {}
    return {*pinned, *_engine_voices(), *_voices_in(voices_dir())}


def _declared_voice_backends(voice_id: str) -> dict[str, Any]:
    import tempfile
    from types import SimpleNamespace

    from .voicerepo import _parse_pins, fetch_repo_manifest, packaged_pins_path, parse_repo_manifest
    from .voices import _engine_voices, _voices_in, voices_dir

    shipped = {**_engine_voices(), **_voices_in(voices_dir())}
    if voice_id in shipped:
        return {voice_id: shipped[voice_id]}
    pins_path = packaged_pins_path()
    pins = _parse_pins(pins_path.read_text(encoding="utf-8"), pins_path) if pins_path.is_file() else {}
    pin = pins.get(voice_id)
    if pin is None:
        return {}
    scratch = Path(tempfile.gettempdir()) / "crucible-declared-voices"
    text, where = fetch_repo_manifest(scratch, pin)
    repo = parse_repo_manifest(text, where)
    return {voice_id: SimpleNamespace(backends=dict(repo.arms))}


def backends_declaring(kind: str, subject_id: str) -> list[str]:
    loaders: dict[str, Any] = {
        "model": (
            load_all_manifests,
            load_all_asr_manifests,
            load_all_align_manifests,
        ),
        "voice": (lambda: _declared_voice_backends(subject_id),),
        "rvc": (load_all_rvc_manifests,),
        "denoise": (denoisemodels.load_all_denoise_manifests,),
    }
    found: set[str] = set()
    for load in loaders.get(kind, ()):
        manifest = load().get(subject_id)
        if manifest is not None:
            found.update(manifest.backends)
    if kind not in loaders:
        return sorted(BACKEND_ENGINES)
    return sorted(found)


def stranded_weights(config: Config) -> list[dict[str, Any]]:
    aliases = {
        manifest.id
        for manifest in (*load_all_manifests().values(), *load_all_asr_manifests().values())
        if getattr(manifest, "weights_of", None) is not None
    }
    return [
        entry.to_dict()
        for entry in weights.stranded(
            config,
            ModelManifest.weights_family,
            tuple(BACKEND_ENGINES),
            lambda subject_id: (
                () if subject_id in aliases else backends_declaring("model", subject_id)
            ),
        )
    ]


def find(
    config: Config, backend: Backend, kind: str, subject_id: str
) -> Subject | None:
    for subject in subjects(config, backend):
        if subject.kind == kind and subject.id == subject_id:
            return subject
    return None


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
    if subject is None:
        raise RemoveRefused(
            404,
            "subject_unknown",
            f"this server has no {kind} called {subject_id!r} for "
            f"{backend.kind}. {LISTING} lists every subject it can hold",
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


def _floors_by_model() -> dict[str, list[str]]:
    built, _omitted = lineup.build()
    inverted: dict[str, list[str]] = {}
    for capability_class, model_id in lineup.floors(built).items():
        inverted.setdefault(model_id, []).append(capability_class)
    return inverted


def rows(config: Config, backend: Backend, residency: Residency) -> list[dict[str, Any]]:
    resident = residency.resident

    def is_resident(subject: Subject) -> bool:
        wanted = _RESIDENT_KIND_FOR_JOB_TYPE.get(subject.job_type)
        if wanted is None or resident is None:
            return False
        return resident.kind == wanted and resident.id == subject.id

    try:
        floors = _floors_by_model()
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
                    "floors": (
                        list(floors.get(subject.id, []))
                        if subject.kind == "model"
                        else []
                    ),
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
