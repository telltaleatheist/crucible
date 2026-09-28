from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

from ... import accelerator, jobenv, ttsplan, weights
from ... import residency as residency_module
from ...backend import MLX_DARWIN
from ...clock import utcnow
from ...config import Config
from ...engines import EngineError, NarratorEngine, find_free_port, start_engine
from ...errors import ApiError
from ...narratorengines import DOCUMENT_READERS
from ...narratorvoices import write_document
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_TTS,
    Occupant,
    Residency,
    ResidentVoice,
    say_to,
)
from ...voicecatalog import load_all_voices
from ...voicereference import ReferenceError, VoiceReference, parse_reference
from ...voices import VoiceBackendSpec, VoiceError, VoiceManifest
from ..base import ModelDescriptor
from ..template import ManifestCatalog, parse_params

__all__ = [
    "describe_voices",
    "known_voice",
    "load_voices",
    "occupy_voice",
    "require_loadable",
    "require_reference",
    "validated_params",
    "voice_provenance",
    "voice_rows",
]


def require_reference(
    manifest: VoiceManifest, reference: Any
) -> VoiceReference | None:
    if manifest.kind == "zeroshot":
        if reference is None:
            raise ApiError(
                400,
                "reference_required",
                f"voice {manifest.id!r} is a zeroshot voice: it is the base "
                "weights conditioned on a recording, and this load carries no "
                "`params.reference`. Send "
                '`{"data": "<base64 wav>", "transcript": "<the book-exact text '
                'spoken in it>"}`. Without one the engine would come up in the '
                "model's own voice — a different speaker at 12 % of the "
                "narrator ceiling — under this voice's id",
                {"voice": manifest.id, "kind": manifest.kind},
            )
    elif reference is not None:
        raise ApiError(
            400,
            "reference_not_allowed",
            f"voice {manifest.id!r} is a {manifest.kind} voice and this load "
            "carries a `params.reference`. A checkpoint's voice is in its "
            "weights and a token voice's is in the engine; a reference here "
            "would clone from the clip and leave the weights this load names "
            "doing nothing",
            {"voice": manifest.id, "kind": manifest.kind},
        )
    if reference is None:
        return None
    try:
        return parse_reference(reference.model_dump())
    except ReferenceError as exc:
        raise ApiError(
            400, exc.code, str(exc), {"voice": manifest.id}
        ) from None


VOICES: ManifestCatalog[VoiceManifest] = ManifestCatalog(
    lambda: load_all_voices(),
    VoiceError,
    unreadable_code="voices_unreadable",
    what="voice manifests",
    unknown="manifest for voice",
    unknown_code="unknown_voice",
)

validated_params = parse_params


def load_voices() -> dict[str, VoiceManifest]:
    return VOICES.all()


def known_voice(voice_id: str) -> VoiceManifest:
    return VOICES.known(voice_id)


def describe_voices(config: Config, residency: Residency) -> list[ModelDescriptor]:
    backend_kind = config.backend_kind
    rows: list[ModelDescriptor] = []
    for manifest in load_voices().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision, source, estimate = (
                spec.weights_identity,
                spec.hf_repo if spec.hf_repo is not None else f"path:{spec.path}",
                spec.memory_bytes_estimate,
            )
            installed = weights.installed(config, manifest, spec) is not None
        else:
            revision, source, estimate, installed = "", "", 0, False
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                installed=installed,
                resident=residency.is_resident(KIND_TTS, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


def _orphan(
    voice_id: str, source: str | None, residency: Residency,
    *, leases: Any | None, store: Any | None,
) -> bool | None:
    if leases is None or store is None:
        return None
    if source != "local":
        return False
    if residency.is_resident(KIND_TTS, voice_id):
        return False
    lease = leases.current()
    if lease is not None and lease.subject == voice_id:
        return False
    running = store.running
    if running is not None and running.model == voice_id:
        return False
    return not any(job.model == voice_id for job in store.queued())


VOICE_ROW_FIELDS: dict[str, Any] = {
    "id": None, "display": None, "kind": None, "language": None,
    "narrator_engine": None, "backend_supported": False, "installed": False,
    "resident": False, "orphan": None, "loadable": False, "reason": None,
    "revision": None, "fingerprint": None, "source": None, "identity_basis": None,
    "memory_bytes_estimate": None, "estimate_basis": None, "serving": None,
    "max_chars": None, "max_chars_basis": None, "pace_basis": None,
    "inherited_from": None, "manifest": None, "sample_rate": None, "takes": 0,
    "needs_reference": False, "pace": None,
}


def voice_row(**fields: Any) -> dict[str, Any]:
    stray = sorted(set(fields) - set(VOICE_ROW_FIELDS))
    if stray:
        raise KeyError(f"a /v1/voices row has no field(s) {stray}")
    return {**VOICE_ROW_FIELDS, **fields}


def _unloadable_reason(
    config: Config, backend: Any, manifest: VoiceManifest, spec: VoiceBackendSpec,
    is_installed: bool,
) -> str | None:
    env = jobenv.env_status(
        config.home,
        jobenv.tts_env(manifest.narrator_engine, backend.kind),
        backend.kind,
    )
    estimate = spec.memory_bytes_estimate
    if estimate > backend.gpu.vram_bytes:
        return (
            f"needs {estimate / 1024 ** 3:.1f} GiB and "
            f"{backend.gpu.name} has {backend.gpu.vram_bytes / 1024 ** 3:.1f}"
            " GiB in total"
        )
    if not env.installed:
        return (
            f"the tts env for {manifest.narrator_engine} is not ready: "
            f"{env.detail}"
        )
    if not is_installed and spec.source == weights.LOCAL:
        return (
            f"no weights at {spec.path} — this voice names a directory on "
            "this server, which Crucible does not fetch and cannot replace"
        )
    if not is_installed:
        directory = weights.subject_dir(config, manifest, backend.kind)
        return f"no weights at {directory} — run `{manifest.pull_command}`"
    return None


def _backend_fields(
    config: Config, backend: Any, manifest: VoiceManifest
) -> dict[str, Any]:
    if not manifest.supports(backend.kind):
        return {
            "backend_supported": False,
            "reason": (
                f"{manifest.path.name} has no {backend.kind} block; it declares "
                f"{sorted(manifest.backends)}"
            ),
        }
    spec = manifest.spec(backend.kind)
    is_installed = weights.installed(config, manifest, spec) is not None
    reason = _unloadable_reason(config, backend, manifest, spec, is_installed)
    return {
        "backend_supported": True,
        "installed": is_installed,
        "loadable": reason is None,
        "reason": reason,
        "revision": spec.weights_identity,
        "fingerprint": manifest.fingerprint(backend.kind),
        "source": spec.source,
        "identity_basis": spec.identity_basis,
        "memory_bytes_estimate": spec.memory_bytes_estimate,
        "estimate_basis": spec.estimate_basis,
        "max_chars": spec.max_chars,
        "max_chars_basis": spec.max_chars_basis,
    }


def _served_voice_row(
    config: Config, backend: Any, residency: Residency, manifest: VoiceManifest,
    *, leases: Any | None, store: Any | None,
) -> dict[str, Any]:
    on_backend = _backend_fields(config, backend, manifest)
    return voice_row(
        id=manifest.id,
        display=manifest.display,
        kind=manifest.kind,
        language=manifest.language,
        narrator_engine=manifest.narrator_engine,
        resident=residency.is_resident(KIND_TTS, manifest.id),
        orphan=_orphan(
            manifest.id, on_backend.get("source"), residency, leases=leases, store=store
        ),
        serving=None if manifest.serving is None else manifest.serving.to_dict(),
        pace_basis=manifest.pace_basis,
        inherited_from=manifest.inherited_from,
        manifest=manifest.manifest_source,
        sample_rate=manifest.sample_rate,
        takes=len(manifest.takes),
        needs_reference=manifest.kind == "zeroshot",
        pace=manifest.pace.to_dict(),
        **on_backend,
    )


def voice_rows(
    config: Config, backend: Any, residency: Residency,
    *, leases: Any | None = None, store: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in load_voices().values():
        rows.append(
            _served_voice_row(
                config, backend, residency, manifest, leases=leases, store=store
            )
        )
    from ...voicecatalog import unserved_pins

    for voice_id, (revision, why) in sorted(unserved_pins().items()):
        rows.append(
            voice_row(
                id=voice_id, display=voice_id, orphan=False, reason=why,
                revision=revision, source="pinned", manifest="repo",
            )
        )
    return rows


def require_loadable(
    config: Config, backend: Any, voice_id: str
) -> tuple[VoiceManifest, VoiceBackendSpec, Any]:
    backend_kind = backend.kind
    manifest = known_voice(voice_id)
    if not manifest.supports(backend_kind):
        raise ApiError(
            400,
            "backend_unsupported",
            f"voice {voice_id!r} has no {backend_kind} block; {manifest.path.name} "
            f"declares {sorted(manifest.backends)}",
            {"voice": voice_id, "backend": backend_kind,
             "declared": sorted(manifest.backends)},
        )
    spec = manifest.spec(backend_kind)
    accelerator.refuse_if_larger_than_host(
        model_id=voice_id,
        need_bytes=voice_load_plan(config, backend, manifest, spec).floor_bytes,
        host_total_bytes=backend.gpu.vram_bytes,
        host_name=backend.gpu.name,
    )
    env_spec = jobenv.tts_env(manifest.narrator_engine, backend_kind)
    try:
        python = jobenv.require_env(config.home, env_spec, backend_kind)
    except jobenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"cannot load {voice_id!r}: {exc}",
            {
                "voice": voice_id,
                "narrator_engine": manifest.narrator_engine,
                "env": str(jobenv.env_dir(config.home, env_spec)),
            },
        ) from None
    try:
        installed = weights.require_installed(config, manifest, spec)
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "voice_not_installed",
            str(exc),
            {
                "voice": voice_id,
                "source": spec.source,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "path": spec.path,
            },
        ) from None
    return manifest, spec, (python, installed)


def voice_provenance(backend_kind: str, voice_id: str | None) -> dict[str, Any] | None:
    if voice_id is None:
        return None
    manifest = known_voice(voice_id)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        return {
            "id": voice_id,
            "revision": None,
            "identity_basis": None,
            "fingerprint": None,
        }
    return {
        "id": voice_id,
        "revision": spec.weights_identity,
        "identity_basis": spec.identity_basis,
        "fingerprint": manifest.fingerprint(backend_kind),
    }


def voice_load_plan(
    config: Config, backend: Any, manifest: VoiceManifest, spec: VoiceBackendSpec
) -> "ttsplan.LoadPlan":
    return ttsplan.load_plan(
        manifest,
        spec,
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
    )


def _serving_width(manifest: VoiceManifest, serving_width: int | None) -> int | None:
    if manifest.serving is None:
        return None
    if serving_width is None:
        return manifest.serving.max_num_seqs
    return min(serving_width, manifest.serving.max_num_seqs)


def _build_narrator(
    home: Path,
    manifest: VoiceManifest,
    spec: VoiceBackendSpec,
    weights_dir: Path,
    python: Path,
    log_path: Path,
    reference: VoiceReference | None,
    serving_width: int | None,
) -> NarratorEngine:
    voices = (
        write_document(home, manifest, spec, weights_dir, reference)
        if manifest.narrator_engine in DOCUMENT_READERS
        else None
    )
    serving = manifest.serving
    return residency_module.build_voice_engine(
        manifest.narrator_engine,
        python,
        log_path,
        serving_stack=jobenv.tts_env(manifest.narrator_engine, spec.backend).serving_stack,
        max_num_seqs=_serving_width(manifest, serving_width),
        mem_fraction=None if serving is None else serving.mem_fraction,
        context_length=None if serving is None else serving.context_length,
        voices=voices,
        mlx_total_bytes=(
            accelerator.probe_unified_memory()[1]
            if spec.backend == MLX_DARWIN
            else None
        ),
    )


def confirm_voice_loaded(
    engine: NarratorEngine,
    manifest: VoiceManifest,
    weights_dir: Path,
    say: Callable[[str], None],
) -> dict[str, Any]:
    say(f"loading {manifest.id} into narrator from {weights_dir}")
    loaded = engine.load(
        voice=manifest.id, weights_dir=weights_dir, warm=True, on_progress=say
    )
    reported = loaded.get("sampleRate")
    if not isinstance(reported, int) or isinstance(reported, bool):
        raise EngineError(
            f"{engine.name} loaded {manifest.id} and reported sampleRate "
            f"{reported!r}, which is not a sample rate. Every duration and "
            "every byte count downstream is derived from it"
        )
    if reported != manifest.sample_rate:
        raise EngineError(
            f"{engine.name} renders {manifest.id} at {reported} Hz, but "
            f"{manifest.path.name} declares {manifest.sample_rate}. Crucible "
            "refuses rather than resampling: a FLAC written at the manifest's "
            "rate from bytes generated at the engine's is a chunk of the "
            "wrong length, and nothing in the file would say so. Fix the "
            "manifest, or find out why the engine changed"
        )
    say(
        f"narrator loaded {manifest.id}: engine {loaded.get('engine')!r}, "
        f"backend {loaded.get('backend')!r}, {reported} Hz"
    )
    return loaded


def occupy_voice(
    residency: Residency,
    manifest: VoiceManifest,
    spec: VoiceBackendSpec,
    weights_dir: Path,
    python: Path,
    *,
    reference: VoiceReference | None = None,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
    serving_width: int | None = None,
) -> ResidentVoice:
    say = say_to(on_progress)

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        engine = _build_narrator(
            residency.home,
            manifest,
            spec,
            weights_dir,
            python,
            log_path,
            reference,
            serving_width,
        )
        port = find_free_port()
        say(
            f"starting narrator ({manifest.narrator_engine}) for {manifest.id} "
            f"on {spec.backend}; log {log_path}"
        )
        start_engine(
            engine,
            weights_dir,
            manifest.id,
            port,
            [],
            say,
            timeout,
            confirm=lambda: confirm_voice_loaded(engine, manifest, weights_dir, say),
        )
        resident = ResidentVoice(
            voice_id=manifest.id,
            backend=spec.backend,
            narrator_engine=manifest.narrator_engine,
            revision=spec.weights_identity,
            fingerprint=manifest.fingerprint(spec.backend),
            sample_rate=manifest.sample_rate,
            max_chars=spec.max_chars,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=utcnow(),
            reference=None if reference is None else reference.to_report(),
        )
        return Occupant(resident, engine=engine)

    return cast(ResidentVoice, residency.occupy(KIND_TTS, manifest.id, start, say=say))


Residency.load_voice = occupy_voice
