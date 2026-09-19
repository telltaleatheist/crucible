"""What `load-voice`, `unload-voice` and `tts` all have to agree about.

These were in `crucible/jobs/tts/__init__.py` while the lifecycle pair was the
whole job type. The render door (PHASE3-TTS.md section 6) needs every one of
them — the same manifest lookup, the same backend spec, the same env and weights
refusals in the same order — and it needs them from a module the package's
`__init__` can import, so they moved here rather than being written twice.

Nothing about their behaviour changed in the move, and the refusal names are the
ones `jobs/tts/__init__.py`'s docstring already lists.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from ... import accelerator, jobenv, weights
from ...config import Config
from ...errors import ApiError
from ...residency import KIND_TTS, Residency
from ...voicereference import ReferenceError, VoiceReference, parse_reference
from ...voices import VoiceBackendSpec, VoiceError, VoiceManifest, load_all_voices
from ..base import ModelDescriptor

__all__ = [
    "describe_voices",
    "known_voice",
    "load_voices",
    "require_loadable",
    "require_reference",
    "validated_params",
    "voice_provenance",
    "voice_rows",
]


def require_reference(
    manifest: VoiceManifest, reference: Any
) -> VoiceReference | None:
    """The clip this load may carry, or the first of three refusals by name.

    PHASE3-TTS.md section 5's amendment. A zero-shot voice IS base weights plus
    a recording, so a load without one has nothing to clone from — the base
    model's own voice is a DIFFERENT voice and would be rendered under this
    id — and a load of anything else WITH one is asking the engine to ignore
    the weights it just named.

        reference_required     kind is `zeroshot` and none was sent
        reference_not_allowed  any other kind, and one was
        reference_malformed    not base64, not a WAV, no transcript, too long

    `reference` is the validated `ReferenceInput` (or None) rather than the raw
    body: shape is pydantic's and content is `voicereference`'s.
    """
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


def load_voices() -> dict[str, VoiceManifest]:
    try:
        return load_all_voices()
    except VoiceError as exc:
        raise ApiError(
            500,
            "voices_unreadable",
            f"this server cannot read its voice manifests: {exc}",
        ) from None


def validated_params(
    model: type[BaseModel], params: dict[str, Any], job_type: str
) -> Any:
    """Validate `params` up front, as a named 400 rather than a 500."""
    try:
        return model.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            f"{job_type} params are not valid: "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def known_voice(voice_id: str) -> VoiceManifest:
    """The manifest for this id, or `unknown_voice` by name."""
    manifests = load_voices()
    manifest = manifests.get(voice_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_voice",
            f"no manifest for voice {voice_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def describe_voices(config: Config, residency: Residency) -> list[ModelDescriptor]:
    """The voices as DESIGN.md section 4's row — what the registry reads.

    `/v1/info`'s `tts` capability carries `voice_rows` verbatim instead
    (PHASE3-TTS.md section 8); this is the row `resolve_model` and `crucible
    doctor` read through `describe_models()`. It takes the Config rather than
    the backend kind alone because `installed` is the puller's stamp under
    `config.home`, and it is the same predicate `voice_rows` and
    `require_loadable` read — one fact, one reader (ARCHITECTURE.md R1).
    """
    backend_kind = config.backend_kind
    rows: list[ModelDescriptor] = []
    for manifest in load_voices().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            # A LOCAL BLOCK HAS NO REVISION AND NO REPO (PHASE18-UNCERTIFIED.md
            # section 3), so the row carries what it does have: the identity its
            # registrant asserted, and the directory instead of a repo id. The
            # `path:` prefix is what keeps `source` one column meaning one
            # thing — every other row here is a bare `<owner>/<name>`, and an
            # unprefixed directory beside those would be a reader's problem to
            # tell apart. (`crucible/catalog.py` prefixes BOTH shapes, `hf:`
            # included; this row does not, so only the new shape is marked.)
            revision, source, estimate = (
                spec.weights_identity,
                spec.hf_repo if spec.hf_repo is not None else f"path:{spec.path}",
                spec.memory_bytes_estimate,
            )
            # The same predicate `voice_rows` and `require_loadable` read: the
            # puller's stamp, at the revision this host's block pins.
            installed = weights.installed(config, manifest, spec) is not None
        else:
            # A backend this manifest has no block for has nothing to install.
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


def voice_rows(
    config: Config, backend: Any, residency: Residency
) -> list[dict[str, Any]]:
    """`GET /v1/voices` — PHASE3-TTS.md section 2.

    These same rows are the `tts` capability's rows in `GET /v1/info`: one shape,
    one producer, the same rule and the same reason as `llm`'s models. One voice,
    one description; a client never reconciles two.

    `loadable` answers "is everything this host needs in place", which is a fact
    about the disk. Like `model_rows` it deliberately does **not** run
    nvidia-smi: the accelerator's state changes between a listing and a request,
    so the guard runs at load time. A row saying `loadable: true` can still be
    refused with `accelerator_busy`.

    **`sampling` is deliberately not on the row**, nor are the EOS levers, the
    token-budget formula or the engine flags. That is engine tuning, it is the
    server's, and publishing it invites a client to send it back. What a client
    gets is the shape it must pack to (`pace`, `max_chars`) and the identity it
    must record (`fingerprint`).

    `[voice.serving].max_num_seqs` IS NOT ON THE ROW EITHER, by that same rule
    and by the division-of-knowledge ruling behind it: it is how wide the
    server admits, a Crucible-side configuration number, and a client has no
    decision to make with it. It reaches narrator through the engine's
    environment (`crucible/engines/narrator.py`) and stops there.
    """
    backend_kind = backend.kind
    rows: list[dict[str, Any]] = []
    for manifest in load_voices().values():
        supported = manifest.supports(backend_kind)
        estimate: int | None = None
        basis: str | None = None
        revision: str | None = None
        source: str | None = None
        identity_basis: str | None = None
        fingerprint: str | None = None
        max_chars: int | None = None
        max_chars_basis: str | None = None
        is_installed = False
        reason: str | None = None
        if not supported:
            reason = (
                f"{manifest.path.name} has no {backend_kind} block; it declares "
                f"{sorted(manifest.backends)}"
            )
        else:
            spec = manifest.spec(backend_kind)
            estimate = spec.memory_bytes_estimate
            basis = spec.estimate_basis
            # THE BLOCK'S IDENTITY AND NOT `spec.revision`, because this row
            # states `fingerprint == f"{id}@{revision}"` and a local block has
            # no revision: read off the raw field, a local voice published a
            # fingerprint naming a checkpoint beside a `revision` of null, and
            # the two halves of one record disagreed. `weights_identity` is the
            # one owner of that fact (crucible/voices.py) and `identity_basis`
            # below is what says how much it is worth.
            revision = spec.weights_identity
            source = spec.source
            identity_basis = spec.identity_basis
            fingerprint = manifest.fingerprint(backend_kind)
            max_chars = spec.max_chars
            max_chars_basis = spec.max_chars_basis
            is_installed = weights.installed(config, manifest, spec) is not None
            env = jobenv.env_status(
                config.home,
                jobenv.tts_env(manifest.narrator_engine, backend_kind),
                backend_kind,
            )
            if estimate > backend.gpu.vram_bytes:
                # Not loadable here at all, so say so instead of asking for an
                # 8.5 GB download first.
                reason = (
                    f"needs {estimate / 1024 ** 3:.1f} GiB and "
                    f"{backend.gpu.name} has {backend.gpu.vram_bytes / 1024 ** 3:.1f}"
                    " GiB in total"
                )
            elif not env.installed:
                reason = (
                    f"the tts env for {manifest.narrator_engine} is not ready: "
                    f"{env.detail}"
                )
            elif not is_installed and spec.source == weights.LOCAL:
                # A LOCAL VOICE IS NOT PULLABLE, so the reason must not tell its
                # reader to pull it (PHASE18-UNCERTIFIED.md section 3). The
                # directory belongs to whatever put it there, and a screening
                # merge being gone is the expected end of its life rather than a
                # broken install.
                reason = (
                    f"no weights at {spec.path} — this voice names a directory on "
                    "this server, which Crucible does not fetch and cannot replace"
                )
            elif not is_installed:
                directory = weights.weights_dir(
                    config, manifest.weights_family, manifest.id, backend_kind
                )
                reason = (
                    f"no weights at {directory} — run "
                    f"`crucible voices pull {manifest.id}`"
                )
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "kind": manifest.kind,
                "language": manifest.language,
                "narrator_engine": manifest.narrator_engine,
                "backend_supported": supported,
                "installed": is_installed,
                "resident": residency.is_resident(KIND_TTS, manifest.id),
                "loadable": reason is None,
                "reason": reason,
                # These four live in the backend block this host may not have,
                # and are null rather than 0 or "" when it does not: a 0 estimate
                # would read as "needs nothing" and an empty revision as a pin.
                "revision": revision,
                "fingerprint": fingerprint,
                # WHERE THE BYTES COME FROM, and how much `fingerprint` is
                # worth. `"pinned"` means the sha was fetched and stamped and
                # the identity is VERIFIED; `"local"` means a directory on this
                # machine whose identity the registrant ASSERTED and nothing
                # checked. Both on the row for `estimate_basis`'s reason: a
                # client comparing two renders must not be able to mistake one
                # kind of identity for the other.
                "source": source,
                "identity_basis": identity_basis,
                "memory_bytes_estimate": estimate,
                # Whether somebody watched the card for that number or it came
                # off the engine's own configured reservation. On the row rather
                # than only in the manifest, so nothing downstream can mistake
                # one for the other (crucible/voices.py).
                "estimate_basis": basis,
                "max_chars": max_chars,
                # HOW THE CAP AND THE BAND WERE GOT, beside the numbers
                # themselves (PHASE21 sections 2.1 and 6). `"measured"` is a
                # sweep on these weights on this arm; `"placeholder"` is a
                # number somebody wrote down so the arm could be served at all;
                # `"inherited"` is a pace taken from a predecessor run. NULL
                # MEANS THIS VOICE'S MANIFEST SCHEMA CANNOT SAY, which is a
                # third statement and not a fourth word for "measured" — every
                # `voices/*.toml` reports null, because that schema has no such
                # key. Both ride on the row for `estimate_basis`'s reason: an
                # inherited pace is indistinguishable from a measured one at
                # the point of use, and that is exactly how deathstalker's
                # 16.64 survived onto weights that measured 15.91.
                "max_chars_basis": max_chars_basis,
                "pace_basis": manifest.pace_basis,
                # WHICH KIND OF FILE THIS ROW'S FACTS CAME OUT OF: `"repo"` is
                # a `crucible-voice.toml` in the weights' own repo at the
                # pinned revision, `"override"` a whole manifest written to
                # this machine through `PUT /v1/voices/{id}`, `"engine"` the
                # narrator engine's own base behaviour (section 2.6), and
                # `"packaged"` one of the manifests this build still ships,
                # which section 8.3 deletes.
                "manifest": manifest.manifest_source,
                "sample_rate": manifest.sample_rate,
                # How many rungs this voice's ladder has, so a client can ask
                # how many takes exist BEFORE it submits one — `take: N` is
                # refused as `unknown_take` past the end and never clamped,
                # and a client spreading N candidates across the ladder (which
                # is what BookForge's Correct Sentences does) has to know N.
                # The rungs' NUMBERS are deliberately not here, for the same
                # reason `sampling` is not: they are engine tuning, they are
                # the server's, and publishing them invites a client to send
                # them back.
                "takes": len(manifest.takes),
                # Whether a `load-voice` for this row must carry a reference
                # clip (`params.reference`) — true for a zeroshot voice and
                # false for every other kind. On the row so a picker can show
                # the clip field before the load is refused
                # (`reference_required`), and derived from `kind` rather than
                # left for a client to derive, because "which kinds need one"
                # is the server's rule.
                "needs_reference": manifest.kind == "zeroshot",
                "pace": manifest.pace.to_dict(),
            }
        )
    return rows


def require_loadable(
    config: Config, backend: Any, voice_id: str
) -> tuple[VoiceManifest, VoiceBackendSpec, Any]:
    """Manifest, backend spec, interpreter and weights, or the named refusal.

    The order is `jobs/llm`'s, and deliberately so: what can never be fixed, then
    what an install or a pull would fix, then what the live accelerator says.
    """
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
        need_bytes=spec.memory_bytes_estimate,
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
                # Both null on a local block, which is what it has: no repo was
                # named and no commit was pinned. The directory is in the
                # message `weights.require_installed` already wrote.
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "path": spec.path,
            },
        ) from None
    return manifest, spec, (python, installed)


def voice_provenance(backend_kind: str, voice_id: str | None) -> dict[str, Any] | None:
    """The `model` block of a tts artifact's provenance sidecar.

    For `tts` the model IS the voice (PHASE3-TTS.md section 6), so the sidecar
    names it with the same three keys every other type uses rather than inventing
    a fourth word for the same idea. The revision is this host's backend pin,
    which is a statement about bytes: a load refuses weights pulled at any other
    revision, so the pin the manifest names is the checkpoint the engine read —
    and a finished audiobook that says which voice rendered it should also say
    which merge of that voice, because two merges of one fine-tune are two
    narrators.

    ON A LOCAL VOICE THE REVISION IS THE ASSERTED IDENTITY, and `identity_basis`
    beside it says so (PHASE18-UNCERTIFIED.md section 3.1). Leaving `revision`
    null there would be the worse of the two available lies: a sidecar whose
    `fingerprint` names a checkpoint and whose `revision` says nothing reads as
    a render whose weights were never established, when in fact they were
    stated — just by a person rather than by a sha. What must never happen is
    an ASSERTED identity being read as a VERIFIED one, and that is what the
    basis is for.
    """
    if voice_id is None:
        return None
    manifest = known_voice(voice_id)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        # Unreachable through the API: `preflight` refuses `backend_unsupported`
        # before a job exists. A sidecar still has to say something true if it is
        # reached another way, and inventing a revision is not it.
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
