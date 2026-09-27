from __future__ import annotations

from typing import Any

from fastapi import Request, Response

from ...config import Config
from ...errors import ApiError
from ...jobs import disabled_error, voice_rows
from ...voicerepo import Pin as VoicePin
from ...voicerepo import home_pins_path, remove_home_pin, voice_for_pin, write_home_pin
from ...voices import (
    VoiceError,
    load_all_voices,
    remove_home_voice,
    voice_document,
    write_home_voice,
)
from ...weights import WeightsError, resolve_revision
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    app, config, backend, residency = ctx.app, ctx.config, ctx.backend, ctx.residency

    @private.get("/voices")
    async def voices(request: Request) -> list[dict[str, Any]]:
        """Every voice this build has a manifest for, and where it stands here."""
        if not config.enable_tts:
            raise disabled_error("tts", config)
        return voice_rows(
            config, backend, residency,
            leases=request.app.state.leases, store=request.app.state.store,
        )

    @private.get("/voices/{voice_id}/manifest")
    async def voice_manifest(voice_id: str) -> dict[str, Any]:
        """One voice's settings as a whole local manifest document, ready to edit and
        send back with `PUT`. `not_carried` names what the local schema cannot hold.
        """
        if not config.enable_tts:
            raise disabled_error("tts", config)
        found = load_all_voices().get(voice_id)
        if found is None:
            raise ApiError(
                404,
                "unknown_voice",
                f"there is no voice {voice_id!r} on this server",
                {"id": voice_id},
            )
        document, not_carried = voice_document(found)
        return {
            "id": found.id,
            "manifest": found.manifest_source,
            "path": str(found.path),
            "document": document,
            "not_carried": not_carried,
        }

    def _pinned_backends(live: Config, block: dict[str, Any]) -> dict[str, Any]:
        backends = block.get("backends")
        if not isinstance(backends, dict):
            return block
        pinned: dict[str, Any] = {}
        for name, spec in backends.items():
            if not isinstance(spec, dict):
                pinned[name] = spec
                continue
            repo = spec.get("hf_repo")
            if spec.get("revision") is None and isinstance(repo, str) and repo != "":
                pinned[name] = {**spec, "revision": resolve_revision(live, repo)}
                continue
            pinned[name] = {k: v for k, v in spec.items()
                            if not (k == "revision" and v is None)}
        return {**block, "backends": pinned}

    def _repin(live: Config, voice_id: str, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ApiError(
                400,
                "voice_invalid",
                "`pin` must be a table of hf_repo and revision, got "
                f"{type(body).__name__}",
            )
        unknown = sorted(set(body) - {"hf_repo", "revision"})
        if unknown:
            raise ApiError(
                400,
                "voice_invalid",
                f"`pin` carries unknown key(s) {unknown}; a pin is exactly "
                "hf_repo and revision. Everything else about a voice lives in "
                "its own repo's crucible-voice.toml at that revision",
            )
        repo = body.get("hf_repo")
        if not isinstance(repo, str) or repo.strip() == "":
            raise ApiError(
                400, "voice_invalid", "`pin` names no hf_repo"
            )
        revision = body.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ApiError(
                400,
                "voice_invalid",
                "`pin.revision` must be a 40-character commit sha, or null to "
                f"resolve this repo's head, got {type(revision).__name__}",
            )
        if revision is None or revision.strip() == "":
            try:
                revision = resolve_revision(live, repo)
            except WeightsError as exc:
                raise ApiError(400, "revision_unresolved", str(exc)) from exc

        candidate = VoicePin(
            id=voice_id,
            hf_repo=repo,
            revision=revision,
            path=home_pins_path(),
        )
        try:
            voice_for_pin(candidate)
            pin = write_home_pin(voice_id, repo, revision)
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc

        rows = [row for row in voice_rows(live, backend, residency,
                                          leases=app.state.leases,
                                          store=app.state.store)
                if row.get("id") == voice_id]
        return {"voice": rows[0] if rows else None, "path": str(pin.path)}

    @private.put("/voices/{voice_id}")
    async def voice_write(request: Request, voice_id: str) -> dict[str, Any]:
        """Pin a voice to a repo revision (`{"pin": ...}`) or write a local manifest
        override (`{"voice": ...}`), and return its `/v1/voices` row. A missing revision
        is resolved to the repo's head.
        """
        if not config.enable_tts:
            raise disabled_error("tts", config)
        live: Config = request.app.state.config

        try:
            document = await request.json()
        except Exception as exc:
            raise ApiError(
                400, "voice_invalid", f"the request body is not JSON: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ApiError(
                400,
                "voice_invalid",
                "the body must be the manifest document — a table with a `voice` "
                "key, exactly as voices/<id>.toml holds it",
            )

        if residency.resident_kind == "tts" and residency.resident_id == voice_id:
            raise ApiError(
                409,
                "voice_in_use",
                f"voice {voice_id!r} is loaded on this server right now, so its "
                "manifest cannot be rewritten underneath it. Unload it and try "
                "again",
                {"id": voice_id},
            )

        has_pin = "pin" in document
        has_voice = "voice" in document
        if has_pin and has_voice:
            raise ApiError(
                400,
                "voice_invalid",
                "the body carries both a `pin` and a `voice`. They are two "
                "different decisions — which published revision this machine "
                "serves, and a whole manifest written on this machine — and a "
                "request making both leaves which one wins to the loader. Send "
                "one",
            )
        if not has_pin and not has_voice:
            raise ApiError(
                400,
                "voice_invalid",
                "the body must be either {\"pin\": {hf_repo, revision}} — the "
                "repo and commit this machine serves this voice from — or "
                "{\"voice\": {...}}, the manifest document exactly as "
                "voices/<id>.toml holds it",
            )

        if has_pin:
            return _repin(live, voice_id, document["pin"])

        block = document.get("voice")
        if isinstance(block, dict):
            document = {**document, "voice": _pinned_backends(live, block)}

        try:
            manifest = write_home_voice(voice_id, document)
        except WeightsError as exc:
            raise ApiError(400, "revision_unresolved", str(exc)) from exc
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc

        rows = [row for row in voice_rows(live, backend, residency,
                                          leases=app.state.leases,
                                          store=app.state.store)
                if row.get("id") == manifest.id]
        return {"voice": rows[0] if rows else None, "path": str(manifest.path)}

    @private.delete("/voices/{voice_id}", status_code=204)
    async def voice_remove(request: Request, voice_id: str) -> Response:
        """Remove this machine's pin or override for a voice; a shipped voice reverts to
        its packaged manifest. Never deletes weights; an unknown id answers 204.
        """
        if not config.enable_tts:
            raise disabled_error("tts", config)
        if residency.resident_kind == "tts" and residency.resident_id == voice_id:
            raise ApiError(
                409,
                "voice_in_use",
                f"voice {voice_id!r} is loaded on this server right now. Unload "
                "it before removing the manifest it is running from",
                {"id": voice_id},
            )
        try:
            gone_pin = remove_home_pin(voice_id)
            gone_voice = remove_home_voice(voice_id)
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc
        if not gone_pin and not gone_voice:
            if voice_id in load_all_voices():
                raise ApiError(
                    404,
                    "voice_not_custom",
                    f"{voice_id!r} is a shipped voice here, not one this server "
                    "added, and this door removes only what was added. The "
                    "packaged set is the install and is restored by it",
                    {"id": voice_id},
                )
        return Response(status_code=204)
