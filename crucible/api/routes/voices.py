from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request, Response

from ...errors import ApiError
from ...jobs import voice_rows
from ...voicecatalog import check_updates, load_all_voices
from ...voicerepo import pinned_backends, remove_home_pin, repin
from ...voices import VoiceError, remove_home_voice, voice_document, write_home_voice
from ...weights import WeightsError, resolve_revision
from ..context import AppContext, Routers
from ..deps import tts_enabled
from ..responses import VoiceInfo


async def _manifest_document(request: Request) -> dict[str, Any]:
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
    return document


def _refuse_both_or_neither(document: dict[str, Any]) -> None:
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


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency
    tts_on = [tts_enabled(config)]

    def rows() -> list[dict[str, Any]]:
        return voice_rows(config, backend, residency, store=ctx.store)

    def written(voice_id: str, path: Any) -> dict[str, Any]:
        found = [row for row in rows() if row.get("id") == voice_id]
        return {"voice": found[0] if found else None, "path": str(path)}

    def refuse_if_resident(voice_id: str, why: str) -> None:
        if residency.resident_kind == "tts" and residency.resident_id == voice_id:
            raise ApiError(409, "voice_in_use", why, {"id": voice_id})

    @private.get(
        "/voices",
        dependencies=tts_on,
        response_model=list[VoiceInfo],
        response_model_exclude_unset=True,
    )
    async def voices() -> list[dict[str, Any]]:
        """Every voice this build has a manifest for, and where it stands here."""
        return rows()

    @private.post("/voices/updates", dependencies=tts_on)
    async def voice_updates() -> dict[str, Any]:
        """Look up, on the Hub, the tag every voice here follows, and say which ones a
        pull would move. The only request that resolves a tag; GET /v1/voices never does.
        """
        return {"voices": await asyncio.to_thread(check_updates, config.home)}

    @private.get("/voices/{voice_id}/manifest", dependencies=tts_on)
    async def voice_manifest(voice_id: str) -> dict[str, Any]:
        """One voice's settings as a whole local manifest document, ready to edit and
        send back with `PUT`. `not_carried` names what the local schema cannot hold.
        """
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

    @private.put("/voices/{voice_id}", dependencies=tts_on)
    async def voice_write(request: Request, voice_id: str) -> dict[str, Any]:
        """Pin a voice to a repo revision (`{"pin": ...}`) or write a local manifest
        override (`{"voice": ...}`), and return its `/v1/voices` row. A missing revision
        is resolved to the repo's head.
        """
        document = await _manifest_document(request)
        refuse_if_resident(
            voice_id,
            f"voice {voice_id!r} is loaded on this server right now, so its "
            "manifest cannot be rewritten underneath it. Unload it and try "
            "again",
        )
        _refuse_both_or_neither(document)
        if "pin" in document:
            pin = repin(config, voice_id, document["pin"], resolve=resolve_revision)
            return written(voice_id, pin.path)

        block = document.get("voice")
        if isinstance(block, dict):
            document = {
                **document,
                "voice": pinned_backends(config, block, resolve=resolve_revision),
            }
        try:
            manifest = write_home_voice(voice_id, document)
        except WeightsError as exc:
            raise ApiError(400, "revision_unresolved", str(exc)) from exc
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc
        return written(manifest.id, manifest.path)

    @private.delete("/voices/{voice_id}", status_code=204, dependencies=tts_on)
    async def voice_remove(voice_id: str) -> Response:
        """Remove this machine's pin or override for a voice; a shipped voice reverts to
        its packaged manifest. Never deletes weights; an unknown id answers 204.
        """
        refuse_if_resident(
            voice_id,
            f"voice {voice_id!r} is loaded on this server right now. Unload "
            "it before removing the manifest it is running from",
        )
        try:
            gone_pin = remove_home_pin(voice_id)
            gone_voice = remove_home_voice(voice_id)
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc
        if not gone_pin and not gone_voice and voice_id in load_all_voices():
            raise ApiError(
                404,
                "voice_not_custom",
                f"{voice_id!r} is a shipped voice here, not one this server "
                "added, and this door removes only what was added. The "
                "packaged set is the install and is restored by it",
                {"id": voice_id},
            )
        return Response(status_code=204)
