from __future__ import annotations

from typing import Any

from fastapi import Request, Response

from ...config import Config
from ...errors import ApiError
from ...jobs import disabled_error, voice_rows
from ...voicerepo import Pin as VoicePin
from ...voicerepo import home_pins_path, remove_home_pin, voice_for_pin, write_home_pin
from ...voices import VoiceError, load_all_voices, remove_home_voice, voice_document, write_home_voice
from ...weights import WeightsError, resolve_revision
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    app, config, backend, residency = ctx.app, ctx.config, ctx.backend, ctx.residency

    # ---------------------------------------------------------------- voices

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
        """One voice's settings as a whole LOCAL manifest document, for editing.

        What `PUT /v1/voices/{id}` with a `voice` body takes, whatever the voice's
        settings came out of: a repo's `crucible-voice.toml` at its pin, this
        machine's override, a packaged file, or the engine's own row. The
        operator console edits this and sends it back as an override (Owen,
        2026-09-26: a person must be able to configure a voice by hand).

        `manifest` says which kind of file that was. `not_carried` names what
        the local schema cannot hold (a repo's `pace_basis`, for one), so an
        override made from a pinned voice is not silently poorer than it.
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
        """Fill in a missing `revision` per backend from the repo's head sha.

        ── Why a caller is allowed to leave it out ────────────────────────────

        `voices.py` requires a full 40-character sha and refuses a branch name,
        because a pull has to be reproducible. That is right and it makes the
        field impossible for a person to supply from a link: nobody knows their
        own repo's head sha, and looking it up needs Hub access and, for a
        private repo, a token. The engine has both.

        ── ABSENT AND NULL BOTH MEAN "PIN IT FOR ME", and a STRING IS OBEYED ──

        A caller that names a revision gets exactly that revision, unresolved
        and unchecked here — asking for an older commit is a real thing to want
        and this is not the door that second-guesses it. Only the absence is
        filled, so nothing a caller wrote is ever replaced.

        A block that is not a table, or names no `hf_repo`, is left ALONE rather
        than repaired: the validator has a sentence for each of those and it is
        better than anything invented here.

        THAT IS ALSO WHAT KEEPS A LOCAL BLOCK OUT OF THIS
        (PHASE18-UNCERTIFIED.md section 3). One that names `path` names no
        `hf_repo`, so there is nothing to resolve a revision from and nothing
        here touches it — which is right rather than lucky: a directory has no
        commit to look up, and a `revision` beside a `path` is refused by the
        validator anyway.
        """
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
            # `None` with no repo to resolve from would reach the validator as a
            # type error about `revision`, which is the wrong sentence. Drop the
            # key so it is reported as missing, which is what it is.
            pinned[name] = {k: v for k, v in spec.items()
                            if not (k == "revision" and v is None)}
        return {**block, "backends": pinned}

    # ------------------------------------------------- voices this host owns
    #
    # PHASE3-TTS.md section 2 gave `voices/*.toml` one home: the package. That
    # made deploying a voice mean cutting a release, which Owen ended on
    # 2026-09-16 — *"we dont have to cut a new release every time we deploy a
    # model do we? ... i train models all the time. nearly every night"* — with
    # the `<CRUCIBLE_HOME>/voices` overlay that `voice_dirs()` reads after the
    # packaged set.
    #
    # THESE TWO DOORS ARE WHAT MAKE THAT OVERLAY REACHABLE. Until now the only
    # way to fill it was to put a file on the machine by hand, which works for
    # the engine you are sitting at and not at all for one across the room —
    # and "across the room" is the ordinary case, because the Mac is a backend
    # and nobody wants to ssh to add a voice they just trained.
    #
    # THE BODY IS THE FILE. A request carries the same `{"voice": {...}}`
    # document the TOML holds, and `crucible/voices.py` validates it with the
    # same `_parse` every packaged manifest goes through. A pydantic mirror of
    # the manifest schema would be a second author of that format, and the two
    # would disagree the first time a field grew a type — silently, because the
    # file would still parse.

    def _repin(live: Config, voice_id: str, body: Any) -> dict[str, Any]:
        """`{"pin": {...}}` — write this machine's pins row. Section 2.4.

        THE PIN IS LOADED BEFORE IT IS WRITTEN. `voice_for_pin` fetches the
        `crucible-voice.toml` at that revision, parses it and merges it with this
        box's `[tts.<engine>]` table, so a revision that carries no manifest, a
        manifest this build's schema cannot read, and a server that has never
        been told what this engine costs are each refused HERE, by name, with
        nothing written. A door that wrote first would leave a server holding a
        pin nothing can load and would report that by breaking the catalog.

        `revision: null` and an ABSENT revision both mean "pin the head for me",
        exactly as `_pinned_backends` reads them on the other body: nobody knows
        their own repo's head sha, and the engine is what holds the HuggingFace
        credential and does the fetching.
        """
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
        """Add or replace a voice this machine owns. Returns its `/v1/voices` row.

        `revision` MAY BE OMITTED per backend, and that is the whole reason a
        person can paste a repo id: a manifest needs a full commit sha so a pull
        is reproducible, and resolving one is the engine's job because the
        engine is what fetches and what holds the HuggingFace credential.

        REPLACING A PACKAGED VOICE IS ALLOWED and is not an accident: the
        overlay is documented to win on a shared id, and deleting the overlay
        brings the packaged voice back, which is what makes trying an override
        safe. A door that refused would make the safe thing impossible and the
        unsafe thing (editing the install) the only way.
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

        # A voice that is ON THE CARD is not re-described underneath itself: its
        # pace, cap and sampling are in use by a render that is already running,
        # and a manifest is read per job rather than held, so the change would
        # land mid-book. Refused by name, with what to do about it.
        if residency.resident_kind == "tts" and residency.resident_id == voice_id:
            raise ApiError(
                409,
                "voice_in_use",
                f"voice {voice_id!r} is loaded on this server right now, so its "
                "manifest cannot be rewritten underneath it. Unload it and try "
                "again",
                {"id": voice_id},
            )

        # TWO BODIES, TOLD APART BY SHAPE (PHASE21 section 2.4), and the two are
        # different decisions rather than two spellings of one.
        #
        #   {"pin": {...}}    REPIN. The voice's facts live in its own repo at
        #                     the named sha, and this machine is choosing which
        #                     sha. This is what a deploy does, per machine, and
        #                     it is the door this phase exists to make ordinary.
        #
        #   {"voice": {...}}  A LOCAL OVERRIDE: a whole manifest, exactly as
        #                     `voices/<id>.toml` holds it, written into this
        #                     machine's overlay. The PHASE18 `path` + `identity`
        #                     arm lives here, and so does a person retuning a
        #                     shipped voice on their own machine.
        #
        # BOTH IS REFUSED and so is NEITHER. A body carrying both has two
        # answers to "what is this voice" and the loader would pick one
        # (`load_all_voices` prefers the override); a body carrying neither is
        # not a manifest at all.
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
        # The row is drawn from the same reader every other caller uses, so what
        # comes back is what `GET /v1/voices` will say — not an echo of the
        # request, which would report a save the file may have shaped otherwise.
        return {"voice": rows[0] if rows else None, "path": str(manifest.path)}

    @private.delete("/voices/{voice_id}", status_code=204)
    async def voice_remove(request: Request, voice_id: str) -> Response:
        """Delete this machine's own manifest for a voice. The packaged set stands.

        A voice that only ever existed in the overlay goes entirely; one that was
        SHADOWING a packaged voice reverts to the packaged manifest, which is the
        undo for an override that did not work out.

        This deletes the MANIFEST, never the weights. They are a subject like any
        other and `DELETE /v1/catalog/voice/{id}` is what removes them — two
        doors because they are two decisions, and somebody re-describing a voice
        they have just downloaded 8 GB of should not lose the download.
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
        # THE PIN ROW OR THE OVERRIDE, WHICHEVER THIS ID HAS (section 2.4), and
        # both are tried rather than one being guessed at from the request:
        # `PUT` writes one or the other and a person deleting a voice knows only
        # that they added it. Never the weights — those are a catalog subject and
        # `DELETE /v1/catalog/voice/{id}` is what removes them, because somebody
        # re-describing a voice they have just downloaded 8.5 GB of should not
        # lose the download.
        try:
            gone_pin = remove_home_pin(voice_id)
            gone_voice = remove_home_voice(voice_id)
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc
        if not gone_pin and not gone_voice:
            # IDEMPOTENT ON THE STATE REACHED (the ladder's ask, 2026-09-21): a
            # DELETE whose answer was lost to a restart is sent again, and the
            # second one must not read as a failure — `ladder-screen` stayed
            # registered three hours because it did. So a voice this server
            # simply does not have answers 204: the state asked for is the
            # state there is. The 404 stays for a voice that IS here and is not
            # this server's own — packaged, engine-base or a shipped pin —
            # because that one cannot be removed by this door and saying 204
            # would be a lie about it.
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
