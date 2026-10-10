from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Request

from ... import (
    catalog,
    jobflags,
    llmconcurrency,
    lowvram,
    pairing,
    ttslevers,
    upstreamrecord,
    upstreams,
)
from ... import settings as settings_module
from ...capabilitystore import low_vram_not_offered, set_low_vram
from ...cardfacts import card_for
from ...config import Config, load_config, mint_token, rewrite_config
from ...errors import ApiError, ConfigError
from ...inflight import read_act
from ...interfaces import InterfaceError
from ..caller import client_agent
from ..context import AppContext, Routers


async def _json_body(request: Request, what: str) -> Any:
    try:
        return json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(400, "invalid_request", f"the {what} body is not JSON: {exc}") from None


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    def _installed_subjects() -> dict[str, bool]:
        return {row["id"]: row["installed"] for row in catalog.rows(config, backend, residency)}

    def _document(read: Config) -> dict[str, Any]:
        return settings_module.live_document(
            read,
            installed=_installed_subjects(),
            resident=residency.resident_model,
            resident_voice=residency.resident_voice,
            job_types=jobflags.rows(read, backend),
            bound=(ctx.bind_host, ctx.bind_port),
        )

    def _sync_pairing(read: Config) -> str | None:
        """The local pairing line follows a new name or token at once (the Windows host
        and this computer's apps read it); the sentence when it could not be written."""
        try:
            pairing.sync_pairing_file(
                read.home, name=read.name, port=ctx.bind_port, token=read.token
            )
        except pairing.PairingFileError as exc:
            return str(exc)
        return None

    @private.get("/settings")
    async def get_settings() -> dict[str, Any]:
        """Every setting in config.toml this server reads, and what the running server
        is doing with them: where each class's work runs, the upstreams, the [server],
        [auth], [jobs], [queue], [hf], [tts.<engine>] and (on mlx-darwin)
        [video_desktop] keys, every job type's flag and verdict, the address it listens
        on now (`bound`) and the keys that wait for a restart (`restart_pending`). No
        secret is returned: `key_hint`, `token_hint` and `hf.token_hint` show the last
        four characters.
        """
        return _document(config)

    @private.put("/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        """Apply a partial settings patch, whole or not at all. Live without a restart,
        except `host` and `port`: those are written and listed in `restart_pending`
        until the server starts again (`POST /v1/server/restart`). A `port` change is
        refused `port_fixed_by_windows_host` on a PC, whose Windows host reaches its
        engine on 7100. `hf_token` is write-only (null removes it); `video_desktop` sets
        `[video_desktop]` keys on mlx-darwin (null: the default) and refuses a value out
        of range by name. Answers the full settings document after the write.
        """
        act = read_act(request.headers)
        try:
            patch = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the settings body is not JSON: {exc}"
            ) from None
        resolved = settings_module.resolve(config, patch)
        named_before = config.name
        await asyncio.to_thread(
            lambda: settings_module.apply(
                config,
                resolved,
                gpu_vendor=backend.gpu.vendor,
                card=card_for(config.home, backend.gpu),
            )
        )
        if resolved.changed:
            ctx.settings_history.record(
                act=act,
                client=client_agent(request),
                changed=resolved.changed,
            )
        answer = _document(config)
        answer["pairing_file_error"] = (
            _sync_pairing(config) if config.name != named_before else None
        )
        return answer

    @private.put("/settings/audio/low-vram")
    async def put_audio_low_vram(request: Request) -> dict[str, Any]:
        """Set `[audio] low_vram` with `{"state": "on" | "off" | "auto"}`: `on` and `off`
        are the operator's and Crucible never changes them; `auto` lets Crucible turn it
        on exactly where this card cannot hold a splittable audio model whole. Decides
        the audio capability and `[jobs] enable_audio` again, as `crucible audio
        low-vram` does. Answers the full settings document after the write.
        """
        act = read_act(request.headers)
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the low_vram body is not JSON: {exc}"
            ) from None
        if not isinstance(body, dict) or set(body) != {"state"} or (
            body["state"] not in lowvram.STATES
        ):
            raise ApiError(
                400,
                "invalid_request",
                "the body is exactly {\"state\": ...} with one of "
                f"{list(lowvram.STATES)}, got {body!r}",
                {"field": "state", "choices": list(lowvram.STATES)},
            )
        state = body["state"]
        if state == lowvram.ON and not lowvram.splittable(config.backend_kind):
            raise ApiError(
                409,
                "low_vram_not_offered",
                low_vram_not_offered(config.backend_kind),
                {"field": "state"},
            )
        done = await asyncio.to_thread(lambda: set_low_vram(config, backend, state))
        changed = [f"[audio] low_vram = {state}"]
        if done.redecided.recorded.low_vram_change is not None:
            changed.append(done.redecided.recorded.low_vram_change)
        ctx.settings_history.record(
            act=act, client=client_agent(request), changed=changed
        )
        # Not adopted here: the config follower adopts the file before the next request
        # and takes up a job type the new verdict turned on, which adopting now would
        # skip. The answer is read from the file just written.
        return _document(load_config(config.home))

    @private.put("/settings/llm/concurrency")
    async def put_llm_concurrency(request: Request) -> dict[str, Any]:
        """Set how many requests one chat model runs at once on this server, with
        `{"model": id, "width": n}`, or `{"model": id, "width": null}` for what its
        manifest states. Only lower than the manifest. Read when the model loads: a
        model on the card keeps its width (`llm_concurrency[].running`) until it is
        loaded again. Answers the full settings document after the write.
        """
        act = read_act(request.headers)
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the concurrency body is not JSON: {exc}"
            ) from None
        if (
            not isinstance(body, dict)
            or set(body) != {"model", "width"}
            or not isinstance(body["model"], str)
            or not (body["width"] is None or type(body["width"]) is int)
        ):
            raise ApiError(
                400,
                "invalid_request",
                'the body is exactly {"model": "<id>", "width": <whole number> | '
                f"null}}, got {body!r}",
                {"field": "width"},
            )
        model, width = body["model"], body["width"]
        try:
            await asyncio.to_thread(
                lambda: llmconcurrency.set_concurrency(config, model, width)
            )
        except ConfigError as exc:
            named = str(exc).split(":", 1)[0]
            code = named if named.startswith("concurrency_") else "config_refused"
            raise ApiError(409, code, str(exc), {"field": "width"}) from None
        ctx.settings_history.record(
            act=act,
            client=client_agent(request),
            changed=[
                f"[llm.concurrency] {model} = "
                + ("the manifest's" if width is None else str(width))
            ],
        )
        return _document(load_config(config.home))

    @private.put("/settings/jobs/{job_type}")
    async def put_job_type(request: Request, job_type: str) -> dict[str, Any]:
        """Turn a job type on or off with `{"enabled": true | false}`, as `crucible jobs
        enable|disable` does and with the same refusals: `job_type_undecided` (no
        capability record), `job_type_cannot_hold` (the card cannot hold it) and
        `env_not_built` (install it instead). Taken up by the next request; nothing
        restarts. Answers the full settings document after the write.
        """
        act = read_act(request.headers)
        body = await _json_body(request, "job type")
        if not isinstance(body, dict) or set(body) != {"enabled"} or not isinstance(
            body["enabled"], bool
        ):
            raise ApiError(
                400,
                "invalid_request",
                f'the body is exactly {{"enabled": true | false}}, got {body!r}',
                {"field": "enabled"},
            )
        if job_type not in jobflags.JOB_TYPE_NAMES:
            raise ApiError(
                404,
                "unknown_job_type",
                f"{job_type!r} is not a job type with a [jobs] flag; they are "
                f"{list(jobflags.JOB_TYPE_NAMES)}",
                {"job_type": job_type},
            )
        on = body["enabled"]
        try:
            written = await asyncio.to_thread(
                lambda: jobflags.set_enabled(config, backend, job_type, on)
            )
        except jobflags.EnableRefused as exc:
            raise ApiError(409, exc.code, exc.sentence, {"job_type": job_type}) from None
        if written is not None:
            ctx.settings_history.record(
                act=act,
                client=client_agent(request),
                changed=[f"[jobs] {jobflags.flag(job_type)} = {str(on).lower()}"],
            )
        return _document(load_config(config.home))

    @private.put("/settings/tts/{engine}")
    async def put_tts_engine(request: Request, engine: str) -> dict[str, Any]:
        """Change `[tts.<engine>]` with an object of its keys: `memory_bytes_estimate`,
        `estimate_basis`, `estimate_note`, `max_num_seqs`, `max_num_seqs_note`,
        `mem_fraction`, `mem_fraction_note`, `context_length`, `context_length_note`
        (null removes an optional one). Checked by the config's own rules: a number
        carries its note, and a refusal is `tts_lever_invalid` naming the rule. A voice
        reads them when it loads, so a voice on the card keeps its numbers until it
        loads again (`tts_engines[].resident`). Answers the full settings document.
        """
        act = read_act(request.headers)
        body = await _json_body(request, "engine")
        if not isinstance(body, dict) or not body:
            raise ApiError(
                400,
                "invalid_request",
                f"the body is an object of [tts.{engine}] keys to set, got {body!r}",
                {"field": "body"},
            )
        try:
            await asyncio.to_thread(lambda: ttslevers.set_levers(config, engine, body))
        except ConfigError as exc:
            named, _, sentence = str(exc).partition(": ")
            raise ApiError(409 if named == ttslevers.UNSET else 400, named, sentence,
                           {"engine": engine}) from None
        ctx.settings_history.record(
            act=act,
            client=client_agent(request),
            changed=[
                f"[tts.{engine}] {key} = " + ("removed" if value is None else repr(value))
                for key, value in sorted(body.items())
            ],
        )
        return _document(load_config(config.home))

    @private.post("/settings/token/rotate")
    async def rotate_token(request: Request) -> dict[str, Any]:
        """Replace this server's bearer token with a new one, at once. Every app paired
        with the old token, and the page that asked, is refused from the next request:
        the answer is the only place the new `token` is said, with the `pairing` lines
        that carry it. The local pairing file is rewritten with it
        (`pairing_file_error` says when it could not be).
        """
        act = read_act(request.headers)
        token = mint_token()
        await asyncio.to_thread(lambda: rewrite_config(config, token=token))
        config.adopt(load_config(config.home))
        ctx.settings_history.record(
            act=act, client=client_agent(request), changed=["[auth] token rotated"]
        )
        try:
            urls = pairing.reachable_urls(
                ctx.bind_host,
                ctx.bind_port,
                config.advertise + config.tailscale_advertise + config.lan_advertise,
            )
        except InterfaceError as exc:
            urls, lines_error = [], (
                f"this server is bound to {ctx.bind_host!r} and cannot list its own "
                f"interfaces, so there are no pairing lines to show: {exc}"
            )
        else:
            lines_error = None
        return {
            "token": config.token,
            "pairing": pairing.pairing_lines(config.name, urls, config.token),
            "pairing_lines_error": lines_error,
            "pairing_file_error": _sync_pairing(config),
        }

    @private.post("/settings/upstreams/{name}/test")
    async def test_upstream(request: Request, name: str) -> dict[str, Any]:
        """List what an upstream serves, using the body's `key` or `url` when given,
        else the stored record. Never cached.
        """
        upstreamrecord.require_name(name, "the path")
        raw = await request.body()
        if raw.strip() == b"":
            probe = None
        else:
            try:
                probe = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(
                    400, "invalid_request", f"the test body is not JSON: {exc}"
                ) from None
        if probe is None or probe == {}:
            record = config.upstream(name)
            if record is None:
                raise ApiError(
                    400,
                    "upstream_unconfigured",
                    f"{name} is not configured on this server and the request "
                    f"carried no {upstreamrecord.UPSTREAM_FIELD[name]!r} to test "
                    "with. Send one to check it before saving it",
                    {
                        "field": f"upstreams.{name}."
                        f"{upstreamrecord.UPSTREAM_FIELD[name]}",
                        "upstream": name,
                    },
                )
        else:
            record = upstreamrecord.record_from_patch(
                name, probe, f"the test body for {name}"
            )
        return {"models": await upstreams.list_models(ctx.http, record)}
