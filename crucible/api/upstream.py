from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import Request, Response
from starlette.background import BackgroundTask

from .. import upstreamrecord, upstreams
from ..config import Config
from ..errors import ApiError
from ..inflight import Entry, InFlight, read_act
from ..sampling import SAMPLING_HEADER
from .context import AppContext
from .deps import error_response
from .proxy import (
    PROXY_CONNECT_TIMEOUT,
    RelayResponse,
    attempts_note,
    billed_attempts_note,
    post_unless_the_caller_leaves,
    sent_across_the_wire,
    sent_before_any_byte,
    set_model_in_frame,
)

__all__ = [
    "UPSTREAM_LIST_SECONDS",
    "UpstreamLists",
    "forward_to_upstream",
    "upstream_facts",
    "upstream_model_rows",
]


# How long a vendor's model list is served from memory before it is asked again. A
# picker reads the list every time a job modal opens (bookforge, 2026-10-25); the vendors'
# lists change on the scale of weeks.
UPSTREAM_LIST_SECONDS = 600.0


@dataclass(frozen=True)
class _Listing:
    record: upstreamrecord.UpstreamRecord
    models: tuple[str, ...]
    at: float
    wall: str


class UpstreamLists:
    """Each configured upstream's model list, asked of the vendor at most once per
    UPSTREAM_LIST_SECONDS and again at once when its key or URL changes. A vendor that
    does not answer is weather: the last list it gave is served, marked stale, with the
    reason, and an upstream that has never answered lists only what a route names."""

    def __init__(self) -> None:
        self._listings: dict[str, _Listing] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def read(
        self, client: httpx.AsyncClient, record: upstreamrecord.UpstreamRecord
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        lock = self._locks.setdefault(record.name, asyncio.Lock())
        async with lock:
            held = self._listings.get(record.name)
            if held is not None and held.record != record:
                held = None
            if held is not None and time.monotonic() - held.at < UPSTREAM_LIST_SECONDS:
                return held.models, {"listed_at": held.wall, "stale": False, "error": None}
            try:
                models = tuple(await upstreams.list_models(client, record))
            except ApiError as exc:
                error = {"code": exc.code, "message": exc.message}
                if held is None:
                    return (), {"listed_at": None, "stale": False, "error": error}
                return held.models, {"listed_at": held.wall, "stale": True, "error": error}
            wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._listings[record.name] = _Listing(record, models, time.monotonic(), wall)
            return models, {"listed_at": wall, "stale": False, "error": None}


def upstream_facts(name: str, model_id: str) -> dict[str, Any] | None:
    """What Crucible itself knows about forwarding a chat to this model, for a picker:
    the `tool_choice` values it sends on, and for Anthropic its rule row. None: Crucible
    has no rule for it and forwards the chat as sent."""
    if name == "ollama":
        return {"tool_choice": ["auto"], "reasoning_efforts": None, "sampling": None,
                "structured_output": None}
    if name != "anthropic":
        return None
    found = upstreams.anthropic_rule(model_id)
    if found is None:
        return None
    _, rule = found
    return {
        "tool_choice": ["auto", "none", "required", "function"] if rule.forced_tool
        else ["auto", "none"],
        "reasoning_efforts": list(rule.efforts),
        "sampling": rule.sampling,
        "structured_output": rule.structured_output,
    }


async def upstream_model_rows(
    config: Config, client: httpx.AsyncClient, lists: UpstreamLists
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every model each configured upstream serves, as `<upstream>/<id>`, plus every
    upstream model a route names; and, per configured upstream, when its list was read
    and why it could not be."""
    routed_for: dict[str, list[str]] = {}
    for entry in config.routes:
        routed_for.setdefault(entry.model, []).append(entry.capability)
    rows: list[dict[str, Any]] = []
    status: dict[str, dict[str, Any]] = {}
    for name in upstreamrecord.UPSTREAM_NAMES:
        record = config.upstream(name)
        served: tuple[str, ...] = ()
        if record is not None:
            served, status[name] = await lists.read(client, record)
        ids = [f"{name}/{model}" for model in served]
        ids += [m for m in routed_for if m.partition("/")[0] == name and m not in ids]
        for model in ids:
            rows.append({
                "id": model,
                "object": "model",
                "owned_by": name,
                "upstream": name,
                "routed_for": routed_for.get(model, []),
                "listed": model.partition("/")[2] in served,
                "facts": upstream_facts(name, model.partition("/")[2]),
            })
    return rows, status


# The upstreams whose chats are billed. Ollama's are not, and a request the wire lost on
# the way to it is sent again as an engine's is.
BILLED_UPSTREAMS: frozenset[str] = frozenset({"anthropic", "openai"})


def _sent(name: str) -> Callable[..., Awaitable[Any]]:
    return sent_before_any_byte if name in BILLED_UPSTREAMS else sent_across_the_wire


def _upstream_unreachable(name: str, url: str, exc: Exception) -> ApiError:
    if name in BILLED_UPSTREAMS:
        return ApiError(
            502,
            "upstream_unreachable",
            f"{name} did not answer at {url}: {type(exc).__name__}: {exc}"
            f"{billed_attempts_note(exc)}",
            {"upstream": name},
        )
    return ApiError(
        502,
        "upstream_unreachable",
        f"{name} did not answer at {url}{attempts_note(exc)}: {type(exc).__name__}: "
        f"{exc}",
        {"upstream": name},
    )


def _rate_limited(name: str, response: httpx.Response, body: bytes) -> Response:
    headers: dict[str, str] = {}
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return error_response(
        ApiError(
            429,
            "upstream_rate_limited",
            f"{name} rate-limited this request: "
            f"{upstreams.upstream_message(body)}. This server never resends a "
            "billed request once it may have reached the provider — the caller waits",
            {
                "upstream": name,
                "retry_after": retry_after,
                "upstream_status": response.status_code,
            },
        ),
        headers,
    )


def _upstream_rejected(name: str, status_code: int, body: bytes) -> ApiError:
    return ApiError(
        502,
        "upstream_rejected",
        f"{name} refused this request with {status_code}: "
        f"{upstreams.upstream_message(body)}",
        {"upstream": name, "upstream_status": status_code},
    )


def _set_model_in_response(raw: bytes, model_id: str) -> bytes:
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw
    if not isinstance(body, dict):
        return raw
    body["model"] = model_id
    return json.dumps(body).encode("utf-8")


async def forward_to_upstream(
    ctx: AppContext,
    request: Request,
    requested: str,
    body: dict[str, Any],
    *,
    client_agent: str | None,
) -> Response:
    name, model_id = upstreamrecord.require_upstream_model(requested)
    record = ctx.config.upstream(name)
    if record is None:
        raise ApiError(
            409,
            "upstream_unconfigured",
            f"{requested!r} names the {name} upstream and this server has no "
            f"{upstreamrecord.UPSTREAM_FIELD[name]} for it. Configure it with "
            f"`PUT /v1/settings` — nothing here falls back to a local model, "
            "because a route is a decision somebody made and not a guess this "
            "server gets to improvise",
            {"upstream": name, "model": requested},
        )

    act = read_act(request.headers)
    client = ctx.http
    if name == "ollama":
        forwarded = await upstreams.forward_ollama(
            client, record, model_id, body, ctx.ollama_contexts
        )
    else:
        forwarded = upstreams.forward_body(name, model_id, body)
    sampling_headers = {
        SAMPLING_HEADER: json.dumps(
            forwarded.sources, separators=(",", ":"), sort_keys=True
        ),
        upstreams.CONTEXT_HEADER: json.dumps(
            forwarded.context, separators=(",", ":"), sort_keys=True
        ),
    }
    url = upstreams.chat_url(record)
    headers = upstreams.chat_headers(record)
    inflight = ctx.inflight
    entry = inflight.open(act=act, model=requested, client=client_agent)
    try:
        if body.get("stream") is True:
            return await _stream_from_upstream(
                client,
                record,
                url,
                headers,
                forwarded,
                requested,
                sampling_headers,
                when_relayed=_close_inflight(inflight, entry),
            )
        try:
            upstream = await _sent(name)(
                lambda: post_unless_the_caller_leaves(
                    client, url, forwarded.body, request, headers
                ),
                where=f"{name} at {url}",
            )
        except httpx.HTTPError as exc:
            raise _upstream_unreachable(name, url, exc) from None
        if upstream is None:
            return error_response(
                ApiError(
                    499,
                    "client_disconnected",
                    f"the caller closed the connection before {name} answered; "
                    "the upstream request was cancelled with it",
                )
            )
        payload = upstream.content
        if upstream.status_code == 429:
            return _rate_limited(name, upstream, payload)
        if upstream.status_code != 200:
            raise _upstream_rejected(name, upstream.status_code, payload)
        if forwarded.dialect != upstreams.DIALECT_OPENAI:
            try:
                document = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(
                    502,
                    "upstream_rejected",
                    f"{name} answered 200 with a body that is not JSON: {exc}",
                    {"upstream": name, "upstream_status": 200},
                ) from None
            translated = (
                upstreams.anthropic_to_openai(document, requested)
                if forwarded.dialect == upstreams.DIALECT_ANTHROPIC
                else upstreams.ollama_to_openai(document, requested)
            )
            content = json.dumps(translated).encode("utf-8")
        else:
            content = _set_model_in_response(payload, requested)
        return Response(
            content=content,
            status_code=200,
            media_type="application/json",
            headers=sampling_headers,
        )
    except BaseException:
        inflight.close(entry)
        raise
    finally:
        if body.get("stream") is not True:
            inflight.close(entry)


def _close_inflight(inflight: InFlight, entry: Entry) -> Callable[[], Awaitable[None]]:
    async def done() -> None:
        inflight.close(entry)

    return done


async def _stream_from_upstream(
    client: httpx.AsyncClient,
    record: upstreamrecord.UpstreamRecord,
    url: str,
    headers: dict[str, str],
    forwarded: upstreams.Forwarded,
    requested: str,
    extra_headers: dict[str, str],
    when_relayed: Callable[[], Awaitable[None]],
) -> Response:
    name = record.name
    upstream_request = client.build_request(
        "POST",
        url,
        content=forwarded.body,
        headers=headers,
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT, read=None, write=60.0, pool=10.0
        ),
    )
    try:
        upstream = await _sent(name)(
            lambda: client.send(upstream_request, stream=True),
            where=f"{name} at {url}",
        )
    except httpx.HTTPError as exc:
        await when_relayed()
        raise _upstream_unreachable(name, url, exc) from None

    if upstream.status_code != 200:
        payload = await upstream.aread()
        await upstream.aclose()
        if upstream.status_code == 429:
            response = _rate_limited(name, upstream, payload)
            response.background = BackgroundTask(when_relayed)
            return response
        await when_relayed()
        raise _upstream_rejected(name, upstream.status_code, payload)

    if forwarded.dialect != upstreams.DIALECT_OPENAI:
        translator: (
            upstreams.AnthropicStreamTranslator | upstreams.OllamaStreamTranslator
        ) = (
            upstreams.AnthropicStreamTranslator(
                requested, include_usage=forwarded.include_usage
            )
            if forwarded.dialect == upstreams.DIALECT_ANTHROPIC
            else upstreams.OllamaStreamTranslator(
                requested, include_usage=forwarded.include_usage
            )
        )

        async def relay() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_bytes():
                for frame in translator.feed(chunk):
                    yield frame
            for frame in translator.finish():
                yield frame

    else:

        async def relay() -> AsyncIterator[bytes]:
            buffer = b""
            async for chunk in upstream.aiter_bytes():
                buffer += chunk
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    yield set_model_in_frame(frame, requested) + b"\n\n"
            if buffer:
                yield set_model_in_frame(buffer, requested)

    return RelayResponse(
        relay(),
        upstream=upstream,
        when_relayed=when_relayed,
        status_code=200,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            **extra_headers,
        },
    )
