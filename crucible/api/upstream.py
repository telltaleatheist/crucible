from __future__ import annotations

import json
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
    post_unless_the_caller_leaves,
    sent_across_the_wire,
    set_model_in_frame,
)

__all__ = [
    "forward_to_upstream",
    "routed_upstream_rows",
]


def routed_upstream_rows(config: Config) -> list[dict[str, Any]]:
    routed_for: dict[str, list[str]] = {}
    for entry in config.routes:
        routed_for.setdefault(entry.model, []).append(entry.capability)
    rows: list[dict[str, Any]] = []
    for name in upstreamrecord.UPSTREAM_NAMES:
        for model, classes in routed_for.items():
            if model.partition("/")[0] != name:
                continue
            rows.append(
                {
                    "id": model,
                    "object": "model",
                    "owned_by": name,
                    "upstream": name,
                    "routed_for": classes,
                }
            )
    return rows


def _upstream_unreachable(name: str, url: str, exc: Exception) -> ApiError:
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
            f"{upstreams.upstream_message(body)}. This server never retries a "
            "billed request — the caller waits",
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
            upstream = await sent_across_the_wire(
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
        upstream = await sent_across_the_wire(
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
            upstreams.AnthropicStreamTranslator(requested)
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
