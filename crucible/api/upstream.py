from __future__ import annotations

import json
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from .. import upstreams
from ..config import Config
from ..errors import ApiError
from ..inflight import Entry, InFlight, read_act
from ..sampling import SAMPLING_HEADER
from .proxy import (
    PROXY_CONNECT_TIMEOUT,
    _RelayResponse,
    _attempts,
    _post_unless_the_caller_leaves,
    _sent_across_the_wire,
    _set_model_in_frame,
)

# --------------------------------------------------- forwarding to an upstream
#
# PHASE15-HOST.md section 3.4. Owen, 2026-09-14: *"they dont have ollama
# fallbacks or cloud anything at all … one contract, one SDK, one API, one
# communication method."* The provider code left BookForge and Foundry; this is
# where it landed, and `crucible/upstreams.py` holds everything that differs
# between the three.
#
# WHAT THIS PATH DELIBERATELY DOES NOT DO. No lease, no lane, no settlement:
# nothing was on the card, so there is nothing to ask about when the answer is
# written. It DOES open an `inflight` record, because `/v1/activity` must be
# able to say "translating on anthropic" — a chat that is invisible is the
# defect `crucible/inflight.py` exists to have fixed, and where it runs does
# not change that.
#
# AND IT NEVER RETRIES. A `429` comes back to the caller with the upstream's own
# `Retry-After` and the CALLER waits. A request that reached the upstream may
# already be billed, and a server that quietly sent it twice would be spending
# somebody's money to smooth a graph.


def _refuse_lease_on_an_upstream(subject_id: str) -> None:
    """`lease_not_needed` — there is nothing on the card to hold in place.

    PHASE15-HOST.md section 3.4. A lease is the promise not to MOVE what is
    resident, and an upstream model is never resident: no load, no eviction,
    nothing another job could take away. A server that accepted the lease would
    be issuing a promise about a card the work never touches, and the client
    that took it would hold the one lease this server has, refusing everybody
    else's `load-model` for a run that is happening in somebody else's
    datacentre.

    Both doors that can name a model call this — `POST /v1/models/{id}/lease`
    and a `load-model` job — because both mistakes come from the same wrong
    belief and a client owed one sentence should not get two.
    """
    if upstreams.split_model(subject_id) is None:
        return
    raise ApiError(
        409,
        "lease_not_needed",
        "an upstream model is never resident; send the chat",
        {"model": subject_id},
    )


def _routed_upstream_rows(config: Config) -> list[dict[str, Any]]:
    """Every distinct upstream model a route names, in `UPSTREAM_NAMES` order.

    DISTINCT, because `translate` and `simplify` routed to the same Anthropic
    model are one thing a client may send and two classes that send it;
    `routed_for` is where the second fact goes. Two identical rows would make a
    client's model picker show the same entry twice.

    No `created`, no `max_model_len`, no `defaults`: this server did not load
    it, cannot see its context window and has no manifest for it. Absent is the
    honest value, and a client that sizes `max_tokens` against `max_model_len`
    already skips the clamp when the field is missing (CLIENT-SURFACES.md 6.1).
    """
    routed_for: dict[str, list[str]] = {}
    for entry in config.routes:
        routed_for.setdefault(entry.model, []).append(entry.capability)
    rows: list[dict[str, Any]] = []
    for name in upstreams.UPSTREAM_NAMES:
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
        f"{name} did not answer at {url}{_attempts(exc)}: {type(exc).__name__}: "
        f"{exc}",
        {"upstream": name},
    )


def _rate_limited(name: str, response: httpx.Response, body: bytes) -> Response:
    """The upstream's own 429, passed through with its own `Retry-After`.

    Passed through rather than translated, because the only correct response to
    a rate limit is for the thing that decided to make the request to decide
    when to make it again. `Retry-After` is the upstream's number and this
    server has no better one; it is copied when it is there and absent when it
    is not, never invented.
    """
    headers: dict[str, str] = {}
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return JSONResponse(
        status_code=429,
        headers=headers,
        content=ApiError(
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
        ).body(),
    )


def _upstream_rejected(name: str, status_code: int, body: bytes) -> ApiError:
    """Anything the upstream refused, with the upstream's own words.

    ONE name for every non-200 that is not a 429, and a `502` for all of them.
    Section 3.4 spells out the `401` case; the rest — a model id the account
    cannot reach, an overloaded region, a body the provider did not like — are
    the same event from this server's side: *the hop failed and the upstream
    said why*. `details.upstream_status` carries the real number, so nothing is
    lost by not multiplying the names, and a client is never left reading a
    Crucible refusal that sounds like Crucible's own fault without the
    provider's sentence beside it.
    """
    return ApiError(
        502,
        "upstream_rejected",
        f"{name} refused this request with {status_code}: "
        f"{upstreams.upstream_message(body)}",
        {"upstream": name, "upstream_status": status_code},
    )


def _set_model_in_response(raw: bytes, model_id: str) -> bytes:
    """Name the id the CALLER asked for in a completion the upstream named its own.

    Anthropic and OpenAI both echo the bare model name they were sent, and the
    caller sent `<upstream>/<model>`. Crucible's id is the contract in both
    directions (`_restore_model_id`'s rule, one hop further out).
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw
    if not isinstance(body, dict):
        return raw
    body["model"] = model_id
    return json.dumps(body).encode("utf-8")


async def _forward_to_upstream(
    request: Request,
    requested: str,
    body: dict[str, Any],
    *,
    client_agent: str | None,
) -> Response:
    """One chat, sent to the upstream the operator configured."""
    live: Config = request.app.state.config
    name, model_id = upstreams.require_upstream_model(requested)
    record = live.upstream(name)
    if record is None:
        # 409 and not 404: the model id is well formed and this server knows
        # the upstream — what is missing is the operator's key, which is a
        # state of the server rather than a mistake in the request. The app
        # that reads this shows its settings window, which is 5.2's whole
        # point.
        raise ApiError(
            409,
            "upstream_unconfigured",
            f"{requested!r} names the {name} upstream and this server has no "
            f"{upstreams.UPSTREAM_FIELD[name]} for it. Configure it with "
            f"`PUT /v1/settings` — nothing here falls back to a local model, "
            "because a route is a decision somebody made and not a guess this "
            "server gets to improvise",
            {"upstream": name, "model": requested},
        )

    # Before the work, so an unknown act is a 400 rather than a BILLED
    # completion reported under a name nobody knows — and before Ollama's
    # context lookup, which is a round trip a refused request should not cost.
    act = read_act(request.headers)
    client: httpx.AsyncClient = request.app.state.http
    if name == "ollama":
        # Section 3.4a: Ollama is spoken natively and its body cannot be built
        # without the `num_ctx` it runs at, which may have to be asked for.
        forwarded = await upstreams.forward_ollama(
            client, record, model_id, body, request.app.state.ollama_contexts
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
    inflight: InFlight = request.app.state.inflight
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
            upstream = await _sent_across_the_wire(
                lambda: _post_unless_the_caller_leaves(
                    client, url, forwarded.body, request, headers
                ),
                where=f"{name} at {url}",
            )
        except httpx.HTTPError as exc:
            raise _upstream_unreachable(name, url, exc) from None
        if upstream is None:
            # The caller hung up. Nothing reads this; it exists so the handler
            # returns something that is not shaped like a completion.
            return JSONResponse(
                status_code=499,
                content=ApiError(
                    499,
                    "client_disconnected",
                    f"the caller closed the connection before {name} answered; "
                    "the upstream request was cancelled with it",
                ).body(),
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
        # Only the non-streamed paths reach here with the record still open;
        # the streamed one hands its close to the relay's end and returns
        # above. `close` is idempotent (crucible/inflight.py), so the two
        # cannot double-count and neither can leave a row behind.
        if body.get("stream") is not True:
            inflight.close(entry)


def _close_inflight(inflight: InFlight, entry: Entry) -> Callable[[], Awaitable[None]]:
    """The end of a streamed upstream completion. No settlement: no card moved."""

    async def done() -> None:
        inflight.close(entry)

    return done


async def _stream_from_upstream(
    client: httpx.AsyncClient,
    record: upstreams.UpstreamRecord,
    url: str,
    headers: dict[str, str],
    forwarded: upstreams.Forwarded,
    requested: str,
    extra_headers: dict[str, str],
    when_relayed: Callable[[], Awaitable[None]],
) -> Response:
    """A streamed completion from an upstream, in OpenAI chunk shape.

    The upstream response is opened before anything is returned, so a refusal
    comes back with its own status and body rather than as a 200 whose stream
    turns out to be an error — the same rule `_proxy_stream` follows for the
    local engine.

    Anthropic's SSE and Ollama's NDJSON are different protocols and are
    TRANSLATED (`upstreams.AnthropicStreamTranslator`,
    `upstreams.OllamaStreamTranslator`); OpenAI's is relayed with one
    substitution, the `model` the caller asked for.
    """
    name = record.name
    upstream_request = client.build_request(
        "POST",
        url,
        content=forwarded.body,
        headers=headers,
        # No read deadline, for `_proxy_stream`'s reason: a completion emits a
        # token at a time and may think for a long while before the first one.
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT, read=None, write=60.0, pool=10.0
        ),
    )
    try:
        # `_proxy_stream`'s reason: before the first byte, a retry costs nothing.
        upstream = await _sent_across_the_wire(
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
            # SSE frames end at a blank line, so the relay holds a partial
            # frame until it has one. Whatever is left when the upstream stops
            # is forwarded as it stands rather than swallowed.
            buffer = b""
            async for chunk in upstream.aiter_bytes():
                buffer += chunk
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    yield _set_model_in_frame(frame, requested) + b"\n\n"
            if buffer:
                yield _set_model_in_frame(buffer, requested)

    return _RelayResponse(
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
