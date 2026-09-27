from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from ..engines import chat_admission
from ..errors import ApiError
from ..inflight import Entry, InFlight
from ..residency import Residency
from ..sampling import Applied
from ..settle import Settlement

#: The proxy waits on the engine, not on a clock it invented. A streamed
#: completion has no read timeout at all (the engine emits a token at a time and
#: may think for a while before the first one); a non-streamed one gets a long
#: but finite ceiling so a wedged engine surfaces as an error rather than a hang.
PROXY_CONNECT_TIMEOUT = 10.0
PROXY_READ_TIMEOUT = 900.0

#: How long the proxy keeps an idle socket to an engine before letting it go.
#: BELOW the engine's own keep-alive, and the gap is the whole point: vLLM's
#: uvicorn closes an idle connection after `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` = 5 s
#: (vllm 0.29.0, vllm/envs.py:109), and httpx's pool keeps one for the same
#: 5.0 s by default (`httpx.Limits().keepalive_expiry`), so at that boundary the
#: proxy could hand a request to a socket the engine had just closed. On
#: 2026-09-21 22:03:20 it did: one `ReadError` with an empty message from an
#: engine that answered nine other completions in the same window, a 502 the
#: client took as fatal, and a book's page read dropped its lease over it. At
#: two seconds no pooled socket is ever older than the engine's patience.
PROXY_KEEPALIVE_EXPIRY = 2.0

#: Transport faults that mean the request was LOST ON THE WAY, not answered and
#: not refused: a reset, a peer that closed a keep-alive socket under us, a
#: half-written request. That is weather, not misconfiguration (CLAUDE.md,
#: "harden transients"), so the proxy sends the request once more on a fresh
#: socket — httpx discards the connection a network error happened on, and
#: `PROXY_KEEPALIVE_EXPIRY` keeps the pool clear of the next stale one. Timeouts
#: are NOT in this set on purpose: an engine that has not answered inside
#: `PROXY_READ_TIMEOUT` is wedged, and repeating that would be a second 900 s.
LOST_ON_THE_WIRE: tuple[type[Exception], ...] = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
#: The whole budget, stated: the first attempt and one more. Not a loop.
WIRE_ATTEMPTS = 2

#: The proxy sends the client's own bytes, so it declares the type itself rather
#: than letting httpx serialise a document and label it.
JSON_HEADERS = {"Content-Type": "application/json"}


def _chat_body(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(
            400, "invalid_request", f"the chat request body is not JSON: {exc}"
        ) from None
    if not isinstance(body, dict):
        raise ApiError(
            400,
            "invalid_request",
            f"the chat request body must be a JSON object, got "
            f"{type(body).__name__}",
        )
    return body


def _forward_body(raw: bytes, applied: Applied, resident: Any) -> bytes:
    """The client's chat body on its way to the engine.

    The proxy owns exactly one field (PHASE2-LLM.md section 5), so where the
    engine already answers to the Crucible id — vLLM, which takes
    `--served-model-name` — there is nothing to substitute and the bytes the
    client sent are the bytes the engine reads. That is worth more than
    tidiness: `response_format.json_schema.schema` is a grammar Foundry hands to
    the guided-decoding backend (CLIENT-SURFACES.md section 6.2), and
    re-encoding somebody else's grammar on the way past is not the proxy's job.

    Where the two names differ — mlx-lm has no `--served-model-name` and answers
    to the resolved weights directory — one field has to change, so the document
    is re-serialised with `model` replaced in the position it already held.
    Nothing else is added, removed or reordered.

    Since phase 2's section 9 there is a second reason to re-serialise: a
    manifest default the request left room for. It is held to the same rule —
    **the bytes pass through untouched unless something actually changed**, so a
    request that states every knob this server knows about reaches the engine
    exactly as it was written, grammar and all.
    """
    if not applied.changed and resident.engine_model_name == resident.model_id:
        return raw
    document = dict(applied.body)
    if resident.engine_model_name != resident.model_id:
        document["model"] = resident.engine_model_name
    return json.dumps(document).encode("utf-8")


async def _watch_for_disconnect(request: Request) -> None:
    """Return the moment the caller's connection has gone away.

    A WAIT ON `receive`, NOT A POLL OF `is_disconnected()` (2026-09-24). The
    poll asks `receive` inside an already-cancelled scope and takes whatever is
    there; how that answer survives the trip depends on every layer between the
    server and the route, and one layer (`BaseHTTPMiddleware`, from 1.0.18) lost
    it every time — a caller that had hung up read as present until the engine
    answered. A blocked `receive` is the server's own statement: uvicorn wakes
    it on `connection_lost` and it returns `http.disconnect`. That is the
    earliest anyone on this side can know, with no clock to choose.

    Only for a route that has read its whole body — both callers have (the
    chat door reads `request.body()`, the decision door's body is a parsed
    parameter) — so what `receive` has left to say is the disconnect. An
    `http.request` that arrives anyway is an empty tail and is passed over;
    anything else is a fault in the server and is raised, not waited past.
    Cancelling this while it waits consumes nothing: the next reader of
    `receive` finds the channel as it was.
    """
    while True:
        message = await request.receive()
        kind = message.get("type")
        if kind == "http.disconnect":
            return
        if kind != "http.request":
            raise RuntimeError(
                f"the caller's channel said {kind!r} while its request was being "
                "answered; only `http.request` or `http.disconnect` can come there"
            )


async def _post_unless_the_caller_leaves(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    request: Request,
    headers: dict[str, str],
) -> httpx.Response | None:
    """The upstream POST, raced against the caller hanging up.

    A bare `await client.post(...)` is not enough, and the gap is not cosmetic:
    nothing inside it watches the caller's own socket, so somebody who gives up
    after five seconds leaves the engine generating to the end of its token
    budget with no one to hand the answer to. Crucible runs one job at a time
    (DESIGN.md section 6), so that is not wasted time in the abstract — it is the
    next job's time. Dropping the connection is also the *only* cancel either app
    has for a chat (CLIENT-SURFACES.md, closing section), which makes this the
    cancel path rather than a refinement of one.

    Returns the engine's response, or None when the caller went first. In that
    case the upstream request has already been cancelled, and cancelling it is
    what closes the socket the engine is writing to — which is how the engine
    learns to stop.
    """
    return await _unless_the_caller_leaves(
        client.post(url, content=body, headers=headers), request
    )


async def _sent_across_the_wire(
    attempt: Callable[[], Awaitable[Any]], *, where: str
) -> Any:
    """`attempt()`, sent once more on a fresh socket if the wire loses it.

    `LOST_ON_THE_WIRE` says which faults qualify and why; `WIRE_ATTEMPTS` is the
    entire budget. Every loss is said in the server log by name, first attempt
    or last, so a socket that keeps dying is visible there rather than folded
    into a success. A caller that hangs up during the first attempt is not a
    loss: `_post_unless_the_caller_leaves` returns None for that, and None is
    returned, never repeated.
    """
    attempt_number = 0
    while True:
        attempt_number += 1
        try:
            return await attempt()
        except LOST_ON_THE_WIRE as exc:
            detail = type(exc).__name__ + (f": {exc}" if str(exc) else "")
            if attempt_number >= WIRE_ATTEMPTS:
                print(
                    f"crucible: lost on the wire to {where} ({detail}), attempt "
                    f"{attempt_number} of {WIRE_ATTEMPTS}; giving up",
                    file=sys.stderr,
                )
                raise
            print(
                f"crucible: lost on the wire to {where} ({detail}), attempt "
                f"{attempt_number} of {WIRE_ATTEMPTS}; sending it again on a "
                "fresh socket",
                file=sys.stderr,
            )


def _attempts(exc: Exception) -> str:
    """How many times the wire was tried, for the 502 that ends it.

    A fault in `LOST_ON_THE_WIRE` only reaches a 502 after the whole budget;
    anything else (a timeout) was tried once, and saying "on 2 attempts" of it
    would be a lie.
    """
    if isinstance(exc, LOST_ON_THE_WIRE):
        return f" on {WIRE_ATTEMPTS} attempts"
    return ""


def _caller_gone(resident: Any) -> JSONResponse:
    """What the proxy answers a caller who is no longer there to read it.

    Nothing reads this: the socket it would travel down is closed. It exists
    because the handler still has to return something, and returning a body
    shaped like a completion would be a lie told to the log. 499 is nginx's code
    for a client that closed the request, and Crucible borrows it rather than
    inventing one.
    """
    return JSONResponse(
        status_code=499,
        content=ApiError(
            499,
            "client_disconnected",
            f"the caller closed the connection before the engine serving "
            f"{resident.model_id!r} answered; the engine's request was cancelled "
            "with it",
        ).body(),
    )


def _refuse_an_exited_engine(residency: Residency, resident: Any) -> None:
    """502 `engine_exited`: the resident model's engine is no longer running.

    Asked before anything is sent, so a request to a dead engine is answered at
    once and by name instead of by a connect retry that ends in
    `engine_unreachable`. A chat already in flight when the engine exits is
    failed by the dropped connection itself. ContentStudio's 2026-09-25 hangs
    were an engine that did NOT exit (mlx-lm's generation thread died and the
    process stayed up), which the `mlx-lm-fatal-generation-thread` env patch
    turns into this.
    """
    code = residency.engine_exit_code
    if code is None:
        return
    raise ApiError(
        502,
        "engine_exited",
        f"the engine serving {resident.model_id!r} exited with code {code} and is "
        f"not serving. Its log is {resident.log_path}. Unload the model and load "
        "it again",
    )


def _model_not_resident(requested: str, resident: Any, answering: str) -> ApiError:
    """409: the named model is not the one on the card, and none will be loaded.

    One body for the chat door and the decision door (PHASE22 section 2.1: "the
    same body the chat door gives"), with only the noun changed.
    """
    return ApiError(
        409,
        "model_not_resident",
        f"{requested!r} is not resident on this server; "
        + (f"{resident.model_id!r} is. " if resident is not None else "no model is. ")
        + f"Crucible never loads a model to answer {answering} — submit "
        'a {"type": "load-model"} job first.',
        {"requested": requested, "resident": None if resident is None
         else resident.model_id},
    )


async def _unless_the_caller_leaves(
    work: Awaitable[Any], request: Request
) -> Any:
    """`work`, raced against the caller hanging up; None if the caller went first.

    THE ONE OWNER of "the caller left, so the engine stops" for every door that
    answers once rather than streaming: the chat door's POST
    (`_post_unless_the_caller_leaves`) and the whole of a decision. A decision
    is a prime and up to sixteen questions, and ONE watcher cancelling the
    whole of it is what closes every in-flight socket AND takes back every
    question still waiting at its gate — so nothing more is sent to an engine
    nobody is listening to. Sixteen watchers on one ASGI `receive` would be
    sixteen readers of one channel.

    When the caller goes first, `work` is cancelled and AWAITED before this
    returns, so the cancellation has reached httpx and closed the sockets by
    the time the door closes its `InFlight` row and asks the settlement about
    the card. The door is cancelled itself (the server shutting down): the work
    goes with it.
    """
    task = asyncio.ensure_future(work)
    watch = asyncio.create_task(_watch_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        raise
    finally:
        watch.cancel()
    if task in done:
        return task.result()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        # Ours: cancelled two lines above, and awaited so the cancellation
        # reaches httpx and closes the sockets. Letting it propagate would
        # report the caller's own departure as this request being cancelled.
        pass
    except Exception as exc:
        # The work failed in the same instant the caller left. There is nobody
        # to answer, so it is said where a person can find it and the caller's
        # departure is what this returns.
        print(
            f"crucible: work for a caller who left failed as it went: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    # The watcher finished, which is the caller leaving — or the watcher's own
    # fault, raised here now that the work is down rather than lost with it.
    watch.result()
    return None


def _engine_unreachable(resident: Any, exc: Exception) -> ApiError:
    return ApiError(
        502,
        "engine_unreachable",
        f"the engine serving {resident.model_id!r} at {resident.base_url} did not "
        f"answer{_attempts(exc)}: {type(exc).__name__}: {exc}. Its log is "
        f"{resident.log_path}",
    )


def _restore_model_id(raw: bytes, resident: Any) -> bytes:
    """Put Crucible's id back where the proxy substituted the engine's name.

    The request's `model` is rewritten on the way in to the name the engine
    answers to, because mlx-lm has no `--served-model-name` and answers to the
    resolved weights directory (`crucible/engines/mlx_lm.py`). OpenAI engines
    echo the name they were asked for, so without this the completion would come
    back naming a directory on the server's disk — and would name the Crucible id
    on vLLM, which *does* take a served name, so the answer to "what did I just
    talk to" would depend on the backend. Crucible's id is the contract in both
    directions; this is the second half of the one substitution the proxy makes,
    and nothing else in the body is touched.

    A backend whose engine already answers to the id (vLLM) is relayed byte for
    byte, as before.
    """
    if resident.engine_model_name == resident.model_id:
        return raw
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(
            502,
            "engine_response_unreadable",
            f"the engine serving {resident.model_id!r} answered 200 with a body "
            f"that is not JSON: {exc}. Its log is {resident.log_path}",
        ) from None
    if not isinstance(body, dict):
        raise ApiError(
            502,
            "engine_response_unreadable",
            f"the engine serving {resident.model_id!r} answered 200 with a JSON "
            f"{type(body).__name__}, not a completion object. Its log is "
            f"{resident.log_path}",
        )
    body["model"] = resident.model_id
    return json.dumps(body).encode("utf-8")


def _restore_model_id_in_frame(frame: bytes, resident: Any) -> bytes:
    """The same substitution inside one SSE frame of a streamed completion."""
    return _set_model_in_frame(frame, resident.model_id)


def _set_model_in_frame(frame: bytes, model_id: str) -> bytes:
    """Name `model_id` in every JSON `data:` chunk of one SSE frame.

    Only `data:` lines carrying a JSON chunk are touched, and only their `model`
    field. `data: [DONE]`, comments, and any line the engine frames some other
    way are passed through as they arrived: mid-stream there is no way to raise,
    and a frame Crucible does not recognise is the engine's to explain.

    Two callers, one substitution: the local proxy puts Crucible's id back where
    it wrote the engine's, and the upstream proxy puts `<upstream>/<model>` back
    where it wrote the bare id. Same rule — **the id the caller asked for is the
    id the answer names** — so it is one function rather than two that could
    come to frame SSE differently.
    """
    lines = frame.split(b"\n")
    changed = False
    for index, line in enumerate(lines):
        if not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: "):]
        if payload.strip() == b"[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(chunk, dict) or "model" not in chunk:
            continue
        chunk["model"] = model_id
        lines[index] = b"data: " + json.dumps(chunk).encode("utf-8")
        changed = True
    return b"\n".join(lines) if changed else frame


def _chat_over(settlement: Settlement) -> Callable[[], Awaitable[None]]:
    """OWEN'S RULING, 2026-09-14: the chat is over, so who still has the card?

    A chat holds nothing and reserves nothing, which is right while it runs and
    is exactly why its END is worth asking at: a client that did not lease has
    now said everything it is going to say, and if the lane, the lease and the
    claim are all clear the card goes. **A run of chats with no lease therefore
    reloads its model between requests**, which is the bill for not stating an
    intention rather than a bug in the rule (crucible/settle.py).

    A CLEANUP FAILURE IS NOT AN OPERATION FAILURE. An engine that will not stop
    is said, loudly, in the server log — it does not turn a completion that
    arrived into a 500, and it does not break a stream that had already been
    delivered.
    """

    async def over() -> None:
        await asyncio.to_thread(
            settlement.settle_quietly, "the last chat completion finished"
        )

    return over


def _after_the_stream(
    inflight: InFlight, entry: Entry, chat_over: Callable[[], Awaitable[None]]
) -> Callable[[], Awaitable[None]]:
    """Close the record and ask about the card, once the relay is really done."""

    async def done() -> None:
        inflight.close(entry)
        await chat_over()

    return done


class _RelayResponse(StreamingResponse):
    """A streamed relay whose upstream is closed however the relay ends.

    A caller who drops a streamed completion has to reach the engine, or it goes
    on producing tokens for nobody — and on one exclusive lane that is the next
    job's time. What closes the engine's end is closing the upstream response, so
    the only question is who is certain to do it.

    Not the relay generator, is the answer. Measured 2026-09-13 against uvicorn
    0.52 and starlette 1.6: uvicorn advertises ASGI `spec_version` 2.3, so
    `StreamingResponse.__call__` takes its task-group branch, a disconnect
    cancels the task pulling from the generator, the `CancelledError` lands in
    the generator's frame and a `finally` there would run. Read the 2.4 branch of
    that same function, though, and the disconnect arrives as an `OSError` out of
    `send` — raised *outside* the generator, which is then left suspended at its
    `yield` until asyncio's async-generator finalizer gets to it, whenever that
    is. Same proxy, same engine, two different answers depending on which branch
    Starlette takes.

    Which is not a thing this proxy should depend on. The upstream's lifetime
    belongs to the response that owns it, not to the generator that happens to be
    reading from it, and `__call__`'s `finally` runs on every one of those paths.
    """

    def __init__(
        self,
        *args: Any,
        upstream: httpx.Response,
        when_relayed: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._upstream = upstream
        # THE END OF A STREAMED COMPLETION, for the same reason the upstream's
        # close lives here rather than in the generator: this `finally` is the
        # one place that runs on every path Starlette can take. It is where the
        # chat stops being in flight and where the card is asked about.
        self._when_relayed = when_relayed

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._upstream.aclose()
            await self._when_relayed()


async def _proxy_stream(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    resident: Any,
    extra_headers: dict[str, str],
    when_relayed: Callable[[], Awaitable[None]],
) -> Response:
    """Forward a streamed completion, SSE framing intact.

    The upstream response is opened before anything is returned, so a refusal
    from the engine comes back with the engine's own status code and body rather
    than as a 200 whose stream turns out to be an error.

    The bytes are relayed unchanged except for the one field the proxy
    substituted on the way in (see `_restore_model_id`); on a backend whose
    engine answers to the Crucible id there is nothing to undo and the relay is
    byte for byte.
    """
    log_path = resident.log_path
    request = client.build_request(
        "POST",
        url,
        content=body,
        headers=JSON_HEADERS,
        # A streamed completion emits a token at a time and may think for a long
        # while before the first one; there is no honest read deadline here.
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT, read=None, write=60.0, pool=10.0
        ),
    )
    try:
        # Opening the stream is the only moment a retry is honest here: nothing
        # has been relayed yet. A socket that dies mid-stream is the client's
        # to notice, because half of an answer has already gone out.
        upstream = await _sent_across_the_wire(
            lambda: client.send(request, stream=True),
            where=f"the engine serving {resident.model_id!r}",
        )
    except httpx.HTTPError as exc:
        raise ApiError(
            502,
            "engine_unreachable",
            f"the resident engine at {url} did not answer{_attempts(exc)}: "
            f"{type(exc).__name__}: {exc}. Its log is {log_path}",
        ) from None

    if upstream.status_code != 200:
        payload = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=payload,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers=dict(extra_headers),
            # There is no relay on this path, so the end of the completion is
            # here: the record closes and the card is asked about, after the
            # engine's own refusal has been written out.
            background=BackgroundTask(when_relayed),
        )

    async def relay() -> AsyncIterator[bytes]:
        if resident.engine_model_name == resident.model_id:
            async for chunk in upstream.aiter_bytes():
                yield chunk
            return
        # SSE frames end at a blank line, so the relay holds a partial frame
        # until it has one. Whatever is left when the engine stops is
        # forwarded as it stands rather than swallowed.
        buffer = b""
        async for chunk in upstream.aiter_bytes():
            buffer += chunk
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                yield _restore_model_id_in_frame(frame, resident) + b"\n\n"
        if buffer:
            yield _restore_model_id_in_frame(buffer, resident)

    return _RelayResponse(
        relay(),
        upstream=upstream,
        when_relayed=when_relayed,
        status_code=200,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        # The sampling audit rides on the response headers, which is the one
        # place a streamed completion HAS to put it: there is nowhere in an SSE
        # body to add a field a client would not have to learn to skip.
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            **extra_headers,
        },
    )


def _chat_limit_of(residency: Residency) -> tuple[int | None, str | None]:
    """The chat door's admission limit for whatever model is resident, and why.

    `(None, None)` when no model is resident: the limit is a property of the
    ENGINE, and with nothing loaded there is no engine to ask. That is not the
    same as "unlimited", and `/v1/activity` reports it as null rather than as a
    number, so a client reading the field cannot mistake an empty card for a
    door that will take anything.
    """
    resident = residency.resident_model
    if resident is None:
        return (None, None)
    return chat_admission(resident.engine, resident.engine_args)


def _chat_queue_full(
    *,
    resident: Any,
    limit: int,
    basis: str | None,
    wait: int | None,
) -> Response:
    """503: this engine already has everything it can run, and it will not queue.

    A REFUSAL RATHER THAN A HELD SOCKET, which is the whole point. The failure
    this replaces looked like a healthy server: the request was accepted, the
    connection stayed open, nothing was generated, and the client found out at
    its own deadline. A caller that is told "full, try in 12 seconds" can pace
    itself; a caller holding an accepted socket cannot.

    503 and not 429: nothing here is a rate limit or a quota. The engine is
    genuinely at capacity for a moment, which is what 503 means, and it is the
    status a client is most likely to already treat as "wait and retry".

    A `JSONResponse` rather than a raised `ApiError` for one reason: `Retry-After`
    is a HEADER, and `ApiError` carries a body. `_rate_limited` below does the
    same for the same reason, and states the rule both follow — a `Retry-After`
    is real or it is absent, never invented. Here it is the median of what
    completions on this engine have recently taken, and a server that has
    finished none omits the header.
    """
    error = ApiError(
        503,
        "chat_queue_full",
        f"this server already has {limit} chat completion(s) open on "
        f"{resident.model_id!r} and its {resident.engine} engine will not queue "
        "another: "
        + (basis or "no basis stated")
        + ". Nothing was sent to the engine, so this request cost nothing and "
        "can be made again"
        + ("" if wait is None else f"; about {wait}s is what completions on this "
           "engine have recently been taking"),
        {
            "model": resident.model_id,
            "engine": resident.engine,
            "max_in_flight": limit,
            "max_in_flight_basis": basis,
            "retry_after": wait,
        },
    )
    headers = {} if wait is None else {"Retry-After": str(wait)}
    return JSONResponse(status_code=503, headers=headers, content=error.body())
