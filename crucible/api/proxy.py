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

PROXY_CONNECT_TIMEOUT = 10.0
PROXY_READ_TIMEOUT = 900.0

PROXY_KEEPALIVE_EXPIRY = 2.0

LOST_ON_THE_WIRE: tuple[type[Exception], ...] = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
WIRE_ATTEMPTS = 2

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
    if not applied.changed and resident.engine_model_name == resident.model_id:
        return raw
    document = dict(applied.body)
    if resident.engine_model_name != resident.model_id:
        document["model"] = resident.engine_model_name
    return json.dumps(document).encode("utf-8")


async def _watch_for_disconnect(request: Request) -> None:
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
    return await _unless_the_caller_leaves(
        client.post(url, content=body, headers=headers), request
    )


async def _sent_across_the_wire(
    attempt: Callable[[], Awaitable[Any]], *, where: str
) -> Any:
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
    if isinstance(exc, LOST_ON_THE_WIRE):
        return f" on {WIRE_ATTEMPTS} attempts"
    return ""


def _caller_gone(resident: Any) -> JSONResponse:
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
        pass
    except Exception as exc:
        print(
            f"crucible: work for a caller who left failed as it went: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
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
    return _set_model_in_frame(frame, resident.model_id)


def _set_model_in_frame(frame: bytes, model_id: str) -> bytes:
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
    async def over() -> None:
        await asyncio.to_thread(
            settlement.settle_quietly, "the last chat completion finished"
        )

    return over


def _after_the_stream(
    inflight: InFlight, entry: Entry, chat_over: Callable[[], Awaitable[None]]
) -> Callable[[], Awaitable[None]]:
    async def done() -> None:
        inflight.close(entry)
        await chat_over()

    return done


class _RelayResponse(StreamingResponse):
    def __init__(
        self,
        *args: Any,
        upstream: httpx.Response,
        when_relayed: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._upstream = upstream
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
    log_path = resident.log_path
    request = client.build_request(
        "POST",
        url,
        content=body,
        headers=JSON_HEADERS,
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT, read=None, write=60.0, pool=10.0
        ),
    )
    try:
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
            background=BackgroundTask(when_relayed),
        )

    async def relay() -> AsyncIterator[bytes]:
        if resident.engine_model_name == resident.model_id:
            async for chunk in upstream.aiter_bytes():
                yield chunk
            return
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
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            **extra_headers,
        },
    )


def _chat_limit_of(residency: Residency) -> tuple[int | None, str | None]:
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
