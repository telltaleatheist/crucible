from __future__ import annotations

import io
import json
import mimetypes
import urllib.request
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from .. import VERSION
from ..protocol import API_HEADER, API_VERSION, user_agent
from .connection import Connection
from .errors import ClientRefusal

REQUEST_TIMEOUT_SECONDS = 900.0

DOWNLOAD_CHUNK_BYTES = 64 * 1024

JSON = "application/json"

EVENT_STREAM = "text/event-stream"

USER_AGENT = user_agent("cli", VERSION)


def request_headers(token: str | None) -> dict[str, str]:
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    headers[API_HEADER] = str(API_VERSION)
    headers["User-Agent"] = USER_AGENT
    return headers


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener()


def open_url(
    origin: str,
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: bytes | None = None,
    content_type: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    headers = request_headers(token)
    if content_type is not None:
        headers["Content-Type"] = content_type
    if extra_headers is not None:
        headers.update(extra_headers)
    request = urllib.request.Request(origin + path, data=body, headers=headers, method=method)
    return opener().open(request, timeout=timeout)


def open_connection(
    connection: Connection,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    return open_url(
        connection.url,
        method,
        path,
        token=connection.token,
        body=body,
        content_type=content_type,
        extra_headers=extra_headers,
        timeout=timeout,
    )


def json_bytes(value: Any) -> bytes:
    return json.dumps(value).encode("utf-8")


def call(
    connection: Connection,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    body = None if json_body is None else json_bytes(json_body)
    with open_connection(
        connection,
        method,
        path,
        body=body,
        content_type=None if body is None else JSON,
        extra_headers=extra_headers,
    ) as response:
        if response.status == 204:
            return None
        raw = response.read()
    if raw == b"":
        return None
    return json.loads(raw.decode("utf-8"))


def stream_lines(response: Any) -> Iterator[str]:
    for raw_line in response:
        yield raw_line.decode("utf-8").rstrip("\n").rstrip("\r")


def sse_frames(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    current: dict[str, Any] = {}
    for line in lines:
        if line == "":
            if current:
                yield current
                current = {}
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "id":
            current["id"] = int(value)
        elif field == "event":
            current["event"] = value
        elif field == "data":
            current["data"] = json.loads(value)
    if current:
        yield current


def follow(
    connection: Connection, path: str, *, last_event_id: int = 0
) -> Iterator[dict[str, Any]]:
    extra = {"Accept": EVENT_STREAM}
    if last_event_id:
        extra["Last-Event-ID"] = str(last_event_id)
    with open_connection(connection, "GET", path, extra_headers=extra, timeout=None) as response:
        yield from sse_frames(stream_lines(response))


def chat_frames(
    connection: Connection, path: str, body: Any, extra_headers: dict[str, str] | None
) -> Iterator[Any]:
    headers = {"Accept": EVENT_STREAM}
    if extra_headers is not None:
        headers.update(extra_headers)
    with open_connection(
        connection, "POST", path,
        body=json_bytes(body),
        content_type=JSON,
        extra_headers=headers,
        timeout=None,
    ) as response:
        for line in stream_lines(response):
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                break
            yield json.loads(payload)


def download(
    connection: Connection,
    path: str,
    open_sink: Callable[[], AbstractContextManager[BinaryIO]],
) -> int:
    written = 0
    with open_connection(connection, "GET", path) as response, open_sink() as sink:
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            sink.write(chunk)
            written += len(chunk)
    return written


def multipart_file(source: Path, boundary: str) -> bytes:
    guessed, _ = mimetypes.guess_type(source.name)
    part_type = guessed if guessed is not None else "application/octet-stream"
    buffer = io.BytesIO()
    buffer.write(f"--{boundary}\r\n".encode("ascii"))
    buffer.write(
        f'Content-Disposition: form-data; name="file"; filename="{source.name}"\r\n'
        f"Content-Type: {part_type}\r\n\r\n".encode("utf-8")
    )
    buffer.write(source.read_bytes())
    buffer.write(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return buffer.getvalue()


def upload(connection: Connection, source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise ClientRefusal(f"input_missing: {source} is not a file")
    boundary = uuid.uuid4().hex
    with open_connection(
        connection,
        "POST",
        "/v1/uploads",
        body=multipart_file(source, boundary),
        content_type=f"multipart/form-data; boundary={boundary}",
    ) as response:
        return json.loads(response.read().decode("utf-8"))
