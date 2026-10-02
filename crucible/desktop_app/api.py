from __future__ import annotations

import urllib.error
import urllib.parse
from typing import Any, Callable, Iterator

from ..client import transport
from ..client.connection import Connection, resolve
from ..client.errors import ClientRefusal, error_in, next_step, unreachable
from ..errors import CrucibleError


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status


def quoted(*parts: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in parts)


def refusal_from_http(exc: urllib.error.HTTPError, connection: Connection | None) -> ApiError:
    _body, error = error_in(exc.read())
    if error is None:
        return ApiError("http_error", f"the server answered HTTP {exc.code} with no reason. "
                        "Run `crucible doctor` to see what is wrong", exc.code)
    said = next_step(error, connection)
    if said is not None:
        return ApiError(error.code, said.split(": ", 1)[1], exc.code)
    return ApiError(error.code, error.message, exc.code)


class LocalApi:
    def __init__(self, connect: Callable[[], Connection] = resolve) -> None:
        self._connect = connect
        self._connection: Connection | None = None

    def connection(self) -> Connection:
        if self._connection is None:
            try:
                self._connection = self._connect()
            except (ClientRefusal, CrucibleError, OSError) as exc:
                code, _, message = str(exc).partition(": ")
                raise ApiError(code if message else "no_local_engine", message or str(exc)) from None
        return self._connection

    def forget(self) -> None:
        self._connection = None

    def _guarded(self, work: Callable[[Connection], Any]) -> Any:
        connection = self.connection()
        try:
            return work(connection)
        except urllib.error.HTTPError as exc:
            raise refusal_from_http(exc, connection) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.forget()
            code, _, message = unreachable(connection, exc).partition(": ")
            raise ApiError(code, message) from None
        except ValueError as exc:
            raise ApiError("bad_answer", f"the server's answer was not JSON ({exc}). "
                           "Run `crucible doctor` to see what is wrong") from None

    def get(self, path: str) -> Any:
        return self._guarded(lambda c: transport.call(c, "GET", path))

    def send(self, method: str, path: str, body: Any = None) -> Any:
        return self._guarded(lambda c: transport.call(c, method, path, json_body=body))

    def follow(self, path: str, last_event_id: int = 0) -> Iterator[dict[str, Any]]:
        connection = self.connection()
        try:
            yield from transport.follow(connection, path, last_event_id=last_event_id)
        except urllib.error.HTTPError as exc:
            raise refusal_from_http(exc, connection) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            code, _, message = unreachable(connection, exc).partition(": ")
            raise ApiError(code, message) from None
