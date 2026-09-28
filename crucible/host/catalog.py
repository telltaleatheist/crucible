from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from ..platform.errors import HostError
from ..platform.runner import Runner
from ..protocol import API_HEADER, API_VERSION, LOOPBACK, api_headers
from ..wsl import default_user_argv

CATALOG_TIMEOUT_SECONDS = 60.0

SUBMIT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class Subject:
    kind: str
    id: str
    name: str
    installed: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.id)

    def __str__(self) -> str:
        return f"{self.kind} {self.id}"


def parse_catalog(document: Any, where: str) -> list[Subject]:
    rows = document.get("rows") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise HostError(
            "catalog_unreadable",
            f"{where} did not answer a catalog: expected an object with `rows`, "
            f"got {type(document).__name__}"
            + (f" with keys {sorted(document)}" if isinstance(document, dict) else "")
            + ". Nothing is migrated off a document this step cannot read.",
        )
    subjects: list[Subject] = []
    for row in rows:
        if not isinstance(row, dict):
            raise HostError(
                "catalog_unreadable",
                f"{where}: a subject row is a {type(row).__name__}, not an object",
            )
        for field in ("kind", "id", "installed"):
            if field not in row:
                raise HostError(
                    "catalog_unreadable",
                    f"{where}: a subject row has no {field!r}; its fields were "
                    f"{sorted(row)}",
                )
        if (not isinstance(row["kind"], str) or not row["kind"]
                or not isinstance(row["id"], str) or not row["id"]
                or type(row["installed"]) is not bool):
            raise HostError("catalog_unreadable", f"{where}: kind/id must be non-empty strings and installed must be a boolean")
        subjects.append(
            Subject(
                kind=str(row["kind"]),
                id=str(row["id"]),
                name=str(row.get("name") or row["id"]),
                installed=bool(row["installed"]),
            )
        )
    return subjects


class CatalogRefusal(HostError):
    def __init__(self, code: str, message: str, who: str = "") -> None:
        super().__init__(code, message)
        self.who = who


def refusal_from(body: bytes, status: int, where: str, what: str) -> CatalogRefusal:
    code = f"http_{status}"
    message = body.decode("utf-8", "replace").strip()[:400]
    who = ""
    try:
        document = json.loads(body.decode("utf-8"))
        error = document.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
            details = error.get("details")
            if isinstance(details, dict) and details.get("who"):
                who = str(details["who"])
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    return CatalogRefusal(code, f"{where}: {what}: {message}", who)


class CatalogPort(Protocol):
    @property
    def where(self) -> str:
        ...

    def installed_subjects(self) -> list[Subject]: ...

    def pull(self, subject: Subject) -> None:
        ...

    def remove(self, subject: Subject) -> None:
        ...


class HttpCatalog:
    def __init__(self, base_url: str, token: str, *, where: str) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._where = where

    @property
    def where(self) -> str:
        return self._where

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None, timeout_s: float
    ) -> bytes:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = api_headers(self._token)
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self._base}{path}", data=data, method=method, headers=headers
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise refusal_from(exc.read(), exc.code, self._where, f"{method} {path}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise HostError(
                "catalog_unreachable",
                f"{self._where} did not answer {method} {path}: {exc}. Nothing is "
                "removed from a machine whose catalog cannot be read.",
            ) from None

    def installed_subjects(self) -> list[Subject]:
        body = self._request("GET", "/v1/catalog", None, CATALOG_TIMEOUT_SECONDS)
        return [row for row in parse_catalog(json.loads(body), self._where) if row.installed]

    def pull(self, subject: Subject) -> None:
        self._request(
            "POST",
            "/v1/tasks",
            {"type": "pull", "kind": subject.kind, "id": subject.id},
            SUBMIT_TIMEOUT_SECONDS,
        )

    def remove(self, subject: Subject) -> None:
        self._request(
            "DELETE",
            f"/v1/catalog/{subject.kind}/{subject.id}",
            None,
            CATALOG_TIMEOUT_SECONDS,
        )


class StoppedWindowsCatalog:
    def __init__(self, config, backend, pending: set[tuple[str, str]]) -> None:
        from ..backend import LLAMA_WINDOWS
        if config.backend_kind != LLAMA_WINDOWS or backend.kind != LLAMA_WINDOWS:
            raise HostError("migration_source_invalid", "Cleanup requires the native Windows catalog")
        self._config = config
        self._backend = backend
        self._pending = set(pending)

    @property
    def where(self) -> str:
        return "the stopped Windows engine"

    def _subjects(self):
        from ..catalog import subjects
        return [row for row in subjects(self._config, self._backend) if row.kind != "engine"]

    def installed_subjects(self) -> list[Subject]:
        rows = self._subjects()
        unknown = self._pending - {(row.kind, row.id) for row in rows}
        if unknown:
            raise HostError("migration_cleanup_subject_unknown", f"The native catalog cannot retire these recorded subjects: {sorted(unknown)}")
        return [Subject(row.kind, row.id, row.name or row.id, True)
                for row in rows if row.installed() is not None or (row.kind, row.id) in self._pending]

    def pull(self, subject: Subject) -> None:
        raise HostError("migration_source_readonly", "The stopped Windows engine cannot download models")

    def remove(self, subject: Subject) -> None:
        from ..errors import CrucibleError
        from ..weights import WeightsShared
        from .cleanup_record import cleanup_subjects, record_cleanup
        row = next((row for row in self._subjects() if (row.kind, row.id) == subject.key), None)
        if row is None:
            raise CatalogRefusal("subject_unknown", f"The native catalog no longer declares {subject}")
        try:
            saved = cleanup_subjects(self._config.home)
            if subject.key not in saved:
                record_cleanup(self._config.home, saved | {subject.key})
            row.remove()
            self._pending.discard(subject.key)
        except WeightsShared as exc:
            raise CatalogRefusal(exc.code, str(exc)) from exc
        except (OSError, CrucibleError) as exc:
            raise CatalogRefusal("subject_remove_failed", str(exc)) from exc


class GuestCatalog:
    STATUS_MARK = "\n<<<status:"

    def __init__(
        self, runner: Runner, distro: str, token: str, port: int, *, where: str
    ) -> None:
        self._runner = runner
        self._distro = distro
        self._token = token
        self._port = port
        self._where = where

    @property
    def where(self) -> str:
        return self._where

    def curl_argv(self, method: str, path: str, body: str | None) -> list[str]:
        words = [
            "curl",
            "-sS",
            "-X",
            method,
            "-H",
            f"Authorization: Bearer {self._token}",
            "-H",
            f"{API_HEADER}: {API_VERSION}",
            "-w",
            f"{self.STATUS_MARK}%{{http_code}}",
        ]
        if body is not None:
            words += ["-H", "Content-Type: application/json", "-d", body]
        words.append(f"http://{LOOPBACK}:{self._port}{path}")
        return default_user_argv(self._distro, words)

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None, timeout_s: float
    ) -> bytes:
        payload = None if body is None else json.dumps(body)
        result = self._runner.run(self.curl_argv(method, path, payload), timeout_s=timeout_s)
        if not result.ok:
            raise HostError(
                "catalog_unreachable",
                f"{self._where} did not answer {method} {path}: {result.output_tail()}. "
                "Nothing is removed from a machine whose catalog cannot be read.",
            )
        text, mark, status_text = result.stdout.rpartition(self.STATUS_MARK)
        if mark == "" or status_text.strip() == "":
            raise HostError(
                "catalog_unreadable",
                f"{self._where}: curl printed no status for {method} {path}; it said "
                f"{result.stdout.strip()[:200]!r}",
            )
        status = int(status_text.strip())
        if status >= 400:
            raise refusal_from(text.encode("utf-8"), status, self._where, f"{method} {path}")
        return text.encode("utf-8")

    def installed_subjects(self) -> list[Subject]:
        body = self._request("GET", "/v1/catalog", None, CATALOG_TIMEOUT_SECONDS)
        return [row for row in parse_catalog(json.loads(body), self._where) if row.installed]

    def pull(self, subject: Subject) -> None:
        self._request(
            "POST",
            "/v1/tasks",
            {"type": "pull", "kind": subject.kind, "id": subject.id},
            SUBMIT_TIMEOUT_SECONDS,
        )

    def remove(self, subject: Subject) -> None:
        self._request(
            "DELETE",
            f"/v1/catalog/{subject.kind}/{subject.id}",
            None,
            CATALOG_TIMEOUT_SECONDS,
        )
