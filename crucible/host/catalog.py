"""The two catalogs the weights migration reads, and the door it deletes through.

PHASE15-HOST.md 3.5 (the weights rule) and 3.5a (`DELETE /v1/catalog/{kind}/{id}`).

THE HOST NEVER TOUCHES A WEIGHTS FILE. 3.5a exists precisely so that it does
not: `crucible/weights.py` owns where a subject's bytes live, and a host that
deleted a directory it had composed itself would be a second owner of that
layout. Preparation reads the two live APIs. After activation, retirement calls
the stopped native catalog's existing owner functions; port 7100 then belongs
to the guest and must never receive a deletion meant for the Windows copy.

TWO PORTS, BECAUSE THE TWO SERVERS ARE REACHED DIFFERENTLY
-----------------------------------------------------------
During the move both servers want `127.0.0.1:7100`, and on Windows the
Windows one has it. So:

* the WINDOWS server is reached over loopback from this process
  (`HttpCatalog`);
* the GUEST server is reached by running `curl` INSIDE the distro
  (`GuestCatalog`), which is the only address that is unambiguously its own.

They speak the same three verbs and answer the same refusals, so the sequence
in `installer.py` never branches on which side it is talking to — it is the
ORDER that carries 3.5's rule, not the transport.

ONE TOKEN. After `migrate-config` the guest holds the Windows server's token
(4.3), which is what lets one bearer open both doors, and is the same fact
that keeps every paired app paired.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import HostError
from .runner import Runner
from .. import API_VERSION

#: A catalog read is cheap, and a server that will not answer one is a server
#: this step cannot reason about — so the timeout is short and the failure is
#: a refusal rather than a retry.
CATALOG_TIMEOUT_SECONDS = 60.0

#: A pull is submitted as a TASK and returns as soon as it is accepted; the
#: waiting is the caller's poll of the catalog, not a long request.
SUBMIT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class Subject:
    """One row of `GET /v1/catalog`, reduced to what the migration needs."""

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
    """`GET /v1/catalog`'s body to subjects. Refuses a shape it cannot read.

    NOT tolerant, and the reason is specific to this caller: a catalog that
    half-parses is indistinguishable from a server that ships fewer subjects,
    and the difference here decides whether a weights file is deleted on
    Windows before the guest has its own.
    """
    rows = document.get("subjects") if isinstance(document, dict) else None
    if rows is None and isinstance(document, list):
        rows = document
    if not isinstance(rows, list):
        raise HostError(
            "catalog_unreadable",
            f"{where} did not answer a catalog: expected an object with "
            f"`subjects`, got {type(document).__name__}. Nothing is migrated off "
            "a document this step cannot read.",
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
    """A server's own refusal, with its code kept verbatim.

    `subject_in_use` is the whole reason this class exists: the migration's
    retry turns on that exact code, and a generic "it failed" would make the
    step skip a subject that 3.5 says it must never skip. `who` carries
    `details.who` (3.5a) — the lease, the resident model or the task holding
    it — so the sentence a person reads names what to close.
    """

    def __init__(self, code: str, message: str, who: str = "") -> None:
        super().__init__(code, message)
        self.who = who


def refusal_from(body: bytes, status: int, where: str, what: str) -> CatalogRefusal:
    """A server's `{"error": {code, message, details}}` as a refusal."""
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
        # A body that is not the error envelope is still evidence; the status
        # and the text are kept rather than replaced with a guess.
        pass
    return CatalogRefusal(code, f"{where}: {what}: {message}", who)


class CatalogPort(Protocol):
    """What the migration may do to a server. Three verbs and no more."""

    @property
    def where(self) -> str:
        """A name for a sentence: "the Windows engine", "the guest"."""

    def installed_subjects(self) -> list[Subject]: ...

    def pull(self, subject: Subject) -> None:
        """`POST /v1/tasks {type: pull}`. Returns once it is accepted."""

    def remove(self, subject: Subject) -> None:
        """`DELETE /v1/catalog/{kind}/{id}` (3.5a). Refuses by the server's name."""


class HttpCatalog:
    """The server this process can dial. Loopback, bearer, stdlib only."""

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
        headers = {"Authorization": f"Bearer {self._token}", "X-Crucible-Api": str(API_VERSION)}
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
    """Retire weights through their catalog owner after the native process exits.

    Port 7100 belongs to the guest after activation; using the old HTTP client
    there would delete the destination. The controller must establish ownership
    and shutdown before constructing this adapter. Runtime binaries stay local.
    """

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
        # A deletion interrupted after removing its stamp must still be resumed.
        return [Subject(row.kind, row.id, row.name or row.id, True)
                for row in rows if row.installed() is not None or (row.kind, row.id) in self._pending]

    def pull(self, subject: Subject) -> None:
        raise HostError("migration_source_readonly", "The stopped Windows engine cannot download models")

    def remove(self, subject: Subject) -> None:
        from ..errors import CrucibleError
        from .installer import cleanup_subjects, record_cleanup
        for row in self._subjects():
            if (row.kind, row.id) == subject.key:
                try:
                    # Downloads completed during preparation can add a subject.
                    # Record it before deletion, so interruption after its stamp
                    # disappears still has a named catalog operation to resume.
                    saved = cleanup_subjects(self._config.home)
                    if subject.key not in saved:
                        record_cleanup(self._config.home, saved | {subject.key})
                    row.remove()
                    self._pending.discard(subject.key)
                except (OSError, CrucibleError) as exc:
                    raise CatalogRefusal("subject_remove_failed", str(exc)) from exc
                return
        raise CatalogRefusal("subject_unknown", f"The native catalog no longer declares {subject}")


class GuestCatalog:
    """The server inside the distro, reached with the guest's own `curl`.

    Not through a port forward and not through `\\wsl$`: while the move runs,
    the WINDOWS server holds `127.0.0.1:7100` on this machine, so the guest's
    only unambiguous address is its own loopback, from inside.

    `--exec`, always: wsl.exe pre-expands `$var` in its implicit-shell form,
    and every wsl call in this system uses `--exec` for that reason. The JSON
    body travels as ONE argv element, so no quoting rule on either side of
    wsl.exe can change a byte of it.
    """

    #: `-w` appends the status after the body. NOT `-f`: that would hide the
    #: refusal document, and the refusal's CODE is the thing this whole step
    #: turns on.
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
        argv = [
            "wsl.exe",
            "-d",
            self._distro,
            "--exec",
            "curl",
            "-sS",
            "-X",
            method,
            "-H",
            f"Authorization: Bearer {self._token}",
            "-H",
            f"X-Crucible-Api: {API_VERSION}",
            "-w",
            f"{self.STATUS_MARK}%{{http_code}}",
        ]
        if body is not None:
            argv += ["-H", "Content-Type: application/json", "-d", body]
        argv.append(f"http://127.0.0.1:{self._port}{path}")
        return argv

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None, timeout_s: float
    ) -> bytes:
        payload = None if body is None else json.dumps(body)
        result = self._runner.run(self.curl_argv(method, path, payload), timeout_s=timeout_s)
        if not result.ok:
            raise HostError(
                "catalog_unreachable",
                f"{self._where} did not answer {method} {path}: {result.said()}. "
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
