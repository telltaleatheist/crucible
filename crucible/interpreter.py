from __future__ import annotations

import errno
import http.client
import json
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from .errors import CrucibleError
from .weights import ProgressHook, sha256_of


class InterpreterError(CrucibleError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StandalonePython:
    python_version: str
    release: str
    asset: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            "https://github.com/astral-sh/python-build-standalone/releases/"
            f"download/{self.release}/{self.asset}"
        )


SERVER_PYTHON = "3.11"

INTERPRETERS: dict[tuple[str, str], StandalonePython] = {
    (CUDA_LINUX, "3.11"): StandalonePython(
        python_version="3.11.16",
        release="20260901",
        asset="cpython-3.11.16+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
        sha256="faa0758583a63f14c5eee516af82738403b59c13edda6fc0a21d953febd89eed",
    ),
    (MLX_DARWIN, "3.11"): StandalonePython(
        python_version="3.11.16",
        release="20260901",
        asset="cpython-3.11.16+20260901-aarch64-apple-darwin-install_only.tar.gz",
        sha256="50424fa409e8ae84b82a3052522f64695b47dff2158b70bb7358e0ebd6c085c9",
    ),
    (LLAMA_WINDOWS, "3.11"): StandalonePython(
        python_version="3.11.16",
        release="20260901",
        asset="cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz",
        sha256="6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6",
    ),
    (CUDA_LINUX, "3.12"): StandalonePython(
        python_version="3.12.14",
        release="20260901",
        asset="cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
        sha256="936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22",
    ),
}

INTERPRETERS_DIRNAME = "interpreters"

STAMP_NAME = "crucible-interpreter.json"


def interpreters_dir(home: Path) -> Path:
    return home / INTERPRETERS_DIRNAME


def interpreter_dir(home: Path, python_version: str) -> Path:
    return interpreters_dir(home) / python_version


def pin_for(backend_kind: str, minor: str) -> StandalonePython:
    found = INTERPRETERS.get((backend_kind, minor))
    if found is None:
        raise InterpreterError(
            "interpreter_not_pinned",
            f"this build pins no python {minor} for {backend_kind!r}; it pins "
            + ", ".join(
                f"{version} on {kind}" for kind, version in sorted(INTERPRETERS)
            )
            + ". Add the row to crucible/interpreter.py with the sha256 from "
            "python-build-standalone's own SHA256SUMS",
        )
    return found


def interpreter_python(root: Path, backend_kind: str) -> Path:
    if backend_kind == LLAMA_WINDOWS:
        return root / "python.exe"
    if backend_kind in (CUDA_LINUX, MLX_DARWIN):
        return root / "bin" / "python"
    raise InterpreterError(
        "interpreter_not_pinned",
        f"{backend_kind!r} has no interpreter layout; the backends are "
        f"{sorted({kind for kind, _ in INTERPRETERS})}",
    )


PROGRESS_PREFIX = "crucible-progress "


def progress_line(bytes_done: int, bytes_total: int | None, file: str) -> str:
    return PROGRESS_PREFIX + json.dumps(
        {"bytes_done": bytes_done, "bytes_total": bytes_total, "file": file}
    )


def parse_progress_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"bytes_done", "bytes_total", "file"}:
        return None
    return payload


# A dropped download is weather (Owen, 2026-11-04: "resumable downloads ... a dropped
# connection mid multi-GB env resumes, not restarts"): it is continued from the bytes already
# on disk, FETCH_PAUSE_S apart, up to FETCH_ATTEMPTS tries, about five minutes of no network.
FETCH_ATTEMPTS = 30
FETCH_PAUSE_S = 10.0
PARTIAL_SUFFIX = ".partial"
# Answers that are the request's fault, not the network's: a later try gets the same answer.
_NOT_WEATHER = frozenset({400, 401, 403, 404, 405, 410})


class _Weather(InterpreterError):
    """A try that failed for a reason a later try may not meet."""


def partial_path(destination: Path) -> Path:
    return destination.with_name(destination.name + PARTIAL_SUFFIX)


def fetch(
    url: str,
    destination: Path,
    *,
    on_progress: ProgressHook | None = None,
    timeout: int = 120,
    chunk: int = 1 << 20,
    attempts: int = FETCH_ATTEMPTS,
    pause_s: float = FETCH_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
    on_wait: Callable[[str], None] | None = None,
) -> None:
    """Download `url` to `destination`, resuming: the bytes land in `<destination>.partial`,
    which a failed try (and a killed process) leaves where it is, and the next try asks for
    the rest with a Range request. Only a finished transfer is renamed to `destination`; the
    caller's digest check says whether the bytes are right. An answer that is the request's
    fault (a 404) ends the fetch at once; anything else is waited on, and `on_wait` hears each
    wait in a sentence, so someone watching can tell it from a hang."""
    if attempts < 1:
        raise InterpreterError(
            "interpreter_download_failed",
            f"attempts={attempts} is not a number of tries; a download that is "
            "not attempted has not failed either.",
        )
    partial = partial_path(destination)
    destination.unlink(missing_ok=True)
    last: InterpreterError | None = None
    for attempt in range(attempts):
        try:
            _fetch_once(url, partial, on_progress=on_progress, timeout=timeout, chunk=chunk)
        except _Weather as exc:
            last = exc
            if attempt + 1 < attempts:
                if on_wait is not None:
                    on_wait(
                        f"{exc.message}; waiting {pause_s:g} s and trying again "
                        f"(try {attempt + 1} of {attempts})"
                    )
                sleep(pause_s)
            continue
        partial.replace(destination)
        return
    assert last is not None
    have = partial.stat().st_size if partial.exists() else 0
    raise InterpreterError(
        "interpreter_download_failed",
        f"{last.message}; after {attempts} tries {pause_s:g} s apart, {have:_} bytes are kept "
        f"at {partial} and the next attempt continues from them",
    )


def _fetch_once(
    url: str,
    partial: Path,
    *,
    on_progress: ProgressHook | None = None,
    timeout: int = 120,
    chunk: int = 1 << 20,
) -> None:
    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if have > 0:
        request.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length_header = response.headers.get("Content-Length")
            length = int(length_header) if length_header is not None else None
            if response.status == 206 and have > 0:
                done, mode = have, "ab"
            else:
                # A 200 is the whole file: the server ignored the Range, so it starts over.
                done, mode = 0, "wb"
            total = None if length is None else done + length
            with partial.open(mode) as handle:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if on_progress is not None:
                        on_progress(done, total, partial.name.removesuffix(PARTIAL_SUFFIX))
            if total is not None and done < total:
                raise _Weather(
                    "interpreter_download_failed",
                    f"{url} ended at {done:_} of {total:_} bytes",
                )
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have > 0:
            # The kept bytes are not a prefix the server can continue: start over.
            partial.unlink(missing_ok=True)
            raise _Weather(
                "interpreter_download_failed",
                f"{url} would not continue from byte {have:_} (HTTP 416); starting it over",
            ) from None
        if exc.code in _NOT_WEATHER:
            raise InterpreterError(
                "interpreter_download_failed",
                f"{url} answered HTTP {exc.code} {exc.reason}",
            ) from None
        raise _Weather(
            "interpreter_download_failed",
            f"{url} answered HTTP {exc.code} {exc.reason}",
        ) from None
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise InterpreterError(
                "interpreter_download_failed",
                f"the disk filled while fetching {url} into {partial}: {exc}",
            ) from None
        raise _Weather(
            "interpreter_download_failed", f"could not fetch {url}: {exc}"
        ) from None
    except (http.client.HTTPException, ValueError) as exc:
        raise _Weather(
            "interpreter_download_failed", f"could not fetch {url}: {exc!r}"
        ) from None


def download_interpreter(
    pin: StandalonePython,
    dest: Path,
    backend_kind: str,
    *,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(dest.parent), prefix=".interpreter-"
    ) as scratch:
        work = Path(scratch)
        archive = work / pin.asset
        if on_line is not None:
            on_line(f"fetching {pin.asset} from python-build-standalone")
        fetch(pin.url, archive, on_progress=on_progress, on_wait=on_line)
        digest = sha256_of(archive)
        if digest != pin.sha256:
            raise InterpreterError(
                "interpreter_sha_mismatch",
                f"{pin.asset} hashes to {digest} and crucible/interpreter.py "
                f"pins {pin.sha256}. Nothing was unpacked",
            )
        unpacked = work / "unpacked"
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(unpacked)
        except (tarfile.TarError, OSError) as exc:
            raise InterpreterError(
                "interpreter_unpack_failed", f"could not open {pin.asset}: {exc}"
            ) from None
        tree = unpacked / "python"
        if not tree.is_dir():
            raise InterpreterError(
                "interpreter_unpack_failed",
                f"{pin.asset} has no top-level python/ directory; "
                f"it unpacked {sorted(p.name for p in unpacked.iterdir())}",
            )
        python = interpreter_python(tree, backend_kind)
        if not python.is_file():
            raise InterpreterError(
                "interpreter_unpack_failed",
                f"{pin.asset} unpacked without a {python.relative_to(tree)}",
            )
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(tree), str(dest))
    (dest / STAMP_NAME).write_text(
        json.dumps(
            {
                "python_version": pin.python_version,
                "release": pin.release,
                "asset": pin.asset,
                "sha256": pin.sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return interpreter_python(dest, backend_kind)


def ensure_interpreter(
    home: Path,
    backend_kind: str,
    minor: str,
    *,
    pin: StandalonePython | None = None,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> Path:
    chosen = pin if pin is not None else pin_for(backend_kind, minor)
    dest = interpreter_dir(home, chosen.python_version)
    stamp = dest / STAMP_NAME
    if stamp.is_file():
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        python = interpreter_python(dest, backend_kind)
        if recorded.get("sha256") == chosen.sha256 and python.is_file():
            if on_line is not None:
                on_line(f"python {chosen.python_version}: already at {dest}")
            return python
    return download_interpreter(
        chosen, dest, backend_kind, on_line=on_line, on_progress=on_progress
    )
