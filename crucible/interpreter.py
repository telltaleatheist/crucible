from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
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


def fetch(
    url: str,
    destination: Path,
    *,
    on_progress: ProgressHook | None = None,
    timeout: int = 120,
    chunk: int = 1 << 20,
    attempts: int = 1,
) -> None:
    if attempts < 1:
        raise InterpreterError(
            "interpreter_download_failed",
            f"attempts={attempts} is not a number of tries; a download that is "
            "not attempted has not failed either.",
        )
    last: InterpreterError | None = None
    for attempt in range(attempts):
        try:
            _fetch_once(url, destination, on_progress=on_progress, timeout=timeout, chunk=chunk)
            return
        except InterpreterError as exc:
            last = exc
            if attempt + 1 < attempts:
                destination.unlink(missing_ok=True)
    assert last is not None
    raise last


def _fetch_once(
    url: str,
    destination: Path,
    *,
    on_progress: ProgressHook | None = None,
    timeout: int = 120,
    chunk: int = 1 << 20,
) -> None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header is not None else None
            done = 0
            with destination.open("wb") as handle:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if on_progress is not None:
                        on_progress(done, total, destination.name)
    except urllib.error.HTTPError as exc:
        raise InterpreterError(
            "interpreter_download_failed",
            f"{url} answered HTTP {exc.code} {exc.reason}",
        ) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise InterpreterError(
            "interpreter_download_failed", f"could not fetch {url}: {exc}"
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
        fetch(pin.url, archive, on_progress=on_progress)
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
