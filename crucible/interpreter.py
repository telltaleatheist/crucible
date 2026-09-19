"""The CPython Crucible runs on — from its publisher, pinned, verified, once.

PHASE20-CODE-NOT-ENVIRONMENTS.md. **A deploy ships CODE.** An interpreter is
not code we wrote, so it is not on our releases: it is downloaded from
astral-sh/python-build-standalone at a pinned release, checked against a digest
read out of that release's own `SHA256SUMS`, and never fetched again once a
tree on this machine is stamped with that digest.

This table used to live in `crucible/envpack.py`, where a pack build read it.
The packs are gone (PHASE20 section 6) and the table is not: it is the one
answer to "which interpreter", and it now has TWO readers rather than one —

  * the INSTALLER (`sdk/bootstrap`, `install.sh`, `install.ps1`) downloads the
    server's own 3.11 into `<CRUCIBLE_HOME>/server/` and pip-installs the
    release's wheel into it;
  * `jobenv.interpreter_for` downloads a recipe's own into
    `<CRUCIBLE_HOME>/interpreters/<version>/` when a recipe names a version the
    server does not run.

Why a standalone CPython and not the host's
-------------------------------------------
Nothing in Crucible may depend on a distro's Python, its version or its
packages (PHASE14 section 0, still true). `install_only` rather than the full
build: it is the layout `uv` ships — one `python/` directory, relocatable by
construction (relative rpaths, no absolute paths baked in) — and the full
archive carries debug symbols and a build manifest nothing here reads.

`install_only` on WINDOWS is the same archive shape and a different tree:
`python/python.exe`, `python/Scripts/`, `python/Lib/`, `python/DLLs/`, and no
`bin/` whatsoever. That is why the interpreter is asked for by
`interpreter_python(root, backend_kind)` and never spelled inline.

Why the PATH search is gone
---------------------------
`interpreter_for` used to look for `python<major.minor>` on PATH when a recipe
wanted a version the server did not run. A venv inherits the version AND the
build of whatever made it, so that was an env assembled out of an interpreter
of unknown provenance and unknown digest — a distro's, a conda's, somebody's
`uv python install` — sitting under 13 GB of pinned wheels. One publisher, one
pin, one digest: the same rule the recipes themselves follow.
"""

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
    """An interpreter could not be pinned, fetched or unpacked. By name.

    Every one of these is a name a person can search for:
    `interpreter_not_pinned`, `interpreter_download_failed`,
    `interpreter_sha_mismatch`, `interpreter_unpack_failed`.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StandalonePython:
    """One python-build-standalone asset, pinned by version AND by digest.

    ONE PLACE. The release tag, the asset name, the URL and the sha256 are all
    here and nowhere else, because two spellings of "which interpreter" is two
    machines running two of them under one version number.
    """

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


#: The minor the SERVER runs, on every backend. `requires-python = ">=3.11"` is
#: the floor `pyproject.toml` declares and the recipes were resolved by pip
#: against 3.11 (`envs/asr/cuda-linux.txt` records one pin that moved for
#: exactly that reason), so moving it would silently re-resolve every recipe
#: `crucible doctor` checks an env against.
SERVER_PYTHON = "3.11"

#: Every interpreter this build can install, by (backend, `major.minor`).
#:
#: All FOUR digests were read from the release's own `SHA256SUMS` — the three
#: 3.11 rows on 2026-09-14 and the 3.12 row on 2026-09-16
#: (https://github.com/astral-sh/python-build-standalone/releases/download/
#: 20260901/SHA256SUMS), not from a download and not from memory.
#:
#: PHASE15 4.4 first wrote the Windows asset as
#: `x86_64-pc-windows-msvc-SHARED-install_only` and then corrected itself from
#: that file: the word "shared" appears ZERO times in the 20260901 SHA256SUMS
#: (against 90 "static"), because the `-shared` infix is retired and the
#: Windows `install_only` build IS the shared one. A pin nobody can download is
#: an install that fails on the operator's machine and nowhere else, so
#: `tests/test_interpreter.py` pins the absence of "-shared".
#:
#: The 3.12 row is ONE RECIPE's: `envs/tts/higgs-v3-cuda-linux.txt` installs
#: sglang-omni 0.1.4, which pulls torch 2.13.0+cu130 and flashinfer built
#: against 3.12. It is keyed by version rather than by job type because the
#: requirement belongs to what is installed — see `jobenv.RECIPE_PYTHON`.
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
    # Same release and same CPython as the two backends, which is the property
    # that matters: the Windows host runs the same server code.
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

#: `<home>/interpreters/` — every CPython a recipe asked for that the server
#: does not itself run. Beside `server/` and `envs/`, and not under either:
#: `crucible doctor` reads every directory under `envs/` as one env, and the
#: server's own interpreter is the installer's rather than ours.
INTERPRETERS_DIRNAME = "interpreters"

#: What says a downloaded interpreter finished and WHICH bytes it is. Read
#: before anything downloads, so a second `crucible install` costs nothing.
STAMP_NAME = "crucible-interpreter.json"


def interpreters_dir(home: Path) -> Path:
    return home / INTERPRETERS_DIRNAME


def interpreter_dir(home: Path, python_version: str) -> Path:
    return interpreters_dir(home) / python_version


def pin_for(backend_kind: str, minor: str) -> StandalonePython:
    """The pinned interpreter for this backend and `major.minor`, or a refusal.

    A version nobody pinned is refused BY NAME rather than resolved from
    upstream's latest: an interpreter with no digest beside it is bytes nobody
    checked, and the refusal names what this build does pin so the next pin is
    one edit in one table.
    """
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
    """The interpreter inside an unpacked tree, whichever layout it has.

    PUBLIC because PHASE15 4.4 names it: python-build-standalone's Windows
    `install_only` tree is `python.exe` / `pythonw.exe` / `Scripts\\` / `Lib\\`
    / `DLLs\\` and has NO `bin/`, so an install and a venv that each spell
    `root / "bin" / "python"` are two places that have to learn the same thing
    and one of them will not.

    An unknown backend is refused rather than guessed at. Guessing `bin/python`
    would produce "unpacked without a bin/python" for a tree that is perfectly
    fine, which is a refusal that sends its reader to the wrong file.
    """
    if backend_kind == LLAMA_WINDOWS:
        return root / "python.exe"
    if backend_kind in (CUDA_LINUX, MLX_DARWIN):
        return root / "bin" / "python"
    raise InterpreterError(
        "interpreter_not_pinned",
        f"{backend_kind!r} has no interpreter layout; the backends are "
        f"{sorted({kind for kind, _ in INTERPRETERS})}",
    )


# ------------------------------------------------------------------ progress


#: A line `crucible install` prints so the install TASK can turn a download's
#: bytes into the `progress {bytes_done, bytes_total, file}` event the pull
#: task already emits (PHASE13 section 3.3).
#:
#: THIS IS NOT A LOG SCRAPE (R4). The install task runs the console script as a
#: CHILD PROCESS and reads its stdout — that pipe is the only channel between
#: the two, and the alternative is an extra inherited file descriptor, which is
#: a second transport to keep working on two platforms for one event shape. So
#: the line is a declared WIRE with one owner (this module writes it and parses
#: it), carrying JSON rather than prose, and the human-readable line an
#: operator reads is printed separately and is not parsed by anything.
#:
#: It lives HERE because the interpreter download is the one thing left in an
#: install that moves bytes we can count: the recipes' wheels come from pip,
#: which reports its own progress in its own prose.
PROGRESS_PREFIX = "crucible-progress "


def progress_line(bytes_done: int, bytes_total: int | None, file: str) -> str:
    return PROGRESS_PREFIX + json.dumps(
        {"bytes_done": bytes_done, "bytes_total": bytes_total, "file": file}
    )


def parse_progress_line(line: str) -> dict[str, Any] | None:
    """The three fields, or None when this is an ordinary line."""
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


# ------------------------------------------------------------ the download


def _fetch(
    url: str,
    destination: Path,
    *,
    on_progress: ProgressHook | None = None,
    timeout: int = 120,
    chunk: int = 1 << 20,
) -> None:
    """One archive, into `destination`. No resume: it is 30 MB, not 8 GB.

    The pack downloader that used to live here resumed with a `Range` request
    because a part was up to 1900 MiB. An interpreter is one small file whose
    digest is checked the moment it lands, so a half-finished one is deleted
    and fetched again rather than appended to — which is also the only way to
    be sure a proxy that ignored the range did not build a corrupt archive.
    """
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
    """Fetch, verify by digest, unpack into `dest`, stamp. Returns its python.

    THE ORDER IS THE POINT. Nothing lands at `dest` until the digest matched
    and the tree unpacked whole: the archive goes to a scratch directory, the
    tree is built beside `dest`, and the move is the last act. A refusal here
    therefore leaves the machine exactly as it found it, which is what makes
    running the install again the correct answer to every one of them.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(dest.parent), prefix=".interpreter-"
    ) as scratch:
        work = Path(scratch)
        archive = work / pin.asset
        if on_line is not None:
            on_line(f"fetching {pin.asset} from python-build-standalone")
        _fetch(pin.url, archive, on_progress=on_progress)
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
        # `install_only` archives carry ONE top-level `python/` directory, and
        # that directory IS the interpreter — so the tree is moved rather than
        # the archive's root, which would put everything one level too deep and
        # leave `bin/python` where nothing looks for it.
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
            # A tree with no stamp, or one whose stamp named other bytes. It is
            # under `<home>/interpreters/<version>/`, which is ours and nobody
            # else's, so replacing it is finishing an install rather than
            # touching somebody's machine.
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
    """`<home>/interpreters/<version>/`'s python, downloading it once.

    ONCE, AND THE STAMP IS WHY. The bytes are pinned by digest, so a tree whose
    stamp names that digest IS those bytes — there is nothing a second download
    could correct, and an install that re-fetched 30 MB on every run would be
    the same "rebuilt because code changed" shape PHASE20 exists to delete.

    `pin` is an ARGUMENT for the tests, never a second source: a caller that
    does not name one gets `pin_for`'s answer, and there is no third place a
    version could come from.
    """
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
