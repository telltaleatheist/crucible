"""A job type's own venv — `~/.crucible/envs/<type>/` (PHASE4-AUDIO.md section 0).

`llm` never needed this module, because vLLM and mlx-lm are *servers*: Crucible
starts one and talks HTTP to it. The phase 4 types are **libraries** —
faster-whisper, ultimate-rvc and the Qwen3 aligner are imported, not connected to
— and each one's torch pin is incompatible with the others and with the server's.
So each gets a venv of its own, built from `envs/<type>/<backend>.txt`, and its
worker script is run as `<that venv's python> <worker.py>`.

One env per job type, never one giant env (DESIGN.md section 5). A recipe is
pinned exactly, every line of it, for the reason `envs/llm/cuda-linux.txt` gives:
a later install must not drift into a different torch and a different numerical
result.

Why this is not `crucible/jobenv.py`
------------------------------------
It should be. `jobenv` is this module with `job_type` hardcoded to `"llm"`, and
the two share almost every line. They are apart because phase 4 was built beside
phases 2 and 3 in one tree and `jobenv.py` was live under another builder's feet
while this was written. Folding `jobenv` into this module — `llm_env_dir(home)`
becomes `worker_env_dir(home, "llm")` and the `EnvStatus` shape is already
identical — is a follow-up, and a mechanical one.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import CrucibleError

RECIPES_DIR_ENV = "CRUCIBLE_WORKER_RECIPES_DIR"

#: What `crucible doctor` and `crucible install <type>` report the version of: the
#: one library the env exists for, so a half-built or wrong-backend env is obvious
#: at a glance rather than after a 3 GB model pull.
HEADLINE_PACKAGE: dict[str, str] = {
    "align": "qwen-asr",
    "asr": "faster-whisper",
    "rvc": "ultimate-rvc",
}

#: Job types that have a worker env at all. `llm` is deliberately absent: it is
#: `jobenv`'s, until the two modules are merged. `denoise` is absent for a
#: different reason — it has no env of its own; see below.
WORKER_JOB_TYPES: tuple[str, ...] = ("align", "asr", "rvc")

#: Which job types one env serves. Almost always itself; `rvc` is the exception
#: and `denoise` is why.
#:
#: audio-separator is torch, the rvc env already holds the exact torch it wants,
#: and BookForge runs both out of one env today (`electron/denoise-bridge.ts`
#: reaches for the RVC env's python). A second venv would be a second 3 GB torch
#: on disk to drive the same card.
#:
#: It is a table rather than a fact each caller knows because two of them need
#: it and they are far apart: `crucible install rvc` decides the capability flag
#: for both types, and `crucible doctor` names the install command that would
#: turn a type on. A second copy of "denoise lives in the rvc env" is exactly
#: the shape ARCHITECTURE.md R1 is about.
JOB_TYPES_SERVED_BY_ENV: dict[str, tuple[str, ...]] = {
    "align": ("align",),
    "asr": ("asr",),
    "rvc": ("rvc", "denoise"),
}


def env_for_job_type(job_type: str) -> str | None:
    """Which env's recipe builds what this job type needs, or None.

    None means no worker env is involved at all (`llm` has `jobenv`'s, `echo`
    and the lifecycle types have none), which is a different answer from "an env
    that does not exist".
    """
    for env, served in JOB_TYPES_SERVED_BY_ENV.items():
        if job_type in served:
            return env
    return None


class WorkerEnvError(CrucibleError):
    """A worker env is missing, or could not be built. Carries the reason."""


@dataclass(frozen=True)
class EnvStatus:
    """What `crucible doctor` prints for one worker env."""

    job_type: str
    installed: bool
    path: Path
    detail: str
    python_version: str | None
    packages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_type": self.job_type,
            "installed": self.installed,
            "path": str(self.path),
            "detail": self.detail,
            "python_version": self.python_version,
            "packages": dict(self.packages),
        }


# ------------------------------------------------------------------- layout


def worker_env_dir(home: Path, job_type: str) -> Path:
    return home / "envs" / job_type


def worker_env_python(home: Path, job_type: str) -> Path:
    """The venv interpreter the worker script is run with."""
    return worker_env_dir(home, job_type) / "bin" / "python"


def _stamp_path(home: Path, job_type: str) -> Path:
    """Written only after `pip install -r <recipe>` returns 0."""
    return worker_env_dir(home, job_type) / "crucible-env.json"


def recipes_dir(job_type: str) -> Path:
    """Where `envs/<job_type>/*.txt` live. Refuses by name if absent."""
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        root = Path(override).expanduser()
        if not root.is_dir():
            raise WorkerEnvError(f"{RECIPES_DIR_ENV}={override!r} is not a directory")
    else:
        root = Path(__file__).resolve().parent.parent / "envs"
    path = root / job_type
    if not path.is_dir():
        raise WorkerEnvError(
            f"no env recipes for job type {job_type!r} at {path}; crucible must run "
            f"from a checkout (pip install -e .) or ${RECIPES_DIR_ENV} must point at "
            "the recipes"
        )
    return path


def recipe_for(job_type: str, backend_kind: str) -> Path:
    """The recipe for this host's backend, or a named refusal.

    A backend with no recipe is not a shrug. `asr` has no `mlx-darwin.txt`
    because CTranslate2 has no Metal backend (crucible/asrmodels.py says it at
    length), and the refusal here names the backends that *are* shipped so the
    reader is not left wondering whether the file was forgotten.
    """
    root = recipes_dir(job_type)
    path = root / f"{backend_kind}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.txt"))
        raise WorkerEnvError(
            f"no {job_type} env recipe for backend {backend_kind!r} at {path}; this "
            f"build ships {job_type} recipes for {available}"
        )
    return path


#: A PEP 508 direct reference at a full commit sha: `name @ git+<url>@<sha>`.
#: The sha is required and a branch name is refused, for the reason every
#: manifest's `revision` is a 40-character commit: a branch is a moving target
#: and a pin is a statement about bytes.
_DIRECT_REFERENCE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*@\s*(?P<url>git\+[^\s@]+@(?P<sha>[0-9a-f]{40}))$"
)


def _canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _requirement_lines(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        yield stripped


def recipe_pins(path: Path) -> dict[str, str]:
    """The `name==version` pins in a recipe, by lower-cased name.

    Direct references (`name @ git+url@sha`) are not pins in this sense and are
    deliberately left out: `pip list` reports a git install's declared *version*,
    not its commit, so checking one here would compare a sha against `0.5.11` and
    call every correctly built env broken. They are checked separately, against
    `pip freeze`, by `installed_direct_refs`.
    """
    pins: dict[str, str] = {}
    for stripped in _requirement_lines(path):
        if _DIRECT_REFERENCE.match(stripped):
            continue
        name, separator, version = stripped.partition("==")
        if separator != "==":
            raise WorkerEnvError(
                f"{path.name}: {stripped!r} is neither a `name==version` pin nor a "
                "`name @ git+<url>@<40-character sha>` direct reference; every "
                "requirement in a recipe is pinned exactly, and a branch name is "
                "not a pin"
            )
        pins[_canonical(name)] = version.strip()
    return pins


def recipe_direct_refs(path: Path) -> dict[str, str]:
    """The `name @ git+url@sha` requirements in a recipe, by lower-cased name.

    `rvc` is the only user and it is not an indulgence: `generate convert-dir`,
    the warm-model batch command this whole job type is built on, exists ONLY in
    Owen's fork (`telltaleatheist/ultimate-rvc`, branch `bookforge`) and not in
    the `ultimate-rvc` on PyPI. Both call themselves version 0.5.11, so the
    version is not an identity here — the commit is, and that is what this pins
    and what `installed_direct_refs` checks.
    """
    refs: dict[str, str] = {}
    for stripped in _requirement_lines(path):
        match = _DIRECT_REFERENCE.match(stripped)
        if match is not None:
            refs[_canonical(match.group("name"))] = match.group("url")
    return refs


# ------------------------------------------------------------------- status


def installed_packages(home: Path, job_type: str) -> dict[str, str]:
    """`pip list` from the env, by lower-cased name. {} if there is no venv."""
    python = worker_env_python(home, job_type)
    if not python.is_file():
        return {}
    completed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "list",
            "--format=json",
            "--disable-pip-version-check",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise WorkerEnvError(
            f"`pip list` in {worker_env_dir(home, job_type)} exited "
            f"{completed.returncode}: {completed.stderr.strip() or 'no output'}"
        )
    return {
        _canonical(entry["name"]): entry["version"]
        for entry in json.loads(completed.stdout)
    }


def installed_direct_refs(home: Path, job_type: str) -> dict[str, str]:
    """`pip freeze`'s direct references from the env, by lower-cased name.

    A second subprocess, and only ever run for a job type whose recipe HAS a
    direct reference, because `pip list` cannot answer this: for a git install it
    reports the version the project declares and says nothing about the commit.
    `pip freeze` writes the reference back out in the form the recipe used, which
    is what makes the two comparable.
    """
    python = worker_env_python(home, job_type)
    if not python.is_file():
        return {}
    completed = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise WorkerEnvError(
            f"`pip freeze` in {worker_env_dir(home, job_type)} exited "
            f"{completed.returncode}: {completed.stderr.strip() or 'no output'}"
        )
    refs: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        match = _DIRECT_REFERENCE.match(line.strip())
        if match is not None:
            refs[_canonical(match.group("name"))] = match.group("url")
    return refs


def env_status(home: Path, job_type: str, backend_kind: str) -> EnvStatus:
    """Is this type's env there, and does it hold exactly what the recipe pins?"""
    directory = worker_env_dir(home, job_type)
    python = worker_env_python(home, job_type)
    if not python.is_file():
        return EnvStatus(
            job_type=job_type,
            installed=False,
            path=directory,
            detail=f"no venv at {directory} — run `crucible install {job_type}`",
            python_version=None,
            packages={},
        )
    stamp = _stamp_path(home, job_type)
    if not stamp.is_file():
        return EnvStatus(
            job_type=job_type,
            installed=False,
            path=directory,
            detail=(
                f"{directory} exists but {stamp.name} does not: the last "
                f"`crucible install {job_type}` did not finish. Re-run it."
            ),
            python_version=None,
            packages={},
        )
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["backend"] != backend_kind:
        return EnvStatus(
            job_type=job_type,
            installed=False,
            path=directory,
            detail=(
                f"{directory} was installed for backend {record['backend']!r}, this "
                f"host is {backend_kind!r} — run `crucible install {job_type} --force`"
            ),
            python_version=record["python_version"],
            packages={},
        )

    present = installed_packages(home, job_type)
    recipe = recipe_for(job_type, backend_kind)
    pins = recipe_pins(recipe)
    wrong = sorted(
        f"{name} is {present.get(name, 'absent')}, recipe pins {version}"
        for name, version in pins.items()
        if present.get(name) != version
    )
    references = recipe_direct_refs(recipe)
    if references:
        # Only asked when the recipe has one, so no job type pays for a second
        # `pip` subprocess to learn that it has no git installs.
        built = installed_direct_refs(home, job_type)
        wrong += sorted(
            f"{name} is {built.get(name, 'absent')}, recipe pins {url}"
            for name, url in references.items()
            if built.get(name) != url
        )
    if wrong:
        return EnvStatus(
            job_type=job_type,
            installed=False,
            path=directory,
            detail=(
                f"{directory} does not match {backend_kind}.txt: " + "; ".join(wrong)
            ),
            python_version=record["python_version"],
            packages=present,
        )
    headline = HEADLINE_PACKAGE[job_type]
    if headline in references:
        # A git install. `pip list` reports the version the project DECLARES,
        # which for `ultimate-rvc` is 0.5.11 for both Owen's fork and the PyPI
        # release — so the version says nothing and the commit says everything.
        # The doctor line names the commit for the same reason the recipe pins it.
        headline_detail = f"{headline} @ {references[headline].rsplit('@', 1)[1][:12]}"
    elif headline in present:
        headline_detail = f"{headline} {present[headline]}"
    else:
        raise WorkerEnvError(
            f"{directory} matches {backend_kind}.txt, but {headline!r} — the "
            f"package the {job_type} env exists for — is not installed in it. "
            "Either the recipe no longer installs it or HEADLINE_PACKAGE names "
            "the wrong thing; both are bugs in this build, not in the env."
        )
    return EnvStatus(
        job_type=job_type,
        installed=True,
        path=directory,
        detail=(
            f"{headline_detail}, python {record['python_version']}, "
            f"{len(present)} packages"
        ),
        python_version=record["python_version"],
        packages=present,
    )


def require_env(home: Path, job_type: str, backend_kind: str) -> Path:
    """The env's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, job_type, backend_kind)
    if not status.installed:
        raise WorkerEnvError(status.detail)
    return worker_env_python(home, job_type)


# ------------------------------------------------------------------ install


def install_worker_env(
    home: Path,
    job_type: str,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> EnvStatus:
    """Create `~/.crucible/envs/<type>/` and install this backend's recipe.

    `on_line` is called with each line of pip's output so the CLI can show it.
    Returns the resulting status. Raises WorkerEnvError naming what went wrong.
    """
    recipe = recipe_for(job_type, backend_kind)
    directory = worker_env_dir(home, job_type)
    stamp = _stamp_path(home, job_type)

    if directory.exists() and not force:
        existing = env_status(home, job_type, backend_kind)
        if existing.installed:
            return existing
        if stamp.is_file():
            raise WorkerEnvError(
                f"{directory} exists but does not match this host: {existing.detail}. "
                "Pass --force to rebuild it."
            )
        # A half-built venv from an interrupted install: no stamp, so nothing
        # downstream has ever trusted it. Rebuilding it is the only correct move.

    started = time.monotonic()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        shutil.rmtree(directory)

    _run(
        [sys.executable, "-m", "venv", str(directory)],
        f"could not create the venv at {directory}",
        on_line,
    )
    python = worker_env_python(home, job_type)
    if not python.is_file():
        raise WorkerEnvError(
            f"`python -m venv {directory}` returned 0 but there is no {python}"
        )
    _run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        f"could not upgrade pip in the {job_type} env",
        on_line,
    )
    _run(
        [str(python), "-m", "pip", "install", "-r", str(recipe)],
        f"could not install {recipe} into {directory}",
        on_line,
    )

    version = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print('.'.join(map(str, sys.version_info[:3])))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    elapsed = time.monotonic() - started
    stamp.write_text(
        json.dumps(
            {
                "job_type": job_type,
                "backend": backend_kind,
                "recipe": recipe.name,
                "python_version": version,
                "seconds": round(elapsed, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return env_status(home, job_type, backend_kind)


def _run(command: list[str], failure: str, on_line: Callable[[str], None] | None) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        del tail[:-40]
        if on_line is not None:
            on_line(line)
    code = process.wait()
    if code != 0:
        raise WorkerEnvError(
            f"{failure}: `{' '.join(command)}` exited {code}\n" + "\n".join(tail)
        )
