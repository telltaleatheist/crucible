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

from . import jobenv
from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError

RECIPES_DIR_ENV = "CRUCIBLE_WORKER_RECIPES_DIR"

#: One name for the stamp, shared with `jobenv` (which declares it) so the two
#: env modules cannot leave two different files behind.
ENV_STAMP_NAME = jobenv.ENV_STAMP_NAME

#: What `crucible doctor` and `crucible install <type>` report the version of:
#: the one library the env exists for, so a half-built or wrong-backend env is
#: obvious at a glance rather than after a 3 GB model pull.
#:
#: PER (JOB TYPE, BACKEND), since 2026-09-14, because `asr` stopped having one
#: answer: `cuda-linux` installs faster-whisper (CTranslate2) and `mlx-darwin`
#: installs mlx-whisper (MLX), and they are two libraries rather than two builds
#: of one. The other two types have one library on both backends and say so by
#: repeating it — written out rather than defaulted, because "this type has the
#: same headline everywhere" is a fact about those recipes and not a rule, and
#: the next type to gain a second engine must be a KeyError here rather than a
#: doctor line quietly naming a package the env does not contain.
HEADLINE_PACKAGE: dict[tuple[str, str], str] = {
    ("align", CUDA_LINUX): "qwen-asr",
    ("align", MLX_DARWIN): "qwen-asr",
    ("asr", CUDA_LINUX): "faster-whisper",
    ("asr", MLX_DARWIN): "mlx-whisper",
    ("rvc", CUDA_LINUX): "ultimate-rvc",
    ("rvc", MLX_DARWIN): "ultimate-rvc",
}


def headline_package(job_type: str, backend_kind: str) -> str:
    """The library this env exists for. Refuses a pair nobody has decided."""
    found = HEADLINE_PACKAGE.get((job_type, backend_kind))
    if found is None:
        raise WorkerEnvError(
            f"no headline package for the {job_type!r} env on {backend_kind!r}; "
            f"this build knows {sorted(HEADLINE_PACKAGE)}"
        )
    return found

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
    #: The recipe's two halves as stamped. `jobenv.EnvStatus` carries the same
    #: three fields and says why at length; the two classes stay separate only
    #: until the modules are merged (see this module's header).
    environment_sha256: str | None = None
    direct_references: dict[str, str] | None = None
    recipe_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_type": self.job_type,
            "installed": self.installed,
            "path": str(self.path),
            "detail": self.detail,
            "python_version": self.python_version,
            "packages": dict(self.packages),
            "environment_sha256": self.environment_sha256,
            "direct_references": (
                None if self.direct_references is None
                else dict(self.direct_references)
            ),
        }


# ------------------------------------------------------------------- layout


def worker_env_dir(home: Path, job_type: str) -> Path:
    return home / "envs" / job_type


def worker_env_python(home: Path, job_type: str) -> Path:
    """The venv interpreter the worker script is run with."""
    return worker_env_dir(home, job_type) / "bin" / "python"


def stamp_path(home: Path, job_type: str) -> Path:
    """Written after pip returned 0 in this venv, and at no other moment.

    Named after `jobenv`'s constant so the two env modules cannot write two
    different filenames.
    """
    return worker_env_dir(home, job_type) / ENV_STAMP_NAME


def recipes_dir(job_type: str) -> Path:
    """Where `envs/<job_type>/*.txt` live. Refuses by name if absent."""
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        root = Path(override).expanduser()
        if not root.is_dir():
            raise WorkerEnvError(f"{RECIPES_DIR_ENV}={override!r} is not a directory")
    else:
        root = Path(__file__).resolve().parent / "envs"
    path = root / job_type
    if not path.is_dir():
        raise WorkerEnvError(
            f"no env recipes for job type {job_type!r} at {path}; they are package "
            f"data and this install has lost them, or ${RECIPES_DIR_ENV} must "
            "point at them"
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
    stamp = stamp_path(home, job_type)
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
    # `.get` for these three alone: a stamp written before PHASE20 hashed the
    # recipe whole and recorded no references. See `jobenv.env_status`, which
    # says why at length — absent is an answer, not a default.
    environment_sha256 = record.get("environment_sha256")
    direct_references = record.get("direct_references")
    recipe_text = record.get("recipe_text")
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
            environment_sha256=environment_sha256,
            direct_references=direct_references,
            recipe_text=recipe_text,
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
            environment_sha256=environment_sha256,
            direct_references=direct_references,
            recipe_text=recipe_text,
        )
    headline = headline_package(job_type, backend_kind)
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
        environment_sha256=environment_sha256,
        direct_references=direct_references,
        recipe_text=recipe_text,
    )


def require_env(home: Path, job_type: str, backend_kind: str) -> Path:
    """The env's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, job_type, backend_kind)
    if not status.installed:
        raise WorkerEnvError(status.detail)
    return worker_env_python(home, job_type)


# ------------------------------------------------------------------ install


def plan_install(
    home: Path, job_type: str, backend_kind: str, *, force: bool = False
) -> jobenv.EnvPlan:
    """What `install_worker_env` would do here, without doing any of it.

    `jobenv.plan_env` is THE rule and this is the worker env's spelling of the
    question. Two copies of "does this env need anything" is two answers, and
    the doctor and the installer must give one (ARCHITECTURE.md R1).
    """
    return jobenv.plan_env(
        directory=worker_env_dir(home, job_type),
        stamp=stamp_path(home, job_type),
        recipe=recipe_for(job_type, backend_kind),
        backend_kind=backend_kind,
        installed=env_status(home, job_type, backend_kind).installed,
        force=force,
        install_command=f"crucible install {job_type}",
    )


def install_worker_env(
    home: Path,
    job_type: str,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> EnvStatus:
    """Bring `~/.crucible/envs/<type>/` to this backend's recipe, and stamp it.

    PHASE20 section 4, the same sequence `jobenv.install_env` walks: the plan
    decides, the venv is rebuilt only under `--force`, and every other drift is
    pip into the venv that is already there.

    `on_line` is called with each line of pip's output so the CLI can show it.
    Returns the resulting status. Raises WorkerEnvError naming what went wrong.
    """
    recipe = recipe_for(job_type, backend_kind)
    directory = worker_env_dir(home, job_type)
    try:
        plan = plan_install(home, job_type, backend_kind, force=force)
    except jobenv.EnvError as exc:
        # The shared planner speaks `jobenv`'s error; this module's callers
        # catch this module's, so it is re-raised rather than leaking a second
        # exception type out of one command.
        raise WorkerEnvError(str(exc)) from exc
    if plan.action == jobenv.PLAN_NOTHING:
        return env_status(home, job_type, backend_kind)
    if on_line is not None:
        on_line(f"{plan.action}: {plan.detail}")

    started = time.monotonic()
    python_version: str | None = None

    if plan.action == jobenv.PLAN_BUILD:
        # BEFORE the venv, and long before pip dials a mirror. The shared
        # guard, for the shared reason — see `jobenv.refuse_without_room`.
        try:
            jobenv.refuse_without_room(
                job_type=job_type, recipe=recipe, directory=directory
            )
        except jobenv.EnvError as exc:
            raise WorkerEnvError(str(exc)) from exc
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
    else:
        python = worker_env_python(home, job_type)
        # The venv was not rebuilt, so its interpreter is the one the stamp
        # already names.
        python_version = json.loads(
            stamp_path(home, job_type).read_text(encoding="utf-8")
        )["python_version"]

    if plan.action == jobenv.PLAN_REFERENCES:
        # One git sha moved and the environment half hashed the same, so the
        # 3 GB of torch under `ultimate-rvc` is exactly what the recipe asks
        # for. `--no-deps` because those dependencies were just proved
        # unchanged; `--force-reinstall` because both Owen's fork and the PyPI
        # release call themselves 0.5.11, so pip sees nothing to do.
        for line in plan.lines:
            _run(
                [
                    str(python), "-m", "pip", "install",
                    "--no-deps", "--force-reinstall", line,
                ],
                f"could not reinstall {line} into {directory}",
                on_line,
            )
    else:
        _run(
            [str(python), "-m", "pip", "install", "-r", str(recipe)],
            f"could not install {recipe} into {directory}",
            on_line,
        )

    if python_version is None:
        python_version = subprocess.run(
            [
                str(python),
                "-c",
                "import sys; print('.'.join(map(str, sys.version_info[:3])))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
    stamp_path(home, job_type).write_text(
        json.dumps(
            {
                "job_type": job_type,
                "backend": backend_kind,
                "recipe": recipe.name,
                # THE TWO HALVES, apart, hashed and parsed by `jobenv` so the
                # two env modules cannot disagree about what a recipe says.
                "environment_sha256": jobenv.environment_sha256(recipe),
                "direct_references": jobenv.recipe_direct_references(recipe),
                "recipe_text": jobenv.recipe_text(recipe),
                "python_version": python_version,
                "seconds": round(time.monotonic() - started, 1),
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


# ---------------------------------------------------------------------------
# THE LIBRARIES A WORKER FINDS AT COMPUTE TIME, WHICH ARE NOT THE ONES IT
# FINDS AT IMPORT TIME
# ---------------------------------------------------------------------------
#
# MEASURED 2026-09-15 on owens-pc, against a `crucible doctor` that reported
# `asr` as `ready: True` with the weights installed and 0 problems. The first
# real transcribe answered:
#
#     asr_window_failed — window 0 (0s):
#     RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
#
# The library was never missing. `nvidia-cublas-cu12` ships it inside the env at
# `site-packages/nvidia/cublas/lib/libcublas.so.12`, and it was there the whole
# time. What was missing is the loader path: pip puts CUDA libraries in
# per-package directories that the dynamic linker has no reason to search, and
# ctranslate2 links them with no RPATH pointing at that layout.
#
# WHY IT LOOKS LIKE HEALTH, and why the doctor could not have caught it. The
# model LOADS without these — `WhisperModel(..., device="cuda")` constructs
# fine, which is what any readiness probe would check. ctranslate2 resolves
# cuBLAS LAZILY, at the first matrix multiply, so the failure is not at import,
# not at load, and not at the first request either: it is at the first COMPUTE.
# Every check short of actually transcribing a second of audio passes.
#
# It cost a wrong diagnosis on the way in, which is worth recording: the first
# A/B ran `WhisperModel(...)` with and without the path, both succeeded, and the
# hypothesis was discarded as disproved. The experiment was measuring the wrong
# moment. Reproducing the real failure — transcribe, not load — showed the
# control failing with the job's exact message and the treatment returning three
# segments.
#
# DERIVED FROM THE ENV, never hardcoded: whatever `nvidia/*/lib` directories that
# env actually contains, plus the package's own bundled `.libs`. A list written
# here would go stale the day a recipe pins a different CUDA package set, and the
# staleness would look exactly like this defect does.


def cuda_library_path(env_dir: Path) -> str | None:
    """`LD_LIBRARY_PATH` additions for an env whose CUDA libs came from pip.

    `None` when there is nothing to add, so a caller can tell "no CUDA packages
    here" (a CPU env, a Mac) from "an empty path", and pass nothing rather than
    an empty variable that would shadow the inherited one.
    """
    site = sorted(env_dir.glob("lib/python*/site-packages"))
    if not site:
        return None
    packages = site[0]
    directories: list[str] = []
    # Every `nvidia/<package>/lib` this env actually has.
    nvidia = packages / "nvidia"
    if nvidia.is_dir():
        for child in sorted(nvidia.iterdir()):
            lib = child / "lib"
            if lib.is_dir():
                directories.append(str(lib))
    # auditwheel-style bundled libraries (`ctranslate2.libs`, and friends).
    for bundled in sorted(packages.glob("*.libs")):
        if bundled.is_dir():
            directories.append(str(bundled))
    return os.pathsep.join(directories) if directories else None


def worker_environment(env_dir: Path, inherited: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a worker in `env_dir` needs, over what it inherits.

    PREPENDED rather than replacing: an operator who has set `LD_LIBRARY_PATH`
    for their own reasons keeps it, and the env's own libraries win only over
    the search order, never over the variable.
    """
    base = dict(os.environ if inherited is None else inherited)
    addition = cuda_library_path(env_dir)
    if addition is None:
        return {}
    existing = base.get("LD_LIBRARY_PATH", "")
    return {
        "LD_LIBRARY_PATH": addition + (os.pathsep + existing if existing else "")
    }
