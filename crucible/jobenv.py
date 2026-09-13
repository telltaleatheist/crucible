"""A job type's env — `~/.crucible/envs/<key>/` (PHASE2-LLM.md section 2).

One env per job type, never one giant env (DESIGN.md section 5). Each is a venv
built from the server's own interpreter, with a recipe (`envs/<job type>/
<name>.txt`) installed from PyPI into it. The engines are then started as
subprocesses of that venv's python — the server process itself never imports
torch, vLLM, mlx or narrator.

Why an env is not simply one per job type
-----------------------------------------
`llm` is: one venv, `~/.crucible/envs/llm/`, whichever backend the host is.
`tts` is not, and the reason is in narrator's dependency matrix rather than in
Crucible's design (PHASE3-TTS.md section 4). Orpheus needs `vllm==0.7.3` — the
last version whose V0 engine takes per-request logits processors, which is what
the EOS boost *is* — and Higgs v3 needs `vllm-omni` against a much later torch;
installing both into one env resolves torch twice and breaks whichever loses. So
on `cuda-linux` there are two tts envs and the voice manifest's `narrator_engine`
picks which one a load uses, while on `mlx-darwin` the two engines genuinely do
share one, and there is one env there.

So an env is named by an `EnvSpec`, and each job type states its own naming rule
in its own constructor below — `llm_env()` and `tts_env()` — where the two can be
read against each other.

This file was `crucible/llmenv.py` until the `tts` job type needed the same
machinery. Nothing about the `llm` env's layout, stamp or refusals changed in the
move.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CrucibleError

RECIPES_DIR_ENV = "CRUCIBLE_RECIPES_DIR"

#: What `crucible doctor` and `crucible install llm` report the version of. The
#: engine module of each backend, so a wrong-backend env is obvious at a glance.
BACKEND_HEADLINE_PACKAGE: dict[str, str] = {
    "cuda-linux": "vllm",
    "mlx-darwin": "mlx-lm",
}

#: `tts`'s headline is the same package on both backends, because narrator is
#: what the env exists for on either — the engine underneath it (SGLang-Omni on
#: `cuda-linux`, mlx-audio on `mlx-darwin`) is narrator's own dependency and not
#: Crucible's, and a doctor line naming it would be reporting a level down.
NARRATOR_PACKAGE = "narrator"


class EnvError(CrucibleError):
    """A job type's env is missing, or could not be built. Carries the reason."""


@dataclass(frozen=True)
class EnvSpec:
    """Which env, and which recipe builds it.

    `key` is the directory under `~/.crucible/envs/`; `recipe_name` is the
    `<name>.txt` inside `envs/<job_type>/`. Two fields rather than one because
    they answer two different questions — what is installed here, and what
    installs it — and `tts` on `cuda-linux` is where they differ.
    """

    job_type: str
    key: str
    recipe_name: str
    headline: str


def llm_env(backend_kind: str) -> EnvSpec:
    """The one `llm` env. One per host, whichever backend it is."""
    if backend_kind not in BACKEND_HEADLINE_PACKAGE:
        raise EnvError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_HEADLINE_PACKAGE)}"
        )
    return EnvSpec(
        job_type="llm",
        key="llm",
        recipe_name=backend_kind,
        headline=BACKEND_HEADLINE_PACKAGE[backend_kind],
    )


def tts_env(narrator_engine: str, backend_kind: str) -> EnvSpec:
    """The `tts` env this narrator engine runs in on this backend.

    On `cuda-linux` the two engines cannot share a venv (see the module
    docstring), so the engine is in the env's name and in the recipe's. On
    `mlx-darwin` they can and do, so there is one env and one recipe, named for
    the backend the way `llm`'s are.
    """
    if backend_kind not in BACKEND_HEADLINE_PACKAGE:
        raise EnvError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_HEADLINE_PACKAGE)}"
        )
    if backend_kind == "cuda-linux":
        return EnvSpec(
            job_type="tts",
            key=f"tts-{narrator_engine}",
            recipe_name=f"{narrator_engine}-{backend_kind}",
            headline=NARRATOR_PACKAGE,
        )
    return EnvSpec(
        job_type="tts",
        key="tts",
        recipe_name=backend_kind,
        headline=NARRATOR_PACKAGE,
    )


@dataclass(frozen=True)
class EnvStatus:
    """What `crucible doctor` prints for one env."""

    installed: bool
    path: Path
    detail: str
    python_version: str | None
    packages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "path": str(self.path),
            "detail": self.detail,
            "python_version": self.python_version,
            "packages": dict(self.packages),
        }


# ------------------------------------------------------------------- layout


def env_dir(home: Path, spec: EnvSpec) -> Path:
    return home / "envs" / spec.key


def env_python(home: Path, spec: EnvSpec) -> Path:
    """The venv interpreter the engines are spawned from."""
    return env_dir(home, spec) / "bin" / "python"


def _stamp_path(home: Path, spec: EnvSpec) -> Path:
    """Written only after `pip install -r <recipe>` returns 0."""
    return env_dir(home, spec) / "crucible-env.json"


def recipes_dir(job_type: str) -> Path:
    """Where `envs/<job_type>/*.txt` live. Refuses by name if absent.

    `$CRUCIBLE_RECIPES_DIR`, when set, is the recipe ROOT and the job type is a
    directory under it — one variable for every job type, rather than one per
    type, which could point two halves of a build at two checkouts.
    """
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        root = Path(override).expanduser()
        if not root.is_dir():
            raise EnvError(f"{RECIPES_DIR_ENV}={override!r} is not a directory")
        path = root / job_type
        if not path.is_dir():
            raise EnvError(
                f"{RECIPES_DIR_ENV}={override!r} holds no {job_type!r} directory"
            )
        return path
    path = Path(__file__).resolve().parent.parent / "envs" / job_type
    if not path.is_dir():
        raise EnvError(
            f"no {job_type} env recipes at {path}; crucible must run from a "
            f"checkout (pip install -e .) or ${RECIPES_DIR_ENV} must point at them"
        )
    return path


def recipe_for(spec: EnvSpec) -> Path:
    """The recipe that builds this env, or a named refusal."""
    root = recipes_dir(spec.job_type)
    path = root / f"{spec.recipe_name}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.txt"))
        raise EnvError(
            f"no {spec.job_type} env recipe for {spec.recipe_name!r} at {path}; "
            f"this build ships recipes for {available}"
        )
    return path


def recipe_pins(path: Path) -> dict[str, str]:
    """The `name==version` pins in a recipe, by lower-cased name."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        name, separator, version = stripped.partition("==")
        if separator != "==":
            raise EnvError(
                f"{path.name}: {stripped!r} is not a `name==version` pin; every "
                "requirement in a recipe is pinned exactly"
            )
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


# ------------------------------------------------------------------- status


def installed_packages(home: Path, spec: EnvSpec) -> dict[str, str]:
    """`pip list` from this venv, by lower-cased name. {} if there is no venv."""
    python = env_python(home, spec)
    if not python.is_file():
        return {}
    completed = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise EnvError(
            f"`pip list` in {env_dir(home, spec)} exited {completed.returncode}: "
            f"{completed.stderr.strip() or 'no output'}"
        )
    return {
        entry["name"].lower().replace("_", "-"): entry["version"]
        for entry in json.loads(completed.stdout)
    }


def env_status(home: Path, spec: EnvSpec, backend_kind: str) -> EnvStatus:
    """Is this env there, and does it hold what the recipe pins?"""
    directory = env_dir(home, spec)
    python = env_python(home, spec)
    install = f"crucible install {spec.job_type}"
    if not python.is_file():
        return EnvStatus(
            installed=False,
            path=directory,
            detail=f"no venv at {directory} — run `{install}`",
            python_version=None,
            packages={},
        )
    stamp = _stamp_path(home, spec)
    if not stamp.is_file():
        return EnvStatus(
            installed=False,
            path=directory,
            detail=(
                f"{directory} exists but {stamp.name} does not: the last "
                f"`{install}` did not finish. Re-run it."
            ),
            python_version=None,
            packages={},
        )
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["backend"] != backend_kind:
        return EnvStatus(
            installed=False,
            path=directory,
            detail=(
                f"{directory} was installed for backend {record['backend']!r}, this "
                f"host is {backend_kind!r} — run `{install} --force`"
            ),
            python_version=record["python_version"],
            packages={},
        )

    present = installed_packages(home, spec)
    recipe = recipe_for(spec)
    pins = recipe_pins(recipe)
    wrong = sorted(
        f"{name} is {present.get(name, 'absent')}, recipe pins {version}"
        for name, version in pins.items()
        if present.get(name) != version
    )
    if wrong:
        return EnvStatus(
            installed=False,
            path=directory,
            detail=f"{directory} does not match {recipe.name}: " + "; ".join(wrong),
            python_version=record["python_version"],
            packages=present,
        )
    return EnvStatus(
        installed=True,
        path=directory,
        detail=(
            f"{spec.headline} {present[spec.headline]}, python "
            f"{record['python_version']}, {len(present)} packages"
        ),
        python_version=record["python_version"],
        packages=present,
    )


def require_env(home: Path, spec: EnvSpec, backend_kind: str) -> Path:
    """This venv's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, spec, backend_kind)
    if not status.installed:
        raise EnvError(status.detail)
    return env_python(home, spec)


# ------------------------------------------------------------------ install


def install_env(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Any = None,
) -> EnvStatus:
    """Create `~/.crucible/envs/<key>/` and install this env's recipe.

    `on_line` is called with each line of pip's output so the CLI can show it.
    Returns the resulting status. Raises EnvError naming what went wrong.
    """
    recipe = recipe_for(spec)
    directory = env_dir(home, spec)
    stamp = _stamp_path(home, spec)

    if directory.exists() and not force:
        existing = env_status(home, spec, backend_kind)
        if existing.installed:
            return existing
        if stamp.is_file():
            raise EnvError(
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
    python = env_python(home, spec)
    if not python.is_file():
        raise EnvError(
            f"`python -m venv {directory}` returned 0 but there is no {python}"
        )
    _run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        f"could not upgrade pip in the {spec.job_type} env",
        on_line,
    )
    _run(
        [str(python), "-m", "pip", "install", "-r", str(recipe)],
        f"could not install {recipe} into {directory}",
        on_line,
    )

    version = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    elapsed = time.monotonic() - started
    stamp.write_text(
        json.dumps(
            {
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
    return env_status(home, spec, backend_kind)


def _run(command: list[str], failure: str, on_line: Any) -> None:
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
        raise EnvError(f"{failure}: `{' '.join(command)}` exited {code}\n" + "\n".join(tail))
