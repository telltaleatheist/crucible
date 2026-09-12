"""The `llm` job type's env — `~/.crucible/envs/llm/` (PHASE2-LLM.md section 2).

One env per job type, never one giant env (DESIGN.md section 5). This one is a
venv built from the server's own interpreter, with the host backend's recipe
(`envs/llm/<backend>.txt`) installed from PyPI into it. The engines are then
started as subprocesses of that venv's python — the server process itself never
imports torch, vLLM or mlx.
"""

from __future__ import annotations

import json
import os
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


class EnvError(CrucibleError):
    """The llm env is missing, or could not be built. Carries the reason."""


@dataclass(frozen=True)
class EnvStatus:
    """What `crucible doctor` prints for the llm env."""

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


def llm_env_dir(home: Path) -> Path:
    return home / "envs" / "llm"


def llm_env_python(home: Path) -> Path:
    """The venv interpreter the engines are spawned from."""
    return llm_env_dir(home) / "bin" / "python"


def llm_env_pip(home: Path) -> Path:
    return llm_env_dir(home) / "bin" / "pip"


def _stamp_path(home: Path) -> Path:
    """Written only after `pip install -r <recipe>` returns 0."""
    return llm_env_dir(home) / "crucible-env.json"


def recipes_dir() -> Path:
    """Where `envs/llm/*.txt` live. Refuses by name if absent."""
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise EnvError(f"{RECIPES_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent.parent / "envs" / "llm"
    if not path.is_dir():
        raise EnvError(
            f"no env recipes at {path}; crucible must run from a checkout "
            f"(pip install -e .) or ${RECIPES_DIR_ENV} must point at them"
        )
    return path


def recipe_for(backend_kind: str) -> Path:
    """The recipe for this host's backend, or a named refusal."""
    root = recipes_dir()
    path = root / f"{backend_kind}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.txt"))
        raise EnvError(
            f"no llm env recipe for backend {backend_kind!r} at {path}; this build "
            f"ships recipes for {available}"
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


def installed_packages(home: Path) -> dict[str, str]:
    """`pip list` from the llm venv, by lower-cased name. {} if there is no venv."""
    python = llm_env_python(home)
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
            f"`pip list` in {llm_env_dir(home)} exited {completed.returncode}: "
            f"{completed.stderr.strip() or 'no output'}"
        )
    return {
        entry["name"].lower().replace("_", "-"): entry["version"]
        for entry in json.loads(completed.stdout)
    }


def env_status(home: Path, backend_kind: str) -> EnvStatus:
    """Is the llm env there, and does it hold what the recipe pins?"""
    directory = llm_env_dir(home)
    python = llm_env_python(home)
    if not python.is_file():
        return EnvStatus(
            installed=False,
            path=directory,
            detail=f"no venv at {directory} — run `crucible install llm`",
            python_version=None,
            packages={},
        )
    stamp = _stamp_path(home)
    if not stamp.is_file():
        return EnvStatus(
            installed=False,
            path=directory,
            detail=(
                f"{directory} exists but {stamp.name} does not: the last "
                "`crucible install llm` did not finish. Re-run it."
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
                f"host is {backend_kind!r} — run `crucible install llm --force`"
            ),
            python_version=record["python_version"],
            packages={},
        )

    present = installed_packages(home)
    pins = recipe_pins(recipe_for(backend_kind))
    wrong = sorted(
        f"{name} is {present.get(name, 'absent')}, recipe pins {version}"
        for name, version in pins.items()
        if present.get(name) != version
    )
    if wrong:
        return EnvStatus(
            installed=False,
            path=directory,
            detail=f"{directory} does not match {backend_kind}.txt: " + "; ".join(wrong),
            python_version=record["python_version"],
            packages=present,
        )
    headline = BACKEND_HEADLINE_PACKAGE[backend_kind]
    return EnvStatus(
        installed=True,
        path=directory,
        detail=(
            f"{headline} {present[headline]}, python {record['python_version']}, "
            f"{len(present)} packages"
        ),
        python_version=record["python_version"],
        packages=present,
    )


def require_env(home: Path, backend_kind: str) -> Path:
    """The llm venv's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, backend_kind)
    if not status.installed:
        raise EnvError(status.detail)
    return llm_env_python(home)


# ------------------------------------------------------------------ install


def install_llm_env(
    home: Path,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Any = None,
) -> EnvStatus:
    """Create `~/.crucible/envs/llm/` and install the host backend's recipe.

    `on_line` is called with each line of pip's output so the CLI can show it.
    Returns the resulting status. Raises EnvError naming what went wrong.
    """
    recipe = recipe_for(backend_kind)
    directory = llm_env_dir(home)
    stamp = _stamp_path(home)

    if directory.exists() and not force:
        existing = env_status(home, backend_kind)
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
        import shutil

        shutil.rmtree(directory)

    _run(
        [sys.executable, "-m", "venv", str(directory)],
        f"could not create the venv at {directory}",
        on_line,
    )
    python = llm_env_python(home)
    if not python.is_file():
        raise EnvError(
            f"`python -m venv {directory}` returned 0 but there is no {python}"
        )
    _run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        "could not upgrade pip in the llm env",
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
    return env_status(home, backend_kind)


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
