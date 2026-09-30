from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import envpatches, interpreter
from .audiomodels import engines_on as audio_engines_on
from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .jobtypes import ENVS, FAMILIES, LLM_ENV
from .narratorengines import HIGGS_V3
from .videomodels import engines_on as video_engines_on

RECIPES_DIR_ENV = "CRUCIBLE_RECIPES_DIR"

ENV_STAMP_NAME = "crucible-env.json"

BACKEND_HEADLINE_PACKAGE: dict[str, str] = {
    "cuda-linux": "vllm",
    "mlx-darwin": "mlx-lm",
}

NARRATOR_PACKAGE = "narrator"

CUDA_LINUX_SERVING_STACK: dict[str, str] = {
    HIGGS_V3: "sglang-omni",
}

RECIPE_PYTHON: dict[str, str] = {
    "higgs-v3-cuda-linux": "3.12",
}

AUDIO_JOB_TYPE = "audio"

AUDIO_ENGINE_HEADLINE: dict[str, str] = {
    "stable-audio-3": "stable-audio-3",
    "yue2": "yue2-infer",
}

AUDIO_ENGINE_MODULE: dict[str, str] = {
    "stable-audio-3": "stable_audio_3",
    "yue2": "yue2",
}

AUDIO_CUDA_EXTRA_MODULE: dict[str, str] = {
    "stable-audio-3": "flash_attn",
}

VIDEO_JOB_TYPE = "video"

VIDEO_ENGINE_HEADLINE: dict[str, str] = {
    "ltx": "diffusers",
}

VIDEO_ENGINE_MODULES: dict[str, str] = {
    "ltx": "diffusers, gguf, torchao, av",
}

WORKER_JOB_TYPES: tuple[str, ...] = tuple(env.name for env in ENVS if env.worker)

WORKER_HEADLINE_PACKAGE: dict[tuple[str, str], str] = {
    ("align", CUDA_LINUX): "qwen-asr",
    ("align", MLX_DARWIN): "qwen-asr",
    ("asr", CUDA_LINUX): "faster-whisper",
    ("asr", MLX_DARWIN): "mlx-whisper",
    ("rvc", CUDA_LINUX): "ultimate-rvc",
    ("rvc", MLX_DARWIN): "ultimate-rvc",
    ("image", CUDA_LINUX): "diffusers",
    ("image", MLX_DARWIN): "mflux",
    ("segment", CUDA_LINUX): "transformers",
    ("segment", MLX_DARWIN): "transformers",
}

SEGMENT_SMOKE_IMPORT = "transformers.models.sam2, timm, kornia, einops, torchvision"

JOB_TYPES_SERVED_BY_ENV: dict[str, tuple[str, ...]] = {
    env.name: tuple(family.name for family in FAMILIES if family.env == env)
    for env in ENVS
    if env.worker
}

INSTALLABLE_JOB_TYPES: tuple[str, ...] = tuple(env.name for env in ENVS)

INSTALLER_FOR: dict[str, str] = {
    **{env.name: env.name for env in ENVS},
    **{family.name: family.env.name for family in FAMILIES if family.env is not None},
    "pages": LLM_ENV.name,
}


def _module_of(package: str) -> str:
    return package.replace("-", "_")


SMOKE_IMPORT: dict[str, dict[str, str]] = {
    "llm": {
        backend_kind: _module_of(package)
        for backend_kind, package in BACKEND_HEADLINE_PACKAGE.items()
    },
    **{
        job_type: {
            backend_kind: _module_of(package)
            for (worker, backend_kind), package in WORKER_HEADLINE_PACKAGE.items()
            if worker == job_type
        }
        for job_type in WORKER_JOB_TYPES
    },
    "segment": {
        CUDA_LINUX: SEGMENT_SMOKE_IMPORT,
        MLX_DARWIN: SEGMENT_SMOKE_IMPORT,
    },
    "tts-higgs-v3": {CUDA_LINUX: NARRATOR_PACKAGE},
    "tts": {MLX_DARWIN: NARRATOR_PACKAGE},
    **{
        f"{AUDIO_JOB_TYPE}-{engine}": {
            backend_kind: ", ".join(
                [module]
                + ([AUDIO_CUDA_EXTRA_MODULE[engine]]
                   if backend_kind == CUDA_LINUX and engine in AUDIO_CUDA_EXTRA_MODULE
                   else [])
            )
            for backend_kind in (CUDA_LINUX, MLX_DARWIN)
            if engine in audio_engines_on(backend_kind)
        }
        for engine, module in AUDIO_ENGINE_MODULE.items()
    },
    **{
        f"{VIDEO_JOB_TYPE}-{engine}": {
            backend_kind: modules
            for backend_kind in (CUDA_LINUX, MLX_DARWIN)
            if engine in video_engines_on(backend_kind)
        }
        for engine, modules in VIDEO_ENGINE_MODULES.items()
    },
}


@dataclass(frozen=True)
class NoInstaller:
    words: str
    shared_with: str | None


def no_installer(job_type: str) -> NoInstaller | None:
    if job_type in INSTALLABLE_JOB_TYPES:
        return None
    shared = INSTALLER_FOR.get(job_type)
    if shared is not None:
        return NoInstaller(
            f"job type {job_type!r} has no installer of its own: it shares "
            f"{shared!r}'s env, so installing {shared!r} is what builds it",
            shared,
        )
    return NoInstaller(
        f"there is no installer for job type {job_type!r}; this build "
        f"installs {sorted(INSTALLABLE_JOB_TYPES)}",
        None,
    )


class EnvError(CrucibleError):
    ...


@dataclass(frozen=True)
class EnvSpec:

    job_type: str
    key: str
    recipe_name: str
    headline: str
    serving_stack: str | None = None
    python_version: str | None = None


def llm_env(backend_kind: str) -> EnvSpec:
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
    if backend_kind not in BACKEND_HEADLINE_PACKAGE:
        raise EnvError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_HEADLINE_PACKAGE)}"
        )
    if backend_kind == "cuda-linux":
        recipe_name = f"{narrator_engine}-{backend_kind}"
        return EnvSpec(
            job_type="tts",
            key=f"tts-{narrator_engine}",
            recipe_name=recipe_name,
            headline=NARRATOR_PACKAGE,
            serving_stack=CUDA_LINUX_SERVING_STACK.get(narrator_engine),
            python_version=RECIPE_PYTHON.get(recipe_name),
        )
    return EnvSpec(
        job_type="tts",
        key="tts",
        recipe_name=backend_kind,
        headline=NARRATOR_PACKAGE,
    )


def audio_env(engine: str, backend_kind: str) -> EnvSpec:
    if engine not in audio_engines_on(backend_kind):
        raise EnvError(
            f"no audio engine {engine!r} on {backend_kind!r}; the audio engines "
            f"there are {list(audio_engines_on(backend_kind))}"
        )
    recipe_name = f"{engine}-{backend_kind}"
    return EnvSpec(
        job_type=AUDIO_JOB_TYPE,
        key=f"{AUDIO_JOB_TYPE}-{engine}",
        recipe_name=recipe_name,
        headline=AUDIO_ENGINE_HEADLINE[engine],
        python_version=RECIPE_PYTHON.get(recipe_name),
    )


def audio_envs(backend_kind: str) -> tuple[EnvSpec, ...]:
    engines = audio_engines_on(backend_kind)
    if not engines:
        raise EnvError(
            f"audio has no engine on {backend_kind!r}; it runs on "
            f"{sorted(k for k in (CUDA_LINUX, MLX_DARWIN) if audio_engines_on(k))}"
        )
    return tuple(audio_env(engine, backend_kind) for engine in engines)


def video_env(engine: str, backend_kind: str) -> EnvSpec:
    if engine not in video_engines_on(backend_kind):
        raise EnvError(
            f"no video engine {engine!r} on {backend_kind!r}; the video engines "
            f"there are {list(video_engines_on(backend_kind))}"
        )
    recipe_name = f"{engine}-{backend_kind}"
    return EnvSpec(
        job_type=VIDEO_JOB_TYPE,
        key=f"{VIDEO_JOB_TYPE}-{engine}",
        recipe_name=recipe_name,
        headline=VIDEO_ENGINE_HEADLINE[engine],
        python_version=RECIPE_PYTHON.get(recipe_name),
    )


def video_envs(backend_kind: str) -> tuple[EnvSpec, ...]:
    engines = video_engines_on(backend_kind)
    if not engines:
        raise EnvError(
            f"video has no engine on {backend_kind!r}; it runs on "
            f"{sorted(k for k in (CUDA_LINUX, MLX_DARWIN) if video_engines_on(k))} "
            "only (LTX-2.5 needs a CUDA card; docs/internals/video.md, \"Backends\")"
        )
    return tuple(video_env(engine, backend_kind) for engine in engines)


def worker_env(job_type: str, backend_kind: str) -> EnvSpec:
    headline = WORKER_HEADLINE_PACKAGE.get((job_type, backend_kind))
    if headline is None:
        raise EnvError(
            f"no {job_type!r} env on {backend_kind!r}; this build has worker "
            f"envs for {sorted(WORKER_HEADLINE_PACKAGE)}"
        )
    return EnvSpec(
        job_type=job_type,
        key=job_type,
        recipe_name=backend_kind,
        headline=headline,
    )


@dataclass(frozen=True)
class EnvStatus:

    installed: bool
    path: Path
    detail: str
    python_version: str | None
    packages: dict[str, str]
    environment_sha256: str | None = None
    direct_references: dict[str, str] | None = None
    recipe_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
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


def env_dir(home: Path, spec: EnvSpec) -> Path:
    return home / "envs" / spec.key


def env_python(home: Path, spec: EnvSpec) -> Path:
    return env_dir(home, spec) / "bin" / "python"


def stamp_path(home: Path, spec: EnvSpec) -> Path:
    return env_dir(home, spec) / ENV_STAMP_NAME


def recipes_dir(job_type: str) -> Path:
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
    path = Path(__file__).resolve().parent / "envs" / job_type
    if not path.is_dir():
        raise EnvError(
            f"no {job_type} env recipes at {path}; they are package data and "
            f"this install has lost them, or ${RECIPES_DIR_ENV} must point at them"
        )
    return path


DEFAULT_INDEX_URL = "https://pypi.org/simple"

HF_ENDPOINT_ENV = "HF_ENDPOINT"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

_INDEX_OPTION = re.compile(
    r"^(?:--index-url|--extra-index-url|-f|--find-links)[=\s]+(?P<url>\S+)$"
)


def recipe_roots() -> list[Path]:
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        root = Path(override).expanduser()
    else:
        root = Path(__file__).resolve().parent / "envs"
    if not root.is_dir():
        raise EnvError(
            f"no env recipes at {root}; they are package data and this install "
            f"has lost them, or ${RECIPES_DIR_ENV} must point at them"
        )
    return sorted(path for path in root.iterdir() if path.is_dir())


def _index_urls_in(text: str) -> Iterator[str]:
    for line in _option_lines(text):
        found = _INDEX_OPTION.match(line)
        if found is not None:
            yield found.group("url")


def _recipe_index_options() -> Iterator[str]:
    for directory in recipe_roots():
        for recipe in sorted(directory.glob("*.txt")):
            yield from _index_urls_in(recipe_text(recipe))


def recipe_index_urls() -> list[str]:
    urls = [DEFAULT_INDEX_URL]
    hub = os.environ.get(HF_ENDPOINT_ENV) or DEFAULT_HF_ENDPOINT
    for url in (*_recipe_index_options(), hub):
        if url not in urls:
            urls.append(url)
    return urls


def recipe_for(spec: EnvSpec) -> Path:
    root = recipes_dir(spec.job_type)
    path = root / f"{spec.recipe_name}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.txt"))
        raise EnvError(
            f"no {spec.job_type} env recipe for {spec.recipe_name!r} at {path}; "
            f"this build ships recipes for {available}"
        )
    return path


_DIRECT_REFERENCE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*@\s*(?P<url>\S+)\s*$"
)

_VCS_COMMIT = re.compile(r"@(?P<sha>[0-9a-f]{40})(?:#|$)")

_ARCHIVE_DIGEST = re.compile(r"#sha256=(?P<sha>[0-9a-f]{64})$")


def _requirement_lines(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        yield stripped


def recipe_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for stripped in _requirement_lines(path):
        if _DIRECT_REFERENCE.match(stripped):
            continue
        name, separator, version = stripped.partition("==")
        if separator != "==":
            raise EnvError(
                f"{path.name}: {stripped!r} is not a `name==version` pin or a "
                "`name @ url` direct reference; every requirement in a recipe is "
                "pinned exactly"
            )
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


_ARCHIVE_BYTES = re.compile(
    r"^#\s*archive-bytes:\s*(?P<bytes>\d+)(?P<citation>\s+\S.*?)?\s*$"
)


def recipe_archive_bytes(path: Path) -> int:
    found: list[re.Match[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ARCHIVE_BYTES.match(line.strip())
        if match is not None:
            found.append(match)
    if not found:
        raise EnvError(
            f"recipe_unsized: {path.name} carries no `# archive-bytes:` line, so "
            "there is no telling what installing it needs. The measured sizes "
            "are docs/internals/config-envs-weights.md's \"Archive sizes\" table and they go "
            "in the recipe's header; guessing one here would be a number "
            "somebody later believes"
        )
    if len(found) > 1:
        raise EnvError(
            f"recipe_size_invalid: {path.name} carries {len(found)} "
            "`# archive-bytes:` lines. One recipe, one size, one owner"
        )
    match = found[0]
    if match.group("citation") is None:
        raise EnvError(
            f"recipe_size_invalid: {path.name}'s `# archive-bytes:` line names "
            "no source. Where a number was measured is part of the number: "
            "write it after the integer"
        )
    size = int(match.group("bytes"))
    if size <= 0:
        raise EnvError(
            f"recipe_size_invalid: {path.name} states an archive size of {size} "
            "bytes, which no env has ever weighed"
        )
    return size


def _filesystem_of(directory: Path) -> Path:
    path = directory.expanduser().absolute()
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    raise EnvError(
        f"env_disk_unreadable: no existing directory above {directory}, so the "
        "free space where this env would land cannot be measured"
    )


def refuse_without_room(*, job_type: str, recipe: Path, directory: Path) -> None:
    required = recipe_archive_bytes(recipe)
    filesystem = _filesystem_of(directory)
    free = shutil.disk_usage(filesystem).free
    if free >= required:
        return
    raise EnvError(
        f"env_disk: installing {job_type!r} needs at least "
        f"{required / 1_000_000_000:.1f} GB free and {filesystem} has "
        f"{free / 1_000_000_000:.1f} GB ({free} bytes of the {required} "
        f"{recipe.name} states). That figure is the ARCHIVE size measured "
        "for this recipe (docs/internals/config-envs-weights.md, \"Archive sizes\") and an unpacked env is larger, so it is a "
        "floor and not an estimate. Free space on that drive, or move "
        "$CRUCIBLE_HOME to one that has it, before running this again"
    )


def recipe_direct_references(path: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for stripped in _requirement_lines(path):
        match = _DIRECT_REFERENCE.match(stripped)
        if match is None:
            continue
        commit = _VCS_COMMIT.search(match.group("url")) or _ARCHIVE_DIGEST.search(
            match.group("url")
        )
        if commit is None:
            raise EnvError(
                f"{path.name}: {stripped!r} names no commit. A direct reference "
                "is pinned by `@<40-character sha>` before any `#fragment`, or, "
                "for a wheel URL, by a `#sha256=<64 hex digits>` fragment; a "
                "branch name is not a pin"
            )
        name = match.group("name").strip().lower().replace("_", "-")
        references[name] = commit.group("sha")
    return references


def installed_direct_references(home: Path, spec: EnvSpec) -> dict[str, str]:
    root = env_dir(home, spec) / "lib"
    found: dict[str, str] = {}
    if not root.is_dir():
        return found
    for record in root.glob("python*/site-packages/*.dist-info/direct_url.json"):
        try:
            document = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvError(f"could not read {record}: {exc}") from None
        commit = document.get("vcs_info", {}).get("commit_id") or (
            document.get("archive_info", {}).get("hashes", {}).get("sha256")
        )
        if not commit:
            continue
        name = record.parent.name.split("-")[0].lower().replace("_", "-")
        found[name] = commit
    return found


def installed_packages(home: Path, spec: EnvSpec) -> dict[str, str]:
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


@dataclass(frozen=True)
class _EnvFindings:

    installed: bool
    detail: str
    stamped: dict[str, Any] | None = None
    packages: dict[str, str] | None = None


def _stamped(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_sha256": environment_digest(record["recipe_text"]),
        "direct_references": record["direct_references"],
        "recipe_text": record["recipe_text"],
        "python_version": record["python_version"],
    }


def _pin_drift(pins: dict[str, str], present: dict[str, str]) -> list[str]:
    return sorted(
        f"{name} is {present.get(name, 'absent')}, recipe pins {version}"
        for name, version in pins.items()
        if present.get(name) != version
    )


def _reference_drift(pinned: dict[str, str], built_from: dict[str, str]) -> list[str]:
    return sorted(
        f"{name} was installed from "
        f"{built_from.get(name, 'no recorded commit')}, recipe pins {commit}"
        for name, commit in pinned.items()
        if built_from.get(name) != commit
    )


def _headline_words(
    spec: EnvSpec, directory: Path, recipe: Path,
    pinned_references: dict[str, str], present: dict[str, str],
) -> str:
    if spec.headline in pinned_references:
        return f"{spec.headline} @ {pinned_references[spec.headline][:12]}"
    if spec.headline in present:
        return f"{spec.headline} {present[spec.headline]}"
    raise EnvError(
        f"{directory} matches {recipe.name}, but {spec.headline!r} — the "
        f"package the {spec.key} env exists for — is not installed in it. "
        "Either the recipe no longer installs it or the headline names the "
        "wrong thing; both are bugs in this build, not in the env."
    )


def _contents_findings(
    home: Path, spec: EnvSpec, directory: Path, stamped: dict[str, Any]
) -> _EnvFindings:
    present = installed_packages(home, spec)
    recipe = recipe_for(spec)
    wrong = _pin_drift(recipe_pins(recipe), present)
    built_from = installed_direct_references(home, spec)
    pinned_references = recipe_direct_references(recipe)
    wrong += _reference_drift(pinned_references, built_from)
    if wrong:
        detail = f"{directory} does not match {recipe.name}: " + "; ".join(wrong)
        return _EnvFindings(False, detail, stamped, present)
    headline = _headline_words(spec, directory, recipe, pinned_references, present)
    detail = (
        f"{headline}, python {stamped['python_version']}, "
        f"{len(present)} packages"
    )
    return _EnvFindings(True, detail, stamped, present)


def _env_findings(home: Path, spec: EnvSpec, backend_kind: str) -> _EnvFindings:
    directory = env_dir(home, spec)
    install = f"crucible install {spec.job_type}"
    if not env_python(home, spec).is_file():
        return _EnvFindings(False, f"no venv at {directory} — run `{install}`")
    stamp = stamp_path(home, spec)
    if not stamp.is_file():
        return _EnvFindings(False, (
            f"{directory} exists but {stamp.name} does not: the last "
            f"`{install}` did not finish. Re-run it."
        ))
    record = _read_stamp(stamp)
    if record is None:
        return _EnvFindings(False, (
            f"{stamp} was written by an older Crucible and does not say what "
            f"this env was built from — run `{install}`"
        ))
    stamped = _stamped(record)
    if record["backend"] != backend_kind:
        return _EnvFindings(False, (
            f"{directory} was installed for backend {record['backend']!r}, this "
            f"host is {backend_kind!r} — run `{install} --force`"
        ), stamped)
    return _contents_findings(home, spec, directory, stamped)


def env_status(home: Path, spec: EnvSpec, backend_kind: str) -> EnvStatus:
    found = _env_findings(home, spec, backend_kind)
    stamped = found.stamped or {}
    return EnvStatus(
        installed=found.installed,
        path=env_dir(home, spec),
        detail=found.detail,
        python_version=stamped.get("python_version"),
        packages={} if found.packages is None else found.packages,
        environment_sha256=stamped.get("environment_sha256"),
        direct_references=stamped.get("direct_references"),
        recipe_text=stamped.get("recipe_text"),
    )


def require_env(home: Path, spec: EnvSpec, backend_kind: str) -> Path:
    status = env_status(home, spec, backend_kind)
    if not status.installed:
        raise EnvError(status.detail)
    return env_python(home, spec)


def interpreter_for(
    spec: EnvSpec,
    home: Path,
    backend_kind: str,
    *,
    on_line: Any = None,
    on_progress: Any = None,
) -> str:
    wanted = spec.python_version
    if wanted is None:
        return sys.executable
    running = ".".join(map(str, sys.version_info[:2]))
    if running == wanted:
        return sys.executable
    return str(
        interpreter.ensure_interpreter(
            home, backend_kind, wanted, on_line=on_line, on_progress=on_progress
        )
    )


PLAN_NOTHING = "nothing"
PLAN_REFERENCES = "narrator_sha_drift"
PLAN_RECIPE = "env_recipe_drift"
PLAN_BUILD = "build"


@dataclass(frozen=True)
class EnvPlan:

    action: str
    detail: str
    lines: tuple[str, ...] = ()


def _plan_before_the_stamp(
    directory: Path, stamp: Path, recipe: Path, force: bool, install_command: str
) -> EnvPlan | None:
    if force:
        return EnvPlan(
            PLAN_BUILD,
            f"--force: {directory} is deleted and built again from {recipe.name}",
        )
    if not directory.exists():
        return EnvPlan(PLAN_BUILD, f"there is no {directory}")
    if not stamp.is_file():
        return EnvPlan(
            PLAN_BUILD,
            f"{directory} exists but {stamp.name} does not: the last "
            f"`{install_command}` did not finish",
        )
    return None


def _refuse_what_pip_cannot_repair(
    record: dict[str, Any],
    directory: Path,
    recipe: Path,
    backend_kind: str,
    install_command: str,
) -> None:
    if record["backend"] != backend_kind:
        raise EnvError(
            f"{directory} was installed for backend {record['backend']!r} and "
            f"this host is {backend_kind!r}. A venv full of one backend's "
            "wheels is not re-pointed at another's by pip, so this is the one "
            f"drift that is genuinely a rebuild: `{install_command} --force`"
        )
    after = recipe_text(recipe)
    problems = unverifiable_recipe_changes(record["recipe_text"], after, recipe.name)
    if problems:
        raise EnvError(
            f"{directory} cannot be brought to {recipe.name} by pip: "
            + "; ".join(problems)
            + ". Neither of those is a change pip acting on this recipe would "
            "make: an index URL moves which wheel a pin resolves to while the "
            "pin itself stays satisfied, and a requirement that vanished stays "
            f"installed. `{install_command} --force` rebuilds it."
        )


def _moved_references_plan(
    recipe: Path, stamped: dict[str, Any], here: dict[str, Any]
) -> EnvPlan:
    moved = sorted(
        name for name in set(stamped) | set(here) if stamped.get(name) != here.get(name)
    )
    lines = tuple(
        line for line in _requirement_lines(recipe) if _reference_name(line) in moved
    )
    if len(lines) != len(moved):
        raise EnvError(
            f"{recipe.name} pins {moved} at commits this env was not built "
            f"from, and only {len(lines)} of those are lines in the file. A "
            "reference that moved must be a line this install can reinstall"
        )
    return EnvPlan(
        PLAN_REFERENCES,
        ", ".join(
            f"{name} {(stamped.get(name) or 'absent')[:12]} -> "
            f"{(here.get(name) or 'absent')[:12]}"
            for name in moved
        ),
        lines,
    )


def _recipe_drift(record: dict[str, Any], recipe: Path) -> EnvPlan | None:
    here_environment = environment_sha256(recipe)
    here_references = recipe_direct_references(recipe)
    stamped_environment = environment_digest(record["recipe_text"])
    stamped_references = record["direct_references"]
    if stamped_environment != here_environment:
        return EnvPlan(
            PLAN_RECIPE,
            f"{recipe.name}'s environment half moved "
            f"{stamped_environment[:12]} -> {here_environment[:12]}",
        )
    if stamped_references != here_references:
        return _moved_references_plan(recipe, stamped_references, here_references)
    return None


def plan_env(
    *,
    directory: Path,
    stamp: Path,
    recipe: Path,
    backend_kind: str,
    installed: bool,
    force: bool,
    install_command: str,
) -> EnvPlan:
    early = _plan_before_the_stamp(directory, stamp, recipe, force, install_command)
    if early is not None:
        return early
    record = _read_stamp(stamp)
    if record is None:
        return EnvPlan(
            PLAN_BUILD,
            f"{stamp} was written by an older Crucible and does not say what "
            f"{directory} was built from",
        )
    _refuse_what_pip_cannot_repair(record, directory, recipe, backend_kind, install_command)
    drift = _recipe_drift(record, recipe)
    if drift is not None:
        return drift
    if not installed:
        return EnvPlan(
            PLAN_RECIPE, f"{directory} does not hold what {recipe.name} pins"
        )
    return EnvPlan(PLAN_NOTHING, f"{directory} is what {recipe.name} says")


_STAMP_KEYS = (
    "backend",
    "environment_sha256",
    "direct_references",
    "recipe_text",
    "python_version",
)


def _read_stamp(stamp: Path) -> dict[str, Any] | None:
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if not all(key in record for key in _STAMP_KEYS):
        return None
    return record


def _reference_name(line: str) -> str | None:
    match = _DIRECT_REFERENCE.match(line)
    if match is None:
        return None
    return match.group("name").strip().lower().replace("_", "-")


def plan_install(
    home: Path, spec: EnvSpec, backend_kind: str, *, force: bool = False
) -> EnvPlan:
    return plan_env(
        directory=env_dir(home, spec),
        stamp=stamp_path(home, spec),
        recipe=recipe_for(spec),
        backend_kind=backend_kind,
        installed=env_status(home, spec, backend_kind).installed,
        force=force,
        install_command=f"crucible install {spec.job_type}",
    )


def _build_venv(
    home: Path, spec: EnvSpec, backend_kind: str, recipe: Path, on_line: Any
) -> Path:
    directory = env_dir(home, spec)
    refuse_without_room(
        job_type=spec.job_type, recipe=recipe, directory=directory
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        shutil.rmtree(directory)
    _run(
        [
            interpreter_for(spec, home, backend_kind, on_line=on_line),
            "-m",
            "venv",
            str(directory),
        ],
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
    return python


def _stamped_python_version(home: Path, spec: EnvSpec) -> str | None:
    return json.loads(
        stamp_path(home, spec).read_text(encoding="utf-8")
    )["python_version"]


def _reinstall_references(
    python: Path, lines: tuple[str, ...], directory: Path, on_line: Any
) -> None:
    for line in lines:
        _run(
            [
                str(python), "-m", "pip", "install",
                "--no-deps", "--force-reinstall", line,
            ],
            f"could not reinstall {line} into {directory}",
            on_line,
        )


def _install_recipe(
    spec: EnvSpec, backend_kind: str, python: Path, recipe: Path,
    directory: Path, on_line: Any,
) -> None:
    _run(
        [str(python), "-m", "pip", "install", "-r", str(recipe)],
        f"could not install {recipe} into {directory}",
        on_line,
    )
    try:
        envpatches.apply(
            spec.job_type, directory, python, recipe_pins(recipe), on_line=on_line
        )
        if spec.job_type == "tts" and backend_kind == "cuda-linux":
            envpatches.ensure_cuda_toolkit_links(directory, on_line=on_line)
    except envpatches.PatchError as exc:
        raise EnvError(str(exc)) from exc


def _python_version_of(python: Path) -> str:
    return subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print('.'.join(map(str, sys.version_info[:3])))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def install_env(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Any = None,
) -> EnvStatus:
    recipe = recipe_for(spec)
    directory = env_dir(home, spec)
    plan = plan_install(home, spec, backend_kind, force=force)
    if plan.action == PLAN_NOTHING:
        return env_status(home, spec, backend_kind)
    if on_line is not None:
        on_line(f"{plan.action}: {plan.detail}")
    started = time.monotonic()
    python_version: str | None = None
    if plan.action == PLAN_BUILD:
        python = _build_venv(home, spec, backend_kind, recipe, on_line)
    else:
        python = env_python(home, spec)
        python_version = _stamped_python_version(home, spec)
    if plan.action == PLAN_REFERENCES:
        _reinstall_references(python, plan.lines, directory, on_line)
    else:
        _install_recipe(spec, backend_kind, python, recipe, directory, on_line)
    if python_version is None:
        python_version = _python_version_of(python)
    _write_stamp(
        home, spec, backend_kind,
        recipe=recipe,
        python_version=python_version,
        references=recipe_direct_references(recipe),
        seconds=round(time.monotonic() - started, 1),
    )
    return env_status(home, spec, backend_kind)


_CRLF = b"\r\n"
_LF = b"\n"


def recipe_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(_CRLF, _LF)).hexdigest()


def environment_sha256(path: Path) -> str:
    return environment_digest(recipe_text(path))


def environment_digest(text: str) -> str:
    kept = []
    for line in text.splitlines():
        content = line.split(" #", 1)[0].strip()
        if not content or content.startswith("#"):
            continue
        if _DIRECT_REFERENCE.match(content) is None:
            kept.append(content)
    body = "".join(line + "\n" for line in kept)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _write_stamp(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    recipe: Path,
    python_version: str | None,
    references: dict[str, str],
    seconds: float | None,
) -> None:
    stamp_path(home, spec).write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": recipe.name,
                "environment_sha256": environment_sha256(recipe),
                "direct_references": dict(references),
                "recipe_text": recipe_text(recipe),
                "python_version": python_version,
                "seconds": seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def recipe_text(path: Path) -> str:
    return path.read_bytes().replace(_CRLF, _LF).decode("utf-8")


def _option_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("-")
    ]


def unverifiable_recipe_changes(before: str, after: str, recipe_name: str) -> list[str]:
    problems: list[str] = []

    was, now = _option_lines(before), _option_lines(after)
    if was != now:
        for line in [x for x in was if x not in now]:
            problems.append(f"{recipe_name} no longer says {line!r}")
        for line in [x for x in now if x not in was]:
            problems.append(f"{recipe_name} now says {line!r}, and it did not")

    gone = sorted(_names_in(before) - _names_in(after))
    for name in gone:
        problems.append(
            f"{recipe_name} no longer requires {name!r}, which is still installed"
        )
    return problems


def _names_in(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        match = _DIRECT_REFERENCE.match(stripped)
        raw = match.group("name") if match else stripped.partition("==")[0]
        found.add(raw.strip().lower().replace("_", "-"))
    return found


_HOST_COMPILERS: tuple[tuple[str, str], ...] = (("gcc", "g++"), ("cc", "c++"), ("clang", "clang++"))


def build_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    import sysconfig

    environment = dict(os.environ if base is None else base)
    if "CC" in environment:
        return environment
    recorded = (sysconfig.get_config_var("CC") or "").split()
    if recorded and shutil.which(recorded[0]) is not None:
        return environment
    for c_compiler, cpp_compiler in _HOST_COMPILERS:
        if shutil.which(c_compiler) is not None:
            environment["CC"] = c_compiler
            environment["CXX"] = cpp_compiler if shutil.which(cpp_compiler) else c_compiler
            environment["LDSHARED"] = f"{c_compiler} -pthread -shared"
            return environment
    return environment


class PipFailure:

    _COLLECTING = re.compile(r"^\s*Collecting (?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;]+)")
    _BUILDING = re.compile(r"Building wheel for (?P<name>[A-Za-z0-9._-]+) \(")
    _FAILED_BUILD = re.compile(
        r"(?:Failed building wheel for|Failed to build) (?!installable )'?(?P<name>[A-Za-z0-9._-]+)'?"
    )
    _NO_COMPILER = re.compile(
        r"No such file or directory: '(?P<cc>clang\+\+|clang|gcc|g\+\+|cc|c\+\+)'"
        r"|(?:unable to execute|command) '(?P<cc2>[^']+)'(?: failed)?: No such file"
    )
    _NO_HEADER = re.compile(r"fatal error: (?P<header>[\w./-]+\.h): No such file")
    _NO_MATCH = re.compile(r"No matching distribution found for (?P<req>\S+)")
    _CONFLICT = re.compile(r"Cannot install (?P<what>.+?) because these package versions")
    _NETWORK = re.compile(
        r"Temporary failure in name resolution|Name or service not known|"
        r"Network is unreachable|Max retries exceeded|Could not fetch URL|ConnectTimeout"
    )
    _NO_GIT = re.compile(r"Cannot find command 'git'|No such file or directory: 'git'")

    def __init__(self) -> None:
        self.versions: dict[str, str] = {}
        self.building: str | None = None
        self.failed_build: str | None = None
        self.build_reason: str | None = None
        self.no_match: str | None = None
        self.conflict: str | None = None
        self.hash_mismatch = False
        self.network = False
        self.disk_full = False
        self.no_git = False
        self.last_error: str | None = None

    def feed(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        found = self._COLLECTING.search(text)
        if found:
            self.versions[found["name"].lower()] = found["version"]
        found = self._BUILDING.search(text)
        if found and self.failed_build is None:
            self.building = found["name"]
        found = self._FAILED_BUILD.search(text)
        if found and self.failed_build is None:
            self.failed_build = found["name"]
        found = self._NO_COMPILER.search(text)
        if found and self.build_reason is None:
            self.build_reason = (
                "there is no C compiler on this host (it asked for "
                f"{found['cc'] or found['cc2']})"
            )
        found = self._NO_HEADER.search(text)
        if found and self.build_reason is None:
            self.build_reason = f"a C header it needs is missing ({found['header']})"
        found = self._NO_MATCH.search(text)
        if found and self.no_match is None:
            self.no_match = found["req"]
        found = self._CONFLICT.search(text)
        if found and self.conflict is None:
            self.conflict = found["what"]
        if "DO NOT MATCH THE HASHES" in text:
            self.hash_mismatch = True
        if self._NETWORK.search(text):
            self.network = True
        if "No space left on device" in text:
            self.disk_full = True
        if self._NO_GIT.search(text):
            self.no_git = True
        if (text.startswith("ERROR:") or text.startswith("error:")) and not (
            "subprocess-exited-with-error" in text or "Failed" in text
        ):
            self.last_error = text

    def _named(self, name: str) -> str:
        version = self.versions.get(name.lower())
        return f"{name} {version}" if version else name

    def summary(self) -> str | None:
        if self.disk_full:
            return "the disk filled up while pip was installing (No space left on device)"
        if self.no_git:
            return (
                "git is not installed on this host, and the recipe installs a "
                "package from a git commit"
            )
        if self.failed_build is not None or (self.build_reason and self.building):
            name = self._named(self.failed_build or self.building or "a package")
            return f"{name} did not build: " + (
                self.build_reason or (self.last_error or "its build step failed")
            )
        if self.no_match is not None:
            if self.network:
                return (
                    f"{self.no_match} could not be downloaded: the package index "
                    "could not be reached (a network failure)"
                )
            return f"{self.no_match} is not on the package index (no matching distribution)"
        if self.conflict is not None:
            return f"the recipe's pins conflict: {self.conflict} cannot be installed together"
        if self.hash_mismatch:
            return "a downloaded package did not match the hash pip expected"
        if self.network:
            return "the package index could not be reached (a network failure)"
        return self.last_error


def _run(command: list[str], failure: str, on_line: Any) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_environment(),
    )
    tail: list[str] = []
    watch = PipFailure()
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        del tail[:-40]
        watch.feed(line)
        if on_line is not None:
            on_line(line)
    code = process.wait()
    if code != 0:
        raise EnvError(failure_message(failure, command, code, watch, tail))


def failure_message(
    failure: str, command: list[str], code: int, watch: PipFailure, tail: list[str]
) -> str:
    said = watch.summary()
    head = f"{failure}: {said}" if said else failure
    return f"{head}\n`{' '.join(command)}` exited {code}\n" + "\n".join(tail)
