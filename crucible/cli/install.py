from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from .. import capability, hosttools, interpreter, jobenv, llamacpp
from ..backend import Backend, CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from ..config import Config
from ..errors import ConfigError, NoViableBackend
from ..voices import NARRATOR_ENGINE_SAMPLING
from . import common
from .capability import _capability_step, _measure_step
from .common import _backend_mismatch, _env_spec, _fail


INSTALLABLE_JOB_TYPES = ("llm", "tts", *jobenv.WORKER_JOB_TYPES)

INSTALLER_FOR: dict[str, str] = {
    **{name: name for name in INSTALLABLE_JOB_TYPES},
    **{
        job_type: env
        for env, served in jobenv.JOB_TYPES_SERVED_BY_ENV.items()
        for job_type in served
    },
    "pages": "llm",
}

SMOKE_IMPORT: dict[str, dict[str, str]] = {
    "llm": {CUDA_LINUX: "vllm", MLX_DARWIN: "mlx_lm"},
    "asr": {CUDA_LINUX: "faster_whisper", MLX_DARWIN: "mlx_whisper"},
    "align": {CUDA_LINUX: "qwen_asr", MLX_DARWIN: "qwen_asr"},
    "rvc": {CUDA_LINUX: "ultimate_rvc", MLX_DARWIN: "ultimate_rvc"},
    "tts-higgs-v3": {CUDA_LINUX: "narrator"},
    "tts": {MLX_DARWIN: "narrator"},
}


def _smoke_import(python: Path, key: str, backend_kind: str) -> str | None:
    module = SMOKE_IMPORT.get(key, {}).get(backend_kind)
    if module is None:
        return (
            f"there is no smoke import recorded for the {key!r} env on "
            f"{backend_kind}; crucible/cli/install.py's SMOKE_IMPORT is the owner of "
            "that fact and an env nothing proved can be imported is not one "
            "this command will call installed"
        )
    completed = subprocess.run(
        [str(python), "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode == 0:
        return None
    said = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
    return (
        f"env_smoke_failed: the env installed but `import {module}` in it "
        f"exited {completed.returncode}: " + " / ".join(said[-3:] or ["no output"])
    )


def cmd_install(args: argparse.Namespace) -> int:
    if args.job_type not in INSTALLABLE_JOB_TYPES:
        shared = INSTALLER_FOR.get(args.job_type)
        if shared is not None:
            return _fail(
                f"job type {args.job_type!r} has no installer of its own: it "
                f"shares {shared!r}'s engine, so installing {shared!r} is what "
                f"builds it. Run `crucible install {shared}`"
            )
        return _fail(
            f"there is no installer for job type {args.job_type!r}; this build "
            f"installs {sorted(INSTALLABLE_JOB_TYPES)}"
        )
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = common.detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force`"
        )
    if backend.kind == LLAMA_WINDOWS:
        return _install_llama_windows(config, backend, args)
    try:
        spec = _env_spec(args.job_type, args.narrator_engine, backend.kind)
        recipe = jobenv.recipe_for(spec)
    except jobenv.EnvError as exc:
        return _fail(str(exc))
    print(f"backend: {backend.kind}")
    print(f"recipe:  {recipe}")
    print(f"target:  {jobenv.env_dir(config.home, spec)}")
    started = time.monotonic()
    try:
        status = jobenv.install_env(
            config.home,
            spec,
            backend.kind,
            force=args.force,
            on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
        )
    except (jobenv.EnvError, interpreter.InterpreterError) as exc:
        return _fail(str(exc))
    elapsed = time.monotonic() - started
    if not status.installed:
        return _fail(f"the env did not come out installed: {status.detail}")
    refusal = _smoke_import(
        jobenv.env_python(config.home, spec), spec.key, backend.kind
    )
    if refusal is not None:
        return _fail(refusal)
    print(f"installed in {elapsed:.0f}s: {status.detail}")
    for name in sorted(status.packages):
        if name in (
            spec.headline, "torch", "numpy", "transformers", "mlx",
            "ctranslate2", "onnxruntime",
        ):
            print(f"  {name}=={status.packages[name]}")
    refusal = _ensure_tools(config, args)
    if refusal is not None:
        return _fail(refusal)
    _measure_step(config, backend, gpu=not args.no_gpu_measure)
    return _capability_step(
        config,
        backend,
        *jobenv.JOB_TYPES_SERVED_BY_ENV.get(args.job_type, (args.job_type,)),
    )


def _ensure_tools(config: Config, args: argparse.Namespace) -> str | None:
    try:
        print(
            hosttools.ensure_ffmpeg(
                config.home,
                on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
            )
        )
    except hosttools.HostToolError as exc:
        return (
            f"the env is installed, but Crucible's ffmpeg is not: {exc}. "
            "Installing again retries only the ffmpeg"
        )
    try:
        print(hosttools.ensure_silero_vad(config.home))
    except hosttools.HostToolError as exc:
        print(
            f"speech detector: not placed ({exc}); an asr job that asks for "
            "speech_only will fetch it"
        )
    return None


def _install_llama_windows(
    config: Config, backend: Backend, args: argparse.Namespace
) -> int:
    if args.job_type in capability.WSL_ONLY_JOB_TYPES:
        return _fail(
            f"needs_wsl: {args.job_type} — {capability.NEEDS_WSL_REASON}. "
            f"The {LLAMA_WINDOWS} backend serves the llm classes and pages "
            "from llama.cpp; tts, asr, align, rvc and denoise are Python "
            "engines and run in the WSL2 guest"
        )
    if args.narrator_engine is not None:
        return _fail(
            "--narrator-engine names which tts env to build, and tts is not "
            f"served on {LLAMA_WINDOWS}"
        )
    build = llamacpp.build_for(backend.gpu.vendor)
    print(f"backend: {backend.kind} ({backend.gpu.name})")
    print(f"engine:  llama.cpp {llamacpp.LLAMA_CPP_RELEASE} ({build})")
    print(f"target:  {llamacpp.engine_dir(config)}")
    for asset in llamacpp.assets_for(build):
        print(f"  {asset.name}  {asset.bytes / 1e6:.0f} MB  sha256 {asset.sha256}")
    started = time.monotonic()
    try:
        found = llamacpp.pull(
            config,
            build,
            force=args.force,
            on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
        )
    except llamacpp.EngineSubjectError as exc:
        return _fail(str(exc))
    print(
        f"installed in {time.monotonic() - started:.0f}s: {found.path} "
        f"({found.bytes / 1e9:.2f} GB)"
    )
    _measure_step(config, backend, gpu=not args.no_gpu_measure)
    return _capability_step(config, backend, args.job_type)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    install = subparsers.add_parser(
        "install",
        help="build this job type's env from its recipe, with pip",
    )
    install.add_argument(
        "job_type",
        choices=sorted(INSTALLABLE_JOB_TYPES),
        help=(
            "the job type to install. 'rvc' also installs 'denoise', which "
            "shares its env (audio-separator is torch, and the rvc env already "
            "holds the torch it wants)"
        ),
    )
    install.add_argument(
        "--narrator-engine",
        default=None,
        choices=sorted(NARRATOR_ENGINE_SAMPLING),
        help=(
            "which tts env to build; required for 'tts' because cuda-linux "
            "names one venv per narrator engine, and refused for 'llm'"
        ),
    )
    install.add_argument(
        "--force",
        action="store_true",
        help=(
            "delete the env and build it again. The answer to a genuinely "
            "broken one, and the only thing that deletes an env: an ordinary "
            "run pips this recipe into the venv that is already there"
        ),
    )
    install.add_argument(
        "--verbose", action="store_true", help="echo pip's output line by line"
    )
    install.add_argument(
        "--no-gpu-measure",
        action="store_true",
        help=(
            "measure the card with nvidia-smi only, and leave the GPU rungs "
            "(torch, CUDA graphs, vLLM) for `crucible ladder` later"
        ),
    )
    install.set_defaults(func=cmd_install)
