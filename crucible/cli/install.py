from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from .. import capabilitywords, hosttools, interpreter, jobenv, llamacpp, verdict
from ..backend import Backend, LLAMA_WINDOWS
from ..config import Config
from ..jobenv import INSTALLABLE_JOB_TYPES, INSTALLER_FOR, SMOKE_IMPORT
from ..narratorengines import NARRATOR_ENGINE_SAMPLING
from ..jobenv import INSTALLABLE_JOB_TYPES, SMOKE_IMPORT
from ..voices import NARRATOR_ENGINE_SAMPLING
from . import common
from .capability import _capability_step, _measure_step
from .common import _env_spec, _fail


def _smoke_import(python: Path, key: str, backend_kind: str) -> str | None:
    module = SMOKE_IMPORT.get(key, {}).get(backend_kind)
    if module is None:
        return (
            f"there is no smoke import recorded for the {key!r} env on "
            f"{backend_kind}; crucible/jobenv.py's SMOKE_IMPORT is the owner of "
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
    missing = jobenv.no_installer(args.job_type)
    if missing is not None:
        if missing.shared_with is not None:
            return _fail(f"{missing.words}. Run `crucible install {missing.shared_with}`")
        return _fail(missing.words)
    config, backend = common.here()
    if backend.kind == LLAMA_WINDOWS:
        return _install_llama_windows(config, backend, args)
    if args.job_type == jobenv.AUDIO_JOB_TYPE:
        return _install_audio(config, backend, args)
    if args.job_type == jobenv.VIDEO_JOB_TYPE:
        return _install_video(config, backend, args)
    try:
        spec = _env_spec(args.job_type, args.narrator_engine, backend.kind)
    except jobenv.EnvError as exc:
        return _fail(str(exc))
    refusal = _build_env(config, backend, spec, args)
    if refusal is not None:
        return _fail(refusal)
    refusal = _ensure_tools(config, args)
    if refusal is not None:
        return _fail(refusal)
    _measure_step(config, backend, gpu=not args.no_gpu_measure)
    return _capability_step(
        config,
        backend,
        *jobenv.JOB_TYPES_SERVED_BY_ENV.get(args.job_type, (args.job_type,)),
    )


def _build_env(
    config: Config, backend: Backend, spec: jobenv.EnvSpec, args: argparse.Namespace
) -> str | None:
    try:
        recipe = jobenv.recipe_for(spec)
    except jobenv.EnvError as exc:
        return str(exc)
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
        return str(exc)
    elapsed = time.monotonic() - started
    if not status.installed:
        return f"the env did not come out installed: {status.detail}"
    refusal = _smoke_import(
        jobenv.env_python(config.home, spec), spec.key, backend.kind
    )
    if refusal is not None:
        return refusal
    print(f"installed in {elapsed:.0f}s: {status.detail}")
    for name in sorted(status.packages):
        if name in (
            spec.headline, "torch", "numpy", "transformers", "mlx",
            "ctranslate2", "onnxruntime",
        ):
            print(f"  {name}=={status.packages[name]}")
    return None


def _install_audio(config: Config, backend: Backend, args: argparse.Namespace) -> int:
    if args.narrator_engine is not None:
        return _fail(
            "--narrator-engine names which tts env to build and means nothing for "
            "'audio'; run `crucible install audio` without it"
        )
    try:
        specs = jobenv.audio_envs(backend.kind)
    except jobenv.EnvError as exc:
        return _fail(str(exc))
    for spec in specs:
        print(f"audio engine: {spec.key.removeprefix('audio-')}")
        refusal = _build_env(config, backend, spec, args)
        if refusal is not None:
            return _fail(
                f"{refusal}. The audio envs already built are kept; running "
                "`crucible install audio` again builds only what is missing"
            )
    _measure_step(config, backend, gpu=not args.no_gpu_measure)
    return _capability_step(config, backend, jobenv.AUDIO_JOB_TYPE)


def _install_video(config: Config, backend: Backend, args: argparse.Namespace) -> int:
    if args.narrator_engine is not None:
        return _fail(
            "--narrator-engine names which tts env to build and means nothing for "
            "'video'; run `crucible install video` without it"
        )
    try:
        specs = jobenv.video_envs(backend.kind)
    except jobenv.EnvError as exc:
        return _fail(str(exc))
    for spec in specs:
        print(f"video engine: {spec.key.removeprefix('video-')}")
        refusal = _build_env(config, backend, spec, args)
        if refusal is not None:
            return _fail(refusal)
    refusal = _ensure_tools(config, args)
    if refusal is not None:
        return _fail(refusal)
    _measure_step(config, backend, gpu=not args.no_gpu_measure)
    return _capability_step(config, backend, jobenv.VIDEO_JOB_TYPE)


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
    if args.job_type in verdict.WSL_ONLY_JOB_TYPES:
        return _fail(
            f"needs_wsl: {args.job_type} — {capabilitywords.NEEDS_WSL_REASON}. "
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
            "holds the torch it wants). 'audio' builds one env per audio engine "
            "this backend runs (Stable Audio 3; YuE2 on cuda-linux). 'video' "
            "builds the LTX-2.5 env (cuda-linux only) and places the ffmpeg that "
            "muxes its clips"
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
