from __future__ import annotations

import argparse
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from .. import capabilitywords, hosttools, interpreter, jobenv, llamacpp, verdict
from ..backend import LLAMA_WINDOWS, Backend
from ..config import Config
from ..jobenv import INSTALLABLE_JOB_TYPES, SMOKE_IMPORT
from ..narratorengines import NARRATOR_ENGINE_SAMPLING
from . import common
from .capability import _capability_step, _measure_step
from .common import _env_spec, _fail

PROGRESS_SECONDS = 20.0

_COLLECTING = re.compile(r"^Collecting (?P<name>[A-Za-z0-9._-]+)")
_BUILDING = re.compile(r"^\s*Building wheel for (?P<name>[A-Za-z0-9._-]+)")
_INSTALLING = "Installing collected packages: "


class EnvProgress:
    """What an env build is doing, said while pip says nothing to this terminal.

    pip runs for minutes (132 s and 166 s for the two audio envs on 2026-10-08), and
    without --verbose its lines are kept back, so the build was silent between
    "target:" and "installed in". This reads the lines it is fed and says, every
    PROGRESS_SECONDS, how long it has been and where pip is: how many packages it has
    collected, which wheel it is building, how many it is installing. It speaks on its
    own clock, so a wheel that takes minutes to build is still accounted for.
    """

    def __init__(
        self,
        say: Callable[[str], None],
        *,
        every: float = PROGRESS_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._say = say
        self._every = every
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.collected = 0
        self.building: str | None = None
        self.installing: int | None = None
        self.finished = False

    def feed(self, line: str) -> None:
        announce: str | None = None
        with self._lock:
            if _COLLECTING.match(line):
                self.collected += 1
                self.building = None
                self.installing = None
                self.finished = False
            elif line.startswith("Successfully installed"):
                self.installing = None
                self.finished = True
            elif (found := _BUILDING.match(line)) is not None:
                self.building = None if "finished with status" in line else found.group("name")
            elif line.startswith(_INSTALLING):
                self.building = None
                self.installing = len(
                    [name for name in line[len(_INSTALLING):].split(",") if name.strip()]
                )
                announce = f"  installing {self.installing} packages"
        if announce is not None:
            self._say(announce)

    def sentence(self) -> str:
        with self._lock:
            if self.installing is not None:
                return f"installing {self.installing} packages"
            if self.building is not None:
                return f"building the wheel for {self.building}"
            if self.finished:
                return "pip has finished; finishing the env"
            if self.collected:
                return f"{self.collected} packages collected so far"
            return "setting up the venv and pip"

    def tick(self) -> None:
        elapsed = self._clock() - self._started
        self._say(f"  still installing, {elapsed:.0f} s: {self.sentence()}")

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            self.tick()

    def __enter__(self) -> "EnvProgress":
        self._thread = threading.Thread(target=self._run, name="crucible-env-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


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
    def say(line: str) -> None:
        print(line, flush=True)

    def on_line(line: str) -> None:
        progress.feed(line)
        if args.verbose:
            say(f"  {line}")

    try:
        with EnvProgress(say) as progress:
            status = jobenv.install_env(
                config.home, spec, backend.kind, force=args.force, on_line=on_line
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
    refusal = _ensure_tools(config, args)
    if refusal is not None:
        return _fail(refusal)
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
    if hosttools.needs_c_compiler(config.home):
        try:
            print(
                hosttools.ensure_zig(
                    config.home,
                    on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
                )
            )
        except hosttools.HostToolError as exc:
            return (
                f"the env is installed, but Crucible's C compiler is not: {exc}. "
                "Triton in it compiles C the first time a kernel runs, and its "
                "engine is refused until the compiler is placed. Installing again "
                "retries only the compiler"
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
            "this backend runs (Stable Audio 3; YuE2 on cuda-linux). 'segment' "
            "builds one env that runs both segment models (BiRefNet and SAM 2.1). "
            "'video' builds the LTX-2.5 env (diffusers on cuda-linux, ltx-2-mlx on "
            "mlx-darwin) and places the ffmpeg that muxes its clips"
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
