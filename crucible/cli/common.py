from __future__ import annotations

import sys

from .. import jobenv
from ..backend import (
    Backend,
    CUDA_LINUX,
    LLAMA_WINDOWS,
    WINDOWS_REFUSAL,
    backend_not_here,
    detect_backend,
)
from ..config import Config, load_config
from ..errors import ConfigError, NoViableBackend
from ..voices import NARRATOR_ENGINE_SAMPLING


EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2


def _fail(message: str) -> int:
    print(f"crucible: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _backend_mismatch(recorded: str, backend: Backend) -> str:
    """The ONE sentence a recorded backend gets when it is not this host's.

    PHASE15-HOST.md section 3.5: *"a backend runs where its engine runs and
    nowhere else"* — `llama-windows` off win32 is as wrong as `cuda-linux` on
    it, and both are `backend_not_here`. `crucible init --backend`,
    `crucible serve` and `crucible service install` all print this, so there
    is one wording for one fact.

    `WINDOWS_REFUSAL` is appended for the ONE mismatch it still describes: a
    `cuda-linux` config found on a Windows host. That config is not wrong
    about wanting vLLM — it is wrong about where vLLM runs, which is inside
    the WSL2 guest — and that is worth saying once, here, where it is true.
    """
    sentence = backend_not_here(recorded, backend.kind, backend.platform)
    if recorded == CUDA_LINUX and backend.kind == LLAMA_WINDOWS:
        sentence = f"{sentence}. {WINDOWS_REFUSAL}"
    return f"backend_not_here: {sentence}"


def _env_spec(
    job_type: str, narrator_engine: str | None, backend_kind: str
) -> jobenv.EnvSpec:
    """The env `crucible install <job type>` builds on this host.

    `tts` needs a second word on `cuda-linux` — the env there is named per
    narrator engine — so the flag is required for it and refused for `llm`,
    rather than quietly ignored on the type that has only one env.
    """
    if job_type in jobenv.WORKER_JOB_TYPES:
        return jobenv.worker_env(job_type, backend_kind)
    if job_type == "llm":
        if narrator_engine is not None:
            raise jobenv.EnvError(
                "--narrator-engine names which tts env to build and means nothing "
                "for 'llm', which has exactly one env per host"
            )
        return jobenv.llm_env(backend_kind)
    if narrator_engine is None:
        raise jobenv.EnvError(
            "`crucible install tts` needs --narrator-engine "
            f"({' or '.join(sorted(NARRATOR_ENGINE_SAMPLING))}): on cuda-linux "
            "the env is named per narrator engine, because two of them cannot "
            "share a venv — each pins its own serving stack against its own "
            "torch — and there is no default"
        )
    if narrator_engine not in NARRATOR_ENGINE_SAMPLING:
        raise jobenv.EnvError(
            f"{narrator_engine!r} is not one of narrator's engines; they are "
            f"{sorted(NARRATOR_ENGINE_SAMPLING)}"
        )
    return jobenv.tts_env(narrator_engine, backend_kind)


def _models_config() -> tuple[Config, Backend] | int:
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    return config, backend
