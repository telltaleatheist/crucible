from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .. import API_HEADER, API_VERSION, jobenv
from ..backend import (
    Backend,
    CUDA_LINUX,
    LLAMA_WINDOWS,
    WINDOWS_REFUSAL,
    backend_not_here,
    detect_backend,
)
from ..config import Config, load_config
from ..errors import ConfigError, CrucibleError, NoViableBackend
from ..protocol import USER_AGENT_HEADER, user_agent
from ..narratorengines import NARRATOR_ENGINE_SAMPLING


EXIT_OK = 0
EXIT_REFUSED = 1

SERVER_PROBE_SECONDS = 2.0

REINIT_COMMAND = "crucible init --force"

DRIVER_HINTS: dict[str, str] = {
    "linux": (
        "Crucible on Linux needs an NVIDIA card and its driver. Install the "
        "NVIDIA driver on Windows if this is WSL (the guest sees it by itself, "
        "nothing is installed inside), or the distribution's NVIDIA driver "
        "package on a bare Linux host, then run `crucible doctor` again"
    ),
    "darwin": (
        "Crucible on a Mac means Apple silicon (arm64) with mlx importable "
        "by this Python; an Intel Mac is not a Crucible host"
    ),
    "win32": (
        "Crucible on Windows serves llama.cpp; with no NVIDIA driver it "
        "runs on the CPU, so this refusal means the host itself could not "
        "be read. `crucible doctor` shows what was found"
    ),
}


class Refusal(CrucibleError):
    ...


def _fail(message: str) -> int:
    print(f"crucible: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _mismatch_sentence(recorded: str, backend: Backend) -> str:
    sentence = backend_not_here(recorded, backend.kind, backend.platform)
    if recorded == CUDA_LINUX and backend.kind == LLAMA_WINDOWS:
        sentence = f"{sentence}. {WINDOWS_REFUSAL}"
    return sentence


def _backend_mismatch(recorded: str, backend: Backend) -> str:
    return f"backend_not_here: {_mismatch_sentence(recorded, backend)}"


def backend_changed_fix(config: Config, backend: Backend) -> str:
    return (
        f"{_mismatch_sentence(config.backend_kind, backend)} ({config.path}); "
        f"re-run `{REINIT_COMMAND}` on this host, which re-detects the backend "
        "and rewrites the config for it"
    )


def backend_hint() -> str:
    return DRIVER_HINTS.get(
        sys.platform,
        "supported hosts are Linux with an NVIDIA card and Apple silicon macOS",
    )


def no_viable_backend(exc: NoViableBackend) -> str:
    return f"no viable backend: {exc.reason}. {backend_hint()}"


def _load(home: Path | None, tolerate_stale_record: bool) -> Config:
    if tolerate_stale_record:
        return load_config(home, tolerate_stale_record=True)
    if home is not None:
        return load_config(home)
    return load_config()


def here(
    home: Path | None = None, *, tolerate_stale_record: bool = False
) -> tuple[Config, Backend]:
    try:
        config = _load(home, tolerate_stale_record)
    except ConfigError as exc:
        raise Refusal(str(exc)) from exc
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        raise Refusal(no_viable_backend(exc)) from exc
    if backend.kind != config.backend_kind:
        raise Refusal("backend_not_here: " + backend_changed_fix(config, backend))
    return config, backend


def loopback_url(config: Config) -> str:
    host = config.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{config.port}"


def server_here(config: Config, backend: Backend):
    from ..apiclient import Connection

    url = loopback_url(config)
    request = urllib.request.Request(
        url + "/v1/info",
        headers={
            "Authorization": f"Bearer {config.token}",
            API_HEADER: str(API_VERSION),
            USER_AGENT_HEADER: user_agent("cli"),
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=SERVER_PROBE_SECONDS) as response:
            info = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    if not isinstance(info, dict):
        return None
    server = info.get("server")
    host = info.get("host")
    if not isinstance(server, dict) or not isinstance(host, dict):
        return None
    if server.get("name") != config.name or host.get("backend") != backend.kind:
        return None
    return Connection(url=url, token=config.token, name=config.name, source="local")


def _env_spec(
    job_type: str, narrator_engine: str | None, backend_kind: str
) -> jobenv.EnvSpec:
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
