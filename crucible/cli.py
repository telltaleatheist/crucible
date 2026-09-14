"""The `crucible` command line.

    crucible init       mint the token, write the config, record the backend
    crucible install    build a job type's env, then decide whether the card fits it
    crucible capability what this host can hold, and why; --write records it
    crucible serve      run the API in the foreground
    crucible service    install/start/stop the machine service that runs `serve`
    crucible models     list and pull model weights
    crucible voices     list and pull voice weights
    crucible doctor     probe the host and every job type; exit 0 only when healthy
    crucible token      print the bearer token (needs --show)

Exit codes: 0 success, 1 refused (named reason on stderr), 2 usage.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import (
    API_VERSION,
    VERSION,
    capability,
    jobenv,
    narratorpatches,
    rvcbase,
    service,
    weights,
    workerenv,
)
from .alignmodels import (
    AlignManifest,
    AlignManifestError,
    load_all_align_manifests,
)
from .asrmodels import AsrManifest, AsrManifestError, load_all_asr_manifests
from .backend import WINDOWS_REFUSAL, Backend, detect_backend
from .config import (
    CAPABILITY_FLAGS,
    CRUCIBLE_HOME_ENV,
    DEFAULT_DESKTOP_ALLOWANCE_BYTES,
    DEFAULT_HOST,
    DEFAULT_PORT,
    MLX_DESKTOP_ALLOWANCE_FRACTION,
    Config,
    config_mode,
    config_path,
    crucible_home,
    default_desktop_allowance_bytes,
    default_server_name,
    load_config,
    mint_token,
    write_config,
)
from .errors import ConfigError, NoViableBackend
from .jobs import ALL_JOB_TYPES, build_registry
from .rvcmodels import RvcManifestError, load_all_rvc_manifests, load_rvc_manifest
from .manifests import (
    ManifestError,
    ModelManifest,
    load_all_manifests,
    load_manifest,
)
from .voices import (
    NARRATOR_ENGINE_SAMPLING,
    VoiceError,
    load_all_voices,
    load_voice,
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2


def _fail(message: str) -> int:
    print(f"crucible: {message}", file=sys.stderr)
    return EXIT_REFUSED


# --------------------------------------------------------------------- init


def cmd_init(args: argparse.Namespace) -> int:
    home = crucible_home()
    path = config_path(home)
    if path.exists() and not args.force:
        return _fail(
            f"{path} already exists; pass --force to replace it (this mints a new "
            "token and every client will need the new one)"
        )

    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")

    # The host reserve is resolved HERE rather than by argparse, because it
    # depends on the backend that was just detected and on the size of its pool
    # (config.default_desktop_allowance_bytes says why the two backends cannot
    # share a number). `None` means the operator did not state one.
    if args.desktop_allowance_bytes is None:
        desktop_allowance_bytes = default_desktop_allowance_bytes(
            backend.kind, backend.gpu.vram_bytes
        )
    else:
        desktop_allowance_bytes = args.desktop_allowance_bytes

    token = mint_token()
    written = write_config(
        home,
        name=args.name if args.name is not None else default_server_name(),
        host=args.host,
        port=args.port,
        token=token,
        backend_kind=backend.kind,
        enable_echo=args.enable_echo,
        enable_llm=args.enable_llm,
        enable_asr=args.enable_asr,
        enable_tts=args.enable_tts,
        enable_align=args.enable_align,
        enable_rvc=args.enable_rvc,
        enable_denoise=args.enable_denoise,
        desktop_allowance_bytes=desktop_allowance_bytes,
    )
    print(f"backend:  {backend.kind} ({backend.gpu.name}, {backend.detail})")
    print(f"config:   {written} (mode {config_mode(written)})")
    print(f"serving:  http://{args.host}:{args.port}/v1")
    print(f"echo job: {'enabled' if args.enable_echo else 'disabled'}")
    print(f"llm job:  {'enabled' if args.enable_llm else 'disabled'}")
    print(f"asr job:  {'enabled' if args.enable_asr else 'disabled'}")
    print(f"tts job:  {'enabled' if args.enable_tts else 'disabled'}")
    print(f"align:    {'enabled' if args.enable_align else 'disabled'}")
    print(f"rvc job:  {'enabled' if args.enable_rvc else 'disabled'}")
    print(f"denoise:  {'enabled' if args.enable_denoise else 'disabled'}")
    source = "stated" if args.desktop_allowance_bytes is not None else (
        f"{backend.kind} default"
    )
    print(
        f"desktop:  {desktop_allowance_bytes / 1024 ** 3:.1f} GiB of "
        f"{backend.gpu.vram_bytes / 1024 ** 3:.1f} GiB treated as this host's own "
        f"desktop, not somebody's job ({source})"
    )
    print("token:    minted; print it with `crucible token --show`")
    return EXIT_OK


# -------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force` "
            "on this host"
        )

    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port

    from .api import create_app  # imported here so `init`/`token` stay light

    app = create_app(config, backend)

    print(f"crucible {VERSION} (api {API_VERSION}) — {config.name}")
    print(f"backend: {backend.kind} ({backend.gpu.name})")
    print(f"listening on http://{host}:{port}/v1")
    if host in ("127.0.0.1", "localhost", "::1"):
        print(
            "bound to loopback: only this host can reach it. To serve the tailnet, "
            "pass --host 0.0.0.0 (or the tailnet IP); the bearer token is the lock."
        )
    else:
        print("bound beyond loopback: the bearer token is the only lock.")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return EXIT_OK


# ------------------------------------------------------------------ service


def _service_context() -> tuple[Config, Backend, str] | int:
    """Config, backend and this host's service mechanism, or a printed refusal.

    The backend is DETECTED and compared against the config, exactly as
    `install` and `capability` do, rather than read off the config alone. A
    service is a promise that `crucible serve` will keep running on this host,
    and `serve` itself refuses when the detected backend and the recorded one
    disagree — so installing a unit in that state would install a unit that
    cannot start.
    """
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force`"
        )
    try:
        mechanism = service.mechanism_for(backend.kind)
    except service.ServiceError as exc:
        return _fail(str(exc))
    return config, backend, mechanism


def cmd_service_install(args: argparse.Namespace) -> int:
    """`crucible service install` — PHASE5-APPS.md 6.0, PHASE11-SERVICE.md.

    Host and port come from `config.toml` AT INSTALL TIME and are baked into the
    unit's `ExecStart`, which means a config edited afterwards is not what the
    service serves until this is re-run. That is stated in the phase doc and
    printed here, because the alternative — a unit that re-reads the config —
    is a unit whose behaviour changes without anybody installing anything.
    """
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    config, _backend, mechanism = resolved
    try:
        lines = service.install(
            mechanism,
            home=service.user_home(),
            server_name=config.name,
            executable=sys.executable,
            crucible_home=config.home,
            host=config.host,
            port=config.port,
            runner=service.subprocess_runner,
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    print(f"mechanism: {mechanism}")
    for line in lines:
        print(line)
    print(
        f"serving:  http://{config.host}:{config.port}/v1 — read from "
        f"{config.path} now and written into the definition. Change either and "
        "re-run `crucible service install`."
    )
    return EXIT_OK


def cmd_service_uninstall(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        lines = service.uninstall(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    return EXIT_OK


def cmd_service_start(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        lines = service.start(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    return EXIT_OK


def cmd_service_stop(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        lines = service.stop(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    return EXIT_OK


def cmd_service_status(args: argparse.Namespace) -> int:
    """Running or not, with the pid and the definition's path.

    **Exit 0 only when it is running**, so a script can gate on it the way it
    gates on `crucible doctor`. A service that is installed and stopped is a
    server nothing can reach, and reporting that as success would make this verb
    useless to the only thing that would automate it.
    """
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        state = service.status(
            mechanism, service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps(state.to_dict(), indent=2))
        return EXIT_OK if state.running else EXIT_REFUSED
    print(f"mechanism:  {state.mechanism}")
    print(
        f"definition: {state.definition} "
        f"({'present' if state.installed else 'NOT THERE'})"
    )
    print(f"running:    {'yes' if state.running else 'NO'}")
    print(f"pid:        {state.pid if state.pid is not None else '-'}")
    print(f"detail:     {state.detail}")
    if state.mechanism == service.SYSTEMD:
        # REPORTED, never assumed: `loginctl enable-linger` is the operator's,
        # and without it this server dies with the session that installed it.
        if state.linger is True:
            linger = "on — this server survives a logout and starts at boot"
        elif state.linger is False:
            linger = (
                "OFF — this server stops when your last session ends. "
                f"`sudo loginctl enable-linger {getpass.getuser()}` grants it"
            )
        else:
            linger = "UNKNOWN — loginctl could not be asked"
        print(f"linger:     {linger}")
    return EXIT_OK if state.running else EXIT_REFUSED


# -------------------------------------------------------------- capability


def _decide_here(config: Config, backend: Backend) -> tuple[capability.Decision, ...]:
    """Every capability class, decided against this host's accelerator.

    The size comes from `backend.gpu.vram_bytes` — the one owner of "how big is
    this card" — and NOT from `accelerator.read_state().free_bytes`. A capability
    is a fact about the host; free VRAM is a fact about this second, and a browser
    open while `crucible install` runs must not permanently disable TTS on a card
    that could hold it. The runtime guard already owns the other question and
    refuses `insufficient_memory` with the measured figure at load time
    (crucible/accelerator.py).
    """
    return capability.decide_all(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
    )


def _write_capability(
    config: Config,
    backend: Backend,
    decisions: tuple[capability.Decision, ...],
    flags: dict[str, bool],
) -> Path:
    """Rewrite config.toml with new capability flags and the record behind them.

    `write_config` writes the whole document, so everything that is not being
    changed is read back off the loaded `Config` and written out again — the token
    included. That is deliberate rather than incidental: an in-place TOML edit
    would have to round-trip comments and would be one more thing that can lose a
    token, and `crucible init --force` (which mints a NEW token and breaks every
    client) must never become the repair for a capability decision.
    """
    values = {
        flag: flags.get(flag, getattr(config, flag)) for flag in CAPABILITY_FLAGS
    }
    return write_config(
        config.home,
        name=config.name,
        host=config.host,
        port=config.port,
        token=config.token,
        backend_kind=config.backend_kind,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        capability=capability.record(
            backend.kind,
            total_bytes=backend.gpu.vram_bytes,
            desktop_allowance_bytes=config.desktop_allowance_bytes,
            decisions=decisions,
        ),
        **values,
    )


def _print_decisions(
    config: Config, backend: Backend, decisions: tuple[capability.Decision, ...]
) -> None:
    budget = capability.available_bytes(
        backend.gpu.vram_bytes, config.desktop_allowance_bytes
    )
    gib = 1024 ** 3
    print(f"backend:  {backend.kind} ({backend.gpu.name})")
    print(
        f"pool:     {backend.gpu.vram_bytes / gib:.1f} GiB "
        f"{capability.POOL_NAME[backend.kind]}"
    )
    print(
        f"reserve:  {config.desktop_allowance_bytes / gib:.1f} GiB for this host "
        "itself"
    )
    print(f"budget:   {budget / gib:.1f} GiB available to a job")
    for decision in decisions:
        mark = "yes" if decision.enabled else "NO"
        print(f"{decision.capability:<10} {mark:<4} {decision.reason}")


def cmd_capability(args: argparse.Namespace) -> int:
    """`crucible capability` — what this host can hold, and why.

    **Why this is a verb of its own and not only a step inside `install`.** The
    decision depends on three things that move independently of the envs: the card
    (Owen swaps GPUs between machines), `desktop_allowance_bytes` (an operator may
    state one), and the manifests (a new quantization ships with a release and
    changes what fits). Any of those changing means the recorded verdict is stale,
    and the only door to re-deciding would otherwise be `crucible install`, which
    rebuilds a multi-gigabyte venv to answer a question about arithmetic. A
    capability that can only be re-decided by reinstalling is a capability nobody
    re-decides.

    It is a **dry run by default**. `--write` is the one that touches config.toml,
    and even then it may only turn a flag OFF, never on: a flag means "this server
    offers this type", which needs the card to fit AND the env to exist, and only
    `crucible install` knows the second. Turning a flag off because the model can
    no longer fit is safe in the direction that matters; turning one on because
    the arithmetic works would advertise a job type with no env behind it.
    """
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force`"
        )
    decisions = _decide_here(config, backend)

    # Only the falling edge. See the docstring: `install` owns the rising one.
    turn_off = {
        f"enable_{name}": False
        for name in sorted({d.job_type for d in decisions})
        if getattr(config, f"enable_{name}")
        and not capability.job_type_enabled(name, decisions)
    }

    if args.json:
        print(
            json.dumps(
                {
                    "backend": backend.kind,
                    "total_bytes": backend.gpu.vram_bytes,
                    "desktop_allowance_bytes": config.desktop_allowance_bytes,
                    "available_bytes": capability.available_bytes(
                        backend.gpu.vram_bytes, config.desktop_allowance_bytes
                    ),
                    "classes": [d.to_dict() for d in decisions],
                    "job_types": {
                        name: capability.job_type_enabled(name, decisions)
                        for name in sorted({d.job_type for d in decisions})
                    },
                    "written": bool(args.write),
                    "turned_off": sorted(turn_off),
                },
                indent=2,
            )
        )
    else:
        _print_decisions(config, backend, decisions)

    if not args.write:
        if not args.json:
            print(
                "dry run: nothing written. Pass --write to record this in "
                f"{config.path}"
            )
        return EXIT_OK

    written = _write_capability(config, backend, decisions, turn_off)
    if not args.json:
        print(f"recorded in {written}")
        for flag in sorted(turn_off):
            print(f"TURNED OFF: [jobs] {flag} — this host cannot hold it")
        if not turn_off:
            print(
                "no flag changed: `crucible capability` only turns a type OFF; "
                "turning one on is `crucible install <type>`"
            )
    return EXIT_OK


# ------------------------------------------------------------------ install


#: Every job type `crucible install` can build an env for. Two shapes of env
#: sit behind it — `jobenv` for the types whose work is an engine SERVER
#: (llm, tts) and `workerenv` for the types whose work is a library in its
#: own venv (asr, and align and rvc after it). The two modules are one
#: module's worth of code twice over and merging them is a named follow-up;
#: this tuple is the one place the difference does not leak.
INSTALLABLE_JOB_TYPES = ("llm", "tts", *workerenv.WORKER_JOB_TYPES)

#: Which `crucible install <type>` builds the env a job type needs. Almost
#: always itself; `denoise` is the exception, because it shares the `rvc` env
#: (`workerenv.JOB_TYPES_SERVED_BY_ENV` is the owner of that fact). `crucible
#: doctor` reads this so the command it suggests is one that exists.
INSTALLER_FOR: dict[str, str] = {
    **{name: name for name in INSTALLABLE_JOB_TYPES},
    **{
        job_type: env
        for env, served in workerenv.JOB_TYPES_SERVED_BY_ENV.items()
        for job_type in served
    },
}


def cmd_install(args: argparse.Namespace) -> int:
    if args.job_type not in INSTALLABLE_JOB_TYPES:
        return _fail(
            f"there is no installer for job type {args.job_type!r}; this build "
            f"installs {sorted(INSTALLABLE_JOB_TYPES)}"
        )
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force`"
        )
    # Which installer a type uses is a fact about the SHAPE of its work, not
    # about its name: `llm` and `tts` run an engine server and get a `jobenv`;
    # `asr`, and `align` and `rvc` after it, run a library in its own venv and
    # get a `workerenv` (PHASE4-AUDIO.md section 0). `workerenv.WORKER_JOB_TYPES`
    # is the list of the second kind, so asking it is the question, rather than
    # testing for one name and assuming everything else is the other.
    if args.job_type in workerenv.WORKER_JOB_TYPES:
        return _install_worker_env(config, backend, args)

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
    except jobenv.EnvError as exc:
        return _fail(str(exc))
    elapsed = time.monotonic() - started
    if not status.installed:
        return _fail(f"the env did not come out installed: {status.detail}")
    print(f"installed in {elapsed:.0f}s: {status.detail}")
    for name in sorted(status.packages):
        if name in (spec.headline, "torch", "numpy", "transformers", "mlx"):
            print(f"  {name}=={status.packages[name]}")
    return _capability_step(config, backend, args.job_type)


def _install_worker_env(
    config: Config, backend: Backend, args: argparse.Namespace
) -> int:
    """`crucible install <type>` for a type whose work runs in its own venv.

    PHASE4-AUDIO.md section 0: the phase 4 types are libraries rather than
    servers, so each gets an env of its own and a worker script run with that
    env's python. The `llm` branch above does the same job through `jobenv`; the
    two modules are one module's worth of code twice over, and merging them is a
    follow-up (crucible/workerenv.py says so at the top).
    """
    try:
        recipe = workerenv.recipe_for(args.job_type, backend.kind)
    except workerenv.WorkerEnvError as exc:
        return _fail(str(exc))
    print(f"backend: {backend.kind}")
    print(f"recipe:  {recipe}")
    print(f"target:  {workerenv.worker_env_dir(config.home, args.job_type)}")
    started = time.monotonic()
    try:
        status = workerenv.install_worker_env(
            config.home,
            args.job_type,
            backend.kind,
            force=args.force,
            on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
        )
    except workerenv.WorkerEnvError as exc:
        return _fail(str(exc))
    elapsed = time.monotonic() - started
    if not status.installed:
        return _fail(f"the env did not come out installed: {status.detail}")
    print(f"installed in {elapsed:.0f}s: {status.detail}")
    headline = workerenv.HEADLINE_PACKAGE[args.job_type]
    for name in sorted(status.packages):
        if name in (headline, "ctranslate2", "numpy", "onnxruntime"):
            print(f"  {name}=={status.packages[name]}")
    # One env can serve more than one job type — `rvc`'s also carries
    # audio-separator, which is `denoise` — and the flag for each of them is
    # decided here, because this is the door that has just built the thing they
    # share.
    return _capability_step(
        config, backend, *workerenv.JOB_TYPES_SERVED_BY_ENV[args.job_type]
    )


def _capability_step(config: Config, backend: Backend, *job_types: str) -> int:
    """The selection step `crucible install` gains (PHASE9-CAPABILITY.md §2).

    It runs AFTER the env is built, and the order is deliberate in both
    directions. Not before, because a flag saying "this server offers tts" must
    not be written by a run whose pip install then failed. Not skipped when the
    card turns out to be too small, because the env is still the right thing to
    have on disk — the card is what is wrong, and a second GPU or a smaller
    allowance makes the same env usable without rebuilding it.

    It writes the record and the flag, and THEN refuses. R6: partial work
    survives failure, and here the partial work is the only durable answer to
    "why is tts off on this box" — throwing it away to make the exit code tidy
    would leave the operator with a refusal and nothing to read.

    Unlike `crucible capability --write`, this one may turn a flag ON, because it
    is the door that has just established the other half of the claim: the env
    exists.

    `job_types` is more than one when one env serves more than one type — the
    `rvc` env also carries audio-separator, which is `denoise`. Each gets its
    own verdict, because they are different arithmetic against the same card,
    and the exit code is a refusal if ANY of them came out disabled: the
    operator asked for an env and one of the things it was for cannot run here.
    """
    decisions = _decide_here(config, backend)
    flags: dict[str, bool] = {}
    disabled: list[str] = []
    print("capability:")
    for job_type in job_types:
        enabled = capability.job_type_enabled(job_type, decisions)
        flags[f"enable_{job_type}"] = enabled
        mine = [d for d in decisions if d.job_type == job_type]
        for decision in mine:
            mark = "yes" if decision.enabled else "NO"
            print(f"  {decision.capability:<10} {mark:<4} {decision.reason}")
        if not enabled:
            disabled.append(
                f"{job_type!r} is DISABLED on this host: "
                + "; ".join(f"{d.capability} — {d.reason}" for d in mine)
            )
    written = _write_capability(config, backend, decisions, flags)
    print(f"recorded in {written}")
    if disabled:
        return _fail(
            "the env is installed, but "
            + " / ".join(disabled)
            + ". The false flag(s) are written, with the numbers, so the refusal "
            "a client gets will name them."
        )
    for flag in flags:
        print(f"[jobs] {flag} = true")
    return EXIT_OK


def _env_spec(
    job_type: str, narrator_engine: str | None, backend_kind: str
) -> jobenv.EnvSpec:
    """The env `crucible install <job type>` builds on this host.

    `tts` needs a second word on `cuda-linux` — there are two envs there, one
    per narrator engine — so the flag is required for it and refused for `llm`,
    rather than quietly ignored on the type that has only one env.
    """
    if job_type == "llm":
        if narrator_engine is not None:
            raise jobenv.EnvError(
                "--narrator-engine names which tts env to build and means nothing "
                "for 'llm', which has exactly one env per host"
            )
        return jobenv.llm_env(backend_kind)
    if narrator_engine is None:
        raise jobenv.EnvError(
            "`crucible install tts` needs --narrator-engine (higgs-v3 or orpheus): "
            "on cuda-linux the two cannot share a venv, because Orpheus pins "
            "vllm 0.7.3 for its per-request logits processors and Higgs v3 needs a "
            "far later torch"
        )
    if narrator_engine not in NARRATOR_ENGINE_SAMPLING:
        raise jobenv.EnvError(
            f"{narrator_engine!r} is not one of narrator's engines; they are "
            f"{sorted(NARRATOR_ENGINE_SAMPLING)}"
        )
    return jobenv.tts_env(narrator_engine, backend_kind)


# ------------------------------------------------------------------- models


def _all_manifests() -> dict[str, "ModelManifest | AsrManifest | AlignManifest"]:
    """Every model this build ships, from all three manifest directories, by id.

    `models/`, `asr/` and `align/` are three directories with three loaders
    (crucible/asrmodels.py explains why they are not one yet), but from the
    command line there is a single namespace of model ids, because `crucible
    models pull <id>` is a single question. A collision between any two would
    make that question ambiguous, so it is refused rather than settled by which
    directory was read first.

    `rvc/` is deliberately NOT in here. Its weights are a single archive fetched
    by name and unpacked rather than a repo snapshot, so `crucible models pull`
    could not serve one; they live under their own `crucible rvc` command and
    their own subtree of `~/.crucible`, which is also what keeps an RVC model
    named `sigma` from colliding with a narrator voice of the same name.
    """
    merged: dict[str, "ModelManifest | AsrManifest | AlignManifest"] = dict(
        load_all_manifests()
    )
    for extra in (load_all_asr_manifests(), load_all_align_manifests()):
        for model_id, manifest in extra.items():
            if model_id in merged:
                raise ManifestError(
                    f"{model_id!r} is declared by both {merged[model_id].path} and "
                    f"{manifest.path}; a model id names one model"
                )
            merged[model_id] = manifest
    return merged


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


def cmd_models_list(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = _all_manifests()
    except (ManifestError, AsrManifestError, AlignManifestError) as exc:
        return _fail(str(exc))
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        rows.append(
            {
                "id": manifest.id,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                # An ASR manifest carries no context. Whisper's window is 30
                # seconds of audio and is not a number anybody sets, so null
                # here means "this model has no such knob", not "unknown".
                "context_default": (
                    manifest.context_for(backend.kind)
                    if isinstance(manifest, ModelManifest)
                    else None
                ),
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"not pulled — `crucible models pull {manifest.id}`"
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else "not pulled"
        )
        print(f"{row['id']:<16} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_models_pull(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = _all_manifests()
    except (ManifestError, AsrManifestError, AlignManifestError) as exc:
        return _fail(str(exc))
    manifest = manifests.get(args.model)
    if manifest is None:
        return _fail(
            f"no manifest for model {args.model!r}; this build ships "
            f"{sorted(manifests)}"
        )
    if not manifest.supports(backend.kind):
        return _fail(
            f"model {args.model!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]} for {backend.kind}")
    try:
        result = weights.pull(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


# ------------------------------------------------------------------- voices


def cmd_voices_list(args: argparse.Namespace) -> int:
    """Every voice manifest this build ships and where it stands on this host.

    The same shape as `crucible models list`, and deliberately not the
    `/v1/voices` row: this command answers "what is on this disk", which a person
    runs before a load, while the row answers "what can this server be asked for",
    which a client reads.
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = load_all_voices()
    except VoiceError as exc:
        return _fail(str(exc))
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "kind": manifest.kind,
                "narrator_engine": manifest.narrator_engine,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "estimate_basis": spec.estimate_basis,
                "max_chars": spec.max_chars,
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"not pulled — `crucible voices pull {manifest.id}`"
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else "not pulled"
        )
        print(f"{row['id']:<22} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_voices_pull(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifest = load_voice(args.voice)
    except VoiceError as exc:
        return _fail(str(exc))
    if not manifest.supports(backend.kind):
        return _fail(
            f"voice {args.voice!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]} for {backend.kind}")
    try:
        result = weights.pull(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


# ---------------------------------------------------------------------- rvc


def cmd_rvc_list(args: argparse.Namespace) -> int:
    """Every RVC manifest this build ships and where it stands on this host.

    Its own command rather than a row in `crucible models list`, for the reason
    `_all_manifests` gives: an RVC model's weights are one archive fetched by
    name, not a repo snapshot, so `models pull` could not fetch one — and the ids
    are a separate namespace, which is what stops an RVC model called `sigma`
    from colliding with the narrator voice of the same name.
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = load_all_rvc_manifests()
    except RvcManifestError as exc:
        return _fail(str(exc))
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "model_name": manifest.model_name,
                "has_index": manifest.has_index,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "archive": spec.archive,
                "revision": spec.revision,
                "archive_bytes": spec.archive_bytes,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"not pulled — `crucible rvc pull {manifest.id}`"
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else "not pulled"
        )
        print(f"{row['id']:<22} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_rvc_pull(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifest = load_rvc_manifest(args.model)
    except RvcManifestError as exc:
        return _fail(str(exc))
    if not manifest.supports(backend.kind):
        return _fail(
            f"RVC model {args.model!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(
        f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]}:{spec.archive} "
        f"for {backend.kind}"
    )
    try:
        result = weights.pull_archive(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def cmd_rvc_pull_base(args: argparse.Namespace) -> int:
    """`crucible rvc pull-base` — the engine's shared assets, at a pinned sha.

    Its own verb rather than a step inside `crucible install rvc`, for the
    reason every other weights pull is its own verb: installing an env and
    fetching 600 MB of weights are different acts with different failure modes,
    and `crucible install llm` does not pull a 19 GB model either. One set, one
    command, one owner (PHASE4-AUDIO.md section 4.1).
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, _backend = resolved
    try:
        assets = rvcbase.load_rvc_base()
    except rvcbase.RvcBaseError as exc:
        return _fail(str(exc))
    print(
        f"{assets.id}: {assets.hf_repo}@{assets.revision[:12]}, "
        f"{len(assets.files)} file(s), {assets.total_bytes / 1e9:.2f} GB"
    )
    for entry in assets.files:
        print(f"  {entry.target} — {entry.why}")
    try:
        result = rvcbase.pull(
            config, assets, force=args.force, on_line=lambda line: print(f"  {line}")
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    absent = rvcbase.missing(config, assets)
    if absent:
        # Unreachable unless something removed a file between the place and
        # this read; said out loud rather than reported as success, because the
        # next thing to look at this tree is a job that will fail inside urvc.
        return _fail(
            f"the pull finished but {sorted(absent)} are not under "
            f"{rvcbase.base_root(config)}"
        )
    print(f"{assets.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


# ------------------------------------------------------------------- doctor


def _job_type_reports(config: Config, backend: Backend) -> list[dict[str, Any]]:
    registry = build_registry(config, backend)
    reports: list[dict[str, Any]] = []
    for name in sorted(ALL_JOB_TYPES):
        plugin = registry.get(name)
        if plugin is None:
            reports.append(
                {
                    "name": name,
                    "enabled": False,
                    "ready": False,
                    "detail": "not enabled in config.toml",
                    "models": [],
                }
            )
            continue
        status = plugin.check(backend)
        reports.append(
            {
                "name": name,
                "enabled": True,
                "ready": status.ready,
                "detail": status.detail,
                "models": [m.to_dict() for m in plugin.describe_models()],
            }
        )
    return reports


def _env_report(
    report: dict[str, Any], label: str, home: Path, spec: jobenv.EnvSpec,
    backend_kind: str,
) -> dict[str, Any]:
    """One env's status, appending a problem to the report when it is not ready."""
    try:
        status = jobenv.env_status(home, spec, backend_kind)
    except jobenv.EnvError as exc:
        report["problems"].append(f"{label}: {exc}")
        return {"installed": False, "detail": str(exc)}
    if not status.installed:
        report["problems"].append(f"{label}: {status.detail}")
    return status.to_dict()


def _capability_report(
    report: dict[str, Any], config: Config, backend: Backend
) -> None:
    """What was decided here, whether it is still true, and whether it agrees.

    Three checks, and only ONE of them is a problem, which is the point:

    * **The record is stale.** It names a different backend or a different pool
      size than this host now has. A swapped card is the case this catches, and it
      is caught by comparing NUMBERS rather than by writing down a date — the date
      a decision was made says nothing about whether it is still right.
    * **A flag is on that the numbers refuse.** `enable_tts = true` with every tts
      class recorded disabled. This is the dangerous direction and the only
      PROBLEM: the server is advertising a job type whose first request is an OOM.
    * **A flag is off that the numbers allow.** Printed as a NOTE, never a
      problem. It is the ordinary state of a host whose env for that type has not
      been built yet, and `crucible install <type>` is the thing that changes it.
    """
    record = config.capability
    if record is None:
        report["capability"] = None
        return
    entry: dict[str, Any] = {
        **record.to_dict(),
        "stale": False,
        "could_enable": [],
    }
    if record.backend_kind != backend.kind:
        entry["stale"] = True
        report["problems"].append(
            f"capability_stale: the record was decided on {record.backend_kind} "
            f"and this host is {backend.kind}; re-run `crucible capability --write`"
        )
    if record.total_bytes != backend.gpu.vram_bytes:
        entry["stale"] = True
        report["problems"].append(
            f"capability_stale: the record was decided against "
            f"{record.total_bytes / 1024 ** 3:.1f} GiB and this host has "
            f"{backend.gpu.vram_bytes / 1024 ** 3:.1f} GiB; re-run "
            "`crucible capability --write`"
        )
    for name in sorted({cls.job_type for cls in capability.CLASSES}):
        rows = [
            record.row(cls.name) for cls in capability.classes_for_job_type(name)
        ]
        known = [row for row in rows if row is not None]
        if not known:
            continue
        fits = any(row.enabled for row in known)
        flagged = getattr(config, f"enable_{name}")
        if flagged and not fits:
            report["problems"].append(
                f"capability_contradicted: [jobs] enable_{name} is true and "
                "nothing behind it fits this host — "
                + "; ".join(f"{row.capability}: {row.reason}" for row in known)
            )
        # `echo` is deliberately not here: it fits every card (it never touches
        # one) and there is no `crucible install echo`, so suggesting one would
        # be a note whose action does not exist. `INSTALLER_FOR` is the owner of
        # "which command installs this", so it is the thing asked — and it is
        # what keeps the note for `denoise` pointing at `crucible install rvc`,
        # the env it actually shares, rather than at a command that does not
        # exist.
        if fits and not flagged and name in INSTALLER_FOR:
            entry["could_enable"].append(name)
    report["capability"] = entry


def _doctor_report() -> dict[str, Any]:
    home = crucible_home()
    report: dict[str, Any] = {
        "healthy": False,
        "crucible": {"version": VERSION, "api_version": API_VERSION},
        "home": str(home),
        "config": None,
        "backend": None,
        "job_types": [],
        "llm_env": None,
        "worker_envs": [],
        "tts_envs": {},
        "narrator_patches": [],
        "capability": None,
        "problems": [],
    }

    try:
        backend = detect_backend()
        report["backend"] = backend.to_dict()
    except NoViableBackend as exc:
        report["problems"].append(f"no_viable_backend: {exc.reason}")
        backend = None

    try:
        config = load_config(home)
    except ConfigError as exc:
        report["problems"].append(f"config: {exc}")
        config = None

    if config is not None:
        mode = config_mode(config.path)
        report["config"] = {
            "path": str(config.path),
            "mode": mode,
            "name": config.name,
            "host": config.host,
            "port": config.port,
            "enable_echo": config.enable_echo,
            "enable_llm": config.enable_llm,
            "enable_asr": config.enable_asr,
            "enable_tts": config.enable_tts,
            "enable_align": config.enable_align,
            "enable_rvc": config.enable_rvc,
            "enable_denoise": config.enable_denoise,
            "desktop_allowance_bytes": config.desktop_allowance_bytes,
            "backend_kind": config.backend_kind,
            # Which capability flags this config did not carry. A config written
            # before a job type existed reads that type as off, which is the only
            # answer that does not invalidate every server on an upgrade — and
            # this is how it says so out loud instead of looking like a choice.
            "flags_absent": list(config.flags_absent),
        }
        if mode != "0o600":
            report["problems"].append(
                f"config_permissions: {config.path} is mode {mode}; the token should "
                "be readable only by its owner (chmod 600)"
            )
        if backend is not None and backend.kind != config.backend_kind:
            report["problems"].append(
                f"backend_changed: config says {config.backend_kind}, this host is "
                f"{backend.kind}"
            )

    if config is not None and backend is not None:
        _capability_report(report, config, backend)
        if config.enable_llm:
            report["llm_env"] = _env_report(
                report,
                "llm_env",
                config.home,
                jobenv.llm_env(backend.kind),
                backend.kind,
            )
        for job_type in workerenv.WORKER_JOB_TYPES:
            if not getattr(config, f"enable_{job_type}"):
                continue
            try:
                worker_env = workerenv.env_status(config.home, job_type, backend.kind)
                report["worker_envs"].append(worker_env.to_dict())
                if not worker_env.installed:
                    report["problems"].append(f"{job_type}_env: {worker_env.detail}")
            except workerenv.WorkerEnvError as exc:
                report["worker_envs"].append(
                    {"job_type": job_type, "installed": False, "detail": str(exc)}
                )
                report["problems"].append(f"{job_type}_env: {exc}")
        if config.enable_tts:
            # One row per narrator engine, because on cuda-linux they are two
            # separate venvs and a voice load picks by its manifest's
            # `narrator_engine`. On mlx-darwin both names resolve to the same
            # env, and the two rows say so by carrying the same path.
            report["tts_envs"] = {
                engine: _env_report(
                    report,
                    f"tts_env[{engine}]",
                    config.home,
                    jobenv.tts_env(engine, backend.kind),
                    backend.kind,
                )
                for engine in sorted(NARRATOR_ENGINE_SAMPLING)
            }
            # The two site-packages edits pip cannot express (PHASE3-TTS.md
            # section 4). They are reported SEPARATELY from the env row and not
            # folded into it, because an env whose pins all match is otherwise
            # reported ready — and a reader has no way to tell that from an env
            # that will render every chunk with 240 ms of garbage on the end.
            patched_spec = jobenv.tts_env(
                narratorpatches.PATCHED_ENGINE, backend.kind
            )
            report["narrator_patches"] = narratorpatches.check(
                jobenv.env_dir(config.home, patched_spec),
                jobenv.recipe_pins(jobenv.recipe_for(patched_spec)),
            )
            for entry in report["narrator_patches"]:
                # `applied` is not the test. Both patches edit the vLLM stack,
                # which `mlx-darwin`'s recipe does not install, and a Mac that
                # has nothing to patch is sound rather than broken.
                if entry["status"] not in narratorpatches.SOUND_STATUSES:
                    report["problems"].append(
                        f"narrator_patch[{entry['id']}]: {entry['status']} — "
                        f"{entry['detail']}. {entry['why']}"
                    )
        report["job_types"] = _job_type_reports(config, backend)
        for entry in report["job_types"]:
            if entry["enabled"] and not entry["ready"]:
                report["problems"].append(
                    f"job_type_not_ready: {entry['name']}: {entry['detail']}"
                )

    report["healthy"] = not report["problems"]
    return report


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _doctor_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"crucible {VERSION} (api {API_VERSION})")
        print(f"home:    {report['home']}")
        backend = report["backend"]
        if backend is None:
            print("backend: NONE")
        else:
            gpu = backend["gpu"]
            print(
                f"backend: {backend['kind']} on {backend['platform']}/{backend['arch']}"
            )
            print(
                f"gpu:     {gpu['vendor']} {gpu['name']} "
                f"({gpu['vram_bytes'] / 1024 ** 3:.1f} GiB) — {backend['detail']}"
            )
        config = report["config"]
        if config is None:
            print("config:  MISSING")
        else:
            print(f"config:  {config['path']} (mode {config['mode']})")
            print(f"serves:  {config['name']} on {config['host']}:{config['port']}")
            if config["flags_absent"]:
                absent = ", ".join(config["flags_absent"])
                print(
                    f"note:    this config predates {absent}; those job types are "
                    "off. Add the keys to [jobs] to turn them on — do NOT run "
                    "`crucible init --force`, which mints a new token"
                )
        env = report["llm_env"]
        if env is not None:
            mark = "ready" if env["installed"] else "NOT READY"
            print(f"llm env: {mark} — {env['detail']}")
        for worker_env in report["worker_envs"]:
            mark = "ready" if worker_env["installed"] else "NOT READY"
            print(
                f"{worker_env['job_type']} env: {mark} — {worker_env['detail']}"
            )
        for engine, entry in sorted(report["tts_envs"].items()):
            mark = "ready" if entry["installed"] else "NOT READY"
            print(f"tts env ({engine}): {mark} — {entry['detail']}")
        for entry in report["narrator_patches"]:
            if entry["status"] == narratorpatches.NOT_APPLICABLE:
                mark = "n/a"
            else:
                mark = "applied" if entry["applied"] else entry["status"].upper()
            print(f"narrator patch ({entry['id']}): {mark} — {entry['detail']}")
        capability_entry = report["capability"]
        if capability_entry is None:
            print(
                "capability: NOT DECIDED — nothing has probed this host's card "
                "against the models; run `crucible capability`"
            )
        else:
            for row in capability_entry["classes"]:
                mark = "yes" if row["enabled"] else "NO"
                print(f"capability {row['capability']}: {mark} — {row['reason']}")
            for name in capability_entry["could_enable"]:
                print(
                    f"note:    this host can hold {name}, and [jobs] enable_{name} "
                    f"is off — `crucible install {INSTALLER_FOR[name]}` builds its "
                    "env and turns it on"
                )
        for entry in report["job_types"]:
            mark = "ready" if entry["ready"] else ("off" if not entry["enabled"] else "NOT READY")
            print(f"job {entry['name']}: {mark} — {entry['detail']}")
        for problem in report["problems"]:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print("healthy" if report["healthy"] else "unhealthy")
    return EXIT_OK if report["healthy"] else EXIT_REFUSED


# -------------------------------------------------------------------- token


def cmd_token(args: argparse.Namespace) -> int:
    if not args.show:
        return _fail("pass --show to print the bearer token")
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    print(config.token)
    return EXIT_OK


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description="One inference server, many client apps.",
        epilog=(
            f"State lives under ${CRUCIBLE_HOME_ENV} (default ~/.crucible). "
            "Crucible runs on Linux with an NVIDIA card and on Apple Silicon macOS."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crucible {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init", help="detect the backend, mint a token, write config.toml"
    )
    init.add_argument("--force", action="store_true", help="replace an existing config")
    init.add_argument("--name", default=None, help="server name (default crucible@<hostname>)")
    init.add_argument("--host", default=DEFAULT_HOST, help=f"default bind host ({DEFAULT_HOST})")
    init.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default port ({DEFAULT_PORT})")
    init.add_argument(
        "--enable-echo",
        action="store_true",
        help="register the echo test job type ([jobs] enable_echo)",
    )
    init.add_argument(
        "--enable-llm",
        action="store_true",
        help="register the load-model / unload-model job types and the OpenAI "
        "proxy ([jobs] enable_llm)",
    )
    init.add_argument(
        "--enable-asr",
        action="store_true",
        help="register the asr (faster-whisper transcription) job type "
        "([jobs] enable_asr)",
    )
    init.add_argument(
        "--enable-tts",
        action="store_true",
        help="register the load-voice / unload-voice job types and /v1/voices "
        "([jobs] enable_tts)",
    )
    init.add_argument(
        "--enable-align",
        action="store_true",
        help="register the align (Qwen3-ForcedAligner) and unload-aligner job "
        "types ([jobs] enable_align)",
    )
    init.add_argument(
        "--enable-rvc",
        action="store_true",
        help="register the rvc (ultimate-rvc voice conversion) job type "
        "([jobs] enable_rvc)",
    )
    init.add_argument(
        "--enable-denoise",
        action="store_true",
        help="register the denoise (audio-separator stem separation) job type; "
        "it shares the rvc env ([jobs] enable_denoise)",
    )
    init.add_argument(
        "--desktop-allowance-bytes",
        type=int,
        default=None,
        help=(
            "VRAM this host's own desktop holds, which the accelerator guard does "
            "not count as somebody's job. Defaults PER BACKEND once the card is "
            f"detected: cuda-linux {DEFAULT_DESKTOP_ALLOWANCE_BYTES} = 3 GiB flat, "
            f"mlx-darwin {MLX_DESKTOP_ALLOWANCE_FRACTION:.0%} of unified memory "
            "because the model and the whole OS share one pool. Use 0 on a "
            "headless box"
        ),
    )
    init.set_defaults(func=cmd_init)

    install = subparsers.add_parser(
        "install", help="create a job type's env and install its recipe"
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
            "which tts env to build; required for 'tts' because cuda-linux has "
            "one venv per narrator engine, and refused for 'llm'"
        ),
    )
    install.add_argument(
        "--force", action="store_true", help="rebuild the env from scratch"
    )
    install.add_argument(
        "--verbose", action="store_true", help="echo pip's output line by line"
    )
    install.set_defaults(func=cmd_install)

    capability_parser = subparsers.add_parser(
        "capability",
        help="what this host's card can hold, and why; --write records it",
    )
    capability_parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "record the verdict in config.toml. It may only turn a job type OFF; "
            "turning one on needs its env, which is `crucible install <type>`"
        ),
    )
    capability_parser.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    capability_parser.set_defaults(func=cmd_capability)

    models = subparsers.add_parser("models", help="list and pull model weights")
    model_commands = models.add_subparsers(dest="models_command", required=True)

    models_list = model_commands.add_parser(
        "list", help="every manifest this build ships and where it stands here"
    )
    models_list.add_argument("--json", action="store_true", help="machine-readable")
    models_list.set_defaults(func=cmd_models_list)

    models_pull = model_commands.add_parser(
        "pull", help="fetch a model's weights at the manifest's pinned revision"
    )
    models_pull.add_argument("model", help="the Crucible model id, e.g. qwen3.5-9b")
    models_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    models_pull.set_defaults(func=cmd_models_pull)

    voices = subparsers.add_parser("voices", help="list and pull voice weights")
    voice_commands = voices.add_subparsers(dest="voices_command", required=True)

    voices_list = voice_commands.add_parser(
        "list", help="every voice manifest this build ships and where it stands here"
    )
    voices_list.add_argument("--json", action="store_true", help="machine-readable")
    voices_list.set_defaults(func=cmd_voices_list)

    voices_pull = voice_commands.add_parser(
        "pull", help="fetch a voice's weights at the manifest's pinned revision"
    )
    voices_pull.add_argument("voice", help="the Crucible voice id, e.g. deathstalker")
    voices_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    voices_pull.set_defaults(func=cmd_voices_pull)

    rvc = subparsers.add_parser("rvc", help="list and pull RVC voice-conversion models")
    rvc_commands = rvc.add_subparsers(dest="rvc_command", required=True)

    rvc_list = rvc_commands.add_parser(
        "list", help="every RVC manifest this build ships and where it stands here"
    )
    rvc_list.add_argument("--json", action="store_true", help="machine-readable")
    rvc_list.set_defaults(func=cmd_rvc_list)

    rvc_pull = rvc_commands.add_parser(
        "pull", help="fetch and unpack an RVC model at the manifest's pinned revision"
    )
    rvc_pull.add_argument("model", help="the Crucible RVC id, e.g. deathstalker-rvc-v1")
    rvc_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    rvc_pull.set_defaults(func=cmd_rvc_pull)

    rvc_pull_base = rvc_commands.add_parser(
        "pull-base",
        help="fetch ultimate-rvc's shared base assets (the embedder and the "
        "pitch predictors) — the engine's, not any model's",
    )
    rvc_pull_base.add_argument(
        "--force", action="store_true", help="re-pull even if they are already there"
    )
    rvc_pull_base.set_defaults(func=cmd_rvc_pull_base)

    serve = subparsers.add_parser("serve", help="run the API in the foreground")
    serve.add_argument("--host", default=None, help="bind host (default from config)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    serve.add_argument("--log-level", default="info", help="uvicorn log level")
    serve.set_defaults(func=cmd_serve)

    service_parser = subparsers.add_parser(
        "service",
        help="the machine service that runs `crucible serve` (PHASE11-SERVICE.md)",
        description=(
            "A local Crucible is a service and no app owns it (Owen, 2026-09-13; "
            "PHASE5-APPS.md section 6.0). On cuda-linux that is a systemd USER "
            "unit, on mlx-darwin a launchd agent. Every verb is idempotent."
        ),
    )
    service_commands = service_parser.add_subparsers(
        dest="service_command", required=True
    )

    service_install = service_commands.add_parser(
        "install",
        help="write the unit or plist for this host, enable it and start it",
    )
    service_install.set_defaults(func=cmd_service_install)

    service_uninstall = service_commands.add_parser(
        "uninstall", help="stop the service, forget it, and remove its definition"
    )
    service_uninstall.set_defaults(func=cmd_service_uninstall)

    service_start = service_commands.add_parser(
        "start", help="make sure the installed service is running"
    )
    service_start.set_defaults(func=cmd_service_start)

    service_stop = service_commands.add_parser(
        "stop", help="stop the service without forgetting it"
    )
    service_stop.set_defaults(func=cmd_service_stop)

    service_status = service_commands.add_parser(
        "status",
        help="running or not, with the pid and the unit/plist path; exit 0 only "
        "when it is running",
    )
    service_status.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    service_status.set_defaults(func=cmd_service_status)

    doctor = subparsers.add_parser("doctor", help="probe the host and the job types")
    doctor.add_argument("--json", action="store_true", help="machine-readable report")
    doctor.set_defaults(func=cmd_doctor)

    token = subparsers.add_parser("token", help="print the bearer token")
    token.add_argument("--show", action="store_true", help="required; prints the secret")
    token.set_defaults(func=cmd_token)

    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        print(f"crucible: {WINDOWS_REFUSAL}", file=sys.stderr)
        return EXIT_REFUSED
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
