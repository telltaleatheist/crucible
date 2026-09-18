"""The `crucible` command line.

    crucible init       mint the token, write the config, record the backend
    crucible install    download a job type's env pack, then decide whether the card fits it
    crucible envpack    build the packs a release carries (developer / CI)
    crucible capability what this host can hold, and why; --write records it
    crucible serve      run the API in the foreground
    crucible service    install/start/stop the machine service that runs `serve`
    crucible orchestrator  win32 only: the tray that manages this machine's engine
                        (`crucible host` is the same verb, deprecated)
    crucible models     list and pull model weights
    crucible voices     list and pull voice weights
    crucible doctor     probe the host and every job type; exit 0 only when healthy
    crucible token      print the bearer token (--show) or the pairing line (--url)
    crucible uninstall  install, run backwards; weights kept unless --purge-weights

Exit codes: 0 success, 1 refused (named reason on stderr), 2 usage.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import (
    API_VERSION,
    VERSION,
    capability,
    catalog,
    denoisemodels,
    envpack,
    hosttools,
    jobenv,
    llamacpp,
    narratorpatches,
    pairing,
    rvcbase,
    service,
    uninstall,
    weights,
    workerenv,
)
from .alignmodels import (
    AlignManifest,
    AlignManifestError,
    load_all_align_manifests,
)
from .asrmodels import AsrManifest, AsrManifestError, load_all_asr_manifests
from .backend import (
    BACKEND_KINDS,
    CUDA_LINUX,
    LLAMA_WINDOWS,
    WINDOWS_REFUSAL,
    Backend,
    backend_not_here,
    detect_backend,
)
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
from .errors import ConfigError, CrucibleError, NoViableBackend
from .interfaces import InterfaceError
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


# --------------------------------------------------------------------- host


def cmd_orchestrator(args: argparse.Namespace) -> int:
    """`crucible orchestrator` — PHASE15 section 4, PHASE17. Windows only.

    **`crucible host` is the same verb**, kept as an argparse alias and
    deprecated in PHASE17-ORCHESTRATOR.md section 7 rather than in code: the
    Startup shortcut installed on Owen's PC on 2026-09-15 has
    `-m crucible.cli host` baked into it, and `--install-startup` still writes
    exactly that string, so a pack rebuilt after tonight starts the tray the
    shortcut already points at. The alias goes when a release changes the
    shortcut, and that is not tonight.

    The verb is refused `host_windows_only` everywhere else, and that is not a
    platform check standing in for a feature check: on Linux and macOS the
    server runs ON the machine and its own service manager supervises it
    (4.4, "no host on the Mac"). There is nothing for a tray to own.

    Three shapes, and the two that are not the tray exit without starting one:

      --install-startup   write the Startup item and print its path
      --remove-startup    delete it, and say whether there was one
      (bare)              the tray
    """
    from .host import startup as host_startup
    from .host.errors import HostError
    from .host.runner import ProcessRunner

    if sys.platform != "win32":
        return _fail(
            "host_windows_only: `crucible orchestrator` is a Windows verb. On "
            f"{sys.platform} the server runs on this machine and "
            f"{'systemd' if sys.platform == 'linux' else 'launchd'} already "
            "supervises it — `crucible service status` is the question you "
            "are asking."
        )

    runner = ProcessRunner(sys.platform, os.environ)
    try:
        if args.install_startup:
            outcome = host_startup.install(runner)
            print(outcome.detail)
            return EXIT_OK
        if args.remove_startup:
            outcome = host_startup.remove(runner)
            print(outcome.detail)
            return EXIT_OK
    except HostError as exc:
        return _fail(f"{exc.code}: {exc.message}")

    from .host.app import run as run_host

    try:
        if not args.headless:
            from .desktop import tray
            tray()
            return EXIT_OK
        return run_host(headless=args.headless)
    except HostError as exc:
        return _fail(f"{exc.code}: {exc.message}")


# --------------------------------------------------------------------- init


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


def carried_from(path: Path) -> tuple[str, dict[str, Any]]:
    """`--config-from`: the token, the routes and the upstreams, and NOTHING else.

    PHASE15-HOST.md 4.3. The host writes this file at 0600 when it moves a
    Windows Crucible into the WSL guest and deletes it afterwards; the point
    of the flag is that the TOKEN survives, so every app that paired with this
    machine stays paired.

    Three things and no fourth. The host, the port, the name, the backend and
    the job flags belong to the machine being INITIALISED, not to the one
    being left — a guest that inherited `backend = "llama-windows"` would
    refuse to serve on its own card, and a guest that inherited a desktop
    allowance measured against somebody's iGPU would hold the wrong number.
    """
    import tomllib

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"config_from_unreadable: {path} could not be read: {exc}")
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config_from_unreadable: {path} is not TOML: {exc}")
    auth = document.get("auth")
    token = auth.get("token") if isinstance(auth, dict) else None
    if not isinstance(token, str) or token.strip() == "":
        raise ConfigError(
            f"config_from_no_token: {path} has no [auth] token. The point of "
            "--config-from is that the token survives the move; a file without "
            "one carries nothing."
        )
    carried: dict[str, Any] = {}
    for section in ("routes", "upstreams"):
        value = document.get(section)
        if isinstance(value, dict):
            carried[section] = value
    return token, carried


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

    # `--backend` STATES what the caller expects this host to be, and is
    # checked against what it is. Section 2: *"`crucible init --backend
    # llama-windows` is legal only on win32 … `cuda-linux`/`mlx-darwin` on
    # win32 are refused the same way"*. `crucible host` passes it (4.3) so a
    # host that somehow ran on the wrong machine says so here instead of
    # writing a config the server would refuse to start from.
    if args.backend is not None and args.backend != backend.kind:
        return _fail(_backend_mismatch(args.backend, backend))

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

    # The token is minted HERE unless the caller brought one. `--token` exists
    # for `@crucible/bootstrap` (PHASE12-BOOTSTRAP.md): the app that installs a
    # local server mints the token on its own side and hands it over, so it
    # already holds what it would otherwise have to read back out of the file.
    # A blank one is refused — a config with an empty token is a server nothing
    # can reach, and `load_config` would refuse it anyway.
    carried: dict[str, Any] = {}
    if args.config_from is not None:
        if args.token is not None:
            return _fail(
                "--config-from and --token both name a token, and two answers to "
                "one question is not a thing this command picks between. Pass one."
            )
        try:
            token, carried = carried_from(Path(args.config_from))
        except ConfigError as exc:
            return _fail(str(exc))
    elif args.token is not None:
        token = args.token
        if token.strip() == "" or any(ch.isspace() for ch in token):
            return _fail("--token must be a non-empty string with no whitespace")
    else:
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
        carried_tables=carried or None,
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
    if args.config_from is not None:
        print(
            f"carried:  the token and {sorted(carried) or 'no other table'} from "
            f"{args.config_from} (4.3); every app that paired stays paired"
        )
    print(
        "token:    "
        + (
            "carried; "
            if args.config_from is not None
            else ("as given; " if args.token is not None else "minted; ")
        )
        + "print it with `crucible token --show`"
    )
    # THE PAIRING FILE (PHASE15-HOST.md section 3.6). Written here, at 0600,
    # beside the config, so an app on this machine connects without anybody
    # typing a token — and rewritten by `--force`, which mints a new one.
    paired = _write_pairing_file(
        home,
        name=args.name if args.name is not None else default_server_name(),
        port=args.port,
        token=token,
    )
    print(f"pairing:  {paired} ({_pairing_permission(paired)})")
    _print_pairing(
        args.name if args.name is not None else default_server_name(),
        args.host,
        args.port,
        token,
    )
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
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force` on this host"
        )

    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port

    # THE PAIRING FILE IS WRITTEN AT STARTUP TOO (PHASE15-HOST.md 3.6, amended
    # 2026-09-14). `init` and `service install` wrote it and nothing else did,
    # so a server that EXISTED before this phase had none and an app on its own
    # machine was told there was no engine there — measured on the Mac after
    # its upgrade. The line is written when it is absent OR when it does not
    # match what this config says, because a rotated token, a renamed server
    # or a moved port each leave a file that is worse than no file: it points
    # an app at a door with the wrong key.
    #
    # The line is the CONFIG's, not this run's `--host`/`--port` overrides:
    # 3.6's file answers "an app on THIS machine wants in", and a developer
    # running `crucible serve --port 7999` for an afternoon must not repoint
    # every app on the box at a server that is about to stop.
    try:
        _sync_pairing_file(config)
    except pairing.PairingFileError as exc:
        # NOT fatal, and NOT silent. The server is the thing being started and
        # it works without this file; what the file changes is whether an app
        # has to be told a token by hand. Refusing to serve over it would be
        # the tail wagging the dog, and swallowing it would be a machine where
        # connect quietly stopped working.
        print(f"crucible: pairing file NOT written: {exc}", file=sys.stderr)

    from .api import create_app  # imported here so `init`/`token` stay light

    app = create_app(config, backend)
    # WHERE IT IS REALLY LISTENING, not where the file says. `--host` and
    # `--port` override the config for this run, and `GET /v1/setup` builds its
    # pairing lines from the bind address — so a server started
    # `crucible serve --host 0.0.0.0` on a config that says `127.0.0.1` must
    # hand out its interface addresses, not a loopback nobody else can dial.
    app.state.bind_host = host
    app.state.bind_port = port

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

    if getattr(args, "controller_stdin", False):
        from .host.child_lifecycle import run_owned_server
        run_owned_server(app, host=host, port=port, log_level=args.log_level)
    else:
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
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force`"
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
    # Section 3.6 again: a server installed as a service is the one an app is
    # most likely to meet without a person present, so the file it reads is
    # written here too — with the SAME token, so nothing that had paired is
    # unpaired by installing a unit.
    paired = _write_pairing_file(
        config.home, name=config.name, port=config.port, token=config.token
    )
    print(f"pairing:  {paired} ({_pairing_permission(paired)})")
    _print_pairing(config.name, config.host, config.port, config.token)
    from .local import publish_installation
    publish_installation(config.home)
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
        return EXIT_OK if state.running is True else EXIT_REFUSED
    print(f"mechanism:  {state.mechanism}")
    print(
        f"definition: {state.definition} "
        f"({'present' if state.installed else 'NOT THERE'})"
    )
    # Three answers, because there are three. `None` is "no manager could be
    # asked", and reporting that as NO would be this command inventing a fact.
    if state.running is True:
        runs = "yes"
    elif state.running is False:
        runs = "NO"
    else:
        runs = "UNKNOWN - no manager could be asked"
    print(f"running:    {runs}")
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
    return EXIT_OK if state.running is True else EXIT_REFUSED


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
        # WHICH POOL THIS IS, which the size alone cannot say: on
        # `llama-windows` 24 GiB is a card on one machine and system RAM on
        # another, and the row's words differ (`crucible/capability.py`'s
        # `pool_name` and the cpu-build sentence).
        gpu_vendor=backend.gpu.vendor,
        # The app selections this config carries. A probe that ignored them
        # would write a record naming a different model than the settings
        # document reports, and nothing would be comparing the two.
        chosen={entry.capability: entry.model for entry in config.local_models},
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
        # THE OPERATOR'S RETENTION WINDOW SURVIVES A CAPABILITY WRITE, on the
        # same terms as the routes below: `write_config` writes the whole
        # document, so omitting it would quietly put a server back to the
        # default seven days the next time `crucible install` ran.
        retention_days=config.retention_days,
        capability=capability.record(
            backend.kind,
            total_bytes=backend.gpu.vram_bytes,
            desktop_allowance_bytes=config.desktop_allowance_bytes,
            decisions=decisions,
            # THE OPERATOR'S ROUTES SURVIVE A CAPABILITY WRITE, and they have
            # to be passed for that to be true: `write_config` writes the whole
            # document, so a `crucible install` that omitted them would unroute
            # a server somebody configured this morning and put the local rows
            # back over the upstream ones (PHASE15-HOST.md section 2).
            routes={entry.capability: entry.model for entry in config.routes},
        ),
        routes=config.routes,
        upstreams=config.upstreams,
        advertise=config.advertise,
        tailscale_advertise=config.tailscale_advertise,
        lan_advertise=config.lan_advertise,
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
#:
#: ITS PARTNER IS `envpack.SMOKE_IMPORT`, which says what a built pack for each
#: of these names must be able to import before it becomes a release asset. It
#: lives over there because `envpack` cannot import this module without a cycle;
#: `tests/test_envpack.py` is what keeps the two from drifting apart (R1).
INSTALLER_FOR: dict[str, str] = {
    **{name: name for name in INSTALLABLE_JOB_TYPES},
    **{
        job_type: env
        for env, served in workerenv.JOB_TYPES_SERVED_BY_ENV.items()
        for job_type in served
    },
    # `pages` is a CAPABILITY CLASS and not a job type — PHASE3-VLM.md section
    # 1: *"there is no `vlm-pages` job type"*, dots.ocr is served through the
    # same `llm` proxy as every text model, on every backend. So it has no
    # installer of its own and never will, and naming it here is what turns
    # `crucible install pages` (which PHASE15-HOST.md 3.5 writes out) from
    # "there is no installer for 'pages'" into a sentence that says `llm`.
    "pages": "llm",
}


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
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force`"
        )
    if backend.kind == LLAMA_WINDOWS:
        return _install_llama_windows(config, backend, args)
    # THE DEFAULT IS A DOWNLOAD (PHASE14-ENVPACKS.md section 3.1). `--build` is
    # the developer's path and an ARGUMENT: nothing below chooses it because a
    # download failed, and every way a download can fail has a name of its own.
    if not args.build:
        return _install_from_pack(config, backend, args)
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


def _install_llama_windows(
    config: Config, backend: Backend, args: argparse.Namespace
) -> int:
    """`crucible install` on `llama-windows`. PHASE15-HOST.md 3.5 and 7.4 item 4.

    THERE IS NO ENV ON THIS BACKEND. `llm` (and therefore `pages`, which
    shares it) is served by `llama-server.exe` from the pinned llama.cpp
    release — the `engine` subject — so `install llm` fetches that and
    nothing else. The five Python job types are refused `needs_wsl` with
    `capability.NEEDS_WSL_REASON`, which is already the sentence their
    capability rows carry, so an operator reads one sentence and not two
    spellings of it.

    `--build` is refused rather than ignored: there is no recipe to build
    from, and a flag that silently did the download instead would be a flag
    whose name lies.
    """
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
    if args.build:
        return _fail(
            f"--build builds an env from a recipe, and {LLAMA_WINDOWS} has no "
            "env: its engine is llama.cpp's own release, fetched at a pinned "
            f"tag ({llamacpp.LLAMA_CPP_RELEASE}). Run without --build"
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
    return _capability_step(config, backend, args.job_type)


def _env_key_for(job_type: str, narrator_engine: str | None, backend_kind: str) -> str:
    """The directory under `~/.crucible/envs/` this job type installs into.

    Two modules own two halves of that answer — `workerenv` names its envs
    after the job type, `jobenv` names `tts`'s after the narrator engine — and
    the pack is named after the DIRECTORY, so this is where the question is
    asked once rather than in each of the two install paths.
    """
    if job_type in workerenv.WORKER_JOB_TYPES:
        if narrator_engine is not None:
            raise jobenv.EnvError(
                "--narrator-engine names which tts env to build and means "
                f"nothing for {job_type!r}, which has exactly one env per host"
            )
        # ASKED FOR ITS REFUSAL, not for its answer. A type with no recipe on
        # this backend — `align` and `asr` on the Mac — must be refused by the
        # module that knows WHY (CTranslate2 has no Metal backend, and
        # `envs/asr/mlx-darwin.md` is prose saying so). Asking the pack table
        # first would answer "there is no pack called 'align'", which is true
        # and tells its reader nothing they can act on.
        workerenv.recipe_for(job_type, backend_kind)
        return job_type
    spec = _env_spec(job_type, narrator_engine, backend_kind)
    jobenv.recipe_for(spec)
    return spec.key


def _install_from_pack(
    config: Config, backend: Backend, args: argparse.Namespace
) -> int:
    """`crucible install <type>` — the default: download, verify, unpack.

    The already-installed short-circuit is the same one `--build` has and for
    the same reason, one order of magnitude louder: re-running the command must
    not fetch eight gigabytes to arrive at the env that is already there.
    """
    try:
        key = _env_key_for(args.job_type, args.narrator_engine, backend.kind)
        target = envpack.target_for_env_key(key, backend.kind)
    except (jobenv.EnvError, workerenv.WorkerEnvError, envpack.PackError) as exc:
        return _fail(str(exc))

    if not args.force:
        existing = _env_status_for(config, backend, args.job_type, args.narrator_engine)
        if existing is not None and existing.installed:
            print(f"already installed: {existing.detail}")
            # AND THE STAMP, WHICH IS THE OTHER HALF OF "INSTALLED". The env
            # holding what the recipe pins says nothing about whether the stamp
            # still names the recipe bytes in this build; when it does not,
            # `crucible doctor` reports `pack_recipe_drift` and sends the
            # operator here. THIS is the line that has to answer, because this
            # is the path `crucible install` actually takes - `install_env` is
            # only reached under `--build`, so a drift branch there alone is one
            # the operator never runs.
            if args.job_type not in workerenv.WORKER_JOB_TYPES:
                try:
                    spec = _env_spec(
                        args.job_type, args.narrator_engine, backend.kind
                    )
                    if jobenv.reconcile_stamp(
                        config.home, spec, backend.kind,
                        on_line=lambda line: print(f"  {line}"),
                    ):
                        print("stamp corrected: no rebuild needed")
                except jobenv.EnvError as exc:
                    return _fail(str(exc))
            return _capability_step(config, backend, *_types_served(args.job_type))

    print(f"backend: {backend.kind}")
    print(f"pack:    {target.name} {VERSION}")
    print(f"target:  {target.env_dir(config.home)}")
    last = [0.0]

    def on_progress(done: int, total: int | None, name: str) -> None:
        # Two audiences, two lines (R4). The sentinel is the install TASK's wire
        # and carries no prose; the human line is throttled prose and carries no
        # contract. `crucible/envpack.py` owns the sentinel's shape.
        print(envpack.progress_line(done, total, name), flush=True)
        now = time.monotonic()
        if now - last[0] < 1.0:
            return
        last[0] = now
        share = f"{100 * done / total:.0f}%" if total else "?"
        print(f"  {name}: {done / 1e9:.2f} GB ({share})", flush=True)

    try:
        entry = envpack.install_pack(
            config.home,
            target,
            VERSION,
            location=args.manifest_url,
            on_line=lambda line: print(f"  {line}", flush=True),
            on_progress=on_progress,
        )
    except envpack.PackError as exc:
        return _fail(f"{exc.code}: {exc.message}")

    status = _env_status_for(config, backend, args.job_type, args.narrator_engine)
    if status is None or not status.installed:
        detail = "no status could be read" if status is None else status.detail
        return _fail(
            f"the pack unpacked but the env did not come out installed: {detail}"
        )
    print(f"installed from pack {entry.sha256[:12]}: {status.detail}")
    return _capability_step(config, backend, *_types_served(args.job_type))


def _types_served(job_type: str) -> tuple[str, ...]:
    """Which capability flags this install decides. `rvc` decides two."""
    return workerenv.JOB_TYPES_SERVED_BY_ENV.get(job_type, (job_type,))


def _env_status_for(
    config: Config, backend: Backend, job_type: str, narrator_engine: str | None
) -> "jobenv.EnvStatus | workerenv.EnvStatus | None":
    """This job type's env status, from whichever module owns it, or None.

    None means the question itself could not be asked — no recipe for this
    backend, an engine that does not exist — and the caller says so rather than
    reporting a missing env, which would be a different fact.
    """
    try:
        if job_type in workerenv.WORKER_JOB_TYPES:
            return workerenv.env_status(config.home, job_type, backend.kind)
        spec = _env_spec(job_type, narrator_engine, backend.kind)
        return jobenv.env_status(config.home, spec, backend.kind)
    except (jobenv.EnvError, workerenv.WorkerEnvError):
        return None


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
    headline = workerenv.headline_package(args.job_type, backend.kind)
    for name in sorted(status.packages):
        # The headline plus the packages whose version is the thing most
        # likely to be quietly wrong. `mlx` is here for the same reason
        # `ctranslate2` is: it is the engine under the headline, and an
        # mlx that resolved differently is a different numerical path.
        if name in (headline, "ctranslate2", "mlx", "numpy", "onnxruntime"):
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


# ------------------------------------------------------------------ envpack


def cmd_envpack_list(args: argparse.Namespace) -> int:
    """Every (pack, backend) a tag carries. `scripts/release.sh` reads this."""
    rows = [
        {"name": name, "backend": backend}
        for name, backend in envpack.every_pack()
        if args.backend is None or backend == args.backend
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        print(f"{row['name']}\t{row['backend']}")
    return EXIT_OK


def cmd_envpack_build(args: argparse.Namespace) -> int:
    """`crucible envpack build <name>` — produce (or check) one pack.

    No `--backend`: pip installs wheels for the machine it runs on, so a pack
    is built on the backend it targets and nowhere else (section 3.3). The
    build host's platform is the answer, and a host that is neither is refused
    by name rather than producing a tree of wrong-platform wheels.
    """
    try:
        backend_kind = envpack.build_backend_kind()
        target = envpack.pack_target(args.name, backend_kind)
    except envpack.PackError as exc:
        return _fail(f"{exc.code}: {exc.message}")
    out = Path(args.out).expanduser().resolve()
    print(f"backend: {backend_kind}")
    print(f"recipe:  {target.recipe}")
    print(f"out:     {out}")

    if args.check:
        try:
            entry = envpack.check_pack(out, target, VERSION)
        except envpack.PackError as exc:
            return _fail(f"{exc.code}: {exc.message}")
        print(
            f"ok: {entry.name}/{entry.backend} {entry.bytes / 1e9:.2f} GB in "
            f"{len(entry.parts)} part(s), sha {entry.sha256[:12]}, recipe "
            f"{entry.recipe_sha256[:12]}"
        )
        return EXIT_OK

    try:
        entry = envpack.build_pack(
            target, VERSION, out, on_line=lambda line: print(f"  {line}", flush=True)
        )
    except envpack.PackError as exc:
        return _fail(f"{exc.code}: {exc.message}")
    print(json.dumps(entry.to_dict(), indent=2))
    return EXIT_OK


def _env_spec(
    job_type: str, narrator_engine: str | None, backend_kind: str
) -> jobenv.EnvSpec:
    """The env `crucible install <job type>` builds on this host.

    `tts` needs a second word on `cuda-linux` — the env there is named per
    narrator engine — so the flag is required for it and refused for `llm`,
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


def cmd_remove(args: argparse.Namespace) -> int:
    """`crucible remove <kind> <id>` — PHASE15-HOST.md 3.5a, from a terminal.

    REFUSES IDENTICALLY TO THE DOOR, and it does so by asking the same
    questions in the same order: an unknown kind or id first (true whatever
    this server is doing), then not-installed, then in-use. What it CANNOT
    ask is whether a running server holds the subject — that is a fact about
    a process this command is not inside, and the names it would need
    (`Residency`, `Leases`, the task store) live in one. So it asks the two
    it can and says so: on a machine with a server running, the door is the
    one to use, and `DELETE /v1/catalog/{kind}/{id}` is what the host calls.
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
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force`"
        )
    subject = catalog.find(config, backend, args.kind, args.id)
    if subject is None:
        return _fail(
            f"subject_unknown: this server has no {args.kind} called "
            f"{args.id!r} for {backend.kind}. `crucible catalog` lists every "
            "subject it can hold"
        )
    found = subject.installed()
    if found is None:
        return _fail(
            f"subject_not_installed: {args.kind} {args.id!r} is not installed "
            "on this server, so there is nothing to remove"
        )
    try:
        gone = subject.remove()
    except weights.RemoveFailed as exc:
        return _fail(f"subject_remove_failed: {exc}")
    except CrucibleError as exc:
        return _fail(f"subject_remove_failed: {type(exc).__name__}: {exc}")
    if args.json:
        print(json.dumps(
            {
                "kind": args.kind,
                "id": args.id,
                "path": str(gone),
                "bytes_freed": found.bytes,
            },
            indent=2,
        ))
    else:
        print(f"removed:  {args.kind} {args.id}")
        print(f"path:     {gone}")
        print(f"freed:    {found.bytes / 1e9:.2f} GB")
    return EXIT_OK


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


# ------------------------------------------------------------------ denoise


def cmd_denoise_list(args: argparse.Namespace) -> int:
    """Every denoise manifest this build ships and where it stands here.

    Its own command rather than a row in `crucible models list`, for `crucible
    rvc list`'s reason one job type along: a separator's weights are two named
    files placed under names an engine resolves by, not a repo snapshot, so
    `models pull` could not fetch one — and the ids are their own namespace.
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = denoisemodels.load_all_denoise_manifests()
    except denoisemodels.DenoiseManifestError as exc:
        return _fail(str(exc))
    root = denoisemodels.denoise_models_root(config.home)
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
        found = denoisemodels.installed(config.home, manifest, spec)
        absent = denoisemodels.missing(config.home, manifest)
        # Stamped and present are different facts and this prints both. A
        # stamp with a file missing beside it is not installed; two files
        # somebody placed by hand are usable and unstamped, which is the state
        # every host was in before this command existed.
        if found is not None:
            detail = f"{found.bytes / 1e9:.2f} GB at {found.path}"
        elif not absent:
            detail = (
                f"both files are in {root} but Crucible did not place them — "
                f"`{denoisemodels.PULL_COMMAND} {manifest.id} --force` to pin them"
            )
        else:
            detail = f"not pulled — `{denoisemodels.PULL_COMMAND} {manifest.id}`"
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "model_filename": manifest.model_filename,
                "config_filename": manifest.config_filename,
                "primary_stem": manifest.primary_stem,
                "sample_rate": manifest.sample_rate,
                "backend_supported": True,
                "installed": found is not None,
                "present": not absent,
                "missing": absent,
                "root": str(root),
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "model_path": spec.model_path,
                "config_path": spec.config_path,
                "total_bytes": spec.total_bytes,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "detail": detail,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else (
                "unstamped" if row["present"] else "not pulled"
            )
        )
        print(f"{row['id']:<22} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_denoise_pull(args: argparse.Namespace) -> int:
    """`crucible denoise pull <id>` — the checkpoint and its config, at the pin.

    Both files, both digests, one revision, into the flat directory
    audio-separator reads by name. The layout is `crucible/denoisemodels.py`'s
    and the job reads the same function, so what this places is what a job
    looks for (ARCHITECTURE.md R1).
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifest = denoisemodels.load_denoise_manifest(args.model)
    except denoisemodels.DenoiseManifestError as exc:
        return _fail(str(exc))
    if not manifest.supports(backend.kind):
        return _fail(
            f"denoise model {args.model!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(
        f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]}, 2 file(s), "
        f"{spec.total_bytes / 1e9:.2f} GB for {backend.kind}"
    )
    for entry in denoisemodels.model_files(manifest, spec):
        print(f"  {entry.target} — {entry.why}")
    try:
        result = denoisemodels.pull(
            config,
            manifest,
            spec,
            force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    absent = denoisemodels.missing(config.home, manifest)
    if absent:
        # Unreachable unless something removed a file between the place and
        # this read; said out loud rather than reported as success, because the
        # next thing to look at this tree is a job that will fail inside
        # audio-separator. `rvc pull-base`'s rule, and its reason.
        return _fail(
            f"the pull finished but {sorted(absent)} are not under "
            f"{denoisemodels.denoise_models_root(config.home)}"
        )
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
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


def _llama_engine_report(
    report: dict[str, Any], config: Config, backend: Backend
) -> dict[str, Any]:
    """`llama-windows`'s llm engine row: llama.cpp at the pinned tag.

    Shaped like `_env_report`'s answer — `installed` and `detail`, and a
    problem appended when it is not ready — so the printer below and every
    JSON reader treat the two backends' engines the same way. What it is NOT
    is an env: no recipe, no packages, no `pack_sha256`, and the provenance
    that matters is the tag and which build this machine takes.
    """
    build = llamacpp.build_for(backend.gpu.vendor)
    found = llamacpp.installed(config, build)
    entry: dict[str, Any] = {
        "installed": found is not None,
        "engine": "llama-server",
        "tag": llamacpp.LLAMA_CPP_RELEASE,
        "build": build,
        "path": str(llamacpp.engine_dir(config)),
        "detail": llamacpp.doctor_line(config, backend.gpu.vendor),
    }
    if found is None:
        report["problems"].append(f"llm_env: {entry['detail']}")
    else:
        entry["bytes"] = found.bytes
        entry["pulled"] = found.pulled
    return entry


def _env_report(
    report: dict[str, Any], label: str, home: Path, spec: jobenv.EnvSpec,
    backend_kind: str,
) -> dict[str, Any]:
    """One env's status, appending a problem to the report when it is not ready."""
    try:
        status = jobenv.env_status(home, spec, backend_kind)
        recipe = jobenv.recipe_for(spec)
    except jobenv.EnvError as exc:
        report["problems"].append(f"{label}: {exc}")
        return {"installed": False, "detail": str(exc)}
    if not status.installed:
        report["problems"].append(f"{label}: {status.detail}")
    entry = status.to_dict()
    entry["provenance"] = _provenance(report, label, status, recipe)
    return entry


def _provenance(
    report: dict[str, Any],
    label: str,
    status: "jobenv.EnvStatus | workerenv.EnvStatus",
    recipe: Path,
) -> dict[str, Any]:
    """Where this env came from, and whether its recipe has moved since.

    PHASE14-ENVPACKS.md section 5: `doctor` reports each env's pack sha beside
    its recipe hash. The comparison is the point rather than the display — an
    env installed from a pack that was built from a recipe this checkout no
    longer has is the drift `crucible install` refuses at download time, and
    an env already on disk has nobody else to notice it.
    """
    here = jobenv.recipe_sha256(recipe) if recipe.is_file() else None
    entry = {
        "source": "pack" if status.pack_sha256 else "built",
        "pack_sha256": status.pack_sha256,
        "recipe_sha256": status.recipe_sha256,
        "recipe_sha256_now": here,
        "recipe": recipe.name,
        "drifted": bool(
            status.recipe_sha256 is not None
            and here is not None
            and status.recipe_sha256 != here
        ),
    }
    if entry["drifted"]:
        report["problems"].append(
            f"{label}: pack_recipe_drift — this env was installed from a "
            f"{recipe.name} hashing {status.recipe_sha256[:12]} and the one in "
            f"this build hashes {here[:12]}. Re-run its `crucible install`, "
            "which now either corrects the stamp (when the edit was to "
            "comments or to pins it can re-verify package by package) or "
            "names the change that needs `--force`, and why"
        )
    return entry


def _provenance_line(entry: dict[str, Any]) -> str:
    """The one-line form `crucible doctor` prints after an env's detail."""
    if entry["source"] == "pack":
        where = f"pack {entry['pack_sha256'][:12]}"
    else:
        where = "built here"
    if entry["recipe_sha256"] is None:
        recipe = f"{entry['recipe']} hash not recorded (installed before 0.6.0)"
    elif entry["drifted"]:
        recipe = (
            f"{entry['recipe']} {entry['recipe_sha256'][:12]} != "
            f"{entry['recipe_sha256_now'][:12]} HERE"
        )
    else:
        recipe = f"{entry['recipe']} {entry['recipe_sha256'][:12]}"
    return f"{where}, {recipe}"


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
        # THE TWO PATHS, because the Mac audit of 2026-09-14 found the same
        # message twice and only one of the two readings was a defect. A
        # `crucible doctor` over a non-login `ssh mac '<cmd>'` reported
        # `job tts: NOT READY — there is no ffmpeg on PATH` while the service
        # was healthy: ffmpeg was at /opt/homebrew/bin, the plist carried that
        # directory, and `launchctl print` confirmed the running process had
        # it. The doctor was right about the shell it was in and silent about
        # the one that matters.
        #
        # Crucible WROTE the service's PATH, so it can read it back
        # (`service.read_recorded_path`) and put the two side by side. `agree`
        # is computed rather than left to the reader, and `null` when there is
        # nothing to compare — three states, not a boolean that would make "no
        # service" read as "they differ".
        "path": None,
        "problems": [],
    }

    try:
        backend = detect_backend()
        report["backend"] = backend.to_dict()
    except NoViableBackend as exc:
        report["problems"].append(f"no_viable_backend: {exc.reason}")
        backend = None

    shell_path = hosttools.search_path()
    path_report: dict[str, Any] = {
        "shell": shell_path,
        "service": None,
        "mechanism": None,
        "definition": None,
        "agree": None,
    }
    if backend is not None:
        try:
            mechanism = service.mechanism_for(backend.kind)
        except service.ServiceError:
            # A backend with no supervisor is not a defect here; `crucible
            # service` is the door that refuses it by name.
            mechanism = None
        if mechanism is not None:
            recorded = service.read_recorded_path(mechanism, service.user_home())
            path_report["mechanism"] = mechanism
            path_report["definition"] = str(
                service.definition_path(mechanism, service.user_home())
            )
            path_report["service"] = recorded
            if recorded is not None:
                path_report["agree"] = recorded == shell_path
    report["path"] = path_report

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
        # NOT ON WIN32. A Windows file has no POSIX mode: `os.chmod` there
        # sets one read-only bit and `stat` reports 0o666 whatever the ACL
        # says, so this check reads a number the OS does not enforce and
        # reports a problem on every healthy Windows server. What restricts a
        # file there is its ACL (`crucible/pairing.py`'s `icacls_argv`), and a
        # doctor line about it is owed — recorded as owed rather than faked
        # with a number that means nothing.
        if sys.platform != "win32" and mode != "0o600":
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
            # TWO SHAPES OF ENGINE, and `doctor` names each in its own terms
            # (PHASE15-HOST.md 3.5, 7.4 item 5). On `cuda-linux` and
            # `mlx-darwin` the llm engine is a Python env with a recipe and a
            # provenance; on `llama-windows` it is llama.cpp's own release at
            # a pinned tag, which has no recipe and no packages and whose
            # provenance is the tag and the digests. Asking `jobenv` for an
            # env that cannot exist was how this crashed the first time it ran
            # on a real Windows box.
            if backend.kind == LLAMA_WINDOWS:
                report["llm_env"] = _llama_engine_report(report, config, backend)
            else:
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
                entry = worker_env.to_dict()
                entry["provenance"] = _provenance(
                    report,
                    f"{job_type}_env",
                    worker_env,
                    workerenv.recipe_for(job_type, backend.kind),
                )
                report["worker_envs"].append(entry)
                if not worker_env.installed:
                    report["problems"].append(f"{job_type}_env: {worker_env.detail}")
            except workerenv.WorkerEnvError as exc:
                report["worker_envs"].append(
                    {"job_type": job_type, "installed": False, "detail": str(exc)}
                )
                report["problems"].append(f"{job_type}_env: {exc}")
        if config.enable_tts:
            # One row per narrator engine, because on cuda-linux each is its
            # own venv and a voice load picks by its manifest's
            # `narrator_engine`. On mlx-darwin every name resolves to the same
            # env, and the rows say so by carrying the same path.
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
            # AND THE TWO CUDA SYMLINKS, on cuda-linux, WHEN THERE IS AN ENV TO
            # ASK ABOUT. Reported in the same rows for the same reason, and
            # needed MORE here than the patches are: the SGLang stack has no
            # site-packages patches at all, so without these this section would
            # be empty on the very host whose env holds the one thing that can
            # be silently missing.
            #
            # GATED ON THE ENV EXISTING, and that gate is the point rather than
            # an optimisation. With no `tts` env installed the links cannot be
            # there, and saying so would raise TWO problems — "lib64 is missing"
            # and "lib/libcudart.so is missing" — for one cause the env row
            # already states in full ("no venv at ... run `crucible install
            # tts`"). Three sentences about one fact is how a reader ends up
            # chasing the wrong one. The patches avoid this a different way
            # (`not_applicable`, when the recipe does not install what they
            # edit); this recipe DOES pin nvidia-cuda-runtime-cu13, so the
            # honest answer is not "not applicable" but "not yet asked".
            patched_env = jobenv.env_dir(config.home, patched_spec)
            if (
                backend.kind == "cuda-linux"
                and narratorpatches.site_packages(patched_env) is not None
            ):
                report["narrator_patches"].extend(
                    narratorpatches.check_cuda_toolkit_links(patched_env)
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
        path_entry = report["path"]
        if path_entry is not None:
            print(f"PATH (this shell):   {path_entry['shell'] or '(empty)'}")
            if path_entry["service"] is None:
                # Named, not omitted. "No service is installed" and "the
                # service has no PATH" are different facts and a missing line
                # would read as either.
                where = path_entry["definition"]
                print(
                    "PATH (the service):  none recorded — no "
                    f"{path_entry['mechanism'] or 'service'} definition at {where}"
                    if where
                    else "PATH (the service):  none recorded"
                )
            else:
                print(f"PATH (the service):  {path_entry['service']}")
                if path_entry["agree"] is False:
                    # Not a PROBLEM: they differ on every correctly installed
                    # host, because a login shell has more than a launchd
                    # agent's recorded PATH needs. It is said out loud because
                    # every line below this one was measured in the FIRST of
                    # the two.
                    print(
                        "note:    the two differ, which is normal. Every line "
                        "below is what THIS shell can see; the service sees "
                        "the second one"
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
            # 3.5's line: `backend: llama-windows on windows/x86_64 —
            # llama.cpp <tag> (cuda-12.4 | cpu)`. Printed from the ROW rather
            # than composed here, so the JSON and the text cannot disagree.
            label = "engine " if env.get("engine") == "llama-server" else "llm env"
            print(f"{label}: {mark} — {env['detail']}")
            if "provenance" in env:
                print(f"         {_provenance_line(env['provenance'])}")
        for worker_env in report["worker_envs"]:
            mark = "ready" if worker_env["installed"] else "NOT READY"
            print(
                f"{worker_env['job_type']} env: {mark} — {worker_env['detail']}"
            )
            if "provenance" in worker_env:
                print(f"         {_provenance_line(worker_env['provenance'])}")
        for engine, entry in sorted(report["tts_envs"].items()):
            mark = "ready" if entry["installed"] else "NOT READY"
            print(f"tts env ({engine}): {mark} — {entry['detail']}")
            if "provenance" in entry:
                print(f"         {_provenance_line(entry['provenance'])}")
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


# ---------------------------------------------------------------- uninstall


def cmd_uninstall(args: argparse.Namespace) -> int:
    """`crucible uninstall` — `crucible/uninstall.py` is where the whole of it is.

    This function does three things and no fourth: build the plan, run it
    unless `--dry-run`, and print it. Every decision — the order, what is
    kept, what is refused by name — belongs to the module, because the module
    is what the tests exercise and what `install.sh --uninstall` reaches
    through this verb.

    **The home is `crucible_home()` and never a flag.** `CRUCIBLE_HOME` is the
    one owner of where a server's state is, on every platform
    (`crucible/config.py`), and a `--home` here would be a second way to name
    it — which on a command that deletes directories is the difference between
    one answer and two.

    **The backend is READ, not detected.** `detect_backend()` probes a card,
    and an uninstall must run on a machine whose driver has already gone, whose
    config has already been half-removed by an interrupted run, or which simply
    has no GPU free tonight. What the plan reports is `[backend] kind` out of
    the config when there is a config, and `null` when there is not.
    """
    try:
        home = crucible_home()
        built = uninstall.plan(
            home=home,
            platform=sys.platform,
            env=os.environ,
            runner=service.subprocess_runner,
            purge_weights=args.purge_weights,
            wsl_too=args.wsl_too,
        )
    except CrucibleError as exc:
        return _fail(str(exc))

    if not args.dry_run:
        built = uninstall.run(built)

    if args.json:
        print(json.dumps(built.to_dict(), indent=2))
        return EXIT_OK if not built.fatal else EXIT_REFUSED

    print(f"home:      {built.home}")
    print(f"platform:  {built.platform} ({built.mechanism})")
    print(
        "backend:   "
        + (
            built.backend_kind
            if built.backend_kind is not None
            else "unrecorded — this home has no readable config.toml"
        )
    )
    print(
        "mode:      "
        + (
            "DRY RUN — nothing below has been touched"
            if built.dry_run
            else "live"
        )
    )
    print(f"weights:   {'PURGED' if built.purge_weights else 'kept unless named below'}")
    print("")
    for step in built.steps:
        size = f"  [{uninstall.gib(step.bytes)}]" if step.bytes else ""
        mark = {
            uninstall.REMOVE: "remove",
            uninstall.STOP: "stop  ",
            uninstall.KEEP: "keep  ",
        }[step.action]
        if step.refused is not None:
            mark = "SKIP  " if not step.refused.fatal else "FAILED"
        print(f"{mark}  {step.name:<26} {step.target}{size}")
        print(f"          {step.what}")
        if step.refused is not None:
            print(f"          {step.refused.code}: {step.refused.message}")
        for line in step.detail:
            print(f"          {line}")
    kept = built.kept()
    print("")
    if kept["weights_bytes"]:
        print(
            f"kept:      {uninstall.gib(kept['weights_bytes'])} of weights. "
            "`--purge-weights` is what deletes them."
        )
    if not built.dry_run:
        print(f"freed:     {uninstall.gib(built.removed_bytes())}")
    if built.fatal:
        return _fail(
            "uninstall_incomplete: "
            + "; ".join(
                f"{step.name} — {step.refused.code}"
                for step in built.fatal
                if step.refused is not None
            )
            + ". Everything else was removed"
        )
    return EXIT_OK


# -------------------------------------------------------------------- token


def _write_pairing_file(home: Path, *, name: str, port: int, token: str) -> Path:
    """`<CRUCIBLE_HOME>/pairing`, through the ONE writer (`crucible/pairing.py`).

    PHASE15-HOST.md section 3.6. The file holds the LOOPBACK line whatever the
    server is bound to — it answers *"an app on THIS machine wants in"*, and
    the answer to that is never a LAN address — so this is where (name, port,
    token) becomes that line; `pairing.write_pairing_file` owns everything
    after it, including the Windows ACL.
    """
    return pairing.write_pairing_file(
        home, pairing.pairing_line(name, f"http://{DEFAULT_HOST}:{port}", token)
    )


def _sync_pairing_file(config: Config) -> None:
    """Write `<home>/pairing` when it is absent or does not match the config.

    PHASE15-HOST.md 3.6, as amended: `crucible serve` is the third writer,
    and it is the one that covers a server that already existed. Comparison
    is on the LINE, which is exactly the four facts an app needs — name,
    host, port, token — so there is no second notion of "matches" to keep in
    step with the writer.
    """
    wanted = pairing.pairing_line(
        config.name, f"http://{DEFAULT_HOST}:{config.port}", config.token
    )
    if pairing.read_pairing_file(config.home) == wanted:
        return
    written = pairing.write_pairing_file(config.home, wanted)
    print(f"pairing:  {written} ({_pairing_permission(written)})")


def _pairing_permission(path: Path) -> str:
    """What restricts the file, said in the platform's own vocabulary.

    A Windows file has no mode, and printing `config_mode`'s answer there
    would report a number the OS does not enforce.
    """
    if sys.platform == "win32":
        return "ACL: this user only"
    return f"mode {config_mode(path)}"


def _pairing_lines(
    name: str, host: str, port: int, token: str, advertise: tuple[str, ...] = ()
) -> list[str] | str:
    """The lines, or the sentence saying why there are none.

    PHASE13-OPERATOR.md section 3.1. A refusal is returned rather than raised
    because the two callers want different things done with it: `token --url`
    has nothing else to print and exits 1, while `init` and `service install`
    have already succeeded and merely have one fewer thing to tell the
    operator.

    **The loopback line comes first, always** (PHASE15-HOST.md section 3.6).
    It is what `<CRUCIBLE_HOME>/pairing` holds, and *"`crucible token --url`
    prints the same"* is only true if it is printed. On a `127.0.0.1` bind it
    IS `reachable_urls`' one entry and is printed once; on a wildcard bind
    `reachable_urls` has no loopback entry at all, and without this an app on
    the server's own machine would be handed whichever interface the OS listed
    first.
    """
    loopback = pairing.pairing_line(name, f"http://{DEFAULT_HOST}:{port}", token)
    try:
        urls = pairing.reachable_urls(host, port, advertise)
    except InterfaceError as exc:
        return (
            f"this host will not list its own interfaces, so there is no "
            f"pairing line for a wildcard bind: {exc}"
        )
    if not urls:
        return (
            f"bound to {host} and this host has no non-loopback IPv4 address, "
            "so nothing else can reach it yet"
        )
    lines = [loopback]
    for line in pairing.pairing_lines(name, urls, token):
        if line not in lines:
            lines.append(line)
    return lines


def _print_pairing(
    name: str, host: str, port: int, token: str, advertise: tuple[str, ...] = ()
) -> None:
    """The one block `init`, `service install` and `token --url` all print.

    Owen, 2026-09-14: nobody types a token twice. The line carries the name,
    the address and the secret, so the person setting up BookForge pastes one
    string into one field instead of reading three values off a terminal.
    """
    result = _pairing_lines(name, host, port, token, advertise)
    if isinstance(result, str):
        print(f"pairing: {result}", file=sys.stderr)
        return
    print("pairing: paste one of these into an app's Crucible server door —")
    for line in result:
        print(f"  {line}")


def cmd_token(args: argparse.Namespace) -> int:
    """`crucible token --show` prints the secret; `--url` prints the whole door.

    `--url` needs no `--show`, and that is not laxity: the flag's name says it
    prints a URL, and the pairing line's whole purpose is to be handed to an
    app. Requiring two flags to print one string would be a ceremony that
    protects nothing — the token is already behind a file mode 0600 and a
    terminal somebody is sitting at.
    """
    if not args.show and not args.url:
        return _fail("pass --show to print the bearer token, or --url to print "
                     "the pairing line an app's connect door takes")
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    if args.show:
        print(config.token)
    if args.url:
        result = _pairing_lines(
            config.name, config.host, config.port, config.token,
            config.advertise + config.tailscale_advertise + config.lan_advertise
        )
        if isinstance(result, str):
            return _fail(result)
        for line in result:
            print(line)
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
    from .local import add_parser as add_local_parser
    add_local_parser(subparsers)

    init = subparsers.add_parser(
        "init", help="detect the backend, mint a token, write config.toml"
    )
    init.add_argument("--force", action="store_true", help="replace an existing config")
    init.add_argument(
        "--backend",
        default=None,
        choices=sorted(BACKEND_KINDS),
        help=(
            "the backend this host is EXPECTED to be, checked against what it "
            "detects. A backend runs where its engine runs and nowhere else "
            "(PHASE15-HOST.md 3.5), so this never chooses one — it refuses "
            "backend_not_here when the two disagree. `crucible host` passes "
            "--backend llama-windows"
        ),
    )
    init.add_argument("--name", default=None, help="server name (default crucible@<hostname>)")
    init.add_argument("--host", default=DEFAULT_HOST, help=f"default bind host ({DEFAULT_HOST})")
    init.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default port ({DEFAULT_PORT})")
    init.add_argument(
        "--token",
        default=None,
        help=(
            "use this bearer token instead of minting one. For an installer "
            "that mints on its own side (@crucible/bootstrap). It still appears "
            "in the pairing line this command ends with, which is the point of "
            "that line"
        ),
    )
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

    init.add_argument(
        "--config-from",
        metavar="FILE",
        help=(
            "take the token, [routes] and [upstreams] out of this TOML file "
            "instead of minting a token (PHASE15-HOST.md 4.3). The host writes "
            "it at 0600 when it moves a Windows Crucible into the WSL guest and "
            "deletes it after, so every app that paired stays paired"
        ),
    )

    install = subparsers.add_parser(
        "install",
        help="download this job type's published env pack and unpack it "
        "(--build installs its recipe with pip instead)",
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
        "--force", action="store_true", help="rebuild the env from scratch"
    )
    install.add_argument(
        "--verbose", action="store_true", help="echo pip's output line by line"
    )
    install.add_argument(
        "--build",
        action="store_true",
        help=(
            "build the env here from its recipe with pip, instead of "
            "downloading the published pack. The developer's path, and an "
            "argument: nothing chooses it because a download failed"
        ),
    )
    install.add_argument(
        "--manifest-url",
        default=None,
        help=(
            "read envpacks.json from here instead of this version's release "
            f"(also ${envpack.MANIFEST_URL_ENV}). For a mirror or a test; "
            "parts are fetched from beside it"
        ),
    )
    install.set_defaults(func=cmd_install)

    envpack_parser = subparsers.add_parser(
        "envpack",
        help="build and check the environment packs a release carries",
        description=(
            "A pack is a relocatable CPython with one recipe installed into "
            "it (PHASE14-ENVPACKS.md). `crucible install` downloads one; this "
            "is the verb that produces one, on the backend it targets."
        ),
    )
    envpack_commands = envpack_parser.add_subparsers(
        dest="envpack_command", required=True
    )
    # WIN32 RUNS THIS ONE, like every other verb since the gate in `main()`
    # went. PHASE15-HOST.md 4.4: the `host` pack is built by `crucible envpack
    # build host` on a `windows-latest` runner, which is the only way the
    # Windows pack can exist at all — pip resolves wheels for the machine it
    # runs on. `build_backend_kind()` is what refuses the wrong platform, by
    # name and with the three backends in the sentence.

    envpack_list = envpack_commands.add_parser(
        "list", help="every (pack, backend) a tag carries"
    )
    envpack_list.add_argument(
        "--backend",
        default=None,
        choices=sorted(envpack.STANDALONE_PYTHON),
        help="only this backend's packs",
    )
    envpack_list.add_argument("--json", action="store_true", help="machine-readable")
    envpack_list.set_defaults(func=cmd_envpack_list)

    envpack_build = envpack_commands.add_parser(
        "build",
        help="build one pack for THIS machine's backend, smoke-test it, and "
        "record it in <out>/envpacks.json",
    )
    envpack_build.add_argument(
        "name",
        help="the pack name — `crucible envpack list` prints them. It is the "
        "env DIRECTORY's name, so on cuda-linux the tts pack is 'tts-higgs-v3'",
    )
    envpack_build.add_argument(
        "--out",
        default="packs",
        help="where the parts and envpacks.json are written (default ./packs)",
    )
    envpack_build.add_argument(
        "--check",
        action="store_true",
        help="build nothing: verify the parts already in --out against "
        "envpacks.json and against this checkout's recipe",
    )
    envpack_build.set_defaults(func=cmd_envpack_build)

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

    remove = subparsers.add_parser(
        "remove",
        help="delete an installed subject's files (PHASE15-HOST.md 3.5a)",
    )
    remove.add_argument(
        "kind",
        choices=list(catalog.KINDS),
        help="the subject kind, as `crucible catalog` and GET /v1/catalog spell it",
    )
    remove.add_argument("id", help="the subject id, e.g. qwen3.5-9b")
    remove.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    remove.set_defaults(func=cmd_remove)

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

    denoise = subparsers.add_parser(
        "denoise", help="list and pull separator checkpoints for the denoise job"
    )
    denoise_commands = denoise.add_subparsers(dest="denoise_command", required=True)

    denoise_list = denoise_commands.add_parser(
        "list", help="every denoise manifest this build ships and where it stands here"
    )
    denoise_list.add_argument("--json", action="store_true", help="machine-readable")
    denoise_list.set_defaults(func=cmd_denoise_list)

    denoise_pull = denoise_commands.add_parser(
        "pull",
        help="fetch a separator's checkpoint and its config at the manifest's "
        "pinned revision, into the directory audio-separator reads by name",
    )
    denoise_pull.add_argument(
        "model", help="the Crucible denoise id, e.g. denoise-roformer"
    )
    denoise_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    denoise_pull.set_defaults(func=cmd_denoise_pull)

    host_parser = subparsers.add_parser(
        "orchestrator",
        # `host` KEPT, and it is the spelling the installed Startup shortcut
        # uses (PHASE17-ORCHESTRATOR.md section 7). Deprecated in the doc, not
        # in code, so tonight's tray survives a pack rebuild.
        aliases=["host"],
        help="win32 only: the tray that manages this machine's engine",
        description=(
            "The Windows ORCHESTRATOR (PHASE15-HOST.md section 4, PHASE17): a "
            "notification-area icon that boots this machine's engine at login, "
            "claims it, watches it, restarts it, and runs the move from the "
            "Windows engine to WSL2 when the operator page asks. It serves zero "
            "job types and carries no data — control is Windows's, data is the "
            "card's. Refused `host_windows_only` on Linux and macOS, where the "
            "service manager already supervises the server. `crucible host` is "
            "the same verb and is deprecated."
        ),
    )
    host_parser.add_argument(
        "--install-startup",
        action="store_true",
        help="write the Startup shortcut and exit (this verb OWNS that file)",
    )
    host_parser.add_argument(
        "--remove-startup",
        action="store_true",
        help="delete the Startup shortcut and exit",
    )
    host_parser.set_defaults(func=cmd_orchestrator)
    host_parser.add_argument("--headless", action="store_true", help="Run the controller independently of the tray")

    serve = subparsers.add_parser("serve", help="run the API in the foreground")
    serve.add_argument("--host", default=None, help="bind host (default from config)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    serve.add_argument("--log-level", default="info", help="uvicorn log level")
    serve.add_argument("--controller-stdin", action="store_true", help=argparse.SUPPRESS)
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

    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="undo an install, in the inverse order; weights are KEPT unless "
        "--purge-weights",
        description=(
            "The exact inverse of `install.sh` / `crucible install`, step by "
            "named step: stop the server, remove the service, remove the job "
            "envs, the pairing file, the config and the working state — and "
            "then keep the weights, which are the expensive part "
            "(PHASE15-HOST.md 3.5), unless --purge-weights says otherwise. "
            "It asks nothing: the flags decide. It removes nothing outside "
            "$CRUCIBLE_HOME and the service entry it wrote, nothing it cannot "
            "name, and never the relocatable interpreter it is running from — "
            "`install.sh --uninstall` removes that after this returns."
        ),
    )
    uninstall_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every step and touch nothing. The plan is the SAME object "
        "the real run performs, so there is no second description of what "
        "would happen",
    )
    uninstall_parser.add_argument(
        "--purge-weights",
        action="store_true",
        help=(
            "also delete the six subject directories — models, voices, rvc, "
            "rvc-base, denoise-models, engines. Tens of gigabytes, and a "
            "reinstall re-downloads every byte"
        ),
    )
    uninstall_parser.add_argument(
        "--wsl-too",
        action="store_true",
        help=(
            f"win32 only: first run the guest's own `crucible uninstall` inside "
            f"the {uninstall.CRUCIBLE_DISTRO!r} distro, with these same flags. "
            "Refused by name when that distro is not there. The distro itself "
            "is never unregistered — every other distro on the machine is "
            "yours, and so is that decision"
        ),
    )
    uninstall_parser.add_argument(
        "--json", action="store_true", help="machine-readable; the shape an app reads"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall)

    doctor = subparsers.add_parser("doctor", help="probe the host and the job types")
    doctor.add_argument("--json", action="store_true", help="machine-readable report")
    doctor.set_defaults(func=cmd_doctor)

    token = subparsers.add_parser(
        "token", help="print the bearer token, or the pairing line an app takes"
    )
    token.add_argument("--show", action="store_true", help="prints the secret")
    token.add_argument(
        "--url",
        action="store_true",
        help=(
            "print the pairing line for each address this server is reachable "
            "on — crucible://<name>@<host>:<port>/#<token>. It carries the "
            "token, which is what the flag name says"
        ),
    )
    token.set_defaults(func=cmd_token)

    from .sharing import add_parser as add_sharing_parser
    add_sharing_parser(subparsers)
    from .lan import add_parser as add_lan_parser
    add_lan_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Every verb, on every platform. THERE IS NO PLATFORM GATE HERE.

    There was, twice. First a total one: `crucible` refused to do anything at
    all on Windows, because vLLM and SGLang do not run there. Then, for one
    session, an opt-in flag (`win32_ok`) that let `host` and `envpack` through
    and kept the refusal for everything else, because the `llama-windows`
    backend was being built on another branch and a verb that reached a
    missing backend would have printed a worse sentence.

    Both are gone, because section 0's amendment and section 3.5 say what the
    answer is: **Windows IS a backend**, every verb runs on win32, and the one
    thing that must be true there is that `backend_kind` is `llama-windows`.
    That is not a question about a verb, so it is not asked here — it is asked
    where a backend is read, by `_backend_mismatch` (`init --backend`,
    `serve`, `service install`), which refuses `backend_not_here` and names
    both kinds. A platform test standing in for a backend test was the shape
    R1 forbids: two owners for "can this machine do it".

    `crucible host` still refuses off win32, by its own name
    (`host_windows_only`), because a tray on a machine whose service manager
    already supervises the server is a second owner of presence — a feature
    check, not a platform one wearing a feature's clothes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
