"""The `crucible` command line.

    crucible init       mint the token, write the config, record the backend
    crucible install    build a job type's env from its recipe, then decide whether the card fits it
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

    crucible api        THE CLIENT HALF: submit jobs, stream tts, chat, read
                        state — over HTTP, against a server that may be this
                        machine's, the one in WSL, or one across the network.
                        Every verb above acts on THIS machine's installation and
                        takes no address; these take --url and --token. See
                        crucible/apiclient.py and docs/API-CLI.md.

Exit codes: 0 success, 1 refused (named reason on stderr), 2 usage.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import (
    API_VERSION,
    KEEP_ALIVE_SECONDS,
    VERSION,
    capability,
    catalog,
    denoisemodels,
    envpatches,
    hosttools,
    interpreter,
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
    MLX_DARWIN,
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
    declared_tts_footprints,
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
        # THIS BOX'S SERVING FOOTPRINT PER NARRATOR ENGINE (PHASE21 section
        # 2.3). Written here because a voice that comes out of its own repo
        # carries no machine facts at all, and a server that has never been told
        # what Higgs costs on it refuses such a voice by name rather than
        # guessing. The numbers are the ones every packaged manifest declared on
        # 2026-09-19, with their citations; `declared_tts_footprints` is the one
        # function that states them, so section 9's ruling 2 — leave it unset
        # until `crucible capability` measures one — is a deleted argument
        # rather than an unpicked writer.
        tts_engines=declared_tts_footprints(backend.kind),
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
    for footprint in declared_tts_footprints(backend.kind):
        print(
            f"tts {footprint.engine}: "
            f"{footprint.memory_bytes_estimate / 1024 ** 3:.1f} GiB per resident "
            f"voice ({footprint.estimate_basis}), {footprint.max_num_seqs} in "
            f"flight — this box's figure, rewritable in config.toml "
            f"[tts.{footprint.engine}]"
        )
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

    # WEIGHTS PULLED UNDER A RENAMED ASR ID MOVE TO THE NEW ONE, before the
    # first request can ask whether they are installed (Owen's asr lineup
    # ruling, 2026-09-24; `jobs/asr.adopt_renamed_asr_weights`). Every line is
    # printed, moved or left, so the log says what happened to the bytes.
    from .jobs.asr import adopt_renamed_asr_weights

    for line in adopt_renamed_asr_weights(config):
        print(f"crucible: asr weights: {line}", file=sys.stderr)

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
        # `timeout_keep_alive` IS STATED, and the default is what broke.
        # uvicorn holds an idle connection 5 s; Node's undici keeps a pooled
        # one about 4 s, so a client's next request lands on a socket this
        # server is closing and reads ECONNRESET while the server is fine.
        # Four times: align's first `GET /v1/info` after a render's last
        # artifact fetch, Sep 18 and Sep 19 against the PC and 2026-09-20 00:57
        # against the Mac on 127.0.0.1. See `KEEP_ALIVE_SECONDS`.
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=args.log_level,
            timeout_keep_alive=KEEP_ALIVE_SECONDS,
        )
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
        # THIS BOX'S TTS FOOTPRINT SURVIVES THE REWRITE, on the same terms as
        # the retention window above (PHASE21 section 2.3): `write_config`
        # writes the whole document, so leaving `[tts.*]` out would take away
        # the one place a repo-manifest voice gets its memory estimate and its
        # serving width, and every such voice would start refusing with
        # `engine_footprint_unset`.
        tts_engines=config.tts_engines,
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
#: `SMOKE_IMPORT` below is its partner and they are now in ONE file. The table
#: used to live in `crucible/envpack.py` — which could not import this module
#: without a cycle — and a pytest tied the two together instead. The packs are
#: gone (PHASE20 section 6) and with them the reason for the separation, so the
#: two halves of "what `crucible install <type>` produces" sit beside each
#: other and cannot drift at all (ARCHITECTURE.md R1).
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

#: What an env must be able to IMPORT before `crucible install` calls it done,
#: keyed by env directory and then by backend.
#:
#: A pack build used to run this before an archive became a release asset. There
#: is no build and no asset now — the env is assembled on the machine that will
#: use it — so the check moved to the end of the install, where it answers the
#: same question about the same bytes: pip returning 0 says the wheels resolved,
#: and says nothing about whether the thing they are for loads.
#:
#: The module name is not the distribution name and the difference is not
#: cosmetic: `mlx-lm` imports as `mlx_lm`, `faster-whisper` as `faster_whisper`,
#: `qwen-asr` as `qwen_asr`, `ultimate-rvc` as `ultimate_rvc`. A table written
#: from `HEADLINE_PACKAGE` would fail on four of six envs.
SMOKE_IMPORT: dict[str, dict[str, str]] = {
    "llm": {CUDA_LINUX: "vllm", MLX_DARWIN: "mlx_lm"},
    # `asr` is TWO ENGINES, so the smoke import differs by backend: a Mac env
    # that imported `faster_whisper` would fail every install, and one that
    # imported nothing would be called ready without being opened.
    "asr": {CUDA_LINUX: "faster_whisper", MLX_DARWIN: "mlx_whisper"},
    "align": {CUDA_LINUX: "qwen_asr", MLX_DARWIN: "qwen_asr"},
    "rvc": {CUDA_LINUX: "ultimate_rvc", MLX_DARWIN: "ultimate_rvc"},
    # The tts env's KEY is the env directory's name, and it differs by backend
    # for the reason `jobenv.tts_env` gives: on cuda-linux two narrator engines
    # cannot share a venv, so the engine is in the name.
    "tts-higgs-v3": {CUDA_LINUX: "narrator"},
    "tts": {MLX_DARWIN: "narrator"},
}


def _smoke_import(python: Path, key: str, backend_kind: str) -> str | None:
    """Import this env's headline module in it. The refusal, or None.

    Not a fallback and not advisory: an env that cannot import the library it
    exists for is not installed, whatever pip said, and the operator finds out
    here rather than at chunk 900 of somebody's book.
    """
    module = SMOKE_IMPORT.get(key, {}).get(backend_kind)
    if module is None:
        return (
            f"there is no smoke import recorded for the {key!r} env on "
            f"{backend_kind}; crucible/cli.py's SMOKE_IMPORT is the owner of "
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
    # ONE PATH, AND IT IS THE RECIPE (PHASE20 section 3, item 4). `crucible
    # install` used to default to downloading an environment PACK from the
    # release and reach the recipe only under `--build`; the packs are gone, so
    # the developer's path became everybody's and the flag it hid behind went
    # with them. What the recipe path does to an env that is already there is
    # `jobenv.plan_install`'s answer, not this function's.
    #
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
    refusal = _smoke_import(
        workerenv.worker_env_python(config.home, args.job_type),
        args.job_type,
        backend.kind,
    )
    if refusal is not None:
        return _fail(refusal)
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
    except weights.WeightsShared as exc:
        # The door's code verbatim (PHASE22 section 2.9); the message already
        # begins with it and names every alias holding the folder.
        return _fail(str(exc))
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
                "revision": spec.weights_identity,
                "source": spec.source,
                "identity_basis": spec.identity_basis,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "estimate_basis": spec.estimate_basis,
                "max_chars": spec.max_chars,
                "max_chars_basis": spec.max_chars_basis,
                "pace_basis": manifest.pace_basis,
                "inherited_from": manifest.inherited_from,
                "manifest": manifest.manifest_source,
                # A LOCAL VOICE IS NOT PULLABLE, so its line must not offer a
                # pull (PHASE18-UNCERTIFIED.md section 3): `crucible voices
                # pull` on one is refused by name, and a screening merge that
                # is gone is the expected end of its life rather than a broken
                # install. The `/v1/voices` row's `reason` says the same thing
                # for a client; this is the same question asked at a terminal.
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"no weights at {spec.path} — this voice names a "
                    "directory on this server, which Crucible does not fetch "
                    "and cannot replace"
                    if spec.source == weights.LOCAL
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
    # WHAT THE BLOCK NAMES, in whichever of its two shapes (PHASE18-
    # UNCERTIFIED.md section 3). `spec.revision[:12]` subscripted None on a
    # local block, so this line raised a TypeError BEFORE `weights.pull` could
    # refuse it by name — the operator got a traceback where there is a
    # sentence saying the bytes are somebody else's.
    if spec.source == weights.LOCAL:
        print(f"{manifest.id}: {spec.path} for {backend.kind}")
    else:
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


def _repo_at(reference: str) -> tuple[str, str] | None:
    """`<owner>/<name>@<sha>` split, or None because this is not one."""
    repo, sep, revision = reference.partition("@")
    if not sep or "/" not in repo:
        return None
    return repo, revision


def _repo_manifest_for(reference: str):
    """A `RepoManifest` from `<repo>@<sha>` or from a local file.

    One reader for both `check` and `card`, because "what does this manifest
    say" must not have two answers depending on which command asked.
    """
    from .voicerepo import Pin, fetch_repo_manifest, parse_repo_manifest

    named = _repo_at(reference)
    if named is None:
        path = Path(reference)
        if not path.is_file():
            raise VoiceError(
                f"{reference!r} is neither an <owner>/<name>@<sha> reference nor "
                "a file on this machine"
            )
        return parse_repo_manifest(path.read_text(encoding="utf-8"), path), None
    repo, revision = named
    pin = Pin(id="probe", hf_repo=repo, revision=revision, path=Path(reference))
    text, path = fetch_repo_manifest(crucible_home(), pin)
    return parse_repo_manifest(text, path), pin


def cmd_voices_pin(args: argparse.Namespace) -> int:
    """`crucible voices pin <id> <repo>@<sha>` — what a deploy does per machine.

    THE PIN IS LOADED BEFORE IT IS WRITTEN, exactly as `PUT /v1/voices/{id}`
    loads it: a revision that carries no `crucible-voice.toml`, a schema this
    build cannot read, and a machine with no `[tts.<engine>]` table are each
    refused here with nothing written.
    """
    from .voicerepo import Pin, home_pins_path, voice_for_pin, write_home_pin

    named = _repo_at(args.reference)
    if named is None:
        return _fail(
            f"{args.reference!r} is not an <owner>/<name>@<sha> reference. A pin "
            "is a repo AND a commit: the manifest at that sha and the weights at "
            "that sha are the same commit, which is the whole point"
        )
    repo, revision = named
    try:
        voice = voice_for_pin(
            Pin(id=args.voice, hf_repo=repo, revision=revision, path=home_pins_path())
        )
        pin = write_home_pin(args.voice, repo, revision)
    except VoiceError as exc:
        return _fail(str(exc))
    print(f"{args.voice}: {pin.hf_repo}@{pin.revision[:12]} -> {pin.path}")
    print(f"  {voice.display} ({voice.kind}, {voice.narrator_engine}), arms "
          f"{sorted(voice.backends)}")
    print(f"  pull the weights with `crucible voices pull {args.voice}`")
    return EXIT_OK


def cmd_voices_check(args: argparse.Namespace) -> int:
    """Parse a manifest exactly as the loader would, and print it or the refusal.

    What the training side runs before it pushes. It merges with THIS machine's
    `[tts.<engine>]` table rather than with an invented one, because the
    question the command answers is "would a server load this", and a server
    that has not been told what the engine costs would not.
    """
    from .config import tts_engine_footprints
    from .voicerepo import Pin, merge

    try:
        repo, pin = _repo_manifest_for(args.reference)
    except VoiceError as exc:
        return _fail(str(exc))
    engine = repo.voice["narrator_engine"]
    footprint = tts_engine_footprints(crucible_home()).get(engine)
    if footprint is None:
        return _fail(
            f"engine_footprint_unset: this manifest is served by {engine!r} and "
            f"this machine's config states no [tts.{engine}] table, so the "
            "checks that depend on what the engine costs here cannot run and a "
            "server here would refuse the voice. Run `crucible init`, or add the "
            "table to config.toml"
        )
    # A LOCAL FILE HAS NO PIN, and the merge needs one to fill `hf_repo` and
    # `revision` in. A stand-in is used and SAID: what this command answers is
    # whether the manifest's own numbers survive the loader, and the repo and
    # sha play no part in that.
    identity = (
        Pin(id=args.id, hf_repo=pin.hf_repo, revision=pin.revision, path=pin.path)
        if pin is not None
        else Pin(
            id=args.id,
            hf_repo="checked/locally",
            revision="0" * 40,
            path=Path(args.reference),
        )
    )
    try:
        voice = merge(repo, identity, footprint)
    except VoiceError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps(voice.to_dict(), indent=2))
        return EXIT_OK
    print(f"{voice.display} ({voice.kind}, {voice.narrator_engine}, "
          f"{voice.language}, {voice.sample_rate} Hz)")
    pace = voice.pace
    if pace.pace_chars_per_sec is None:
        print("pace:     not measured — an uncertified voice (PHASE18 4.1)")
    else:
        print(
            f"pace:     {pace.pace_chars_per_sec} chars/s ({voice.pace_basis}), "
            f"band {pace.min_chars_per_sec}-{pace.max_chars_per_sec}"
        )
        if voice.inherited_from is not None:
            print(f"          inherited from {voice.inherited_from}")
    if pace.safe_min_chars is not None:
        print(f"packs:    {pace.safe_min_chars}-{pace.safe_max_chars} chars")
    elif pace.target_chars is not None:
        print(f"packs:    {pace.target_chars} chars")
    for arm in sorted(voice.backends):
        spec = voice.backends[arm]
        # "not measured" rather than a blank or a zero: a voice may state no
        # cap since 2026-09-19 (PHASE18-UNCERTIFIED.md section 4), and an
        # operator reading `cap None` would not know whether the number is
        # missing or the field is broken.
        cap = (
            "not measured"
            if spec.max_chars is None
            else f"{spec.max_chars} ({spec.max_chars_basis})"
        )
        print(
            f"{arm}: cap {cap}, sampling "
            + ", ".join(f"{k} {v}" for k, v in sorted(spec.sampling.items()))
        )
    print(f"takes:    {len(voice.takes)} rung(s)")
    return EXIT_OK


def cmd_voices_card(args: argparse.Namespace) -> int:
    """Render the repo's README from its manifest, and with --upload commit it."""
    from . import voicecard
    from .voicerepo import REPO_MANIFEST_NAME
    from .weights import hf_token_at

    try:
        repo, pin = _repo_manifest_for(args.reference)
    except VoiceError as exc:
        return _fail(str(exc))
    if pin is None:
        if args.upload:
            return _fail(
                "--upload needs an <owner>/<name>@<sha> reference: a local file "
                "names no repo to commit to"
            )
        # A LOCAL FILE HAS NO CARD TO MERGE INTO, so what is printed is the two
        # blocks this command owns. That is what the training side wants before
        # it pushes: see the frontmatter and the limits section that the repo's
        # README will get, without a repo yet existing.
        print("---")
        print(voicecard.render_frontmatter(repo, ""))
        print("---")
        print()
        print(voicecard.render_limits(repo), end="")
        return EXIT_OK

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        return _fail(f"huggingface_hub is not importable: {exc}")
    token = hf_token_at(config_path(crucible_home()))
    try:
        card_path = hf_hub_download(
            repo_id=pin.hf_repo,
            filename="README.md",
            revision=pin.revision,
            local_dir=str(
                crucible_home()
                / "voice-manifests"
                / pin.hf_repo.replace("/", "--")
                / pin.revision
            ),
            token=token,
        )
    except Exception as exc:
        return _fail(
            f"could not read README.md from {pin.hf_repo}@{pin.revision[:12]}: "
            f"{type(exc).__name__}: {exc}"
        )
    existing = Path(card_path).read_text(encoding="utf-8")
    try:
        rendered, added = voicecard.render_card(repo, existing)
    except VoiceError as exc:
        return _fail(str(exc))
    if added:
        print(
            "note: this card had no limits section; one was ADDED after the "
            "frontmatter"
        )
    if not args.upload:
        print(rendered, end="")
        print(
            f"\n--- not uploaded. Pass --upload to commit this README.md to "
            f"{pin.hf_repo}, rendered from its own {REPO_MANIFEST_NAME}.",
        )
        return EXIT_OK
    if rendered == existing:
        print(f"{pin.hf_repo}: the card already says what the manifest says")
        return EXIT_OK
    try:
        HfApi(token=token).upload_file(
            path_or_fileobj=rendered.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=pin.hf_repo,
            commit_message=(
                f"Render README.md from {REPO_MANIFEST_NAME} "
                f"(crucible voices card)"
            ),
        )
    except Exception as exc:
        return _fail(f"could not upload README.md to {pin.hf_repo}: {exc}")
    print(f"{pin.hf_repo}: README.md rendered from {REPO_MANIFEST_NAME} and pushed")
    return EXIT_OK


def cmd_voices_export(args: argparse.Namespace) -> int:
    """A packaged manifest as a `crucible-voice.toml`, plus the rows it drops."""
    from . import voicecard
    from .voices import MANIFEST_REPO

    try:
        manifest = load_voice(args.voice)
    except VoiceError as exc:
        return _fail(str(exc))
    if manifest.manifest_source == MANIFEST_REPO:
        return _fail(
            f"voice {args.voice!r} already comes out of a repo manifest at its "
            "pin, so there is nothing to convert. Read it with `crucible voices "
            "check`"
        )
    try:
        text, dropped = voicecard.export_manifest(
            manifest,
            pace_basis=args.pace_basis,
            measured_from=args.measured_from,
            inherited_from=args.inherited_from,
            max_chars_basis=args.max_chars_basis,
            uncertified=args.uncertified,
        )
    except VoiceError as exc:
        return _fail(str(exc))
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{args.voice}: wrote {args.out}")
    else:
        print(text, end="")
    # PRINTED, NEVER DROPPED SILENTLY (section 4). Every line is a machine fact
    # that moved rather than vanished, and the person converting the file is the
    # one who has to put it where it now lives.
    print(f"\n# {len(dropped)} row(s) this file does NOT carry:", file=sys.stderr)
    for line in dropped:
        print(f"#   {line}", file=sys.stderr)
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
    entry["provenance"] = _provenance(
        report,
        label,
        status,
        recipe,
        _plan_or_refusal(lambda: jobenv.plan_install(home, spec, backend_kind)),
    )
    return entry


def _plan_or_refusal(call: Any) -> "jobenv.EnvPlan | str":
    """The plan, or the sentence the planner refused with.

    `crucible doctor` reports what an install WOULD do, so a planner refusal is
    a doctor problem rather than a doctor crash — and it is the same sentence
    the operator gets when they run the install, because it comes from the same
    function (ARCHITECTURE.md R1).
    """
    try:
        return call()
    except (jobenv.EnvError, workerenv.WorkerEnvError) as exc:
        return str(exc)


def _provenance(
    report: dict[str, Any],
    label: str,
    status: "jobenv.EnvStatus | workerenv.EnvStatus",
    recipe: Path,
    plan: "jobenv.EnvPlan | str",
) -> dict[str, Any]:
    """What this env was installed from, and what an install would do to it now.

    TWO DRIFTS, NAMED APART (PHASE20 section 4). `env_recipe_drift` is the
    environment half — torch, SGLang, the wheels — and costs a `pip install -r`
    into the venv that is there. `narrator_sha_drift` is one git sha in one
    line and costs one `pip install --no-deps`. A single `pack_recipe_drift`
    could not tell them apart, and told every reader the same wrong thing about
    both: that an env had to be rebuilt.

    The verdict is `jobenv.plan_install`'s rather than this function's, because
    the doctor's sentence and the installer's remedy must be one sentence.
    """
    entry: dict[str, Any] = {
        "recipe": recipe.name,
        "environment_sha256": status.environment_sha256,
        "environment_sha256_now": (
            jobenv.environment_sha256(recipe) if recipe.is_file() else None
        ),
        "direct_references": (
            None if status.direct_references is None
            else dict(status.direct_references)
        ),
        "action": plan if isinstance(plan, str) else plan.action,
        "detail": plan if isinstance(plan, str) else plan.detail,
    }
    if isinstance(plan, str):
        report["problems"].append(f"{label}: {plan}")
    elif plan.action != jobenv.PLAN_NOTHING:
        report["problems"].append(
            f"{label}: {plan.action} — {plan.detail}. "
            f"`crucible install` brings it up to {recipe.name}"
        )
    return entry


def _provenance_line(entry: dict[str, Any]) -> str:
    """The one-line form `crucible doctor` prints after an env's detail."""
    if entry["environment_sha256"] is None:
        recipe = f"{entry['recipe']} halves not recorded (installed before 0.7.0)"
    elif entry["environment_sha256"] != entry["environment_sha256_now"]:
        recipe = (
            f"{entry['recipe']} {entry['environment_sha256'][:12]} != "
            f"{(entry['environment_sha256_now'] or 'absent')[:12]} HERE"
        )
    else:
        recipe = f"{entry['recipe']} {entry['environment_sha256'][:12]}"
    references = entry["direct_references"] or {}
    if references:
        recipe += ", " + ", ".join(
            f"{name} @ {commit[:12]}" for name, commit in sorted(references.items())
        )
    if entry["action"] == jobenv.PLAN_NOTHING:
        return recipe
    return f"{recipe} — {entry['action']}"


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
        "llm_patches": [],
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
        # WEIGHTS NO MANIFEST OWNS (`catalog.stranded_weights`). Reported and
        # never counted as a problem: bytes on a disk are not an unhealthy
        # server, and deleting them is the operator's decision. Null when no
        # config was readable, because there is then no home to look in.
        "stranded_weights": None,
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

    if config is not None:
        report["stranded_weights"] = catalog.stranded_weights(config)

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
                    _plan_or_refusal(
                        lambda: workerenv.plan_install(
                            config.home, job_type, backend.kind
                        )
                    ),
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
        if config.enable_llm:
            # THE `llm` ENV's PATCHES (`crucible/envpatches.py`): mlx-lm's
            # `top_logprobs` 11 -> 40, which `MlxLmEngine` states as its cap
            # and refuses to start without. Selected by the recipe's pins, so
            # cuda-linux (vLLM) reads `not_applicable`; llama-windows has no
            # llm recipe at all and is asked with none, which answers the same.
            if backend.kind == LLAMA_WINDOWS:
                llm_env_dir, llm_pins = config.home / "envs" / "none", {}
            else:
                llm_spec = jobenv.llm_env(backend.kind)
                llm_env_dir = jobenv.env_dir(config.home, llm_spec)
                llm_pins = jobenv.recipe_pins(jobenv.recipe_for(llm_spec))
            report["llm_patches"] = envpatches.check("llm", llm_env_dir, llm_pins)
            for entry in report["llm_patches"]:
                # `no_env` is the llm env row's fact, stated there in full with
                # the command that fixes it; a second problem for the same
                # cause is how a reader ends up chasing the wrong sentence.
                if entry["status"] not in narratorpatches.SOUND_STATUSES and (
                    entry["status"] != narratorpatches.NO_ENV
                ):
                    report["problems"].append(
                        f"llm_patch[{entry['id']}]: {entry['status']} — "
                        f"{entry['detail']}. {entry['why']}. Run `crucible env "
                        "patch llm`"
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
        for entry in report["llm_patches"]:
            if entry["status"] == narratorpatches.NOT_APPLICABLE:
                mark = "n/a"
            else:
                mark = "applied" if entry["applied"] else entry["status"].upper()
            print(f"llm patch ({entry['id']}): {mark} — {entry['detail']}")
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
        for entry in report["stranded_weights"] or ():
            why = entry["note"] or (
                f"no manifest in this build declares {entry['id']!r} on "
                f"{entry['backend']}"
            )
            print(
                f"note:    {entry['bytes'] / 1e9:.2f} GB of weights at "
                f"{entry['path']} belong to nothing: {why}. Nothing will use "
                "them; delete the directory to reclaim the space"
            )
        for problem in report["problems"]:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print("healthy" if report["healthy"] else "unhealthy")
    return EXIT_OK if report["healthy"] else EXIT_REFUSED


# ---------------------------------------------------------------- env patch


def cmd_env_patch(args: argparse.Namespace) -> int:
    """Apply and CHECK one env type's site-packages patches, in place.

    The installer's `env-patch-llm` step (`sdk/bootstrap/src/steps.ts`, PHASE22-DECIDE.md
    section 2.6.1): an upgrade installs a new wheel and restarts, and never runs
    `crucible install`, so `install_env`'s patch step never runs on an env whose
    recipe has not moved. This is that step without the pip. Exit 0 only when
    every row is `applied` or `not_applicable`; anything else is refused by name.

    No env installed is NOT a failure: there is nothing to patch, and
    `crucible install <type>` applies the patches before it stamps the env.
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
    if envpatches.patches_for(args.job_type) is None:
        return _fail(
            f"job type {args.job_type!r} carries no site-packages patches; the "
            f"types that do are {list(envpatches.patched_job_types())}"
        )
    if args.job_type == "llm" and backend.kind == LLAMA_WINDOWS:
        # No Python env at all: the engine is llama.cpp's own release.
        rows = envpatches.check("llm", config.home / "envs" / "none", {})
    else:
        try:
            spec = _env_spec(args.job_type, args.narrator_engine, backend.kind)
            recipe = jobenv.recipe_for(spec)
            pins = jobenv.recipe_pins(recipe)
        except jobenv.EnvError as exc:
            return _fail(str(exc))
        directory = jobenv.env_dir(config.home, spec)
        python = jobenv.env_python(config.home, spec)
        if not python.is_file():
            print(
                f"{args.job_type} env: not installed at {directory}; nothing to "
                f"patch (`crucible install {args.job_type}` applies them)"
            )
            return EXIT_OK
        try:
            rows = envpatches.apply(
                args.job_type, directory, python, pins, on_line=print
            )
        except narratorpatches.PatchError as exc:
            return _fail(f"env_patch_failed: {exc}")
    for row in rows:
        print(f"{args.job_type} patch ({row['id']}): {row['status']} — {row['detail']}")
    unsound = [r for r in rows if r["status"] not in narratorpatches.SOUND_STATUSES]
    if unsound:
        return _fail(
            "env_patch_failed: "
            + "; ".join(f"{r['id']} is {r['status']}" for r in unsound)
        )
    return EXIT_OK


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

    voices_pin = voice_commands.add_parser(
        "pin", help="point a voice id at a repo and commit (PHASE21 section 2.4)"
    )
    voices_pin.add_argument("voice", help="the Crucible voice id, e.g. mistborn")
    voices_pin.add_argument(
        "reference", help="<owner>/<name>@<40-character sha>"
    )
    voices_pin.set_defaults(func=cmd_voices_pin)

    voices_check = voice_commands.add_parser(
        "check",
        help="parse a crucible-voice.toml exactly as the loader would",
    )
    voices_check.add_argument(
        "reference", help="<owner>/<name>@<sha>, or a path to a local file"
    )
    voices_check.add_argument(
        "--id",
        default="probe",
        help="the id to check it under; only the refusal messages see it",
    )
    voices_check.add_argument("--json", action="store_true", help="machine-readable")
    voices_check.set_defaults(func=cmd_voices_check)

    voices_card = voice_commands.add_parser(
        "card", help="render a repo's README.md from its crucible-voice.toml"
    )
    voices_card.add_argument(
        "reference", help="<owner>/<name>@<sha>, or a path to a local file"
    )
    voices_card.add_argument(
        "--upload",
        action="store_true",
        help="commit the rendered README.md to the repo (needs an HF token)",
    )
    voices_card.set_defaults(func=cmd_voices_card)

    voices_export = voice_commands.add_parser(
        "export",
        help="a packaged manifest as a crucible-voice.toml, machine rows dropped",
    )
    voices_export.add_argument("voice", help="the Crucible voice id")
    voices_export.add_argument("--out", help="write here instead of to stdout")
    voices_export.add_argument(
        "--pace-basis",
        choices=("measured", "inherited"),
        help="how this voice's pace was got; the packaged schema cannot say",
    )
    voices_export.add_argument(
        "--measured-from",
        help="what a measured pace was measured on; required with "
        "--pace-basis measured",
    )
    voices_export.add_argument(
        "--inherited-from",
        help="which run and checkpoint an inherited pace came from, and why "
        "these weights have no ladder; required with --pace-basis inherited",
    )
    voices_export.add_argument(
        "--max-chars-basis",
        choices=("measured", "placeholder"),
        help="how the per-arm caps were got; the packaged schema cannot say",
    )
    voices_export.add_argument(
        "--uncertified",
        action="store_true",
        help="say on purpose that this voice has no measured pace",
    )
    voices_export.set_defaults(func=cmd_voices_export)

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

    env_parser = subparsers.add_parser(
        "env", help="operate on an installed job-type env without rebuilding it"
    )
    env_commands = env_parser.add_subparsers(dest="env_command", required=True)
    env_patch = env_commands.add_parser(
        "patch",
        help=(
            "apply and check this env type's site-packages patches in place; "
            "exits non-zero by name unless every one is in (a deploy runs it)"
        ),
    )
    env_patch.add_argument("job_type", choices=sorted(envpatches.patched_job_types()))
    env_patch.add_argument(
        "--narrator-engine",
        default=None,
        choices=sorted(NARRATOR_ENGINE_SAMPLING),
        help="which tts env; required for 'tts', refused for 'llm'",
    )
    env_patch.set_defaults(func=cmd_env_patch)

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

    # THE CLIENT HALF, and the one namespace in this file whose verbs take an
    # address. Everything above acts on THIS machine's installation and has
    # nothing to point at; `crucible api …` speaks HTTP to a server that may be
    # in WSL or on the Mac. Imported here rather than at module scope for
    # `local`'s and `sharing`'s reason — `build_parser` is the only caller and a
    # CLI's import time is its `--help` time.
    from .apiclient import add_parser as add_api_parser
    add_api_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Every verb, on every platform. THERE IS NO PLATFORM GATE HERE.

    There was, twice. First a total one: `crucible` refused to do anything at
    all on Windows, because vLLM and SGLang do not run there. Then, for one
    session, an opt-in flag (`win32_ok`) that let `host` and the since-deleted
    pack builder through and kept the refusal for everything else, because the
    `llama-windows`
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
