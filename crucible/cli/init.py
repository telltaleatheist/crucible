from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import ladder
from ..backend import BACKEND_KINDS, Backend, MLX_DARWIN
from ..config import (
    Config,
    DEFAULT_DESKTOP_ALLOWANCE_BYTES,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_RETENTION_DAYS,
    DESKTOP_BASES,
    DESKTOP_BASIS_DECLARED,
    DESKTOP_BASIS_MEASURED,
    DESKTOP_BASIS_STATED,
    MLX_DESKTOP_ALLOWANCE_FRACTION,
    config_mode,
    config_path,
    crucible_home,
    declared_tts_footprints,
    default_desktop_allowance_bytes,
    default_server_name,
    mint_token,
    write_config,
)
from ..errors import ConfigError, NoViableBackend
from . import common
from .common import EXIT_OK, _backend_mismatch, _fail
from .token import PAIRING_NOT_PRINTED, _pairing_permission, _write_pairing_file


def carried_from(path: Path) -> tuple[str, dict[str, Any]]:
    """`--config-from`: the token, the routes and the upstreams, and NOTHING else.

    PHASE15-HOST.md 4.3. The host writes this file at 0600 when it moves a
    Windows Crucible into the WSL guest and deletes it afterwards; the point
    of the flag is that the TOKEN survives, so every app that paired with this
    machine stays paired.

    Three tables here, and the desktop reserve beside them (`carried_reserve`).
    The host, the port, the name, the backend and the job flags belong to the
    machine being INITIALISED, not to the one being left — a guest that
    inherited `backend = "llama-windows"` would refuse to serve on its own card.

    THE RESERVE IS CARRIED SINCE 2026-09-26, reversing what this said before.
    The Windows server and the WSL guest share one card and one desktop, and a
    reserve is now a measured or stated fact about that desktop with its basis
    beside it: re-deciding it in the guest would re-measure a reserve somebody
    stated (Owen's 3 GiB, kept for streaming) or lower it on a quiet minute.
    The iGPU worry that kept it out is gone with the basis: a Windows server on
    no NVIDIA card writes a "declared" reserve, never a measured one.
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


def carried_reserve(path: Path) -> tuple[int, str, str]:
    """`--config-from`'s desktop reserve: (bytes, basis, note)."""
    import tomllib

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"config_from_unreadable: {path} could not be read: {exc}")
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config_from_unreadable: {path} is not TOML: {exc}")
    section = document.get("accelerator")
    value = section.get("desktop_allowance_bytes") if isinstance(section, dict) else None
    basis = section.get("desktop_allowance_basis") if isinstance(section, dict) else None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or basis not in DESKTOP_BASES
    ):
        raise ConfigError(
            f"config_from_no_reserve: {path} has no [accelerator] "
            "desktop_allowance_bytes and desktop_allowance_basis to carry"
        )
    note = section.get("desktop_allowance_note", "")
    return value, basis, note if isinstance(note, str) else ""


def _existing_stated_reserve(path: Path) -> tuple[int, str] | None:
    """A STATED reserve in the config `init --force` is about to replace.

    Owen, 2026-09-26: an existing reserve is never changed automatically. A
    re-init mints a new token; it is not a request to lower owens-pc's 3 GiB
    because nothing was streaming that minute. So a stated reserve survives the
    re-init;
    `crucible capability --measure-desktop` is the deliberate way to replace it.
    """
    if not path.exists():
        return None
    try:
        existing = common.load_config(path.parent)
    except ConfigError:
        return None
    if existing.desktop_allowance_basis != DESKTOP_BASIS_STATED:
        return None
    return existing.desktop_allowance_bytes, existing.desktop_allowance_note


def _decide_reserve(
    args: argparse.Namespace, backend: Backend, home: Path
) -> tuple[int, str, str, str]:
    """`crucible init`'s desktop reserve: (bytes, basis, note, how it was reached).

    In order, first answer wins:

    1. `--desktop-allowance-bytes` — stated, and it always wins.
    2. `--config-from` carrying a reserve — its value and basis, unchanged.
    3. `--force` over a config whose reserve is stated — kept
       (`_existing_stated_reserve`).
    4. An NVIDIA card nvidia-smi answers for, with nothing of Crucible's on
       it — MEASURED (`ladder.measure_desktop_reserve`, 2026-09-26).
    5. Otherwise the backend's declared rule
       (`config.default_desktop_allowance_bytes`), with the reason it was not
       measured in the note. A Mac always lands here: its reserve is a share of
       unified memory, not a desktop on a card.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    if args.desktop_allowance_bytes is not None:
        return (
            args.desktop_allowance_bytes,
            DESKTOP_BASIS_STATED,
            f"given to `crucible init --desktop-allowance-bytes` on {today}",
            "stated",
        )
    if args.config_from is not None:
        value, basis, note = carried_reserve(Path(args.config_from))
        return value, basis, note, f"{basis}, carried from {args.config_from}"
    if args.force:
        kept = _existing_stated_reserve(config_path(home))
        if kept is not None:
            value, note = kept
            return (
                value,
                DESKTOP_BASIS_STATED,
                note,
                "stated, kept from the config this replaced; "
                "`crucible capability --measure-desktop` re-measures it",
            )
    declared = default_desktop_allowance_bytes(backend.kind, backend.gpu.vram_bytes)
    if backend.kind == MLX_DARWIN:
        return (
            declared,
            DESKTOP_BASIS_DECLARED,
            f"{MLX_DESKTOP_ALLOWANCE_FRACTION * 100:.0f}% of unified memory",
            f"{backend.kind} default",
        )
    existing: Config | None = None
    if config_path(home).exists():
        try:
            existing = common.load_config(home)
        except ConfigError:
            existing = None
    reserve, why_not = ladder.measure_desktop_reserve(existing, backend, port=args.port)
    if reserve is not None:
        return reserve.allowance_bytes, DESKTOP_BASIS_MEASURED, reserve.note, "measured"
    return (
        declared,
        DESKTOP_BASIS_DECLARED,
        f"not measured on {today}: {why_not}",
        f"{backend.kind} default; not measured: {why_not}",
    )


def cmd_init(args: argparse.Namespace) -> int:
    home = crucible_home()
    path = config_path(home)
    if path.exists() and not args.force:
        return _fail(
            f"{path} already exists; pass --force to replace it (this mints a new "
            "token and every client will need the new one)"
        )

    try:
        backend = common.detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")

    # `--backend` STATES what the caller expects this host to be, and is
    # checked against what it is. Section 2: *"`crucible init --backend
    # llama-windows` is legal only on win32 … `cuda-linux`/`mlx-darwin` on
    # win32 are refused the same way"*. `crucible orchestrator` passes it (4.3) so a
    # host that somehow ran on the wrong machine says so here instead of
    # writing a config the server would refuse to start from.
    if args.backend is not None and args.backend != backend.kind:
        return _fail(_backend_mismatch(args.backend, backend))

    # The host reserve is resolved HERE rather than by argparse, because it
    # depends on the backend that was just detected, the size of its pool and,
    # on an NVIDIA card since 2026-09-26, what its desktop actually holds
    # (`_decide_reserve` gives the order). BEFORE the config is written, so a
    # measurement never sees a half-written home.
    try:
        (
            desktop_allowance_bytes,
            desktop_basis,
            desktop_note,
            desktop_source,
        ) = _decide_reserve(args, backend, home)
    except ConfigError as exc:
        return _fail(str(exc))

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
        retention_days=DEFAULT_RETENTION_DAYS,
        desktop_allowance_basis=desktop_basis,
        desktop_allowance_note=desktop_note,
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
    print(
        f"desktop:  {desktop_allowance_bytes / 1024 ** 3:.1f} GiB of "
        f"{backend.gpu.vram_bytes / 1024 ** 3:.1f} GiB treated as this host's own "
        f"desktop, not somebody's job ({desktop_source})"
    )
    if desktop_basis == DESKTOP_BASIS_MEASURED:
        print(f"          {desktop_note}")
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
    print(PAIRING_NOT_PRINTED)
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
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
            "backend_not_here when the two disagree. `crucible orchestrator` passes "
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
            "not count as somebody's job. Stating it always wins. Otherwise, on "
            "an NVIDIA card with nothing of Crucible's on it, init MEASURES the "
            "desktop (peak + max(peak, 1 GiB), at most 3 GiB); else defaults PER "
            f"BACKEND: cuda {DEFAULT_DESKTOP_ALLOWANCE_BYTES} = 3 GiB flat, "
            # `%%`: argparse formats help with `%`, and a bare "25% of" is
            # read as a `% o` directive and crashes `init --help`.
            f"mlx-darwin {MLX_DESKTOP_ALLOWANCE_FRACTION * 100:.0f}%% of unified memory "
            "because the model and the whole OS share one pool. Use 0 on a "
            "headless box"
        ),
    )
    init.set_defaults(func=cmd_init)

    init.add_argument(
        "--config-from",
        metavar="FILE",
        help=(
            "take the token, [routes], [upstreams] and the [accelerator] desktop "
            "reserve (with its basis) out of this TOML file "
            "instead of minting a token (PHASE15-HOST.md 4.3). The host writes "
            "it at 0600 when it moves a Windows Crucible into the WSL guest and "
            "deletes it after, so every app that paired stays paired"
        ),
    )
