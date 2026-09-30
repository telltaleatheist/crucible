from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import ladder
from ..backend import BACKEND_KINDS, MLX_DARWIN, Backend
from ..capabilityrecord import (
    DESKTOP_BASES,
    DESKTOP_BASIS_DECLARED,
    DESKTOP_BASIS_MEASURED,
    DESKTOP_BASIS_STATED,
)
from ..config import (
    DEFAULT_DESKTOP_ALLOWANCE_BYTES,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_RETENTION_DAYS,
    MLX_DESKTOP_ALLOWANCE_FRACTION,
    Config,
    config_mode,
    config_path,
    crucible_home,
    default_desktop_allowance_bytes,
    default_server_name,
    mint_token,
    write_config,
)
from ..errors import ConfigError, NoViableBackend
from ..memorybudget import gib_text
from ..narratorengines import declared_tts_footprints
from . import common
from .common import EXIT_OK, _backend_mismatch, _fail
from .token import PAIRING_NOT_PRINTED, _pairing_permission, _write_pairing_file


def carried_from(path: Path) -> tuple[str, dict[str, Any]]:
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

    if args.backend is not None and args.backend != backend.kind:
        return _fail(_backend_mismatch(args.backend, backend))

    try:
        (
            desktop_allowance_bytes,
            desktop_basis,
            desktop_note,
            desktop_source,
        ) = _decide_reserve(args, backend, home)
    except ConfigError as exc:
        return _fail(str(exc))

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
        enable_image=args.enable_image,
        enable_audio=args.enable_audio,
        enable_video=args.enable_video,
        desktop_allowance_bytes=desktop_allowance_bytes,
        retention_days=DEFAULT_RETENTION_DAYS,
        desktop_allowance_basis=desktop_basis,
        desktop_allowance_note=desktop_note,
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
    print(f"image:    {'enabled' if args.enable_image else 'disabled'}")
    print(f"audio:    {'enabled' if args.enable_audio else 'disabled'}")
    print(f"video:    {'enabled' if args.enable_video else 'disabled'}")
    for footprint in declared_tts_footprints(backend.kind):
        print(
            f"tts {footprint.engine}: "
            f"{gib_text(footprint.memory_bytes_estimate)} per resident "
            f"voice ({footprint.estimate_basis}), {footprint.max_num_seqs} in "
            f"flight — this box's figure, rewritable in config.toml "
            f"[tts.{footprint.engine}]"
        )
    print(
        f"desktop:  {gib_text(desktop_allowance_bytes)} of "
        f"{gib_text(backend.gpu.vram_bytes)} treated as this host's own "
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
            "(docs/internals/engines-and-capability.md, \"Backends\"), so this never chooses one — it refuses "
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
        "--enable-image",
        action="store_true",
        help="register the image (Qwen-Image text-to-image), load-image and "
        "unload-image job types ([jobs] enable_image)",
    )
    init.add_argument(
        "--enable-audio",
        action="store_true",
        help="register the audio (sound effects, music and songs: Stable Audio 3, "
        "YuE2), load-audio and unload-audio job types ([jobs] enable_audio)",
    )
    init.add_argument(
        "--enable-video",
        action="store_true",
        help="register the video (LTX-2.5 text- and image-to-video with sound, "
        "cuda-linux only), load-video and unload-video job types ([jobs] enable_video)",
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
            "instead of minting a token (docs/internals/cli.md, \"init\"). The host writes "
            "it at 0600 when it moves a Windows Crucible into the WSL guest and "
            "deletes it after, so every app that paired stays paired"
        ),
    )
