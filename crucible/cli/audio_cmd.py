from __future__ import annotations

import argparse

from .. import lowvram
from ..audiomodels import AudioManifestError, load_all_audio_manifests
from ..capabilitystore import low_vram_for, low_vram_not_offered, set_low_vram
from ..config import load_config
from ..errors import ConfigError
from ..memorybudget import gib_text
from . import common
from .common import EXIT_OK, _fail


def low_vram_models(backend_kind: str) -> dict[str, tuple[int, int]]:
    """The audio models whose manifest on this backend can be held half at a time:
    id -> (whole-model bytes, low-VRAM bytes)."""
    return {
        model_id: (spec.memory_bytes_estimate, spec.low_vram_memory_bytes_estimate)
        for model_id, manifest in load_all_audio_manifests().items()
        if (spec := manifest.backends.get(backend_kind)) is not None
        and spec.low_vram_memory_bytes_estimate is not None
    }


def cmd_audio_low_vram(args: argparse.Namespace) -> int:
    config, backend = common.here()
    try:
        honoured = low_vram_models(config.backend_kind)
    except AudioManifestError as exc:
        return _fail(str(exc))
    figures = "; ".join(
        f"{model_id} {gib_text(low)} instead of {gib_text(whole)}"
        for model_id, (whole, low) in sorted(honoured.items())
    )
    now = low_vram_for(config, backend)
    if args.state is None:
        print(f"[audio] low_vram is {'on' if config.audio_low_vram else 'off'} in {config.path}")
        print(now.words)
        print(
            f"honoured on {config.backend_kind} by: {figures}"
            if honoured
            else f"no audio model on {config.backend_kind} can be held half at a time"
        )
        return EXIT_OK
    if args.state == lowvram.ON and not honoured:
        return _fail(f"low_vram_not_offered: {low_vram_not_offered(config.backend_kind)}")
    if now.state == args.state and not now.changed:
        print(f"[audio] low_vram is already {args.state} in {config.path}")
        return EXIT_OK
    try:
        done = set_low_vram(config, backend, args.state)
    except ConfigError as exc:
        return _fail(str(exc))
    after = done.low_vram
    print(f"[audio] low_vram = {'true' if after.on else 'false'} — {done.path}")
    print(after.words)
    if after.on:
        print(
            f"an audio model that can be split now holds only the half a stage uses "
            f"on the card ({figures}); the audio is the same"
        )
    print(
        "a running server reads it on its next request. An audio model already "
        "resident stays loaded the way it was until it comes off the card (an "
        "unload-audio job, or the server clearing the card when nothing holds it)"
    )
    # What the card can hold for audio changed with the setting, so the recorded verdict
    # (and [jobs] enable_audio, which follows it) was decided again by set_low_vram, not
    # left stale until someone thinks to run `crucible capability --write`.
    from .capability import _print_redecided

    return _print_redecided(load_config(config.home), backend, done.redecided, ("audio",))


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    audio = subparsers.add_parser(
        "audio",
        help="settings of the audio job type ([audio] in config.toml)",
    )
    commands = audio.add_subparsers(dest="audio_command", required=True)
    low_vram = commands.add_parser(
        "low-vram",
        help=(
            "[audio] low_vram: hold one half of a splittable audio model (YuE2) on "
            "the card at a time, for a card too small to hold it whole. Crucible "
            "turns it on by itself where the card needs it; on or off is yours and "
            "it never changes it, auto hands it back. No argument shows the setting"
        ),
    )
    low_vram.add_argument("state", nargs="?", choices=lowvram.STATES, default=None)
    low_vram.set_defaults(func=cmd_audio_low_vram)
