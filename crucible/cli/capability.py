from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from .. import capability, ladder
from ..backend import Backend, MLX_DARWIN
from ..config import (
    CAPABILITY_FLAGS,
    Config,
    DESKTOP_BASIS_MEASURED,
    desktop_reserve_words,
    write_config,
)
from . import common
from .common import EXIT_OK, _fail


def _decide_here(config: Config, backend: Backend) -> tuple[capability.Decision, ...]:
    return capability.decide_all(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        gpu_vendor=backend.gpu.vendor,
        chosen={entry.capability: entry.model for entry in config.local_models},
        card=ladder.card_for(config.home, backend.gpu),
    )


def _card_facts(home: Path, backend: Backend) -> dict[str, Any]:
    return ladder.card_for(home, backend.gpu).to_dict()


def _card_line(facts: dict[str, Any]) -> str:
    if facts["compute_capability"] is None:
        return (
            "compute capability not reported (no NVIDIA card, or a driver too "
            "old to answer `nvidia-smi --query-gpu=compute_cap`); card features "
            "unknown"
        )
    features = facts["features"] or {}

    def said(name: str) -> str:
        value = features.get(name)
        word = "yes" if value else "NO"
        floor = facts["floors"].get(name)
        return f"{name} {word}" + (f" (needs {floor})" if floor else "")

    known = [name for name in features if features[name] is not None]
    measured = (
        f"; measured {facts['measured_at']}" if facts.get("measured_at") else ""
    )
    return (
        f"{facts['sm']} ({facts['compute_capability']}): "
        + ", ".join(said(name) for name in known)
        + measured
    )


def _write_capability(
    config: Config,
    backend: Backend,
    decisions: tuple[capability.Decision, ...],
    flags: dict[str, bool],
) -> Path:
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
        desktop_allowance_basis=config.desktop_allowance_basis,
        desktop_allowance_note=config.desktop_allowance_note,
        retention_days=config.retention_days,
        tts_engines=config.tts_engines,
        capability=capability.record(
            backend.kind,
            total_bytes=backend.gpu.vram_bytes,
            desktop_allowance_bytes=config.desktop_allowance_bytes,
            decisions=decisions,
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
    if backend.kind != MLX_DARWIN and backend.gpu.vendor != capability.CPU_VENDOR:
        print(f"card:     {_card_line(_card_facts(config.home, backend))}")
    print(
        f"pool:     {backend.gpu.vram_bytes / gib:.1f} GiB "
        f"{capability.POOL_NAME[backend.kind]}"
    )
    print(
        "reserve:  "
        + desktop_reserve_words(
            config.desktop_allowance_bytes, config.desktop_allowance_basis
        )
    )
    if config.desktop_allowance_note:
        print(f"          {config.desktop_allowance_note}")
    print(f"budget:   {budget / gib:.1f} GiB available to a job")
    for decision in decisions:
        mark = "yes" if decision.enabled else "NO"
        print(f"{decision.capability:<10} {mark:<4} {decision.reason}")


def cmd_capability(args: argparse.Namespace) -> int:
    config, backend = common.here(tolerate_stale_record=True)

    before: Config | None = None
    reserve: ladder.DesktopReserve | None = None
    if args.measure_desktop:
        reserve, why_not = ladder.measure_desktop_reserve(config, backend)
        if reserve is None:
            return _fail(
                f"desktop_not_measured: {why_not}. Nothing was written; the "
                f"reserve stays {config.desktop_allowance_bytes / 1024 ** 3:.1f} GiB "
                f"({config.desktop_allowance_basis})"
            )
        before = config
        config = dataclasses.replace(
            config,
            desktop_allowance_bytes=reserve.allowance_bytes,
            desktop_allowance_basis=DESKTOP_BASIS_MEASURED,
            desktop_allowance_note=reserve.note,
        )
    decisions = _decide_here(config, backend)

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
                    "card": _card_facts(config.home, backend),
                    "total_bytes": backend.gpu.vram_bytes,
                    "desktop_allowance_bytes": config.desktop_allowance_bytes,
                    "desktop_allowance_basis": config.desktop_allowance_basis,
                    "desktop_allowance_note": config.desktop_allowance_note,
                    "desktop_reserve": desktop_reserve_words(
                        config.desktop_allowance_bytes, config.desktop_allowance_basis
                    ),
                    "desktop_remeasured": (
                        None
                        if before is None or reserve is None
                        else {
                            "old_bytes": before.desktop_allowance_bytes,
                            "old_basis": before.desktop_allowance_basis,
                            "new_bytes": reserve.allowance_bytes,
                            "desktop_least_bytes": reserve.sample.least_bytes,
                            "desktop_peak_bytes": reserve.sample.peak_bytes,
                        }
                    ),
                    "available_bytes": capability.available_bytes(
                        backend.gpu.vram_bytes, config.desktop_allowance_bytes
                    ),
                    "classes": [d.to_dict() for d in decisions],
                    "job_types": {
                        name: capability.job_type_enabled(name, decisions)
                        for name in sorted({d.job_type for d in decisions})
                    },
                    "written": bool(args.write or args.measure_desktop),
                    "turned_off": sorted(turn_off),
                },
                indent=2,
            )
        )
    else:
        if before is not None and reserve is not None:
            gib = 1024 ** 3
            print(
                f"desktop:  was {before.desktop_allowance_bytes / gib:.1f} GiB "
                f"({before.desktop_allowance_basis}), now "
                f"{reserve.allowance_bytes / gib:.1f} GiB (measured); budget "
                f"{capability.available_bytes(backend.gpu.vram_bytes, before.desktop_allowance_bytes) / gib:.1f}"
                f" -> {capability.available_bytes(backend.gpu.vram_bytes, reserve.allowance_bytes) / gib:.1f} GiB"
            )
        _print_decisions(config, backend, decisions)

    if not args.write and before is None:
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


def _measure_step(config: Config, backend: Backend, *, gpu: bool) -> None:
    print("measuring this card:")
    try:
        ladder.run(
            config,
            backend,
            gpu=gpu,
            on_line=lambda line: print(f"  {line}"),
        )
    except (ladder.LadderError, OSError) as exc:
        print(f"  the measurement did not complete ({exc}); deciding on what is known")


def cmd_ladder(args: argparse.Namespace) -> int:
    config, backend = common.here(tolerate_stale_record=True)
    rungs = tuple(args.rung) if args.rung else ladder.RUNGS
    say = None if args.json else (lambda line: print(line))
    try:
        ladder.run(config, backend, rungs, gpu=not args.no_gpu, on_line=say)
    except ladder.LadderError as exc:
        return _fail(str(exc))
    if args.json:
        print(
            json.dumps(
                {
                    "card": _card_facts(config.home, backend),
                    "record": ladder.summary(config.home, backend.gpu),
                },
                indent=2,
            )
        )
    else:
        print(f"card: {_card_line(_card_facts(config.home, backend))}")
        print(f"recorded in {ladder.record_path(config.home)}")
    return EXIT_OK


def _capability_step(config: Config, backend: Backend, *job_types: str) -> int:
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
    card = ladder.card_for(config.home, backend.gpu)
    pool = capability.pool_name(backend.kind, backend.gpu.vendor)
    for job_type in job_types:
        plan = capability.install_plan(
            job_type,
            decisions,
            card=card,
            total_bytes=backend.gpu.vram_bytes,
            pool=pool,
            desktop_allowance_bytes=config.desktop_allowance_bytes,
            desktop_basis=config.desktop_allowance_basis,
        )
        print(f"your card ({plan['card_words']}), for {job_type}:")
        for row in plan["classes"]:
            print(f"  {row['line']}")
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


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    ladder_parser = subparsers.add_parser(
        "ladder",
        help="measure what this card can do (the install's measurement ladder)",
    )
    ladder_parser.add_argument(
        "--rung",
        action="append",
        choices=list(ladder.RUNGS),
        default=None,
        help="run only this rung (repeatable); every rung by default",
    )
    ladder_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="the nvidia-smi rung only; nothing is put on the card",
    )
    ladder_parser.add_argument("--json", action="store_true", help="machine-readable")
    ladder_parser.set_defaults(func=cmd_ladder)

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
        "--measure-desktop",
        action="store_true",
        help=(
            "re-measure what this PC's desktop holds on the card (nvidia-smi, "
            "five seconds) and replace the reserve with it, whatever its basis; "
            "refused while anything of Crucible's is on the card. Shows old and "
            "new and writes, as --write does"
        ),
    )
    capability_parser.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    capability_parser.set_defaults(func=cmd_capability)
