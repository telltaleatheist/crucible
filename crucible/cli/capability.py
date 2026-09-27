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
from ..errors import ConfigError, NoViableBackend
from . import common
from .common import EXIT_OK, _fail


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
        # WHAT THIS CARD IS, beside how big (fresh-install #48): its
        # generation and the ladder's measured facts. A 6 GB Turing card and a
        # 6 GB Ampere one hold the same bytes and run different precisions.
        card=ladder.card_for(config.home, backend.gpu),
    )


def _card_facts(home: Path, backend: Backend) -> dict[str, Any]:
    """What this card IS, beside how much it holds: `crucible capability` and
    `crucible doctor` print the same facts from this one function, which is
    `ladder.card_for`'s card (declared facts plus measured ones).

    A feature is null where it is unknown — no NVIDIA card, a driver that
    would not report `compute_cap`, or a fact the ladder has not measured —
    and never false for that reason, which would read as a card that failed.
    """
    return ladder.card_for(home, backend.gpu).to_dict()


def _card_line(facts: dict[str, Any]) -> str:
    """One line a person can read: `sm_75 (7.5): bf16 NO (needs 8.0), ...`."""
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

    # A fact nobody has measured is left off the line rather than printed as
    # "unknown" on every host: the ladder's GPU rungs only exist for the Linux
    # engines, and a line that nags where they cannot run is noise.
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
        # AND ITS BASIS, or a measured reserve would come back from every
        # install as "stated" (`write_config`'s parameter says why).
        desktop_allowance_basis=config.desktop_allowance_basis,
        desktop_allowance_note=config.desktop_allowance_note,
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
    if backend.kind != MLX_DARWIN and backend.gpu.vendor != capability.CPU_VENDOR:
        # A Mac and a cardless Windows box have no compute capability to
        # report, and a line saying "unknown" there would read as a fault.
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

    `--measure-desktop` (2026-09-26) is the one door that REPLACES a reserve
    after init: it samples the desktop (`ladder.measure_desktop_reserve`, the
    same rule `crucible init` uses), refuses by name while anything of
    Crucible's is on the card, prints old and new, and writes the new reserve
    with basis "measured" and the record decided on it. A person runs it;
    nothing runs it for them, because a stated reserve is never changed on its
    own (owens-pc keeps 3 GiB for streaming).
    """
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = common.detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force`"
        )

    # `--measure-desktop` (2026-09-26): the DELIBERATE re-measure, for a
    # machine initialised before `crucible init` measured — kylies-pc holds
    # the 3 GiB every NVIDIA card used to get. Nothing else ever replaces a
    # stated reserve, so this refuses rather than guesses when anything of
    # Crucible's is on the card, says old and new, and writes.
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
                    "card": _card_facts(config.home, backend),
                    "total_bytes": backend.gpu.vram_bytes,
                    "desktop_allowance_bytes": config.desktop_allowance_bytes,
                    "desktop_allowance_basis": config.desktop_allowance_basis,
                    "desktop_allowance_note": config.desktop_allowance_note,
                    "desktop_reserve": desktop_reserve_words(
                        config.desktop_allowance_bytes, config.desktop_allowance_basis
                    ),
                    # Old and new, when `--measure-desktop` replaced it.
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
    """The measurement ladder, run by install before it decides (crucible/ladder.py).

    Owen, 2026-09-26: *"our measurement tool should determine how much space is
    available, whether tensors are available, cuda graphs, vllm, etc. and
    install the best the user can use"*. So install measures FIRST and then
    decides: `_decide_here` reads the record this writes (`ladder.card_for`),
    and "the best the user can use" is the best-first walk over what this card
    measured it can start, at the precision it can run.

    Never a refusal of the install. A GPU rung that finds the card in use waits
    (`waiting`, nothing recorded as failed) and the install goes on with what
    is known; unknown refuses nothing. `gpu=False` is `--no-gpu-measure`.
    """
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
    """`crucible ladder` — measure what this card can do, on demand.

    The install's own step (`_measure_step`), for a card that was in use then,
    a driver or card changed since (`crucible doctor` says the record is
    stale), or a person who wants to know. `--no-gpu` puts nothing on the card.
    It records; `crucible capability --write` (or the next install) decides
    from what it recorded.
    """
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = common.detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
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
    # THE SAME WORDS THE UI's CONFIRMATION SHOWS (`capability.install_plan`),
    # so a person installing from a console reads what one installing from
    # the operator page was asked to accept.
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
