from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .. import (
    API_VERSION,
    VERSION,
    capability,
    catalog,
    envpatches,
    hosttools,
    jobenv,
    ladder,
    llamacpp,
    service,
)
from ..backend import Backend, LLAMA_WINDOWS, MLX_DARWIN
from ..config import (
    Config,
    DESKTOP_BASIS_MEASURED,
    config_mode,
    crucible_home,
    desktop_reserve_words,
)
from ..errors import ConfigError, NoViableBackend
from ..jobs import ALL_JOB_TYPES, build_registry
from ..voices import NARRATOR_ENGINE_SAMPLING
from . import common
from .capability import _card_facts, _card_line
from .common import EXIT_OK, EXIT_REFUSED, _backend_mismatch, _env_spec, _fail
from .install import INSTALLER_FOR


def _no_python_env_dir(home: Path) -> Path:
    return home / "envs" / "none"


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
                    "awaiting_weights": False,
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
                "awaiting_weights": status.awaiting_weights,
                "models": [m.to_dict() for m in plugin.describe_models()],
            }
        )
    return reports


def _llama_engine_report(
    report: dict[str, Any], config: Config, backend: Backend
) -> dict[str, Any]:
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
    try:
        return call()
    except jobenv.EnvError as exc:
        return str(exc)


def _provenance(
    report: dict[str, Any],
    label: str,
    status: jobenv.EnvStatus,
    recipe: Path,
    plan: "jobenv.EnvPlan | str",
) -> dict[str, Any]:
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
    if entry["environment_sha256"] is None:
        recipe = f"{entry['recipe']} not stamped"
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
        "card": None,
        "ladder": None,
        "job_types": [],
        "llm_env": None,
        "worker_envs": [],
        "tts_envs": {},
        "cuda_toolkit_links": [],
        "llm_patches": [],
        "capability": None,
        "path": None,
        "stranded_weights": None,
        "ffmpeg": None,
        "notes": [],
        "problems": [],
    }

    try:
        backend = common.detect_backend()
        report["backend"] = backend.to_dict()
        report["card"] = _card_facts(home, backend)
        report["ladder"] = ladder.summary(home, backend.gpu)
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
        config = common.load_config(home)
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
            "desktop_allowance_basis": config.desktop_allowance_basis,
            "desktop_allowance_note": config.desktop_allowance_note,
            "desktop_reserve": desktop_reserve_words(
                config.desktop_allowance_bytes, config.desktop_allowance_basis
            ),
            "backend_kind": config.backend_kind,
        }
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
        report["ffmpeg"] = hosttools.ffmpeg_report(config.home)

    if config is not None and backend is not None:
        _capability_report(report, config, backend)
        if config.enable_llm:
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
        for job_type in jobenv.WORKER_JOB_TYPES:
            if not getattr(config, f"enable_{job_type}"):
                continue
            try:
                spec = jobenv.worker_env(job_type, backend.kind)
            except jobenv.EnvError as exc:
                report["worker_envs"].append(
                    {"job_type": job_type, "installed": False, "detail": str(exc)}
                )
                report["problems"].append(f"{job_type}_env: {exc}")
                continue
            entry = _env_report(
                report, f"{job_type}_env", config.home, spec, backend.kind
            )
            report["worker_envs"].append({"job_type": job_type, **entry})
        if config.enable_tts:
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
            if backend.kind == "cuda-linux":
                for engine in sorted(NARRATOR_ENGINE_SAMPLING):
                    engine_env = jobenv.env_dir(
                        config.home, jobenv.tts_env(engine, backend.kind)
                    )
                    if envpatches.site_packages(engine_env) is None:
                        continue
                    for entry in envpatches.check_cuda_toolkit_links(engine_env):
                        report["cuda_toolkit_links"].append(
                            {"engine": engine, **entry}
                        )
            for entry in report["cuda_toolkit_links"]:
                if entry["status"] not in envpatches.SOUND_STATUSES:
                    report["problems"].append(
                        f"cuda_toolkit_link[{entry['engine']}:{entry['id']}]: "
                        f"{entry['status']} — {entry['detail']}. {entry['why']}"
                    )
        if config.enable_llm:
            if backend.kind == LLAMA_WINDOWS:
                llm_env_dir, llm_pins = _no_python_env_dir(config.home), {}
            else:
                llm_spec = jobenv.llm_env(backend.kind)
                llm_env_dir = jobenv.env_dir(config.home, llm_spec)
                llm_pins = jobenv.recipe_pins(jobenv.recipe_for(llm_spec))
            report["llm_patches"] = envpatches.check("llm", llm_env_dir, llm_pins)
            for entry in report["llm_patches"]:
                if entry["status"] not in envpatches.SOUND_STATUSES and (
                    entry["status"] != envpatches.NO_ENV
                ):
                    report["problems"].append(
                        f"llm_patch[{entry['id']}]: {entry['status']} — "
                        f"{entry['detail']}. {entry['why']}. Run `crucible env "
                        "patch llm`"
                    )
        report["job_types"] = _job_type_reports(config, backend)
        for entry in report["job_types"]:
            if entry["enabled"] and not entry["ready"]:
                if entry.get("awaiting_weights"):
                    report["notes"].append(
                        f"job_type_awaiting_weights: {entry['name']}: {entry['detail']}"
                    )
                    continue
                report["problems"].append(
                    f"job_type_not_ready: {entry['name']}: {entry['detail']}"
                )

    if (
        config is not None
        and backend is not None
        and config.desktop_allowance_basis != DESKTOP_BASIS_MEASURED
    ):
        card_rung = ((report.get("ladder") or {}).get("rungs") or {}).get(ladder.CARD)
        seen = ((card_rung or {}).get("facts") or {}).get("desktop_bytes_max")
        if isinstance(seen, int):
            would = ladder.desktop_allowance_from(seen)
            if would < config.desktop_allowance_bytes:
                gib = 1024 ** 3
                report["notes"].append(
                    f"desktop_reserve_unmeasured: the reserve is "
                    f"{config.desktop_allowance_bytes / gib:.1f} GiB "
                    f"({config.desktop_allowance_basis}); the ladder saw this "
                    f"desktop at up to {seen / gib:.1f} GiB, which would keep "
                    f"{would / gib:.1f} GiB. `crucible capability "
                    "--measure-desktop` re-measures it"
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
            if (
                report["card"] is not None
                and backend["kind"] != MLX_DARWIN
                and gpu["vendor"] != capability.CPU_VENDOR
            ):
                print(f"card:    {_card_line(report['card'])}")
            measured = report.get("ladder")
            if measured is not None:
                if measured["stale"]:
                    print(f"note:    {measured['stale']}")
                for name, row in measured["rungs"].items():
                    print(f"ladder {name}: {row['outcome']} — {row['detail']}")
        path_entry = report["path"]
        if path_entry is not None:
            print(f"PATH (this shell):   {path_entry['shell'] or '(empty)'}")
            if path_entry["service"] is None:
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
            print(f"reserve: {config['desktop_reserve']}")
            if config["desktop_allowance_note"]:
                print(f"         {config['desktop_allowance_note']}")
        env = report["llm_env"]
        if env is not None:
            mark = "ready" if env["installed"] else "NOT READY"
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
        for entry in report["cuda_toolkit_links"]:
            mark = "applied" if entry["applied"] else entry["status"].upper()
            print(
                f"cuda toolkit link ({entry['engine']}, {entry['id']}): {mark} — "
                f"{entry['detail']}"
            )
        for entry in report["llm_patches"]:
            if entry["status"] == envpatches.NOT_APPLICABLE:
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
        tool = report["ffmpeg"]
        if tool is not None:
            if tool["source"] == "crucible":
                print(f"ffmpeg:  {tool['path']} (Crucible's pinned {tool['pinned_version']})")
            elif tool["source"] == "path" and tool["pinned_version"] is None:
                print(
                    f"ffmpeg:  {tool['path']} (from PATH; there is no pinned "
                    f"build for {tool['platform']})"
                )
            elif tool["source"] == "path":
                print(
                    f"ffmpeg:  {tool['path']} (from PATH). note: Crucible's pinned "
                    f"{tool['pinned_version']} is not in {tool['tools_bin']}; "
                    "installing any job type puts it there"
                )
            else:
                print(
                    "ffmpeg:  NONE — "
                    + (
                        f"Crucible's pinned {tool['pinned_version']} is not in "
                        f"{tool['tools_bin']}; installing any job type puts it there"
                        if tool["pinned_version"] is not None
                        else f"no pinned build for {tool['platform']}, and none on PATH"
                    )
                )
        for entry in report["job_types"]:
            if entry["ready"]:
                mark = "ready"
            elif not entry["enabled"]:
                mark = "off"
            elif entry.get("awaiting_weights"):
                mark = "waiting for weights"
            else:
                mark = "NOT READY"
            print(f"job {entry['name']}: {mark} — {entry['detail']}")
        for entry in report["stranded_weights"] or ():
            why = (
                f"no manifest in this build declares {entry['id']!r} on "
                f"{entry['backend']}"
            )
            print(
                f"note:    {entry['bytes'] / 1e9:.2f} GB of weights at "
                f"{entry['path']} belong to nothing: {why}. Nothing will use "
                "them; delete the directory to reclaim the space"
            )
        for note in report["notes"]:
            print(f"note:    {note}")
        for problem in report["problems"]:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print("healthy" if report["healthy"] else "unhealthy")
    return EXIT_OK if report["healthy"] else EXIT_REFUSED


def cmd_env_patch(args: argparse.Namespace) -> int:
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
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force`"
        )
    if envpatches.patches_for(args.job_type) is None:
        return _fail(
            f"job type {args.job_type!r} carries no site-packages patches; the "
            f"types that do are {list(envpatches.patched_job_types())}"
        )
    if args.job_type == "llm" and backend.kind == LLAMA_WINDOWS:
        rows = envpatches.check("llm", _no_python_env_dir(config.home), {})
    else:
        try:
            spec = _env_spec(args.job_type, None, backend.kind)
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
        except envpatches.PatchError as exc:
            return _fail(f"env_patch_failed: {exc}")
    for row in rows:
        print(f"{args.job_type} patch ({row['id']}): {row['status']} — {row['detail']}")
    unsound = [r for r in rows if r["status"] not in envpatches.SOUND_STATUSES]
    if unsound:
        return _fail(
            "env_patch_failed: "
            + "; ".join(f"{r['id']} is {r['status']}" for r in unsound)
        )
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
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
    env_patch.set_defaults(func=cmd_env_patch)
