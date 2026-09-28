from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .. import (
    API_VERSION,
    VERSION,
    capabilityclasses,
    catalog,
    envpatches,
    hosttools,
    jobenv,
    ladder,
    llamacpp,
    service,
    verdict,
)
from ..backend import LLAMA_WINDOWS, MLX_DARWIN, Backend
from ..capabilityrecord import DESKTOP_BASIS_MEASURED, desktop_reserve_words
from ..config import Config, config_mode, crucible_home
from ..errors import ConfigError, NoViableBackend
from ..jobs import ALL_JOB_TYPES, build_registry
from ..memorybudget import gib_text
from ..narratorengines import NARRATOR_ENGINE_SAMPLING
from . import common
from .capability import _card_facts, _card_line
from .common import EXIT_OK, EXIT_REFUSED, _env_spec, _fail
from .install import INSTALLER_FOR

CAPABILITY_WRITE = "crucible capability --write"
DOCTOR_JSON = "crucible doctor --json"
RERUN_DOCTOR = "crucible doctor"
PATCH_LLM = "crucible env patch llm"


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    fix: str

    def __post_init__(self) -> None:
        if not self.fix.strip():
            raise ValueError(
                f"doctor finding {self.code!r} names no fix; every problem doctor "
                "reports must say what to run"
            )

    @classmethod
    def run(cls, code: str, detail: str, fix: str) -> "Finding":
        message = detail if f"`{fix}`" in detail else f"{detail}. Run `{fix}`"
        return cls(code, message, fix)

    @property
    def line(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class Section:
    name: str
    facts: dict[str, Any]
    findings: tuple[Finding, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Host:
    home: Path
    backend: Backend | None
    backend_refusal: NoViableBackend | None
    config: Config | None
    config_refusal: ConfigError | None
    ladder: dict[str, Any] | None


Check = Callable[[Host], Section]

REPORT_DEFAULTS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("healthy", lambda: False),
    ("crucible", lambda: {"version": VERSION, "api_version": API_VERSION}),
    ("home", lambda: None),
    ("config", lambda: None),
    ("backend", lambda: None),
    ("card", lambda: None),
    ("ladder", lambda: None),
    ("job_types", list),
    ("llm_env", lambda: None),
    ("worker_envs", list),
    ("tts_envs", dict),
    ("cuda_toolkit_links", list),
    ("llm_patches", list),
    ("capability", lambda: None),
    ("path", lambda: None),
    ("stranded_weights", lambda: None),
    ("ffmpeg", lambda: None),
    ("notes", list),
    ("problems", list),
)


def survey(home: Path) -> Host:
    try:
        backend, backend_refusal = common.detect_backend(), None
    except NoViableBackend as exc:
        backend, backend_refusal = None, exc
    summary = None if backend is None else ladder.summary(home, backend.gpu)
    try:
        config, config_refusal = common.load_config(home, tolerate_stale_record=True), None
    except ConfigError as exc:
        config, config_refusal = None, exc
    return Host(home, backend, backend_refusal, config, config_refusal, summary)


def _no_python_env_dir(home: Path) -> Path:
    return home / "envs" / "none"


def _tts_install_command(engine: str) -> str:
    return f"crucible install tts --narrator-engine {engine}"


def _install_command(family: str) -> str | None:
    installer = INSTALLER_FOR.get(family)
    if installer is None:
        return None
    if installer != "tts":
        return f"crucible install {installer}"
    return " or ".join(
        _tts_install_command(engine) for engine in sorted(NARRATOR_ENGINE_SAMPLING)
    )


def check_backend(host: Host) -> Section:
    backend = host.backend
    if backend is None:
        assert host.backend_refusal is not None
        return Section("backend", {}, (
            Finding(
                "no_viable_backend",
                f"{host.backend_refusal.reason}. {common.backend_hint()}",
                RERUN_DOCTOR,
            ),
        ))
    return Section("backend", {
        "backend": backend.to_dict(),
        "card": _card_facts(host.home, backend),
        "ladder": host.ladder,
    })


def check_path(host: Host) -> Section:
    shell_path = hosttools.search_path()
    entry: dict[str, Any] = {
        "shell": shell_path,
        "service": None,
        "mechanism": None,
        "definition": None,
        "agree": None,
    }
    mechanism = None
    if host.backend is not None:
        try:
            mechanism = service.mechanism_for(host.backend.kind)
        except service.ServiceError:
            mechanism = None
    if mechanism is not None:
        recorded = service.read_recorded_path(mechanism, service.user_home())
        entry["mechanism"] = mechanism
        entry["definition"] = str(service.definition_path(mechanism, service.user_home()))
        entry["service"] = recorded
        if recorded is not None:
            entry["agree"] = recorded == shell_path
    return Section("path", {"path": entry})


def _config_refusal(exc: ConfigError) -> Finding:
    text = str(exc)
    if "`crucible" not in text:
        return Finding.run("config", text, common.REINIT_COMMAND)
    start = text.index("`crucible") + 1
    end = text.find("`", start)
    return Finding("config", text, text[start:end] if end != -1 else common.REINIT_COMMAND)


def check_config(host: Host) -> Section:
    config = host.config
    if config is None:
        assert host.config_refusal is not None
        return Section("config", {}, (_config_refusal(host.config_refusal),))
    mode = config_mode(config.path)
    facts = {
        "path": str(config.path),
        "mode": mode,
        "name": config.name,
        "host": config.host,
        "port": config.port,
        **{
            f"enable_{job_type}": getattr(config, f"enable_{job_type}")
            for job_type in ("echo", "llm", "asr", "tts", "align", "rvc", "denoise")
        },
        "desktop_allowance_bytes": config.desktop_allowance_bytes,
        "desktop_allowance_basis": config.desktop_allowance_basis,
        "desktop_allowance_note": config.desktop_allowance_note,
        "desktop_reserve": desktop_reserve_words(
            config.desktop_allowance_bytes, config.desktop_allowance_basis
        ),
        "backend_kind": config.backend_kind,
    }
    findings: list[Finding] = []
    if sys.platform != "win32" and mode != "0o600":
        chmod = f"chmod 600 {config.path}"
        findings.append(Finding(
            "config_permissions",
            f"{config.path} is mode {mode}; the token should be readable only by "
            f"its owner. Run `{chmod}`",
            chmod,
        ))
    backend = host.backend
    if backend is not None and backend.kind != config.backend_kind:
        findings.append(Finding(
            "backend_changed",
            common.backend_changed_fix(config, backend),
            common.REINIT_COMMAND,
        ))
    return Section("config", {"config": facts}, tuple(findings))


def check_files(host: Host) -> Section:
    config = host.config
    if config is None:
        return Section("files", {})
    return Section("files", {
        "stranded_weights": catalog.stranded_weights(config),
        "ffmpeg": hosttools.ffmpeg_report(config.home),
    })


def _stale_findings(config: Config, backend: Backend) -> list[Finding]:
    record = config.capability
    assert record is not None
    stale: list[str] = []
    if record.backend_kind != backend.kind:
        stale.append(
            f"the record was decided on {record.backend_kind} "
            f"and this host is {backend.kind}; re-run `{CAPABILITY_WRITE}`"
        )
    if record.total_bytes != backend.gpu.vram_bytes:
        stale.append(
            f"the record was decided against {gib_text(record.total_bytes)} and this "
            f"host has {gib_text(backend.gpu.vram_bytes)}; re-run `{CAPABILITY_WRITE}`"
        )
    if record.desktop_allowance_bytes != config.desktop_allowance_bytes:
        stale.append(
            f"the record was decided with a {gib_text(record.desktop_allowance_bytes)} "
            f"desktop reserve and [accelerator] now says "
            f"{gib_text(config.desktop_allowance_bytes)}; re-run `{CAPABILITY_WRITE}`"
        )
    return [Finding("capability_stale", message, CAPABILITY_WRITE) for message in stale]


def check_capability(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None:
        return Section("capability", {})
    record = config.capability
    if record is None:
        return Section("capability", {"capability": None})
    findings = _stale_findings(config, backend)
    could_enable: list[str] = []
    for name in sorted({cls.job_type for cls in capabilityclasses.CLASSES}):
        rows = [record.row(cls.name) for cls in capabilityclasses.classes_for_job_type(name)]
        known = [row for row in rows if row is not None]
        if not known:
            continue
        fits = any(row.enabled for row in known)
        flagged = getattr(config, f"enable_{name}")
        if flagged and not fits:
            findings.append(Finding(
                "capability_contradicted",
                f"[jobs] enable_{name} is true and nothing behind it fits this host — "
                + "; ".join(f"{row.capability}: {row.reason}" for row in known)
                + f". `{CAPABILITY_WRITE}` turns the flag off",
                CAPABILITY_WRITE,
            ))
        if fits and not flagged and name in INSTALLER_FOR:
            could_enable.append(name)
    entry = {
        **record.to_dict(),
        "stale": any(finding.code == "capability_stale" for finding in findings),
        "could_enable": could_enable,
    }
    return Section("capability", {"capability": entry}, tuple(findings))


def _plan_or_refusal(call: Callable[[], "jobenv.EnvPlan"]) -> "jobenv.EnvPlan | str":
    try:
        return call()
    except jobenv.EnvError as exc:
        return str(exc)


def _provenance(
    label: str,
    status: jobenv.EnvStatus,
    recipe: Path,
    plan: "jobenv.EnvPlan | str",
    install: str,
) -> tuple[dict[str, Any], list[Finding]]:
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
        return entry, [Finding.run(label, plan, install)]
    if plan.action != jobenv.PLAN_NOTHING:
        return entry, [Finding(
            label,
            f"{plan.action} — {plan.detail}. `{install}` brings it up to {recipe.name}",
            install,
        )]
    return entry, []


def _env_report(
    label: str, home: Path, spec: jobenv.EnvSpec, backend_kind: str, install: str
) -> tuple[dict[str, Any], list[Finding]]:
    try:
        status = jobenv.env_status(home, spec, backend_kind)
        recipe = jobenv.recipe_for(spec)
    except jobenv.EnvError as exc:
        return {"installed": False, "detail": str(exc)}, [Finding.run(label, str(exc), install)]
    findings = [] if status.installed else [Finding.run(label, status.detail, install)]
    entry = status.to_dict()
    entry["provenance"], drift = _provenance(
        label,
        status,
        recipe,
        _plan_or_refusal(lambda: jobenv.plan_install(home, spec, backend_kind)),
        install,
    )
    return entry, findings + drift


def _llama_engine_report(config: Config, backend: Backend) -> tuple[dict[str, Any], list[Finding]]:
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
        return entry, [Finding("llm_env", entry["detail"], llamacpp.INSTALL_COMMAND)]
    entry["bytes"] = found.bytes
    entry["pulled"] = found.pulled
    return entry, []


def check_llm_env(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None or not config.enable_llm:
        return Section("llm_env", {})
    if backend.kind == LLAMA_WINDOWS:
        entry, findings = _llama_engine_report(config, backend)
    else:
        entry, findings = _env_report(
            "llm_env",
            config.home,
            jobenv.llm_env(backend.kind),
            backend.kind,
            llamacpp.INSTALL_COMMAND,
        )
    return Section("llm_env", {"llm_env": entry}, tuple(findings))


def check_worker_envs(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None:
        return Section("worker_envs", {})
    entries: list[dict[str, Any]] = []
    findings: list[Finding] = []
    for job_type in jobenv.WORKER_JOB_TYPES:
        if not getattr(config, f"enable_{job_type}"):
            continue
        install = f"crucible install {job_type}"
        label = f"{job_type}_env"
        try:
            spec = jobenv.worker_env(job_type, backend.kind)
        except jobenv.EnvError as exc:
            entries.append({"job_type": job_type, "installed": False, "detail": str(exc)})
            findings.append(Finding.run(label, str(exc), install))
            continue
        entry, found = _env_report(label, config.home, spec, backend.kind, install)
        entries.append({"job_type": job_type, **entry})
        findings.extend(found)
    return Section("worker_envs", {"worker_envs": entries}, tuple(findings))


def _cuda_toolkit_links(home: Path, backend_kind: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        engine_env = jobenv.env_dir(home, jobenv.tts_env(engine, backend_kind))
        if envpatches.site_packages(engine_env) is None:
            continue
        for entry in envpatches.check_cuda_toolkit_links(engine_env):
            links.append({"engine": engine, **entry})
    return links


def check_tts_envs(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None or not config.enable_tts:
        return Section("tts_envs", {})
    envs: dict[str, Any] = {}
    findings: list[Finding] = []
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        envs[engine], found = _env_report(
            f"tts_env[{engine}]",
            config.home,
            jobenv.tts_env(engine, backend.kind),
            backend.kind,
            _tts_install_command(engine),
        )
        findings.extend(found)
    links = _cuda_toolkit_links(config.home, backend.kind) if backend.kind == "cuda-linux" else []
    for entry in links:
        if entry["status"] not in envpatches.SOUND_STATUSES:
            install = _tts_install_command(entry["engine"])
            findings.append(Finding(
                f"cuda_toolkit_link[{entry['engine']}:{entry['id']}]",
                f"{entry['status']} — {entry['detail']}. {entry['why']}. "
                f"`{install}` re-links it",
                install,
            ))
    return Section(
        "tts_envs", {"tts_envs": envs, "cuda_toolkit_links": links}, tuple(findings)
    )


def check_llm_patches(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None or not config.enable_llm:
        return Section("llm_patches", {})
    if backend.kind == LLAMA_WINDOWS:
        env_dir, pins = _no_python_env_dir(config.home), {}
    else:
        spec = jobenv.llm_env(backend.kind)
        env_dir = jobenv.env_dir(config.home, spec)
        pins = jobenv.recipe_pins(jobenv.recipe_for(spec))
    rows = envpatches.check("llm", env_dir, pins)
    findings = tuple(
        Finding(
            f"llm_patch[{entry['id']}]",
            f"{entry['status']} — {entry['detail']}. {entry['why']}. Run `{PATCH_LLM}`",
            PATCH_LLM,
        )
        for entry in rows
        if entry["status"] not in envpatches.SOUND_STATUSES
        and entry["status"] != envpatches.NO_ENV
    )
    return Section("llm_patches", {"llm_patches": rows}, findings)


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


def _not_ready(entry: dict[str, Any]) -> Finding:
    install = _install_command(ALL_JOB_TYPES[entry["name"]])
    if install is None:
        return Finding(
            "job_type_not_ready",
            f"{entry['name']}: {entry['detail']}. `{DOCTOR_JSON}` carries the detail",
            DOCTOR_JSON,
        )
    return Finding(
        "job_type_not_ready",
        f"{entry['name']}: {entry['detail']}. `{install}` builds what it runs on",
        install,
    )


def check_job_types(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None:
        return Section("job_types", {})
    reports = _job_type_reports(config, backend)
    unready = [entry for entry in reports if entry["enabled"] and not entry["ready"]]
    notes = tuple(
        f"job_type_awaiting_weights: {entry['name']}: {entry['detail']}"
        for entry in unready
        if entry.get("awaiting_weights")
    )
    findings = tuple(_not_ready(entry) for entry in unready if not entry.get("awaiting_weights"))
    return Section("job_types", {"job_types": reports}, findings, notes)


def check_desktop_reserve(host: Host) -> Section:
    config, backend = host.config, host.backend
    if config is None or backend is None:
        return Section("desktop_reserve", {})
    if config.desktop_allowance_basis == DESKTOP_BASIS_MEASURED:
        return Section("desktop_reserve", {})
    card_rung = ((host.ladder or {}).get("rungs") or {}).get(ladder.CARD)
    seen = ((card_rung or {}).get("facts") or {}).get("desktop_bytes_max")
    if not isinstance(seen, int):
        return Section("desktop_reserve", {})
    would = ladder.desktop_allowance_from(seen)
    if would >= config.desktop_allowance_bytes:
        return Section("desktop_reserve", {})
    return Section("desktop_reserve", {}, notes=(
        f"desktop_reserve_unmeasured: the reserve is "
        f"{gib_text(config.desktop_allowance_bytes)} "
        f"({config.desktop_allowance_basis}); the ladder saw this "
        f"desktop at up to {gib_text(seen)}, which would keep "
        f"{gib_text(would)}. `crucible capability "
        "--measure-desktop` re-measures it",
    ))


CHECKS: tuple[Check, ...] = (
    check_backend,
    check_path,
    check_config,
    check_files,
    check_capability,
    check_llm_env,
    check_worker_envs,
    check_tts_envs,
    check_llm_patches,
    check_job_types,
    check_desktop_reserve,
)


def empty_report(home: Path) -> dict[str, Any]:
    report = {key: make() for key, make in REPORT_DEFAULTS}
    report["home"] = str(home)
    return report


def assemble(home: Path, sections: Iterable[Section]) -> dict[str, Any]:
    report = empty_report(home)
    for section in sections:
        stray = set(section.facts) - set(report)
        if stray:
            raise KeyError(f"doctor check {section.name} reports unknown keys {sorted(stray)}")
        report.update(section.facts)
        report["problems"].extend(finding.line for finding in section.findings)
        report["notes"].extend(section.notes)
    report["healthy"] = not report["problems"]
    return report


def _doctor_report(checks: Iterable[Check] = CHECKS) -> dict[str, Any]:
    host = survey(crucible_home())
    return assemble(host.home, (check(host) for check in checks))


def _mark(ready: bool) -> str:
    return "ready" if ready else "NOT READY"


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


def _env_lines(title: str, entry: dict[str, Any]) -> Iterator[str]:
    yield f"{title}: {_mark(entry['installed'])} — {entry['detail']}"
    if "provenance" in entry:
        yield f"         {_provenance_line(entry['provenance'])}"


def lines_header(report: dict[str, Any]) -> Iterator[str]:
    yield f"crucible {VERSION} (api {API_VERSION})"
    yield f"home:    {report['home']}"


def lines_backend(report: dict[str, Any]) -> Iterator[str]:
    backend = report["backend"]
    if backend is None:
        yield "backend: NONE"
        return
    gpu = backend["gpu"]
    yield f"backend: {backend['kind']} on {backend['platform']}/{backend['arch']}"
    yield (
        f"gpu:     {gpu['vendor']} {gpu['name']} "
        f"({gib_text(gpu['vram_bytes'])}) — {backend['detail']}"
    )
    if (
        report["card"] is not None
        and backend["kind"] != MLX_DARWIN
        and gpu["vendor"] != verdict.CPU_VENDOR
    ):
        yield f"card:    {_card_line(report['card'])}"
    measured = report.get("ladder")
    if measured is not None:
        if measured["stale"]:
            yield f"note:    {measured['stale']}"
        for name, row in measured["rungs"].items():
            yield f"ladder {name}: {row['outcome']} — {row['detail']}"


def lines_path(report: dict[str, Any]) -> Iterator[str]:
    entry = report["path"]
    if entry is None:
        return
    yield f"PATH (this shell):   {entry['shell'] or '(empty)'}"
    if entry["service"] is None:
        where = entry["definition"]
        yield (
            "PATH (the service):  none recorded — no "
            f"{entry['mechanism'] or 'service'} definition at {where}"
            if where
            else "PATH (the service):  none recorded"
        )
        return
    yield f"PATH (the service):  {entry['service']}"
    if entry["agree"] is False:
        yield (
            "note:    the two differ, which is normal. Every line "
            "below is what THIS shell can see; the service sees "
            "the second one"
        )


def lines_config(report: dict[str, Any]) -> Iterator[str]:
    config = report["config"]
    if config is None:
        yield "config:  MISSING"
        return
    yield f"config:  {config['path']} (mode {config['mode']})"
    yield f"serves:  {config['name']} on {config['host']}:{config['port']}"
    yield f"reserve: {config['desktop_reserve']}"
    if config["desktop_allowance_note"]:
        yield f"         {config['desktop_allowance_note']}"


def lines_envs(report: dict[str, Any]) -> Iterator[str]:
    env = report["llm_env"]
    if env is not None:
        yield from _env_lines(
            "engine " if env.get("engine") == "llama-server" else "llm env", env
        )
    for worker_env in report["worker_envs"]:
        yield from _env_lines(f"{worker_env['job_type']} env", worker_env)
    for engine, entry in sorted(report["tts_envs"].items()):
        yield from _env_lines(f"tts env ({engine})", entry)


def lines_patches(report: dict[str, Any]) -> Iterator[str]:
    for entry in report["cuda_toolkit_links"]:
        mark = "applied" if entry["applied"] else entry["status"].upper()
        yield (
            f"cuda toolkit link ({entry['engine']}, {entry['id']}): {mark} — "
            f"{entry['detail']}"
        )
    for entry in report["llm_patches"]:
        if entry["status"] == envpatches.NOT_APPLICABLE:
            mark = "n/a"
        else:
            mark = "applied" if entry["applied"] else entry["status"].upper()
        yield f"llm patch ({entry['id']}): {mark} — {entry['detail']}"


def lines_capability(report: dict[str, Any]) -> Iterator[str]:
    entry = report["capability"]
    if entry is None:
        yield (
            "capability: NOT DECIDED — nothing has probed this host's card "
            "against the models; run `crucible capability`"
        )
        return
    for row in entry["classes"]:
        mark = "yes" if row["enabled"] else "NO"
        yield f"capability {row['capability']}: {mark} — {row['reason']}"
    for name in entry["could_enable"]:
        yield (
            f"note:    this host can hold {name}, and [jobs] enable_{name} "
            f"is off — `crucible install {INSTALLER_FOR[name]}` builds its "
            "env and turns it on"
        )


def lines_ffmpeg(report: dict[str, Any]) -> Iterator[str]:
    tool = report["ffmpeg"]
    if tool is None:
        return
    pinned, where = tool["pinned_version"], tool["tools_bin"]
    if tool["source"] == "crucible":
        yield f"ffmpeg:  {tool['path']} (Crucible's pinned {pinned})"
    elif tool["source"] == "path" and pinned is None:
        yield (
            f"ffmpeg:  {tool['path']} (from PATH; there is no pinned "
            f"build for {tool['platform']})"
        )
    elif tool["source"] == "path":
        yield (
            f"ffmpeg:  {tool['path']} (from PATH). note: Crucible's pinned "
            f"{pinned} is not in {where}; installing any job type puts it there"
        )
    elif pinned is not None:
        yield (
            f"ffmpeg:  NONE — Crucible's pinned {pinned} is not in {where}; "
            "installing any job type puts it there"
        )
    else:
        yield f"ffmpeg:  NONE — no pinned build for {tool['platform']}, and none on PATH"


def _job_mark(entry: dict[str, Any]) -> str:
    if entry["ready"]:
        return "ready"
    if not entry["enabled"]:
        return "off"
    if entry.get("awaiting_weights"):
        return "waiting for weights"
    return "NOT READY"


def lines_job_types(report: dict[str, Any]) -> Iterator[str]:
    for entry in report["job_types"]:
        yield f"job {entry['name']}: {_job_mark(entry)} — {entry['detail']}"


def lines_stranded_weights(report: dict[str, Any]) -> Iterator[str]:
    for entry in report["stranded_weights"] or ():
        why = f"no manifest in this build declares {entry['id']!r} on {entry['backend']}"
        yield (
            f"note:    {entry['bytes'] / 1e9:.2f} GB of weights at "
            f"{entry['path']} belong to nothing: {why}. Nothing will use "
            "them; delete the directory to reclaim the space"
        )


def lines_notes(report: dict[str, Any]) -> Iterator[str]:
    for note in report["notes"]:
        yield f"note:    {note}"


TEXT_SECTIONS: tuple[Callable[[dict[str, Any]], Iterable[str]], ...] = (
    lines_header,
    lines_backend,
    lines_path,
    lines_config,
    lines_envs,
    lines_patches,
    lines_capability,
    lines_ffmpeg,
    lines_job_types,
    lines_stranded_weights,
    lines_notes,
)


def render_text(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    out = [line for render in TEXT_SECTIONS for line in render(report)]
    err = [f"PROBLEM: {problem}" for problem in report["problems"]]
    return out, err


def render(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2))
        return
    out, err = render_text(report)
    for line in out:
        print(line)
    for line in err:
        print(line, file=sys.stderr)
    print("healthy" if report["healthy"] else "unhealthy")


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _doctor_report()
    render(report, args.json)
    return EXIT_OK if report["healthy"] else EXIT_REFUSED


def cmd_env_patch(args: argparse.Namespace) -> int:
    config, backend = common.here()
    job_type = args.job_type
    if envpatches.patches_for(job_type) is None:
        return _fail(
            f"job type {job_type!r} carries no site-packages patches; the "
            f"types that do are {list(envpatches.patched_job_types())}"
        )
    if job_type == "llm" and backend.kind == LLAMA_WINDOWS:
        rows = envpatches.check(job_type, _no_python_env_dir(config.home), {})
    else:
        try:
            spec = _env_spec(job_type, None, backend.kind)
            recipe = jobenv.recipe_for(spec)
            pins = jobenv.recipe_pins(recipe)
        except jobenv.EnvError as exc:
            return _fail(str(exc))
        directory = jobenv.env_dir(config.home, spec)
        python = jobenv.env_python(config.home, spec)
        if not python.is_file():
            print(
                f"{job_type} env: not installed at {directory}; nothing to "
                f"patch (`crucible install {job_type}` applies them)"
            )
            return EXIT_OK
        try:
            rows = envpatches.apply(job_type, directory, python, pins, on_line=print)
        except envpatches.PatchError as exc:
            return _fail(f"env_patch_failed: {exc}")
    for row in rows:
        print(f"{job_type} patch ({row['id']}): {row['status']} — {row['detail']}")
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
