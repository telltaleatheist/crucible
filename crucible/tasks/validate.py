from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .. import capability, catalog, jobenv, tasks
from ..backend import Backend
from ..config import Config
from ..errors import ApiError
from ..jobs.llm import llm_engine_status
from ..voices import NARRATOR_ENGINE_SAMPLING
from . import hostdoor
from .states import TASK_TYPES

MODULE_KEYS = frozenset({"name", "version", "job_types", "needs", "subjects"})


@dataclass(frozen=True)
class ModuleEntry:

    name: str
    job_type: str | None = None
    narrator_engine: str | None = None
    kind: str | None = None
    subject_id: str | None = None
    capability_class: str | None = None


def require_installable(job_type: str) -> None:
    missing = jobenv.no_installer(job_type)
    if missing is None:
        return
    installer = missing.shared_with
    if installer is not None:
        raise ApiError(
            400,
            "unknown_job_type",
            f"{missing.words}. Name {installer!r} instead",
            {"job_type": job_type, "installed_by": installer},
        )
    raise ApiError(400, "unknown_job_type", missing.words)


def require_narrator_engine(job_type: str, narrator_engine: str | None) -> None:
    if job_type == "tts":
        if narrator_engine is None:
            raise ApiError(
                400,
                "narrator_engine_required",
                "installing 'tts' needs narrator_engine: on cuda-linux two "
                "engines cannot share a venv, so there is one env per engine "
                f"and no default. This build knows "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}",
            )
        if narrator_engine not in NARRATOR_ENGINE_SAMPLING:
            raise ApiError(
                400,
                "narrator_engine_required",
                f"{narrator_engine!r} is not one of narrator's engines; they are "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}",
            )
        return
    if narrator_engine is not None:
        raise ApiError(
            400,
            "narrator_engine_refused",
            f"narrator_engine names which tts env to build and means nothing for "
            f"{job_type!r}, which has exactly one env per host",
        )


def env_installed(config: Config, backend: Backend, job_type: str, engine: str | None) -> bool:
    try:
        if job_type == "llm":
            return llm_engine_status(config, backend).installed
        if job_type in jobenv.WORKER_JOB_TYPES:
            spec = jobenv.worker_env(job_type, backend.kind)
        else:
            spec = jobenv.tts_env(engine or "", backend.kind)
        return jobenv.env_status(config.home, spec, backend.kind).installed
    except jobenv.EnvError:
        return False


def install_label(job_type: str, narrator_engine: str | None) -> str:
    return f"install {job_type}" + (f" ({narrator_engine})" if narrator_engine else "")


def validate_pull(config: Config, backend: Backend, request: dict[str, Any]) -> None:
    kind, subject_id = request["kind"], request["id"]
    if kind not in catalog.KINDS:
        raise ApiError(
            404,
            "unknown_subject",
            f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
            {"kind": kind, "id": subject_id},
        )
    subject = catalog.find(config, backend, kind, subject_id)
    if subject is None:
        raise ApiError(
            404,
            "unknown_subject",
            f"this server has no {kind} called {subject_id!r} for "
            f"{backend.kind}. GET /v1/catalog lists every subject it can hold",
            {"kind": kind, "id": subject_id},
        )
    if subject.installed() is not None:
        raise ApiError(
            409,
            "already_installed",
            f"{kind} {subject_id!r} is already installed on this server. A pull "
            f"of an installed subject is refused rather than skipped; to replace "
            f"it deliberately, run `{subject.pull_command} --force` on the server",
            {"kind": kind, "id": subject_id},
        )


def validate_install(config: Config, backend: Backend, request: dict[str, Any]) -> None:
    job_type, narrator_engine = request["job_type"], request.get("narrator_engine")
    require_installable(job_type)
    require_narrator_engine(job_type, narrator_engine)
    if tasks.env_installed(config, backend, job_type, narrator_engine):
        raise ApiError(
            409,
            "job_type_installed",
            f"job type {job_type!r}"
            + (f" ({narrator_engine})" if narrator_engine else "")
            + " already has its env on this server. To rebuild it deliberately, "
            "run `crucible install` with --force on the server",
            {"job_type": job_type, "narrator_engine": narrator_engine},
        )


def validate_module_request(config: Config, backend: Backend, request: dict[str, Any]) -> None:
    validate_module(config, backend, request["module"])


def validate_engine(config: Config, backend: Backend, request: dict[str, Any]) -> None:
    hostdoor.door_for_move(backend, request["target"])


def validate_engine_restart(config: Config, backend: Backend, request: dict[str, Any]) -> None:
    hostdoor.door_for_restart()


RequestValidator = Callable[[Config, Backend, dict[str, Any]], None]

VALIDATORS: dict[str, RequestValidator] = {
    "pull": validate_pull,
    "install": validate_install,
    "module": validate_module_request,
    "engine": validate_engine,
    "engine-restart": validate_engine_restart,
}


def validate_request(config: Config, backend: Backend, request: dict[str, Any]) -> None:
    task_type = request["type"]
    validator = VALIDATORS.get(task_type)
    if validator is None:
        raise ApiError(
            400,
            "invalid_request",
            f"{task_type!r} is not a task type; they are {list(TASK_TYPES)}",
        )
    validator(config, backend, request)


def touches_the_registry(request: dict[str, Any]) -> bool:
    task_type = request["type"]
    if task_type == "install":
        return True
    if task_type != "module":
        return False
    module = request.get("module")
    if not isinstance(module, dict):
        return False
    declared = module.get("job_types")
    return isinstance(declared, list) and len(declared) > 0


EntryOrProblem = ModuleEntry | str


def job_type_entry(config: Config, backend: Backend, where: str, raw: Any) -> EntryOrProblem:
    if not isinstance(raw, dict):
        return f"{where}: must be an object with a `type`"
    stray = sorted(set(raw) - {"type", "narrator_engine"})
    if stray:
        return f"{where}: unknown key(s) {stray}"
    job_type = raw.get("type")
    engine = raw.get("narrator_engine")
    if not isinstance(job_type, str):
        return f"{where}: `type` must be a string"
    if engine is not None and not isinstance(engine, str):
        return f"{where}: `narrator_engine` must be a string"
    try:
        require_installable(job_type)
        require_narrator_engine(job_type, engine)
    except ApiError as exc:
        return f"{where}: {exc.message}"
    return ModuleEntry(
        name=install_label(job_type, engine), job_type=job_type, narrator_engine=engine
    )


def need_entry(config: Config, backend: Backend, where: str, raw: Any) -> EntryOrProblem:
    if not isinstance(raw, dict):
        return f"{where}: must be an object with a `class`"
    stray = sorted(set(raw) - {"class"})
    if stray:
        return (
            f"{where}: unknown key(s) {stray}. A need is a CLASS and nothing "
            "else; an app that wants one specific model names it under "
            "`subjects`, which is a choice and says so"
        )
    capability_class = raw.get("class")
    if not isinstance(capability_class, str):
        return f"{where}: `class` must be a string"
    if capability_class not in capability.BY_NAME:
        return (
            f"{where}: {capability_class!r} is not a capability class; they "
            f"are {sorted(capability.BY_NAME)}"
        )
    return ModuleEntry(name=f"resolve {capability_class}", capability_class=capability_class)


def subject_entry(config: Config, backend: Backend, where: str, raw: Any) -> EntryOrProblem:
    if not isinstance(raw, dict):
        return f"{where}: must be an object with `kind` and `id`"
    stray = sorted(set(raw) - {"kind", "id"})
    if stray:
        return f"{where}: unknown key(s) {stray}"
    kind, subject_id = raw.get("kind"), raw.get("id")
    if not isinstance(kind, str) or not isinstance(subject_id, str):
        return f"{where}: `kind` and `id` must both be strings"
    if kind not in catalog.KINDS:
        return f"{where}: {kind!r} is not a subject kind; they are {list(catalog.KINDS)}"
    if catalog.find(config, backend, kind, subject_id) is None:
        return f"{where}: this server has no {kind} called {subject_id!r} for {backend.kind}"
    return ModuleEntry(name=f"pull {kind} {subject_id}", kind=kind, subject_id=subject_id)


MODULE_TABLES: tuple[tuple[str, Callable[[Config, Backend, str, Any], EntryOrProblem]], ...] = (
    ("job_types", job_type_entry),
    ("needs", need_entry),
    ("subjects", subject_entry),
)


def module_header_problems(module: dict[str, Any]) -> list[str]:
    problems = [
        f"{key}: a module needs a non-empty string {key}"
        for key in ("name", "version")
        if not isinstance(module.get(key), str) or module[key].strip() == ""
    ]
    unknown = sorted(set(module) - MODULE_KEYS)
    if unknown:
        problems.append(
            f"unknown key(s) {unknown}; a module carries exactly name, version, "
            "job_types, needs and subjects"
        )
    return problems


def validate_module(config: Config, backend: Backend, module: Any) -> list[ModuleEntry]:
    if not isinstance(module, dict):
        raise ApiError(
            400,
            "invalid_module",
            f"a module is a JSON object, got {type(module).__name__}",
        )
    problems = module_header_problems(module)
    entries: list[ModuleEntry] = []
    for key, entry_of in MODULE_TABLES:
        rows = module.get(key, [])
        if not isinstance(rows, list):
            problems.append(f"{key}: must be a list")
            continue
        for index, raw in enumerate(rows):
            outcome = entry_of(config, backend, f"{key}[{index}]", raw)
            if isinstance(outcome, ModuleEntry):
                entries.append(outcome)
            else:
                problems.append(outcome)
    if not entries and not problems:
        problems.append(
            "a module with no job_types and no subjects asks for nothing; if that "
            "is what this app needs, it needs no module"
        )
    if problems:
        raise ApiError(
            400,
            "invalid_module",
            "this module was not run because "
            + (
                "it has a problem: " if len(problems) == 1 else
                f"it has {len(problems)} problems: "
            )
            + "; ".join(problems),
            {"problems": problems},
        )
    return entries


def module_document(entries: Iterable[ModuleEntry]) -> list[str]:
    return [entry.name for entry in entries]
