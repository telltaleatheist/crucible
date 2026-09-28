from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from . import VERSION, catalog, lineup
from .capabilityclasses import BY_NAME, models_by_class
from .verdict import WSL_ONLY_JOB_TYPES
from .backend import LLAMA_WINDOWS
from .errors import ApiError, CrucibleError
from .manifests import BACKEND_ENGINES
from .tasks import require_installable, require_narrator_engine

DIR_NAME = "modules"

def file_name(app: str) -> str:
    return f"{app}.module.json"


class ModuleError(CrucibleError):
    ...


def check_class(capability_class: str) -> set[str]:
    if capability_class not in BY_NAME:
        raise ModuleError(
            f"{capability_class!r} is not a capability class; this build knows "
            f"{sorted(BY_NAME)}"
        )
    served = models_by_class()
    if capability_class not in served:
        raise ModuleError(
            f"the {capability_class!r} class does not select a model — it selects "
            f"{BY_NAME[capability_class].noun}, which are their own namespace. "
            f"Name what you want with a [[subjects]] entry instead; a generator "
            f"picking one of them would be choosing on the app's behalf"
        )
    return served[capability_class]


def resolve_class(capability_class: str, named: str | None) -> str:
    candidates = check_class(capability_class)
    if not candidates:
        raise ModuleError(
            f"nothing in this build's catalog serves the {capability_class!r} "
            "class, so a module cannot ask for it"
        )

    if named is not None:
        if named not in candidates:
            raise ModuleError(
                f"{named!r} does not serve the {capability_class!r} class; the "
                f"models that do are {sorted(candidates)}"
            )
        return named

    floor = lineup.floors(lineup.build()[0]).get(capability_class)
    if floor is not None:
        return floor
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ModuleError(
        f"the {capability_class!r} class is served by {sorted(candidates)} and "
        f"none of them declares itself the floor, so which one an app needs is "
        f"the app's choice and not this generator's. Say it in the declaration: "
        f'`model = "{sorted(candidates)[0]}"`'
    )


def declared_models() -> list[str]:
    return catalog.declared_ids()["model"]


def read_declaration(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise ModuleError(f"could not read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ModuleError(f"{path.name}: not valid TOML: {exc}") from exc
    unknown = sorted(set(document) - {"module", "job_types", "needs", "subjects"})
    if unknown:
        raise ModuleError(
            f"{path.name}: unknown top-level key(s) {unknown}; a declaration "
            "carries [module], [[job_types]], [[needs]] and [[subjects]]"
        )
    block = document.get("module")
    if not isinstance(block, dict) or not isinstance(block.get("name"), str):
        raise ModuleError(f"{path.name}: [module] needs a string `name`")
    if sorted(set(block) - {"name"}):
        raise ModuleError(
            f"{path.name}: [module] takes only `name`. The version is DERIVED "
            "from the content — see crucible/modules.py"
        )
    return document


def build(declaration: dict[str, Any], where: str) -> dict[str, Any]:
    name = declaration["module"]["name"]

    job_types: list[dict[str, Any]] = []
    for index, raw in enumerate(declaration.get("job_types", [])):
        at = f"{where}: job_types[{index}]"
        if not isinstance(raw, dict):
            raise ModuleError(f"{at}: must be a table with a `type`")
        unknown = sorted(set(raw) - {"type", "narrator_engine"})
        if unknown:
            raise ModuleError(f"{at}: unknown key(s) {unknown}")
        job_type = raw.get("type")
        engine = raw.get("narrator_engine")
        if not isinstance(job_type, str):
            raise ModuleError(f"{at}: `type` must be a string")
        if engine is not None and not isinstance(engine, str):
            raise ModuleError(f"{at}: `narrator_engine` must be a string")
        try:
            require_installable(job_type)
            require_narrator_engine(job_type, engine)
        except ApiError as exc:
            raise ModuleError(f"{at}: {exc.message}") from None
        entry: dict[str, Any] = {"type": job_type}
        if engine is not None:
            entry["narrator_engine"] = engine
        entry["backends"] = sorted(
            backend for backend in BACKEND_ENGINES
            if backend != LLAMA_WINDOWS or job_type not in WSL_ONLY_JOB_TYPES
        )
        job_types.append(entry)

    declared = catalog.declared_ids()
    subjects: list[dict[str, Any]] = []

    def add(kind: str, subject_id: str) -> None:
        entry: dict[str, Any] = {
            "kind": kind,
            "id": subject_id,
            "backends": catalog.backends_declaring(kind, subject_id),
        }
        if entry not in subjects:
            subjects.append(entry)

    needs: list[dict[str, str]] = []
    for index, raw in enumerate(declaration.get("needs", [])):
        at = f"{where}: needs[{index}]"
        if not isinstance(raw, dict):
            raise ModuleError(f"{at}: must be a table with a `class`")
        unknown = sorted(set(raw) - {"class", "model"})
        if unknown:
            raise ModuleError(f"{at}: unknown key(s) {unknown}")
        capability_class = raw.get("class")
        named = raw.get("model")
        if not isinstance(capability_class, str):
            raise ModuleError(f"{at}: `class` must be a string")
        if named is not None and not isinstance(named, str):
            raise ModuleError(f"{at}: `model` must be a string")
        try:
            check_class(capability_class)
        except ModuleError as exc:
            raise ModuleError(f"{at}: {exc}") from None
        if named is not None:
            if named not in declared_models():
                raise ModuleError(
                    f"{at}: this build has no model called {named!r}; it ships "
                    f"{declared_models()}"
                )
            add("model", named)
            continue
        entry = {"class": capability_class}
        if entry not in needs:
            needs.append(entry)

    for index, raw in enumerate(declaration.get("subjects", [])):
        at = f"{where}: subjects[{index}]"
        if not isinstance(raw, dict):
            raise ModuleError(f"{at}: must be a table with `kind` and `id`")
        unknown = sorted(set(raw) - {"kind", "id"})
        if unknown:
            raise ModuleError(f"{at}: unknown key(s) {unknown}")
        kind, subject_id = raw.get("kind"), raw.get("id")
        if not isinstance(kind, str) or not isinstance(subject_id, str):
            raise ModuleError(f"{at}: `kind` and `id` must both be strings")
        if kind not in declared:
            raise ModuleError(
                f"{at}: {kind!r} is not a subject kind; they are "
                f"{list(catalog.KINDS)}"
            )
        if subject_id not in declared[kind]:
            raise ModuleError(
                f"{at}: this build has no {kind} called {subject_id!r}; it ships "
                f"{declared[kind]}"
            )
        add(kind, subject_id)

    if not job_types and not subjects and not needs:
        raise ModuleError(
            f"{where}: this declaration asks for nothing. An app that needs "
            "nothing from a server needs no module"
        )
    body = {
        "name": name,
        "job_types": job_types,
        "needs": needs,
        "subjects": subjects,
    }
    return {"name": name, "version": version_of(body), **body}


def version_of(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{VERSION}+{digest}"


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def check(existing_text: str, fresh: dict[str, Any]) -> list[str]:
    try:
        existing = json.loads(existing_text)
    except json.JSONDecodeError as exc:
        return [f"the checked-in file is not valid JSON: {exc}"]
    if not isinstance(existing, dict):
        return ["the checked-in file is not a JSON object"]
    problems: list[str] = []
    for key in sorted(set(existing) | set(fresh)):
        if existing.get(key) != fresh.get(key):
            problems.append(
                f"{key}: checked in {json.dumps(existing.get(key))}, "
                f"generator says {json.dumps(fresh.get(key))}"
            )
    return problems


__all__ = [
    "DIR_NAME",
    "ModuleError",
    "build",
    "check",
    "check_class",
    "declared_models",
    "file_name",
    "read_declaration",
    "render",
    "resolve_class",
    "version_of",
]
