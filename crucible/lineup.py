from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .capability import classes_for_model
from .errors import CrucibleError
from .manifests import (
    GgufLocal,
    LocalForm,
    ModelManifest,
    OllamaLocal,
    load_all_manifests,
)

SCHEMA = 2

FILE_NAME = "foundry-lineup.json"

GB = 1_000_000_000

PROVENANCE_KEY = "generated_from"

_SHA = re.compile(r"^[0-9a-f]{40}$")


class LineupError(CrucibleError):
    ...


def gigabytes(value: int) -> float:
    return round(value / GB, 2)


def local_row(local: LocalForm) -> dict[str, Any]:
    size = {
        "downloadGB": gigabytes(local.download_bytes),
        "needsGB": {
            "value": gigabytes(local.needs_bytes),
            "basis": local.needs_basis,
        },
    }
    if isinstance(local, OllamaLocal):
        return {"kind": "ollama", "tag": local.tag, **size}
    if isinstance(local, GgufLocal):
        return {
            "kind": "gguf",
            "hf_repo": local.hf_repo,
            "revision": local.revision,
            "file": local.file,
            "mmproj": local.mmproj,
            **size,
        }
    raise TypeError(f"{type(local).__name__} is not a local form this module draws")


def model_row(manifest: ModelManifest, classes: tuple[str, ...]) -> dict[str, Any]:
    local = manifest.local
    if local is None:
        raise LineupError(
            f"{manifest.id} has no [local] table; a model without one is omitted "
            "from the lineup, not drawn"
        )
    if manifest.display is None or manifest.description is None:
        raise LineupError(
            f"{manifest.id}: a [local] table needs [model] display and description"
        )
    if not classes:
        raise LineupError(
            f"{manifest.id} carries a [local] table but no capability class in "
            f"crucible/capability.py names its family {manifest.family!r}. A "
            "lineup row lights a tile, and this one would light none; either the "
            "class table is missing a class or this model has no business in "
            "the lineup"
        )
    stray = [name for name in local.minimum_for if name not in classes]
    if stray:
        raise LineupError(
            f"{manifest.id}: [local] minimum_for names {stray}, which this model "
            f"does not serve — its classes are {list(classes)}. A model can only "
            "be the floor for a class it can run"
        )
    return {
        "id": manifest.id,
        "classes": list(classes),
        "label": manifest.display,
        "description": manifest.description,
        "local": local_row(local),
        "minimum": any(name in classes for name in local.minimum_for),
        "minimumFor": list(local.minimum_for),
    }


def build() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    omitted: list[str] = []
    for manifest in load_all_manifests().values():
        if manifest.local is None:
            omitted.append(manifest.id)
            continue
        rows.append(model_row(manifest, classes_for_model(manifest.id)))
    return rows, omitted


def floors(rows: list[dict[str, Any]]) -> dict[str, str]:
    found: dict[str, str] = {}
    for row in rows:
        for name in row["minimumFor"]:
            if name in found:
                raise LineupError(
                    f"two models floor the {name!r} class: {found[name]!r} and "
                    f"{row['id']!r}. A class has one floor; remove `minimum_for = "
                    f"[\"{name}\"]` from one of their `[local]` tables."
                )
            found[name] = row["id"]
    return dict(sorted(found.items()))


def document(rows: list[dict[str, Any]], generated_from: str) -> dict[str, Any]:
    return {
        PROVENANCE_KEY: generated_from,
        "schema": SCHEMA,
        "floors": floors(rows),
        "models": rows,
    }


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def git_head(repo_root: Path) -> str:
    try:
        ran = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LineupError(f"could not run git to read HEAD: {exc}") from exc
    if ran.returncode != 0:
        raise LineupError(
            f"git rev-parse HEAD failed in {repo_root}: {ran.stderr.strip()}"
        )
    sha = ran.stdout.strip()
    if not _SHA.match(sha):
        raise LineupError(f"git rev-parse HEAD printed {sha!r}, not a commit sha")
    return sha


def content(doc: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != PROVENANCE_KEY}


def check(existing_text: str, fresh: dict[str, Any]) -> list[str]:
    try:
        existing = json.loads(existing_text)
    except json.JSONDecodeError as exc:
        return [f"the checked-in file is not valid JSON: {exc}"]
    if not isinstance(existing, dict):
        return ["the checked-in file is not a JSON object"]

    problems: list[str] = []
    have = content(existing)
    want = content(fresh)
    if have.get("schema") != want["schema"]:
        problems.append(
            f"schema: checked in {have.get('schema')!r}, generator says {want['schema']!r}"
        )
    unknown = sorted(set(have) - set(want))
    if unknown:
        problems.append(f"top-level key(s) the generator does not write: {unknown}")

    old_rows = have.get("models")
    if not isinstance(old_rows, list):
        return problems + ["models: the checked-in file has no models list"]
    old_by_id = {row.get("id"): row for row in old_rows if isinstance(row, dict)}
    new_by_id = {row["id"]: row for row in want["models"]}
    for model_id in sorted(set(old_by_id) - set(new_by_id)):
        problems.append(f"{model_id}: in the checked-in file, not in the manifests")
    for model_id in sorted(set(new_by_id) - set(old_by_id)):
        problems.append(f"{model_id}: in the manifests, not in the checked-in file")
    for model_id in sorted(set(old_by_id) & set(new_by_id)):
        old_row, new_row = old_by_id[model_id], new_by_id[model_id]
        if old_row == new_row:
            continue
        changed = sorted(
            key
            for key in set(old_row) | set(new_row)
            if old_row.get(key) != new_row.get(key)
        )
        problems.append(f"{model_id}: differs in {changed}")
    old_order = [row.get("id") for row in old_rows]
    new_order = [row["id"] for row in want["models"]]
    if set(old_by_id) == set(new_by_id) and old_order != new_order:
        problems.append(
            f"models: the same rows in a different order; rows are in id order, "
            f"{new_order}"
        )
    return problems


__all__ = [
    "FILE_NAME",
    "GB",
    "LineupError",
    "PROVENANCE_KEY",
    "SCHEMA",
    "build",
    "check",
    "content",
    "document",
    "gigabytes",
    "git_head",
    "local_row",
    "model_row",
    "render",
]
