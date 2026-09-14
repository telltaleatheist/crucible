"""The lineup Foundry vendors: every model's LOCAL form, read off the manifests.

Owen, 2026-09-13, via Foundry: Crucible's model manifests are the catalog of
record for Foundry's local lineup too — the Ollama / llama.cpp fallback it runs
when no Crucible is present — so that "what can this machine run" has ONE owner.
Foundry's app vendors the JSON this module writes and compares it by content.

What this module is, and is not
-------------------------------
It is a READER. Every fact in `foundry-lineup.json` is read through
`crucible/manifests.py` (the `[model]` and `[local]` tables) and
`crucible/capability.py` (`classes_for_model`, which walks the same `CLASSES`
table install selects on). It re-parses no TOML, derives no class of its own
and invents no number: a row that needs a fact the manifest does not state is a
refusal by name, not a default (ARCHITECTURE.md R1 — one fact, one owner).

It is not on the wire. A Crucible never runs Ollama, so `/v1/models` does not
carry `[local]`; the JSON file is that table's one door, and `scripts/
gen-foundry-lineup.py` is what opens it. `--check` is the guard that the file
in the repo still equals what the manifests say (R2: a guard that is red is a
broken guard, so CI runs it).

The shape, which Foundry's reader is built against
--------------------------------------------------
    {
      "generated_from": "<crucible git sha>",
      "schema": 2,
      "floors": {"translate": "qwen3.8-27b-4bit", "simplify": "qwen3.8-27b-4bit"},
      "models": [
        {
          "id": "<crucible model id>",
          "classes": ["clean"],
          "label": "...",
          "description": "...",
          "local": {"kind": "ollama", "tag": "...",
                    "downloadGB": 19.32,
                    "needsGB": {"value": 20.82, "basis": "declared"}}
                 | {"kind": "gguf", "hf_repo": "...", "revision": "<sha>",
                    "file": "...", "mmproj": "..." | null,
                    "downloadGB": ..., "needsGB": {...}},
          "minimum": false,
          "minimumFor": []
        }
      ]
    }

Rows are in id order — the order `load_all_manifests` documents and `/v1/models`
lists in. A model without a `[local]` table is OMITTED, not emitted with
`local: null`: a machine without Crucible cannot run it, and a row that says so
would be a tile a picker has to grey out for a reason it cannot state.

`generated_from` is the commit the generator RAN on, so it is always one commit
behind the file that carries it (the file lands in the next commit). `check`
therefore compares everything BUT that key: a check that included it would go
red on every commit, and a red guard is a broken guard.

`floors` names, per capability class, THE model that floors it — the same fact
`minimumFor` carries per row, gathered into one place a reader can ask. It was
added 2026-09-14 because the fact had grown a second owner without anyone
deciding it should. Foundry vendors this file AND keeps `model-lineup-local.json`
for models it adds on its own, and its reader took the SMALLEST declared floor
across both. So a local row declaring a 9B the translate floor silently overruled
this catalog on every machine that fits a 9B and not a 27B — against Owen's
ruling that translate and simplify need a 27B-class model, a Crucible serving
the class, or a cloud provider, and never a 9B locally (2026-09-14). "Smallest
wins" cannot tell a legitimate smaller floor from one that contradicts a ruling.

So the rule this key states: **the catalog of record owns the floor for every
class it floors.** A consumer that also carries its own additions reads `floors`
first, and a local row declaring a floor for a class named here is a
contradiction it should refuse by name rather than average in. A class absent
from `floors` has no floor from this catalog and a consumer may set its own —
`analysis` is deliberately such a class (Owen named translate and simplify, and
analysis carries no floor).
"""

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

#: Bumped when a row's shape changes in a way Foundry's reader must know about.
#: 2 (2026-09-14) added the top-level `floors` table.
SCHEMA = 2

#: Where the file lives: the repo root, beside `models/`.
FILE_NAME = "foundry-lineup.json"

#: Decimal gigabytes, because that is what Ollama prints (`ollama list` says
#: "19 GB" for 19_321_189_044 bytes) and what Foundry's picker draws. The
#: manifests keep bytes; the conversion happens here, once, and nowhere else.
GB = 1_000_000_000

#: The one top-level key `check` ignores. See the module docstring.
PROVENANCE_KEY = "generated_from"

_SHA = re.compile(r"^[0-9a-f]{40}$")


class LineupError(CrucibleError):
    """A manifest says something the lineup cannot honestly draw."""


def gigabytes(value: int) -> float:
    """Bytes to GB at two decimals — the precision a tile prints."""
    return round(value / GB, 2)


def local_row(local: LocalForm) -> dict[str, Any]:
    """The `local` object of one row, in Foundry's vocabulary."""
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
    """One row, or a refusal naming what the manifest got wrong.

    `classes` is passed in rather than looked up so the caller decides where the
    catalog is; `build` is that caller and asks `capability.classes_for_model`.
    """
    local = manifest.local
    if local is None:
        raise LineupError(
            f"{manifest.id} has no [local] table; a model without one is omitted "
            "from the lineup, not drawn"
        )
    if manifest.display is None or manifest.description is None:
        # The loader already refuses this pair; the check is here so a manifest
        # built in memory cannot reach a row with a null label either.
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
    """Every row, in id order, and the ids that were omitted for having no local.

    Reads THE catalog — `manifests_dir()`, which honours `$CRUCIBLE_MODELS_DIR` —
    and takes no directory of its own, because `classes_for_model` reads the same
    catalog through the same door and two arguments naming one directory would be
    two owners of where the manifests are.
    """
    rows: list[dict[str, Any]] = []
    omitted: list[str] = []
    for manifest in load_all_manifests().values():
        if manifest.local is None:
            omitted.append(manifest.id)
            continue
        rows.append(model_row(manifest, classes_for_model(manifest.id)))
    return rows, omitted


def floors(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Class to the model id that floors it, gathered from the rows themselves.

    Derived, never declared: a class appears here because some row's
    `minimumFor` names it, which is the manifest's `[local] minimum_for`. Two
    rows flooring one class is a refusal rather than a pick — "the smallest
    wins" is exactly the rule that let a second owner overrule this catalog
    (see the module docstring), and it has no place inside the owner either.
    """
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
    """The file's top level, in the order Foundry's reader was shown."""
    return {
        PROVENANCE_KEY: generated_from,
        "schema": SCHEMA,
        "floors": floors(rows),
        "models": rows,
    }


def render(doc: dict[str, Any]) -> str:
    """The bytes that go in the file: two-space indent, UTF-8, one trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def git_head(repo_root: Path) -> str:
    """The commit the generator is running on, or a refusal naming why not."""
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
    """Everything the check compares: the document less its provenance."""
    return {key: value for key, value in doc.items() if key != PROVENANCE_KEY}


def check(existing_text: str, fresh: dict[str, Any]) -> list[str]:
    """How the checked-in file differs from a fresh build, as sentences.

    Empty means they agree. Each sentence names a model id or a top-level key, so
    the operator regenerating the file knows which manifest moved.
    """
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
