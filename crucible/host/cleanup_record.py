from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..atomicjson import write_json
from ..platform.errors import HostError
from ..platform.quarantine import quarantine

CLEANUP_RECORD = "migration-cleanup.json"

CLEANUP_RECORD_INVALID = "migration_cleanup_record_invalid"


def cleanup_subjects(home: Path) -> set[tuple[str, str]]:
    record = home / CLEANUP_RECORD
    try:
        value = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostError(CLEANUP_RECORD_INVALID, f"{record} is not a JSON document ({exc})") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("subjects"), list):
        raise HostError(CLEANUP_RECORD_INVALID, f"{record} is not a migration cleanup record this build knows")
    result = set()
    for row in value["subjects"]:
        if (not isinstance(row, list) or len(row) != 2
                or not all(isinstance(part, str) and part for part in row) or row[0] == "engine"):
            raise HostError(CLEANUP_RECORD_INVALID, f"{record} names a model subject that is not one: {row!r}")
        result.add(tuple(row))
    return result


def quarantine_bad_cleanup_record(home: Path, log: Callable[[str], None]) -> Path | None:
    try:
        cleanup_subjects(home)
    except HostError as exc:
        if exc.code != CLEANUP_RECORD_INVALID:
            raise
        aside = quarantine(home / CLEANUP_RECORD)
        log(
            f"model cleanup: {exc.message}. It was moved to {aside} and the cleanup "
            "it described is dropped: the Windows copies of the models that moved "
            f"into the guest stay under {home} and cost disk only. `crucible "
            "uninstall --purge-weights` on this Windows side removes them with "
            "everything else; nothing needs doing to keep using Crucible."
        )
        return aside
    return None


def record_cleanup(home: Path, subjects: set[tuple[str, str]]) -> None:
    write_json(home / CLEANUP_RECORD, {"schema_version": 1, "subjects": [list(row) for row in sorted(subjects)]})
