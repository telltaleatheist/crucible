from __future__ import annotations

import re
from typing import Any, Mapping

from .errors import CrucibleError

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

VOICE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def check_table(
    where: str,
    table: dict[str, Any],
    required: Mapping[str, Any],
    optional: Mapping[str, Any] | None = None,
    *,
    error: type[CrucibleError],
) -> None:
    optional = optional or {}
    allowed = set(required) | set(optional)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise error(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(allowed)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise error(f"{where}: missing required key(s) {missing}")
    for key, kind in {**required, **optional}.items():
        if key not in table:
            continue
        value = table[key]
        kinds = kind if isinstance(kind, tuple) else (kind,)
        wrong = not isinstance(value, kinds)
        if int in kinds and isinstance(value, bool):
            wrong = True
        if wrong:
            named = " or ".join(k.__name__ for k in kinds)
            raise error(
                f"{where}: {key} must be {named}, got "
                f"{type(value).__name__}"
            )
