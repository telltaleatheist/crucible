from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomicjson import write_json
from .clock import utcnow
from .errors import CrucibleError
from .platform.quarantine import quarantine
from .tomltable import REVISION_PATTERN
from .weights import STAMP_NAME

REFS_FILE = "voice-refs.json"

DEFAULT_REF = "crucible"

REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

CHECK_UPDATES_COMMAND = "crucible voices check-updates"


class VoiceRefError(CrucibleError):
    ...


@dataclass(frozen=True)
class RefCheck:
    hf_repo: str
    ref: str
    revision: str | None
    checked_at: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hf_repo": self.hf_repo,
            "ref": self.ref,
            "revision": self.revision,
            "checked_at": self.checked_at,
            "error": self.error,
        }


def refs_path(home: Path) -> Path:
    return home / REFS_FILE


def _key(hf_repo: str, ref: str) -> str:
    return f"{hf_repo}@{ref}"


def _check_of(entry: Any) -> RefCheck | None:
    if not isinstance(entry, dict):
        return None
    revision = entry.get("revision")
    if revision is not None and not (
        isinstance(revision, str) and REVISION_PATTERN.match(revision)
    ):
        return None
    try:
        return RefCheck(
            hf_repo=str(entry["hf_repo"]),
            ref=str(entry["ref"]),
            revision=revision,
            checked_at=str(entry["checked_at"]),
            error=None if entry.get("error") is None else str(entry["error"]),
        )
    except KeyError:
        return None


def read_checks(home: Path) -> dict[str, RefCheck]:
    path = refs_path(home)
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        quarantine(path)
        return {}
    if not isinstance(document, dict):
        quarantine(path)
        return {}
    found: dict[str, RefCheck] = {}
    for entry in document.values():
        check = _check_of(entry)
        if check is not None:
            found[_key(check.hf_repo, check.ref)] = check
    return found


def cached(home: Path, hf_repo: str, ref: str) -> RefCheck | None:
    return read_checks(home).get(_key(hf_repo, ref))


def record(home: Path, check: RefCheck) -> RefCheck:
    checks = read_checks(home)
    checks[_key(check.hf_repo, check.ref)] = check
    write_json(
        refs_path(home),
        {key: checks[key].to_dict() for key in sorted(checks)},
    )
    return check


def resolve_ref(home: Path, hf_repo: str, ref: str) -> str:
    from .config import config_path
    from .weights import hf_token_at

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise VoiceRefError(f"huggingface_hub is not importable: {exc}") from exc
    try:
        info = HfApi(token=hf_token_at(config_path(home))).model_info(
            hf_repo, revision=ref
        )
    except Exception as exc:
        raise VoiceRefError(
            f"could not resolve {hf_repo}@{ref} on HuggingFace "
            f"({type(exc).__name__}: {exc}). This machine stays on the revision "
            f"it has; run `{CHECK_UPDATES_COMMAND}` again once the Hub answers"
        ) from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not REVISION_PATTERN.match(sha):
        raise VoiceRefError(
            f"HuggingFace answered for {hf_repo}@{ref} without a commit sha "
            f"({sha!r}); run `{CHECK_UPDATES_COMMAND}` again"
        )
    return sha


def check(home: Path, hf_repo: str, ref: str) -> RefCheck:
    try:
        sha = resolve_ref(home, hf_repo, ref)
    except VoiceRefError as exc:
        earlier = cached(home, hf_repo, ref)
        return record(
            home,
            RefCheck(
                hf_repo=hf_repo,
                ref=ref,
                revision=None if earlier is None else earlier.revision,
                checked_at=utcnow(),
                error=str(exc),
            ),
        )
    return record(
        home,
        RefCheck(hf_repo=hf_repo, ref=ref, revision=sha, checked_at=utcnow(), error=None),
    )


def pulled_revision(home: Path, voice_id: str, hf_repo: str) -> str | None:
    root = home / "voices" / voice_id
    if not root.is_dir():
        return None
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stamp = arm_dir / STAMP_NAME
        if not stamp.is_file():
            continue
        try:
            stamped = json.loads(stamp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(stamped, dict) or stamped.get("hf_repo") != hf_repo:
            continue
        revision = stamped.get("revision")
        if isinstance(revision, str) and REVISION_PATTERN.match(revision):
            return revision
    return None


def served_revision(home: Path, voice_id: str, hf_repo: str, ref: str) -> str | None:
    pulled = pulled_revision(home, voice_id, hf_repo)
    if pulled is not None:
        return pulled
    latest = cached(home, hf_repo, ref)
    return None if latest is None else latest.revision


def unresolved_reason(voice_id: str, hf_repo: str, ref: str, home: Path) -> str:
    latest = cached(home, hf_repo, ref)
    why = (
        "this machine has never looked the tag up"
        if latest is None
        else f"the last look-up failed at {latest.checked_at}: {latest.error}"
    )
    return (
        f"voice_ref_unresolved: voice {voice_id!r} follows {hf_repo}@{ref}, and "
        f"nothing of it is pulled here and {why}. Run `{CHECK_UPDATES_COMMAND}` "
        f"(or `crucible voices pull {voice_id}`) on this machine"
    )


__all__ = [
    "CHECK_UPDATES_COMMAND",
    "DEFAULT_REF",
    "REFS_FILE",
    "REF_PATTERN",
    "RefCheck",
    "VoiceRefError",
    "cached",
    "check",
    "pulled_revision",
    "read_checks",
    "record",
    "refs_path",
    "resolve_ref",
    "served_revision",
    "unresolved_reason",
]
