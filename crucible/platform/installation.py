from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from .. import VERSION
from ..atomicjson import write_json
from .errors import LocalError

RECORD = "installation.json"

RELEASE_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _release_numbers(value: str) -> tuple[int, ...]:
    match = RELEASE_PATTERN.match(value.strip())
    if match is None:
        raise LocalError(
            f"release_unreadable: {value!r} is not a Crucible release, so it "
            "cannot be compared with one"
        )
    return tuple(int(part) for part in match.groups())


def release_order(left: str, right: str) -> int:
    first, second = _release_numbers(left), _release_numbers(right)
    if first == second:
        return 0
    return -1 if first < second else 1


def _console_interpreter() -> Path:
    executable = Path(sys.executable).absolute()
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    if not executable.is_file():
        raise LocalError(f"local_runtime_missing: {executable}")
    return executable


def publish_installation(home: Path | None = None) -> Path:
    if home is None:
        from ..config import crucible_home

        home = crucible_home()
    home = home.resolve()
    record = {
        "schema_version": 1, "platform": sys.platform, "release": VERSION,
        "home": str(home),
        "control": {"command": str(_console_interpreter()), "args": ["-m", "crucible.cli", "local"],
                    "cwd": str(PACKAGE_ROOT)},
    }
    return write_json(home / RECORD, record, private=True)


def installed_control(home: Path) -> dict[str, Any]:
    path = Path(home) / RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        control = record["control"]
        command, cwd = control["command"], control["cwd"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LocalError(
            f"installation_record_unreadable: {path} could not be read ({exc}). "
            "Run `crucible local register` to write it again, then run this again"
        ) from exc
    return {"command": command, "cwd": cwd}
