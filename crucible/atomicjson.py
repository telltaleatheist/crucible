from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

PRIVATE_MODE = 0o600


def write_json(path: Path, value: Any, *, indent: int | None = 2, private: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    staged = Path(staged_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as staged_file:
            json.dump(value, staged_file, indent=indent)
            staged_file.write("\n")
        if private:
            staged.chmod(PRIVATE_MODE)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return path
