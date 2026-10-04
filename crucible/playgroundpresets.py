"""Named playground presets: a form's parameters saved under a name, per model, on the server.

Owen, 2026-10-03: "a save preset button". They live in the server's home
(`playground-presets.json`), so they survive a browser's storage being cleared and are
the same from every browser that opens this server's playground. One server's presets are
not another's, like every other piece of server state.

The file is `{"<model id>": {"<preset name>": {"params": {...}, "saved_at": "<iso>"}}}`,
written whole to a temporary file and moved into place, under one lock.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .clock import utcnow
from .errors import ApiError

FILE_NAME = "playground-presets.json"

NAME = re.compile(r"[^\x00-\x1f]{1,80}")
MODEL = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

# What a preset may hold: the form's own params, as JSON scalars. Seeds are not kept - a
# preset is a sound, not one take of it.
PARAMS = frozenset({
    "prompt", "tags", "lyrics", "negative_prompt", "duration_s", "steps", "cfg",
    "instrumental", "format", "width", "height", "num_frames", "fps",
})
MAX_BYTES = 64 * 1024


def _refused(code: str, message: str, **details: Any) -> ApiError:
    return ApiError(400, code, message, details or None)


class Presets:
    def __init__(self, home: Path) -> None:
        self.path = Path(home) / FILE_NAME
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ApiError(500, "presets_unreadable", f"{self.path} is not a JSON object")
        return document

    def _write(self, document: dict[str, dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(".json.writing")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def of(self, model: str) -> list[dict[str, Any]]:
        self._check_model(model)
        with self._lock:
            saved = self._read().get(model, {})
        return [
            {"name": name, "params": entry["params"], "saved_at": entry["saved_at"]}
            for name, entry in sorted(saved.items(), key=lambda item: item[0].casefold())
        ]

    def save(self, model: str, name: str, params: Any) -> dict[str, Any]:
        self._check_model(model)
        if not isinstance(name, str) or not NAME.fullmatch(name.strip()):
            raise _refused("preset_name_invalid", "a preset name is 1-80 characters with no control characters")
        name = name.strip()
        if not isinstance(params, dict) or not params:
            raise _refused("preset_params_invalid", "params must be a non-empty object of the form's fields")
        unknown = sorted(set(params) - PARAMS)
        if unknown:
            raise _refused(
                "preset_params_invalid",
                f"a preset keeps only the form's own params; {unknown} are not among {sorted(PARAMS)}"
                + (" (a seed is one take, not part of a sound)" if "seed" in unknown else ""),
                unknown=unknown,
            )
        if not all(isinstance(v, (str, int, float, bool)) for v in params.values()):
            raise _refused("preset_params_invalid", "each param must be a string, number or true/false")
        if len(json.dumps(params)) > MAX_BYTES:
            raise _refused("preset_too_large", f"a preset is at most {MAX_BYTES // 1024} kB")
        entry = {"params": dict(params), "saved_at": utcnow()}
        with self._lock:
            document = self._read()
            document.setdefault(model, {})[name] = entry
            self._write(document)
        return {"name": name, **entry}

    def delete(self, model: str, name: str) -> None:
        self._check_model(model)
        with self._lock:
            document = self._read()
            if name not in document.get(model, {}):
                raise ApiError(404, "preset_not_found", f"{model} has no preset {name!r}")
            del document[model][name]
            if not document[model]:
                del document[model]
            self._write(document)

    @staticmethod
    def _check_model(model: str) -> None:
        if not MODEL.fullmatch(model):
            raise _refused("model_id_invalid", f"{model!r} is not a model id")


__all__ = ["FILE_NAME", "Presets"]
