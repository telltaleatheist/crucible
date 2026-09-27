from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any

CHUNK = 1024

VOICE_MANIFEST = "crucible-voice.toml"


class FakeHub:

    def __init__(self, *, chunks: int = 4, delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.started = threading.Event()
        self.asked: list[tuple[str, str]] = []
        self.allowed: list[list[str] | None] = []
        self.absent: set[str] = set()
        self.files: dict[str, bytes] = {}


    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        token: str | None = None,
        max_workers: int = 1,
        tqdm_class: Any = None,
        allow_patterns: Any = None,
        **_ignored: Any,
    ) -> str:
        self.asked.append((repo_id, revision))
        self.allowed.append(None if allow_patterns is None else list(allow_patterns))
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        if allow_patterns is not None:
            for name in allow_patterns:
                if name in self.absent:
                    continue
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                self._stream(path, tqdm_class)
            return str(target)
        (target / "config.json").write_text("{}\n", encoding="utf-8")
        self._stream(target / "model.safetensors", tqdm_class)
        return str(target)

    def hf_hub_download(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
        token: str | None = None,
        tqdm_class: Any = None,
        **_ignored: Any,
    ) -> str:
        if filename not in self.files and filename == VOICE_MANIFEST:
            from .conftest import _pinned_manifest_or_offline

            return _pinned_manifest_or_offline(
                repo_id, filename, revision=revision, local_dir=local_dir
            )
        self.asked.append((repo_id, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.files.get(filename)
        if payload is None:
            raise AssertionError(
                f"the fake hub was asked for {filename!r}, which no test declared"
            )
        target.write_bytes(payload)
        self._tick(tqdm_class, len(payload), filename)
        return str(target)


    def _stream(self, path: Path, tqdm_class: Any) -> None:
        total = self.chunks * CHUNK
        bar = (
            None
            if tqdm_class is None
            else tqdm_class(total=total, unit="B", desc=path.name, initial=0)
        )
        with path.open("wb") as handle:
            for _ in range(self.chunks):
                handle.write(b"x" * CHUNK)
                handle.flush()
                self.started.set()
                if bar is not None:
                    bar.update(CHUNK)
                if self.delay:
                    time.sleep(self.delay)
        if bar is not None:
            bar.close()

    def _tick(self, tqdm_class: Any, size: int, name: str) -> None:
        if tqdm_class is None:
            return
        bar = tqdm_class(total=size, unit="B", desc=name, initial=0)
        self.started.set()
        bar.update(size)
        bar.close()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = ["CHUNK", "FakeHub", "sha256"]
