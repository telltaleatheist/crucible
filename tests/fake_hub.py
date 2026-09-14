"""A HuggingFace hub that writes bytes from nowhere.

Every task test that pulls goes through this. It is not a stub in the "return a
canned value" sense: it drives the real `weights.pull`, writes real files into a
real temp `CRUCIBLE_HOME`, and — the part that matters — **drives the
`tqdm_class` Crucible hands it**, chunk by chunk, so the progress hook and
therefore the cancel path are exercised exactly as a 19 GB snapshot would
exercise them.

C: has about 5 GB free on the machine this was written on, and the suite runs on
hosted runners with no HF token, so nothing here may touch the network. What is
faked is the transport and nothing else: the stamp, the digest checks, the
directory layout and the cancel are all the shipping code's.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any

#: One "chunk" of a fake download. Small, because the point is the number of
#: hook calls rather than the number of bytes.
CHUNK = 1024


class FakeHub:
    """Stands in for `snapshot_download` and `hf_hub_download`.

    `started` is set once the first chunk is written, so a test can know the
    download is genuinely in flight before it cancels — rather than sleeping a
    tenth of a second and hoping, which is the kind of assertion that passes on
    a fast laptop and fails in CI.
    """

    def __init__(self, *, chunks: int = 4, delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.started = threading.Event()
        #: Every `(repo_id, revision)` this hub was asked for, in order.
        self.asked: list[tuple[str, str]] = []
        #: The `allow_patterns` of each snapshot call, in order. `None` where
        #: the caller asked for the whole repo.
        self.allowed: list[list[str] | None] = []
        #: Names this hub pretends the revision does not carry.
        self.absent: set[str] = set()
        #: Files `hf_hub_download` should produce, by their path in the repo.
        #: A test that pulls an archive or a named-file set fills this in.
        self.files: dict[str, bytes] = {}

    # ------------------------------------------------------------ transport

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
            # THE REAL HUB BEHAVES LIKE THIS, and it is the half that matters
            # for PHASE15 file-aware pulls: only the named files arrive, and a
            # name the revision does not carry simply does not appear. No
            # error, nothing. `self.absent` is how a test says "this repo does
            # not have that one", which is exactly the case the post-check in
            # `weights.pull` exists for.
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

    # ------------------------------------------------------------- internals

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
                    # The cancel lives in here. A `PullCancelled` raised by the
                    # hook travels out of this call, out of the fake, and into
                    # `weights.pull` — the same path a real download takes.
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
