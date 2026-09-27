from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

UPLOAD_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class Blob:
    blob_id: str
    bytes: int
    sha256: str
    filename: str | None

    def receipt(self) -> dict[str, object]:
        return {"blob_id": self.blob_id, "bytes": self.bytes, "sha256": self.sha256}


def blob_path(uploads_dir: str | Path, blob_id: str) -> Path:
    return Path(uploads_dir) / blob_id


def sidecar_path(uploads_dir: str | Path, blob_id: str) -> Path:
    return Path(uploads_dir) / f"{blob_id}.json"


async def store_upload(
    uploads_dir: str | Path,
    read: Callable[[int], Awaitable[bytes]],
    filename: str | None,
) -> Blob:
    blob_id = uuid.uuid4().hex
    digest = hashlib.sha256()
    written = 0
    with blob_path(uploads_dir, blob_id).open("wb") as handle:
        while True:
            chunk = await read(UPLOAD_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
            written += len(chunk)
    blob = Blob(blob_id=blob_id, bytes=written, sha256=digest.hexdigest(), filename=filename)
    meta = {
        "blob_id": blob.blob_id,
        "bytes": blob.bytes,
        "sha256": blob.sha256,
        "filename": blob.filename,
    }
    sidecar_path(uploads_dir, blob_id).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return blob


def recorded_sha256(uploads_dir: str | Path, blob_id: str, size: int) -> str | None:
    try:
        meta = json.loads(sidecar_path(uploads_dir, blob_id).read_text(encoding="utf-8"))
        if int(meta.get("bytes", -1)) == size:
            return str(meta["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None
