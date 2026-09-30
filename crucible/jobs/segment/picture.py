"""The size of a PNG, JPEG or WebP read from its header, without an image library.

The server has no Pillow (only the segment env does), and a point outside the
picture must be refused before the model loads, not by a worker that would
exit on the error. So the job reads width and height from the file's own
header here; the worker opens the same file with Pillow and gets the same
numbers (EXIF orientation is not applied on either side).
"""

from __future__ import annotations

import struct
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

JPEG_SOI = b"\xff\xd8"

JPEG_FRAME_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)

JPEG_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})

HEADER_BYTES = 64 * 1024


def _png(head: bytes) -> tuple[int, int] | None:
    if len(head) < 24 or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def _webp(head: bytes) -> tuple[int, int] | None:
    chunk = head[12:16]
    if chunk == b"VP8 " and len(head) >= 30 and head[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", head[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and len(head) >= 25 and head[20] == 0x2F:
        bits = int.from_bytes(head[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(head) >= 30:
        width = int.from_bytes(head[24:27], "little") + 1
        height = int.from_bytes(head[27:30], "little") + 1
        return width, height
    return None


def _jpeg(data: bytes) -> tuple[int, int] | None:
    at = 2
    while at + 4 <= len(data):
        if data[at] != 0xFF:
            return None
        marker = data[at + 1]
        if marker == 0xFF:
            at += 1
            continue
        if marker in JPEG_STANDALONE_MARKERS:
            at += 2
            continue
        (length,) = struct.unpack(">H", data[at + 2 : at + 4])
        if marker in JPEG_FRAME_MARKERS:
            if at + 9 > len(data):
                return None
            height, width = struct.unpack(">HH", data[at + 5 : at + 9])
            return width, height
        if marker == 0xDA or length < 2:
            return None
        at += 2 + length
    return None


def picture_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from the header, or None when the header does not say."""
    with path.open("rb") as handle:
        head = handle.read(HEADER_BYTES)
    if head.startswith(PNG_SIGNATURE):
        found = _png(head)
    elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        found = _webp(head)
    elif head.startswith(JPEG_SOI):
        found = _jpeg(head)
        if found is None and len(head) == HEADER_BYTES:
            found = _jpeg(path.read_bytes())
    else:
        found = None
    if found is None or found[0] <= 0 or found[1] <= 0:
        return None
    return int(found[0]), int(found[1])


__all__ = ["picture_size"]
