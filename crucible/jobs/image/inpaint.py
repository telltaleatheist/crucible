"""The mask, shared by both image engines: the region, its feather, the latent grid, the
per-step blend and the paste-back.

The worker loads this beside itself (`workerio.load_sibling`), in an image env that has numpy
and Pillow. The controller imports it too, but only for `picture_size`, which reads a header in
plain Python; numpy and Pillow are imported inside the functions that need them.

Conventions: a mask is a picture the size of the input image, read as luminance; white (128 and
up after scaling to the job's size) is the region to regenerate, black is kept. The region is
stretched to width x height exactly as the input is.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

LATENT_TILE = 16

DEFAULT_MASK_BLUR = 8

MAX_MASK_BLUR = 256

REGION_THRESHOLD = 128


class MaskRefused(ValueError):
    """A mask the job cannot use, with the error code the caller is refused by."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def picture_size(path: Any) -> "tuple[int, int] | None":
    """(width, height) from a PNG, JPEG or WebP header, or None if the header is not one of them.

    Plain Python, so the controller can compare an input and its mask before it loads a model.
    A JPEG's EXIF orientation is not applied here; the worker, which applies it, checks again.
    """
    with open(path, "rb") as handle:
        head = handle.read(64)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return struct.unpack(">II", head[16:24])
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return _webp_size(head)
        if head[:3] == b"\xff\xd8\xff":
            handle.seek(2)
            return _jpeg_size(handle)
    return None


def _webp_size(head: bytes) -> "tuple[int, int] | None":
    chunk = head[12:16]
    if chunk == b"VP8 " and head[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", head[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and head[20] == 0x2F:
        bits = int.from_bytes(head[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1
    return None


_JPEG_FRAMES = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg_size(handle) -> "tuple[int, int] | None":
    while True:
        byte = handle.read(1)
        while byte and byte != b"\xff":
            byte = handle.read(1)
        while byte == b"\xff":
            byte = handle.read(1)
        if not byte:
            return None
        marker = byte[0]
        if marker in (0x01, *range(0xD0, 0xDA)):
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) < 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        if marker in _JPEG_FRAMES:
            frame = handle.read(5)
            if len(frame) < 5:
                return None
            height, width = struct.unpack(">HH", frame[1:5])
            return width, height
        handle.seek(length - 2, 1)


def open_picture(path: Any):
    """A picture as the model sees it: EXIF orientation applied (as mflux's loader does)."""
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).copy()


def region_of(picture, width: int, height: int):
    """The region to regenerate at the job's size: float32 (height, width), 1 inside, 0 outside."""
    import numpy as np
    from PIL import Image

    gray = picture.convert("L")
    if gray.size != (width, height):
        gray = gray.resize((width, height), Image.BILINEAR)
    return (np.asarray(gray) >= REGION_THRESHOLD).astype(np.float32)


def feathered(region, blur: int):
    """The paste-back weight: 1 deep inside the region, falling to 0 over `blur` pixels inside its
    edge, and exactly 0 everywhere outside it, so a kept pixel is never touched."""
    import numpy as np
    from PIL import Image, ImageFilter

    if blur <= 0:
        return region.copy()
    picture = Image.fromarray((region * 255).astype(np.uint8), mode="L")
    blurred = np.asarray(picture.filter(ImageFilter.GaussianBlur(blur / 2)), dtype=np.float32) / 255
    return np.clip(2.0 * blurred - 1.0, 0.0, 1.0) * region


def latent_grid(region, tile: int = LATENT_TILE):
    """The region on the latent grid: a latent is regenerated when any pixel of its tile is.

    Max, not nearest: a one-pixel brush stroke still reaches the latent it falls in.
    """
    height, width = region.shape
    if height % tile or width % tile:
        raise ValueError(f"a {width}x{height} region does not tile by {tile}")
    return region.reshape(height // tile, tile, width // tile, tile).max(axis=(1, 3))


def packed(grid):
    """(1, h*w, 1): both engines pack 2.1 latents as a row-major spatial flatten, (1, h*w, C)."""
    return grid.reshape(1, -1, 1)


def blend_step(latents, clean, noise, mask, sigma: float):
    """RePaint-lite: keep the model's latents inside the mask and put the input back outside it,
    noised to the sigma the latents are now at (diffusers' inpaint pipelines, scale_noise)."""
    return mask * latents + (1.0 - mask) * ((1.0 - sigma) * clean + sigma * noise)


def paste_back(generated, original, feather):
    """The generated picture inside the mask, the input's own pixels outside it."""
    import numpy as np
    from PIL import Image

    width, height = generated.size
    source = original.convert("RGB")
    if source.size != (width, height):
        source = source.resize((width, height), Image.LANCZOS)
    made = np.asarray(generated.convert("RGB"), dtype=np.float32)
    kept = np.asarray(source, dtype=np.float32)
    weight = feather[:, :, None]
    mixed = weight * made + (1.0 - weight) * kept
    return Image.fromarray(np.clip(np.rint(mixed), 0, 255).astype(np.uint8), mode="RGB")


def outside_drift(generated, original, region) -> float:
    """How far the generated picture strayed from the input where the mask keeps it: the mean
    absolute difference, 0 to 255, over the kept pixels, before the paste-back hides it.

    The last denoising step puts the input's own latents back outside the mask, so a blend
    that held leaves only the VAE's round trip here (a few units). A picture made without
    regard to the input differs by tens.
    """
    import numpy as np
    from PIL import Image

    width, height = generated.size
    source = original.convert("RGB")
    if source.size != (width, height):
        source = source.resize((width, height), Image.LANCZOS)
    kept = region == 0
    if not kept.any():
        return 0.0
    made = np.asarray(generated.convert("RGB"), dtype=np.float32)
    difference = np.abs(made - np.asarray(source, dtype=np.float32))
    return float(difference[kept].mean())


@dataclass(frozen=True)
class Mask:
    region: Any
    feather: Any
    latent: Any
    original: Any
    grid: Any = None

    @property
    def coverage(self) -> float:
        return float(self.region.mean())


def load_mask(mask_path: str, image_path: str, width: int, height: int, blur: int) -> Mask:
    original = open_picture(image_path)
    drawn = open_picture(mask_path)
    if drawn.size != original.size:
        raise MaskRefused(
            "mask_size_mismatch",
            f"the mask is {drawn.size[0]}x{drawn.size[1]} and the image "
            f"{original.size[0]}x{original.size[1]}; draw the mask on the image's own canvas",
        )
    region = region_of(drawn, width, height)
    if not region.any():
        raise MaskRefused(
            "mask_empty",
            "the mask selects nothing (no pixel is 128 or brighter); white marks what to "
            "regenerate, black what to keep",
        )
    grid = latent_grid(region)
    return Mask(
        region=region,
        feather=feathered(region, blur),
        latent=packed(grid),
        original=original,
        grid=grid,
    )
