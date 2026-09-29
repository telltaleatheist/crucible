from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "crucible" / "desktop_app" / "assets"
INSTALLER = ROOT / "installer" / "windows"
WELCOME_SIZE = (164, 314)

MASTER = 1024
EMBER = (214, 106, 48, 255)
WHITE = (255, 255, 255, 255)
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
ICNS_SIZES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
PNG_SIZES = (16, 32, 64, 256, 512)


def glyph(size: int) -> Image.Image:
    scale = size / MASTER
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    inset = round(40 * scale)
    draw.rounded_rectangle((inset, inset, size - inset, size - inset),
                           radius=round(220 * scale), fill=EMBER)
    def at(*points: float) -> list[float]:
        return [round(p * scale) for p in points]

    draw.rounded_rectangle(at(230, 300, 794, 390), radius=round(45 * scale), fill=WHITE)
    draw.polygon(at(280, 390, 744, 390, 660, 700, 364, 700), fill=WHITE)
    draw.rounded_rectangle(at(364, 600, 660, 780), radius=round(90 * scale), fill=WHITE)
    draw.rounded_rectangle(at(330, 430, 694, 470), radius=round(20 * scale), fill=EMBER)
    return image


def sized(size: int) -> Image.Image:
    return glyph(MASTER).resize((size, size), Image.LANCZOS) if size < 64 else glyph(size)


def write(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for size in PNG_SIZES:
        path = out / f"icon-{size}.png"
        sized(size).save(path)
        written.append(path)
    master = glyph(MASTER)
    master.save(out / "crucible.ico", sizes=ICO_SIZES)
    master.save(out / "crucible.icns", sizes=ICNS_SIZES)
    return written + [out / "crucible.ico", out / "crucible.icns"]


def welcome_bitmap(out: Path) -> Path:
    width, height = WELCOME_SIZE
    image = Image.new("RGB", WELCOME_SIZE, EMBER[:3])
    mark = glyph(MASTER).crop((150, 150, MASTER - 150, MASTER - 150)).resize((120, 120), Image.LANCZOS)
    plate = Image.new("RGBA", mark.size, EMBER)
    plate.alpha_composite(mark)
    image.paste(plate.convert("RGB"), ((width - 120) // 2, height // 3 - 60))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "welcome.bmp"
    image.save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw Crucible's app icon into crucible/desktop_app/assets")
    parser.add_argument("--out", type=Path, default=ASSETS)
    parser.add_argument("--installer-out", type=Path, default=INSTALLER)
    args = parser.parse_args(argv)
    for path in write(args.out) + [welcome_bitmap(args.installer_out)]:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
