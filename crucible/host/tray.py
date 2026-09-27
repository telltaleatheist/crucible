from __future__ import annotations

from typing import Any

ICON_SIZE = 64
ICON_FOREGROUND = (214, 106, 48, 255)


def icon_image(colour: tuple[int, int, int, int]) -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, ICON_SIZE - 6, ICON_SIZE - 6), outline=colour, width=8)
    draw.rectangle((ICON_SIZE // 2 - 4, 2, ICON_SIZE // 2 + 4, 16), fill=colour)
    return image
