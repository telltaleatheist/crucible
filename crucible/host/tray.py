from __future__ import annotations

from typing import Any, Callable

from .menu import MenuModel

TOOLTIP_PREFIX = "Crucible"

ICON_SIZE = 64
ICON_FOREGROUND = (214, 106, 48, 255)
ICON_RUNNING = (90, 166, 106, 255)
ICON_STOPPED = (150, 150, 150, 255)


def icon_image(colour: tuple[int, int, int, int]) -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, ICON_SIZE - 6, ICON_SIZE - 6), outline=colour, width=8)
    draw.rectangle((ICON_SIZE // 2 - 4, 2, ICON_SIZE // 2 + 4, 16), fill=colour)
    return image


def build_menu(model: MenuModel, on_click: Callable[[str], None]) -> Any:
    import pystray

    items: list[Any] = [pystray.MenuItem(model.title, None, enabled=False)]
    items.append(pystray.Menu.SEPARATOR)
    for entry in model.items:
        items.append(
            pystray.MenuItem(
                entry.label,
                _handler(entry.item_id, on_click),
                enabled=entry.enabled,
            )
        )
    return pystray.Menu(*items)


def _handler(item_id: str, on_click: Callable[[str], None]) -> Callable[..., None]:
    def clicked(_icon: Any = None, _item: Any = None) -> None:
        on_click(item_id)

    return clicked


def make_icon(model: MenuModel, on_click: Callable[[str], None]) -> Any:
    import pystray

    return pystray.Icon(
        "crucible",
        icon_image(ICON_FOREGROUND),
        f"{TOOLTIP_PREFIX} — {model.title}",
        build_menu(model, on_click),
    )


def update(icon: Any, model: MenuModel, on_click: Callable[[str], None]) -> None:
    icon.title = f"{TOOLTIP_PREFIX} — {model.title}"
    icon.menu = build_menu(model, on_click)
    icon.update_menu()
