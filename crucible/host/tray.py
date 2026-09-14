"""The notification-area icon. PHASE15-HOST.md 4.2.

**This is the only module in the package that cannot be tested**, and it is
deliberately the smallest: it turns a `MenuModel` (which is a pure function and
IS tested, exhaustively) into pystray objects, and it does nothing else. No
decision is made here — not which items to show, not what the title says, not
whether an item is enabled. If you find yourself writing an `if` in this file,
it belongs in `menu.py`.

`pystray` and `pillow` are imported INSIDE the functions. `crucible/host/` is
imported by `crucible/cli.py` and by pytest in WSL, where neither package
exists, and a test suite that cannot import its subject pins nothing.

THE ICON IS DRAWN, NOT SHIPPED
-------------------------------
A `.ico` would be a binary in the wheel, a `package-data` line, and a file
that has to survive the pack's relocation. Sixteen lines of Pillow draw a
legible 64x64 mark that scales to whatever DPI the tray asks for, and the
whole thing is data this file can state.
"""

from __future__ import annotations

from typing import Any, Callable

from .menu import MenuModel

#: What Windows shows on hover, before the menu is opened.
TOOLTIP_PREFIX = "Crucible"

#: The mark: a filled ring on a transparent square. Crucible's shape is a
#: crucible, and at 16 device pixels anything with detail in it is a smudge.
ICON_SIZE = 64
ICON_FOREGROUND = (214, 106, 48, 255)
ICON_RUNNING = (90, 166, 106, 255)
ICON_STOPPED = (150, 150, 150, 255)


def icon_image(colour: tuple[int, int, int, int]) -> Any:
    """A 64x64 RGBA ring. Imported here so the module loads without Pillow."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, ICON_SIZE - 6, ICON_SIZE - 6), outline=colour, width=8)
    draw.rectangle((ICON_SIZE // 2 - 4, 2, ICON_SIZE // 2 + 4, 16), fill=colour)
    return image


def build_menu(model: MenuModel, on_click: Callable[[str], None]) -> Any:
    """The pystray menu for this model. One item per `MenuItem`, in order.

    The title is item zero and disabled, which is how pystray spells a header;
    `menu.title_for` is what decided the words.
    """
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
    """Bind one id. A closure per item, because a lambda in a loop would
    capture the loop variable and every item would fire the last one."""

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
    """Redraw for a new state. Called from the watch thread, every 15 s."""
    icon.title = f"{TOOLTIP_PREFIX} — {model.title}"
    icon.menu = build_menu(model, on_click)
    icon.update_menu()
