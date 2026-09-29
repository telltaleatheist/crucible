from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable

from .theme import Palette

BASE_POINTS = {"darwin": 13, "win32": 10}

INDETERMINATE_MS = 40


class Style:
    def __init__(self, root: tk.Tk, palette: Palette) -> None:
        self.root = root
        self.palette = palette
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0) if sys.platform == "win32" else 1.0
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        base = BASE_POINTS.get(sys.platform, 10)
        mono = tkfont.nametofont("TkFixedFont").actual("family")
        self.fonts = {
            "title": (family, round(base * 2.0), "bold"),
            "brand": (family, round(base * 1.35), "bold"),
            "section": (family, round(base * 1.05), "bold"),
            "strong": (family, base, "bold"),
            "body": (family, base),
            "small": (family, max(8, base - 1)),
            "nav": (family, round(base * 1.05)),
            "mono": (mono, max(8, base - 1)),
        }

    def px(self, value: float) -> int:
        return round(value * self.scale)

    def font(self, name: str) -> tuple:
        return self.fonts[name]


def rounded_points(x1: float, y1: float, x2: float, y2: float, r: float) -> list[float]:
    return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]


def blend(first: str, second: str, amount: float) -> str:
    a = [int(first[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(second[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b))


def colours_for(style: Style, kind: str, enabled: bool) -> tuple[str, str]:
    p = style.palette
    if not enabled:
        return p.secondary, p.muted
    if kind == "primary":
        return p.primary, p.primary_text
    if kind == "danger":
        return p.secondary, p.bad
    return p.secondary, p.secondary_text


class PillButton(tk.Canvas):
    def __init__(self, parent: tk.Misc, style: Style, text: str, command: Callable[[], None] | None,
                 kind: str = "secondary", enabled: bool = True, bg: str | None = None) -> None:
        font = tkfont.Font(font=style.font("strong"))
        width = font.measure(text) + style.px(28)
        height = font.metrics("linespace") + style.px(14)
        super().__init__(parent, width=width, height=height, highlightthickness=0, bd=0,
                         bg=bg or style.palette.bg, cursor="hand2" if enabled else "arrow")
        fill, ink = colours_for(style, kind, enabled)
        self.shape = self.create_polygon(rounded_points(1, 1, width - 1, height - 1, height / 2),
                                         smooth=True, fill=fill, outline=fill)
        self.create_text(width / 2, height / 2, text=text, fill=ink, font=style.font("strong"))
        if enabled and command is not None:
            self.bind("<Button-1>", lambda _e: command())
            hover = blend(fill, ink, 0.15)
            self.bind("<Enter>", lambda _e: self.itemconfigure(self.shape, fill=hover, outline=hover))
            self.bind("<Leave>", lambda _e: self.itemconfigure(self.shape, fill=fill, outline=fill))


class ProgressBar(tk.Canvas):
    def __init__(self, parent: tk.Misc, style: Style, fraction: float | None, bg: str,
                 width: int = 240) -> None:
        height = style.px(6)
        super().__init__(parent, width=style.px(width), height=height, highlightthickness=0, bd=0, bg=bg)
        self.style = style
        self.fraction = fraction
        self.offset = 0.0
        self.job: str | None = None
        self.bind("<Configure>", lambda _e: self.draw())
        self.bind("<Destroy>", lambda _e: self.stop())
        self.draw()

    def stop(self) -> None:
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None

    def draw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), int(self["width"]))
        height = int(self["height"])
        p = self.style.palette
        self.create_polygon(rounded_points(0, 0, width, height, height / 2), smooth=True, fill=p.track)
        if self.fraction is not None:
            filled = max(height, width * self.fraction)
            self.create_polygon(rounded_points(0, 0, filled, height, height / 2), smooth=True, fill=p.ember)
            return
        start = (self.offset % 1.3 - 0.3) * width
        self.create_polygon(rounded_points(max(0, start), 0, min(width, start + width * 0.3), height,
                                           height / 2), smooth=True, fill=p.ember)
        self.offset += 0.02
        self.job = self.after(INDETERMINATE_MS, self.draw)


class Dot(tk.Canvas):
    def __init__(self, parent: tk.Misc, style: Style, colour: str, bg: str) -> None:
        size = style.px(10)
        super().__init__(parent, width=size, height=size, highlightthickness=0, bd=0, bg=bg)
        self.create_oval(1, 1, size - 1, size - 1, fill=colour, outline=colour)


class ScrollArea(tk.Frame):
    def __init__(self, parent: tk.Misc, style: Style) -> None:
        bg = style.palette.bg
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.pack(fill="both", expand=True)
        self.inner.bind("<Configure>", lambda _e: self._fit())
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.bind("<Enter>", lambda _e: self.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda _e: self.unbind_all("<MouseWheel>"))

    def _fit(self) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _wheel(self, event: tk.Event) -> None:
        if self.inner.winfo_height() <= self.canvas.winfo_height():
            return
        step = -int(event.delta / 120) if sys.platform == "win32" else -int(event.delta)
        self.canvas.yview_scroll(step or (-1 if event.delta > 0 else 1), "units")

    def top(self) -> float:
        return self.canvas.yview()[0]

    def restore(self, fraction: float) -> None:
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(fraction)
