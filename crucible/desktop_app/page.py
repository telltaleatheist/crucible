from __future__ import annotations

import tkinter as tk
from typing import Callable

from .screens import Progress
from .theme import tone_colour
from .widgets import Dot, PillButton, ProgressBar, Style

WRAP = 560


class Page:
    def __init__(self, body: tk.Frame, style: Style) -> None:
        self.body = body
        self.style = style
        self.p = style.palette

    def label(self, parent: tk.Misc, text: str, font: str = "body", colour: str | None = None,
              bg: str | None = None, wrap: int | None = None, **pack) -> tk.Label:
        widget = tk.Label(parent, text=text, font=self.style.font(font), fg=colour or self.p.text,
                          bg=bg or self.p.bg, anchor="w", justify="left",
                          wraplength=self.style.px(wrap) if wrap else 0)
        widget.pack(**({"anchor": "w"} | pack))
        return widget

    def frame(self, parent: tk.Misc, bg: str | None = None, **pack) -> tk.Frame:
        widget = tk.Frame(parent, bg=bg or self.p.bg)
        widget.pack(**({"fill": "x"} | pack))
        return widget

    def header(self, title: str, subtitle: str = "", tone: str | None = None) -> None:
        top = self.frame(self.body, pady=(self.style.px(28), self.style.px(4)))
        if tone is not None:
            Dot(top, self.style, tone_colour(self.p, tone), self.p.bg).pack(
                side="left", padx=(0, self.style.px(10)))
        self.label(top, title, "title", side="left")
        if subtitle:
            self.label(self.body, subtitle, "body", self.p.muted, wrap=WRAP,
                       pady=(0, self.style.px(8)))

    def section(self, title: str) -> tk.Frame:
        self.label(self.body, title, "section", self.p.muted,
                   pady=(self.style.px(22), self.style.px(6)))
        return self.frame(self.body)

    def separator(self, parent: tk.Misc) -> None:
        tk.Frame(parent, bg=self.p.line, height=1).pack(fill="x")

    def row(self, parent: tk.Misc, title: str, subtitle: str = "") -> tuple[tk.Frame, tk.Frame, tk.Frame]:
        self.separator(parent)
        outer = self.frame(parent, pady=self.style.px(10))
        left = tk.Frame(outer, bg=self.p.bg)
        left.pack(side="left", fill="x", expand=True)
        right = tk.Frame(outer, bg=self.p.bg)
        right.pack(side="right")
        self.label(left, title, "strong")
        if subtitle:
            self.label(left, subtitle, "small", self.p.muted, wrap=380)
        below = self.frame(parent)
        return left, right, below

    def button(self, parent: tk.Misc, text: str, command: Callable[[], None] | None,
               kind: str = "secondary", enabled: bool = True, side: str = "right") -> PillButton:
        widget = PillButton(parent, self.style, text, command, kind=kind, enabled=enabled,
                            bg=parent.cget("bg"))
        widget.pack(side=side, padx=(self.style.px(8), 0))
        return widget

    def tag(self, parent: tk.Misc, text: str, tone: str, side: str = "right") -> None:
        if text:
            self.label(parent, text, "small", tone_colour(self.p, tone), side=side,
                       padx=(0, 0) if side == "top" else (self.style.px(10), 0), wrap=380)

    def notice(self, parent: tk.Misc, text: str) -> None:
        box = tk.Frame(parent, bg=self.p.notice_bg)
        box.pack(fill="x", pady=(self.style.px(4), self.style.px(10)))
        self.label(box, text, "small", self.p.bad, bg=self.p.notice_bg, wrap=WRAP,
                   padx=self.style.px(12), pady=self.style.px(8))

    def progress(self, parent: tk.Misc, work: Progress,
                 on_cancel: Callable[[Progress], None] | None = None) -> None:
        box = self.frame(parent, pady=(0, self.style.px(10)))
        line = self.frame(box)
        self.label(line, work.title, "body", side="left")
        if work.cancel and on_cancel is not None:
            self.button(line, "Cancel", lambda: on_cancel(work))
        percent = f"{work.fraction * 100:.0f}%  " if work.fraction is not None else ""
        self.label(box, percent + work.detail, "small", self.p.muted, wrap=WRAP)
        ProgressBar(box, self.style, work.fraction, self.p.bg).pack(
            fill="x", pady=(self.style.px(6), 0))

    def facts(self, parent: tk.Misc, facts: tuple) -> None:
        grid = tk.Frame(parent, bg=self.p.bg)
        grid.pack(fill="x")
        for index, fact in enumerate(facts):
            tk.Label(grid, text=fact.label, font=self.style.font("body"), fg=self.p.muted,
                     bg=self.p.bg, anchor="w").grid(row=index, column=0, sticky="nw",
                                                   pady=self.style.px(5), padx=(0, self.style.px(24)))
            tk.Label(grid, text=fact.value, font=self.style.font("body"), fg=self.p.text,
                     bg=self.p.bg, anchor="w", justify="left",
                     wraplength=self.style.px(460)).grid(row=index, column=1, sticky="w")

    def empty(self, parent: tk.Misc, text: str) -> None:
        self.label(parent, text, "body", self.p.muted, wrap=WRAP, pady=self.style.px(12))
