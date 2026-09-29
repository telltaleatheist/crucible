from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any

from . import theme, views
from .api import LocalApi
from .controller import Controller
from .host import LocalHost
from .instance import FOCUS, QUIT, Instance
from .launchers import assets_dir
from .page import Page
from .widgets import ScrollArea, Style

POLL_MS = 120
REFRESH_MS = 2000
THEME_MS = 4000
SIDEBAR_WIDTH = 200
CONTENT_PAD = 36
WINDOWS_APP_ID = "Crucible.App"
DWM_DARK_TITLE = 20


class Asker:
    def __init__(self) -> None:
        self.questions: queue.Queue = queue.Queue()

    def __call__(self, question: str) -> bool:
        answer: dict[str, bool] = {}
        done = threading.Event()
        self.questions.put((question, answer, done))
        done.wait()
        return answer.get("yes", False)

    def answer_pending(self, root: tk.Tk) -> None:
        while not self.questions.empty():
            question, answer, done = self.questions.get_nowait()
            answer["yes"] = messagebox.askyesno("Crucible", question, parent=root)
            done.set()


def prepare_windows() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def dark_title_bar(root: tk.Tk, dark: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWM_DARK_TITLE, ctypes.byref(value),
                                                   ctypes.sizeof(value))
        root.withdraw()
        root.deiconify()
    except (AttributeError, OSError):
        pass


class App:
    def __init__(self, root: tk.Tk, controller: Controller, asker: Asker) -> None:
        self.root = root
        self.c = controller
        self.asker = asker
        self.heard: queue.Queue = queue.Queue()
        self.screen = "home"
        self.shown: tuple | None = None
        self.palette = theme.current()
        self.icons = [tk.PhotoImage(file=str(assets_dir() / f"icon-{size}.png")) for size in (256, 64, 32, 16)]
        self.root.iconphoto(True, *self.icons)
        self.logo = tk.PhotoImage(file=str(assets_dir() / "icon-32.png"))
        self.actions = views.Actions(controller, self.go, self.copy)
        self.build()

    def build(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()
        self.style = Style(self.root, self.palette)
        self.root.configure(bg=self.palette.bg)
        self.root.update_idletasks()
        dark_title_bar(self.root, self.palette.name == "dark")
        self.sidebar = tk.Frame(self.root, bg=self.palette.sidebar, width=self.style.px(SIDEBAR_WIDTH))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.scroll = ScrollArea(self.root, self.style)
        self.scroll.pack(side="left", fill="both", expand=True)
        self.shown = None
        self.draw_sidebar()

    def draw_sidebar(self) -> None:
        for child in self.sidebar.winfo_children():
            child.destroy()
        p, s = self.palette, self.style
        brand = tk.Frame(self.sidebar, bg=p.sidebar)
        brand.pack(fill="x", padx=s.px(18), pady=(s.px(22), s.px(18)))
        tk.Label(brand, image=self.logo, bg=p.sidebar).pack(side="left")
        tk.Label(brand, text="Crucible", font=s.font("brand"), fg=p.text, bg=p.sidebar).pack(
            side="left", padx=(s.px(10), 0))
        for key, title in views.SCREENS:
            self.nav_item(key, title)

    def nav_item(self, key: str, title: str) -> None:
        p, s = self.palette, self.style
        chosen = key == self.screen
        fill = p.selected if chosen else p.sidebar
        item = tk.Label(self.sidebar, text=title, font=s.font("nav"), anchor="w", cursor="hand2",
                        fg=p.text if chosen else p.muted, bg=fill, padx=s.px(12), pady=s.px(7))
        item.pack(fill="x", padx=s.px(10), pady=s.px(1))
        item.bind("<Button-1>", lambda _e: self.go(key))
        if not chosen:
            item.bind("<Enter>", lambda _e: item.configure(fg=p.text))
            item.bind("<Leave>", lambda _e: item.configure(fg=p.muted))

    def go(self, screen: str) -> None:
        self.screen = screen
        self.draw_sidebar()
        self.shown = None
        self.render(scroll_top=True)
        self.c.refresh(screen)

    def copy(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def typing(self) -> bool:
        focus = self.root.focus_get()
        return isinstance(focus, tk.Entry)

    def signature(self) -> tuple:
        view = self.c.view(self.screen)
        return (self.screen, repr(view), repr(sorted(self.c.notices.items())), repr(sorted(self.c.busy)),
                repr(self.c.watch_view()), repr(self.c.screen_errors(self.screen)))

    def render(self, scroll_top: bool = False) -> None:
        signature = self.signature()
        if signature == self.shown or (self.typing() and self.shown is not None and not scroll_top):
            return
        self.shown = signature
        top = 0.0 if scroll_top else self.scroll.top()
        for child in self.scroll.inner.winfo_children():
            child.destroy()
        body = tk.Frame(self.scroll.inner, bg=self.palette.bg)
        body.pack(fill="both", expand=True, padx=self.style.px(CONTENT_PAD), pady=(0, self.style.px(28)))
        views.draw(Page(body, self.style), self.screen, self.c.view(self.screen), self.actions)
        self.scroll.restore(top)

    def poll(self) -> None:
        self.asker.answer_pending(self.root)
        while not self.heard.empty():
            self.hear(self.heard.get_nowait())
        self.render()
        self.root.after(POLL_MS, self.poll)

    def hear(self, word: str) -> None:
        if word == QUIT:
            self.root.destroy()
            return
        if word == FOCUS:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()

    def tick(self) -> None:
        self.c.refresh(self.screen)
        self.root.after(REFRESH_MS, self.tick)

    def watch_theme(self) -> None:
        palette = theme.current()
        if palette != self.palette:
            self.palette = palette
            self.build()
        self.root.after(THEME_MS, self.watch_theme)

    def start(self) -> None:
        self.c.refresh(self.screen)
        self.root.after(POLL_MS, self.poll)
        self.root.after(REFRESH_MS, self.tick)
        self.root.after(THEME_MS, self.watch_theme)


def make_root() -> tk.Tk:
    prepare_windows()
    root = tk.Tk(className="Crucible")
    root.title("Crucible")
    scale = max(1.0, root.winfo_fpixels("1i") / 96.0) if sys.platform == "win32" else 1.0
    root.geometry(f"{round(920 * scale)}x{round(640 * scale)}")
    root.minsize(round(760 * scale), round(520 * scale))
    return root


def run(home: Path, instance: Instance, root: Any = None) -> int:
    root = root if root is not None else make_root()
    asker = Asker()
    controller = Controller(LocalApi(), LocalHost(home), asker)
    app = App(root, controller, asker)
    instance.serve(app.heard.put)
    app.start()
    root.mainloop()
    return 0
