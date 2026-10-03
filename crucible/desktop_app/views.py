from __future__ import annotations

import tkinter as tk
from typing import Any, Callable

from . import screens
from .page import Page

SCREENS = (
    ("home", "Home"),
    ("models", "Models"),
    ("voices", "Voices"),
    ("packages", "Packages"),
    ("activity", "Activity"),
    ("settings", "Settings"),
)

SUBTITLES = {
    "models": "Language, speech and audio models on this computer.",
    "voices": "Narration voices. Download one to read text aloud in it.",
    "packages": "What this computer can do. Install a package to add a kind of work.",
    "activity": "What Crucible is doing right now.",
    "settings": "Connections, sharing and keys.",
}


class Actions:
    def __init__(self, controller: Any, go: Callable[[str], None], copy: Callable[[str], None]) -> None:
        self.c = controller
        self.go = go
        self.copy = copy

    def cancel(self, work: screens.Progress) -> None:
        if work.cancel:
            self.c.cancel(work.target, work.cancel)


def draw_notices(page: Page, parent: tk.Misc, texts: list[str]) -> None:
    for text in texts:
        page.notice(parent, text)


def notice_for(page: Page, parent: tk.Misc, actions: Actions, place: str) -> None:
    if place in actions.c.notices:
        page.notice(parent, actions.c.notices[place])


def draw_home(page: Page, view: screens.HomeView, actions: Actions) -> None:
    page.header(view.headline, view.detail, view.tone)
    notice_for(page, page.body, actions, "status")
    busy = "home" in actions.c.busy
    if view.action:
        line = page.frame(page.body, pady=(page.style.px(8), 0))
        label = "Working..." if busy else view.action_label
        page.button(line, label, lambda: actions.c.server(view.action), kind="primary",
                    enabled=not busy, side="left")
    notice_for(page, page.body, actions, "home")
    if view.facts:
        page.facts(page.section("This computer"), view.facts)
    if view.work:
        box = page.section("Working on")
        for work in view.work:
            page.progress(box, work, actions.cancel)


def draw_down(page: Page, screen: str, view: screens.HomeView, actions: Actions) -> None:
    page.header(dict(SCREENS)[screen], SUBTITLES.get(screen, ""))
    box = page.section(view.headline)
    page.empty(box, view.detail)
    line = page.frame(box)
    page.button(line, "Go to Home", lambda: actions.go("home"), kind="primary", side="left")


def subject_actions(page: Page, right: tk.Frame, row: screens.SubjectRow, actions: Actions) -> None:
    busy = f"{row.kind}:{row.id}" in actions.c.busy
    if row.installed:
        page.button(right, "Remove", lambda: actions.c.remove(row.kind, row.id, row.size),
                    kind="danger", enabled=row.can_remove and not busy)
    else:
        page.button(right, "Download", lambda: actions.c.pull(row.kind, row.id), kind="primary",
                    enabled=row.can_pull and not busy)
    if "reset" in row.extra:
        page.button(right, "Use shipped version", lambda: actions.c.reset_voice(row.id))
    page.tag(right, row.size, screens.IDLE)
    page.tag(right, row.note, row.tone)


def draw_subjects(page: Page, rows: list[screens.SubjectRow], actions: Actions, empty: str) -> None:
    groups = (("On this computer", [row for row in rows if row.installed]),
              ("Available to download", [row for row in rows if not row.installed]))
    watch = actions.c.watch_view()
    if not rows:
        page.empty(page.body, empty)
    for title, members in groups:
        if not members:
            continue
        box = page.section(title)
        for row in members:
            _left, right, below = page.row(box, row.title, row.subtitle)
            subject_actions(page, right, row, actions)
            if watch is not None and watch.title.endswith(f" {row.id}"):
                page.progress(below, watch, actions.cancel)
            place = f"{row.kind}:{row.id}"
            notice_for(page, below, actions, place)


def draw_packages(page: Page, rows: list[screens.PackageRow], actions: Actions) -> None:
    watch = actions.c.watch_view()
    box = page.frame(page.body, pady=(page.style.px(16), 0))
    for row in rows:
        left, right, below = page.row(box, row.title, row.blurb)
        if row.verdict:
            page.tag(left, row.verdict, row.tone, side="top")
        place = f"package:{row.job_type}"
        if row.installed:
            page.tag(right, "Installed", screens.OK)
        elif row.note:
            page.tag(right, row.note, screens.IDLE)
        else:
            engine = row.engines[0] if row.engines else None
            page.button(right, "Install", lambda job=row.job_type, e=engine: actions.c.install(job, e),
                        kind="primary", enabled=row.installable and place not in actions.c.busy)
        if watch is not None and watch.title.startswith(f"Installing {row.job_type}"):
            page.progress(below, watch, actions.cancel)
        notice_for(page, below, actions, place)


def draw_activity(page: Page, view: screens.ActivityView, actions: Actions) -> None:
    box = page.section("Running now")
    notice_for(page, box, actions, "status")
    if not view.work:
        page.empty(box, "Nothing is running.")
    for work in view.work:
        page.progress(box, work, actions.cancel)
    for place, text in actions.c.notices.items():
        if place.startswith("cancel:"):
            page.notice(box, text)
    draw_queue(page, view, actions)
    memory = page.section("Memory")
    kept = (screens.Fact("Kept by", view.session.title),) if view.session else ()
    page.facts(memory, (view.loaded,) + kept)
    history = page.section("Recent downloads and installs")
    if not view.tasks:
        page.empty(history, "None since Crucible last started.")
    for line in view.tasks:
        _left, right, _below = page.row(history, line.title, line.detail)
        page.tag(right, line.state.capitalize(), line.tone)


def draw_queue(page: Page, view: screens.ActivityView, actions: Actions) -> None:
    box = page.section("Queue")
    if view.session is not None:
        _left, right, below = page.row(box, view.session.title, view.session.detail)
        place = f"session:{view.session.session_id}"
        page.button(right, "End", lambda ident=view.session.session_id: actions.c.end_session(ident),
                    kind="danger", enabled=place not in actions.c.busy)
        notice_for(page, below, actions, place)
    if not view.queue and view.session is None:
        page.empty(box, "Nothing is waiting. Jobs, chats and app sessions that arrive while "
                   "Crucible is busy wait here.")
    for line in view.queue:
        _left, right, below = page.row(box, line.title, line.detail)
        place = f"queue:{line.job_id}"
        page.button(right, "Remove", lambda job=line.job_id: actions.c.remove_queued(job),
                    kind="danger", enabled=place not in actions.c.busy)
        page.tag(right, line.waited, screens.IDLE)
        page.tag(right, screens.KIND_WORDS.get(line.kind, line.kind), screens.IDLE)
        notice_for(page, below, actions, place)


def entry(page: Page, parent: tk.Misc, value: str = "", secret: bool = False, width: int = 28) -> tk.Entry:
    p = page.p
    widget = tk.Entry(parent, font=page.style.font("body"), bg=p.entry, fg=p.text, insertbackground=p.text,
                      relief="flat", highlightthickness=1, highlightbackground=p.line,
                      highlightcolor=p.muted, width=width, show="•" if secret else "")
    widget.insert(0, value)
    widget.pack(side="right", ipady=page.style.px(5), padx=(page.style.px(8), 0))
    return widget


def draw_pairing(page: Page, view: screens.SettingsView, actions: Actions) -> None:
    box = page.section("Connect an app or another computer")
    if not view.pairing:
        page.empty(box, "The server did not say how to reach it.")
    for line in view.pairing:
        _left, right, _below = page.row(box, f"Pairing line for {line.where}", line.shown)
        page.button(right, "Copy", lambda text=line.line: actions.copy(text))


def draw_lan(page: Page, view: screens.SettingsView, actions: Actions) -> None:
    box = page.section("Share on your network")
    _left, right, below = page.row(box, "On" if view.lan_on else "Off", view.lan_words)
    if view.lan_supported:
        busy = "lan" in actions.c.busy
        label = "Working..." if busy else ("Stop sharing" if view.lan_on else "Share")
        page.button(right, label, lambda: actions.c.set_lan(not view.lan_on),
                    kind="secondary" if view.lan_on else "primary", enabled=not busy)
    notice_for(page, below, actions, "lan")


def draw_upstreams(page: Page, view: screens.SettingsView, actions: Actions) -> None:
    box = page.section("Cloud providers")
    for row in view.upstreams:
        words = row.shown if row.configured else "Not set"
        _left, right, below = page.row(box, row.label, words)
        if row.configured:
            page.button(right, "Remove", lambda name=row.name, f=row.field: actions.c.save_upstream(name, f, ""),
                        kind="danger")
        field = entry(page, right, "", secret=row.field == "key")
        page.button(right, "Save", lambda name=row.name, f=row.field, box=field:
                    actions.c.save_upstream(name, f, box.get())).pack_configure(before=field)
        place = f"upstream:{row.name}"
        notice_for(page, below, actions, place)


def draw_settings(page: Page, view: screens.SettingsView, actions: Actions) -> None:
    draw_pairing(page, view, actions)
    draw_lan(page, view, actions)
    draw_upstreams(page, view, actions)
    box = page.section("Memory kept for the desktop")
    _left, right, below = page.row(box, "Kept free for other programs, in GB", view.allowance)
    field = entry(page, right, view.allowance_gib, width=6)
    page.button(right, "Save", lambda: actions.c.save_allowance(field.get())).pack_configure(before=field)
    notice_for(page, below, actions, "allowance")
    logs = page.section("Logs")
    _left, right, below = page.row(logs, "Crucible's log files", "Open them when something goes wrong.")
    page.button(right, "Open logs folder", actions.c.open_logs)
    notice_for(page, below, actions, "logs")


def draw(page: Page, screen: str, view: Any, actions: Actions) -> None:
    if screen == "home":
        draw_home(page, view, actions)
        return
    if isinstance(view, screens.HomeView):
        draw_down(page, screen, view, actions)
        return
    page.header(dict(SCREENS)[screen], SUBTITLES.get(screen, ""))
    draw_notices(page, page.body, actions.c.screen_errors(screen))
    DRAWERS[screen](page, view, actions)


DRAWERS = {
    "models": lambda page, view, actions: draw_subjects(
        page, view, actions, "No models are listed for this computer."),
    "voices": lambda page, view, actions: draw_subjects(
        page, view, actions, "No voices yet. Install the Narration package first."),
    "packages": draw_packages,
    "activity": draw_activity,
    "settings": draw_settings,
}
