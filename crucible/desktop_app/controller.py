from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable, Mapping

from .. import clock as wall_clock
from ..errors import CrucibleError
from . import screens
from .api import ApiError, quoted
from .host import Ask
from .progress import TaskWatch

Run = Callable[[Callable[[], None]], None]

PATHS = {
    "info": "/v1/info",
    "activity": "/v1/activity",
    "capability": "/v1/capability",
    "catalog": "/v1/catalog",
    "voices": "/v1/voices",
    "tasks": "/v1/tasks",
    "settings": "/v1/settings",
    "setup": "/v1/setup",
    "queue": "/v1/queue",
}

SLOW_SECONDS = 15.0

# A server that was answering and now misses a status read is weather until it has missed
# them for this long: the status ping waits 3 s, and a server busy with a long render has
# been measured taking 2-3.5 s to answer it (the PC, 2026-10-02), so a single miss flipped
# the window to "Crucible is not running" and back every few seconds. Inside the budget
# the window keeps what it last knew and says the server is slow to answer.
SLOW_ANSWER_BUDGET_SECONDS = 30.0

NEEDS = {
    "home": ("info", "activity", "queue", "capability", "tasks"),
    "models": ("catalog", "tasks", "capability"),
    "voices": ("voices", "catalog", "tasks", "info"),
    "packages": ("info", "capability", "tasks"),
    "activity": ("activity", "queue", "tasks"),
    "settings": ("settings", "setup"),
}

# GET /v1/events (docs/EVENTS.md) says when what these documents show has changed, so they
# are read again when an event says so, never on a timer. `activity`, `queue` and `tasks`
# arrive whole in the stream's snapshot. The rest (info, capability, catalog, voices,
# setup) have no event, so they are read every SLOW_SECONDS, and again when a task ends.
EVENTS_PATH = "/v1/events"
STREAMED = ("activity", "queue", "tasks")
EVENT_COVERED = frozenset(STREAMED + ("settings",))
STALE_ON = {
    "job": ("activity",),
    "card": ("activity",),
    "chat": ("activity",),
    "session": ("activity", "queue"),
    "queue": ("activity", "queue"),
    "settings": ("settings",),
}
TASK_CHANGES = frozenset({"task.running", "task.done", "task.failed", "task.cancelled"})
RECONNECT_SECONDS = (1.0, 2.0, 4.0, 8.0, 15.0)

HOST_ERRORS = (CrucibleError, OSError, ValueError, RuntimeError)


def _slow_not_gone(status: Mapping[str, Any] | Exception) -> bool:
    """A status read that says only that the server did not answer in time. A refused
    connection, a server stopped on purpose, a refused token or something else answering
    is a real state, shown at once."""
    return (isinstance(status, Mapping)
            and status.get("state") in ("unreachable", "unhealthy")
            and status.get("timed_out") is True)


def in_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True).start()


class Controller:
    def __init__(self, api: Any, host: Any, ask: Ask, run: Run = in_thread,
                 clock: Callable[[], float] = time.monotonic,
                 stream: Run = in_thread,
                 wall: Callable[[], datetime] = wall_clock.now,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.api = api
        self.host = host
        self.ask = ask
        self.run = run
        self.clock = clock
        self.stream = stream
        self.wall = wall
        self.sleep = sleep
        self.stale: set[str] = set()
        self.last_event_id = 0
        self.events_open = False
        self.events_started = False
        self.screen = "home"
        self.lock = threading.RLock()
        self.docs: dict[str, Any] = {}
        self.fetched: dict[str, float] = {}
        self.status: Mapping[str, Any] | Exception | None = None
        self.missed_since: float | None = None
        self.lan: Mapping[str, Any] | None = None
        self.notices: dict[str, str] = {}
        self.busy: set[str] = set()
        self.watch: TaskWatch | None = None
        self.version = 0
        self.refreshing = False

    def changed(self) -> None:
        with self.lock:
            self.version += 1

    def running(self) -> bool:
        return isinstance(self.status, Mapping) and self.status.get("state") == "running"

    def doc(self, name: str) -> Any:
        value = self.docs.get(name)
        return None if isinstance(value, ApiError) else value

    def error(self, name: str) -> ApiError | None:
        value = self.docs.get(name)
        return value if isinstance(value, ApiError) else None

    def invalidate(self) -> None:
        with self.lock:
            self.fetched.clear()

    def _due(self, name: str) -> bool:
        seen = self.fetched.get(name)
        if seen is None or name in self.stale:
            return True
        if name in EVENT_COVERED:
            return False
        return self.clock() - seen >= SLOW_SECONDS

    def _read_status(self) -> None:
        try:
            status: Mapping[str, Any] | Exception = self.host.status()
        except HOST_ERRORS as exc:
            status = exc
        answered = isinstance(status, Mapping) and status.get("state") == "running"
        if not answered and self.running() and _slow_not_gone(status):
            now = self.clock()
            if self.missed_since is None:
                self.missed_since = now
            waited = now - self.missed_since
            if waited < SLOW_ANSWER_BUDGET_SECONDS:
                self.notices["status"] = (
                    f"Crucible is slow to answer (no reply for {waited:.0f} s); it is "
                    "probably busy with a long job. Showing what it last said."
                )
                return
        self.missed_since = None
        self.notices.pop("status", None)
        self.status = status

    def _read(self, names: tuple[str, ...]) -> None:
        for name in names:
            if not self._due(name):
                continue
            try:
                value = self.api.get(PATHS[name])
            except ApiError as exc:
                value = exc
            with self.lock:
                self.docs[name] = value
                self.fetched[name] = self.clock()
                self.stale.discard(name)

    def _read_lan(self) -> None:
        try:
            self.lan = self.host.lan_record()
        except HOST_ERRORS as exc:
            self.notices["lan"] = str(exc)

    def refresh_now(self, screen: str) -> None:
        self.screen = screen
        self._read_status()
        if self.running():
            self.start_events()
            self._idle_passed()
            self._read(NEEDS.get(screen, ()))
            if screen == "settings":
                self._read_lan()
            self._follow_running_task()
        self.changed()

    def refresh(self, screen: str) -> None:
        with self.lock:
            if self.refreshing:
                return
            self.refreshing = True

        def work() -> None:
            try:
                self.refresh_now(screen)
            finally:
                self.refreshing = False
        self.run(work)

    def start_events(self) -> None:
        """Follow GET /v1/events for as long as this window lives: one stream, resumed
        with Last-Event-ID after a drop, a fresh snapshot after a gap."""
        with self.lock:
            if self.events_started:
                return
            self.events_started = True
        self.stream(self._events_forever)

    def _events_forever(self) -> None:
        failures = 0
        while True:
            if self.follow_events():
                failures = 0
            self.sleep(RECONNECT_SECONDS[min(failures, len(RECONNECT_SECONDS) - 1)])
            failures += 1

    def follow_events(self) -> bool:
        """Read the stream until it ends, and say whether it opened. It ends when the
        connection drops, on `overflow` and on `server.stopping`; the caller reconnects,
        and the server resumes from `last_event_id` or sends a snapshot with `gap`."""
        opened = False
        try:
            for frame in self.api.follow(EVENTS_PATH, last_event_id=self.last_event_id):
                opened = True
                self._on_event(frame)
        except ApiError:
            pass
        finally:
            self.events_open = False
            self.changed()
        return opened

    def _on_event(self, frame: Mapping[str, Any]) -> None:
        name = str(frame.get("event") or "")
        data = frame.get("data") or {}
        ident = frame.get("id")
        if isinstance(ident, int):
            self.last_event_id = ident
        if name == "snapshot":
            self._take_snapshot(data)
            return
        if name == "job.progress":
            self._job_progress(data)
            return
        with self.lock:
            self.stale.update(STALE_ON.get(name.partition(".")[0], ()))
            if name in TASK_CHANGES:
                self.stale.add("tasks")
            if name in TASK_CHANGES - {"task.running"}:
                self.fetched.clear()
            any_stale = bool(self.stale)
        if any_stale:
            self.refresh(self.screen)
        self.changed()

    def _take_snapshot(self, data: Mapping[str, Any]) -> None:
        with self.lock:
            if data.get("gap"):
                self.fetched.clear()
            now = self.clock()
            self.docs["activity"] = data.get("activity")
            self.docs["queue"] = data.get("queue")
            self.docs["tasks"] = {"tasks": data.get("tasks") or []}
            for name in STREAMED:
                self.fetched[name] = now
                self.stale.discard(name)
            self.events_open = True
        self._follow_running_task()
        self.changed()

    def _job_progress(self, data: Mapping[str, Any]) -> None:
        activity = self.doc("activity")
        if not isinstance(activity, Mapping):
            return
        rows = []
        for row in activity.get("running") or []:
            if row.get("job_id") == data.get("job_id"):
                row = {**row, "progress": data.get("fraction"), "message": data.get("message")}
            rows.append(row)
        with self.lock:
            self.docs["activity"] = {**activity, "running": rows}
        self.changed()

    def _idle_passed(self) -> None:
        """A session's idle deadline also moves when its client touches it, which no event
        says, so a countdown that ran out with the session still open is read again."""
        activity = self.doc("activity")
        session = activity.get("session") if isinstance(activity, Mapping) else None
        deadline = session.get("idle_deadline") if isinstance(session, Mapping) else None
        if not isinstance(deadline, str):
            return
        if datetime.fromisoformat(deadline) <= self.wall():
            with self.lock:
                self.stale.add("activity")

    def _follow_running_task(self) -> None:
        task = screens.running_task(self.doc("tasks"))
        with self.lock:
            if task is None or (self.watch is not None and self.watch.task_id == task.get("task_id")):
                return
            self.watch = TaskWatch.of(task)
            watch = self.watch
        self.run(lambda: self.follow(watch))

    def follow(self, watch: TaskWatch) -> None:
        path = "/v1/tasks/" + quoted(watch.task_id)
        try:
            for frame in self.api.follow(path + "/events"):
                watch.apply(str(frame.get("event")), frame.get("data") or {})
                self.changed()
                if watch.ended:
                    break
            watch.settle(self.api.get(path))
        except ApiError as exc:
            watch.finish("failed", {"code": exc.code, "message": exc.message})
        self.invalidate()
        self.changed()

    def act(self, place: str, work: Callable[[], None]) -> None:
        with self.lock:
            if place in self.busy:
                return
            self.busy.add(place)
            self.notices.pop(place, None)
        self.changed()

        def guarded() -> None:
            try:
                work()
            except ApiError as exc:
                self.notices[place] = screens.refusal_text(exc)
            except HOST_ERRORS as exc:
                self.notices[place] = str(exc)
            finally:
                self.busy.discard(place)
                self.invalidate()
                self.changed()
        self.run(guarded)

    def server(self, action: str) -> None:
        def work() -> None:
            self.status = self.host.act(action)
            self.api.forget()
        self.act("home", work)

    def _plan_says_yes(self, query: str, fallback: str) -> bool:
        try:
            plan = self.api.get("/v1/capability/plan?" + query)
        except ApiError as exc:
            if exc.status != 404:
                raise
            plan = {}
        return self.ask(str(plan.get("confirm") or fallback))

    def _submit(self, request: dict[str, Any]) -> None:
        admitted = self.api.send("POST", "/v1/tasks", request)
        if isinstance(admitted, Mapping) and admitted.get("task_id"):
            task = {"task_id": admitted["task_id"], "type": request["type"], "request": request}
            with self.lock:
                self.watch = TaskWatch.of(task)
                watch = self.watch
            self.run(lambda: self.follow(watch))

    def pull(self, kind: str, subject_id: str) -> None:
        def work() -> None:
            if self._plan_says_yes("subject=" + quoted(subject_id), f"Download {subject_id}?"):
                self._submit({"type": "pull", "kind": kind, "id": subject_id})
        self.act(f"{kind}:{subject_id}", work)

    def remove(self, kind: str, subject_id: str, size: str) -> None:
        def work() -> None:
            frees = f" This frees {size}." if size else ""
            if self.ask(f"Remove {subject_id} from this computer?{frees}\n\n"
                        "The files are deleted. Getting it back is another download."):
                self.api.send("DELETE", "/v1/catalog/" + quoted(kind, subject_id))
        self.act(f"{kind}:{subject_id}", work)

    def install(self, job_type: str, engine: str | None) -> None:
        def work() -> None:
            if self._plan_says_yes("job_type=" + quoted(job_type), f"Install {job_type}?"):
                request: dict[str, Any] = {"type": "install", "job_type": job_type}
                if engine:
                    request["narrator_engine"] = engine
                self._submit(request)
        self.act(f"package:{job_type}", work)

    def cancel(self, target: str, ident: str) -> None:
        def work() -> None:
            if target == "task":
                self.api.send("DELETE", "/v1/tasks/" + quoted(ident))
            elif self.ask("Cancel this job? The app that sent it will be told it was cancelled."):
                self.api.send("DELETE", "/v1/jobs/" + quoted(ident))
        self.act(f"cancel:{ident}", work)

    def remove_queued(self, job_id: str) -> None:
        def work() -> None:
            if self.ask("Remove this job from the queue? It will not run, and the app that "
                        "sent it will be told it was removed."):
                self.api.send("DELETE", "/v1/queue/" + quoted(job_id))
        self.act(f"queue:{job_id}", work)

    def end_session(self, session_id: str) -> None:
        def work() -> None:
            if self.ask("End this app's session? What it is running finishes, nothing more "
                        "of its runs, and the app is told an operator ended it."):
                self.api.send("DELETE", "/v1/queue/" + quoted(session_id))
        self.act(f"session:{session_id}", work)

    def reset_voice(self, voice_id: str) -> None:
        def work() -> None:
            if self.ask(f"Go back to the version of {voice_id} that came with Crucible?"):
                self.api.send("DELETE", "/v1/voices/" + quoted(voice_id))
        self.act(f"voice:{voice_id}", work)

    def save_upstream(self, name: str, field: str, value: str) -> None:
        patch = {"upstreams": {name: {field: value.strip()} if value.strip() else None}}

        def work() -> None:
            self.docs["settings"] = self.api.send("PUT", "/v1/settings", patch)
        self.act(f"upstream:{name}", work)

    def save_allowance(self, gib_text: str) -> None:
        def work() -> None:
            try:
                value = round(float(gib_text) * screens.GIB)
            except ValueError:
                raise ApiError("allowance_not_a_number",
                               f"{gib_text!r} is not a number of GB; type one, like 3") from None
            self.docs["settings"] = self.api.send("PUT", "/v1/settings", {"desktop_allowance_bytes": value})
        self.act("allowance", work)

    def set_lan(self, on: bool) -> None:
        def work() -> None:
            report = self.host.set_lan(on, self.ask)
            if report.get("next"):
                self.notices["lan"] = str(report["next"])
            self.lan = self.host.lan_record()
        self.act("lan", work)

    def open_logs(self) -> None:
        self.act("logs", lambda: self.host.open_logs())

    def watch_view(self) -> screens.Progress | None:
        watch = self.watch
        return None if watch is None else watch.view()

    def home(self) -> screens.HomeView:
        if self.status is None:
            return screens.HomeView(headline="Looking for Crucible", tone=screens.IDLE,
                                    detail="Checking whether it is running")
        view = screens.home_view(self.status, self.doc("info"), self.doc("activity"),
                                 self.doc("capability"), self.doc("queue"))
        watch = self.watch_view()
        if watch is None or not self.running():
            return view
        return screens.HomeView(**{**view.__dict__, "work": (watch,) + view.work})

    def view(self, screen: str) -> Any:
        if screen == "home":
            return self.home()
        if not self.running():
            return self.home()
        if screen == "models":
            return screens.model_rows(self.doc("catalog"), self.doc("tasks"))
        if screen == "voices":
            return screens.voice_rows(self.doc("voices"), self.doc("catalog"), self.doc("tasks"),
                                      self.doc("info"))
        if screen == "packages":
            return screens.package_rows(self.doc("info"), self.doc("capability"), self.doc("tasks"))
        if screen == "activity":
            return screens.activity_view(self.doc("activity"), self.doc("tasks"), self.watch_view(),
                                         self.doc("queue"), self.wall())
        return screens.settings_view(self.doc("settings"), self.doc("setup"), self.lan,
                                     self.host.lan_supported())

    def screen_errors(self, screen: str) -> list[str]:
        return [screens.refusal_text(error) for name in NEEDS.get(screen, ())
                if (error := self.error(name)) is not None]
