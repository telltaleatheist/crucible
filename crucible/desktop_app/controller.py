from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

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
}

LIVE = frozenset({"activity", "tasks"})

SLOW_SECONDS = 15.0

NEEDS = {
    "home": ("info", "activity", "capability", "tasks"),
    "models": ("catalog", "tasks", "capability"),
    "voices": ("voices", "catalog", "tasks", "info"),
    "packages": ("info", "capability", "tasks"),
    "activity": ("activity", "tasks"),
    "settings": ("settings", "setup"),
}

HOST_ERRORS = (CrucibleError, OSError, ValueError, RuntimeError)


def in_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True).start()


class Controller:
    def __init__(self, api: Any, host: Any, ask: Ask, run: Run = in_thread,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.api = api
        self.host = host
        self.ask = ask
        self.run = run
        self.clock = clock
        self.lock = threading.RLock()
        self.docs: dict[str, Any] = {}
        self.fetched: dict[str, float] = {}
        self.status: Mapping[str, Any] | Exception | None = None
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
        return seen is None or name in LIVE or self.clock() - seen >= SLOW_SECONDS

    def _read_status(self) -> None:
        try:
            self.status = self.host.status()
        except HOST_ERRORS as exc:
            self.status = exc

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

    def _read_lan(self) -> None:
        try:
            self.lan = self.host.lan_record()
        except HOST_ERRORS as exc:
            self.notices["lan"] = str(exc)

    def refresh_now(self, screen: str) -> None:
        self._read_status()
        if self.running():
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
                                 self.doc("capability"))
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
            return screens.activity_view(self.doc("activity"), self.doc("tasks"), self.watch_view())
        return screens.settings_view(self.doc("settings"), self.doc("setup"), self.lan,
                                     self.host.lan_supported())

    def screen_errors(self, screen: str) -> list[str]:
        return [screens.refusal_text(error) for name in NEEDS.get(screen, ())
                if (error := self.error(name)) is not None]
