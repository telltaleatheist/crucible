from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any, Awaitable, Callable

from .. import catalog, tasks
from ..backend import Backend
from ..clock import utcnow
from ..config import Config
from ..errors import ApiError
from ..settle import Held
from ..weights import PullCancelled, WeightsError
from . import hostdoor, runner
from .states import (
    CANCELLED,
    DONE,
    FAILED,
    HISTORY,
    TERMINAL_STATES,
    ReloadRefused,
    Task,
    TaskCancelled,
    TaskFailedByHost,
)
from .validate import (
    ModuleEntry,
    install_label,
    touches_the_registry,
    validate_module,
    validate_request,
)

ALREADY_TRUE = (
    "A module says what must be true, so an entry that is already true is "
    "skipped rather than refused"
)


class TaskStore:

    def __init__(
        self,
        config: Config,
        backend: Backend,
        *,
        reload: Callable[[], list[str]],
        holder: Callable[[], Held | None],
        take_up: Callable[[], list[str]] | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._reload = reload
        self._holder = holder
        self._take_up = take_up
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._running_id: str | None = None
        self._runner: asyncio.Task[None] | None = None
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def stop(self) -> None:
        runner_task, self._runner = self._runner, None
        if runner_task is None:
            return
        running = self.running
        if running is not None:
            self._request_cancel(running)
        runner_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass

    @property
    def running(self) -> Task | None:
        return None if self._running_id is None else self._tasks[self._running_id]

    def get(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise ApiError(
                404,
                "unknown_task",
                f"no task {task_id} on this server. Tasks are held in memory and "
                f"the last {HISTORY} are kept, so this one may have finished "
                "before a restart or been pushed out by newer ones",
            )
        return task

    def recent(self) -> list[Task]:
        return [self._tasks[task_id] for task_id in reversed(self._order)]

    def refuse_if_busy(self) -> None:
        running = self.running
        if running is None:
            return
        raise ApiError(
            409,
            "task_busy",
            f"this server is already running task {running.id} ({running.type}), "
            f"started {running.started}. One operator task at a time: they write "
            "to the same config and the same weights tree, and two at once would "
            "be two answers about what is installed. Watch "
            f"GET /v1/tasks/{running.id}/events for its end",
            {"task_id": running.id, "type": running.type, "since": running.started},
        )

    def refuse_if_the_card_is_held(self) -> None:
        held = self._holder()
        if held is None:
            return
        raise ApiError(
            409,
            "server_busy",
            f"this server cannot install anything right now: {held}. An install "
            "rewrites config.toml and reloads this server's job registry, so it "
            "waits until nothing holds the card",
            {"fact": held.fact, "who": held.who, **held.details, "door": "operator"},
        )

    def submit(self, request: dict[str, Any], *, on_submit: bool = False) -> Task:
        validate_request(self._config, self._backend, request)
        self.refuse_if_busy()
        if touches_the_registry(request) and not on_submit:
            self.refuse_if_the_card_is_held()

        now = utcnow()
        task = Task(
            id=uuid.uuid4().hex,
            type=request["type"],
            request=request,
            created=now,
            started=now,
            on_submit=on_submit,
        )
        self._tasks[task.id] = task
        self._order.append(task.id)
        self._prune()
        self._running_id = task.id
        self._loop = asyncio.get_running_loop()
        self._runner = self._loop.create_task(
            self._run(task), name=f"crucible-task-{task.id}"
        )
        return task

    def _prune(self) -> None:
        while len(self._order) > HISTORY:
            oldest = self._order[0]
            if self._tasks[oldest].state not in TERMINAL_STATES:
                raise RuntimeError(
                    f"task {oldest} is {self._tasks[oldest].state} and would be "
                    "pruned; the history must never drop a live task"
                )
            self._order.pop(0)
            self._tasks.pop(oldest, None)
            self._subscribers.pop(oldest, None)

    def cancel(self, task: Task) -> str:
        if task.state in TERMINAL_STATES:
            raise ApiError(
                409,
                "not_running",
                f"task {task.id} is already {task.state}; there is nothing to "
                "cancel",
                {"task_id": task.id, "state": task.state},
            )
        self._request_cancel(task)
        return "cancelling"

    def _request_cancel(self, task: Task) -> None:
        task.cancel_requested = True
        process = task.process
        if process is not None and process.poll() is None:
            process.terminate()

    def append_event(self, task: Task, kind: str, data: dict[str, Any]) -> None:
        task.events.append(
            {"id": len(task.events) + 1, "event": kind, "data": data}
        )
        for waiter in self._subscribers.get(task.id, []):
            waiter.set()

    def subscribe(self, task: Task) -> asyncio.Event:
        waiter = asyncio.Event()
        self._subscribers.setdefault(task.id, []).append(waiter)
        return waiter

    def unsubscribe(self, task: Task, waiter: asyncio.Event) -> None:
        waiters = self._subscribers.get(task.id)
        if waiters is None:
            return
        if waiter in waiters:
            waiters.remove(waiter)
        if not waiters:
            self._subscribers.pop(task.id, None)

    def _from_thread(self, task: Task, kind: str, data: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self.append_event, task, kind, data)

    def _emitter(self, task: Task) -> runner.Emit:
        return lambda kind, data: self._from_thread(task, kind, data)

    def _step(self, task: Task, name: str, index: int, total: int, **extra: Any) -> None:
        self.append_event(
            task, "step", {"name": name, "index": index, "total": total, **extra}
        )

    def _skip(self, task: Task, reason: str) -> None:
        self.append_event(task, "skipped", {"reason": reason})

    def _runs(self) -> dict[str, Callable[[Task], Awaitable[None]]]:
        return {
            "pull": self._run_pull,
            "install": self._run_install,
            "engine": self._run_engine,
            "engine-restart": self._run_engine_restart,
            "module": self._run_module,
        }

    async def _run(self, task: Task) -> None:
        try:
            self.append_event(task, "started", {"type": task.type})
            await self._runs()[task.type](task)
        except (TaskCancelled, PullCancelled):
            self._finish(task, CANCELLED, None)
        except TaskFailedByHost:
            self._finish(task, FAILED, None)
        except ReloadRefused as exc:
            self._finish(
                task,
                FAILED,
                {
                    "code": "reload_refused",
                    "message": (
                        f"the env is installed, but this server would not swap "
                        f"its job registry while {exc.held}. Nothing was undone "
                        f"(R6); run the install again and it will find the env "
                        f"built and reach the reload in seconds"
                    ),
                },
            )
        except ApiError as exc:
            self._finish(task, FAILED, {"code": exc.code, "message": exc.message})
        except asyncio.CancelledError:
            self._finish(task, CANCELLED, None)
            raise
        except Exception as exc:
            self._finish(
                task,
                FAILED,
                {"code": "task_failed", "message": f"{type(exc).__name__}: {exc}"},
            )
        else:
            self._finish(task, DONE, None)

    def _finish(
        self, task: Task, state: str, error: dict[str, str] | None
    ) -> None:
        task.state = state
        task.finished = utcnow()
        task.error = error
        task.process = None
        self._running_id = None
        if state == DONE:
            self.append_event(task, "done", {"unmet": task.unmet})
        elif state == FAILED:
            if error is not None:
                self.append_event(task, "failed", error)
        else:
            self.append_event(task, "cancelled", {})

    async def _run_pull(self, task: Task) -> None:
        task.raise_if_cancelled()
        kind, subject_id = task.request["kind"], task.request["id"]
        subject = catalog.find(self._config, self._backend, kind, subject_id)
        if subject is None:
            raise ApiError(
                404, "unknown_subject", f"no {kind} called {subject_id!r}"
            )
        self._step(task, f"pull {kind} {subject_id}", 1, 1)
        await self._pull(task, subject)

    async def _pull(self, task: Task, subject: catalog.Subject) -> None:
        try:
            await asyncio.to_thread(self._pull_blocking, task, subject)
        except WeightsError as exc:
            raise ApiError(
                500,
                "pull_failed",
                f"pulling {subject.kind} {subject.id!r} failed: {exc}",
                {"kind": subject.kind, "id": subject.id},
            ) from None

    def _pull_blocking(self, task: Task, subject: catalog.Subject) -> None:
        throttle = runner.Throttle(runner.PROGRESS_INTERVAL_SECONDS)

        def on_progress(done: int, total: int | None, name: str) -> None:
            if task.cancel_requested:
                raise PullCancelled(f"task {task.id} was cancelled")
            if throttle.due():
                self._from_thread(
                    task,
                    "progress",
                    {"bytes_done": done, "bytes_total": total, "file": name},
                )

        def on_line(line: str) -> None:
            print(f"crucible: task {task.id}: {line}", file=sys.stderr)

        subject.pull(force=False, on_line=on_line, on_progress=on_progress)

    async def _relay(self, task: Task, door: str, path: str, body: dict[str, Any]) -> str | None:
        return await asyncio.to_thread(
            hostdoor.relay,
            task,
            door,
            path,
            body,
            token=self._config.token,
            emit=self._emitter(task),
        )

    async def _run_engine(self, task: Task) -> None:
        target = task.request["target"]
        door = hostdoor.door_for_move(self._backend, target)
        self._step(task, "hand the move to the host", 1, 1)
        terminal = await self._relay(task, door, hostdoor.HOST_DOOR_PATH, {"target": target})
        if terminal is None:
            raise hostdoor.silent_move(door)
        if terminal == "failed":
            raise TaskFailedByHost(task)

    async def _run_engine_restart(self, task: Task) -> None:
        door = hostdoor.door_for_restart()
        self._step(task, "hand the restart to the orchestrator", 1, 1)
        terminal = await self._relay(task, door, hostdoor.HOST_DOOR_RESTART_PATH, {})
        if terminal is None:
            raise hostdoor.silent_restart(door)
        if terminal == "failed":
            raise TaskFailedByHost(task)

    async def _install_process(self, task: Task, argv: list[str]) -> int:
        code = await asyncio.to_thread(
            runner.run_install_process, task, argv, self._config.home, self._emitter(task)
        )
        task.raise_if_cancelled()
        return code

    async def _run_install(self, task: Task) -> None:
        job_type = task.request["job_type"]
        narrator_engine = task.request.get("narrator_engine")
        task.raise_if_cancelled()
        argv = runner.install_argv(job_type, narrator_engine)
        self._step(task, install_label(job_type, narrator_engine), 1, 2)
        code = await self._install_process(task, argv)
        if code != 0:
            raise ApiError(
                500,
                "install_failed",
                task.reason_first()
                + f"`{' '.join(argv)}` exited {code}. Its output is on this task's "
                "event stream, line by line, and in the server's log; the env "
                "that was built is left on disk (R6) so a re-run does not start "
                "again from nothing",
            )
        self._reload_step(task, index=2, total=2)

    def _reload_step(self, task: Task, *, index: int, total: int) -> None:
        job_types = (
            self._take_up()
            if task.on_submit and self._take_up is not None
            else self._reload()
        )
        self._step(task, "reload", index, total, job_types=job_types)

    async def _run_module(self, task: Task) -> None:
        entries = validate_module(self._config, self._backend, task.request["module"])
        installs = any(entry.job_type is not None for entry in entries)
        total = len(entries) + (1 if installs else 0)
        installed_anything = False
        unmet: list[dict[str, str]] = []
        for index, entry in enumerate(entries, start=1):
            task.raise_if_cancelled()
            if entry.capability_class is not None:
                row = await self._need_step(task, entry, index, total)
                if row is not None:
                    unmet.append(row)
            elif entry.job_type is not None:
                installed_anything |= await self._install_entry_step(task, entry, index, total)
            else:
                await self._subject_step(task, entry, index, total)
        if installs:
            self._module_reload(task, installed_anything, index=total, total=total)
        task.unmet = unmet

    async def _need_step(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> dict[str, str] | None:
        assert entry.capability_class is not None
        resolved = self._resolve_need(task, entry, index, total)
        if resolved is None:
            return self._unmet_row(entry.capability_class)
        if resolved.installed() is not None:
            self._skip(
                task,
                f"{entry.name}: this card selected {resolved.id!r} and it is "
                "already installed",
            )
            return None
        await self._pull(task, resolved)
        return None

    async def _install_entry_step(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> bool:
        assert entry.job_type is not None
        self._step(task, entry.name, index, total)
        if tasks.env_installed(
            self._config, self._backend, entry.job_type, entry.narrator_engine
        ):
            self._skip(task, f"{entry.name}: this server already has that env. {ALREADY_TRUE}")
            return False
        argv = runner.install_argv(entry.job_type, entry.narrator_engine)
        code = await self._install_process(task, argv)
        if code != 0:
            raise ApiError(
                500,
                "install_failed",
                task.reason_first()
                + f"step {index} of {total} ({entry.name}) exited {code}. The "
                "module stops here and every step before it STAYS — the envs "
                "and weights are on disk (R6). Re-posting the module skips what "
                "is already installed and resumes at this step",
            )
        return True

    async def _subject_step(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> None:
        assert entry.kind is not None and entry.subject_id is not None
        subject = catalog.find(self._config, self._backend, entry.kind, entry.subject_id)
        assert subject is not None
        self._step(task, entry.name, index, total)
        if subject.installed() is not None:
            self._skip(task, f"{entry.name}: already installed. {ALREADY_TRUE}")
            return
        await self._pull(task, subject)

    def _module_reload(
        self, task: Task, installed_anything: bool, *, index: int, total: int
    ) -> None:
        if installed_anything:
            self._reload_step(task, index=index, total=total)
            return
        self._step(task, "reload", index, total)
        self._skip(
            task,
            "reload: every job type this module names was already installed, so "
            "this server's registry is already what the module asks for",
        )

    def _resolve_need(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> "catalog.Subject | None":
        assert entry.capability_class is not None
        self._step(task, entry.name, index, total)
        row = self._capability_row(entry.capability_class)
        if row is None or not row.enabled or row.selected == "":
            return None
        subject = catalog.find(self._config, self._backend, "model", row.selected)
        if subject is None:
            return None
        self.append_event(
            task,
            "progress",
            {"line": f"{entry.capability_class}: this card selected {row.selected}"},
        )
        return subject

    def _capability_row(self, capability_class: str) -> Any:
        record = self._config.capability
        return None if record is None else record.row(capability_class)

    def _unmet_row(self, capability_class: str) -> dict[str, str]:
        row = self._capability_row(capability_class)
        if row is None:
            return {
                "class": capability_class,
                "reason": (
                    "this server has no capability record, so it cannot say "
                    "which model serves this class. Run `crucible capability "
                    "--write` on it"
                ),
            }
        if row.selected != "" and row.enabled:
            return {
                "class": capability_class,
                "reason": (
                    f"this card selected {row.selected!r} and this backend has "
                    "no block for it; the capability record is stale. Run "
                    "`crucible capability --write`"
                ),
            }
        return {"class": capability_class, "reason": row.reason}
