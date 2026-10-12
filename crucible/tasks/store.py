from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .. import catalog, jobflags, tasks, voicecatalog
from ..backend import Backend
from ..clock import utcnow
from ..config import Config, load_config
from ..errors import ApiError
from ..events import TASK, EventHub
from ..updating import UpdateHold
from ..weights import PullCancelled, WeightsError
from . import hostdoor, runner
from .modulefit import ModuleFit, Unfit, enables, unmet_row
from .states import (
    CANCELLED,
    DONE,
    FAILED,
    HISTORY,
    TERMINAL_STATES,
    Task,
    TaskCancelled,
    TaskFailedByHost,
)
from .validate import (
    ModuleEntry,
    install_label,
    validate_module,
    validate_request,
)

ALREADY_TRUE = (
    "A module or an install says what must be true, so what is already true is "
    "skipped rather than refused"
)


@dataclass(frozen=True)
class Outcome:
    """What one module entry left on this server: `present` when what it asks for is
    there now (installed by it or already), `built` when it built an env, and the `unmet`
    rows it adds when it is not."""

    present: bool
    built: bool = False
    unmet: tuple[dict[str, str], ...] = ()


class TaskStore:

    def __init__(
        self,
        config: Config,
        backend: Backend,
        *,
        take_up: Callable[[], list[str]],
        in_use: Callable[[catalog.Subject], dict[str, Any] | None] | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._take_up = take_up
        self._in_use = in_use
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._running_id: str | None = None
        self._runner: asyncio.Task[None] | None = None
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self.events = EventHub()
        # The deploy's hold (crucible/updating.py); the app hands every work owner the same one.
        self.updating = UpdateHold()

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

    def submit(self, request: dict[str, Any]) -> Task:
        """Start a task. What holds the card does not stop one: an install builds an env
        (disk, network, CPU) and ends in the additive take-up (`_reload_step`), which
        adds the types it turned on and replaces no plugin a job is running on, so it
        runs beside whatever the card is doing (Victoria's laptop, 2026-10-11: B-Sides'
        module was refused `server_busy` for as long as her albums rendered)."""
        validate_request(self._config, self._backend, request)
        self.updating.refuse_if_holding(f"a {request['type']} task")
        self.refuse_if_busy()

        now = utcnow()
        task = Task(
            id=uuid.uuid4().hex,
            type=request["type"],
            request=request,
            created=now,
            started=now,
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
        self._publish(task, kind, data)

    def _publish(self, task: Task, kind: str, data: dict[str, Any]) -> None:
        """Tell the server-wide stream: `task.running`, `task.step`, `task.progress`
        (throttled) and, from `_finish`, `task.<state>` (docs/EVENTS.md)."""
        if kind == "started":
            self.events.publish(TASK, f"task.{task.state}", task.to_dict())
        elif kind == "step":
            self.events.publish(TASK, "task.step", {"task_id": task.id, **data})
        elif kind == "progress":
            self.events.publish_throttled(
                TASK, "task.progress", f"task:{task.id}", {"task_id": task.id, **data}
            )

    def _publish_end(self, task: Task) -> None:
        self.events.forget(f"task:{task.id}")
        self.events.publish(TASK, f"task.{task.state}", task.to_dict())

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
        self._publish_end(task)

    async def _run_pull(self, task: Task) -> None:
        task.raise_if_cancelled()
        kind, subject_id = task.request["kind"], task.request["id"]
        if kind == "voice":
            await asyncio.to_thread(voicecatalog.refresh_ref, self._config.home, subject_id)
        subject = catalog.find(self._config, self._backend, kind, subject_id)
        if subject is None:
            raise ApiError(
                404, "unknown_subject", f"no {kind} called {subject_id!r}"
            )
        self._refuse_moving_in_use(subject)
        self._step(task, f"pull {kind} {subject_id}", 1, 1)
        await self._pull(task, subject)

    def _refuse_moving_in_use(self, subject: catalog.Subject) -> None:
        revision = subject.would_move()
        if revision is None or subject.installed() is None or self._in_use is None:
            return
        held = self._in_use(subject)
        if held is None:
            return
        raise ApiError(
            409,
            "subject_in_use",
            f"{subject.kind} {subject.id!r} would move to {revision[:12]}, and "
            f"{held['who']}. Its weights are not swapped underneath it: this "
            "machine stays on the revision it has. Once it is free, submit the "
            f"pull again (or run `{subject.pull_command}`)",
            {**held, "revision": revision},
        )

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
        if tasks.env_installed(self._config, self._backend, job_type, narrator_engine):
            await self._turn_on_built(task, job_type, narrator_engine)
            return
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

    async def _turn_on_built(
        self, task: Task, job_type: str, narrator_engine: str | None
    ) -> None:
        """An install of an env that is already built: no build, the module path's
        `enable <type>` steps for every type the env serves, then the take-up. It is the
        door `job_type_disabled` names for a type that is off, and its env being built
        does not make the type on (owens-pc, 2.0.8: `enable_video` false with the video
        env on disk, and the install the refusal pointed at was refused
        `job_type_installed`). A rebuild stays deliberate: `crucible install --force`
        on the server."""
        entry = ModuleEntry(
            name=install_label(job_type, narrator_engine),
            job_type=job_type,
            narrator_engine=narrator_engine,
        )
        fit = await asyncio.to_thread(ModuleFit, self._config, self._backend)
        wanted = enables(self._config, self._backend, [entry])
        total = 1 + len(wanted) + 1
        self._step(task, entry.name, 1, total)
        engine = "" if narrator_engine is None else f" --narrator-engine {narrator_engine}"
        self._skip(
            task,
            f"{entry.name}: this server already has that env, so nothing is built "
            f"(`crucible install {job_type}{engine} --force` on the server rebuilds "
            f"it). {ALREADY_TRUE}",
        )
        await self._enable_and_reload(
            task, [entry], [Outcome(present=True)], fit, wanted, total=total
        )

    def _reload_step(self, task: Task, *, index: int, total: int) -> None:
        self._step(task, "reload", index, total, job_types=self._take_up())

    async def _run_module(self, task: Task) -> None:
        entries = validate_module(self._config, self._backend, task.request["module"])
        fit = await asyncio.to_thread(ModuleFit, self._config, self._backend)
        wanted = enables(self._config, self._backend, entries)
        total = len(entries) + len(wanted) + 1
        outcomes: list[Outcome] = []
        for index, entry in enumerate(entries, start=1):
            task.raise_if_cancelled()
            if entry.capability_class is not None:
                outcome = await self._need_step(task, entry, index, total)
            elif entry.job_type is not None:
                outcome = await self._install_entry_step(task, entry, fit, index, total)
            else:
                outcome = await self._subject_step(task, entry, fit, index, total)
            outcomes.append(outcome)
        await self._enable_and_reload(task, entries, outcomes, fit, wanted, total=total)

    async def _enable_and_reload(
        self,
        task: Task,
        entries: list[ModuleEntry],
        outcomes: list[Outcome],
        fit: ModuleFit,
        wanted: dict[str, list[int]],
        *,
        total: int,
    ) -> None:
        """The tail a module and an install of a built env share: an `enable <type>` step
        for each type the entries are for, then the take-up, and the task's `unmet`."""
        unmet = [row for outcome in outcomes for row in outcome.unmet]
        turned_on = False
        for index, (job_type, asking) in enumerate(wanted.items(), start=len(entries) + 1):
            task.raise_if_cancelled()
            present = [entries[at] for at in asking if outcomes[at].present]
            on, rows = await self._enable_step(
                task, job_type, present, fit, index=index, total=total
            )
            turned_on |= on
            unmet.extend(rows)
        built = any(outcome.built for outcome in outcomes)
        self._module_reload(task, built or turned_on, index=total, total=total)
        task.unmet = unmet

    async def _need_step(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> Outcome:
        assert entry.capability_class is not None
        resolved = self._resolve_need(task, entry, index, total)
        if resolved is None:
            return Outcome(present=False, unmet=(self._unmet_row(entry),))
        if resolved.installed() is not None:
            self._skip(
                task,
                f"{entry.name}: this card selected {resolved.id!r} and it is "
                "already installed",
            )
            return Outcome(present=True)
        await self._pull(task, resolved)
        return Outcome(present=True)

    async def _install_entry_step(
        self, task: Task, entry: ModuleEntry, fit: ModuleFit, index: int, total: int
    ) -> Outcome:
        assert entry.job_type is not None
        self._step(task, entry.name, index, total)
        if tasks.env_installed(
            self._config, self._backend, entry.job_type, entry.narrator_engine
        ):
            self._skip(task, f"{entry.name}: this server already has that env. {ALREADY_TRUE}")
            return Outcome(present=True)
        unfit = fit.install(entry)
        if unfit is not None:
            return self._unfit(task, entry, unfit)
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
        return Outcome(present=True, built=True)

    async def _subject_step(
        self, task: Task, entry: ModuleEntry, fit: ModuleFit, index: int, total: int
    ) -> Outcome:
        assert entry.kind is not None and entry.subject_id is not None
        subject = catalog.find(self._config, self._backend, entry.kind, entry.subject_id)
        assert subject is not None
        self._step(task, entry.name, index, total)
        if subject.installed() is not None:
            self._skip(task, f"{entry.name}: already installed. {ALREADY_TRUE}")
            return Outcome(present=True)
        unfit = fit.subject(entry, subject)
        if unfit is not None:
            return self._unfit(task, entry, unfit)
        await self._pull(task, subject)
        return Outcome(present=True)

    def _unfit(self, task: Task, entry: ModuleEntry, unfit: Unfit) -> Outcome:
        """An entry this card cannot run: skipped by name, the rest of the module goes on
        (Owen, 2026-10-11: "crucible should intelligently know which things will fit on
        the gpu and which wont")."""
        self._skip(task, f"{entry.name}: {unfit.words}")
        return Outcome(present=False, unmet=unfit.rows)

    async def _enable_step(
        self,
        task: Task,
        job_type: str,
        present: list[ModuleEntry],
        fit: ModuleFit,
        *,
        index: int,
        total: int,
    ) -> tuple[bool, list[dict[str, str]]]:
        """Turn on the job type that serves what the module put here, through the door
        `PUT /v1/settings/jobs/{type}` and `crucible jobs enable` share, so a module POST
        leaves the server ready to serve: a `denoise` separator pulled into an `rvc` env
        that was already built is served by `denoise`, which no install was asked to turn
        on (Victoria's laptop, 2026-10-11: B-Sides had to turn it on itself)."""
        name = f"enable {job_type}"
        flag = jobflags.flag(job_type)
        self._step(task, name, index, total)
        if not present:
            self._skip(
                task,
                f"{name}: nothing this module asks of it is on this server (see "
                "the steps before), so there is nothing for it to serve",
            )
            return False, []
        self._config.adopt(await asyncio.to_thread(load_config, self._config.home))
        if getattr(self._config, flag):
            self._skip(task, f"{name}: [jobs] {flag} is already on. {ALREADY_TRUE}")
            return False, []
        try:
            await asyncio.to_thread(
                jobflags.set_enabled, self._config, self._backend, job_type, True
            )
        except jobflags.EnableRefused as exc:
            self._skip(task, f"{name}: {exc.code}: {exc.sentence}")
            rows = {
                capability_class: unmet_row(capability_class, exc.sentence, name)
                for entry in present
                for capability_class in fit.classes_for(entry, job_type)
            }
            return False, list(rows.values())
        self.append_event(task, "progress", {"line": f"[jobs] {flag} = true"})
        return True, []

    def _module_reload(
        self, task: Task, changed: bool, *, index: int, total: int
    ) -> None:
        if changed:
            self._reload_step(task, index=index, total=total)
            return
        self._step(task, "reload", index, total)
        self._skip(
            task,
            "reload: nothing this task asks for changed what this server offers "
            "(every env was already built and every job type already on), so its "
            "registry is already what it asks for",
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

    def _unmet_row(self, entry: ModuleEntry) -> dict[str, str]:
        capability_class = entry.capability_class
        assert capability_class is not None
        row = self._capability_row(capability_class)
        if row is None:
            return unmet_row(
                capability_class,
                "this server has no capability record, so it cannot say which "
                "model serves this class. Run `crucible capability --write` on it",
                entry.name,
            )
        if row.selected != "" and row.enabled:
            return unmet_row(
                capability_class,
                f"this card selected {row.selected!r} and this backend has no block "
                "for it; the capability record is stale. Run `crucible capability "
                "--write`",
                entry.name,
            )
        return unmet_row(capability_class, row.reason, entry.name)
