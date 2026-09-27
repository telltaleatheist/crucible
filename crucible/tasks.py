from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import capability, catalog, interpreter, jobenv
from .backend import LLAMA_WINDOWS, Backend
from .config import Config
from .errors import ApiError, CrucibleError
from .hosttools import searched_note, which
from .jobs.base import utcnow
from .jobs.llm import llm_engine_status
from .settle import Held
from .voices import NARRATOR_ENGINE_SAMPLING
from .weights import PullCancelled, WeightsError

RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED})

TASK_TYPES: tuple[str, ...] = (
    "pull",
    "install",
    "module",
    "engine",
    "engine-restart",
)

ENGINE_TARGETS: tuple[str, ...] = ("wsl",)

HOST_DOOR_ENV = "CRUCIBLE_HOST_DOOR"

HOST_DOOR_PATH = "/install"

HOST_DOOR_RESTART_PATH = "/restart"

ENGINE_RESTART_NEEDS_ORCHESTRATOR = "engine_restart_needs_orchestrator"

HOST_UNREACHABLE = "host_unreachable"
HOST_INSTALL_FAILED = "host_install_failed"

HOST_DOOR_CONNECT_SECONDS = 30.0

HISTORY = 50

PROGRESS_INTERVAL_SECONDS = 0.5

TERMINATE_GRACE_SECONDS = 10.0


class TaskCancelled(CrucibleError):
    ...


class TaskFailedByHost(CrucibleError):

    def __init__(self, task: "Task") -> None:
        super().__init__(f"task {task.id} failed on the host")


def _host_refusal_code(body: str) -> str:
    try:
        payload = json.loads(body)
        code = payload["error"]["code"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "engine_move_needs_host"
    return str(code) if isinstance(code, str) and code else "engine_move_needs_host"


class ReloadRefused(CrucibleError):

    def __init__(self, held: Held) -> None:
        super().__init__(str(held))
        self.held = held


@dataclass
class Task:

    id: str
    type: str
    request: dict[str, Any]
    created: str
    started: str
    state: str = RUNNING
    finished: str | None = None
    error: dict[str, str] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    unmet: list[dict[str, str]] = field(default_factory=list)
    cancel_requested: bool = False
    on_submit: bool = False
    reason: str | None = None
    describe: Callable[["Task"], str | None] | None = None
    process: subprocess.Popen[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.id,
            "type": self.type,
            "request": self.request,
            "state": self.state,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "unmet": self.unmet,
            "message": None if self.describe is None else self.describe(self),
        }


def require_installable(job_type: str) -> None:
    from .cli import INSTALLABLE_JOB_TYPES, INSTALLER_FOR

    if job_type in INSTALLABLE_JOB_TYPES:
        return
    installer = INSTALLER_FOR.get(job_type)
    if installer is not None:
        raise ApiError(
            400,
            "unknown_job_type",
            f"job type {job_type!r} has no installer of its own: it shares "
            f"{installer!r}'s env, so installing {installer!r} is what builds "
            f"it. Name {installer!r} instead",
            {"job_type": job_type, "installed_by": installer},
        )
    raise ApiError(
        400,
        "unknown_job_type",
        f"there is no installer for job type {job_type!r}; this build "
        f"installs {sorted(INSTALLABLE_JOB_TYPES)}",
    )


def require_narrator_engine(job_type: str, narrator_engine: str | None) -> None:
    if job_type == "tts":
        if narrator_engine is None:
            raise ApiError(
                400,
                "narrator_engine_required",
                "installing 'tts' needs narrator_engine: on cuda-linux two "
                "engines cannot share a venv, so there is one env per engine "
                f"and no default. This build knows "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}",
            )
        if narrator_engine not in NARRATOR_ENGINE_SAMPLING:
            raise ApiError(
                400,
                "narrator_engine_required",
                f"{narrator_engine!r} is not one of narrator's engines; they are "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}",
            )
        return
    if narrator_engine is not None:
        raise ApiError(
            400,
            "narrator_engine_refused",
            f"narrator_engine names which tts env to build and means nothing for "
            f"{job_type!r}, which has exactly one env per host",
        )


def env_installed(config: Config, backend: Backend, job_type: str, engine: str | None) -> bool:
    try:
        if job_type == "llm":
            return llm_engine_status(config, backend).installed
        if job_type in jobenv.WORKER_JOB_TYPES:
            spec = jobenv.worker_env(job_type, backend.kind)
        else:
            spec = jobenv.tts_env(engine or "", backend.kind)
        return jobenv.env_status(config.home, spec, backend.kind).installed
    except jobenv.EnvError:
        return False


def _validate_pull(config: Config, backend: Backend, kind: str, subject_id: str) -> None:
    if kind not in catalog.KINDS:
        raise ApiError(
            404,
            "unknown_subject",
            f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
            {"kind": kind, "id": subject_id},
        )
    subject = catalog.find(config, backend, kind, subject_id)
    if subject is None:
        raise ApiError(
            404,
            "unknown_subject",
            f"this server has no {kind} called {subject_id!r} for "
            f"{backend.kind}. GET /v1/catalog lists every subject it can hold",
            {"kind": kind, "id": subject_id},
        )
    if subject.installed() is not None:
        raise ApiError(
            409,
            "already_installed",
            f"{kind} {subject_id!r} is already installed on this server. A pull "
            f"of an installed subject is refused rather than skipped; to replace "
            f"it deliberately, run `{subject.pull_command} --force` on the server",
            {"kind": kind, "id": subject_id},
        )


def _validate_engine(backend: Backend, target: str) -> str:
    if target not in ENGINE_TARGETS:
        raise ApiError(
            400,
            "engine_target_unknown",
            f"{target!r} is not an engine this build moves to; the targets are "
            f"{list(ENGINE_TARGETS)}. Moving BACK to Windows is an explicit "
            "operator act (PHASE15-HOST.md section 6) and is refused rather "
            "than half-done",
            {"target": target, "targets": list(ENGINE_TARGETS)},
        )
    if backend.kind != LLAMA_WINDOWS:
        raise ApiError(
            409,
            "engine_move_not_here",
            f"this server runs the {backend.kind} backend on "
            f"{backend.platform}, and the engine move is a Windows machine "
            "swapping llama.cpp for the WSL2 guest. There is nothing here to "
            "move from",
            {"backend": backend.kind, "platform": backend.platform},
        )
    door = os.environ.get(HOST_DOOR_ENV, "").strip()
    if door == "":
        raise ApiError(
            409,
            "engine_move_needs_host",
            "this server was not started by `crucible orchestrator`, so there is "
            f"nothing to hand the move to (${HOST_DOOR_ENV} is not set). Only "
            "the host can run wsl.exe, prompt for administrator and survive "
            "the reboot the move may need — a server doing it itself would "
            "stop halfway through and take its own event stream with it. "
            "Start the host and press it again from the page",
            {"env": HOST_DOOR_ENV},
        )
    return door.rstrip("/")


def _validate_engine_restart() -> str:
    door = os.environ.get(HOST_DOOR_ENV, "").strip()
    if door == "":
        raise ApiError(
            409,
            ENGINE_RESTART_NEEDS_ORCHESTRATOR,
            "this server was not started by an orchestrator "
            f"(${HOST_DOOR_ENV} is not set), so there is nothing here that "
            "can restart it. Only the orchestrator can run wsl.exe, name the "
            "guest's unit or respawn a child — a server restarting itself "
            "would take its own event stream with it and leave nobody to say "
            "whether it came back. Start `crucible orchestrator` and press it "
            "again from the page",
            {"env": HOST_DOOR_ENV},
        )
    return door.rstrip("/")


def _validate_install(
    config: Config, backend: Backend, job_type: str, narrator_engine: str | None
) -> None:
    require_installable(job_type)
    require_narrator_engine(job_type, narrator_engine)
    if env_installed(config, backend, job_type, narrator_engine):
        raise ApiError(
            409,
            "job_type_installed",
            f"job type {job_type!r}"
            + (f" ({narrator_engine})" if narrator_engine else "")
            + " already has its env on this server. To rebuild it deliberately, "
            "run `crucible install` with --force on the server",
            {"job_type": job_type, "narrator_engine": narrator_engine},
        )


@dataclass(frozen=True)
class ModuleEntry:

    name: str
    job_type: str | None = None
    narrator_engine: str | None = None
    kind: str | None = None
    subject_id: str | None = None
    capability_class: str | None = None


def validate_module(
    config: Config, backend: Backend, module: Any
) -> list[ModuleEntry]:
    problems: list[str] = []

    if not isinstance(module, dict):
        raise ApiError(
            400,
            "invalid_module",
            f"a module is a JSON object, got {type(module).__name__}",
        )
    for key in ("name", "version"):
        if not isinstance(module.get(key), str) or module[key].strip() == "":
            problems.append(f"{key}: a module needs a non-empty string {key}")
    unknown = sorted(
        set(module) - {"name", "version", "job_types", "needs", "subjects"}
    )
    if unknown:
        problems.append(
            f"unknown key(s) {unknown}; a module carries exactly name, version, "
            "job_types, needs and subjects"
        )

    entries: list[ModuleEntry] = []
    raw_types = module.get("job_types", [])
    if not isinstance(raw_types, list):
        problems.append("job_types: must be a list")
        raw_types = []
    for index, raw in enumerate(raw_types):
        where = f"job_types[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object with a `type`")
            continue
        stray = sorted(set(raw) - {"type", "narrator_engine"})
        if stray:
            problems.append(f"{where}: unknown key(s) {stray}")
            continue
        job_type = raw.get("type")
        engine = raw.get("narrator_engine")
        if not isinstance(job_type, str):
            problems.append(f"{where}: `type` must be a string")
            continue
        if engine is not None and not isinstance(engine, str):
            problems.append(f"{where}: `narrator_engine` must be a string")
            continue
        try:
            require_installable(job_type)
            require_narrator_engine(job_type, engine)
        except ApiError as exc:
            problems.append(f"{where}: {exc.message}")
            continue
        entries.append(
            ModuleEntry(
                name=f"install {job_type}"
                + (f" ({engine})" if engine else ""),
                job_type=job_type,
                narrator_engine=engine,
            )
        )

    raw_needs = module.get("needs", [])
    if not isinstance(raw_needs, list):
        problems.append("needs: must be a list")
        raw_needs = []
    for index, raw in enumerate(raw_needs):
        where = f"needs[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object with a `class`")
            continue
        stray = sorted(set(raw) - {"class"})
        if stray:
            problems.append(
                f"{where}: unknown key(s) {stray}. A need is a CLASS and nothing "
                "else; an app that wants one specific model names it under "
                "`subjects`, which is a choice and says so"
            )
            continue
        capability_class = raw.get("class")
        if not isinstance(capability_class, str):
            problems.append(f"{where}: `class` must be a string")
            continue
        if capability_class not in capability.BY_NAME:
            problems.append(
                f"{where}: {capability_class!r} is not a capability class; they "
                f"are {sorted(capability.BY_NAME)}"
            )
            continue
        entries.append(
            ModuleEntry(
                name=f"resolve {capability_class}",
                capability_class=capability_class,
            )
        )

    raw_subjects = module.get("subjects", [])
    if not isinstance(raw_subjects, list):
        problems.append("subjects: must be a list")
        raw_subjects = []
    for index, raw in enumerate(raw_subjects):
        where = f"subjects[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object with `kind` and `id`")
            continue
        stray = sorted(set(raw) - {"kind", "id"})
        if stray:
            problems.append(f"{where}: unknown key(s) {stray}")
            continue
        kind, subject_id = raw.get("kind"), raw.get("id")
        if not isinstance(kind, str) or not isinstance(subject_id, str):
            problems.append(f"{where}: `kind` and `id` must both be strings")
            continue
        if kind not in catalog.KINDS:
            problems.append(
                f"{where}: {kind!r} is not a subject kind; they are "
                f"{list(catalog.KINDS)}"
            )
            continue
        if catalog.find(config, backend, kind, subject_id) is None:
            problems.append(
                f"{where}: this server has no {kind} called {subject_id!r} for "
                f"{backend.kind}"
            )
            continue
        entries.append(
            ModuleEntry(
                name=f"pull {kind} {subject_id}", kind=kind, subject_id=subject_id
            )
        )

    if not entries and not problems:
        problems.append(
            "a module with no job_types and no subjects asks for nothing; if that "
            "is what this app needs, it needs no module"
        )
    if problems:
        raise ApiError(
            400,
            "invalid_module",
            "this module was not run because "
            + (
                "it has a problem: " if len(problems) == 1 else
                f"it has {len(problems)} problems: "
            )
            + "; ".join(problems),
            {"problems": problems},
        )
    return entries


def install_command() -> str:
    sibling = Path(sys.executable).resolve().parent / "crucible"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = which("crucible")
    if found is not None:
        return found
    raise ApiError(
        503,
        "install_command_missing",
        f"this server cannot install anything: there is no `crucible` console "
        f"script at {sibling} and none on PATH {searched_note()}. It is the "
        "script `pip install crucible` writes beside the interpreter this "
        "server runs on",
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
        runner, self._runner = self._runner, None
        if runner is None:
            return
        running = self.running
        if running is not None:
            self._request_cancel(running)
        runner.cancel()
        try:
            await runner
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
        task_type = request["type"]
        if task_type == "pull":
            _validate_pull(self._config, self._backend, request["kind"], request["id"])
        elif task_type == "install":
            _validate_install(
                self._config,
                self._backend,
                request["job_type"],
                request.get("narrator_engine"),
            )
        elif task_type == "module":
            validate_module(self._config, self._backend, request["module"])
        elif task_type == "engine":
            _validate_engine(self._backend, request["target"])
        elif task_type == "engine-restart":
            _validate_engine_restart()
        else:
            raise ApiError(
                400,
                "invalid_request",
                f"{task_type!r} is not a task type; they are {list(TASK_TYPES)}",
            )

        self.refuse_if_busy()
        if _touches_the_registry(task_type, request) and not on_submit:
            self.refuse_if_the_card_is_held()

        now = utcnow()
        task = Task(
            id=uuid.uuid4().hex,
            type=task_type,
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


    async def _run(self, task: Task) -> None:
        try:
            self.append_event(task, "started", {"type": task.type})
            if task.type == "pull":
                await self._run_pull(task)
            elif task.type == "install":
                await self._run_install(task)
            elif task.type == "engine":
                await self._run_engine(task)
            elif task.type == "engine-restart":
                await self._run_engine_restart(task)
            else:
                await self._run_module(task)
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

    def _raise_if_cancelled(self, task: Task) -> None:
        if task.cancel_requested:
            raise TaskCancelled(f"task {task.id} was cancelled")


    async def _run_pull(self, task: Task) -> None:
        await self._pull_one(
            task, task.request["kind"], task.request["id"], index=1, total=1
        )

    async def _pull_one(
        self, task: Task, kind: str, subject_id: str, *, index: int, total: int
    ) -> None:
        self._raise_if_cancelled(task)
        subject = catalog.find(self._config, self._backend, kind, subject_id)
        if subject is None:
            raise ApiError(
                404, "unknown_subject", f"no {kind} called {subject_id!r}"
            )
        self.append_event(
            task,
            "step",
            {"name": f"pull {kind} {subject_id}", "index": index, "total": total},
        )
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
        last = 0.0

        def on_progress(done: int, total: int | None, name: str) -> None:
            if task.cancel_requested:
                raise PullCancelled(f"task {task.id} was cancelled")
            nonlocal last
            now = time.monotonic()
            if now - last < PROGRESS_INTERVAL_SECONDS:
                return
            last = now
            self._from_thread(
                task,
                "progress",
                {"bytes_done": done, "bytes_total": total, "file": name},
            )

        def on_line(line: str) -> None:
            print(f"crucible: task {task.id}: {line}", file=sys.stderr)

        subject.pull(force=False, on_line=on_line, on_progress=on_progress)


    async def _run_engine(self, task: Task) -> None:
        door = _validate_engine(self._backend, task.request["target"])
        self.append_event(
            task,
            "step",
            {"name": "hand the move to the host", "index": 1, "total": 1},
        )
        terminal = await asyncio.to_thread(
            self._relay_blocking, task, door, HOST_DOOR_PATH, {"target": task.request["target"]}
        )
        if terminal is None:
            raise ApiError(
                502,
                HOST_INSTALL_FAILED,
                f"the host's door at {door}{HOST_DOOR_PATH} closed its stream "
                "without saying whether the move finished. Nothing here can "
                "tell a completed install from an abandoned one, so it is "
                "reported as a failure; the host's log says what happened",
                {"door": door},
            )
        if terminal == "failed":
            raise TaskFailedByHost(task)

    def _relay_blocking(
        self, task: Task, door: str, path: str, body_fields: dict[str, Any]
    ) -> str | None:
        import urllib.error
        import urllib.request

        body = json.dumps(body_fields).encode("utf-8")
        request = urllib.request.Request(
            f"{door}{path}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.token}",
            },
        )
        terminal: str | None = None
        try:
            with urllib.request.urlopen(
                request, timeout=HOST_DOOR_CONNECT_SECONDS
            ) as stream:
                for raw in stream:
                    if task.cancel_requested:
                        raise TaskCancelled(f"task {task.id} was cancelled")
                    line = raw.decode("utf-8", "replace").strip()
                    if line == "":
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        self._from_thread(task, "progress", {"line": line})
                        continue
                    name = str(event.get("event") or "progress")
                    data = event.get("data")
                    self._from_thread(
                        task, name, data if isinstance(data, dict) else {}
                    )
                    if name in ("done", "failed", "cancelled"):
                        terminal = name
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ApiError(
                502 if exc.code >= 500 else 409,
                _host_refusal_code(detail),
                f"the host refused the move with HTTP {exc.code}: {detail}",
                {"door": door, "host_status": exc.code},
            ) from None
        except (urllib.error.URLError, OSError) as exc:
            raise ApiError(
                502,
                HOST_UNREACHABLE,
                f"the orchestrator's door at {door}{path} did not answer: "
                f"{type(exc).__name__}: {exc}. A host started this server "
                f"(${HOST_DOOR_ENV} is set) and its door is not answering "
                "now. Start it from the Startup item, or run `crucible orchestrator` "
                "from the host runtime, and press it again",
                {"door": door},
            ) from None
        return terminal


    async def _run_engine_restart(self, task: Task) -> None:
        door = _validate_engine_restart()
        self.append_event(
            task,
            "step",
            {"name": "hand the restart to the orchestrator", "index": 1, "total": 1},
        )
        terminal = await asyncio.to_thread(
            self._relay_blocking, task, door, HOST_DOOR_RESTART_PATH, {}
        )
        if terminal is None:
            raise ApiError(
                502,
                HOST_INSTALL_FAILED,
                f"the orchestrator's door at {door}{HOST_DOOR_RESTART_PATH} "
                "closed its stream without saying whether the engine came "
                "back. That is the EXPECTED shape when the engine being "
                "restarted is the one relaying: read `GET /v1/info` for the "
                "answer, which is the only place it is reliably true",
                {"door": door},
            )
        if terminal == "failed":
            raise TaskFailedByHost(task)


    async def _run_install(self, task: Task) -> None:
        await self._install_one(
            task,
            task.request["job_type"],
            task.request.get("narrator_engine"),
            index=1,
            total=2,
        )
        await self._reload_step(task, index=2, total=2)

    async def _install_one(
        self,
        task: Task,
        job_type: str,
        narrator_engine: str | None,
        *,
        index: int,
        total: int,
    ) -> None:
        self._raise_if_cancelled(task)
        command = install_command()
        argv = [command, "install", job_type, "--verbose"]
        if narrator_engine is not None:
            argv += ["--narrator-engine", narrator_engine]
        self.append_event(
            task,
            "step",
            {
                "name": f"install {job_type}"
                + (f" ({narrator_engine})" if narrator_engine else ""),
                "index": index,
                "total": total,
            },
        )
        code = await asyncio.to_thread(self._run_install_process, task, argv)
        self._raise_if_cancelled(task)
        if code != 0:
            raise ApiError(
                500,
                "install_failed",
                _reason_first(task)
                + f"`{' '.join(argv)}` exited {code}. Its output is on this task's "
                "event stream, line by line, and in the server's log; the env "
                "that was built is left on disk (R6) so a re-run does not start "
                "again from nothing",
            )

    def _run_install_process(self, task: Task, argv: list[str]) -> int:
        environment = dict(os.environ)
        environment["CRUCIBLE_HOME"] = str(self._config.home)
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        task.process = process
        assert process.stdout is not None
        last_bytes = 0.0
        try:
            for line in process.stdout:
                stripped = line.rstrip("\n")
                if task.cancel_requested and process.poll() is None:
                    process.terminate()
                if stripped.startswith("crucible: "):
                    task.reason = stripped[len("crucible: "):].strip() or None
                measured = interpreter.parse_progress_line(stripped)
                if measured is not None:
                    now = time.monotonic()
                    if now - last_bytes < PROGRESS_INTERVAL_SECONDS:
                        continue
                    last_bytes = now
                    self._from_thread(task, "progress", measured)
                    continue
                self._from_thread(task, "progress", {"line": stripped})
            code = process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
        finally:
            task.process = None
        return code

    async def _reload_step(self, task: Task, *, index: int, total: int) -> None:
        job_types = (
            self._take_up()
            if task.on_submit and self._take_up is not None
            else self._reload()
        )
        self.append_event(
            task,
            "step",
            {
                "name": "reload",
                "index": index,
                "total": total,
                "job_types": job_types,
            },
        )


    async def _run_module(self, task: Task) -> None:
        entries = validate_module(self._config, self._backend, task.request["module"])
        total = len(entries)
        installs = [entry for entry in entries if entry.job_type is not None]
        if installs:
            total += 1
        index = 0
        installed_anything = False
        unmet: list[dict[str, str]] = []
        for entry in entries:
            index += 1
            self._raise_if_cancelled(task)
            if entry.capability_class is not None:
                resolved = self._resolve_need(task, entry, index, total)
                if resolved is None:
                    unmet.append(self._unmet_row(entry.capability_class))
                    continue
                if resolved.installed() is not None:
                    self.append_event(
                        task,
                        "skipped",
                        {
                            "reason": f"{entry.name}: this card selected "
                            f"{resolved.id!r} and it is already installed"
                        },
                    )
                    continue
                await self._pull(task, resolved)
                continue
            if entry.job_type is not None:
                self.append_event(
                    task,
                    "step",
                    {"name": entry.name, "index": index, "total": total},
                )
                if env_installed(
                    self._config, self._backend, entry.job_type, entry.narrator_engine
                ):
                    self.append_event(
                        task,
                        "skipped",
                        {
                            "reason": f"{entry.name}: this server already has that "
                            "env. A module says what must be true, so an entry "
                            "that is already true is skipped rather than refused"
                        },
                    )
                    continue
                await self._install_step(task, entry, index, total)
                installed_anything = True
                continue

            assert entry.kind is not None and entry.subject_id is not None
            subject = catalog.find(
                self._config, self._backend, entry.kind, entry.subject_id
            )
            assert subject is not None
            self.append_event(
                task, "step", {"name": entry.name, "index": index, "total": total}
            )
            if subject.installed() is not None:
                self.append_event(
                    task,
                    "skipped",
                    {
                        "reason": f"{entry.name}: already installed. A module says "
                        "what must be true, so an entry that is already true is "
                        "skipped rather than refused"
                    },
                )
                continue
            await self._pull(task, subject)

        if installs:
            index += 1
            if installed_anything:
                await self._reload_step(task, index=index, total=total)
            else:
                self.append_event(
                    task, "step", {"name": "reload", "index": index, "total": total}
                )
                self.append_event(
                    task,
                    "skipped",
                    {
                        "reason": "reload: every job type this module names was "
                        "already installed, so this server's registry is already "
                        "what the module asks for"
                    },
                )
        task.unmet = unmet

    def _resolve_need(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> "catalog.Subject | None":
        assert entry.capability_class is not None
        self.append_event(
            task, "step", {"name": entry.name, "index": index, "total": total}
        )
        row = self._capability_row(entry.capability_class)
        if row is None or not row.enabled or row.selected == "":
            return None
        subject = catalog.find(
            self._config, self._backend, "model", row.selected
        )
        if subject is None:
            return None
        self.append_event(
            task,
            "progress",
            {
                "line": f"{entry.capability_class}: this card selected "
                f"{row.selected}"
            },
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

    async def _install_step(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> None:
        assert entry.job_type is not None
        command = install_command()
        argv = [command, "install", entry.job_type, "--verbose"]
        if entry.narrator_engine is not None:
            argv += ["--narrator-engine", entry.narrator_engine]
        code = await asyncio.to_thread(self._run_install_process, task, argv)
        self._raise_if_cancelled(task)
        if code != 0:
            raise ApiError(
                500,
                "install_failed",
                _reason_first(task)
                + f"step {index} of {total} ({entry.name}) exited {code}. The "
                "module stops here and every step before it STAYS — the envs "
                "and weights are on disk (R6). Re-posting the module skips what "
                "is already installed and resumes at this step",
            )


def _reason_first(task: Task) -> str:
    return "" if task.reason is None else f"{task.reason}. "


def _touches_the_registry(task_type: str, request: dict[str, Any]) -> bool:
    if task_type == "install":
        return True
    if task_type != "module":
        return False
    module = request.get("module")
    if not isinstance(module, dict):
        return False
    declared = module.get("job_types")
    return isinstance(declared, list) and len(declared) > 0


def module_document(entries: Iterable[ModuleEntry]) -> list[str]:
    return [entry.name for entry in entries]


__all__ = [
    "CANCELLED",
    "DONE",
    "FAILED",
    "HISTORY",
    "RUNNING",
    "ENGINE_RESTART_NEEDS_ORCHESTRATOR",
    "HOST_DOOR_RESTART_PATH",
    "TASK_TYPES",
    "TERMINAL_STATES",
    "ModuleEntry",
    "ReloadRefused",
    "Task",
    "TaskCancelled",
    "TaskStore",
    "env_installed",
    "install_command",
    "require_installable",
    "require_narrator_engine",
    "module_document",
    "validate_module",
]
