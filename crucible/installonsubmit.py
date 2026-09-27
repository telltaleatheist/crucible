"""Install on submit: a job for a type this card can run waits for its install.

OWEN'S RULING, 2026-09-26: *"yes, we need to install a missing environment when
a job is submitted"*, under his standing rule that Crucible must be idiot proof:
assume the caller has no idea an "environment" exists. Fresh-install snag #2
(`docs/FRESH-INSTALL-KYLIES-2026-09-26.md`): kylies-pc's rvc env, its base
assets and its model were three commands typed inside the guest, and before
this a job for `rvc` there was refused `job_type_disabled` with an install
request for the client to send (`docs/PROPOSAL-INSTALL-ON-SUBMIT.md`).

WHAT HAPPENS NOW. `POST /v1/jobs` for a type whose refusal is `not_installed`
(the capability record says this card can hold it and nothing has installed
it) is ACCEPTED with a 202. The server starts the same install the operator
page's Install button runs, as a `module` task, so one task carries the env,
the base weights the type needs to run at all (`rvc`'s base assets) and the
voice or model the job names when it is declared and not yet here. The job is
`queued` and waits: `waiting_for` on `GET /v1/jobs/{id}` and a `waiting` event
on its stream say in words what it is waiting for, from the task's own steps
and progress. When the task ends the type is taken up and the job goes on the
lane. If the task fails, the job fails with the install's own one-line reason.

WHAT IS STILL REFUSED AT ONCE, and must be: a type this card CANNOT serve
(`cannot_hold`, or the live install plan says nothing it offers runs here); a
server whose card was never measured (`undecided`: nothing knows whether it
would run, and "never install something that can't run" is the other half of
the ruling); a type with no installer (`echo`); an env already on disk with
its flag off (installing again would change nothing); a model this build does
not declare. `[jobs] install_on_submit = false` puts the old refusal back, for
an operator who wants every install to be his own act.

THE INSTALL MODAL, AND WHY A JOB DOES NOT GET ONE. The Crucible UI asks before
it installs: `GET /v1/capability/plan` (`capability.install_plan`) is what the
modal shows, the card by its name and what it will run at what precision. A job
submitted by an API client has nobody to show a modal to, so the plan's
sentences go into the job's status (`waiting_for.plan`) instead: the caller who
did not know an environment existed can still read what it is getting.

WHY THE JOB IS PARKED OFF THE LANE. A job waiting minutes for pip holds no card,
so it is not on the lane (`JobStore.park`) and the settlement does not count it:
the lane stays free for work that can run now, and an install of this kind is
not gated on the four facts because it ends by TAKING UP the new type (adding
plugins, replacing none, #42) rather than swapping the registry
(`TaskStore.submit`, `on_submit`). When the install lands, the job is admitted
exactly as the door would admit it (the lease, the clearance, `preflight`) and
put on the lane behind whatever is there (`JobStore.enqueue_admitted`): it was
answered 202, and a 202 is a promise.

ONE INSTALL PER ENV. There is one driver and one task lane: every job waiting
is served by the task running now, and the next task is built from what the
jobs still waiting need, with anything already on disk skipped. Two `rvc` jobs
arriving together wait on one install; a third arriving mid-install waits on
the same one, and whatever it needs beyond it (another voice) is the next
task's. A task somebody else started (the operator's own Install) is waited
out, never raced.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import capability as capability_classes
from . import catalog, jobenv, ladder, workerenv
from .backend import Backend
from .config import Config
from .errors import ApiError
from .jobs import ALL_JOB_TYPES
from .jobs.base import QUEUED, Job, utcnow
from .jobs.queue import JobStore
from .tasks import DONE, FAILED, TERMINAL_STATES, Task, TaskStore, env_installed
from .voices import NARRATOR_ENGINE_SAMPLING, load_all_voices

#: The weights a type cannot run AT ALL without, whatever the job names, by
#: capability. `rvc`'s base assets are the engine's (hubert, rmvpe), shared by
#: every voice. Pulled with the install so the job does not fail on
#: `rvc_base_models_missing` the moment its env arrives.
BASE_SUBJECTS: dict[str, tuple[tuple[str, str], ...]] = {
    "rvc": (("rvc-base", catalog.RVC_BASE_ID),),
}

#: Capabilities whose every servable model is a catalog subject, so a model
#: the catalog does not have is a typo and is refused BEFORE anything installs.
#: `llm` and `tts` are not here: a model may be an upstream route, a voice a
#: local directory, and those are the plugin's to judge once it exists.
CATALOG_IS_COMPLETE = frozenset({"rvc", "denoise", "asr", "align"})

#: A released job whose admission refuses for one of these waits and is asked
#: again, rather than failed: each is somebody else holding the card for now
#: (a lease, a streaming session, an engine being cleared), which the door
#: would refuse with and a client would retry. This job cannot retry: it was
#: answered 202.
TRANSIENT_REFUSALS = frozenset(
    {"leased", "engine_in_use", "accelerator_busy", "stream_session_open"}
)

#: How often a job waiting for the card is asked about again.
CARD_RETRY_SECONDS = 5.0

#: How often, at most, a waiting job's status is rewritten from its task.
STATUS_INTERVAL_SECONDS = 1.0

#: A download's pace is not stated until it has run this long: the first
#: seconds of a pull are connection setup, and a remaining time computed from
#: them is a number that moves by minutes.
PACE_AFTER_SECONDS = 3.0


@dataclass(frozen=True)
class Pull:
    """One subject a waiting job needs pulled."""

    kind: str
    id: str
    #: The subject in a person's words: "its base assets", "the RVC voice 'sigma'".
    words: str
    expected_bytes: int | None

    @property
    def step(self) -> str:
        """The module step's name (`TaskStore._run_module`)."""
        return f"pull {self.kind} {self.id}"


@dataclass
class Need:
    """What one waiting job needs before it can go on the lane."""

    job_type: str
    capability: str
    #: The install that builds its env (`INSTALLER_FOR`), and for `tts` which
    #: narrator engine.
    installer: str
    narrator_engine: str | None
    env_bytes: int | None
    pulls: tuple[Pull, ...]
    #: The install plan's sentences for this card, the modal's words.
    plan: str
    since: str = field(default_factory=utcnow)

    @property
    def install_step(self) -> str:
        return f"install {self.installer}" + (
            f" ({self.narrator_engine})" if self.narrator_engine else ""
        )

    def steps(self) -> list[str]:
        return [self.install_step, *(pull.step for pull in self.pulls)]


class InstallOnSubmit:
    """The one driver: parks jobs, runs their installs, releases them."""

    def __init__(
        self,
        config: Config,
        backend: Backend,
        store: JobStore,
        tasks: TaskStore,
        *,
        admit: Callable[[Job], Awaitable[None]],
    ) -> None:
        self._config = config
        self._backend = backend
        self._store = store
        self._tasks = tasks
        #: The door's own admission, for a job whose install has landed: the
        #: take-up, the lease, the clearance, `preflight`, `enqueue_admitted`.
        #: Injected, because it is a statement about how the server is
        #: assembled, which `crucible/api.py` owns.
        self._admit = admit
        self._needs: dict[str, Need] = {}
        self._driver: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        #: (task id, step name) -> (monotonic, bytes) at first sight, for a pace.
        self._pace: dict[tuple[str, str], tuple[float, int]] = {}

    # --------------------------------------------------------------- planning

    def plan(self, job_type: str, model: str | None, refusal: ApiError) -> Need:
        """What installing for this job means, or the refusal it should get instead.

        `refusal` is `disabled_error`'s answer for this type. Only its
        `not_installed` case is installed for; every other case is re-raised as
        it is, because it already says the true thing (the card cannot hold
        it, nothing has measured the card, the type is on and not taken up).
        """
        details = refusal.details or {}
        install = details.get("install")
        if details.get("reason") != "not_installed" or not isinstance(install, dict):
            raise refusal
        if job_type.startswith("unload-"):
            # Nothing of an uninstalled type is on the card to unload, and
            # gigabytes of install to find that out is not a service.
            raise refusal
        capability = ALL_JOB_TYPES[job_type]
        installer = str(install["job_type"])
        engine = self._narrator_engine(installer, model)
        if installer == "tts" and engine is None:
            raise refusal
        if env_installed(self._config, self._backend, installer, engine):
            # On disk, flag off: an install would find nothing to do and the
            # type would stay off. The refusal says what is true.
            raise refusal

        decisions, card, pool = live_decisions(self._config, self._backend)
        total = self._backend.gpu.vram_bytes
        plan = capability_classes.install_plan(
            capability, decisions, card=card, total_bytes=total, pool=pool
        )
        if not plan["usable"]:
            lines = " ".join(row["line"] for row in plan["classes"])
            raise ApiError(
                400,
                "job_type_disabled",
                f"job type {job_type!r} is not installed, and it will not be: "
                f"on this card ({plan['card_words']}) {lines} Installing it "
                "would put a type on this server whose first job fails.",
                {
                    "job_type": job_type,
                    "capability_recorded": True,
                    "fits": [],
                    "reason": "cannot_hold",
                },
            )
        words = [f"Your card ({plan['card_words']}):"]
        words += [f"- {row['line']}" for row in plan["classes"]]

        subjects = catalog.subjects(self._config, self._backend)
        pulls: list[Pull] = []
        for kind, subject_id in BASE_SUBJECTS.get(capability, ()):
            subject = next(
                (s for s in subjects if s.kind == kind and s.id == subject_id), None
            )
            if subject is not None and subject.installed() is None:
                pulls.append(_pull_of(subject))
        if model is None and job_type == capability and capability in CATALOG_IS_COMPLETE:
            offered = sorted(
                s.id
                for s in subjects
                if s.job_type == capability and s.kind not in ("rvc-base", "engine")
            )
            if offered:
                # `resolve_model`'s refusal, made before the install rather
                # than after it: a request that names no model cannot run.
                raise ApiError(
                    400,
                    "model_required",
                    f"job type {job_type!r} requires a model; it offers {offered}. "
                    "Nothing was installed for it",
                )
        if model is not None:
            named = [
                s
                for s in subjects
                if s.id == model
                and s.job_type == capability
                and s.kind not in ("rvc-base", "engine")
            ]
            if not named and capability in CATALOG_IS_COMPLETE:
                offered = sorted(
                    s.id
                    for s in subjects
                    if s.job_type == capability and s.kind not in ("rvc-base", "engine")
                )
                raise ApiError(
                    400,
                    "unknown_model",
                    f"job type {job_type!r} does not serve model {model!r}; it "
                    f"offers {offered}. Nothing was installed for it",
                    {"model": model, "offered": offered},
                )
            for subject in named[:1]:
                runs = _subject_runs(subject.id, decisions, card, total, pool)
                if runs is not None and not runs["usable"]:
                    raise ApiError(
                        400,
                        "model_cannot_run",
                        f"{model!r} cannot run on this card "
                        f"({runs['card_words']}): {' '.join(runs['lines'])} "
                        f"Nothing was installed for it",
                        {"model": model, "lines": runs["lines"]},
                    )
                if runs is not None:
                    words += [f"- {model}: {line}" for line in runs["lines"]]
                if subject.installed() is None:
                    pulls.append(_pull_of(subject))
        return Need(
            job_type=job_type,
            capability=capability,
            installer=installer,
            narrator_engine=engine,
            env_bytes=_env_bytes(installer, engine, self._backend.kind),
            pulls=tuple(pulls),
            plan="\n".join(words),
        )

    def _narrator_engine(self, installer: str, model: str | None) -> str | None:
        """Which tts env: the voice's own engine, or the only one there is."""
        if installer != "tts":
            return None
        if model is not None:
            try:
                voice = load_all_voices().get(model)
            except Exception:  # noqa: BLE001 - an unreadable voice decides nothing
                voice = None
            if voice is not None:
                return voice.narrator_engine
        engines = sorted(NARRATOR_ENGINE_SAMPLING)
        return engines[0] if len(engines) == 1 else None

    # ---------------------------------------------------------------- parking

    def park(self, job: Job, need: Need) -> None:
        """Admit `job` to wait for `need`, and see that something is working on it."""
        self._needs[job.id] = need
        self._store.park(job, self._status(need, None, ours=True))
        self._wake.set()
        if self._driver is None or self._driver.done():
            self._driver = asyncio.get_running_loop().create_task(
                self._drive(), name="crucible-install-on-submit"
            )

    async def stop(self) -> None:
        driver, self._driver = self._driver, None
        if driver is None:
            return
        driver.cancel()
        try:
            await driver
        except asyncio.CancelledError:
            pass

    # ----------------------------------------------------------------- driver

    async def _drive(self) -> None:
        try:
            while True:
                self._forget_ended()
                if not self._needs:
                    return
                await self._release_what_is_ready()
                self._forget_ended()
                if not self._needs:
                    return
                outstanding = {
                    job_id: need
                    for job_id, need in self._needs.items()
                    if self._missing(need)
                }
                running = self._tasks.running
                if running is not None and outstanding:
                    # Somebody else's task has the lane (an operator's Install,
                    # a pull). Waited out, never raced: one task at a time.
                    await self._follow(running, ours=False)
                    continue
                if not outstanding:
                    # Only jobs waiting for the card. Ask again shortly.
                    await self._nap(CARD_RETRY_SECONDS)
                    continue
                await self._run_round(outstanding)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - every waiting job is told
            line = f"the install-on-submit driver failed: {type(exc).__name__}: {exc}"
            print(f"crucible: {line}", file=sys.stderr)
            for job_id in list(self._needs):
                self._fail(job_id, {"code": "install_failed", "message": line})

    async def _run_round(self, outstanding: dict[str, Need]) -> None:
        module = self._module(outstanding.values())
        try:
            task = self._tasks.submit(
                {"type": "module", "module": module}, on_submit=True
            )
        except ApiError as exc:
            for job_id in outstanding:
                self._fail(job_id, {"code": exc.code, "message": exc.message})
            return
        await self._follow(task, ours=True)
        if task.state == DONE:
            for job_id, need in outstanding.items():
                missing = self._missing(need) if job_id in self._needs else []
                if missing:
                    self._fail(
                        job_id,
                        {
                            "code": "install_failed",
                            "message": f"the install task {task.id} finished and "
                            f"this job still has no {', '.join(missing)}",
                        },
                    )
            return
        # FAILED OR CANCELLED: every job that needed the step it stopped on
        # ends with it, INCLUDING one that arrived while it ran. It was waiting
        # on that same install, and running it again at once for the late
        # arrival would be a second install of one env, failing the same way.
        # A job whose own steps never ran waits for the next task.
        failed_step = _last_step(task)
        for job_id, need in list(self._needs.items()):
            if failed_step is not None and failed_step not in need.steps():
                continue
            self._fail(job_id, _failure_of(task, need))

    def _module(self, needs: Any) -> dict[str, Any]:
        """One module for every job waiting, with each env and subject once."""
        job_types: list[dict[str, Any]] = []
        subjects: list[dict[str, str]] = []
        installs: set[tuple[str, str | None]] = set()
        pulled: set[tuple[str, str]] = set()
        for need in needs:
            key = (need.installer, need.narrator_engine)
            if key not in installs:
                installs.add(key)
                entry: dict[str, Any] = {"type": need.installer}
                if need.narrator_engine is not None:
                    entry["narrator_engine"] = need.narrator_engine
                job_types.append(entry)
            for pull in need.pulls:
                if (pull.kind, pull.id) not in pulled:
                    pulled.add((pull.kind, pull.id))
                    subjects.append({"kind": pull.kind, "id": pull.id})
        names = sorted({installer for installer, _ in installs})
        return {
            "name": f"install on submit: {', '.join(names)}",
            "version": "1",
            "job_types": job_types,
            "subjects": subjects,
        }

    async def _release_what_is_ready(self) -> None:
        for job_id, need in list(self._needs.items()):
            if self._missing(need):
                continue
            job = self._job(job_id)
            if job is None:
                continue
            try:
                await self._admit(job)
            except ApiError as exc:
                if exc.code in TRANSIENT_REFUSALS:
                    message = (
                        f"installed; waiting for the card, which is held: {exc.message}"
                    )
                    now = self._store.waiting_for(job) or {}
                    if now.get("message") != message:
                        self._store.update_waiting(
                            job,
                            {
                                **self._status(need, None, ours=True),
                                "reason": "card",
                                "task_id": None,
                                "message": message,
                            },
                        )
                    continue
                self._fail(job_id, {"code": exc.code, "message": exc.message})
                continue
            except Exception as exc:  # noqa: BLE001 - the job is told, the driver lives
                self._fail(
                    job_id,
                    {"code": "job_failed", "message": f"{type(exc).__name__}: {exc}"},
                )
                continue
            self._needs.pop(job_id, None)

    async def _follow(self, task: Task, *, ours: bool) -> None:
        """Rewrite every waiting job's status from `task` until it ends."""
        waiter = self._tasks.subscribe(task)
        last: dict[str, tuple[Any, ...]] = {}
        try:
            while task.state not in TERMINAL_STATES:
                self._forget_ended()
                for job_id, need in self._needs.items():
                    job = self._job(job_id)
                    if job is None:
                        continue
                    status = self._status(need, task, ours=ours)
                    key = (status["message"], status["line"], status["progress"])
                    if last.get(job_id) != key:
                        last[job_id] = key
                        self._store.update_waiting(job, status)
                waiter.clear()
                try:
                    await asyncio.wait_for(waiter.wait(), STATUS_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    pass
                # Throttle: a pull's hook fires twice a second and a status
                # rewritten on each would be an event per chunk.
                await asyncio.sleep(STATUS_INTERVAL_SECONDS / 2)
        finally:
            self._tasks.unsubscribe(task, waiter)

    async def _nap(self, seconds: float) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), seconds)
        except asyncio.TimeoutError:
            pass

    # ---------------------------------------------------------------- reading

    def _missing(self, need: Need) -> list[str]:
        """What this job still lacks on disk, in words. Empty when it lacks nothing."""
        missing: list[str] = []
        if not env_installed(
            self._config, self._backend, need.installer, need.narrator_engine
        ):
            missing.append(f"{need.installer} environment")
        for pull in need.pulls:
            subject = catalog.find(self._config, self._backend, pull.kind, pull.id)
            if subject is not None and subject.installed() is None:
                missing.append(pull.words)
        return missing

    def _job(self, job_id: str) -> Job | None:
        try:
            job = self._store.get(job_id)
        except ApiError:
            return None
        return job if self._store.waiting_for(job) is not None else None

    def _forget_ended(self) -> None:
        """Drop jobs no longer waiting: cancelled, or failed from elsewhere."""
        for job_id in list(self._needs):
            if self._job(job_id) is None:
                self._needs.pop(job_id, None)

    def _fail(self, job_id: str, error: dict[str, str]) -> None:
        self._needs.pop(job_id, None)
        try:
            job = self._store.get(job_id)
        except ApiError:
            return
        if job.status == QUEUED and self._store.waiting_for(job) is not None:
            self._store.fail_unadmitted(job, error)

    def _status(self, need: Need, task: Task | None, *, ours: bool) -> dict[str, Any]:
        """`waiting_for`: what the job waits for, in words and in numbers."""
        phrases = self._phrases(need)
        step = None if task is None else _current_step(task)
        progress, line, pace = (None, None, "")
        if task is not None and step is not None:
            progress, line, pace = self._progress(task, step)
        if task is not None and not ours and not _serves(task, need):
            # Somebody else's task has the lane. Say whose, then ours.
            doing = (
                f"waiting for the {task.type} task already running on this "
                f"server ({task.id[:8]}) to finish, then "
                + _then(list(phrases.values()))
            )
        else:
            names = list(phrases)
            current = step["name"] if step is not None else None
            if current in phrases:
                at = names.index(current)
                now = phrases[current] + pace
                rest = [phrases[name] for name in names[at + 1:]]
                doing = now + (f", then {_joined(rest)}" if rest else "")
            elif current == "reload":
                doing = f"turning {need.job_type} on"
            elif current is not None and phrases:
                # The task is on a step another waiting job needs.
                doing = f"waiting for this server to {current} for another job, then " + _joined(
                    list(phrases.values())
                )
            else:
                doing = _then(list(phrases.values()))
        return {
            "reason": "install",
            "task_id": None if task is None else task.id,
            "message": f"{doing}; your job starts after it",
            "plan": need.plan,
            "steps": need.steps(),
            "step": step,
            "progress": progress,
            "line": line,
            "since": need.since,
        }

    def _phrases(self, need: Need) -> dict[str, str]:
        """Step name -> what it is, in words, for the steps still to do."""
        phrases: dict[str, str] = {}
        if not env_installed(
            self._config, self._backend, need.installer, need.narrator_engine
        ):
            phrases[need.install_step] = (
                f"installing the {need.installer} environment"
                + (f" for {need.narrator_engine}" if need.narrator_engine else "")
                + _size(need.env_bytes)
            )
        for pull in need.pulls:
            phrases[pull.step] = f"pulling {pull.words}{_size(pull.expected_bytes)}"
        return phrases

    def _progress(
        self, task: Task, step: dict[str, Any]
    ) -> tuple[float | None, str | None, str]:
        """The step's fraction, its last line, and a pace phrase, from its events."""
        fraction: float | None = None
        line: str | None = None
        pace = ""
        for event in reversed(task.events):
            if event["event"] == "step":
                break
            if event["event"] != "progress":
                continue
            data = event["data"]
            if line is None and isinstance(data.get("line"), str):
                line = data["line"]
            done, total = data.get("bytes_done"), data.get("bytes_total")
            if fraction is None and isinstance(done, int) and isinstance(total, int) and total > 0:
                fraction = min(1.0, done / total)
                pace = f", {fraction:.0%}" + self._remaining(task, step, done, total)
            if line is not None and fraction is not None:
                break
        return fraction, line, pace

    def _remaining(self, task: Task, step: dict[str, Any], done: int, total: int) -> str:
        key = (task.id, str(step.get("name")))
        now = time.monotonic()
        first = self._pace.setdefault(key, (now, done))
        elapsed = now - first[0]
        moved = done - first[1]
        if elapsed < PACE_AFTER_SECONDS or moved <= 0 or done >= total:
            return ""
        seconds = (total - done) / (moved / elapsed)
        if seconds < 60:
            return ", under a minute left"
        minutes = round(seconds / 60)
        return f", about {minutes} minute{'s' if minutes != 1 else ''} left"


# ------------------------------------------------------------------ helpers


def live_decisions(config: Config, backend: Backend) -> tuple[Any, Any, str]:
    """This card's decisions, decided now: what `crucible install` would record.

    The walk `GET /v1/capability/plan` makes for the install modal, so the
    sentence a job's status carries and the one the modal shows are one answer.
    """
    card = ladder.card_for(config.home, backend.gpu)
    decisions = capability_classes.decide_all(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        gpu_vendor=backend.gpu.vendor,
        chosen={entry.capability: entry.model for entry in config.local_models},
        card=card,
    )
    return decisions, card, capability_classes.pool_name(backend.kind, backend.gpu.vendor)


def _subject_runs(
    subject_id: str, decisions: Any, card: Any, total: int, pool: str
) -> dict[str, Any] | None:
    """The pull modal's verdict on one model, or None when no class offers it."""
    try:
        return capability_classes.subject_plan(
            subject_id, decisions, card=card, total_bytes=total, pool=pool
        )
    except ApiError:
        return None


_KIND_WORDS: dict[str, str] = {
    "model": "the model {id!r}",
    "voice": "the voice {id!r}",
    "rvc": "the RVC voice {id!r}",
    "rvc-base": "its base assets",
    "denoise": "the separator {id!r}",
    "engine": "the llama.cpp engine",
}


def _pull_of(subject: catalog.Subject) -> Pull:
    words = _KIND_WORDS.get(subject.kind, "{id!r}").format(id=subject.id)
    return Pull(subject.kind, subject.id, words, subject.expected_bytes)


def _env_bytes(installer: str, engine: str | None, backend_kind: str) -> int | None:
    """The recipe's own `# archive-bytes:` floor, or None when it has none."""
    try:
        if installer in workerenv.WORKER_JOB_TYPES:
            recipe = workerenv.recipe_for(installer, backend_kind)
        elif installer == "tts":
            recipe = jobenv.recipe_for(jobenv.tts_env(engine or "", backend_kind))
        else:
            return None
        return jobenv.recipe_archive_bytes(recipe)
    except Exception:  # noqa: BLE001 - a size is a courtesy, never a refusal
        return None


def _size(count: int | None) -> str:
    if not count:
        return ""
    if count >= 1e9:
        return f" (about {count / 1e9:.1f} GB)"
    return f" (about {max(1, round(count / 1e6))} MB)"


def _joined(phrases: list[str]) -> str:
    if not phrases:
        return "finishing its install"
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _then(phrases: list[str]) -> str:
    """The first thing, then the rest."""
    if len(phrases) < 2:
        return _joined(phrases)
    return f"{phrases[0]}, then {_joined(phrases[1:])}"


def _current_step(task: Task) -> dict[str, Any] | None:
    for event in reversed(task.events):
        if event["event"] == "step":
            data = event["data"]
            return {"name": data.get("name"), "index": data.get("index"), "total": data.get("total")}
    return None


def _last_step(task: Task) -> str | None:
    step = _current_step(task)
    return None if step is None else step.get("name")


def _serves(task: Task, need: Need) -> bool:
    """Is somebody else's task the install this job waits for anyway?"""
    request = task.request
    if task.type == "install":
        return request.get("job_type") == need.installer and request.get(
            "narrator_engine"
        ) == need.narrator_engine
    return False


def _failure_of(task: Task, need: Need) -> dict[str, str]:
    """The job's error: the install's own one-line reason, first."""
    if task.state == FAILED and task.error is not None:
        code = task.error.get("code", "install_failed")
        reason = task.reason or task.error.get("message", "")
        return {
            "code": code,
            "message": f"installing {need.installer} for this job failed: {reason}",
        }
    if task.state == FAILED:
        return {
            "code": "install_failed",
            "message": f"installing {need.installer} for this job failed; task "
            f"{task.id}'s events say why",
        }
    return {
        "code": "install_cancelled",
        "message": f"the install this job was waiting for (task {task.id}) was "
        "cancelled, so the job was not run",
    }


__all__ = [
    "BASE_SUBJECTS",
    "InstallOnSubmit",
    "Need",
    "Pull",
    "live_decisions",
]
