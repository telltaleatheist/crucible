"""Install on submit: a job whose environment or model is missing gets it installed.

OWEN'S RULINGS. 2026-09-26: *"yes, we need to install a missing environment when
a job is submitted"*, under his standing rule that Crucible must be idiot proof:
assume the caller has no idea an "environment" exists. 2026-09-27, on the model
behind an installed env: *"Yes, it should try to pull the model"*; on a server
that never decided its card: *"Yes, it should automatically be checked"*; and on
what happens to the job meanwhile: *"Crucible isn't responsible for queuing. The
apps that use it are. It grants and releases leases. That's it"*.

Fresh-install snag #2 (`docs/FRESH-INSTALL-KYLIES-2026-09-26.md`): kylies-pc's
rvc env, its base assets and its model were three commands typed inside the
guest (`docs/PROPOSAL-INSTALL-ON-SUBMIT.md`).

WHAT A CLIENT SEES. `POST /v1/jobs` for a type this card can run and has not
installed, or for a declared model/voice (and `rvc`'s base assets) this card can
run and has not pulled, starts the install the operator page's Install and Pull
buttons start, as one `module` task, and is REFUSED `409 installing`:

    installing the rvc environment (about 3.3 GB), then pulling its base
    assets (about 900 MB) and the RVC voice 'sigma' (about 55 MB); submit this
    job again after it. Task 1f2e... is doing it: GET /v1/tasks/1f2e...

`details` carry the task id, the steps, the step it is on, its byte progress and
last line, and `plan`, the install modal's sentences for this card
(`GET /v1/capability/plan`): the UI asks before installing; an API caller has no
modal, so it reads them here. Submitting again while it runs answers the same
refusal with the same task and the progress so far; submitting after it ends is
an ordinary submit. The task's own record carries the sentence too
(`GET /v1/tasks/{id}` `message`).

WHY A REFUSAL AND NOT A HELD JOB. The first build accepted the job and held it
until the install landed, then put it on the lane behind whatever was there,
which made the lane a queue of two and Crucible the owner of an order. Owen's
2026-09-27 answer rules that out: the server answers "is there room now" and
grants or refuses (ARCHITECTURE.md section 3, `JobStore.refuse_if_busy`), and
the app's queue decides what to send next. The other reading, holding the job
and giving it ONE normal admission when the install lands, still makes the
server the thing that remembers a job the app asked for minutes ago and decides
when to run it, and turns "the lane was busy at that moment" into a failure the
app did not cause. So nothing is held: the refusal names the work under way and
the moment to come back, which is the `server_busy` shape an app already
retries on.

ONE INSTALL PER ENV. There is one task lane and one task at a time. A submit
that arrives while an install it needs is running is pointed at that task,
never given a second; one that arrives while an unrelated task runs is told to
come back after it. An install that failed is reported ONCE, with the
install's own one-line reason (`jobenv.failure_message`'s head, `Task.reason`),
to the next submit that needed it; the submit after that tries again. A job is
never answered by a loop of the same failing install.

WHAT IS STILL REFUSED AT ONCE: a type this card cannot serve (`cannot_hold`, or
a live install plan that says nothing it offers runs here), a model the card
cannot run, a type with no installer (`echo`), `unload-*` of an uninstalled
type, an env already on disk with its flag off, a missing or undeclared model
for a type whose models are all in the catalog. A server with NO capability
record decides one first (`crucible/api.py`, `decide_here`) and is then
answered as above. `[jobs] install_on_submit = false` puts the plain refusal
back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import capability as capability_classes
from . import catalog, jobenv, ladder
from .backend import Backend
from .config import Config
from .errors import ApiError
from .jobs import ALL_JOB_TYPES
from .tasks import CANCELLED, FAILED, TERMINAL_STATES, Task, TaskStore, env_installed
from .voices import NARRATOR_ENGINE_SAMPLING, load_all_voices

#: The refusal every missing install is answered with. 409, the `server_busy`
#: family: a fact about this server right now that the app's queue retries.
INSTALLING = "installing"

#: The weights a type cannot run AT ALL without, whatever the job names, by
#: capability. `rvc`'s base assets are the engine's (hubert, rmvpe), shared by
#: every voice.
BASE_SUBJECTS: dict[str, tuple[tuple[str, str], ...]] = {
    "rvc": (("rvc-base", catalog.RVC_BASE_ID),),
}

#: `preflight`'s refusals that mean "the weights are not on this disk". Only on
#: one of these is a pull considered, so a job whose weights are here costs
#: nothing extra.
PULLABLE_REFUSALS = frozenset(
    {
        "model_not_installed",
        "voice_not_installed",
        "rvc_base_models_missing",
        "denoise_model_missing",
    }
)

#: Capabilities whose every servable model is a catalog subject, so a model
#: the catalog does not have is a typo and is refused BEFORE anything installs.
#: `llm` and `tts` are not here: a model may be an upstream route, a voice a
#: local directory, and those are the plugin's to judge once it exists.
CATALOG_IS_COMPLETE = frozenset({"rvc", "denoise", "asr", "align"})

#: A download's pace is not stated until it has run this long: the first
#: seconds of a pull are connection setup, and a remaining time computed from
#: them is a number that moves by minutes.
PACE_AFTER_SECONDS = 3.0

_NOT_A_MODEL = ("rvc-base", "engine")


@dataclass(frozen=True)
class Pull:
    """One subject a job needs pulled."""

    kind: str
    id: str
    #: The subject in a person's words: "its base assets", "the RVC voice 'sigma'".
    words: str
    expected_bytes: int | None

    @property
    def step(self) -> str:
        """The module step's name (`TaskStore._run_module`)."""
        return f"pull {self.kind} {self.id}"


@dataclass(frozen=True)
class Need:
    """What one job needs installed before it can be admitted."""

    job_type: str
    #: The install that builds its env (`INSTALLER_FOR`), or None when the env
    #: is here and only weights are missing; for `tts`, which narrator engine.
    installer: str | None
    narrator_engine: str | None
    env_bytes: int | None
    pulls: tuple[Pull, ...]
    #: The install modal's sentences for this card, or None.
    plan: str | None

    @property
    def install_step(self) -> str | None:
        if self.installer is None:
            return None
        return f"install {self.installer}" + (
            f" ({self.narrator_engine})" if self.narrator_engine else ""
        )

    def steps(self) -> list[str]:
        first = [] if self.install_step is None else [self.install_step]
        return [*first, *(pull.step for pull in self.pulls)]


class InstallOnSubmit:
    """Decides what a job needs installed, starts it once, and says so."""

    def __init__(self, config: Config, backend: Backend, tasks: TaskStore) -> None:
        self._config = config
        self._backend = backend
        self._tasks = tasks
        #: What each install-on-submit task was started for, by task id: its
        #: sentence, and which later submits it serves.
        self._task_needs: dict[str, Need] = {}
        #: Failed tasks whose failure a submit has already been told.
        self._reported: set[str] = set()
        #: (task id, step name) -> (monotonic, bytes) at first sight, for a pace.
        self._pace: dict[tuple[str, str], tuple[float, int]] = {}

    # --------------------------------------------------------------- planning

    @staticmethod
    def installable(job_type: str) -> bool:
        """Could installing ever make this type servable here?"""
        from .cli import INSTALLABLE_JOB_TYPES, INSTALLER_FOR  # cli imports api

        capability = ALL_JOB_TYPES.get(job_type)
        if capability is None or job_type.startswith("unload-"):
            # Nothing of an uninstalled type is on the card to unload, and
            # gigabytes of install to find that out is not a service.
            return False
        return INSTALLER_FOR.get(capability) in INSTALLABLE_JOB_TYPES

    def plan(self, job_type: str, model: str | None, refusal: ApiError) -> Need:
        """What installing for this job means, or the refusal it should get instead.

        `refusal` is `disabled_error`'s answer for this type. Only its
        `not_installed` case is installed for; every other case is re-raised as
        it is, because it already says the true thing.
        """
        details = refusal.details or {}
        install = details.get("install")
        if (
            details.get("reason") != "not_installed"
            or not isinstance(install, dict)
            or not self.installable(job_type)
        ):
            raise refusal
        capability = ALL_JOB_TYPES[job_type]
        installer = str(install["job_type"])
        engine = self._narrator_engine(installer, model)
        if installer == "tts" and engine is None:
            raise refusal
        if env_installed(self._config, self._backend, installer, engine):
            running = self._tasks.running
            earlier = None if running is None else self._task_needs.get(running.id)
            if earlier is not None and (earlier.installer, earlier.narrator_engine) == (
                installer,
                engine,
            ):
                # Built, and the task that built it has not reached its
                # take-up yet (its pulls come first). That task, not a stale
                # "not installed".
                return earlier
            # On disk, flag off: an install would find nothing to do and the
            # type would stay off. The refusal says what is true.
            raise refusal

        decisions, card, pool = live_decisions(self._config, self._backend)
        total = self._backend.gpu.vram_bytes
        plan = capability_classes.install_plan(
            capability,
            decisions,
            card=card,
            total_bytes=total,
            pool=pool,
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            desktop_basis=self._config.desktop_allowance_basis,
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
        if model is None and job_type == capability and capability in CATALOG_IS_COMPLETE:
            offered = _offered(subjects, capability)
            if offered:
                # `resolve_model`'s refusal, made before the install rather
                # than after it: a request that names no model cannot run.
                raise ApiError(
                    400,
                    "model_required",
                    f"job type {job_type!r} requires a model; it offers {offered}. "
                    "Nothing was installed for it",
                )
        pulls, lines = self._pulls(job_type, capability, model, subjects, decisions, card, pool)
        return Need(
            job_type=job_type,
            installer=installer,
            narrator_engine=engine,
            env_bytes=_env_bytes(installer, engine, self._backend.kind),
            pulls=tuple(pulls),
            plan="\n".join(words + lines),
        )

    def pulls_for(self, job_type: str, model: str | None, refusal: ApiError) -> Need | None:
        """The weights an installed type's job is missing, or None to refuse as before.

        Owen, 2026-09-27: *"Yes, it should try to pull the model"*. Asked only
        when `preflight` refused for missing weights (`PULLABLE_REFUSALS`).
        None when nothing declared and pullable is missing, so the plugin's own
        refusal stands.
        """
        if refusal.code not in PULLABLE_REFUSALS:
            return None
        capability = ALL_JOB_TYPES.get(job_type)
        if capability is None:
            return None
        decisions, card, pool = live_decisions(self._config, self._backend)
        subjects = catalog.subjects(self._config, self._backend)
        pulls, lines = self._pulls(job_type, capability, model, subjects, decisions, card, pool)
        if not pulls:
            # Nothing missing NOW, though `preflight` said so a moment ago: a
            # task pulling it finished its download in between and is still
            # finishing. Pointed at that task, not refused with a stale
            # "not installed".
            running = self._tasks.running
            earlier = None if running is None else self._task_needs.get(running.id)
            base = set(BASE_SUBJECTS.get(capability, ()))
            pulls = [
                pull
                for pull in (() if earlier is None else earlier.pulls)
                if (pull.kind, pull.id) in base
                or (pull.id == model and pull.kind not in _NOT_A_MODEL)
            ]
            if not pulls:
                return None
        card_words = capability_classes.describe_card(
            card, self._backend.gpu.vram_bytes, pool
        )
        return Need(
            job_type=job_type,
            installer=None,
            narrator_engine=None,
            env_bytes=None,
            pulls=tuple(pulls),
            plan="\n".join([f"Your card ({card_words}):", *lines]) if lines else None,
        )

    def _pulls(
        self,
        job_type: str,
        capability: str,
        model: str | None,
        subjects: list[catalog.Subject],
        decisions: Any,
        card: Any,
        pool: str,
    ) -> tuple[list[Pull], list[str]]:
        """The base weights and the named subject that are missing, and the
        pull modal's lines for the named one. Refuses a model the card cannot
        run or the catalog does not declare, before anything is fetched."""
        pulls: list[Pull] = []
        lines: list[str] = []
        for kind, subject_id in BASE_SUBJECTS.get(capability, ()):
            subject = next(
                (s for s in subjects if s.kind == kind and s.id == subject_id), None
            )
            if subject is not None and subject.installed() is None:
                pulls.append(_pull_of(subject))
        if model is None:
            return pulls, lines
        named = [
            s
            for s in subjects
            if s.id == model and s.job_type == capability and s.kind not in _NOT_A_MODEL
        ]
        if not named and capability in CATALOG_IS_COMPLETE:
            offered = _offered(subjects, capability)
            raise ApiError(
                400,
                "unknown_model",
                f"job type {job_type!r} does not serve model {model!r}; it "
                f"offers {offered}. Nothing was installed for it",
                {"model": model, "offered": offered},
            )
        for subject in named[:1]:
            runs = _subject_runs(
                subject.id, decisions, card, self._backend.gpu.vram_bytes, pool
            )
            if runs is not None and not runs["usable"]:
                raise ApiError(
                    400,
                    "model_cannot_run",
                    f"{model!r} cannot run on this card ({runs['card_words']}): "
                    f"{' '.join(runs['lines'])} Nothing was installed for it",
                    {"model": model, "lines": runs["lines"]},
                )
            if runs is not None:
                lines += [f"- {model}: {line}" for line in runs["lines"]]
            if subject.installed() is None:
                pulls.append(_pull_of(subject))
        return pulls, lines

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

    # ---------------------------------------------------------------- the act

    def start(self, need: Need) -> ApiError:
        """Start (or find) the install `need` waits on; the refusal to answer with.

        **Event loop only**, and synchronous from the lane check to the task's
        start, so two submits a millisecond apart cannot both start one:
        `TaskStore.submit`'s own atomicity.
        """
        failed = self._unreported_failure(need)
        if failed is not None:
            return failed
        running = self._tasks.running
        if running is not None:
            return self._installing(need, running)
        module = {
            "name": f"install on submit: {need.job_type}",
            "version": "1",
            "job_types": (
                []
                if need.installer is None
                else [
                    {"type": need.installer}
                    | (
                        {"narrator_engine": need.narrator_engine}
                        if need.narrator_engine
                        else {}
                    )
                ]
            ),
            "subjects": [{"kind": pull.kind, "id": pull.id} for pull in need.pulls],
        }
        try:
            task = self._tasks.submit({"type": "module", "module": module}, on_submit=True)
        except ApiError as exc:
            return exc
        self._task_needs[task.id] = need
        task.describe = self.describe
        return self._installing(need, task)

    def describe(self, task: Task) -> str | None:
        """The sentence a task started for a job carries on its own record."""
        need = self._task_needs.get(task.id)
        if need is None or task.state in TERMINAL_STATES:
            return None
        return self._doing(need, task, serves=True)

    def _installing(self, need: Need, task: Task) -> ApiError:
        serves = self._serves(task, need)
        doing = self._doing(need, task, serves=serves)
        step = _current_step(task) if serves else None
        progress, line, _ = (None, None, "") if step is None else self._progress(task, step)
        return ApiError(
            409,
            INSTALLING,
            f"{doing}; submit this job again after it. Task {task.id} is doing "
            f"{'it' if serves else 'the work before it'}: GET /v1/tasks/{task.id}",
            {
                "job_type": need.job_type,
                "task_id": task.id,
                # `installing`: that task is this job's install. `task_busy`:
                # another task has the one lane; this job's install starts on
                # the first submit after it.
                "reason": "installing" if serves else "task_busy",
                "message": doing,
                "plan": need.plan,
                "steps": need.steps(),
                "step": step,
                "progress": progress,
                "line": line,
            },
        )

    def _unreported_failure(self, need: Need) -> ApiError | None:
        """The last install this job needed, if it failed and nobody was told."""
        wanted = set(need.steps())
        for task in self._tasks.recent():
            earlier = self._task_needs.get(task.id)
            if earlier is None or not wanted & set(earlier.steps()):
                continue
            if task.state not in (FAILED, CANCELLED) or task.id in self._reported:
                return None
            self._reported.add(task.id)
            if task.state == CANCELLED:
                return ApiError(
                    409,
                    "install_cancelled",
                    f"the install this job needs (task {task.id}) was cancelled. "
                    "Submitting again starts it again",
                    {"task_id": task.id},
                )
            error = task.error or {}
            reason = task.reason or error.get("message") or f"task {task.id} failed"
            return ApiError(
                409,
                error.get("code", "install_failed"),
                f"installing what this job needs failed: {reason}. Submitting "
                f"again tries again (task {task.id})",
                {"task_id": task.id, "reason": reason},
            )
        return None

    # ---------------------------------------------------------------- reading

    def _serves(self, task: Task, need: Need) -> bool:
        """Is `task` doing (part of) what this job needs?"""
        earlier = self._task_needs.get(task.id)
        if earlier is not None:
            return bool(set(earlier.steps()) & set(need.steps()))
        request = task.request
        if task.type == "install" and need.installer is not None:
            return request.get("job_type") == need.installer and request.get(
                "narrator_engine"
            ) == need.narrator_engine
        if task.type == "pull":
            return any(
                request.get("kind") == pull.kind and request.get("id") == pull.id
                for pull in need.pulls
            )
        return False

    def _doing(self, need: Need, task: Task, *, serves: bool) -> str:
        """What is happening for this job, in words."""
        phrases = self._phrases(need)
        if not serves:
            return (
                f"waiting for the {task.type} task already running on this "
                f"server ({task.id[:8]}) to finish, then {_then(list(phrases.values()))}"
            )
        step = _current_step(task)
        current = None if step is None else step.get("name")
        names = list(phrases)
        if current in phrases:
            _, _, pace = self._progress(task, step)
            at = names.index(current)
            rest = [phrases[name] for name in names[at + 1:]]
            return phrases[current] + pace + (f", then {_joined(rest)}" if rest else "")
        if current == "reload":
            return f"turning {need.job_type} on"
        return _then(list(phrases.values()))

    def _phrases(self, need: Need) -> dict[str, str]:
        """Step name -> what it is, in words, for the steps still to do."""
        phrases: dict[str, str] = {}
        if need.installer is not None and not env_installed(
            self._config, self._backend, need.installer, need.narrator_engine
        ):
            phrases[need.install_step or ""] = (
                f"installing the {need.installer} environment"
                + (f" for {need.narrator_engine}" if need.narrator_engine else "")
                + _size(need.env_bytes)
            )
        for pull in need.pulls:
            subject = catalog.find(self._config, self._backend, pull.kind, pull.id)
            if subject is None or subject.installed() is None:
                phrases[pull.step] = f"pulling {pull.words}{_size(pull.expected_bytes)}"
        return phrases

    def _progress(
        self, task: Task, step: dict[str, Any] | None
    ) -> tuple[float | None, str | None, str]:
        """The step's fraction, its last line, and a pace phrase, from its events."""
        fraction: float | None = None
        line: str | None = None
        pace = ""
        if step is None:
            return fraction, line, pace
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
    sentences a refusal carries and the ones the modal shows are one answer.
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


def _offered(subjects: list[catalog.Subject], capability: str) -> list[str]:
    return sorted(
        s.id for s in subjects if s.job_type == capability and s.kind not in _NOT_A_MODEL
    )


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
        if installer in jobenv.WORKER_JOB_TYPES:
            recipe = jobenv.recipe_for(jobenv.worker_env(installer, backend_kind))
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
            return {
                "name": data.get("name"),
                "index": data.get("index"),
                "total": data.get("total"),
            }
    return None


__all__ = [
    "BASE_SUBJECTS",
    "INSTALLING",
    "InstallOnSubmit",
    "Need",
    "PULLABLE_REFUSALS",
    "Pull",
    "live_decisions",
]
