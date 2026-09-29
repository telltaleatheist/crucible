from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import capabilitywords, catalog, installplan, jobenv, verdict
from .backend import Backend
from .capabilitystore import decide_for
from .cardfacts import card_for
from .config import Config
from .errors import ApiError
from .jobenv import INSTALLABLE_JOB_TYPES, INSTALLER_FOR
from .jobs import ALL_JOB_TYPES
from .jobtypes import FAMILIES, spec_of
from .narratorengines import NARRATOR_ENGINE_SAMPLING
from .voicecatalog import load_all_voices
from .tasks import Task, TaskStore, env_installed
from .tasks.states import CANCELLED, FAILED, TERMINAL_STATES

INSTALLING = "installing"

BASE_SUBJECT_IDS: dict[str, str] = {"rvc-base": catalog.RVC_BASE_ID}

BASE_SUBJECTS: dict[str, tuple[tuple[str, str], ...]] = {
    family.name: tuple((kind, BASE_SUBJECT_IDS[kind]) for kind in family.base_subject_kinds)
    for family in FAMILIES
    if family.base_subject_kinds
}

PULLABLE_REFUSALS: frozenset[str] = frozenset().union(
    *(family.pullable_refusals for family in FAMILIES)
)

ENV_REFUSALS = frozenset({"env_missing"})

INSTALLABLE_REFUSALS = PULLABLE_REFUSALS | ENV_REFUSALS

CATALOG_IS_COMPLETE: frozenset[str] = frozenset(
    family.name for family in FAMILIES if family.catalog_is_complete
)

PACE_AFTER_SECONDS = 3.0

_NOT_A_MODEL = ("rvc-base", "engine")


@dataclass(frozen=True)
class Pull:
    kind: str
    id: str
    words: str
    expected_bytes: int | None

    @property
    def step(self) -> str:
        return f"pull {self.kind} {self.id}"


@dataclass(frozen=True)
class Need:
    job_type: str
    installer: str | None
    narrator_engine: str | None
    env_bytes: int | None
    pulls: tuple[Pull, ...]
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
    def __init__(self, config: Config, backend: Backend, tasks: TaskStore) -> None:
        self._config = config
        self._backend = backend
        self._tasks = tasks
        self._task_needs: dict[str, Need] = {}
        self._reported: set[str] = set()
        self._pace: dict[tuple[str, str], tuple[float, int]] = {}


    @staticmethod
    def installable(job_type: str) -> bool:
        spec = spec_of(job_type)
        return spec is not None and spec.installable

    def plan(self, job_type: str, model: str | None, refusal: ApiError) -> Need:
        details = refusal.details or {}
        install = details.get("install")
        if (
            details.get("reason") != "not_installed"
            or not isinstance(install, dict)
            or not self.installable(job_type)
        ):
            raise refusal
        installer = str(install["job_type"])
        engine = self._narrator_engine(installer, model)
        if installer == "tts" and engine is None:
            raise refusal
        return self._need(job_type, model, installer, engine, refusal, model_required=True)

    def env_for(self, job_type: str, model: str | None, refusal: ApiError) -> Need:
        if refusal.code not in ENV_REFUSALS or not self.installable(job_type):
            raise refusal
        details = refusal.details or {}
        installer = _installer_of(job_type, details.get("env"))
        engine = None
        if installer == "tts":
            named = details.get("narrator_engine")
            engine = named if isinstance(named, str) else self._narrator_engine(installer, model)
            if engine is None:
                raise refusal
        return self._need(job_type, model, installer, engine, refusal, model_required=False)

    def need_for(self, job_type: str, model: str | None, refusal: ApiError) -> Need | None:
        if refusal.code in ENV_REFUSALS:
            return self.env_for(job_type, model, refusal)
        return self.pulls_for(job_type, model, refusal)

    def _need(
        self,
        job_type: str,
        model: str | None,
        installer: str,
        engine: str | None,
        refusal: ApiError,
        *,
        model_required: bool,
    ) -> Need:
        capability = ALL_JOB_TYPES[job_type]
        if env_installed(self._config, self._backend, installer, engine):
            running = self._tasks.running
            earlier = None if running is None else self._task_needs.get(running.id)
            if earlier is not None and (earlier.installer, earlier.narrator_engine) == (
                installer,
                engine,
            ):
                return earlier
            raise refusal

        decisions, card, pool = live_decisions(self._config, self._backend)
        total = self._backend.gpu.vram_bytes
        plan = installplan.install_plan(
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
        if (
            model_required
            and model is None
            and job_type == capability
            and capability in CATALOG_IS_COMPLETE
        ):
            offered = _offered(subjects, capability)
            if offered:
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
        if refusal.code not in PULLABLE_REFUSALS:
            return None
        capability = ALL_JOB_TYPES.get(job_type)
        if capability is None:
            return None
        decisions, card, pool = live_decisions(self._config, self._backend)
        subjects = catalog.subjects(self._config, self._backend)
        pulls, lines = self._pulls(job_type, capability, model, subjects, decisions, card, pool)
        if not pulls:
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
        card_words = capabilitywords.describe_card(
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
        if installer != "tts":
            return None
        if model is not None:
            try:
                voice = load_all_voices().get(model)
            except Exception:
                voice = None
            if voice is not None:
                return voice.narrator_engine
        engines = sorted(NARRATOR_ENGINE_SAMPLING)
        return engines[0] if len(engines) == 1 else None


    def start(self, need: Need) -> ApiError:
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


    def _serves(self, task: Task, need: Need) -> bool:
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


def live_decisions(config: Config, backend: Backend) -> tuple[Any, Any, str]:
    card = card_for(config.home, backend.gpu)
    decisions = decide_for(config, backend, card=card)
    return decisions, card, verdict.pool_name(backend.kind, backend.gpu.vendor)


def _offered(subjects: list[catalog.Subject], capability: str) -> list[str]:
    return sorted(
        s.id for s in subjects if s.job_type == capability and s.kind not in _NOT_A_MODEL
    )


def _subject_runs(
    subject_id: str, decisions: Any, card: Any, total: int, pool: str
) -> dict[str, Any] | None:
    try:
        return installplan.subject_plan(
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


def _installer_of(job_type: str, env: Any) -> str:
    name = Path(str(env)).name if isinstance(env, str) and env else ""
    if name in INSTALLABLE_JOB_TYPES:
        return name
    if name.startswith("tts-"):
        return "tts"
    if name.startswith(f"{jobenv.AUDIO_JOB_TYPE}-"):
        return jobenv.AUDIO_JOB_TYPE
    return INSTALLER_FOR[ALL_JOB_TYPES[job_type]]


def _env_bytes(installer: str, engine: str | None, backend_kind: str) -> int | None:
    try:
        if installer == jobenv.AUDIO_JOB_TYPE:
            return sum(
                jobenv.recipe_archive_bytes(jobenv.recipe_for(spec))
                for spec in jobenv.audio_envs(backend_kind)
            )
        if installer in jobenv.WORKER_JOB_TYPES:
            recipe = jobenv.recipe_for(jobenv.worker_env(installer, backend_kind))
        elif installer == "tts":
            recipe = jobenv.recipe_for(jobenv.tts_env(engine or "", backend_kind))
        else:
            return None
        return jobenv.recipe_archive_bytes(recipe)
    except Exception:
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
    "ENV_REFUSALS",
    "INSTALLABLE_REFUSALS",
    "INSTALLING",
    "InstallOnSubmit",
    "Need",
    "PULLABLE_REFUSALS",
    "Pull",
    "live_decisions",
]
