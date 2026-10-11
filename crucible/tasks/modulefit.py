"""What a module asks for, judged against THIS card before anything is installed.

Owen, 2026-10-11: *"crucible should intelligently know which things will fit on the gpu
and which wont. rvc will fit on victoria's system. it should be installed there."* A module
is one list for every server an app talks to, so an entry this card cannot run is not the
app's mistake and not a reason to stop the rest: it is SKIPPED, said on the task's stream,
and reported in the task's `unmet` with the fit owner's own sentence, as a `need` this
server cannot serve already is. Nothing here decides fit: the decisions are
installplan.live_decisions (the same reading install-on-submit and
`GET /v1/capability/plan` make), and the sentences are install_plan's and subject_plan's.

It also says which job type each entry is FOR, so the module can turn on the type that
serves what it put on disk (a `denoise` separator pulled into the `rvc` env is served by
`denoise`, whose flag no install of `rvc` is asked to set when the env is already there).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import catalog, installplan
from ..backend import Backend
from ..capabilityclasses import BY_NAME, classes_for_job_type
from ..capabilitywords import describe_card
from ..config import Config
from ..jobenv import JOB_TYPES_SERVED_BY_ENV
from .validate import ModuleEntry


@dataclass(frozen=True)
class Unfit:
    """Why an entry is skipped on this card: a sentence for its `skipped` event and its
    rows for the task's `unmet` (``{"class", "reason", "entry"}``)."""

    words: str
    rows: tuple[dict[str, str], ...]


def unmet_row(capability_class: str, reason: str, entry: str) -> dict[str, str]:
    return {"class": capability_class, "reason": reason, "entry": entry}


class ModuleFit:

    def __init__(self, config: Config, backend: Backend) -> None:
        self._config = config
        self._backend = backend
        self._decisions, self._card, self._pool = installplan.live_decisions(config, backend)

    @property
    def card_words(self) -> str:
        return describe_card(self._card, self._backend.gpu.vram_bytes, self._pool)

    def install(self, entry: ModuleEntry) -> Unfit | None:
        """None when anything the env serves runs on this card (`crucible install` turns
        on each type it serves that does); else every class it serves, each with its own
        line."""
        assert entry.job_type is not None
        plans = [
            installplan.install_plan(
                job_type,
                self._decisions,
                card=self._card,
                total_bytes=self._backend.gpu.vram_bytes,
                pool=self._pool,
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                desktop_basis=self._config.desktop_allowance_basis,
            )
            for job_type in served_by(entry.job_type)
        ]
        if any(plan["usable"] for plan in plans):
            return None
        rows = tuple(
            unmet_row(row["capability"], row["line"], entry.name)
            for plan in plans
            for row in plan["classes"]
        )
        return Unfit(
            f"nothing it serves can run on this card ({self.card_words}), so its env "
            "is not built: " + " ".join(row["reason"] for row in rows),
            rows,
        )

    def subject(self, entry: ModuleEntry, subject: catalog.Subject) -> Unfit | None:
        """None when the subject runs on this card, or when no class weighs it (rvc's
        base assets, the llama.cpp engine, a model in a family of its own)."""
        if not installplan.offering_classes(subject.id, self._decisions):
            return None
        plan = installplan.subject_plan(
            subject.id,
            self._decisions,
            card=self._card,
            total_bytes=self._backend.gpu.vram_bytes,
            pool=self._pool,
        )
        if plan["usable"]:
            return None
        rows = tuple(
            unmet_row(row["capability"], row["line"], entry.name)
            for row in plan["classes"]
            if not row["usable"]
        )
        return Unfit(
            f"it cannot run on this card ({plan['card_words']}), so it is not "
            "downloaded: " + " ".join(row["reason"] for row in rows),
            rows,
        )

    def classes_for(self, entry: ModuleEntry, job_type: str) -> list[str]:
        """The classes an enable refusal for `job_type` is reported under, for this
        entry: a need's own class, the classes that weigh a subject, else every class
        of the type."""
        if entry.capability_class is not None:
            return [entry.capability_class]
        if entry.subject_id is not None:
            offered = [
                name
                for name in installplan.offering_classes(entry.subject_id, self._decisions)
                if BY_NAME[name].job_type == job_type
            ]
            if offered:
                return offered
        return [cls.name for cls in classes_for_job_type(job_type)]


def served_by(installer: str) -> tuple[str, ...]:
    """The job types an env serves: `rvc` serves `rvc` and `denoise`."""
    return JOB_TYPES_SERVED_BY_ENV.get(installer, (installer,))


def job_type_of(config: Config, backend: Backend, entry: ModuleEntry) -> str:
    """The job type that serves what this entry puts on the server: the type an install
    names, the class's type for a need, the subject's own for a subject."""
    if entry.job_type is not None:
        return entry.job_type
    if entry.capability_class is not None:
        return BY_NAME[entry.capability_class].job_type
    assert entry.kind is not None and entry.subject_id is not None
    subject = catalog.find(config, backend, entry.kind, entry.subject_id)
    assert subject is not None
    return subject.job_type


def enables(
    config: Config, backend: Backend, entries: list[ModuleEntry]
) -> dict[str, list[int]]:
    """Each job type the module's entries are for, in the order they first appear, with
    the positions of the entries that ask for it."""
    found: dict[str, list[int]] = {}
    for position, entry in enumerate(entries):
        found.setdefault(job_type_of(config, backend, entry), []).append(position)
    return found


__all__ = [
    "ModuleFit",
    "Unfit",
    "enables",
    "job_type_of",
    "served_by",
    "unmet_row",
]
