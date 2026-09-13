"""What this server can actually do, decided from the card it is installed on.

PHASE9-CAPABILITY.md is the ruling. Owen, 2026-09-13: *"the extreme/moderate/fast
settings are irrelevant now. it uses what is available on the system. the user
doesnt set those. crucible does."* This module is the thing that does.

The shape of the decision
-------------------------
A client asks for a **capability class** — "a translate-class model", "a voice" —
and this server answers with the concrete thing it has, or refuses with the number
that stopped it. Section 1 of the phase doc splits the ownership and it is the
reason this file exists at all:

    which FAMILY a task needs        the client   a quality requirement about the work
    which QUANTIZATION runs here     Crucible     how the engine realizes the model
    none of them fits                Crucible     the class is disabled, with the number

So every class is "binary in its family, graded in its quantization". `tts` looks
binary only because Higgs has exactly one quantization — Owen: *"higgs is tied to a
certain size. we cant (or wont) quantize that. if higgs doesnt fit in a card that
crucible is installed on, it's disabled on that gpu."* It is the same rule with a
family of one, and it needs no special case here.

Four decisions this file makes, and why each one is the way it is
-----------------------------------------------------------------
**1. Candidates are ordered by DECLARED SIZE, descending, and the first that fits
wins.** Not by a `precision` field in the manifests, because there is no such
field and adding one would give "how heavily is this quantized" a SECOND owner
alongside `memory_bytes_estimate`, which already answers it (ARCHITECTURE.md R1).
Within one family on one backend the estimate is dominated by the weights, so
bigger *is* less quantized: `qwen3.8-27b` at 56.4 GB against `qwen3.8-27b-4bit` at
21.6 GB on cuda-linux, 55.5 against 33.9 on mlx-darwin. The order comes out right
on both without anybody maintaining a table.

Best-first, never smallest-that-fits. A 4-bit translation is a worse translation
than a bf16 one, and this rule is about the best output the host can hold, not the
most that can be crammed onto it.

**2. The bar is TOTAL memory, not free memory.** A capability is a fact about the
host; free VRAM is a fact about this second. `crucible/accelerator.py` already owns
"is there room right now" and refuses `insufficient_memory` at load time with the
measured figure. If this module also selected on `free_bytes`, a browser open
during `crucible install` would permanently disable TTS on a 24 GB card — a
capability decided by a transient. Total comes from `backend.detect_backend()`,
which is the one owner of "how big is this card" (`nvidia-smi memory.total` on
cuda-linux, `hw.memsize` on mlx-darwin — on Apple Silicon the pool and the machine
are the same thing).

**3. There is no `margin` term, and its absence is the ruling rather than an
omission.** Section 1.3 of the phase doc writes the test as
`estimate + margin <= total - desktop_allowance`, and then section 1.2 answers what
`margin` is: it is `desktop_allowance_bytes`. Owen, 2026-09-13: *"I don't think we
need to measure estimates. I've been using this system the way it is for months and
it works fine. Use the current settings for each."* Adding a second reserve on top
of the first would be a number nobody has measured, invented to feel safe, and it
would disable translate on the 3090 Ti that has been translating for months —
exactly the failure section 1.1 records, where a rule nobody had run against a
known-good answer disagreed with the operator and the rule was the thing that was
wrong.

**4. The two known-good answers this rule is checked against.** Both live in
`tests/test_capability.py` and both predate the rule:

    3090 Ti, 24.0 GiB, 3.0 GiB reserve   -> translate ENABLED on qwen3.8-27b-4bit
                                            (20.1 GiB needed, 21.0 GiB available)
    64 GiB Studio, 25% reserve           -> translate ENABLED on qwen3.8-27b-4bit,
                                            bf16 REFUSED (51.7 GiB against 48.0)

Owen runs both of those today. A selection rule that has never been run against an
answer somebody already knows is a rule nobody has tested.

Units
-----
Every figure this module prints is GiB, because its messages sit beside
`crucible/accelerator.py`'s refusals about the same bytes and a reader comparing
"needs 19.0 GB" with "needs 17.7 GiB" for one number has been given a puzzle
instead of an answer. The phase doc's tables are in GB; the bytes are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import CUDA_LINUX, MLX_DARWIN
from .config import CapabilityRecord, CapabilityRow
from .manifests import load_all_manifests
from .rvcmodels import load_all_rvc_manifests
from .voices import load_all_voices

GIB = 1024 ** 3


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


#: What the pool is called on each backend, because "card" is a lie on a Mac: the
#: model, the compositor and every open app allocate from the same unified memory.
POOL_NAME: dict[str, str] = {
    CUDA_LINUX: "card",
    MLX_DARWIN: "unified memory",
}


@dataclass(frozen=True)
class Candidate:
    """One concrete thing that could satisfy a class on one backend."""

    id: str
    memory_bytes_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "memory_bytes_estimate": self.memory_bytes_estimate}


def _from_catalog(
    load: Callable[[], dict[str, Any]], family: str | None = None
) -> Callable[[str], tuple[Candidate, ...]]:
    """Read one catalog into candidates for a backend, best-first.

    `family` filters `models/` down to one model family — the `[model] family` key
    that is already in the manifests (`qwen3.5`, `qwen3.8`, `dots`), so the class
    table below names a fact the repo already states rather than listing model ids
    that would go stale the day a variant is added.
    """

    def candidates(backend_kind: str) -> tuple[Candidate, ...]:
        found: list[Candidate] = []
        for manifest in load().values():
            if family is not None and manifest.family != family:
                continue
            if not manifest.supports(backend_kind):
                continue
            found.append(
                Candidate(
                    id=manifest.id,
                    memory_bytes_estimate=manifest.spec(
                        backend_kind
                    ).memory_bytes_estimate,
                )
            )
        # Descending by size, then by id. The id is not decoration: every voice in
        # the catalog declares the SAME estimate (Higgs is one engine holding one
        # reservation whatever weights it was started on), and every RVC model
        # declares the same 2.5 GiB, so without a second key the "selected" id
        # would change with dict ordering and two runs on one host would disagree.
        found.sort(key=lambda c: (-c.memory_bytes_estimate, c.id))
        return tuple(found)

    return candidates


@dataclass(frozen=True)
class CapabilityClass:
    """One thing a client can ask this server for, and what satisfies it."""

    #: The name a client asks by, and the key this class is recorded under.
    name: str
    #: Which `[jobs] enable_*` flag this class contributes to. A job type is
    #: enabled when ANY of its classes is, which is why `llm` needs three classes:
    #: a host that can clean and cannot translate is a real and common host.
    job_type: str
    #: What the class is for, in a sentence, for `crucible capability`'s output.
    purpose: str
    #: The noun for the things that satisfy it, so a refusal reads like English.
    noun: str
    #: Every candidate on a backend, best-first. None for a class that never
    #: touches the accelerator.
    candidates: Callable[[str], tuple[Candidate, ...]] | None
    #: Appended to a refusal when nothing fits. This is where a BINARY ruling gets
    #: said out loud, so the operator is not left looking for a smaller variant
    #: that was never going to exist.
    binary_note: str = ""


#: Every capability class this build knows, in report order.
#:
#: `llm` is three classes and not one because the ruling is about the work, not
#: the job type. Owen, 2026-09-13: *"translation is binary per server as well. it
#: should use a 27b to translate. if 27b doesnt fit on the card then it cant
#: translate."* A 12 GB card that cleans happily and cannot translate has to be
#: able to SAY that, and `enable_llm = true` cannot say it.
CLASSES: tuple[CapabilityClass, ...] = (
    CapabilityClass(
        name="echo",
        job_type="echo",
        purpose="the test job type; it never touches the accelerator",
        noun="engines",
        candidates=None,
    ),
    CapabilityClass(
        name="clean",
        job_type="llm",
        purpose="cleanup and the other 9B-class text work",
        noun="qwen3.5 variants",
        candidates=_from_catalog(load_all_manifests, family="qwen3.5"),
        binary_note=(
            "This build ships no 4-bit 9B, so there is nothing smaller to fall "
            "back to (PHASE9-CAPABILITY.md section 1.1)."
        ),
    ),
    CapabilityClass(
        name="translate",
        job_type="llm",
        purpose="translation, which needs a 27B-class model",
        noun="qwen3.8 variants",
        candidates=_from_catalog(load_all_manifests, family="qwen3.8"),
        binary_note=(
            "Translation is binary per server: it needs a 27B and the smallest "
            "this build ships is already 4-bit, so this host cannot translate."
        ),
    ),
    CapabilityClass(
        name="pages",
        job_type="llm",
        purpose="reading page images (the VLM door)",
        noun="page readers",
        candidates=_from_catalog(load_all_manifests, family="dots"),
    ),
    CapabilityClass(
        name="tts",
        job_type="tts",
        purpose="narration",
        noun="voices",
        candidates=_from_catalog(load_all_voices),
        binary_note=(
            "Higgs v3 is not quantized and will not be, so this is not a tuning "
            "choice — the engine is disabled on this accelerator."
        ),
    ),
    CapabilityClass(
        name="asr",
        job_type="asr",
        purpose="transcription",
        noun="whisper models",
        candidates=_from_catalog(load_all_asr_manifests),
    ),
    CapabilityClass(
        name="align",
        job_type="align",
        purpose="forced alignment",
        noun="aligners",
        candidates=_from_catalog(load_all_align_manifests),
    ),
    CapabilityClass(
        name="rvc",
        job_type="rvc",
        purpose="voice conversion",
        noun="RVC models",
        candidates=_from_catalog(load_all_rvc_manifests),
    ),
)

#: By name, for a lookup that refuses rather than returns None on a typo.
BY_NAME: dict[str, CapabilityClass] = {entry.name: entry for entry in CLASSES}


def classes_for_job_type(job_type: str) -> tuple[CapabilityClass, ...]:
    """Every class whose verdict feeds one `enable_*` flag."""
    return tuple(entry for entry in CLASSES if entry.job_type == job_type)


@dataclass(frozen=True)
class Decision:
    """What the walk decided about one class on one host, and the numbers it used."""

    capability: str
    job_type: str
    enabled: bool
    #: The candidate that won, or "" when none did. Never None on the wire: the
    #: config's TOML has no null, and an absent key would read as "not recorded".
    selected: str
    reason: str
    #: How much more memory the SMALLEST candidate would have needed, or 0. This
    #: is the number that turned the class off, and it is stored as a number and
    #: not only inside the sentence, because a sentence is not load-bearing
    #: (ARCHITECTURE.md R4).
    shortfall_bytes: int
    available_bytes: int
    candidates: tuple[Candidate, ...]
    fit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "job_type": self.job_type,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "shortfall_bytes": self.shortfall_bytes,
            "available_bytes": self.available_bytes,
            "fit_count": self.fit_count,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def row(self) -> CapabilityRow:
        return CapabilityRow(
            capability=self.capability,
            enabled=self.enabled,
            selected=self.selected,
            reason=self.reason,
            shortfall_bytes=self.shortfall_bytes,
        )


def available_bytes(total_bytes: int, desktop_allowance_bytes: int) -> int:
    """What a job may have: the pool, less the host's own reserve.

    Never below zero. An allowance larger than the pool is a misconfiguration and
    the arithmetic would otherwise report a NEGATIVE budget, which reads as a
    number rather than as the nonsense it is — `accelerator.unattributed_bytes`
    clamps the mirror image of this for the same reason.
    """
    return max(0, total_bytes - desktop_allowance_bytes)


def decide(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
) -> Decision:
    """Walk one class's candidates best-first and take the first that fits."""
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    pool = POOL_NAME.get(backend_kind)
    if pool is None:
        raise ValueError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(POOL_NAME)}"
        )
    arithmetic = (
        f"{_gib(budget)} available ({_gib(total_bytes)} {pool} less a "
        f"{_gib(desktop_allowance_bytes)} desktop allowance)"
    )

    if entry.candidates is None:
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=True,
            selected="",
            reason=f"always available: {entry.purpose}",
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=(),
            fit_count=0,
        )

    found = entry.candidates(backend_kind)
    if not found:
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=False,
            selected="",
            reason=(
                f"disabled: {entry.purpose} needs {entry.noun}, and this build "
                f"ships none with a {backend_kind} block"
            ),
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=(),
            fit_count=0,
        )

    fitting = [c for c in found if c.memory_bytes_estimate <= budget]
    if fitting:
        best = fitting[0]
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=True,
            selected=best.id,
            reason=(
                f"{best.id} fits: it needs {_gib(best.memory_bytes_estimate)} and "
                f"there is {arithmetic}; {len(fitting)} of {len(found)} "
                f"{entry.noun} fit"
            ),
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=found,
            fit_count=len(fitting),
        )

    smallest = found[-1]
    shortfall = smallest.memory_bytes_estimate - budget
    note = f" {entry.binary_note}" if entry.binary_note else ""
    return Decision(
        capability=entry.name,
        job_type=entry.job_type,
        enabled=False,
        selected="",
        reason=(
            f"disabled: the smallest of {len(found)} {entry.noun} is {smallest.id} "
            f"at {_gib(smallest.memory_bytes_estimate)} and there is only "
            f"{arithmetic} — short by {_gib(shortfall)}.{note}"
        ),
        shortfall_bytes=shortfall,
        available_bytes=budget,
        candidates=found,
        fit_count=0,
    )


def decide_all(
    backend_kind: str, *, total_bytes: int, desktop_allowance_bytes: int
) -> tuple[Decision, ...]:
    """Every class, decided on one host. The order of `CLASSES`."""
    return tuple(
        decide(
            entry,
            backend_kind,
            total_bytes=total_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
        )
        for entry in CLASSES
    )


def record(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    decisions: tuple[Decision, ...],
) -> CapabilityRecord:
    """The decisions, in the shape `config.toml` keeps them."""
    return CapabilityRecord(
        backend_kind=backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        rows=tuple(decision.row() for decision in decisions),
    )


def job_type_enabled(job_type: str, decisions: tuple[Decision, ...]) -> bool:
    """Can this host run ANY of the classes behind one `enable_*` flag?

    The disjunction is the whole reason `llm` has three classes: a 12 GB card that
    cleans and cannot translate is an `llm` server, and says which half it has in
    the per-class rows rather than in prose.
    """
    mine = [d for d in decisions if d.job_type == job_type]
    if not mine:
        raise ValueError(
            f"no capability class feeds {job_type!r}; CLASSES covers "
            f"{sorted({entry.job_type for entry in CLASSES})}"
        )
    return any(d.enabled for d in mine)


__all__ = [
    "BY_NAME",
    "CLASSES",
    "Candidate",
    "CapabilityClass",
    "Decision",
    "available_bytes",
    "classes_for_job_type",
    "decide",
    "decide_all",
    "job_type_enabled",
    "record",
]
