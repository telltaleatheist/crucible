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
from typing import Any, Callable, Mapping

from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from .config import CapabilityRecord, CapabilityRow
from .denoisemodels import load_all_denoise_manifests
from .manifests import BACKEND_ENGINES, load_all_manifests
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
    LLAMA_WINDOWS: "card",
}

#: What `llama-windows` is measuring when there is no card: the machine's RAM,
#: which is where a GGUF on the CPU allocates from. Keyed on the GPU VENDOR
#: rather than on the backend, because one backend kind serves both machines
#: and the pool is the only thing that differs (PHASE15-HOST.md section 3.10:
#: *"no NVIDIA → enabled: true with the reason 'cpu build — slow'"*).
CPU_VENDOR = "cpu"
CPU_POOL_NAME = "system memory"

#: The job types that need the WSL2 engine and will never run natively on
#: Windows: they are PYTHON environments (narrator, whisper, the aligner, urvc,
#: the separator), not llama.cpp. PHASE15-HOST.md section 3.3 gives all five
#: ONE sentence, so an app shows it once instead of five times.
WSL_ONLY_JOB_TYPES: frozenset[str] = frozenset(
    {"tts", "asr", "align", "rvc", "denoise"}
)

NEEDS_WSL_REASON = (
    "this job type needs the WSL2 engine (vLLM/SGLang); install it from the "
    "console"
)

#: What a class answers on a Windows box with no NVIDIA card. Owen: *"a
#: crucible server will run on absolutely anything."* Nothing refuses it — the
#: sentence is the warning, and the row stays enabled.
CPU_BUILD_REASON = (
    "cpu build — slow; the model runs on this machine's CPU"
)

#: Said in front of a class's LOCAL sentence when the operator has routed it
#: upstream. The local answer is kept whole after it (section 3.3), so routing
#: back loses nothing and a reader can see what this host would do on its own.
LOCAL_ANSWER_PREFIX = "the local answer would be: "


@dataclass(frozen=True)
class Candidate:
    """One concrete thing that could satisfy a class on one backend."""

    id: str
    memory_bytes_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "memory_bytes_estimate": self.memory_bytes_estimate}


@dataclass(frozen=True)
class CatalogCandidates:
    """One catalog, read into candidates for a backend, best-first.

    `family` filters `models/` down to one model family — the `[model] family` key
    that is already in the manifests (`qwen3.5`, `qwen3.8`, `dots`), so the class
    table below names a fact the repo already states rather than listing model ids
    that would go stale the day a variant is added.

    A class with fields rather than a closure, because `classes_for_model` below
    has to ask a class WHICH catalog it reads — the lineup Foundry vendors lists
    model manifests only, and must not load the voice catalog to find out that
    `tts` never names a model. A closure cannot be asked; a dataclass can.
    """

    load: Callable[[], dict[str, Any]]
    family: str | None = None

    def __call__(self, backend_kind: str) -> tuple[Candidate, ...]:
        found: list[Candidate] = []
        for manifest in self.load().values():
            if self.family is not None and manifest.family != self.family:
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


def _from_catalog(
    load: Callable[[], dict[str, Any]], family: str | None = None
) -> CatalogCandidates:
    return CatalogCandidates(load, family)


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
    #: May this class's work run somewhere other than this card
    #: (PHASE15-HOST.md section 1)? True for exactly the four chat-shaped `llm`
    #: classes. **Declared here rather than derived from `job_type == "llm"`**,
    #: because `pages` is an `llm` job type and is NOT one of the four: it sends
    #: page IMAGES to a vision model, and "forward it to Anthropic" is a
    #: different feature with a different body that nobody has asked for. A
    #: derivation would have made the two indistinguishable and routed the VLM
    #: the first time somebody typed the wrong class name.
    routable: bool = False


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
        routable=True,
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
        routable=True,
        purpose="translation, which needs a 27B-class model",
        noun="qwen3.8 variants",
        candidates=_from_catalog(load_all_manifests, family="qwen3.8"),
        binary_note=(
            "Translation is binary per server: it needs a 27B and the smallest "
            "this build ships is already 4-bit, so this host cannot translate."
        ),
    ),
    # THREE ACTS, ONE MODEL, THREE CLASSES. `simplify` and `analysis` select the
    # same 27B `translate` does and will answer identically on every card this
    # build knows — and they are still separate classes, on Owen's ruling of
    # 2026-09-13: *"they can't lie to the user and say a translate job is running
    # when it's actually a simplify job. It must accurately represent the job
    # that's running. Previously, before crucible, everything ran under
    # translate."*
    #
    # That is the naming half, and it is the half that decides this table.
    # Folding them into one class would have made a client ask about `translate`
    # in order to learn whether it may simplify, which puts the old lie back at
    # the API boundary — and naming an umbrella after one of its members is
    # exactly how "everything ran under translate" happened in the first place.
    #
    # The usual objection to a duplicated axis — that a field nothing selects on
    # differently is a field that will drift — does not hold here, twice over.
    # These are three DIFFERENT facts that share an answer today, not one fact
    # with three owners; nothing can disagree with anything. And they can already
    # be seen to diverge: `analysis` needs guided decoding (a `response_format`
    # schema) and `pages` needs vision, neither of which is a memory question, so
    # the day a backend serves the 27B without guided decoding this table has
    # somewhere to say so.
    CapabilityClass(
        name="simplify",
        job_type="llm",
        routable=True,
        purpose="simplification, which runs on the same 27B translation needs",
        noun="qwen3.8 variants",
        candidates=_from_catalog(load_all_manifests, family="qwen3.8"),
        binary_note=(
            "Simplification is binary per server for translation's reason: it "
            "needs a 27B and the smallest this build ships is already 4-bit."
        ),
    ),
    CapabilityClass(
        name="analysis",
        job_type="llm",
        routable=True,
        purpose="structured analysis answers, on the same 27B",
        noun="qwen3.8 variants",
        candidates=_from_catalog(load_all_manifests, family="qwen3.8"),
        binary_note=(
            "Analysis is binary per server for translation's reason: it needs a "
            "27B and the smallest this build ships is already 4-bit."
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
    # Its own class even though it shares the rvc ENV, because a class is about
    # what the CARD can hold and the two hold different things: a 913 MB
    # separator and a 2.5 GiB urvc stack are different arithmetic, and a host
    # that can denoise and cannot convert is a host that should say so.
    CapabilityClass(
        name="denoise",
        job_type="denoise",
        purpose="noise removal and stem separation",
        noun="separator models",
        candidates=_from_catalog(load_all_denoise_manifests),
    ),
)

#: By name, for a lookup that refuses rather than returns None on a typo.
BY_NAME: dict[str, CapabilityClass] = {entry.name: entry for entry in CLASSES}

#: The classes a route may name, in report order. Read off the table's own
#: `routable` field, so `[routes]`, `PUT /v1/settings` and the operator page
#: all ask ONE thing which classes those are (ARCHITECTURE.md R1). The day a
#: fifth becomes routable, the flag moves and every door follows.
ROUTABLE_CLASSES: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.routable
)

#: The classes an APP may choose a local model for, in report order: every one
#: with candidates to choose between. Read off the same table as
#: `ROUTABLE_CLASSES` and for the same reason — `[local_models]`,
#: `PUT /v1/settings` and the settings document must ask ONE thing which
#: classes those are (ARCHITECTURE.md R1). `echo` is absent because it has no
#: candidates, not because it was left out by hand.
SELECTABLE_CLASSES: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.candidates is not None
)


def classes_for_job_type(job_type: str) -> tuple[CapabilityClass, ...]:
    """Every class whose verdict feeds one `enable_*` flag."""
    return tuple(entry for entry in CLASSES if entry.job_type == job_type)


def classes_for_model(model_id: str) -> tuple[str, ...]:
    """Every class a MODEL manifest can satisfy, on any backend, in report order.

    Read off the class table and not off the manifest: which classes a model
    serves is decided by `CLASSES` — its family filter, its catalog — and nowhere
    else, so the lineup Foundry vendors (`crucible/lineup.py`) asks here rather
    than carrying a `classes = [...]` key in the manifest that would be a second
    owner of the same fact (ARCHITECTURE.md R1).

    The union over backends, because a class is about what a model is FOR:
    `dots-ocr` reads pages whether or not this host is the one with its
    cuda-linux block, and the machine the lineup describes has no Crucible at all.
    Only the classes that read `models/` are walked — the voice, whisper, aligner
    and RVC catalogs are different namespaces and a model id can never appear in
    them. An id the catalog does not hold is refused by name rather than answered
    with an empty tuple that reads as "a model with no class".
    """
    catalog = load_all_manifests()
    if model_id not in catalog:
        raise ValueError(
            f"{model_id!r} is not a model in this build's catalog; it ships "
            f"{sorted(catalog)}"
        )
    names: list[str] = []
    for entry in CLASSES:
        source = entry.candidates
        if not isinstance(source, CatalogCandidates):
            continue
        if source.load is not load_all_manifests:
            continue
        served = {c.id for kind in BACKEND_ENGINES for c in source(kind)}
        if model_id in served:
            names.append(entry.name)
    return tuple(names)


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


def pool_name(backend_kind: str, gpu_vendor: str) -> str:
    """What the pool this class is measured against is CALLED.

    Two inputs because one backend serves two machines: `llama-windows` on a
    box with an NVIDIA card is measuring the card, and on a box without one it
    is measuring the machine's RAM, which is where a GGUF on the CPU really
    allocates from. Calling both "card" would put a lie in every reason string
    on a CPU-only host, the way "card" was already a lie on a Mac.
    """
    if gpu_vendor == CPU_VENDOR:
        return CPU_POOL_NAME
    pool = POOL_NAME.get(backend_kind)
    if pool is None:
        raise ValueError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(POOL_NAME)}"
        )
    return pool


def decide(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: str | None,
) -> Decision:
    """Walk one class's candidates best-first and take the first that fits.

    `gpu_vendor` is REQUIRED and has no default, because on `llama-windows` it
    is the difference between two true answers and there is no safe guess: a
    machine with no card told it has one would report a 24 GiB pool that does
    not exist, and one with a card told it has none would say "slow" about a
    4090. Every caller has a `Backend` in hand.
    """
    if backend_kind == LLAMA_WINDOWS and entry.job_type in WSL_ONLY_JOB_TYPES:
        # THE FIVE PYTHON JOB TYPES, and they are off for a reason that has
        # nothing to do with the card: narrator, whisper, the aligner, urvc and
        # the separator are Python environments, and this backend is
        # llama.cpp. One sentence for all five (section 3.3), so an app shows
        # it once rather than printing five variations of "install WSL".
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=False,
            selected="",
            reason=NEEDS_WSL_REASON,
            shortfall_bytes=0,
            available_bytes=available_bytes(total_bytes, desktop_allowance_bytes),
            candidates=(),
            fit_count=0,
        )
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    pool = pool_name(backend_kind, gpu_vendor)
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

    # ON A CARDLESS WINDOWS BOX THE ROW STILL LIGHTS, and says what it will
    # cost. Owen: *"a crucible server will run on absolutely anything."* The
    # sentence is the warning; the arithmetic is unchanged, because RAM is a
    # real limit and a 27B in 8 GB does not run slowly, it thrashes.
    cpu_note = (
        f" {CPU_BUILD_REASON}."
        if backend_kind == LLAMA_WINDOWS and gpu_vendor == CPU_VENDOR
        else ""
    )
    fitting = [c for c in found if c.memory_bytes_estimate <= budget]

    if chosen is not None:
        # AN APP'S OWN CHOICE, and the reason the best-first walk below is not
        # the only way a class gets its model. INTENT.md gives the app the
        # choice of its models; what stays HERE is the arithmetic, because
        # whether a model fits is a fact about this card and no app can know it
        # from where it sits. A choice is honoured or it is refused with the
        # numbers — it is never quietly replaced by a different model, which
        # would make the settings document a suggestion.
        picked = next((c for c in found if c.id == chosen), None)
        if picked is None:
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {chosen} was chosen for {entry.name}, and it is "
                    f"not among the {len(found)} {entry.noun} this build ships "
                    f"with a {backend_kind} block"
                ),
                shortfall_bytes=0,
                available_bytes=budget,
                candidates=found,
                fit_count=len(fitting),
            )
        if picked.memory_bytes_estimate > budget:
            # The settings door refuses a choice that does not fit, so reaching
            # here means the MACHINE changed under a choice that did fit when it
            # was made — a config carried to a smaller card, or a desktop
            # allowance raised since. Say that, rather than silently demoting to
            # something that fits and leaving an app to wonder why its model
            # never runs.
            shortfall = picked.memory_bytes_estimate - budget
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {picked.id} was chosen for {entry.name} and needs "
                    f"{_gib(picked.memory_bytes_estimate)}, and there is only "
                    f"{arithmetic} — short by {_gib(shortfall)}. This choice fit "
                    f"the machine it was made on{cpu_note}"
                ),
                shortfall_bytes=shortfall,
                available_bytes=budget,
                candidates=found,
                fit_count=len(fitting),
            )
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=True,
            selected=picked.id,
            reason=(
                f"{picked.id} was chosen for {entry.name}: it needs "
                f"{_gib(picked.memory_bytes_estimate)} and there is {arithmetic}; "
                f"{len(fitting)} of {len(found)} {entry.noun} fit{cpu_note}"
            ),
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=found,
            fit_count=len(fitting),
        )

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
                f"{entry.noun} fit{cpu_note}"
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
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: Mapping[str, str],
) -> tuple[Decision, ...]:
    """Every class, decided on one host. The order of `CLASSES`.

    `chosen` is the app selections — class name to model id — and a class
    absent from it is decided best-first. It is REQUIRED and has no default
    for the reason `gpu_vendor` is: a caller that forgot it would silently
    un-choose every model an app had picked, while the config went on saying
    otherwise, and the two would disagree with nothing comparing them.
    """
    return tuple(
        decide(
            entry,
            backend_kind,
            total_bytes=total_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
            gpu_vendor=gpu_vendor,
            chosen=chosen.get(entry.name),
        )
        for entry in CLASSES
    )


def routed_row(row: CapabilityRow, model: str) -> CapabilityRow:
    """One class's row, as it reads once the operator has routed it upstream.

    PHASE15-HOST.md section 3.3. `selected` becomes the upstream model id —
    *"which is exactly the `model` an app sends to `/v1/openai/chat/completions`"*
    — and `enabled` becomes true, because `PUT /v1/settings` refuses to store a
    route whose upstream is unconfigured, so a stored route is always servable.

    **The local sentence is kept whole**, after `the local answer would be: `.
    Nothing is lost when the operator routes back: the row is rebuilt from
    `decide()` on the next write, and until then a reader can still see what
    this host would do on its own. `shortfall_bytes` is zeroed with the same
    honesty — nothing is short of anything when the work is not on this card.
    """
    return CapabilityRow(
        capability=row.capability,
        enabled=True,
        selected=model,
        reason=(
            f"routed to {model.partition('/')[0]}; "
            f"{LOCAL_ANSWER_PREFIX}{row.reason}"
        ),
        shortfall_bytes=0,
    )


def record(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    decisions: tuple[Decision, ...],
    routes: dict[str, str],
) -> CapabilityRecord:  # noqa: D401 - the docstring below is the contract
    """The decisions, in the shape `config.toml` keeps them, routes applied.

    `routes` is REQUIRED and has no default, which is the whole point of it
    being a parameter. A default of `{}` would mean every caller that forgot it
    silently unrouted the server on its next capability write — and the callers
    are `crucible capability --write`, `crucible install` and
    `PUT /v1/settings`, all three of which rewrite the record of a server an
    operator may have routed hours ago (PHASE15-HOST.md section 2: capability
    is recomputed and re-written on every settings write that touches a route).
    """
    rows = []
    for decision in decisions:
        row = decision.row()
        model = routes.get(decision.capability)
        rows.append(row if model is None else routed_row(row, model))
    return CapabilityRecord(
        backend_kind=backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        rows=tuple(rows),
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
    "CPU_BUILD_REASON",
    "CPU_POOL_NAME",
    "CPU_VENDOR",
    "LOCAL_ANSWER_PREFIX",
    "NEEDS_WSL_REASON",
    "WSL_ONLY_JOB_TYPES",
    "pool_name",
    "ROUTABLE_CLASSES",
    "SELECTABLE_CLASSES",
    "routed_row",
    "Candidate",
    "CapabilityClass",
    "CatalogCandidates",
    "Decision",
    "available_bytes",
    "classes_for_job_type",
    "classes_for_model",
    "decide",
    "decide_all",
    "job_type_enabled",
    "record",
]
