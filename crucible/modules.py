"""An app's module file, written from these manifests. Never typed by hand.

PHASE13-OPERATOR.md section 5.4. A **module** is an app's statement of what it
needs from a server — job types plus subjects — which it POSTs to
`POST /v1/tasks {type: "module"}` and a Crucible then makes true.

WHY THE FILE IS GENERATED, WHICH IS FOUNDRY'S OWN OBJECTION
------------------------------------------------------------
Foundry already vendors `foundry-lineup.json` from these manifests, for the
reason ARCHITECTURE.md R1 gives: *"what can this machine run"* has one owner.
A module file typed beside it would restate the same model ids —
`qwen3.8-27b-4bit` appears in both — with nothing comparing them. Rename a
manifest and the lineup is regenerated and the module is not, and a "Set up
for Foundry" button then asks a server for a subject that does not exist. That
is every row of ARCHITECTURE.md's table, introduced on purpose, in the one file
whose whole job is to be correct about ids.

So an app declares only what Crucible cannot know — which job types it uses,
which capability CLASSES it uses, and any subject it names outright — and this
module resolves every one of those against the catalog and refuses by name if
it cannot. `scripts/gen-modules.py --check` is the guard, beside the lineup's.

WHAT A CLASS RESOLVES TO, AND WHEN THE DECLARATION MUST SAY
------------------------------------------------------------
Three rules, in order, and the third is a refusal on purpose:

1. **A class with a FLOOR** (`crucible/lineup.py`'s `floors`, from a manifest's
   `[local] minimum_for`) resolves to the floor. A floor is by definition the
   smallest model the class may run on at all, which is exactly the one an app
   must have pulled.
2. **A class with exactly ONE candidate** resolves to it. `clean` is such a
   class today, and `pages`.
3. **A class with several candidates and no floor must be NAMED.** `analysis`
   is such a class: both 27Bs serve it and neither floors it. A generator that
   picked — the largest, the smallest, the first — would be inventing a policy
   nobody wrote down, and would change its answer the day a variant shipped.
   The declaration says `model = "qwen3.8-27b-4bit"`, and this checks that the
   model really serves the class it was named for.

`[[needs]]` resolves MODEL subjects only, which is the classes whose candidates
read `models/`. `tts`, `asr`, `align`, `rvc` and `denoise` are classes too, and
each has several candidates that are not models — a voice is not a model and
`crucible/weights.py` keeps their namespaces apart on disk for good reason — so
a need for one of those is refused with the advice to name the subject. That is
not a gap: an app choosing *which* voice is an app making a choice, and a
generator that picked the largest voice in the catalog would be choosing a
narrator.

THE VERSION IS DERIVED
----------------------
`<crucible version>+<12 hex of the sha-256 of the module's own content, version
excluded>`. Two apps at one Crucible version whose needs differ get different
versions; a regeneration that changes nothing keeps the same one. A hand-typed
semver on a generated file is a number somebody forgets to bump.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from . import VERSION, catalog, lineup
from .capability import BY_NAME, CLASSES, WSL_ONLY_JOB_TYPES, CatalogCandidates
from .backend import LLAMA_WINDOWS
from .errors import ApiError, CrucibleError
from .manifests import BACKEND_ENGINES, load_all_manifests
from .tasks import require_installable, require_narrator_engine

#: Where the declarations and the generated files live, relative to the repo.
DIR_NAME = "modules"

#: What a generated file is called, given an app name.
def file_name(app: str) -> str:
    return f"{app}.module.json"


class ModuleError(CrucibleError):
    """A declaration says something the catalog cannot satisfy."""


def _model_classes() -> dict[str, set[str]]:
    """Every capability class that selects a MODEL, and the ids that serve it.

    Read off `CLASSES` the way `capability.classes_for_model` does, and for its
    reason: which classes a model serves is decided by that table and nowhere
    else. The union is taken over every backend, because a module is posted to
    a Mac and a PC alike and must name the same subjects on both.
    """
    served: dict[str, set[str]] = {}
    for entry in CLASSES:
        source = entry.candidates
        if not isinstance(source, CatalogCandidates):
            continue
        if source.load is not load_all_manifests:
            continue
        served[entry.name] = {
            candidate.id
            for backend_kind in BACKEND_ENGINES
            for candidate in source(backend_kind)
        }
    return served


def resolve_class(capability_class: str, named: str | None) -> str:
    """Which model id a declared class means. Refuses rather than picking."""
    if capability_class not in BY_NAME:
        raise ModuleError(
            f"{capability_class!r} is not a capability class; this build knows "
            f"{sorted(BY_NAME)}"
        )
    served = _model_classes()
    if capability_class not in served:
        raise ModuleError(
            f"the {capability_class!r} class does not select a model — it selects "
            f"{BY_NAME[capability_class].noun}, which are their own namespace. "
            f"Name what you want with a [[subjects]] entry instead; a generator "
            f"picking one of them would be choosing on the app's behalf"
        )
    candidates = served[capability_class]
    if not candidates:
        raise ModuleError(
            f"nothing in this build's catalog serves the {capability_class!r} "
            "class, so a module cannot ask for it"
        )

    if named is not None:
        if named not in candidates:
            raise ModuleError(
                f"{named!r} does not serve the {capability_class!r} class; the "
                f"models that do are {sorted(candidates)}"
            )
        return named

    floor = lineup.floors(lineup.build()[0]).get(capability_class)
    if floor is not None:
        return floor
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ModuleError(
        f"the {capability_class!r} class is served by {sorted(candidates)} and "
        f"none of them declares itself the floor, so which one an app needs is "
        f"the app's choice and not this generator's. Say it in the declaration: "
        f'`model = "{sorted(candidates)[0]}"`'
    )


def declared_models() -> list[str]:
    """Every model id this BUILD declares, on any backend. Sorted."""
    return catalog.declared_ids()["model"]


def check_class(capability_class: str) -> None:
    """Is this a class a module may name? Refuses, and resolves NOTHING.

    PHASE15-HOST.md 5.3a. Two of `resolve_class`'s three questions are facts
    about this CHECKOUT and are the same on every machine — a class this
    build does not have, and a class that selects no model — so they are
    asked here, at generation, where the answer can be a red build rather
    than a refused install. The third — WHICH id serves it — is a fact about
    a machine's card and is the server's alone.
    """
    if capability_class not in BY_NAME:
        raise ModuleError(
            f"{capability_class!r} is not a capability class; they are "
            f"{sorted(BY_NAME)}"
        )
    served = _model_classes()
    if capability_class not in served:
        raise ModuleError(
            f"the {capability_class!r} class does not select a model — it "
            "selects a voice, an aligner or nothing at all. Name what it "
            "needs under [[subjects]] instead"
        )


def read_declaration(path: Path) -> dict[str, Any]:
    """Parse one `modules/<app>.toml`, or refuse naming the file."""
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise ModuleError(f"could not read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ModuleError(f"{path.name}: not valid TOML: {exc}") from exc
    unknown = sorted(set(document) - {"module", "job_types", "needs", "subjects"})
    if unknown:
        raise ModuleError(
            f"{path.name}: unknown top-level key(s) {unknown}; a declaration "
            "carries [module], [[job_types]], [[needs]] and [[subjects]]"
        )
    block = document.get("module")
    if not isinstance(block, dict) or not isinstance(block.get("name"), str):
        raise ModuleError(f"{path.name}: [module] needs a string `name`")
    if sorted(set(block) - {"name"}):
        raise ModuleError(
            f"{path.name}: [module] takes only `name`. The version is DERIVED "
            "from the content — see crucible/modules.py"
        )
    return document


def build(declaration: dict[str, Any], where: str) -> dict[str, Any]:
    """One declaration, resolved into the module document an app vendors.

    Every id in the result came out of the catalog rather than out of the
    declaration, except the ones a `[[subjects]]` entry names — and those are
    checked to exist, in the right kind, before they are written.
    """
    name = declaration["module"]["name"]

    job_types: list[dict[str, Any]] = []
    for index, raw in enumerate(declaration.get("job_types", [])):
        at = f"{where}: job_types[{index}]"
        if not isinstance(raw, dict):
            raise ModuleError(f"{at}: must be a table with a `type`")
        unknown = sorted(set(raw) - {"type", "narrator_engine"})
        if unknown:
            raise ModuleError(f"{at}: unknown key(s) {unknown}")
        job_type = raw.get("type")
        engine = raw.get("narrator_engine")
        if not isinstance(job_type, str):
            raise ModuleError(f"{at}: `type` must be a string")
        if engine is not None and not isinstance(engine, str):
            raise ModuleError(f"{at}: `narrator_engine` must be a string")
        try:
            # THE SERVER'S OWN CHECKS, not a second copy of them. A declaration
            # that names a type no installer builds, or a `tts` with no engine,
            # is refused here with the sentence `POST /v1/tasks` would have used
            # — which is the difference between finding it at generation and
            # finding it when somebody presses "Set up for BookForge".
            require_installable(job_type)
            require_narrator_engine(job_type, engine)
        except ApiError as exc:
            raise ModuleError(f"{at}: {exc.message}") from None
        entry: dict[str, Any] = {"type": job_type}
        if engine is not None:
            entry["narrator_engine"] = engine
        # The same core capability policy that reports native Windows support
        # scopes app installation plans. Clients strip this generated metadata
        # before POST, just as they strip subjects[].backends.
        entry["backends"] = sorted(
            backend for backend in BACKEND_ENGINES
            if backend != LLAMA_WINDOWS or job_type not in WSL_ONLY_JOB_TYPES
        )
        job_types.append(entry)

    declared = catalog.declared_ids()
    subjects: list[dict[str, Any]] = []

    def add(kind: str, subject_id: str) -> None:
        # Deduplicated, in first-declared order. Two classes can floor on one
        # model — `translate` and `simplify` both floor on the 4-bit 27B — and
        # a module that named it twice would make the server run two steps
        # where the second is always skipped.
        #
        # `backends` ON EVERY ENTRY, not only the narrowing ones.
        #
        # The assumption this replaces is a few lines up — "a module is posted
        # to a Mac and a PC alike and must name the same subjects on both" —
        # and it stopped being true at the transcribers: CTranslate2 has no
        # Metal backend, so `faster-whisper-*` was cuda-linux and
        # `mlx-whisper-*` mlx-darwin, and a module naming only the first was
        # one the Mac refused WHOLE (`invalid_module`, measured 2026-09-15).
        # Since Owen's asr lineup ruling of 2026-09-24 every transcriber is one
        # id on both backends, but a subject can still exist on one machine
        # only (the Mac's 8-bit 27B), so the field is still owed.
        #
        # Written uniformly because the alternative is a rule about when it
        # appears, and every such rule needs a judgment about which backends
        # "count" — `llama-windows` declares `rvc-base` and can serve none of
        # the job types in this file, so "is it on all three?" answers the
        # wrong question. A derived field on every row needs no such judgment
        # and is read by a machine, which does not mind the repetition.
        #
        # THE KEY NEVER REACHES A SERVER. `tasks.validate_module` refuses a
        # subject carrying an unknown key, so an app filters on this and strips
        # it before posting — which is also what keeps every Crucible already
        # installed able to read the files this generator writes today.
        entry: dict[str, Any] = {
            "kind": kind,
            "id": subject_id,
            "backends": catalog.backends_declaring(kind, subject_id),
        }
        if entry not in subjects:
            subjects.append(entry)

    # NEEDS TRAVEL AS CLASSES, UNRESOLVED (PHASE15-HOST.md 5.3a, found
    # 2026-09-14 by Foundry against the Mac). This generator used to resolve a
    # class to ONE id here — the cuda-linux answer, because it runs on a PC —
    # and post it to every machine. The Mac then refused the WHOLE module
    # `unknown_subject` (dots-ocr has no mlx-darwin block), and even where it
    # did not, `qwen3.8-27b-4bit` is not what the Mac's capability selected
    # (`qwen3.8-27b`). The generator was a second owner of a decision that is
    # the SERVER's: PHASE9 says the capability record is the one place a class
    # is resolved, and the record is per machine.
    #
    # So this half only CHECKS. A class the build does not have, or one that
    # selects no model, is refused here — those are facts about this
    # checkout and are the same on every machine. Which id serves it is not.
    needs: list[dict[str, str]] = []
    for index, raw in enumerate(declaration.get("needs", [])):
        at = f"{where}: needs[{index}]"
        if not isinstance(raw, dict):
            raise ModuleError(f"{at}: must be a table with a `class`")
        unknown = sorted(set(raw) - {"class", "model"})
        if unknown:
            raise ModuleError(f"{at}: unknown key(s) {unknown}")
        capability_class = raw.get("class")
        named = raw.get("model")
        if not isinstance(capability_class, str):
            raise ModuleError(f"{at}: `class` must be a string")
        if named is not None and not isinstance(named, str):
            raise ModuleError(f"{at}: `model` must be a string")
        try:
            check_class(capability_class)
        except ModuleError as exc:
            raise ModuleError(f"{at}: {exc}") from None
        if named is not None:
            # A `model` beside a class is an app OVERRIDING the resolution,
            # which is an explicit choice and therefore an explicit subject.
            # It is checked to exist in SOME backend's block, like every other
            # named id, and it stops being a class on the wire.
            if named not in declared_models():
                raise ModuleError(
                    f"{at}: this build has no model called {named!r}; it ships "
                    f"{declared_models()}"
                )
            add("model", named)
            continue
        entry = {"class": capability_class}
        if entry not in needs:
            needs.append(entry)

    for index, raw in enumerate(declaration.get("subjects", [])):
        at = f"{where}: subjects[{index}]"
        if not isinstance(raw, dict):
            raise ModuleError(f"{at}: must be a table with `kind` and `id`")
        unknown = sorted(set(raw) - {"kind", "id"})
        if unknown:
            raise ModuleError(f"{at}: unknown key(s) {unknown}")
        kind, subject_id = raw.get("kind"), raw.get("id")
        if not isinstance(kind, str) or not isinstance(subject_id, str):
            raise ModuleError(f"{at}: `kind` and `id` must both be strings")
        if kind not in declared:
            raise ModuleError(
                f"{at}: {kind!r} is not a subject kind; they are "
                f"{list(catalog.KINDS)}"
            )
        if subject_id not in declared[kind]:
            raise ModuleError(
                f"{at}: this build has no {kind} called {subject_id!r}; it ships "
                f"{declared[kind]}"
            )
        add(kind, subject_id)

    if not job_types and not subjects and not needs:
        raise ModuleError(
            f"{where}: this declaration asks for nothing. An app that needs "
            "nothing from a server needs no module"
        )
    body = {
        "name": name,
        "job_types": job_types,
        "needs": needs,
        "subjects": subjects,
    }
    return {"name": name, "version": version_of(body), **body}


def version_of(body: dict[str, Any]) -> str:
    """`<crucible version>+<content hash>`. Derived, never typed.

    The hash is over the document WITHOUT its version, which is the only way a
    content hash can live inside the content it hashes. Sorted keys and no
    whitespace, so the digest depends on what the module SAYS and not on how
    this function happened to lay it out.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{VERSION}+{digest}"


def render(document: dict[str, Any]) -> str:
    """The bytes that go in the file. `crucible/lineup.py`'s shape, for its reason."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def check(existing_text: str, fresh: dict[str, Any]) -> list[str]:
    """How the checked-in file differs from a fresh build, as sentences.

    Empty means they agree. Unlike the lineup's check nothing is ignored: a
    module carries no provenance key, and its version is derived from its own
    content, so a drifted version IS a drifted module.
    """
    try:
        existing = json.loads(existing_text)
    except json.JSONDecodeError as exc:
        return [f"the checked-in file is not valid JSON: {exc}"]
    if not isinstance(existing, dict):
        return ["the checked-in file is not a JSON object"]
    problems: list[str] = []
    for key in sorted(set(existing) | set(fresh)):
        if existing.get(key) != fresh.get(key):
            problems.append(
                f"{key}: checked in {json.dumps(existing.get(key))}, "
                f"generator says {json.dumps(fresh.get(key))}"
            )
    return problems


__all__ = [
    "DIR_NAME",
    "ModuleError",
    "build",
    "check",
    "check_class",
    "declared_models",
    "file_name",
    "read_declaration",
    "render",
    "resolve_class",
    "version_of",
]
