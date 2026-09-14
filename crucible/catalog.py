"""Every pullable subject this backend can hold — the one list, read two ways.

PHASE13-OPERATOR.md sections 2 and 3.2. A **subject** is one pullable thing:
`{kind, id}` where `kind ∈ model, voice, rvc, rvc-base, denoise`. Two callers
want that set and they want different halves of it:

* `GET /v1/catalog` wants a row per subject, for the operator page's grid.
* `POST /v1/tasks {type: "pull"}` wants ONE subject by name, and then wants to
  ask it whether it is installed and to fetch it.

Those are the same set, so they are the same function — `subjects()` — and the
route and the task each take what they need from it. Written the other way
round (a list for the page, a `match kind:` in the task runner) the two would
disagree the first time a sixth kind arrived, and they would disagree silently:
a subject the page can see and no task can pull is a button that does nothing.

WHAT THIS MODULE IS, AND IS NOT
-------------------------------
It is a READER, in exactly the sense `crucible/lineup.py` is one. Every field
comes from somewhere that already owns it:

| field | owner |
|---|---|
| `installed`, `installed_bytes` | `crucible/weights.py`'s stamp, through each kind's own `installed()` |
| `expected_bytes` | the manifest, where the manifest declares a file's bytes |
| `floors` | `crucible/lineup.py` — the same table `foundry-lineup.json` carries |
| `resident` | `Residency.resident`, which is what `/v1/activity` reports |
| `job_type` | which loader read the manifest: `models/` is `llm`, `asr/` is `asr`, … |
| `source` | the backend block's `hf_repo` |

There is **no table of kinds** here beyond the five `subjects()` walks, and each
of those five is a loader plus the two functions that already pull and check it.

THE TWO FIELDS THAT ARE OFTEN NULL, AND WHY THEY STAY ON THE WIRE
------------------------------------------------------------------
**`license` is null on every row this build can produce.** No manifest schema in
the repo carries a licence key — not the model, voice, rvc, rvc-base or denoise
one — so there is nothing to read it from. Deriving "Apache-2.0" from a repo
name would be Crucible making a licensing claim about somebody else's weights,
which is exactly the kind of confident wrongness ARCHITECTURE.md is about. The
field is on the wire so the page and the SDK are built against the shape that
will carry it; the day a licence is wanted, a manifest declares it.

**`expected_bytes` is null for models and voices.** The three kinds that fetch
NAMED files with pinned digests declare their sizes (`archive_bytes` for an RVC
model, the summed `bytes` of the base assets, `total_bytes` for a separator), so
those rows carry a real number. A model or a voice is a `snapshot_download` of a
whole repository and no manifest states its size. The tempting substitute —
`[local] download_bytes` — is the Ollama or GGUF artifact's size, a different
file in a different repo, and lending it here would put a plausible wrong number
in front of somebody about to spend twenty minutes on a download.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import denoisemodels, lineup, llamacpp, rvcbase, weights
from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import Backend
from .config import Config
from .errors import ApiError, CrucibleError
from .manifests import load_all_manifests
from .jobs.base import utcnow
from .residency import KIND_ALIGN, KIND_LLM, KIND_TTS, Residency
from .rvcmodels import load_all_rvc_manifests
from .voices import load_all_voices

#: The six words a subject's `kind` may be, in the order `/v1/catalog` lists
#: them. A tuple and not a set, because the order IS the catalog's order.
#:
#: `engine` arrived with PHASE15-HOST.md 3.10: on `llama-windows` the thing
#: that serves a model is `llama-server.exe`, a zip on a GitHub release, and
#: it is pulled, listed, reported installed and REMOVED exactly like weights
#: rather than through a second mechanism with a second page panel. It is
#: LAST because it exists on one backend and a reader scanning the list
#: should meet the four universal kinds first.
KINDS: tuple[str, ...] = ("model", "voice", "rvc", "rvc-base", "denoise", "engine")

#: The one id `rvc-base` has. PHASE13-OPERATOR.md section 2: the base assets are
#: the ENGINE's, not any model's, and there is one set of them — so the subject
#: is `{kind: "rvc-base", id: "base"}` rather than the declaration's engine id.
#: A client asking for `ultimate-rvc` by name would be naming an implementation
#: detail of which engine this build's `rvc` job runs.
RVC_BASE_ID = "base"

#: Which resident kind, if any, a subject of this job type could BE. `rvc` and
#: `denoise` are absent on purpose: nothing of those kinds is ever resident
#: (both run a process that loads, works and exits), so their rows report
#: `resident: false` without asking. Reading them off a residency that can never
#: name them would be a comparison whose answer is structurally fixed, which is
#: how a field becomes wrong the day the structure changes.
_RESIDENT_KIND_FOR_JOB_TYPE: dict[str, str] = {
    "llm": KIND_LLM,
    "align": KIND_ALIGN,
    "tts": KIND_TTS,
}


@dataclass(frozen=True)
class Subject:
    """One pullable thing, and the two things anybody ever does to it.

    `installed` and `pull` are bound callables rather than a `kind` this module
    switches on twice, because the switch is what would rot: the catalog route
    and the pull task would each carry a copy of "how do you fetch an rvc-base",
    and the second copy is always the one that is not updated.
    """

    kind: str
    id: str
    name: str | None
    job_type: str
    expected_bytes: int | None
    source: str
    #: What a person would type to fetch this by hand. Carried so a refusal can
    #: name it, in the idiom `crucible/weights.py` already uses.
    pull_command: str
    installed: Callable[[], weights.InstalledWeights | None]
    pull: Callable[..., weights.InstalledWeights]
    #: Delete this subject's files and return what went (PHASE15-HOST.md
    #: 3.5a). Bound per subject for `installed` and `pull`'s reason: the
    #: alternative is a third `match kind:` that goes stale on its own.
    remove: Callable[[], Path]


def subjects(config: Config, backend: Backend) -> list[Subject]:
    """Every subject this backend can hold, grouped by kind in `KINDS` order.

    Within a kind the order is each loader's, which is id order for all of them
    (`load_all_manifests` and its siblings say so). `model` spans three loaders
    — `models/`, `asr/`, `align/` — so it is three id-ordered runs rather than
    one, which is the honest shape: those are three catalogs that happen to
    share a namespace, not one catalog.

    A subject with no block for this backend is ABSENT rather than listed as
    unsupported: the sentence this answers is "what this backend can hold", and
    an entry for a Mac-only voice on the PC would have nothing truthful to put
    in `installed`, `expected_bytes` or `source`.
    """
    found: list[Subject] = []

    # --- model. Three directories, one namespace, three job types.
    # `crucible/cli.py`'s `_all_manifests` already treats them as one set of ids
    # for `crucible models pull`; this keeps that and adds the job type each
    # came from, which is the fact the page needs and the CLI does not.
    for job_type, loaded in (
        ("llm", load_all_manifests()),
        ("asr", load_all_asr_manifests()),
        ("align", load_all_align_manifests()),
    ):
        for manifest in loaded.values():
            if not manifest.supports(backend.kind):
                continue
            spec = manifest.spec(backend.kind)
            found.append(
                Subject(
                    kind="model",
                    id=manifest.id,
                    # ASR and align manifests carry no display name; null is the
                    # honest answer and the page prints the id.
                    name=getattr(manifest, "display", None),
                    job_type=job_type,
                    expected_bytes=None,
                    source=f"hf:{spec.hf_repo}",
                    pull_command=f"crucible models pull {manifest.id}",
                    installed=_installed_weights(config, manifest, spec),
                    pull=_pull_weights(config, manifest, spec),
                    remove=_remove_weights(config, manifest, spec),
                )
            )

    for voice in load_all_voices().values():
        if not voice.supports(backend.kind):
            continue
        spec = voice.spec(backend.kind)
        found.append(
            Subject(
                kind="voice",
                id=voice.id,
                name=voice.display,
                job_type="tts",
                expected_bytes=None,
                source=f"hf:{spec.hf_repo}",
                pull_command=f"crucible voices pull {voice.id}",
                installed=_installed_weights(config, voice, spec),
                pull=_pull_weights(config, voice, spec),
                remove=_remove_weights(config, voice, spec),
            )
        )

    for model in load_all_rvc_manifests().values():
        if not model.supports(backend.kind):
            continue
        spec = model.spec(backend.kind)
        found.append(
            Subject(
                kind="rvc",
                id=model.id,
                name=model.display,
                job_type="rvc",
                expected_bytes=spec.archive_bytes,
                source=f"hf:{spec.hf_repo}",
                pull_command=f"crucible rvc pull {model.id}",
                installed=_installed_weights(config, model, spec),
                pull=_pull_archive(config, model, spec),
                remove=_remove_weights(config, model, spec),
            )
        )

    # --- rvc-base. Exactly one, and backend-independent: these are the engine's
    # shared assets and the same files serve both backends.
    assets = rvcbase.load_rvc_base()
    found.append(
        Subject(
            kind="rvc-base",
            id=RVC_BASE_ID,
            name=f"{assets.id}'s base assets",
            job_type="rvc",
            expected_bytes=assets.total_bytes,
            source=f"hf:{assets.hf_repo}",
            pull_command=rvcbase.PULL_COMMAND,
            installed=lambda: rvcbase.installed(config, assets),
            pull=lambda **kwargs: rvcbase.pull(config, assets, **kwargs),
            remove=lambda: weights.remove_files(
                rvcbase.base_root(config), assets.targets
            ),
        )
    )

    # --- engine. ONE, on ONE backend, and absent everywhere else: `cuda-linux`
    # and `mlx-darwin` get their engine from a Python env that `crucible
    # install` builds, and a row here for them would be a second answer to
    # "how does this backend get its engine".
    if backend.kind == llamacpp_backend():
        build = llamacpp.build_for(backend.gpu.vendor)
        found.append(
            Subject(
                kind=llamacpp.ENGINE_KIND,
                id=llamacpp.LLAMA_CPP_ID,
                name=f"llama.cpp {llamacpp.LLAMA_CPP_RELEASE} ({build})",
                # The job type it SERVES. `llm` and not a sixth word: on this
                # backend `pages` is an llm-class model like any other
                # (PHASE3-VLM.md section 1, "there is no vlm-pages job type"),
                # and the engine is what starts both.
                job_type="llm",
                expected_bytes=llamacpp.expected_bytes(build),
                source=f"github:ggml-org/llama.cpp@{llamacpp.LLAMA_CPP_RELEASE}",
                pull_command="crucible install llm",
                installed=_installed_engine(config, build),
                pull=_pull_engine(config, build),
                remove=lambda: llamacpp.remove(config),
            )
        )

    for separator in denoisemodels.load_all_denoise_manifests().values():
        if not separator.supports(backend.kind):
            continue
        spec = separator.spec(backend.kind)
        found.append(
            Subject(
                kind="denoise",
                id=separator.id,
                name=separator.display,
                job_type="denoise",
                expected_bytes=spec.total_bytes,
                source=f"hf:{spec.hf_repo}",
                pull_command=f"{denoisemodels.PULL_COMMAND} {separator.id}",
                installed=_installed_denoise(config, separator, spec),
                pull=_pull_denoise(config, separator, spec),
                remove=_remove_denoise(config, separator),
            )
        )
    return found


# Bound one per subject rather than written inline, because a lambda closing
# over a loop variable is the classic way to build five closures that all
# describe the last manifest.


def _installed_weights(
    config: Config, manifest: Any, spec: Any
) -> Callable[[], weights.InstalledWeights | None]:
    return lambda: weights.installed(config, manifest, spec)


def _pull_weights(
    config: Config, manifest: Any, spec: Any
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: weights.pull(config, manifest, spec, **kwargs)


def _pull_archive(
    config: Config, manifest: Any, spec: Any
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: weights.pull_archive(config, manifest, spec, **kwargs)


def llamacpp_backend() -> str:
    """The one backend the `engine` subject exists on.

    A function rather than an import of `backend.LLAMA_WINDOWS` at the top,
    because `crucible/backend.py` is where that name lives and this module is
    a READER: it asks, it does not hold a copy.
    """
    from .backend import LLAMA_WINDOWS

    return LLAMA_WINDOWS


def _installed_engine(
    config: Config, build: str
) -> Callable[[], weights.InstalledWeights | None]:
    return lambda: llamacpp.installed(config, build)


def _pull_engine(
    config: Config, build: str
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: llamacpp.pull(config, build, **kwargs)


def _remove_weights(
    config: Config, manifest: Any, spec: Any
) -> Callable[[], Path]:
    return lambda: weights.remove(config, manifest, spec)


def _remove_denoise(config: Config, manifest: Any) -> Callable[[], Path]:
    # ONE FLAT DIRECTORY holds every separator, so the SET goes and the
    # directory stays (`crucible/denoisemodels.py` says why there is a stamp
    # per model rather than one at the root).
    return lambda: weights.remove_files(
        denoisemodels.denoise_models_root(config.home),
        (manifest.model_filename, manifest.config_filename),
        stamp_name=denoisemodels.stamp_name(manifest),
    )


def _installed_denoise(
    config: Config, manifest: Any, spec: Any
) -> Callable[[], weights.InstalledWeights | None]:
    return lambda: denoisemodels.installed(config.home, manifest, spec)


def _pull_denoise(
    config: Config, manifest: Any, spec: Any
) -> Callable[..., weights.InstalledWeights]:
    return lambda **kwargs: denoisemodels.pull(config, manifest, spec, **kwargs)


def declared_ids() -> dict[str, list[str]]:
    """Every subject id this BUILD declares, per kind, across all backends.

    `subjects()`' question with the host taken out of it, and it has one
    caller: `crucible/modules.py`, which writes a module file an app posts to
    a Mac and a PC alike. A module that named only what the generating machine
    could hold would be a module that installs less on the other one — so the
    existence check there has to be about the catalog, not about a card.

    It walks the SAME loaders `subjects()` does, in this file, so the day a
    sixth kind arrives both answers change together. What it cannot share is
    the row-building, because there is no backend to read a spec from.
    """
    return {
        "model": sorted(
            {
                *load_all_manifests(),
                *load_all_asr_manifests(),
                *load_all_align_manifests(),
            }
        ),
        "voice": sorted(load_all_voices()),
        "rvc": sorted(load_all_rvc_manifests()),
        "rvc-base": [RVC_BASE_ID],
        "denoise": sorted(denoisemodels.load_all_denoise_manifests()),
        # Declared on every machine, because `declared_ids` is the question
        # with the HOST taken out of it (a module written on a Mac installs
        # on a PC). Whether THIS backend has the row is `subjects()`' answer.
        llamacpp.ENGINE_KIND: [llamacpp.LLAMA_CPP_ID],
    }


def find(
    config: Config, backend: Backend, kind: str, subject_id: str
) -> Subject | None:
    """One subject by `{kind, id}`, or None. The caller names the refusal.

    None rather than a raise, because the two callers refuse differently: a
    pull task answers `404 unknown_subject` and a module's validation collects
    every bad entry before refusing once (`invalid_module`).
    """
    for subject in subjects(config, backend):
        if subject.kind == kind and subject.id == subject_id:
            return subject
    return None


def _floors_by_model() -> dict[str, list[str]]:
    """Model id to the capability classes it FLOORS, from `crucible/lineup.py`.

    Inverted from `lineup.floors()` rather than read off `[local] minimum_for`
    directly, so that this route and `foundry-lineup.json` cannot come to
    disagree about what floors what — which is the exact drift the `floors` key
    was added to that file to stop (2026-09-14).
    """
    built, _omitted = lineup.build()
    inverted: dict[str, list[str]] = {}
    for capability_class, model_id in lineup.floors(built).items():
        inverted.setdefault(model_id, []).append(capability_class)
    return inverted


def rows(config: Config, backend: Backend, residency: Residency) -> list[dict[str, Any]]:
    """`GET /v1/catalog`'s body — `subjects()` with the two live facts added.

    A manifest that cannot be read at all fails the whole route by name, and
    that is deliberate. A catalog missing one row looks exactly like a catalog
    of a server that does not ship that subject, and the page would draw the
    second while the first was true.
    """
    resident = residency.resident

    def is_resident(subject: Subject) -> bool:
        wanted = _RESIDENT_KIND_FOR_JOB_TYPE.get(subject.job_type)
        if wanted is None or resident is None:
            return False
        return resident.kind == wanted and resident.id == subject.id

    try:
        floors = _floors_by_model()
        built: list[dict[str, Any]] = []
        for subject in subjects(config, backend):
            found = subject.installed()
            built.append(
                {
                    "kind": subject.kind,
                    "id": subject.id,
                    "name": subject.name,
                    "job_type": subject.job_type,
                    "installed": found is not None,
                    "installed_bytes": None if found is None else found.bytes,
                    "expected_bytes": subject.expected_bytes,
                    # Only a model can floor a capability class; every other
                    # kind gets the empty list because nothing floors on it, not
                    # because nobody looked.
                    "floors": (
                        list(floors.get(subject.id, []))
                        if subject.kind == "model"
                        else []
                    ),
                    # See the module docstring. Null until a manifest declares one.
                    "license": None,
                    "source": subject.source,
                    "resident": is_resident(subject),
                }
            )
    except ApiError:
        raise
    except CrucibleError as exc:
        raise ApiError(
            503,
            "catalog_unreadable",
            f"this server cannot read its own catalog: {type(exc).__name__}: "
            f"{exc}. Nothing is listed rather than some of it, because a catalog "
            "missing a row is indistinguishable from a build that does not ship "
            "it",
        ) from None
    return built


class Removals:
    """The last few subject removals, for `/v1/activity`. In memory.

    PHASE15-HOST.md 3.5a: *"recorded in `/v1/activity` with the act"*. The
    same shape and the same reasoning as `settings.History` — a display of
    "who deleted what just now" when the host, the page and an operator can
    all reach the same server, not an audit log, and a restart forgets it.
    Deleting several gigabytes is the one catalog act nobody can undo, so it
    is the one that most needs to say who asked.
    """

    #: How many are kept. A person looking at this is asking about the last
    #: few minutes; a log is somebody else's job.
    LIMIT = 20

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def record(
        self,
        *,
        kind: str,
        subject_id: str,
        bytes_freed: int | None,
        act: str | None,
        client: str | None,
    ) -> None:
        row = {
            "at": utcnow(),
            "kind": kind,
            "id": subject_id,
            # What the catalog said the subject weighed before it went. None
            # where the stamp did not record one; never a guess and never a
            # re-measurement of a directory that no longer exists.
            "bytes_freed": bytes_freed,
            # Null means the client did not say, exactly as on a chat row.
            "act": act,
            "client": client,
        }
        with self._lock:
            self._rows.append(row)
            del self._rows[: -self.LIMIT]

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._rows))


__all__ = [
    "KINDS",
    "Removals",
    "RVC_BASE_ID",
    "Subject",
    "declared_ids",
    "find",
    "rows",
    "subjects",
]
