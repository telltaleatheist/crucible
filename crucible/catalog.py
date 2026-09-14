"""`GET /v1/catalog` — every pullable subject this backend can hold.

PHASE13-OPERATOR.md section 3.2. The operator page draws one grid from this and
nothing else, which is the whole reason it exists as one route: before it, a
person answering *"what can this machine hold and what has it got"* read five
`crucible … list` commands with five shapes, and a page would have had to make a
sixth by merging them.

WHAT THIS MODULE IS, AND IS NOT
-------------------------------
It is a READER, in exactly the sense `crucible/lineup.py` is one. Every field on
every row comes from somewhere that already owns it:

| field | owner |
|---|---|
| `installed`, `installed_bytes` | `crucible/weights.py`'s stamp, through each kind's own `installed()` |
| `expected_bytes` | the manifest, where the manifest declares a file's bytes |
| `floors` | `crucible/lineup.py` — the same table `foundry-lineup.json` carries |
| `resident` | `Residency.resident`, which is what `/v1/activity` reports |
| `job_type` | which loader read the manifest: `models/` is `llm`, `asr/` is `asr`, … |
| `source` | the backend block's `hf_repo` |

There is **no table in this file**. A `kind` is a directory of manifests and a
function that says whether its weights are on disk, and that is all a row is
(ARCHITECTURE.md R1: where a copy must exist it is derived, never authored
twice).

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

from typing import Any

from . import denoisemodels, lineup, rvcbase, weights
from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import Backend
from .config import Config
from .errors import ApiError, CrucibleError
from .manifests import load_all_manifests
from .residency import KIND_ALIGN, KIND_LLM, KIND_TTS, Residency
from .rvcmodels import load_all_rvc_manifests
from .voices import load_all_voices

#: The one id `rvc-base` has. PHASE13-OPERATOR.md section 2: the base assets are
#: the ENGINE's, not any model's, and there is one set of them — so the subject
#: is `{kind: "rvc-base", id: "base"}` rather than the declaration's engine id.
#: A client asking for `ultimate-rvc` by name would be naming an implementation
#: detail of which engine this build's `rvc` job runs.
RVC_BASE_ID = "base"

#: Which resident kind, if any, a subject of this kind could BE. `rvc`,
#: `rvc-base` and `denoise` are absent on purpose: nothing of those kinds is ever
#: resident (`crucible/jobs/rvc` and `crucible/jobs/denoise` both run a process
#: that loads, works and exits), so their rows report `resident: false` without
#: asking. Reading them off a residency that can never name them would be a
#: comparison whose answer is structurally fixed, which is how a field becomes
#: wrong the day the structure changes.
_RESIDENT_KIND_FOR_JOB_TYPE: dict[str, str] = {
    "llm": KIND_LLM,
    "align": KIND_ALIGN,
    "tts": KIND_TTS,
}


def _floors_by_model() -> dict[str, list[str]]:
    """Model id to the capability classes it FLOORS, from `crucible/lineup.py`.

    Inverted from `lineup.floors()` rather than read off `[local] minimum_for`
    directly, so that this route and `foundry-lineup.json` cannot come to
    disagree about what floors what — which is the exact drift the `floors` key
    was added to that file to stop (2026-09-14).
    """
    rows, _omitted = lineup.build()
    inverted: dict[str, list[str]] = {}
    for capability_class, model_id in lineup.floors(rows).items():
        inverted.setdefault(model_id, []).append(capability_class)
    return inverted


def _row(
    *,
    kind: str,
    subject_id: str,
    name: str | None,
    job_type: str,
    installed: Any,
    expected_bytes: int | None,
    floors: list[str],
    source: str,
    resident: bool,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": subject_id,
        "name": name,
        "job_type": job_type,
        "installed": installed is not None,
        "installed_bytes": None if installed is None else installed.bytes,
        "expected_bytes": expected_bytes,
        "floors": floors,
        # See the module docstring. Null until a manifest declares one.
        "license": None,
        "source": source,
        "resident": resident,
    }


def rows(config: Config, backend: Backend, residency: Residency) -> list[dict[str, Any]]:
    """Every subject this backend can hold, grouped by kind in the order below.

    Within a kind the order is each loader's, which is id order for all of them
    (`load_all_manifests` and its siblings say so). `model` spans three loaders
    — `models/`, `asr/`, `align/` — so it is three id-ordered runs rather than
    one, which is the honest shape: those are three catalogs that happen to
    share a namespace, not one catalog.

    A subject with no block for this backend is ABSENT rather than listed as
    unsupported: the route's sentence is "what this backend can hold", and a row
    for a Mac-only voice on the PC would have nothing truthful to put in
    `installed`, `expected_bytes` or `source`.

    A manifest that cannot be read at all fails the whole route by name, and
    that is deliberate. A catalog missing one row looks exactly like a catalog
    of a server that does not ship that subject, and the page would draw the
    second while the first was true.
    """
    resident = residency.resident

    def is_resident(job_type: str, subject_id: str) -> bool:
        wanted = _RESIDENT_KIND_FOR_JOB_TYPE.get(job_type)
        if wanted is None or resident is None:
            return False
        return resident.kind == wanted and resident.id == subject_id

    try:
        floors = _floors_by_model()
        out: list[dict[str, Any]] = []

        # --- kind: model. Three directories, one namespace, three job types.
        # `crucible/cli.py`'s `_all_manifests` already treats them as one set of
        # ids for `crucible models pull`; this keeps that and adds the job type
        # each came from, which is the fact the page needs and the CLI does not.
        for job_type, loaded in (
            ("llm", load_all_manifests()),
            ("asr", load_all_asr_manifests()),
            ("align", load_all_align_manifests()),
        ):
            for manifest in loaded.values():
                if not manifest.supports(backend.kind):
                    continue
                spec = manifest.spec(backend.kind)
                out.append(
                    _row(
                        kind="model",
                        subject_id=manifest.id,
                        # ASR and align manifests carry no display name; null is
                        # the honest answer and the page prints the id.
                        name=getattr(manifest, "display", None),
                        job_type=job_type,
                        installed=weights.installed(config, manifest, spec),
                        expected_bytes=None,
                        floors=list(floors.get(manifest.id, [])),
                        source=f"hf:{spec.hf_repo}",
                        resident=is_resident(job_type, manifest.id),
                    )
                )

        # --- kind: voice.
        for voice in load_all_voices().values():
            if not voice.supports(backend.kind):
                continue
            spec = voice.spec(backend.kind)
            out.append(
                _row(
                    kind="voice",
                    subject_id=voice.id,
                    name=voice.display,
                    job_type="tts",
                    installed=weights.installed(config, voice, spec),
                    expected_bytes=None,
                    floors=[],
                    source=f"hf:{spec.hf_repo}",
                    resident=is_resident("tts", voice.id),
                )
            )

        # --- kind: rvc. One archive, whose size the manifest states.
        for model in load_all_rvc_manifests().values():
            if not model.supports(backend.kind):
                continue
            spec = model.spec(backend.kind)
            out.append(
                _row(
                    kind="rvc",
                    subject_id=model.id,
                    name=model.display,
                    job_type="rvc",
                    installed=weights.installed(config, model, spec),
                    expected_bytes=spec.archive_bytes,
                    floors=[],
                    source=f"hf:{spec.hf_repo}",
                    resident=False,
                )
            )

        # --- kind: rvc-base. Exactly one, and backend-independent: these are the
        # engine's shared assets and the same files serve both backends.
        assets = rvcbase.load_rvc_base()
        out.append(
            _row(
                kind="rvc-base",
                subject_id=RVC_BASE_ID,
                name=f"{assets.id}'s base assets",
                job_type="rvc",
                installed=rvcbase.installed(config, assets),
                expected_bytes=assets.total_bytes,
                floors=[],
                source=f"hf:{assets.hf_repo}",
                resident=False,
            )
        )

        # --- kind: denoise. Two named files, whose sizes the manifest states.
        for separator in denoisemodels.load_all_denoise_manifests().values():
            if not separator.supports(backend.kind):
                continue
            spec = separator.spec(backend.kind)
            out.append(
                _row(
                    kind="denoise",
                    subject_id=separator.id,
                    name=separator.display,
                    job_type="denoise",
                    installed=denoisemodels.installed(config.home, separator, spec),
                    expected_bytes=spec.total_bytes,
                    floors=[],
                    source=f"hf:{spec.hf_repo}",
                    resident=False,
                )
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
    return out


__all__ = ["RVC_BASE_ID", "rows"]
