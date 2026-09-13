"""Two edits to somebody else's installed package, which pip cannot express.

PHASE3-TTS.md section 4: the `higgs-v3` tts env needs two site-packages patches
re-applied after any upgrade of its pins, and `crucible doctor` reports them by
name. They are not Crucible's patches and this module does not apply them — it
**checks** them, which is the useful half: an env whose pins all match is
reported ready, and a reader has no way to tell that from an env that will render
every chunk with 240 ms of garbage on the end.

Why the table is duplicated rather than imported
------------------------------------------------
It is a mirror of `electron/tool-paths.ts`'s `HIGGS_PATCHES` in the BookForge
repo, markers and all, and the duplication is deliberate for the reason that file
gives about its own: this is the module every env question goes through, and a
Crucible server must not need a BookForge checkout to answer "is this env
sound". The markers were measured on the certifying box (vllm-omni 0.28.0,
2026-09-05) and are quoted rather than paraphrased, so a `git grep` finds both
copies when one moves.

BOTH PATCHES ARE EDITS TO THE vLLM STACK, WHICH ONLY ONE BACKEND HAS
--------------------------------------------------------------------
`envs/tts/higgs-v3-cuda-linux.txt` pins `vllm` and `vllm-omni`;
`envs/tts/mlx-darwin.txt` pins neither, because Higgs v3 on the Mac is narrator's
in-process MLX backend and there is no served vLLM under it at all. So on
`mlx-darwin` these files are not merely unpatched, they are **not there** — and a
check that reports that as `no_such_file` makes `crucible doctor` call a perfectly
sound Mac unhealthy, which is what it did until 2026-09-13.

The fix is not a backend list in this module. That would be a second copy of a
fact the recipe already states, which is the shape `docs/ARCHITECTURE.md` §1
names. Each patch instead declares the **distribution** it edits, and `check`
takes the pins of the recipe that built the env: a patch whose distribution is
not in that recipe is `not_applicable`, and says so by name rather than being
silently skipped. Add `vllm-omni` to the Mac recipe one day and the check starts
running there on its own, with nothing here to remember to change.

A NAME IN THE SPEC IS STALE, AND THIS IS WHERE IT SHOWS
-------------------------------------------------------
PHASE3-TTS.md section 4 and narrator's own `pyproject.toml` both name
`work/patch_tail_trim.py` as the second patch. It is **retired**: the live script
is `patch_sentinel_filter.py`, and the difference is not cosmetic. The retired
one wrote `_trim_trailing_sentinel_frames`, which is why the check below greps
for `_filter_sentinel_frames` instead — grepping for the helper both scripts
write would certify a band-aided env as patched. The `absent` marker is the other
half of the proof: upstream's one-frame trim `[:, :-1]` occurs twice in the
pristine stage processor and zero times once the filter patch is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Statuses a patch check can report. `stale` is its own answer rather than a
#: kind of `missing`, because an env carrying v1 of a patch looks patched to
#: every marker grep and renders wrongly anyway.
APPLIED = "applied"
MISSING = "missing"
STALE = "stale"
NO_FILE = "no_such_file"
NO_ENV = "no_env"
#: The recipe that built this env does not install the distribution this patch
#: edits, so there is nothing here to patch and nothing wrong. Distinct from
#: `no_such_file`, which means the package SHOULD be here and is not.
NOT_APPLICABLE = "not_applicable"

#: The statuses that are not a problem. `crucible doctor` reads this rather than
#: testing `applied`, because `applied` answers "is the marker in the file" and a
#: patch with no file to be in has no honest answer to that question.
SOUND_STATUSES: frozenset[str] = frozenset({APPLIED, NOT_APPLICABLE})


@dataclass(frozen=True)
class NarratorPatch:
    """One edit, where it lands, and how to tell whether it is there."""

    id: str
    #: The pinned distribution this patch edits, as the recipe names it. A patch
    #: whose distribution the recipe does not install is `not_applicable`.
    distribution: str
    #: Relative to the env's `site-packages`.
    rel_path: str
    #: A string the patched file must contain.
    marker: str
    #: A string the patched file must NOT contain, when there is one. A marker
    #: alone answers "did somebody apply something here"; this answers "and is
    #: the thing it replaced actually gone".
    absent_marker: str | None
    #: Present only in the CURRENT version of the patch. Marker present and this
    #: absent is an env carrying an older one, which is reported STALE.
    stale_marker: str | None
    why: str


NARRATOR_PATCHES: tuple[NarratorPatch, ...] = (
    NarratorPatch(
        id="vllm-negative-token-id",
        distribution="vllm",
        rel_path="vllm/v1/engine/input_processor.py",
        marker="min_input_id != -100",
        absent_marker=None,
        stale_marker=None,
        why=(
            "vLLM 0.28's blanket negative-token-id rejection fires on vllm-omni's "
            "audio placeholder (-100), so every voice-clone request returns HTTP "
            "400 and only the default voice can serve"
        ),
    ),
    NarratorPatch(
        id="higgs-sentinel-filter",
        distribution="vllm-omni",
        rel_path=(
            "vllm_omni/model_executor/stage_input_processors/higgs_audio_v3.py"
        ),
        marker="_filter_sentinel_frames",
        absent_marker="[:, :-1]",
        stale_marker="final=%s, window=%d frames",
        why=(
            "without it every rendered chunk ends with ~240 ms of audible garbage "
            "— the ramp-down sentinels are substituted with codec code 0, which is "
            "a VALID code that decodes to real sound, and only one of the seven "
            "frames they smear across is trimmed"
        ),
    ),
)

#: The narrator engine these belong to. Orpheus's env needs neither: they are
#: both edits to the vllm-omni stack Higgs v3 serves through.
PATCHED_ENGINE = "higgs-v3"


def site_packages(env_dir: Path) -> Path | None:
    """This venv's `site-packages`, or None when there is no venv.

    The python version is GLOBBED rather than assumed, exactly as BookForge's own
    patch scripts do it: the env is built from whatever interpreter the server is
    running under, which is not necessarily 3.11.
    """
    for candidate in sorted((env_dir / "lib").glob("python*/site-packages")):
        if candidate.is_dir():
            return candidate
    return None


def check(env_dir: Path, recipe_pins: dict[str, str]) -> list[dict[str, Any]]:
    """One row per patch: what it is, whether it is in, and what breaks if not.

    `recipe_pins` is `jobenv.recipe_pins(jobenv.recipe_for(spec))` for the env
    being checked — the caller passes it rather than this module reading it,
    because this module must work against a directory in a test with no recipes
    dir at all. A patch whose distribution is absent from those pins is
    `not_applicable`: see the module docstring for why that is read off the
    recipe and not off a backend name.
    """
    packages = site_packages(env_dir)
    rows: list[dict[str, Any]] = []
    for patch in NARRATOR_PATCHES:
        if patch.distribution not in recipe_pins:
            rows.append(
                _row(
                    patch,
                    NOT_APPLICABLE,
                    f"this env's recipe does not install {patch.distribution}, so "
                    f"{patch.rel_path} is not here to patch",
                )
            )
            continue
        if packages is None:
            rows.append(_row(patch, NO_ENV, f"there is no venv at {env_dir}"))
            continue
        target = packages / patch.rel_path
        if not target.is_file():
            rows.append(
                _row(
                    patch,
                    NO_FILE,
                    f"{target} is not there; the env does not hold the package "
                    "this patch edits",
                )
            )
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            rows.append(_row(patch, NO_FILE, f"could not read {target}: {exc}"))
            continue
        if patch.marker not in text:
            rows.append(
                _row(patch, MISSING, f"{patch.marker!r} is not in {patch.rel_path}")
            )
            continue
        if patch.absent_marker is not None and patch.absent_marker in text:
            rows.append(
                _row(
                    patch,
                    STALE,
                    f"{patch.rel_path} carries {patch.marker!r} but still has "
                    f"{patch.absent_marker!r} in it, so what the patch replaces "
                    "is still running",
                )
            )
            continue
        if patch.stale_marker is not None and patch.stale_marker not in text:
            rows.append(
                _row(
                    patch,
                    STALE,
                    f"{patch.rel_path} carries an OLDER version of this patch: "
                    f"{patch.marker!r} is there but {patch.stale_marker!r} is not",
                )
            )
            continue
        rows.append(_row(patch, APPLIED, f"{patch.rel_path} carries {patch.marker!r}"))
    return rows


def _row(patch: NarratorPatch, status: str, detail: str) -> dict[str, Any]:
    return {
        "id": patch.id,
        "status": status,
        "applied": status == APPLIED,
        "path": patch.rel_path,
        "detail": detail,
        "why": patch.why,
    }
