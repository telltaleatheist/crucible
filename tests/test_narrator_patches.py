"""Two edits pip cannot express, and the check that says whether they are in.

PHASE3-TTS.md section 4. An env whose pins all match is reported ready, and a
reader has no way to tell that from an env that will render every chunk with
240 ms of garbage on the end — so the patches are checked, by name, against the
markers BookForge measured on the certifying box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible import narratorpatches
from crucible.narratorpatches import (
    APPLIED,
    MISSING,
    NARRATOR_PATCHES,
    NO_ENV,
    NO_FILE,
    STALE,
)

PRISTINE_INPUT_PROCESSOR = "if min_input_id < 0:\n    raise ValueError('oov')\n"
PATCHED_INPUT_PROCESSOR = "if min_input_id < 0 and min_input_id != -100:\n"

#: A stage processor with the filter in and upstream's one-frame trim gone, plus
#: the field only v2 of the patch logs.
PATCHED_STAGE = (
    "def _filter_sentinel_frames(frames):\n"
    "    ...\n"
    "logger.warning('final=%s, window=%d frames', final, window)\n"
)


def env_with(tmp_path: Path, **files: str) -> Path:
    """A venv-shaped directory holding exactly these site-packages files."""
    packages = tmp_path / "env" / "lib" / "python3.11" / "site-packages"
    for patch in NARRATOR_PATCHES:
        body = files.get(patch.id)
        if body is None:
            continue
        target = packages / patch.rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    packages.mkdir(parents=True, exist_ok=True)
    return tmp_path / "env"


def rows(env: Path) -> dict[str, dict]:
    return {row["id"]: row for row in narratorpatches.check(env)}


def test_both_patches_in_is_the_only_applied_answer(tmp_path: Path) -> None:
    found = rows(
        env_with(
            tmp_path,
            **{
                "vllm-negative-token-id": PATCHED_INPUT_PROCESSOR,
                "higgs-sentinel-filter": PATCHED_STAGE,
            },
        )
    )
    assert [row["status"] for row in found.values()] == [APPLIED, APPLIED]
    assert all(row["applied"] for row in found.values())


def test_an_unpatched_file_is_missing_and_says_what_breaks(tmp_path: Path) -> None:
    found = rows(
        env_with(
            tmp_path,
            **{
                "vllm-negative-token-id": PRISTINE_INPUT_PROCESSOR,
                "higgs-sentinel-filter": PATCHED_STAGE,
            },
        )
    )
    entry = found["vllm-negative-token-id"]
    assert entry["status"] == MISSING
    assert entry["applied"] is False
    assert "HTTP 400" in entry["why"]


def test_the_marker_alone_does_not_certify_the_sentinel_filter(
    tmp_path: Path,
) -> None:
    """Marker present AND the thing it replaces gone is the whole proof.

    `[:, :-1]` is upstream's one-frame trim: it occurs twice in the pristine
    stage processor and zero times once the filter patch is in. A file carrying
    the helper and still carrying the trim is a band-aided file, and grepping for
    the helper alone would certify it.
    """
    found = rows(
        env_with(
            tmp_path,
            **{
                "vllm-negative-token-id": PATCHED_INPUT_PROCESSOR,
                "higgs-sentinel-filter": PATCHED_STAGE + "audio = audio[:, :-1]\n",
            },
        )
    )
    entry = found["higgs-sentinel-filter"]
    assert entry["status"] == STALE
    assert "is still running" in entry["detail"]


def test_v1_of_the_sentinel_patch_is_reported_stale_not_applied(
    tmp_path: Path,
) -> None:
    """v1 substituted sentinels before the identity trim, so the trim found
    nothing and every chunk ended in an audible burst. It looks patched to any
    marker grep, which is why the v2-only log field is checked too."""
    v1 = "def _filter_sentinel_frames(frames):\n    ...\n"
    found = rows(
        env_with(
            tmp_path,
            **{
                "vllm-negative-token-id": PATCHED_INPUT_PROCESSOR,
                "higgs-sentinel-filter": v1,
            },
        )
    )
    entry = found["higgs-sentinel-filter"]
    assert entry["status"] == STALE
    assert "an OLDER version of this patch" in entry["detail"]


def test_a_missing_package_is_told_apart_from_a_missing_patch(
    tmp_path: Path,
) -> None:
    found = rows(env_with(tmp_path))
    assert [row["status"] for row in found.values()] == [NO_FILE, NO_FILE]
    assert all("does not hold the package" in row["detail"] for row in found.values())


def test_no_venv_at_all_is_its_own_answer(tmp_path: Path) -> None:
    found = rows(tmp_path / "nothing-here")
    assert [row["status"] for row in found.values()] == [NO_ENV, NO_ENV]


def test_the_patches_belong_to_the_higgs_env(tmp_path: Path) -> None:
    """Orpheus's env needs neither: both are edits to the vllm-omni stack Higgs
    v3 serves through."""
    assert narratorpatches.PATCHED_ENGINE == "higgs-v3"


@pytest.mark.parametrize("patch", NARRATOR_PATCHES, ids=lambda p: p.id)
def test_every_patch_says_what_breaks_without_it(patch) -> None:
    """A row a reader cannot act on is a row nobody acts on."""
    assert patch.why.strip()
    assert patch.rel_path.endswith(".py")
    assert patch.marker.strip()
