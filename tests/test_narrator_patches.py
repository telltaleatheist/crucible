"""Two edits pip cannot express, and the check that says whether they are in.

PHASE3-TTS.md section 4. An env whose pins all match is reported ready, and a
reader has no way to tell that from an env that will render every chunk with
240 ms of garbage on the end — so the patches are checked, by name, against the
markers BookForge measured on the certifying box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from crucible import narratorpatches
from crucible.narratorpatches import (
    APPLIED,
    MISSING,
    NARRATOR_PATCHES,
    NO_ENV,
    NO_FILE,
    NOT_APPLICABLE,
    SOUND_STATUSES,
    STALE,
)

#: The pins of a recipe that installs everything both patches edit — the shape of
#: `envs/tts/higgs-v3-cuda-linux.txt`. Every check below runs against this unless
#: it is about a backend that does not have the vLLM stack at all.
CUDA_PINS = {patch.distribution: "0.28.0" for patch in NARRATOR_PATCHES}

PRISTINE_INPUT_PROCESSOR = "if min_input_id < 0:\n    raise ValueError('oov')\n"
PATCHED_INPUT_PROCESSOR = "if min_input_id < 0 and min_input_id != -100:\n"

#: A stage processor with the filter in and upstream's one-frame trim gone, plus
#: the env var only v3 of the patch reads. v2 logged `final=%s, window=%d frames`
#: and wrote no records; an env carrying that is STALE, because narrator now reads
#: the RECORDS rather than running regexes over the server's log file
#: (ARCHITECTURE.md R4 — a log line is never load-bearing).
PATCHED_STAGE = (
    "def _filter_sentinel_frames(frames):\n"
    "    ...\n"
    "    path = os.environ.get('HIGGS_SENTINEL_REPORT')\n"
)

#: What v2 left in site-packages: the filter in, the trim gone, and no report.
#: It passes every marker grep an env could be given, and narrator cannot read it.
V2_STAGE = (
    "def _filter_sentinel_frames(frames):\n"
    "    ...\n"
    "logger.warning('final=%s, window=%d frames', final, window)\n"
)


@dataclass(frozen=True)
class Ran:
    """What `apply`'s `runner` returns — the two fields it reads, and no more."""

    returncode: int
    stdout: str


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


def rows(env: Path, pins: dict[str, str] | None = None) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in narratorpatches.check(env, CUDA_PINS if pins is None else pins)
    }


# ------------------------------------------------- the backend that has no vLLM


def test_a_backend_without_the_vllm_stack_is_not_applicable_not_broken(
    tmp_path: Path,
) -> None:
    """`crucible doctor` called a sound Mac unhealthy until 2026-09-13.

    Both patches edit `vllm`/`vllm-omni`, which `envs/tts/mlx-darwin.txt` does
    not install — Higgs v3 on the Mac is narrator's in-process MLX backend. The
    old check reported `no_such_file` for both, the doctor turned each into a
    problem, and `healthy` went false on a machine with nothing wrong with it.
    """
    found = rows(tmp_path / "env", pins={"mlx-lm": "0.31.3", "mlx-audio": "0.4.8"})
    assert [row["status"] for row in found.values()] == [
        NOT_APPLICABLE,
        NOT_APPLICABLE,
    ]
    # Not a problem — and the doctor must not read `applied` to learn that,
    # because "is the marker in the file" has no honest answer when there is no
    # file and never will be one.
    assert all(row["status"] in SOUND_STATUSES for row in found.values())
    assert not any(row["applied"] for row in found.values())
    assert "does not install vllm" in found["vllm-negative-token-id"]["detail"]


def test_a_recipe_that_adds_the_stack_starts_being_checked_on_its_own(
    tmp_path: Path,
) -> None:
    """The applicability fact has ONE owner, the recipe. Nothing in
    `narratorpatches` names a backend, so a Mac recipe that one day pins
    `vllm-omni` is checked without an edit here."""
    env = env_with(tmp_path, **{"higgs-sentinel-filter": PATCHED_STAGE})
    assert rows(env, pins={"mlx-lm": "0.31.3"})["higgs-sentinel-filter"]["status"] == (
        NOT_APPLICABLE
    )
    assert rows(env, pins={"vllm-omni": "0.28.0"})["higgs-sentinel-filter"][
        "status"
    ] == APPLIED


def test_the_shipped_cuda_recipe_makes_NEITHER_patch_applicable() -> None:
    """THE STACK FLIP OF 2026-09-15 TURNED THIS TEST INSIDE OUT, and that is
    the right answer rather than a loosened one.

    Both patches edit the vLLM stack — the negative-token-id fix in `vllm` and
    the sentinel filter in `vllm_omni`. Owen ruled that Higgs does not render on
    vllm-omni at all ("we dont use vllm-omni. we use sglang. vllm-omni doesnt
    work for higgs"), so `higgs-v3-cuda-linux.txt` now installs sglang-omni and
    neither distribution is there to patch.

    WHAT MATTERS IS THAT THEY REPORT `not_applicable` AND NOT `missing`.
    SGLang-Omni has its own stage processor and needs no patch — `sgl_served`'s
    header and BookForge's installer both say so in as many words — so a
    `crucible doctor` that called this env unpatched would be calling a sound
    env broken, which is exactly what it did to the Mac until 2026-09-13. The
    selection is by the recipe's own pins, so this needed no code change; this
    test is what proves that."""
    from crucible import jobenv

    pins = jobenv.recipe_pins(jobenv.recipe_for(jobenv.tts_env("higgs-v3", "cuda-linux")))
    for patch in NARRATOR_PATCHES:
        assert patch.distribution not in pins, patch.id


def test_the_shipped_mac_recipe_makes_neither_applicable() -> None:
    from crucible import jobenv

    pins = jobenv.recipe_pins(jobenv.recipe_for(jobenv.tts_env("higgs-v3", "mlx-darwin")))
    for patch in NARRATOR_PATCHES:
        assert patch.distribution not in pins, patch.id


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
    """Both are edits to the vllm-omni stack Higgs v3 serves through, so an
    engine on another stack needs neither."""
    assert narratorpatches.PATCHED_ENGINE == "higgs-v3"


@pytest.mark.parametrize("patch", NARRATOR_PATCHES, ids=lambda p: p.id)
def test_every_patch_says_what_breaks_without_it(patch) -> None:
    """A row a reader cannot act on is a row nobody acts on."""
    assert patch.why.strip()
    assert patch.rel_path.endswith(".py")
    assert patch.marker.strip()


def test_a_v2_env_is_reported_stale_rather_than_applied(tmp_path: Path) -> None:
    """The generation that logged instead of reporting.

    v2 carries `_filter_sentinel_frames` and has upstream's one-frame trim gone,
    so it passes every marker grep an env could be given — and narrator can no
    longer read it, because the proof now reads RECORDS rather than running three
    regexes over the server's log file. An env like this renders a book that
    cannot be proven clean, so `applied` would be a lie and `stale` is the answer.

    Every v1 and v2 env on every machine reports this until the installer
    re-applies the patch. That is intended and loud.
    """
    found = rows(env_with(tmp_path, **{"higgs-sentinel-filter": V2_STAGE}))
    assert found["higgs-sentinel-filter"]["status"] == STALE
    assert not found["higgs-sentinel-filter"]["applied"]


# ------------------------------------------------------------ applying them


def test_the_vendored_appliers_write_the_markers_the_table_greps_for() -> None:
    """The two copies of one fact, compared.

    `envs/tts/patches/*.py` are byte-identical copies of BookForge's
    `electron/scripts/higgs/`, because narrator's wheel does not carry them and a
    Crucible server must not need a BookForge checkout. A copy can drift, and the
    way it would drift invisibly is the pair below disagreeing: a script that
    writes one marker and a table that greps for another leaves `apply` running
    the patch and then refusing the install it just fixed.

    Read out of the script's source rather than by importing it: importing runs
    a module that expects an env prefix in argv.
    """
    for patch in NARRATOR_PATCHES:
        source = narratorpatches.script_path(patch).read_text(encoding="utf-8")
        assert f'REL = "{patch.rel_path}"' in source, patch.id
        assert f'MARKER = "{patch.marker}"' in source, patch.id
        if patch.absent_marker is not None:
            assert f'ABSENT_MARKER = "{patch.absent_marker}"' in source, patch.id


def test_apply_runs_one_script_per_applicable_patch_and_proves_the_result(
    tmp_path: Path,
) -> None:
    """The happy path: pip reverted both files, `apply` puts them back.

    The fake runner writes what the real scripts write, so `check` — the same
    function `crucible doctor` calls — is what decides the install succeeded.
    """
    env = env_with(
        tmp_path,
        **{
            "vllm-negative-token-id": PRISTINE_INPUT_PROCESSOR,
            "higgs-sentinel-filter": "codes = codes[:, :-1]\n",
        },
    )
    packages = env / "lib" / "python3.11" / "site-packages"
    written: dict[str, str] = {
        "vllm-negative-token-id": PATCHED_INPUT_PROCESSOR,
        "higgs-sentinel-filter": PATCHED_STAGE,
    }
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> Ran:
        seen.append(argv)
        for patch in NARRATOR_PATCHES:
            if argv[1].endswith(patch.script):
                (packages / patch.rel_path).write_text(
                    written[patch.id], encoding="utf-8"
                )
        return Ran(0, "PATCHED\n")

    result = narratorpatches.apply(
        env, Path("python"), CUDA_PINS, runner=runner
    )
    assert [row["status"] for row in result] == [APPLIED, APPLIED]
    # One invocation per patch, each handed the ENV DIRECTORY — the scripts glob
    # `<prefix>/lib/python*/site-packages` off it rather than being told a
    # python version.
    assert len(seen) == len(NARRATOR_PATCHES)
    assert all(argv[2] == str(env) for argv in seen)


def test_apply_refuses_when_the_env_still_does_not_carry_the_patch(
    tmp_path: Path,
) -> None:
    """A script that exits 0 and changes nothing must not stamp the env.

    This is the regression of 2026-09-15 one layer in: the exit code is the
    script's own idea of success, and the thing that has to be true is the marker
    `crucible doctor` greps for tomorrow. `apply` re-checks, so the two are one
    fact and `jobenv.install_env` raises before it writes the stamp.
    """
    env = env_with(
        tmp_path,
        **{
            "vllm-negative-token-id": PRISTINE_INPUT_PROCESSOR,
            "higgs-sentinel-filter": "codes = codes[:, :-1]\n",
        },
    )
    with pytest.raises(narratorpatches.PatchError) as raised:
        narratorpatches.apply(
            env, Path("python"), CUDA_PINS, runner=lambda argv: Ran(0, "")
        )
    assert "still does not carry them" in str(raised.value)
    assert MISSING in str(raised.value)


def test_apply_quotes_a_failing_script_rather_than_swallowing_it(
    tmp_path: Path,
) -> None:
    """`ANCHOR_NOT_FOUND` means upstream moved the code this patch edits.

    The scripts say so on stderr and exit 2 rather than skipping quietly, and the
    refusal has to carry that word out to the operator: the patch must be
    re-derived against the new version, which is not something a retry fixes.
    """
    env = env_with(tmp_path, **{"vllm-negative-token-id": "moved on\n"})
    with pytest.raises(narratorpatches.PatchError) as raised:
        narratorpatches.apply(
            env,
            Path("python"),
            CUDA_PINS,
            runner=lambda argv: Ran(2, "ANCHOR_NOT_FOUND\n"),
        )
    assert "ANCHOR_NOT_FOUND" in str(raised.value)
    assert "vllm-negative-token-id" in str(raised.value)


def test_apply_runs_nothing_on_a_backend_whose_recipe_has_no_vllm_stack(
    tmp_path: Path,
) -> None:
    """The Mac again: nothing to patch is not a failure, and not a no-op lie.

    `mlx-darwin`'s tts recipe pins neither distribution, so `apply` runs neither
    script — and the rows it returns still say `not_applicable` by name, so an
    install there records the same answer `crucible doctor` gives.
    """
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> Ran:
        seen.append(argv)
        return Ran(0, "")

    result = narratorpatches.apply(
        tmp_path / "env",
        Path("python"),
        {"mlx-lm": "0.31.3", "mlx-audio": "0.4.8"},
        runner=runner,
    )
    assert seen == []
    assert [row["status"] for row in result] == [NOT_APPLICABLE, NOT_APPLICABLE]
