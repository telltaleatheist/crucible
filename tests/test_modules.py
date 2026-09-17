"""The generated module files, and the rules that make them generatable.

PHASE13-OPERATOR.md section 5.4. `tests/test_lineup.py`'s shape one file along:
the checked-in JSON must equal a fresh build, so a manifest edited without
regenerating is red here as well as in CI.

The interesting half is the RESOLUTION. A class with a floor resolves to it, a
class with one candidate resolves to it, and a class with several and no floor
is REFUSED until the declaration names the model — which is the one place this
generator is allowed to have no answer, because picking would be inventing a
policy nobody wrote down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import VERSION, lineup, modules
from crucible.modules import ModuleError

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = REPO_ROOT / modules.DIR_NAME


def declarations() -> list[Path]:
    found = sorted(MODULES_DIR.glob("*.toml"))
    assert found, f"no declarations in {MODULES_DIR}"
    return found


# ------------------------------------------------------ the checked-in files


@pytest.mark.parametrize("path", declarations(), ids=lambda p: p.stem)
def test_the_checked_in_module_equals_a_fresh_build(path: Path) -> None:
    fresh = modules.build(modules.read_declaration(path), path.name)
    target = MODULES_DIR / modules.file_name(fresh["name"])
    assert target.is_file(), f"run scripts/gen-modules.py to write {target.name}"
    problems = modules.check(target.read_text(encoding="utf-8"), fresh)
    assert problems == [], (
        f"{target.name} has drifted from the manifests; regenerate it with "
        f"python scripts/gen-modules.py"
    )


@pytest.mark.parametrize("path", declarations(), ids=lambda p: p.stem)
def test_the_file_on_disk_is_byte_for_byte_what_render_produces(path: Path) -> None:
    """Apps vendor these bytes; a stray reformat would be a spurious diff."""
    fresh = modules.build(modules.read_declaration(path), path.name)
    target = MODULES_DIR / modules.file_name(fresh["name"])
    assert target.read_bytes() == modules.render(fresh).encode("utf-8")


def test_bookforge_asks_for_what_its_install_door_used_to_print() -> None:
    """The pull list in `electron/crucible/install.ts`, now derived.

    Read against the ids that file names today, because this file REPLACES it
    and a module missing one of them is a machine BookForge cannot use.
    """
    document = json.loads(
        (MODULES_DIR / "bookforge.module.json").read_text(encoding="utf-8")
    )
    assert [entry["type"] for entry in document["job_types"]] == [
        "llm",
        "asr",
        "tts",
        "align",
        "rvc",
    ]
    # EXPLICIT IDS ONLY, and every one of them is a genuine app choice
    # (PHASE15-HOST.md 5.3a): a whisper size, the aligner, a voice, the rvc
    # base, a separator. `qwen3.5-9b` LEFT this set on 2026-09-14 — it is the
    # `clean` class's answer on a PC, and which model serves `clean` is the
    # SERVER's to say, per machine.
    #
    # TWO TRANSCRIBERS, ONE CHOICE. The id of "large-v3" depends on the backend
    # — CTranslate2 has no Metal backend, so `faster-whisper-*` is cuda-linux
    # and `mlx-whisper-*` is mlx-darwin, permanently. Naming only the first is
    # what made the Mac refuse the WHOLE module (`invalid_module`, 2026-09-15);
    # an app sends each server the one it can hold, off the `backends` each
    # entry carries.
    assert {(s["kind"], s["id"]) for s in document["subjects"]} == {
        ("model", "faster-whisper-large-v3"),
        ("model", "mlx-whisper-large-v3"),
        ("model", "qwen3-aligner"),
        ("voice", "higgs-default"),
        ("rvc-base", "base"),
        ("denoise", "denoise-roformer"),
    }
    # AND EACH IS SCOPED, because an app cannot filter on a field that is not
    # there — it would post both whispers and be refused by whichever server it
    # reached.
    scope = {s["id"]: s.get("backends") for s in document["subjects"]}
    assert scope["faster-whisper-large-v3"] == ["cuda-linux"]
    assert scope["mlx-whisper-large-v3"] == ["mlx-darwin"]
    assert scope["qwen3-aligner"] == ["cuda-linux", "mlx-darwin"]
    assert [need["class"] for need in document["needs"]] == ["clean"]
    tts = next(e for e in document["job_types"] if e["type"] == "tts")
    assert tts["narrator_engine"] == "higgs-v3"


def test_generated_job_types_follow_the_core_backend_support_policy() -> None:
    from crucible.backend import LLAMA_WINDOWS
    from crucible.capability import WSL_ONLY_JOB_TYPES
    from crucible.manifests import BACKEND_ENGINES

    document = modules.build(modules.read_declaration(MODULES_DIR / "bookforge.toml"), "bookforge")
    for job in document["job_types"]:
        assert job["backends"] == sorted(
            backend for backend in BACKEND_ENGINES
            if backend != LLAMA_WINDOWS or job["type"] not in WSL_ONLY_JOB_TYPES
        )
    native = [job["type"] for job in document["job_types"] if LLAMA_WINDOWS in job["backends"]]
    assert "llm" in native
    assert "tts" not in native
    assert "align" not in native


def test_foundry_asks_for_the_classes_owens_ruling_names() -> None:
    """FIVE CLASSES AND NO IDS AT ALL, since PHASE15-HOST.md 5.3a.

    Measured by Foundry against the Mac: this file used to carry
    `qwen3.8-27b-4bit` and `dots-ocr` as RESOLVED ids, because the generator
    resolved a class to one model at generation time — the cuda-linux answer,
    because the generator runs on a PC. Posted to the Mac, `validate_module`
    refused the whole module `unknown_subject` (dots-ocr has no mlx-darwin
    block), and `qwen3.8-27b-4bit` is not what that machine's capability
    selected anyway (`qwen3.8-27b`).

    Deduplication moved with the resolution: three classes resolving to one
    model on one card is the SERVER's `skipped`, and here they are three
    different classes and stay three entries.
    """
    document = json.loads(
        (MODULES_DIR / "foundry.module.json").read_text(encoding="utf-8")
    )
    assert [need["class"] for need in document["needs"]] == [
        "clean",
        "translate",
        "simplify",
        "analysis",
        "pages",
    ]
    assert document["subjects"] == []
    # Nothing in this document is a model id, on any machine.
    assert "qwen3" not in json.dumps(document)
    assert "dots-ocr" not in json.dumps(document)


# ------------------------------------------------------------------ versions


@pytest.mark.parametrize("path", declarations(), ids=lambda p: p.stem)
def test_the_version_is_the_crucible_version_and_a_content_hash(path: Path) -> None:
    fresh = modules.build(modules.read_declaration(path), path.name)
    prefix, _, digest = fresh["version"].partition("+")
    assert prefix == VERSION
    assert len(digest) == 12 and all(c in "0123456789abcdef" for c in digest)


def test_two_apps_asking_for_different_things_have_different_versions() -> None:
    built = [
        modules.build(modules.read_declaration(path), path.name)
        for path in declarations()
    ]
    versions = {document["version"] for document in built}
    assert len(versions) == len(built)


def test_the_same_content_hashes_the_same_and_a_change_moves_it() -> None:
    body = {"name": "x", "job_types": [{"type": "llm"}], "subjects": []}
    assert modules.version_of(body) == modules.version_of(dict(body))
    moved = {**body, "subjects": [{"kind": "model", "id": "qwen3.5-9b"}]}
    assert modules.version_of(moved) != modules.version_of(body)


# ---------------------------------------------------------------- resolution


def test_a_class_with_a_floor_resolves_to_the_floor() -> None:
    floors = lineup.floors(lineup.build()[0])
    assert floors, "this build declares no floors, so this test proves nothing"
    for capability_class, model_id in floors.items():
        assert modules.resolve_class(capability_class, None) == model_id


def test_a_class_with_one_candidate_resolves_to_it() -> None:
    assert modules.resolve_class("clean", None) == "qwen3.5-9b"
    assert modules.resolve_class("pages", None) == "dots-ocr"


def test_a_class_with_several_candidates_and_no_floor_is_refused() -> None:
    """`analysis`. The refusal names the models and shows the fix."""
    with pytest.raises(ModuleError) as caught:
        modules.resolve_class("analysis", None)
    assert "qwen3.8-27b-4bit" in str(caught.value)
    assert "model =" in str(caught.value)


def test_a_named_model_is_checked_against_the_class_it_was_named_for() -> None:
    assert modules.resolve_class("analysis", "qwen3.8-27b-4bit") == "qwen3.8-27b-4bit"
    # THE 9B SERVES ANALYSIS NOW, so the model that must be refused here had to
    # change with Owen's 2026-09-16 reversal. `dots-ocr` is the page reader and
    # no text class reaches it, which is what this test needs: a model that
    # genuinely does not serve the class it was named for.
    with pytest.raises(ModuleError, match="does not serve"):
        modules.resolve_class("analysis", "dots-ocr")


def test_the_9b_now_serves_the_three_acts_it_used_to_be_refused_for() -> None:
    """The reversal, at the door a module declaration comes through.

    An app naming the 9B for translation used to be refused by name. Owen:
    *"they cant pick smaller than 9b… i think 9b could do an ok job at
    translation."*
    """
    for name in ("translate", "simplify", "analysis"):
        assert modules.resolve_class(name, "qwen3.5-9b") == "qwen3.5-9b"


def test_a_class_that_does_not_select_a_model_says_to_name_the_subject() -> None:
    """`tts` picks a voice, and a generator choosing one is choosing a narrator."""
    with pytest.raises(ModuleError, match=r"\[\[subjects\]\]"):
        modules.resolve_class("tts", None)


def test_a_class_that_is_not_a_class_is_refused() -> None:
    with pytest.raises(ModuleError, match="not a capability class"):
        modules.resolve_class("transcribe", None)


# ----------------------------------------------------------- bad declarations


def declare(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "app.toml"
    path.write_text(text, encoding="utf-8")
    return path


def refused(tmp_path: Path, text: str, because: str) -> None:
    path = declare(tmp_path, text)
    with pytest.raises(ModuleError, match=because):
        modules.build(modules.read_declaration(path), path.name)


def test_a_subject_that_does_not_exist_is_refused_at_generation(
    tmp_path: Path,
) -> None:
    refused(
        tmp_path,
        '[module]\nname = "x"\n[[subjects]]\nkind = "voice"\nid = "no-such-voice"\n',
        "has no voice",
    )


def test_a_kind_that_is_not_a_kind_is_refused(tmp_path: Path) -> None:
    refused(
        tmp_path,
        '[module]\nname = "x"\n[[subjects]]\nkind = "sorcery"\nid = "a"\n',
        "not a subject kind",
    )


def test_a_job_type_with_no_installer_names_the_one_that_builds_it(
    tmp_path: Path,
) -> None:
    """`denoise` shares the rvc env, so the refusal says to name `rvc`."""
    refused(
        tmp_path,
        '[module]\nname = "x"\n[[job_types]]\ntype = "denoise"\n',
        "shares 'rvc'",
    )


def test_tts_without_a_narrator_engine_is_refused_at_generation(
    tmp_path: Path,
) -> None:
    refused(
        tmp_path,
        '[module]\nname = "x"\n[[job_types]]\ntype = "tts"\n',
        "narrator_engine",
    )


def test_a_declaration_that_asks_for_nothing_is_refused(tmp_path: Path) -> None:
    refused(tmp_path, '[module]\nname = "x"\n', "asks for nothing")


def test_a_version_in_the_declaration_is_refused_because_it_is_derived(
    tmp_path: Path,
) -> None:
    path = declare(tmp_path, '[module]\nname = "x"\nversion = "1.0.0"\n')
    with pytest.raises(ModuleError, match="DERIVED"):
        modules.read_declaration(path)


def test_an_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    path = declare(tmp_path, '[module]\nname = "x"\n[[wants]]\nthing = "a"\n')
    with pytest.raises(ModuleError, match="unknown top-level"):
        modules.read_declaration(path)


# ------------------------------------------------------------------ the check


def test_check_is_empty_only_when_the_documents_agree() -> None:
    path = MODULES_DIR / "foundry.toml"
    fresh = modules.build(modules.read_declaration(path), path.name)
    assert modules.check(modules.render(fresh), fresh) == []
    # Foundry's `subjects` is EMPTY since 5.3a — every id it used to carry
    # is a class the server resolves — so the drift is made in `needs`,
    # which is where its content now lives.
    drifted = {**fresh, "needs": []}
    problems = modules.check(modules.render(drifted), fresh)
    assert any(problem.startswith("needs:") for problem in problems)


def test_a_drifted_version_is_a_drifted_module() -> None:
    """Nothing is ignored, unlike the lineup's `generated_from`."""
    path = MODULES_DIR / "foundry.toml"
    fresh = modules.build(modules.read_declaration(path), path.name)
    stale = {**fresh, "version": "0.0.0+000000000000"}
    assert modules.check(modules.render(stale), fresh) != []


def test_the_bytes_are_lf_on_every_platform() -> None:
    """They are vendored into another repo and compared by content."""
    for path in MODULES_DIR.glob("*.module.json"):
        assert b"\r\n" not in path.read_bytes(), path.name
