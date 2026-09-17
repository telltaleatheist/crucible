"""The lineup Foundry vendors: `foundry-lineup.json`, and the generator behind it.

Owen, 2026-09-13, via Foundry: the model manifests are the catalog of record for
Foundry's LOCAL lineup too, so "what can this machine run" has one owner. The
first test here is the whole ruling as a guard — the checked-in file equals what
the manifests say, or the suite is red — and everything after it is the shape
Foundry's reader is built against, asserted field by field rather than trusted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import lineup
from crucible.capability import classes_for_model
from crucible.lineup import LineupError
from crucible.manifests import load_manifest, manifests_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "gen-foundry-lineup.py"
CHECKED_IN = REPO_ROOT / lineup.FILE_NAME

#: The three models Foundry runs locally today, and the one it cannot.
WITH_LOCAL = ["dots-ocr", "qwen3.5-9b", "qwen3.8-27b-4bit"]
WITHOUT_LOCAL = ["qwen3.8-27b"]

#: Which classes each local model lights, read off `capability.CLASSES` through
#: the same door the generator uses. Written out here so a change to the class
#: table that moves a model is a change this file notices.
CLASSES = {
    "dots-ocr": ["pages"],
    "qwen3.5-9b": ["clean"],
    "qwen3.8-27b-4bit": ["translate", "simplify", "analysis"],
}

#: The exact key order of one row. Foundry's reader compares by content and a
#: reader may be strict about order; ours is the order the ruling was shown.
ROW_KEYS = ["id", "classes", "label", "description", "local", "minimum", "minimumFor"]
OLLAMA_KEYS = ["kind", "tag", "downloadGB", "needsGB"]
GGUF_KEYS = ["kind", "hf_repo", "revision", "file", "mmproj", "downloadGB", "needsGB"]

_SHA = re.compile(r"^[0-9a-f]{40}$")


def _checked_in() -> dict:
    return json.loads(CHECKED_IN.read_text(encoding="utf-8"))


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


# ------------------------------------------------------ the file cannot drift


def test_the_checked_in_lineup_equals_the_generators_output() -> None:
    """The ruling as a guard. A manifest edited without regenerating the file is
    red here and in CI's `--check`, never shipped with two answers."""
    rows, omitted = lineup.build()
    fresh = lineup.document(rows, "0" * 40)
    assert lineup.content(_checked_in()) == lineup.content(fresh)
    assert omitted == WITHOUT_LOCAL


def test_the_checked_in_file_names_the_commit_it_was_generated_from() -> None:
    doc = _checked_in()
    assert list(doc) == [lineup.PROVENANCE_KEY, "schema", "floors", "models"]
    assert _SHA.match(doc[lineup.PROVENANCE_KEY])
    assert doc["schema"] == lineup.SCHEMA == 2


def test_the_checked_in_file_is_exactly_what_render_writes() -> None:
    """Byte-for-byte, so a hand edit that survives the content check (whitespace,
    key order, a trailing newline) is still caught."""
    doc = _checked_in()
    assert CHECKED_IN.read_bytes() == lineup.render(doc).encode("utf-8")
    assert b"\r\n" not in CHECKED_IN.read_bytes()


def test_check_passes_on_the_checked_in_file() -> None:
    ran = _run("--check")
    assert ran.returncode == 0, ran.stderr
    assert "matches the manifests" in ran.stdout


def test_check_fails_by_name_when_a_row_has_moved(tmp_path: Path) -> None:
    doc = _checked_in()
    doc["models"][0]["local"]["downloadGB"] += 1
    drifted = tmp_path / lineup.FILE_NAME
    drifted.write_text(lineup.render(doc), encoding="utf-8")
    ran = _run("--check", "--output", str(drifted))
    assert ran.returncode == 1
    assert "HAS DRIFTED" in ran.stderr
    assert f"{doc['models'][0]['id']}: differs in ['local']" in ran.stderr


def test_check_fails_when_the_file_is_missing(tmp_path: Path) -> None:
    ran = _run("--check", "--output", str(tmp_path / "absent.json"))
    assert ran.returncode == 1
    assert "absent.json" in ran.stderr


def test_the_generator_writes_a_file_its_own_check_accepts(tmp_path: Path) -> None:
    target = tmp_path / "out" / lineup.FILE_NAME
    wrote = _run("--verbose", "--output", str(target))
    assert wrote.returncode == 0, wrote.stderr
    assert "omitted qwen3.8-27b: no [local] table" in wrote.stdout
    assert "3 model(s) with a local form, 1 omitted" in wrote.stdout
    assert lineup.content(json.loads(target.read_text(encoding="utf-8"))) == (
        lineup.content(_checked_in())
    )
    checked = _run("--check", "--output", str(target))
    assert checked.returncode == 0, checked.stderr


# ------------------------------------------------------------------ the shape


def test_rows_are_in_id_order_and_only_models_with_a_local_form() -> None:
    ids = [row["id"] for row in _checked_in()["models"]]
    assert ids == WITH_LOCAL
    assert ids == sorted(ids)
    for model_id in WITHOUT_LOCAL:
        assert load_manifest(model_id).local is None


@pytest.mark.parametrize("model_id", WITH_LOCAL)
def test_every_row_has_exactly_the_keys_foundry_reads(model_id: str) -> None:
    row = next(r for r in _checked_in()["models"] if r["id"] == model_id)
    assert list(row) == ROW_KEYS
    assert isinstance(row["label"], str) and row["label"]
    assert isinstance(row["description"], str) and row["description"]
    assert row["classes"] == CLASSES[model_id]
    assert isinstance(row["minimum"], bool)
    assert isinstance(row["minimumFor"], list)
    local = row["local"]
    assert list(local) == (OLLAMA_KEYS if local["kind"] == "ollama" else GGUF_KEYS)
    assert isinstance(local["downloadGB"], float)
    assert list(local["needsGB"]) == ["value", "basis"]
    assert local["needsGB"]["basis"] in {"measured", "declared"}
    assert local["needsGB"]["value"] >= local["downloadGB"]


def test_the_first_row_is_the_page_reader_as_a_gguf_pair() -> None:
    row = _checked_in()["models"][0]
    assert row == {
        "id": "dots-ocr",
        "classes": ["pages"],
        "label": "dots.ocr",
        "description": load_manifest("dots-ocr").description,
        "local": {
            "kind": "gguf",
            "hf_repo": "ggml-org/dots.ocr-GGUF",
            "revision": "2c093a32ca360a396bc6d87d60408636130b9d9b",
            "file": "dots.ocr-Q8_0.gguf",
            "mmproj": "mmproj-dots.ocr-Q8_0.gguf",
            "downloadGB": 3.24,
            "needsGB": {"value": 4.74, "basis": "declared"},
        },
        "minimum": False,
        "minimumFor": [],
    }


def test_the_cleanup_model_is_the_bf16_ollama_tag() -> None:
    row = next(r for r in _checked_in()["models"] if r["id"] == "qwen3.5-9b")
    assert row["local"] == {
        "kind": "ollama",
        "tag": "qwen3.5:9b-bf16",
        # 19_321_189_044 B, which `ollama list` prints as "19 GB".
        "downloadGB": 19.32,
        "needsGB": {"value": 20.82, "basis": "declared"},
    }
    assert row["minimum"] is False and row["minimumFor"] == []


def test_the_27b_is_the_floor_for_the_three_classes_it_serves() -> None:
    """Owen's tile rule: translate and simplify do not light below it; analysis has no floor."""
    row = next(r for r in _checked_in()["models"] if r["id"] == "qwen3.8-27b-4bit")
    assert row["local"]["kind"] == "ollama"
    assert row["local"]["tag"] == "qwen3.8:27b"
    assert row["local"]["downloadGB"] == 17.74
    assert row["minimum"] is True
    assert row["minimumFor"] == ["translate", "simplify"]
    assert set(row["minimumFor"]) <= set(row["classes"])


def test_gigabytes_are_decimal_at_two_places() -> None:
    """Ollama's GB, not GiB: 19_321_189_044 B is the "19 GB" `ollama list` prints."""
    assert lineup.gigabytes(19_321_189_044) == 19.32
    assert lineup.gigabytes(1_000_000_000) == 1.0
    assert lineup.gigabytes(4_419_026_144) == 4.42


# ------------------------------------------------------- classes, one owner


@pytest.mark.parametrize("model_id", sorted(CLASSES))
def test_classes_come_from_the_capability_table(model_id: str) -> None:
    assert list(classes_for_model(model_id)) == CLASSES[model_id]


def test_a_model_with_no_local_form_still_has_classes() -> None:
    """Omitted from the lineup is not the same as serving nothing: the bf16 27B
    is a translate candidate on a 64 GB Mac; it just has no Ollama form."""
    assert classes_for_model("qwen3.8-27b") == ("translate", "simplify", "analysis")


def test_an_unknown_model_id_is_refused_not_answered_with_no_classes() -> None:
    with pytest.raises(ValueError) as caught:
        classes_for_model("qwen9-1t")
    assert "'qwen9-1t' is not a model in this build's catalog" in str(caught.value)


# ------------------------------------------------- what the generator refuses

FIXTURE = """
[model]
id = "{id}"
family = "{family}"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text"]
display = "Demo"
description = "A fixture."

[local]
kind = "ollama"
tag = "demo:1b"
download_bytes = 1000000000
needs_bytes = 2500000000
needs_basis = "declared"
{extra}
[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
"""


def _catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **fields: str) -> None:
    text = FIXTURE.format(**{"extra": "", **fields})
    (tmp_path / f"{fields['id']}.toml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(tmp_path))
    assert manifests_dir() == tmp_path


def test_a_local_model_no_class_names_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A row lights a tile. A family no class in `CLASSES` names would light none."""
    _catalog(monkeypatch, tmp_path, id="demo-1b", family="demo")
    with pytest.raises(LineupError) as caught:
        lineup.build()
    assert "no capability class in crucible/capability.py names its family 'demo'" in (
        str(caught.value)
    )


def test_a_floor_for_a_class_the_model_does_not_serve_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _catalog(
        monkeypatch,
        tmp_path,
        id="demo-1b",
        family="qwen3.5",
        extra='minimum_for = ["translate"]\n',
    )
    with pytest.raises(LineupError) as caught:
        lineup.build()
    assert "minimum_for names ['translate'], which this model does not serve" in (
        str(caught.value)
    )
    assert "its classes are ['clean']" in str(caught.value)


def test_a_fixture_catalog_builds_the_same_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _catalog(monkeypatch, tmp_path, id="demo-1b", family="qwen3.5")
    rows, omitted = lineup.build()
    assert omitted == []
    assert rows == [
        {
            "id": "demo-1b",
            "classes": ["clean"],
            "label": "Demo",
            "description": "A fixture.",
            "local": {
                "kind": "ollama",
                "tag": "demo:1b",
                "downloadGB": 1.0,
                "needsGB": {"value": 2.5, "basis": "declared"},
            },
            "minimum": False,
            "minimumFor": [],
        }
    ]


def test_check_reads_a_broken_file_as_a_named_problem() -> None:
    rows, _ = lineup.build()
    fresh = lineup.document(rows, "0" * 40)
    assert lineup.check("not json", fresh) == [
        "the checked-in file is not valid JSON: Expecting value: line 1 column 1 (char 0)"
    ]
    assert lineup.check("[]", fresh) == ["the checked-in file is not a JSON object"]
    assert lineup.check('{"schema": 2}', fresh) == [
        "models: the checked-in file has no models list"
    ]


def test_check_names_a_row_that_appeared_or_vanished() -> None:
    rows, _ = lineup.build()
    fresh = lineup.document(rows, "0" * 40)
    fewer = lineup.document(rows[1:], "0" * 40)
    assert lineup.check(lineup.render(fewer), fresh) == [
        f"{rows[0]['id']}: in the manifests, not in the checked-in file"
    ]
    assert lineup.check(lineup.render(fresh), fewer) == [
        f"{rows[0]['id']}: in the checked-in file, not in the manifests"
    ]
    reordered = lineup.document(list(reversed(rows)), "0" * 40)
    problems = lineup.check(lineup.render(reordered), fresh)
    assert len(problems) == 1 and problems[0].startswith(
        "models: the same rows in a different order"
    )


def test_check_ignores_provenance_and_nothing_else() -> None:
    rows, _ = lineup.build()
    fresh = lineup.document(rows, "0" * 40)
    other = lineup.document(rows, "f" * 40)
    assert lineup.check(lineup.render(other), fresh) == []
    schema = dict(fresh, schema=3)
    assert lineup.check(lineup.render(schema), fresh) == [
        "schema: checked in 3, generator says 2"
    ]

def test_floors_names_one_model_per_class_and_only_where_a_manifest_says_so() -> None:
    """The catalog of record owns the floor, in one place a reader can ask.

    Added 2026-09-14 after the fact grew a second owner: Foundry vendors this
    file AND keeps its own additions, and its reader took the SMALLEST declared
    floor across both — so a local row naming a 9B the translate floor silently
    overruled this catalog on every machine that fits a 9B and not a 27B,
    against Owen's ruling that translate and simplify take a 27B-class model, a
    Crucible, or a cloud provider, never a 9B locally.
    """
    rows, _ = lineup.build()
    table = lineup.floors(rows)
    assert table == {"simplify": "qwen3.8-27b-4bit", "translate": "qwen3.8-27b-4bit"}
    # Derived from the rows, never declared beside them.
    for name, model in table.items():
        row = next(r for r in rows if r["id"] == model)
        assert name in row["minimumFor"]
    # A class no manifest floors is absent, and that is its own answer: analysis
    # carries no floor because Owen named translate and simplify.
    assert "analysis" not in table
    assert "clean" not in table and "pages" not in table


def test_two_models_flooring_one_class_is_refused_rather_than_picked() -> None:
    """"The smallest wins" is the rule that let a second owner overrule the
    catalog; it has no place inside the owner either."""
    rows = [
        {"id": "small", "minimumFor": ["translate"]},
        {"id": "large", "minimumFor": ["translate"]},
    ]
    with pytest.raises(lineup.LineupError) as raised:
        lineup.floors(rows)
    assert "two models floor the 'translate' class" in str(raised.value)
    assert "'small'" in str(raised.value) and "'large'" in str(raised.value)


def test_the_document_carries_the_floors_the_rows_state() -> None:
    rows, _ = lineup.build()
    doc = lineup.document(rows, "0" * 40)
    assert doc["schema"] == 2
    assert doc["floors"] == lineup.floors(rows)
    assert list(doc) == ["generated_from", "schema", "floors", "models"]

