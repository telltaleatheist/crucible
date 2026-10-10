from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import lineup
from crucible.capabilityclasses import classes_for_model
from crucible.lineup import LineupError
from crucible.manifests import load_manifest, manifests_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "gen-foundry-lineup.py"
CHECKED_IN = REPO_ROOT / lineup.FILE_NAME

WITH_LOCAL = [
    "dots-ocr", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
    "qwen3.8-27b-4bit",
]
WITHOUT_LOCAL = [
    "qwen3.5-4b-8bit", "qwen3.5-4b-bside", "qwen3.5-4b-bside-4bit", "qwen3.5-9b-vl", "qwen3.8-27b-4bit-vl", "qwen3.8-27b-8bit",
]

CLASSES = {
    "dots-ocr": ["pages"],
    # every text verb runs down to the small tiers (docs/VERB-SIZING.md rule 1)
    "qwen3.5-0.8b": ["clean", "translate", "simplify", "analysis", "generate", "decide"],
    "qwen3.5-2b": ["clean", "translate", "simplify", "analysis", "generate", "decide"],
    "qwen3.5-4b": ["clean", "translate", "simplify", "analysis", "generate", "decide"],
    "qwen3.5-9b": [
        "clean", "translate", "simplify", "analysis", "generate", "decide"
    ],
    "qwen3.8-27b-4bit": ["translate", "simplify", "analysis", "generate", "decide"],
}

ROW_KEYS = ["id", "classes", "label", "description", "local"]
OLLAMA_KEYS = ["kind", "tag", "downloadGB", "needsGB"]
GGUF_KEYS = ["kind", "hf_repo", "revision", "file", "mmproj", "downloadGB", "needsGB"]

_SHA = re.compile(r"^[0-9a-f]{40}$")


def _git_reads_this_checkout() -> bool:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=False
        ).returncode == 0
    except OSError:
        return False


needs_git_here = pytest.mark.skipif(
    not _git_reads_this_checkout(),
    reason="the generator stamps git HEAD, and git on this platform cannot read "
    "this checkout (a worktree made by the other side of WSL)",
)


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


def test_the_checked_in_lineup_equals_the_generators_output() -> None:
    rows, omitted = lineup.build()
    fresh = lineup.document(rows, "0" * 40)
    assert lineup.content(_checked_in()) == lineup.content(fresh)
    assert omitted == WITHOUT_LOCAL


def test_the_checked_in_file_names_the_commit_it_was_generated_from() -> None:
    doc = _checked_in()
    assert list(doc) == [lineup.PROVENANCE_KEY, "schema", "models"]
    assert _SHA.match(doc[lineup.PROVENANCE_KEY])
    assert doc["schema"] == lineup.SCHEMA == 3


def test_the_checked_in_file_is_exactly_what_render_writes() -> None:
    doc = _checked_in()
    assert CHECKED_IN.read_bytes() == lineup.render(doc).encode("utf-8")
    assert b"\r\n" not in CHECKED_IN.read_bytes()


@needs_git_here
def test_check_passes_on_the_checked_in_file() -> None:
    ran = _run("--check")
    assert ran.returncode == 0, ran.stderr
    assert "matches the manifests" in ran.stdout


@needs_git_here
def test_check_fails_by_name_when_a_row_has_moved(tmp_path: Path) -> None:
    doc = _checked_in()
    doc["models"][0]["local"]["downloadGB"] += 1
    drifted = tmp_path / lineup.FILE_NAME
    drifted.write_text(lineup.render(doc), encoding="utf-8")
    ran = _run("--check", "--output", str(drifted))
    assert ran.returncode == 1
    assert "HAS DRIFTED" in ran.stderr
    assert f"{doc['models'][0]['id']}: differs in ['local']" in ran.stderr


@needs_git_here
def test_check_fails_when_the_file_is_missing(tmp_path: Path) -> None:
    ran = _run("--check", "--output", str(tmp_path / "absent.json"))
    assert ran.returncode == 1
    assert "absent.json" in ran.stderr


@needs_git_here
def test_the_generator_writes_a_file_its_own_check_accepts(tmp_path: Path) -> None:
    target = tmp_path / "out" / lineup.FILE_NAME
    wrote = _run("--verbose", "--output", str(target))
    assert wrote.returncode == 0, wrote.stderr
    assert "omitted qwen3.8-27b-8bit: no [local] table" in wrote.stdout
    assert "6 model(s) with a local form, 6 omitted" in wrote.stdout
    assert lineup.content(json.loads(target.read_text(encoding="utf-8"))) == (
        lineup.content(_checked_in())
    )
    checked = _run("--check", "--output", str(target))
    assert checked.returncode == 0, checked.stderr


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
    }


def test_the_cleanup_model_is_the_bf16_ollama_tag() -> None:
    row = next(r for r in _checked_in()["models"] if r["id"] == "qwen3.5-9b")
    assert row["local"] == {
        "kind": "ollama",
        "tag": "qwen3.5:9b-bf16",
        "downloadGB": 19.32,
        "needsGB": {"value": 20.82, "basis": "declared"},
    }


def test_the_27b_row_is_its_ollama_tag() -> None:
    row = next(r for r in _checked_in()["models"] if r["id"] == "qwen3.8-27b-4bit")
    assert row["local"]["kind"] == "ollama"
    assert row["local"]["tag"] == "qwen3.8:27b"
    assert row["local"]["downloadGB"] == 17.74


def test_no_class_has_a_floor_any_more() -> None:
    # Owen 2026-10-09, of the 9B's `minimum_for = ["translate", "simplify"]`: "we can
    # remove it, yes" (docs/VERB-SIZING.md section 7).
    doc = _checked_in()
    assert "floors" not in doc
    for row in doc["models"]:
        assert "minimum" not in row and "minimumFor" not in row


def test_gigabytes_are_decimal_at_two_places() -> None:
    assert lineup.gigabytes(19_321_189_044) == 19.32
    assert lineup.gigabytes(1_000_000_000) == 1.0
    assert lineup.gigabytes(4_419_026_144) == 4.42


@pytest.mark.parametrize("model_id", sorted(CLASSES))
def test_classes_come_from_the_capability_table(model_id: str) -> None:
    assert list(classes_for_model(model_id)) == CLASSES[model_id]


def test_a_model_with_no_local_form_still_has_classes() -> None:
    assert classes_for_model("qwen3.8-27b-8bit") == (
        "translate", "simplify", "analysis", "generate", "decide",
    )


def test_an_unknown_model_id_is_refused_not_answered_with_no_classes() -> None:
    with pytest.raises(ValueError) as caught:
        classes_for_model("qwen9-1t")
    assert "'qwen9-1t' is not a model in this build's catalog" in str(caught.value)


FIXTURE = """
[model]
id = "{id}"
family = "{family}"
params_b = 9
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
    _catalog(monkeypatch, tmp_path, id="demo-1b", family="demo")
    with pytest.raises(LineupError) as caught:
        lineup.build()
    assert "no capability class in crucible/capabilityclasses.py names its family 'demo'" in (
        str(caught.value)
    )


def test_a_fixture_catalog_builds_the_same_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _catalog(monkeypatch, tmp_path, id="demo-1b", family="qwen3.5")
    rows, omitted = lineup.build()
    assert omitted == []
    assert rows == [
        {
            "id": "demo-1b",
            "classes": [
                "clean", "translate", "simplify", "analysis", "generate", "decide"
            ],
            "label": "Demo",
            "description": "A fixture.",
            "local": {
                "kind": "ollama",
                "tag": "demo:1b",
                "downloadGB": 1.0,
                "needsGB": {"value": 2.5, "basis": "declared"},
            },
        }
    ]


def test_check_reads_a_broken_file_as_a_named_problem() -> None:
    rows, _ = lineup.build()
    fresh = lineup.document(rows, "0" * 40)
    assert lineup.check("not json", fresh) == [
        "the checked-in file is not valid JSON: Expecting value: line 1 column 1 (char 0)"
    ]
    assert lineup.check("[]", fresh) == ["the checked-in file is not a JSON object"]
    assert lineup.check('{"schema": 3}', fresh) == [
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
    schema = dict(fresh, schema=2)
    assert lineup.check(lineup.render(schema), fresh) == [
        "schema: checked in 2, generator says 3"
    ]

def test_the_document_is_provenance_schema_and_rows() -> None:
    rows, _ = lineup.build()
    doc = lineup.document(rows, "0" * 40)
    assert doc["schema"] == 3
    assert list(doc) == ["generated_from", "schema", "models"]
