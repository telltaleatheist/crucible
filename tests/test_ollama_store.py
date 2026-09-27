from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import ollamastore

MODEL_DIGEST = "sha256:" + "d7" * 32
PROJECTOR_DIGEST = "sha256:" + "9c" * 32


def build_store(
    tmp_path: Path,
    *,
    tag: str = "qwen3.5:9b-bf16",
    model_bytes: bytes = b"GGUF-the-weights",
    declared: int | None = None,
    projector: bool = False,
    write_blob: bool = True,
) -> Path:
    root = tmp_path / "models"
    (root / "blobs").mkdir(parents=True)
    name, _, label = tag.partition(":")
    folder = root / "manifests" / "registry.ollama.ai" / "library" / name
    folder.mkdir(parents=True)
    layers = [
        {
            "mediaType": ollamastore.MODEL_MEDIA_TYPE,
            "digest": MODEL_DIGEST,
            "size": len(model_bytes) if declared is None else declared,
        },
        {
            "mediaType": "application/vnd.ollama.image.license",
            "digest": "sha256:" + "11" * 32,
            "size": 4,
        },
        {
            "mediaType": "application/vnd.ollama.image.params",
            "digest": "sha256:" + "22" * 32,
            "size": 2,
        },
    ]
    if projector:
        blob = root / "blobs" / PROJECTOR_DIGEST.replace(":", "-", 1)
        blob.write_bytes(b"proj")
        layers.append(
            {
                "mediaType": ollamastore.PROJECTOR_MEDIA_TYPE,
                "digest": PROJECTOR_DIGEST,
                "size": 4,
            }
        )
    (folder / label).write_text(
        json.dumps({"schemaVersion": 2, "layers": layers}), encoding="utf-8"
    )
    if write_blob:
        (root / "blobs" / MODEL_DIGEST.replace(":", "-", 1)).write_bytes(model_bytes)
    return root


def test_the_store_is_found_where_ollama_puts_it(tmp_path: Path) -> None:
    assert ollamastore.store_root(environ={}, home=tmp_path) == (
        tmp_path / ".ollama" / "models"
    )


def test_the_environment_moves_it(tmp_path: Path) -> None:
    moved = tmp_path / "elsewhere"
    assert ollamastore.store_root(
        environ={ollamastore.STORE_ENV: str(moved)}, home=tmp_path
    ) == moved


def test_no_store_and_no_tag_are_different_answers(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    with pytest.raises(ollamastore.OllamaStoreError, match="there is no Ollama store"):
        ollamastore.resident(empty, "qwen3.5:9b-bf16")

    root = build_store(tmp_path)
    with pytest.raises(ollamastore.OllamaStoreError, match="not in this Ollama store"):
        ollamastore.resident(root, "qwen3.5:70b-bf16")


def test_the_model_layer_is_found_past_the_licence(tmp_path: Path) -> None:
    root = build_store(tmp_path)
    weights = ollamastore.resident(root, "qwen3.5:9b-bf16")
    assert weights.model.digest == MODEL_DIGEST
    assert weights.model.path.read_bytes().startswith(b"GGUF")
    assert weights.projector is None


def test_a_manifest_with_no_model_layer_is_a_partial_pull(tmp_path: Path) -> None:
    root = build_store(tmp_path)
    path = ollamastore.manifest_path(root, "qwen3.5:9b-bf16")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["layers"] = [
        layer
        for layer in document["layers"]
        if layer["mediaType"] != ollamastore.MODEL_MEDIA_TYPE
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ollamastore.OllamaStoreError, match="partial pull"):
        ollamastore.resident(root, "qwen3.5:9b-bf16")


def test_a_manifest_pointing_at_a_blob_that_is_gone_is_refused(tmp_path: Path) -> None:
    root = build_store(tmp_path, write_blob=False)
    with pytest.raises(ollamastore.OllamaStoreError, match="is not there"):
        ollamastore.resident(root, "qwen3.5:9b-bf16")


def test_a_truncated_blob_is_refused_here_rather_than_by_llama_cpp(
    tmp_path: Path,
) -> None:
    root = build_store(tmp_path, model_bytes=b"GGUF-short", declared=999_999)
    with pytest.raises(ollamastore.OllamaStoreError, match="bytes and the manifest says"):
        ollamastore.resident(root, "qwen3.5:9b-bf16")


def test_a_vision_model_gets_its_projector_or_a_refusal(tmp_path: Path) -> None:
    with_proj = build_store(tmp_path / "a", projector=True)
    assert ollamastore.resident(
        with_proj, "qwen3.5:9b-bf16", wants_projector=True
    ).projector is not None

    without = build_store(tmp_path / "b")
    with pytest.raises(ollamastore.OllamaStoreError, match="cannot see"):
        ollamastore.resident(without, "qwen3.5:9b-bf16", wants_projector=True)


def test_the_provenance_never_claims_to_be_the_hub_pin(tmp_path: Path) -> None:
    root = build_store(tmp_path)
    weights = ollamastore.resident(root, "qwen3.5:9b-bf16")
    assert weights.provenance == f"ollama:qwen3.5:9b-bf16@{MODEL_DIGEST}"
    assert "unsloth" not in weights.provenance
    assert weights.provenance.startswith("ollama:")


def test_the_reported_size_counts_every_file_that_will_be_loaded(
    tmp_path: Path,
) -> None:
    root = build_store(tmp_path, projector=True)
    weights = ollamastore.resident(root, "qwen3.5:9b-bf16", wants_projector=True)
    assert weights.bytes == weights.model.bytes + weights.projector.bytes


@pytest.mark.parametrize(
    "written,expected",
    [
        ("qwen3.5:9b-bf16", ("registry.ollama.ai", "library", "qwen3.5", "9b-bf16")),
        ("library/qwen3.5:9b-bf16", ("registry.ollama.ai", "library", "qwen3.5", "9b-bf16")),
        (
            "registry.ollama.ai/library/qwen3.5:9b-bf16",
            ("registry.ollama.ai", "library", "qwen3.5", "9b-bf16"),
        ),
        ("hf.co/owen/thing:q4", ("hf.co", "owen", "thing", "q4")),
    ],
)
def test_every_spelling_ollama_accepts_resolves_the_same(
    written: str, expected: tuple[str, str, str, str]
) -> None:
    assert ollamastore.split_tag(written) == expected


@pytest.mark.parametrize("bad", ["", "qwen3.5", "a:b:c", "a/b/c/d:e"])
def test_a_thing_that_is_not_a_tag_is_refused(bad: str) -> None:
    with pytest.raises(ollamastore.OllamaStoreError):
        ollamastore.split_tag(bad)


def test_held_lists_what_could_be_offered_without_a_download(tmp_path: Path) -> None:
    root = build_store(tmp_path)
    assert ollamastore.held(root) == ("qwen3.5:9b-bf16",)
    for tag in ollamastore.held(root):
        assert ollamastore.resident(root, tag).tag == tag


def test_held_is_empty_rather_than_angry_when_there_is_no_store(
    tmp_path: Path,
) -> None:
    assert ollamastore.held(tmp_path / "nothing") == ()
