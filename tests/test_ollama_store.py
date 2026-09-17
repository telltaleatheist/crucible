"""Reading the GGUFs Ollama already has.

Owen, 2026-09-16: *"i dont want to have 16 copies of giant models sitting
around"*, and *"windows crucible should function if there's no wsl… It should be
able to use the ollama models or api keys via Bookforge/foundry."*

Every fixture here is shaped like the real store on Owen's PC, read on
2026-09-16: a `manifests/registry.ollama.ai/library/<name>/<label>` file holding
a v2 manifest whose `application/vnd.ollama.image.model` layer names a
`blobs/sha256-<digest>` file. The sizes are small; the shape is not invented.
"""

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
        # The two layers that are NOT weights, present because the real store has
        # them and a reader that picked the first layer would pick a licence.
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


# ------------------------------------------------------------------ the store


def test_the_store_is_found_where_ollama_puts_it(tmp_path: Path) -> None:
    assert ollamastore.store_root(environ={}, home=tmp_path) == (
        tmp_path / ".ollama" / "models"
    )


def test_the_environment_moves_it(tmp_path: Path) -> None:
    """Read rather than assumed, because a store moved to another drive is
    exactly what somebody does about a C: drive that has filled twice."""
    moved = tmp_path / "elsewhere"
    assert ollamastore.store_root(
        environ={ollamastore.STORE_ENV: str(moved)}, home=tmp_path
    ) == moved


def test_no_store_and_no_tag_are_different_answers(tmp_path: Path) -> None:
    """Five ways of not finding a tag, five things to do about it.

    A `None` return would have collapsed "you do not have Ollama" and "you have
    Ollama and not that model" into one sentence, and they lead somewhere
    different.
    """
    empty = tmp_path / "nothing"
    with pytest.raises(ollamastore.OllamaStoreError, match="there is no Ollama store"):
        ollamastore.resident(empty, "qwen3.5:9b-bf16")

    root = build_store(tmp_path)
    with pytest.raises(ollamastore.OllamaStoreError, match="not in this Ollama store"):
        ollamastore.resident(root, "qwen3.5:70b-bf16")


# ------------------------------------------------------------------ the layer


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
    """The one failure worth the stat() call.

    A short GGUF reaches llama.cpp as a parse error several layers from the file
    that is wrong, which is a bad afternoon. The digest is NOT verified — it is
    up to 19 GB and this runs on the path to a model load — and the docstring
    says so rather than letting a reader assume the stronger check.
    """
    root = build_store(tmp_path, model_bytes=b"GGUF-short", declared=999_999)
    with pytest.raises(ollamastore.OllamaStoreError, match="bytes and the manifest says"):
        ollamastore.resident(root, "qwen3.5:9b-bf16")


def test_a_vision_model_gets_its_projector_or_a_refusal(tmp_path: Path) -> None:
    """Half a vision model is a model that loads and then cannot see."""
    with_proj = build_store(tmp_path / "a", projector=True)
    assert ollamastore.resident(
        with_proj, "qwen3.5:9b-bf16", wants_projector=True
    ).projector is not None

    without = build_store(tmp_path / "b")
    with pytest.raises(ollamastore.OllamaStoreError, match="cannot see"):
        ollamastore.resident(without, "qwen3.5:9b-bf16", wants_projector=True)


# ------------------------------------------------------------- the provenance


def test_the_provenance_never_claims_to_be_the_hub_pin(tmp_path: Path) -> None:
    """THE CENTRAL RULE OF THIS MODULE.

    Ollama's `qwen3.5:9b-bf16` and the manifest's `unsloth/Qwen3.5-9B-GGUF`
    `Q8_0` are two forms of one model quantized by two different people. Serving
    one under the other's fingerprint would be a silent substitution into a
    RECORD — Foundry hashes a fingerprint into its cleanup cache key and
    BookForge stamps it into a book's OPF — so a cache would answer for weights
    that never ran.
    """
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


# ------------------------------------------------------------------- the tags


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
    """A `[local]` table is allowed to be explicit, and a reader should not have
    to know that the short form is the only one that works."""
    assert ollamastore.split_tag(written) == expected


@pytest.mark.parametrize("bad", ["", "qwen3.5", "a:b:c", "a/b/c/d:e"])
def test_a_thing_that_is_not_a_tag_is_refused(bad: str) -> None:
    with pytest.raises(ollamastore.OllamaStoreError):
        ollamastore.split_tag(bad)


def test_held_lists_what_could_be_offered_without_a_download(tmp_path: Path) -> None:
    """For a settings page that offers what is on disk before it offers 19 GB.

    Every tag it returns can be handed straight back to `resident()`, which is
    why the default registry and namespace come back spelled short and anything
    else comes back spelled in full.
    """
    root = build_store(tmp_path)
    assert ollamastore.held(root) == ("qwen3.5:9b-bf16",)
    for tag in ollamastore.held(root):
        assert ollamastore.resident(root, tag).tag == tag


def test_held_is_empty_rather_than_angry_when_there_is_no_store(
    tmp_path: Path,
) -> None:
    """A settings page asks this before it knows whether Ollama is installed."""
    assert ollamastore.held(tmp_path / "nothing") == ()
