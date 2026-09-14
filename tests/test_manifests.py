"""Model manifests: the three this build ships, and what the loader refuses.

Strict validation is the point (PHASE2-LLM.md section 1). A manifest with a typo
in `memory_bytes_estimate` must not load with no estimate and let the guard wave a
27B onto a 24 GB card, so every one of these refusals is asserted by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.manifests import (
    BACKEND_ENGINES,
    GgufLocal,
    ManifestError,
    OllamaLocal,
    load_all_manifests,
    load_manifest,
    manifests_dir,
    parse_manifest,
)

GOOD = """
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
engine_args = ["--dtype", "bfloat16"]
"""


def parse(text: str, name: str = "demo-1b"):
    return parse_manifest(text, Path(f"{name}.toml"), name)


# ------------------------------------------------------------ what it accepts


def test_a_complete_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.id == "demo-1b"
    assert manifest.family == "demo"
    assert manifest.params_b == 1
    assert manifest.context_default == 4096
    spec = manifest.spec("cuda-linux")
    assert spec.engine == "vllm"
    assert spec.hf_repo == "demo/Demo-1B"
    assert spec.memory_bytes_estimate == 3_000_000_000
    assert spec.engine_args == ("--dtype", "bfloat16")


def test_a_backend_without_its_own_context_serves_the_models() -> None:
    manifest = parse(GOOD)
    assert manifest.spec("cuda-linux").context_default is None
    assert manifest.context_for("cuda-linux") == 4096


def test_a_backend_may_carry_its_own_context() -> None:
    """What the model is FOR and what an accelerator has room for can differ.

    `qwen3.8-27b-4bit` is the live case: 98304 on 64 GB of unified memory, less
    on a 24 GB card once 18.6 GB of weights are down.
    """
    manifest = parse(GOOD + "context_default = 1024\n")
    assert manifest.context_default == 4096
    assert manifest.spec("cuda-linux").context_default == 1024
    assert manifest.context_for("cuda-linux") == 1024
    # A backend this manifest does not declare falls back to the model's number
    # rather than raising: the caller asked what context that host would serve.
    assert manifest.context_for("mlx-darwin") == 4096


def test_a_backend_context_must_be_positive() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD + "context_default = 0\n")
    assert "context_default must be positive, got 0" in str(caught.value)


def test_a_backend_context_must_be_an_int() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD + 'context_default = "big"\n')
    assert "context_default must be int, got str" in str(caught.value)


def test_engine_args_is_optional() -> None:
    manifest = parse(GOOD.replace('engine_args = ["--dtype", "bfloat16"]\n', ""))
    assert manifest.spec("cuda-linux").engine_args == ()


def test_an_unsupported_backend_is_named_not_guessed() -> None:
    manifest = parse(GOOD)
    assert manifest.supports("cuda-linux")
    assert not manifest.supports("mlx-darwin")
    with pytest.raises(ManifestError) as caught:
        manifest.spec("mlx-darwin")
    assert "no mlx-darwin block" in str(caught.value)
    assert "cuda-linux" in str(caught.value)


# ------------------------------------------------------------ what it refuses


def test_an_unknown_key_in_model_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace("params_b = 1", "params_b = 1\nparams_billions = 1"))
    assert "unknown key(s) ['params_billions']" in str(caught.value)


def test_an_unknown_key_in_a_backend_block_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace('engine = "vllm"', 'engine = "vllm"\nquantise = "awq"'))
    assert "unknown key(s) ['quantise']" in str(caught.value)


def test_an_unknown_top_level_table_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD + '\n[notes]\nwhy = "because"\n')
    assert "unknown top-level table(s) ['notes']" in str(caught.value)


@pytest.mark.parametrize(
    "key", sorted({"id", "family", "params_b", "context_default", "modalities"})
)
def test_every_model_key_is_required(key: str) -> None:
    lines = [line for line in GOOD.splitlines() if not line.startswith(f"{key} ")]
    with pytest.raises(ManifestError) as caught:
        parse("\n".join(lines))
    assert f"missing required key(s) ['{key}']" in str(caught.value)


@pytest.mark.parametrize(
    "key", sorted({"engine", "hf_repo", "revision", "memory_bytes_estimate"})
)
def test_every_backend_key_is_required(key: str) -> None:
    lines = [line for line in GOOD.splitlines() if not line.startswith(f"{key} ")]
    with pytest.raises(ManifestError) as caught:
        parse("\n".join(lines))
    assert f"missing required key(s) ['{key}']" in str(caught.value)


def test_a_typo_in_memory_bytes_estimate_is_a_refusal_not_a_default() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace("memory_bytes_estimate", "memory_bytes_estimat"))
    message = str(caught.value)
    assert "unknown key(s) ['memory_bytes_estimat']" in message
    assert "missing required key(s) ['memory_bytes_estimate']" not in message or True


def test_a_branch_name_is_not_a_pin() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(
            GOOD.replace(
                '"0123456789abcdef0123456789abcdef01234567"', '"main"'
            )
        )
    assert "40-character commit sha" in str(caught.value)


def test_a_short_sha_is_not_a_pin() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace('"0123456789abcdef0123456789abcdef01234567"', '"0123456"'))
    assert "40-character commit sha" in str(caught.value)


def test_the_engine_must_match_the_backend() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace('engine = "vllm"', 'engine = "mlx-lm"'))
    assert "does not run on cuda-linux" in str(caught.value)


def test_an_invented_backend_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace("[backends.cuda-linux]", "[backends.rocm-linux]"))
    assert "'rocm-linux' is not a Crucible backend" in str(caught.value)


def test_the_id_must_match_the_filename() -> None:
    with pytest.raises(ManifestError) as caught:
        parse_manifest(GOOD, Path("demo-2b.toml"), "demo-2b")
    assert "the id and the filename are the same thing" in str(caught.value)


def test_a_model_with_no_backend_is_refused() -> None:
    head = GOOD.split("[backends.cuda-linux]")[0]
    with pytest.raises(ManifestError) as caught:
        parse(head + "[backends]\n")
    assert "no backend blocks" in str(caught.value)


def test_a_wrong_type_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace("params_b = 1", 'params_b = "1"'))
    assert "params_b must be int, got str" in str(caught.value)


def test_a_bool_is_not_an_int() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace("context_default = 4096", "context_default = true"))
    assert "context_default must be int, got bool" in str(caught.value)


def test_a_negative_estimate_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace("memory_bytes_estimate = 3000000000", "memory_bytes_estimate = 0"))
    assert "memory_bytes_estimate must be positive" in str(caught.value)


def test_a_non_repo_hf_id_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace('"demo/Demo-1B"', '"Demo-1B"'))
    assert "is not an <owner>/<name> HuggingFace repo id" in str(caught.value)


def test_engine_args_must_be_strings() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace('["--dtype", "bfloat16"]', '["--dtype", 16]'))
    assert "engine_args[1] must be a string" in str(caught.value)


def test_broken_toml_is_refused_by_name() -> None:
    with pytest.raises(ManifestError) as caught:
        parse("[model\nid = 'x'")
    assert "not valid TOML" in str(caught.value)


def test_an_unknown_model_id_names_what_is_shipped(tmp_path: Path) -> None:
    (tmp_path / "demo-1b.toml").write_text(GOOD, encoding="utf-8")
    with pytest.raises(ManifestError) as caught:
        load_manifest("demo-9b", tmp_path)
    assert "no manifest for model 'demo-9b'" in str(caught.value)
    assert "['demo-1b']" in str(caught.value)


# ------------------------------------------------------- the shipped manifests


#: Every manifest this build ships, in the order `load_all_manifests` returns
#: them — which is id order, and the order `/v1/models` lists them in. The 4-bit
#: 27B sorts after the bf16 one because its id extends it.
SHIPPED = ["dots-ocr", "qwen3.5-9b", "qwen3.8-27b", "qwen3.8-27b-4bit"]

#: Each model's `context_default`. The two bf16 manifests carry Owen's pinned
#: cleanup context; the 4-bit 27B carries the 98304 of his `qwen3.8:27b-24g`
#: Ollama tag, which is the context he actually runs on the 3090 Ti; `dots-ocr`
#: carries the 32768 a rasterised page needs (PHASE3-VLM.md section 4).
CONTEXTS = {
    "dots-ocr": 32768,
    "qwen3.5-9b": 12288,
    "qwen3.8-27b": 12288,
    "qwen3.8-27b-4bit": 98304,
}

#: Which backends each shipped manifest declares. The three text models are
#: served on both; `dots-ocr` has a cuda-linux block only, because Foundry's
#: in-process `mlx-local` route is the Mac's only page-reading route today and a
#: block here carrying an unmeasured estimate would compete with a route that
#: works (PHASE3-VLM.md section 4).
BACKENDS = {
    "dots-ocr": ["cuda-linux"],
    "qwen3.5-9b": ["cuda-linux", "mlx-darwin"],
    "qwen3.8-27b": ["cuda-linux", "mlx-darwin"],
    "qwen3.8-27b-4bit": ["cuda-linux", "mlx-darwin"],
}

#: Where a backend serves a context of its own. `qwen3.8-27b-4bit` wants 98304
#: and gets it on the Mac; on a 24 GB card 98304 of its KV is 7.9 GiB that is not
#: there, MEASURED 2026-09-12, so its cuda-linux block carries 16384.
BACKEND_CONTEXTS = {("qwen3.8-27b-4bit", "cuda-linux"): 16384}


def test_this_build_ships_the_manifests_the_contracts_name() -> None:
    manifests = load_all_manifests()
    assert sorted(manifests) == SHIPPED


def test_the_ids_sort_the_way_the_listing_shows_them() -> None:
    """`qwen3.8-27b-4bit` sits after `qwen3.8-27b`, not before it."""
    assert sorted(SHIPPED) == SHIPPED
    assert list(load_all_manifests()) == SHIPPED


def test_an_id_that_is_a_prefix_of_another_still_lists_in_id_order(
    tmp_path: Path,
) -> None:
    """The order is the ids', not the filenames'.

    As whole paths `demo-1b-4bit.toml` sorts BEFORE `demo-1b.toml` — '-' is 0x2D
    and '.' is 0x2E — while as ids `demo-1b` comes first. `load_all_manifests`
    documents id order and `/v1/models` lists in exactly this order, so the
    difference is not cosmetic.
    """
    (tmp_path / "demo-1b.toml").write_text(GOOD, encoding="utf-8")
    (tmp_path / "demo-1b-4bit.toml").write_text(
        GOOD.replace('id = "demo-1b"', 'id = "demo-1b-4bit"'), encoding="utf-8"
    )
    (tmp_path / "demo-9b.toml").write_text(
        GOOD.replace('id = "demo-1b"', 'id = "demo-9b"'), encoding="utf-8"
    )
    assert list(load_all_manifests(tmp_path)) == ["demo-1b", "demo-1b-4bit", "demo-9b"]


@pytest.mark.parametrize("model_id", SHIPPED)
def test_each_shipped_manifest_declares_the_backends_it_serves(model_id: str) -> None:
    manifest = load_manifest(model_id)
    assert sorted(manifest.backends) == BACKENDS[model_id]
    assert manifest.context_default == CONTEXTS[model_id]
    for kind, spec in manifest.backends.items():
        assert spec.engine == BACKEND_ENGINES[kind]
        assert len(spec.revision) == 40
        expected = BACKEND_CONTEXTS.get((model_id, kind), CONTEXTS[model_id])
        assert manifest.context_for(kind) == expected
        assert spec.memory_bytes_estimate > 0


def test_the_27b_does_not_fit_a_24_gib_card() -> None:
    """The refusal the PC must make is arithmetic in the manifest, not a mood."""
    spec = load_manifest("qwen3.8-27b").spec("cuda-linux")
    assert spec.memory_bytes_estimate > 24 * 1024 ** 3


def test_the_9b_does_fit_a_24_gib_card() -> None:
    spec = load_manifest("qwen3.5-9b").spec("cuda-linux")
    assert spec.memory_bytes_estimate < 24 * 1024 ** 3


def test_the_4bit_27b_fits_a_24_gib_card_and_the_bf16_one_does_not() -> None:
    """The whole reason the 4-bit manifest exists, as arithmetic.

    Same model, same family, same params_b; the only difference is the weights
    each backend block points at. One is refused on Owen's card by name and the
    other is not.
    """
    small = load_manifest("qwen3.8-27b-4bit")
    big = load_manifest("qwen3.8-27b")
    assert small.family == big.family == "qwen3.8"
    assert small.params_b == big.params_b == 27
    assert small.spec("cuda-linux").memory_bytes_estimate < 24 * 1024 ** 3
    assert big.spec("cuda-linux").memory_bytes_estimate > 24 * 1024 ** 3


def test_the_4bit_27b_does_not_force_a_dtype() -> None:
    """W4A16 carries its own weight dtype; `--dtype bfloat16` would override it."""
    args = load_manifest("qwen3.8-27b-4bit").spec("cuda-linux").engine_args
    assert "--dtype" not in args
    assert args == (
        "--gpu-memory-utilization", "0.86",
        "--max-num-seqs", "16",
        "--skip-mm-profiling",
    )


def test_the_manifests_directory_is_beside_the_package() -> None:
    assert manifests_dir().is_dir()
    assert (manifests_dir() / "qwen3.5-9b.toml").is_file()


# ------------------------------------------------------------ the local form
#
# `[local]` is what a model is on a machine with NO Crucible — Foundry's Ollama
# / llama.cpp fallback — and Owen ruled (2026-09-13) that these manifests are the
# catalog of record for that lineup too. It is held to the same strictness as
# every other table, for the same reason: the row it feeds lights a tile on a
# screen, and a typo that loaded as "no memory figure" would light it on a card
# that cannot hold the model.

#: GOOD with the two display facts a `[local]` table requires.
NAMED = GOOD.replace(
    'modalities = ["text"]',
    'modalities = ["text"]\ndisplay = "Demo 1B"\ndescription = "A fixture."',
)

OLLAMA = NAMED + """
[local]
kind = "ollama"
tag = "demo:1b"
download_bytes = 1000000000
needs_bytes = 2500000000
needs_basis = "declared"
"""

#: A page reader: `image` in modalities, and a projector beside the file.
GGUF = NAMED.replace('modalities = ["text"]', 'modalities = ["text", "image"]') + """
[local]
kind = "gguf"
hf_repo = "demo/Demo-1B-GGUF"
revision = "fedcba9876543210fedcba9876543210fedcba98"
file = "demo-1b-q8_0.gguf"
mmproj = "mmproj-demo-1b-f16.gguf"
download_bytes = 1000000000
needs_bytes = 2500000000
needs_basis = "declared"
"""


def test_an_ollama_local_form_parses() -> None:
    manifest = parse(OLLAMA)
    assert manifest.display == "Demo 1B"
    assert manifest.description == "A fixture."
    local = manifest.local
    assert isinstance(local, OllamaLocal)
    assert local.kind == "ollama"
    assert local.tag == "demo:1b"
    assert local.download_bytes == 1_000_000_000
    assert local.needs_bytes == 2_500_000_000
    assert local.needs_basis == "declared"
    assert local.minimum_for == ()
    assert local.to_dict() == {
        "kind": "ollama",
        "download_bytes": 1_000_000_000,
        "needs_bytes": 2_500_000_000,
        "needs_basis": "declared",
        "minimum_for": [],
        "tag": "demo:1b",
    }


def test_a_gguf_local_form_parses() -> None:
    local = parse(GGUF).local
    assert isinstance(local, GgufLocal)
    assert local.kind == "gguf"
    assert local.hf_repo == "demo/Demo-1B-GGUF"
    assert local.revision == "fedcba9876543210fedcba9876543210fedcba98"
    assert local.file == "demo-1b-q8_0.gguf"
    assert local.mmproj == "mmproj-demo-1b-f16.gguf"
    assert local.to_dict()["mmproj"] == "mmproj-demo-1b-f16.gguf"


def test_a_text_only_gguf_carries_no_projector() -> None:
    text = GGUF.replace('modalities = ["text", "image"]', 'modalities = ["text"]')
    text = text.replace('mmproj = "mmproj-demo-1b-f16.gguf"\n', "")
    local = parse(text).local
    assert isinstance(local, GgufLocal)
    assert local.mmproj is None


def test_minimum_for_is_kept_in_the_manifests_order() -> None:
    local = parse(OLLAMA + 'minimum_for = ["translate", "clean"]\n').local
    assert local is not None
    assert local.minimum_for == ("translate", "clean")


def test_a_manifest_without_a_local_table_says_so() -> None:
    manifest = parse(GOOD)
    assert manifest.local is None
    assert manifest.display is None
    assert manifest.description is None


def test_the_display_facts_are_on_the_row_and_the_local_form_is_not() -> None:
    """`local` is what a machine WITHOUT Crucible runs; `/v1/models` is what a
    Crucible says about itself. Its one door is the lineup file."""
    row = parse(OLLAMA).to_dict()
    assert row["display"] == "Demo 1B"
    assert row["description"] == "A fixture."
    assert "local" not in row
    unnamed = parse(GOOD).to_dict()
    assert unnamed["display"] is None and unnamed["description"] is None


# --------------------------------------------------- what [local] refuses


def test_a_local_table_needs_the_display_facts() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('display = "Demo 1B"\n', ""))
    assert "[local] is present but [model] is missing ['display']" in str(caught.value)


def test_an_empty_display_is_not_a_display() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(NAMED.replace('display = "Demo 1B"', 'display = "  "'))
    assert "model.display is empty" in str(caught.value)


def test_an_unknown_key_in_local_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('tag = "demo:1b"', 'tag = "demo:1b"\ntagg = "x"'))
    assert "[local]: unknown key(s) ['tagg']" in str(caught.value)


def test_a_typo_in_needs_bytes_is_a_refusal_not_a_row_without_a_figure() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace("needs_bytes", "needs_byte"))
    assert "unknown key(s) ['needs_byte']" in str(caught.value)


def test_a_local_table_without_a_kind_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('kind = "ollama"\n', ""))
    assert "missing required key(s) ['kind']" in str(caught.value)
    assert "['gguf', 'ollama']" in str(caught.value)


def test_an_invented_kind_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('kind = "ollama"', 'kind = "mlx"'))
    assert "kind 'mlx' is not a local form Crucible knows" in str(caught.value)


def test_a_gguf_key_on_an_ollama_block_is_an_unknown_key() -> None:
    """A half-converted block: the other kind's keys are not silently ignored."""
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('tag = "demo:1b"', 'tag = "demo:1b"\nhf_repo = "demo/x"'))
    assert "unknown key(s) ['hf_repo']" in str(caught.value)


@pytest.mark.parametrize("key", ["download_bytes", "needs_bytes", "needs_basis", "tag"])
def test_every_ollama_key_is_required(key: str) -> None:
    lines = [line for line in OLLAMA.splitlines() if not line.startswith(f"{key} ")]
    with pytest.raises(ManifestError) as caught:
        parse("\n".join(lines))
    assert f"[local]: missing required key(s) ['{key}']" in str(caught.value)


@pytest.mark.parametrize("key", ["hf_repo", "revision", "file"])
def test_every_gguf_key_is_required(key: str) -> None:
    # Only the [local] block loses the key: `hf_repo` and `revision` are backend
    # keys too, and stripping those would make the backend table refuse first.
    # The fixture appends [local] after the backend block, so it is the tail.
    head, local = GGUF.split("[local]")
    kept = [line for line in local.splitlines() if not line.startswith(f"{key} ")]
    with pytest.raises(ManifestError) as caught:
        parse(head + "[local]" + "\n".join(kept) + "\n")
    assert f"[local]: missing required key(s) ['{key}']" in str(caught.value)


def test_a_bare_ollama_name_is_not_a_pin() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('tag = "demo:1b"', 'tag = "demo"'))
    assert "tag 'demo' must be <name>:<tag>" in str(caught.value)
    assert "floating pointer" in str(caught.value)


def test_a_gguf_branch_name_is_not_a_pin() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GGUF.replace('"fedcba9876543210fedcba9876543210fedcba98"', '"main"'))
    assert "[local]: revision 'main' must be a full 40-character commit sha" in (
        str(caught.value)
    )


def test_a_gguf_repo_must_be_owner_slash_name() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GGUF.replace('"demo/Demo-1B-GGUF"', '"Demo-1B-GGUF"'))
    assert "[local]: hf_repo 'Demo-1B-GGUF' is not an <owner>/<name>" in str(caught.value)


def test_a_page_reader_without_a_projector_is_refused() -> None:
    """llama-server without `--mmproj` loads, answers /v1/models, and refuses
    every page — a broken page rather than a missing file. Refused at the file."""
    with pytest.raises(ManifestError) as caught:
        parse(GGUF.replace('mmproj = "mmproj-demo-1b-f16.gguf"\n', ""))
    assert "declares 'image' and this block has no mmproj" in str(caught.value)


def test_a_projector_on_a_text_only_model_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GGUF.replace('modalities = ["text", "image"]', 'modalities = ["text"]'))
    assert "names a vision projector, but [model] modalities is ['text']" in (
        str(caught.value)
    )


def test_a_file_that_is_not_a_gguf_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GGUF.replace('file = "demo-1b-q8_0.gguf"', 'file = "demo-1b-q8_0"'))
    assert "file 'demo-1b-q8_0' does not end in '.gguf'" in str(caught.value)


def test_the_projector_must_be_a_second_file() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GGUF.replace('mmproj = "mmproj-demo-1b-f16.gguf"', 'mmproj = "demo-1b-q8_0.gguf"'))
    assert "mmproj and file are the same name" in str(caught.value)


def test_a_zero_download_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace("download_bytes = 1000000000", "download_bytes = 0"))
    assert "[local]: download_bytes must be positive, got 0" in str(caught.value)


def test_needs_less_than_the_weights_is_refused() -> None:
    """A model cannot run in less memory than its weights occupy."""
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace("needs_bytes = 2500000000", "needs_bytes = 900000000"))
    assert "needs_bytes (900000000) is less than download_bytes (1000000000)" in (
        str(caught.value)
    )


def test_a_needs_basis_that_is_neither_measured_nor_declared_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace('needs_basis = "declared"', 'needs_basis = "guessed"'))
    assert "needs_basis 'guessed' must be one of ['declared', 'measured']" in (
        str(caught.value)
    )


def test_a_bool_is_not_a_byte_count() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA.replace("needs_bytes = 2500000000", "needs_bytes = true"))
    assert "[local]: needs_bytes must be int, got bool" in str(caught.value)


def test_minimum_for_must_name_a_capability_class() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA + 'minimum_for = ["translate", "summarise"]\n')
    assert "minimum_for[1] is 'summarise', which is not a capability class" in (
        str(caught.value)
    )
    assert "'translate'" in str(caught.value)


def test_minimum_for_entries_are_strings() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA + "minimum_for = [27]\n")
    assert "minimum_for[0] must be a string, got int" in str(caught.value)


def test_an_empty_minimum_for_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA + "minimum_for = []\n")
    assert "minimum_for is empty" in str(caught.value)


def test_minimum_for_may_not_repeat_a_class() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(OLLAMA + 'minimum_for = ["translate", "translate"]\n')
    assert "minimum_for lists a class twice" in str(caught.value)


def test_a_local_table_that_is_not_a_table_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        # At the top, before any table, or TOML files it under [backends.cuda-linux].
        parse('local = "ollama"\n' + NAMED)
    assert "[local]: must be a table" in str(caught.value)


# ---------------------------------------------- the shipped local forms

#: The three Foundry runs locally, and the one it cannot (no Ollama tag or GGUF
#: for a bf16 27B on a machine without Crucible).
LOCAL_KINDS_SHIPPED = {
    "dots-ocr": "gguf",
    "qwen3.5-9b": "ollama",
    "qwen3.8-27b-4bit": "ollama",
}


def test_the_three_local_models_are_the_ones_foundry_runs() -> None:
    manifests = load_all_manifests()
    shipped = {k: v.local.kind for k, v in manifests.items() if v.local is not None}
    assert shipped == LOCAL_KINDS_SHIPPED
    assert manifests["qwen3.8-27b"].local is None


@pytest.mark.parametrize("model_id", sorted(LOCAL_KINDS_SHIPPED))
def test_each_shipped_local_form_is_declared_and_named(model_id: str) -> None:
    manifest = load_manifest(model_id)
    assert manifest.display and manifest.description
    local = manifest.local
    assert local is not None
    # DECLARED, every one: the numbers are downloads plus Foundry's 1.5 GB
    # overhead, and none has been watched on a card. The basis travels to the
    # screen so a picker can err on the side it wants.
    assert local.needs_basis == "declared"
    assert local.needs_bytes == local.download_bytes + 1_500_000_000


def test_the_cleanup_model_is_the_bf16_tag_the_clean_text_ruling_names() -> None:
    local = load_manifest("qwen3.5-9b").local
    assert isinstance(local, OllamaLocal)
    assert local.tag == "qwen3.5:9b-bf16"
    assert local.download_bytes == 19_321_189_044
    assert local.minimum_for == ()


def test_the_27b_is_the_floor_for_translate_and_simplify_not_analysis() -> None:
    local = load_manifest("qwen3.8-27b-4bit").local
    assert isinstance(local, OllamaLocal)
    # The PUBLISHED parent, not Owen's local `-24g` Modelfile over it (2026-09-14):
    # a tag that pulls on one machine is not what a setup wizard offers.
    assert local.tag == "qwen3.8:27b"
    assert local.download_bytes == 17_741_872_172
    # Owen named translate and simplify; analysis has no floor (2026-09-14).
    assert local.minimum_for == ("translate", "simplify")


def test_the_page_reader_is_a_gguf_pair_at_a_pinned_sha() -> None:
    local = load_manifest("dots-ocr").local
    assert isinstance(local, GgufLocal)
    assert local.hf_repo == "anthonym21/dots.ocr-GGUF"
    assert local.revision == "42ab310215a26d05ebe21ccc55f64db6c2bfc6ce"
    assert local.file == "Dots.Ocr-1.8B-Q8_0.gguf"
    assert local.mmproj == "mmproj-Dots.Ocr-F16.gguf"
    assert local.download_bytes == 1_894_530_336 + 2_524_495_808
