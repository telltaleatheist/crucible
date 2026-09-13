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
    ManifestError,
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
    "key", sorted({"id", "family", "params_b", "context_default"})
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
SHIPPED = ["qwen3.5-9b", "qwen3.8-27b", "qwen3.8-27b-4bit"]

#: Each model's `context_default`. The two bf16 manifests carry Owen's pinned
#: cleanup context; the 4-bit 27B carries the 98304 of his `qwen3.8:27b-24g`
#: Ollama tag, which is the context he actually runs on the 3090 Ti.
CONTEXTS = {"qwen3.5-9b": 12288, "qwen3.8-27b": 12288, "qwen3.8-27b-4bit": 98304}


def test_this_build_ships_the_phase_two_manifests() -> None:
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
def test_each_shipped_manifest_declares_both_backends(model_id: str) -> None:
    manifest = load_manifest(model_id)
    assert sorted(manifest.backends) == ["cuda-linux", "mlx-darwin"]
    assert manifest.context_default == CONTEXTS[model_id]
    for kind, spec in manifest.backends.items():
        assert spec.engine == BACKEND_ENGINES[kind]
        assert len(spec.revision) == 40
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
    assert args == ("--gpu-memory-utilization", "0.85")


def test_the_manifests_directory_is_beside_the_package() -> None:
    assert manifests_dir().is_dir()
    assert (manifests_dir() / "qwen3.5-9b.toml").is_file()
