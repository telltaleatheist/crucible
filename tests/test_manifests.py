from __future__ import annotations

from pathlib import Path

import pytest

from crucible.manifests import (
    BACKEND_ENGINES,
    GgufLocal,
    ManifestError,
    OllamaLocal,
    class_family,
    engine_for,
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
trained_context = 262144
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
    manifest = parse(GOOD + "context_default = 1024\n")
    assert manifest.context_default == 4096
    assert manifest.spec("cuda-linux").context_default == 1024
    assert manifest.context_for("cuda-linux") == 1024
    assert manifest.context_for("mlx-darwin") == 4096


def test_a_backend_context_must_be_positive() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD + "context_default = 0\n")
    assert "context_default must be positive, got 0" in str(caught.value)


def test_a_backend_context_must_be_an_int() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD + 'context_default = "big"\n')
    assert "context_default must be int, got str" in str(caught.value)


def test_a_block_without_max_context_is_capped_at_its_own_default() -> None:
    manifest = parse(GOOD)
    assert manifest.spec("cuda-linux").max_context is None
    assert manifest.max_context_for("cuda-linux") == 4096
    over = parse(GOOD + "context_default = 1024\n")
    assert over.max_context_for("cuda-linux") == 1024


def test_a_block_may_state_its_max_context() -> None:
    manifest = parse(GOOD + "max_context = 32768\n")
    assert manifest.spec("cuda-linux").max_context == 32768
    assert manifest.max_context_for("cuda-linux") == 32768
    assert manifest.context_for("cuda-linux") == 4096
    assert manifest.spec("cuda-linux").to_dict()["max_context"] == 32768


@pytest.mark.parametrize(
    ("line", "said"),
    [
        ("max_context = 0\n", "max_context must be positive, got 0"),
        ("max_context = -8192\n", "max_context must be positive, got -8192"),
        ("max_context = 524288\n", "max_context is 524288 and the weights are trained at 262144"),
        ("max_context = 2048\n", "max_context is 2048 and this block serves 4096"),
        ('max_context = "big"\n', "max_context must be int, got str"),
        ("max_context = true\n", "max_context must be int, got bool"),
        ("max_context = 32768.0\n", "max_context must be int, got float"),
    ],
)
def test_an_invalid_max_context_is_refused_by_name(line: str, said: str) -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD + line)
    assert said in str(caught.value)


def test_max_context_is_held_to_the_blocks_own_default_not_the_models() -> None:
    manifest = parse(GOOD + "context_default = 1024\nmax_context = 2048\n")
    assert manifest.max_context_for("cuda-linux") == 2048


def test_the_shipped_maxima_are_the_computed_ones() -> None:
    found = {
        (m.id, kind): m.max_context_for(kind)
        for m in load_all_manifests().values()
        for kind in m.backends
    }
    assert found[("qwen3.8-27b-4bit", "cuda-linux")] == 32768
    assert found[("qwen3.5-9b", "cuda-linux")] == 65536
    assert found[("qwen3.8-27b-8bit", "mlx-darwin")] == 131072
    assert found[("qwen3.8-27b-4bit", "mlx-darwin")] == 131072
    assert found[("qwen3.5-9b", "mlx-darwin")] == 131072
    assert found[("qwen3.8-27b-4bit", "llama-windows")] == 65536
    assert found[("qwen3.5-9b-vl", "cuda-linux")] == 16384
    assert found[("qwen3.8-27b-4bit-vl", "cuda-linux")] == 16384
    assert found[("dots-ocr", "cuda-linux")] == 32768


def test_the_8bit_mac_terms_count_kv_once() -> None:
    terms = load_manifest("qwen3.8-27b-8bit").spec("mlx-darwin").memory
    assert terms.overhead_bytes == 17_818_943_521 - 98_220 * 65_536 == 11_381_997_601
    assert terms.bytes_for(context=12288, concurrency=1) == (
        load_manifest("qwen3.8-27b-8bit").spec("mlx-darwin").memory_bytes_estimate
    )


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


def test_the_engine_must_match_the_backend_and_the_family() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(GOOD.replace('engine = "vllm"', 'engine = "mlx-lm"'))
    assert "does not serve 'text' models on cuda-linux" in str(caught.value)


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
    assert "params_b must be int or float, got str" in str(caught.value)


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


SHIPPED = [
    "dots-ocr", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
    "qwen3.8-27b-4bit", "qwen3.8-27b-8bit",
]
ALIASES = ["qwen3.5-9b-vl", "qwen3.8-27b-4bit-vl"]

CONTEXTS = {
    "dots-ocr": 32768,
    "qwen3.5-9b": 16384,
    "qwen3.5-4b": 16384,
    "qwen3.5-2b": 16384,
    "qwen3.5-0.8b": 8192,
    "qwen3.8-27b-8bit": 12288,
    "qwen3.8-27b-4bit": 98304,
}

BACKENDS = {
    "dots-ocr": ["cuda-linux", "llama-windows", "mlx-darwin"],
    "qwen3.5-9b": ["cuda-linux", "llama-windows", "mlx-darwin"],
    "qwen3.5-4b": ["cuda-linux", "llama-windows", "mlx-darwin"],
    "qwen3.5-2b": ["cuda-linux", "llama-windows", "mlx-darwin"],
    "qwen3.5-0.8b": ["cuda-linux", "llama-windows", "mlx-darwin"],
    "qwen3.8-27b-8bit": ["mlx-darwin"],
    "qwen3.8-27b-4bit": ["cuda-linux", "llama-windows", "mlx-darwin"],
}

BACKEND_CONTEXTS = {
    ("qwen3.8-27b-4bit", "cuda-linux"): 16384,
    ("qwen3.8-27b-4bit", "llama-windows"): 16384,
    ("dots-ocr", "llama-windows"): 16384,
}


def test_this_build_ships_the_manifests_the_contracts_name() -> None:
    manifests = load_all_manifests()
    assert sorted(manifests) == sorted(SHIPPED + ALIASES)


def test_the_ids_sort_the_way_the_listing_shows_them() -> None:
    assert sorted(SHIPPED) == SHIPPED
    assert list(load_all_manifests()) == sorted(SHIPPED + ALIASES)


def test_an_id_that_is_a_prefix_of_another_still_lists_in_id_order(
    tmp_path: Path,
) -> None:
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
        assert spec.engine == engine_for(kind, spec.serves)
        assert len(spec.revision) == 40
        expected = BACKEND_CONTEXTS.get((model_id, kind), CONTEXTS[model_id])
        assert manifest.context_for(kind) == expected
        assert spec.memory_bytes_estimate > 0


def test_the_engine_table_is_one_per_backend_and_family() -> None:
    assert BACKEND_ENGINES == {
        "cuda-linux": {"text": "vllm", "pages": "vllm"},
        "mlx-darwin": {"text": "mlx-lm", "pages": "mlx-vlm"},
        "llama-windows": {"text": "llama-server", "pages": "llama-server"},
    }


def test_the_family_is_read_off_modalities_and_is_not_a_key() -> None:
    assert class_family(["text"]) == "text"
    assert class_family(["text", "image"]) == "pages"
    assert load_manifest("dots-ocr").modalities == ("text", "image")
    assert class_family(load_manifest("dots-ocr").modalities) == "pages"
    assert class_family(load_manifest("qwen3.5-9b").modalities) == "text"


def test_a_text_engine_named_by_a_page_model_is_refused() -> None:
    text = GOOD.replace('modalities = ["text"]', 'modalities = ["text", "image"]')
    text = text.replace("[backends.cuda-linux]", "[backends.mlx-darwin]")
    text = text.replace('engine = "vllm"', 'engine = "mlx-lm"')
    with pytest.raises(ManifestError) as caught:
        parse(text)
    message = str(caught.value)
    assert "does not serve 'pages' models on mlx-darwin" in message
    assert "that pairing's engine is 'mlx-vlm'" in message
    assert "read off [model] modalities" in message


def test_a_page_engine_named_by_a_text_model_is_refused_too() -> None:
    text = GOOD.replace("[backends.cuda-linux]", "[backends.mlx-darwin]")
    text = text.replace('engine = "vllm"', 'engine = "mlx-vlm"')
    with pytest.raises(ManifestError) as caught:
        parse(text)
    assert "does not serve 'text' models on mlx-darwin" in str(caught.value)
    assert "that pairing's engine is 'mlx-lm'" in str(caught.value)


def test_the_8bit_27b_is_not_offered_on_cuda_linux_or_windows() -> None:
    manifest = load_manifest("qwen3.8-27b-8bit")
    assert sorted(manifest.backends) == ["mlx-darwin"]
    assert not manifest.supports("cuda-linux")
    assert not manifest.supports("llama-windows")
    with pytest.raises(ManifestError) as caught:
        manifest.spec("cuda-linux")
    assert "has no cuda-linux block" in str(caught.value)


def test_the_9b_does_fit_a_24_gib_card() -> None:
    spec = load_manifest("qwen3.5-9b").spec("cuda-linux")
    assert spec.memory_bytes_estimate < 24 * 1024 ** 3


def test_the_4bit_27b_fits_a_24_gib_card_and_the_8bit_one_is_mac_only() -> None:
    small = load_manifest("qwen3.8-27b-4bit")
    big = load_manifest("qwen3.8-27b-8bit")
    assert small.family == big.family == "qwen3.8"
    assert small.params_b == big.params_b == 27
    assert small.spec("cuda-linux").memory_bytes_estimate < 24 * 1024 ** 3
    assert not big.supports("cuda-linux")
    assert big.supports("mlx-darwin")


def test_the_4bit_27b_does_not_force_a_dtype() -> None:
    args = load_manifest("qwen3.8-27b-4bit").spec("cuda-linux").engine_args
    assert "--dtype" not in args
    assert args == (
        "--gpu-memory-utilization", "0.86",
        "--max-num-seqs", "16",
        "--skip-mm-profiling",
        "--language-model-only",
    )


def test_the_text_models_are_served_language_model_only() -> None:
    for model_id in ("qwen3.5-9b", "qwen3.8-27b-4bit"):
        args = load_manifest(model_id).spec("cuda-linux").engine_args
        assert "--language-model-only" in args, model_id
        assert "--skip-mm-profiling" in args, model_id


def test_the_page_reader_is_never_language_model_only() -> None:
    manifest = load_manifest("dots-ocr")
    assert "image" in manifest.modalities
    assert "--language-model-only" not in manifest.spec("cuda-linux").engine_args


def test_the_manifests_directory_is_beside_the_package() -> None:
    assert manifests_dir().is_dir()
    assert (manifests_dir() / "qwen3.5-9b.toml").is_file()


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
    row = parse(OLLAMA).to_dict()
    assert row["display"] == "Demo 1B"
    assert row["description"] == "A fixture."
    assert "local" not in row
    unnamed = parse(GOOD).to_dict()
    assert unnamed["display"] is None and unnamed["description"] is None


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
        parse('local = "ollama"\n' + NAMED)
    assert "[local]: must be a table" in str(caught.value)


LOCAL_KINDS_SHIPPED = {
    "dots-ocr": "gguf",
    "qwen3.5-0.8b": "ollama",
    "qwen3.5-2b": "ollama",
    "qwen3.5-4b": "ollama",
    "qwen3.5-9b": "ollama",
    "qwen3.8-27b-4bit": "ollama",
}


def test_the_three_local_models_are_the_ones_foundry_runs() -> None:
    manifests = load_all_manifests()
    shipped = {k: v.local.kind for k, v in manifests.items() if v.local is not None}
    assert shipped == LOCAL_KINDS_SHIPPED
    assert manifests["qwen3.8-27b-8bit"].local is None


@pytest.mark.parametrize("model_id", sorted(LOCAL_KINDS_SHIPPED))
def test_each_shipped_local_form_is_declared_and_named(model_id: str) -> None:
    manifest = load_manifest(model_id)
    assert manifest.display and manifest.description
    local = manifest.local
    assert local is not None
    assert local.needs_basis == "declared"
    assert local.needs_bytes == local.download_bytes + 1_500_000_000


def test_the_cleanup_model_is_the_bf16_tag_the_clean_text_ruling_names() -> None:
    local = load_manifest("qwen3.5-9b").local
    assert isinstance(local, OllamaLocal)
    assert local.tag == "qwen3.5:9b-bf16"
    assert local.download_bytes == 19_321_189_044
    assert local.minimum_for == ("translate", "simplify")


def test_the_27b_no_longer_floors_anything() -> None:
    local = load_manifest("qwen3.8-27b-4bit").local
    assert isinstance(local, OllamaLocal)
    assert local.tag == "qwen3.8:27b"
    assert local.download_bytes == 17_741_872_172
    assert local.minimum_for == ()


def test_the_page_reader_is_a_gguf_pair_at_a_pinned_sha() -> None:
    local = load_manifest("dots-ocr").local
    assert isinstance(local, GgufLocal)
    assert local.hf_repo == "ggml-org/dots.ocr-GGUF"
    assert local.revision == "2c093a32ca360a396bc6d87d60408636130b9d9b"
    assert local.file == "dots.ocr-Q8_0.gguf"
    assert local.mmproj == "mmproj-dots.ocr-Q8_0.gguf"
    assert local.download_bytes == 1_894_530_272 + 1_344_068_512


def test_a_model_id_with_a_slash_is_refused_by_name() -> None:
    text = GOOD.replace('id = "demo-1b"', 'id = "vendor/demo-1b"')
    with pytest.raises(ManifestError, match="manifest_model_id_slash"):
        parse_manifest(text, Path("vendor-demo-1b.toml"), "vendor/demo-1b")
