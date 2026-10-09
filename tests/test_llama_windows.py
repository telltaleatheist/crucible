from __future__ import annotations

from pathlib import Path

import pytest

from crucible import capabilitywords
from crucible.backend import (
    BACKEND_KINDS,
    CUDA_LINUX,
    LLAMA_WINDOWS,
    MLX_DARWIN,
    backend_not_here,
)
from crucible.capabilityclasses import BY_NAME
from crucible.config import WINDOWS_HOME_DIRNAME, crucible_home
from crucible.manifests import BACKEND_ENGINES, load_all_manifests, load_manifest
from crucible.verdict import CPU_POOL_NAME, decide, decide_all, pool_name

GIB = 1024 ** 3

WINDOWS_CARD = 24 * GIB
WINDOWS_RESERVE = 3 * GIB
LAPTOP_RAM = 32 * GIB


def _decide(name: str, total: int, reserve: int, vendor: str, chosen: str | None = None):
    return decide(
        BY_NAME[name],
        LLAMA_WINDOWS,
        total_bytes=total,
        desktop_allowance_bytes=reserve,
        gpu_vendor=vendor,
        chosen=chosen,
        audio_low_vram=False,
    )


def test_windows_is_a_backend_and_llama_server_is_its_engine() -> None:
    assert LLAMA_WINDOWS in BACKEND_KINDS
    assert BACKEND_ENGINES[LLAMA_WINDOWS] == {
        "text": "llama-server",
        "pages": "llama-server",
    }
    assert set(BACKEND_KINDS) == {CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS}


def test_a_backend_runs_where_its_engine_runs_and_nowhere_else() -> None:
    sentence = backend_not_here(CUDA_LINUX, LLAMA_WINDOWS, "windows")
    assert "cuda-linux" in sentence and "llama-windows" in sentence
    assert "A backend runs where its engine runs" in sentence
    other = backend_not_here(LLAMA_WINDOWS, MLX_DARWIN, "darwin")
    assert "llama-windows" in other and "mlx-darwin" in other


def test_detect_windows_answers_with_a_card_or_with_the_machines_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible import backend as backend_module
    from crucible.errors import NoViableBackend

    monkeypatch.setattr(
        backend_module, "probe_nvidia_smi", lambda: ("NVIDIA GeForce RTX 3090 Ti", WINDOWS_CARD)
    )
    with_card = backend_module.detect_windows("AMD64")
    assert with_card.kind == LLAMA_WINDOWS
    assert with_card.platform == "windows"
    assert with_card.gpu.vendor == "nvidia"
    assert with_card.gpu.vram_bytes == WINDOWS_CARD
    assert "cuda build" in with_card.detail

    def no_card() -> tuple[str, int]:
        raise NoViableBackend("nvidia-smi is not on PATH")

    monkeypatch.setattr(backend_module, "probe_nvidia_smi", no_card)
    monkeypatch.setattr(backend_module, "physical_memory_bytes", lambda: LAPTOP_RAM)
    without = backend_module.detect_windows("AMD64")
    assert without.kind == LLAMA_WINDOWS
    assert without.gpu.vendor == "cpu"
    assert without.gpu.vram_bytes == LAPTOP_RAM
    assert "cpu build" in without.detail


def test_the_windows_home_is_localappdata(monkeypatch: pytest.MonkeyPatch) -> None:
    import crucible.config as config_module

    monkeypatch.delenv("CRUCIBLE_HOME", raising=False)
    monkeypatch.setattr(config_module.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(Path("/tmp/AppData/Local")))
    assert crucible_home() == Path("/tmp/AppData/Local") / WINDOWS_HOME_DIRNAME
    monkeypatch.setenv("CRUCIBLE_HOME", "/tmp/elsewhere")
    assert crucible_home() == Path("/tmp/elsewhere")


def test_a_windows_session_with_no_localappdata_is_refused_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.config as config_module
    from crucible.errors import ConfigError

    monkeypatch.delenv("CRUCIBLE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(config_module.sys, "platform", "win32")
    with pytest.raises(ConfigError, match="LOCALAPPDATA"):
        crucible_home()


def test_the_three_published_ggufs_are_named_and_the_fourth_is_absent() -> None:
    catalog = load_all_manifests()
    served = {
        model_id
        for model_id, manifest in catalog.items()
        if manifest.supports(LLAMA_WINDOWS)
    }
    assert served == {
        "dots-ocr", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
        "qwen3.8-27b-4bit",
        "qwen3.5-9b-vl", "qwen3.8-27b-4bit-vl",
    }


def test_every_llama_windows_row_names_the_one_file_it_is() -> None:
    for model_id in (
        "dots-ocr", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
        "qwen3.8-27b-4bit",
    ):
        spec = load_manifest(model_id).spec(LLAMA_WINDOWS)
        assert spec.engine == "llama-server"
        assert spec.file is not None and spec.file.endswith(".gguf"), model_id
        assert spec.memory_bytes_estimate > 0


def test_the_page_reader_names_its_projector_and_the_text_models_do_not() -> None:
    dots = load_manifest("dots-ocr").spec(LLAMA_WINDOWS)
    assert dots.mmproj == "mmproj-dots.ocr-Q8_0.gguf"
    assert dots.files == ("dots.ocr-Q8_0.gguf", "mmproj-dots.ocr-Q8_0.gguf")
    for model_id in ("qwen3.5-9b", "qwen3.8-27b-4bit"):
        spec = load_manifest(model_id).spec(LLAMA_WINDOWS)
        assert spec.mmproj is None, model_id
        assert len(spec.files) == 1, model_id


def test_the_page_reader_pins_the_same_pair_the_local_form_names() -> None:
    manifest = load_manifest("dots-ocr")
    spec = manifest.spec(LLAMA_WINDOWS)
    local = manifest.local
    assert local is not None
    assert spec.hf_repo == local.hf_repo
    assert spec.revision == local.revision
    assert spec.file == local.file
    assert spec.mmproj == local.mmproj


def test_the_projector_comes_from_the_same_project_as_the_binary() -> None:
    spec = load_manifest("dots-ocr").spec(LLAMA_WINDOWS)
    assert spec.hf_repo.split("/")[0] == "ggml-org", (
        f"the page reader's GGUFs come from {spec.hf_repo}, but the engine is "
        "ggml-org/llama.cpp; a third party's conversion may not carry the keys "
        "this build requires"
    )


def test_a_file_on_a_backend_that_pulls_a_whole_repo_is_refused() -> None:
    from crucible.manifests import ManifestError, parse_manifest

    from .test_manifests import GOOD

    text = GOOD.replace(
        "memory_bytes_estimate = 3000000000",
        'memory_bytes_estimate = 3000000000\nfile = "x.gguf"',
    )
    with pytest.raises(ManifestError, match="llama-windows block"):
        parse_manifest(text, Path("demo-1b.toml"), "demo-1b")


def test_a_llama_windows_block_with_no_file_is_refused() -> None:
    from crucible.manifests import ManifestError, parse_manifest

    text = """
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text"]

[backends.llama-windows]
engine = "llama-server"
hf_repo = "demo/Demo-1B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
"""
    with pytest.raises(ManifestError, match="needs `file`"):
        parse_manifest(text, Path("demo-1b.toml"), "demo-1b")


def test_an_image_model_with_no_mmproj_is_refused() -> None:
    from crucible.manifests import ManifestError, parse_manifest

    text = """
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text", "image"]

[backends.llama-windows]
engine = "llama-server"
hf_repo = "demo/Demo-1B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "demo.gguf"
memory_bytes_estimate = 3000000000
"""
    with pytest.raises(ManifestError, match="mmproj"):
        parse_manifest(text, Path("demo-1b.toml"), "demo-1b")


def test_the_five_python_job_types_answer_one_sentence(
) -> None:
    reasons = set()
    for name in ("tts", "asr", "align", "rvc", "denoise"):
        verdict = _decide(name, WINDOWS_CARD, WINDOWS_RESERVE, "nvidia")
        assert verdict.enabled is False, name
        assert verdict.selected == "", name
        assert verdict.shortfall_bytes == 0, name
        reasons.add(verdict.reason)
    assert reasons == {capabilitywords.NEEDS_WSL_REASON}
    assert "WSL2" in capabilitywords.NEEDS_WSL_REASON


def test_the_llm_classes_and_pages_answer_from_the_gguf_table() -> None:
    verdicts = {
        name: _decide(name, WINDOWS_CARD, WINDOWS_RESERVE, "nvidia")
        for name in ("clean", "translate", "simplify", "analysis", "pages")
    }
    assert verdicts["clean"].selected == "qwen3.5-9b"
    assert verdicts["translate"].selected == "qwen3.8-27b-4bit"
    assert verdicts["pages"].selected == "dots-ocr"
    for name, verdict in verdicts.items():
        assert verdict.enabled is True, name
        assert "card" in verdict.reason, name
        assert "cpu build" not in verdict.reason, name


def test_a_windows_box_with_no_card_still_serves_and_says_it_is_slow() -> None:
    verdict = _decide("clean", LAPTOP_RAM, WINDOWS_RESERVE, "cpu")
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.5-9b"
    assert capabilitywords.CPU_BUILD_REASON in verdict.reason
    assert CPU_POOL_NAME in verdict.reason
    assert "card" not in verdict.reason


def test_echo_is_on_and_needs_neither_a_card_nor_wsl() -> None:
    verdict = _decide("echo", 1, 0, "cpu")
    assert verdict.enabled is True


def test_the_whole_record_reads_as_three_kinds_of_answer() -> None:
    decisions = decide_all(
        LLAMA_WINDOWS,
        total_bytes=WINDOWS_CARD,
        desktop_allowance_bytes=WINDOWS_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
        audio_low_vram=False,
    )
    by_name = {d.capability: d for d in decisions}
    assert by_name["echo"].enabled is True
    assert by_name["clean"].enabled is True
    assert by_name["pages"].enabled is True
    assert by_name["tts"].enabled is False
    assert by_name["tts"].reason == capabilitywords.NEEDS_WSL_REASON


def test_a_small_card_turns_the_27b_off_with_the_number(
) -> None:
    verdict = _decide("translate", 12 * GIB, WINDOWS_RESERVE, "nvidia")
    assert verdict.enabled is False
    assert verdict.shortfall_bytes > 0
    assert "short by" in verdict.reason


def test_the_pool_is_named_for_what_it_actually_is() -> None:
    assert pool_name(LLAMA_WINDOWS, "nvidia") == "card"
    assert pool_name(LLAMA_WINDOWS, "cpu") == CPU_POOL_NAME
    assert pool_name(MLX_DARWIN, "apple") == "unified memory"
    with pytest.raises(ValueError, match="not a Crucible backend"):
        pool_name("cuda-windows", "nvidia")
