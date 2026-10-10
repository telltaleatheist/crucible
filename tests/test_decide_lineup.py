from __future__ import annotations

from pathlib import Path

import pytest

from crucible import capabilityclasses, classnames
from crucible.backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from crucible.capabilityclasses import BY_NAME, CHAT_GOAL, DECIDE_GOAL, classes_for_model
from crucible.enginespec import UNSTATED_ENGINE_CONCURRENCY
from crucible.errors import ApiError
from crucible.inflight import ACT_NAMES, read_act
from crucible.manifests import ManifestError, load_manifest, parse_manifest
from crucible.verdict import decide

from .conftest import FAKE_BACKEND

GIB = 1024 ** 3

TEXT_CLASSES = ("clean", "translate", "simplify", "analysis")
TEXT_VERBS = ("clean", "translate", "simplify", "analysis", "generate", "decide")
SMALL_TIERS = ("qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b")
# CUDA also carries the 8-bit 4B, for an 8 GiB card (Victoria's 3070).
# The catalog lists by size, so the 16-bit 2B comes before it; the pick still takes the
# 4B first (params, then bits: docs/VERB-SIZING.md).
CUDA_SMALL_TIERS = ("qwen3.5-4b", "qwen3.5-2b", "qwen3.5-4b-8bit", "qwen3.5-0.8b")


def ids(name: str, backend: str) -> list[str]:
    return [c.id for c in BY_NAME[name].candidates(backend)]


def test_decide_is_a_class_an_act_and_not_routable() -> None:
    entry = BY_NAME["decide"]
    assert entry.job_type == "llm"
    assert entry.plainly == "decide"
    assert entry.routable is False
    assert "decide" not in classnames.ROUTABLE_CLASSES
    assert "decide" in classnames.SELECTABLE_CLASSES
    assert entry.goal is DECIDE_GOAL and entry.goal.params_b == 9
    assert "decide" in ACT_NAMES
    assert read_act({"X-Crucible-Act": "decide"}) == "decide"


def test_an_unknown_act_is_still_refused() -> None:
    with pytest.raises(ApiError) as caught:
        read_act({"X-Crucible-Act": "decides"})
    assert caught.value.code == "unknown_act"


def test_decide_work_cites_the_door_and_is_not_sixteen_states() -> None:
    work = BY_NAME["decide"].work
    assert work is not None
    assert work.tokens == capabilityclasses.DECIDE_STATE_TOKENS == 8192
    assert work.concurrency == 2
    assert f"{UNSTATED_ENGINE_CONCURRENCY} questions" in work.source


def test_every_text_verb_has_its_goal_and_no_floor() -> None:
    """docs/VERB-SIZING.md rule 2 and section 5: the 9B floor is gone; a goal caps the
    automatic pick instead."""
    goals = {name: BY_NAME[name].goal for name in TEXT_VERBS}
    assert {name: goal.params_b for name, goal in goals.items()} == {
        "clean": 9, "translate": 27, "simplify": 27, "analysis": 27, "generate": 27,
        "decide": 9,
    }
    for name in ("translate", "simplify", "analysis", "generate"):
        assert goals[name] is CHAT_GOAL, name
    assert not hasattr(BY_NAME["clean"], "min_params_b")


LINEUPS = {
    ("clean", CUDA_LINUX): ["qwen3.5-9b", *CUDA_SMALL_TIERS],
    ("clean", MLX_DARWIN): ["qwen3.5-9b", *SMALL_TIERS],
    ("clean", LLAMA_WINDOWS): ["qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b"],
    ("translate", CUDA_LINUX): ["qwen3.8-27b-4bit", "qwen3.5-9b", *CUDA_SMALL_TIERS],
    ("translate", MLX_DARWIN): ["qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b",
                                *SMALL_TIERS],
    ("translate", LLAMA_WINDOWS): ["qwen3.8-27b-4bit", "qwen3.5-9b", *SMALL_TIERS],
}


@pytest.mark.parametrize("backend", [CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS])
def test_the_text_classes_run_down_to_the_0_8b(backend: str) -> None:
    """Rule 1: every verb's lineup reaches the smallest text model, so a small card
    gets a smaller model, never "off"."""
    assert ids("clean", backend) == LINEUPS[("clean", backend)]
    for name in ("translate", "simplify", "analysis", "generate"):
        assert ids(name, backend) == LINEUPS[("translate", backend)], name


@pytest.mark.parametrize("backend", [CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS])
def test_decide_offers_every_tier_best_first(backend: str) -> None:
    expected = {
        CUDA_LINUX: ["qwen3.8-27b-4bit-vl", "qwen3.5-9b-vl", "qwen3.8-27b-4bit",
                     "qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3.5-4b-8bit",
                     "qwen3.5-0.8b"],
        MLX_DARWIN: ["qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b-vl",
                     "qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b"],
        LLAMA_WINDOWS: ["qwen3.8-27b-4bit-vl", "qwen3.8-27b-4bit",
                        "qwen3.5-9b-vl", "qwen3.5-9b", "qwen3.5-4b",
                        "qwen3.5-2b", "qwen3.5-0.8b"],
    }[backend]
    assert ids("decide", backend) == expected


@pytest.mark.parametrize("model_id", SMALL_TIERS)
def test_a_small_tier_serves_every_text_verb(model_id: str) -> None:
    assert set(classes_for_model(model_id)) == set(TEXT_VERBS)


def test_the_nine_b_and_the_27bs_keep_their_classes_and_gain_decide() -> None:
    for model_id in ("qwen3.5-9b", "qwen3.8-27b-4bit", "qwen3.8-27b-8bit"):
        classes = classes_for_model(model_id)
        assert "decide" in classes
        assert {"translate", "simplify", "analysis"} <= set(classes)
    assert "clean" in classes_for_model("qwen3.5-9b")


def test_the_3090ti_decides_on_the_9b_it_was_measured_on() -> None:
    verdict = decide(
        BY_NAME["decide"],
        CUDA_LINUX,
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia",
        chosen=None,
        audio_low_vram=False,
    )
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.5-9b"
    nine = next(c for c in verdict.candidates if c.id == "qwen3.5-9b")
    assert nine.need_bytes(BY_NAME["decide"].work) <= verdict.available_bytes


def test_a_six_gig_card_cannot_decide_even_on_the_0_8b() -> None:
    verdict = decide(
        BY_NAME["decide"], CUDA_LINUX, total_bytes=6 * GIB,
        desktop_allowance_bytes=3 * GIB, gpu_vendor="nvidia", chosen=None,
        audio_low_vram=False,
    )
    assert verdict.enabled is False
    assert "qwen3.5-0.8b" in verdict.reason


PINS = {
    "qwen3.5-4b": {
        CUDA_LINUX: ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        MLX_DARWIN: ("mlx-community/Qwen3.5-4B-bf16", "491fdc7c087ba7fb48adcb1253f8e76d011db783"),
        LLAMA_WINDOWS: ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523"),
    },
    "qwen3.5-2b": {
        CUDA_LINUX: ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
        MLX_DARWIN: ("mlx-community/Qwen3.5-2B-bf16", "fb270110eb5a9af244937040c9c3e57addab8ee9"),
        LLAMA_WINDOWS: ("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae"),
    },
    "qwen3.5-0.8b": {
        CUDA_LINUX: ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
        MLX_DARWIN: ("mlx-community/Qwen3.5-0.8B-bf16", "3067585164dbcc505eec73d554349de6a27571a4"),
        LLAMA_WINDOWS: ("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"),
    },
}


@pytest.mark.parametrize("model_id", SMALL_TIERS)
def test_the_small_tiers_are_pinned_to_the_official_repos(model_id: str) -> None:
    manifest = load_manifest(model_id)
    assert manifest.family == "qwen3.5"
    assert manifest.params_b < DECIDE_GOAL.params_b
    assert manifest.defaults.thinking is False
    for backend, (repo, revision) in PINS[model_id].items():
        spec = manifest.spec(backend)
        assert (spec.hf_repo, spec.revision) == (repo, revision)
    assert manifest.modalities == ("text", "image")
    assert manifest.serves(CUDA_LINUX) == ("text", "image")
    assert manifest.serves(LLAMA_WINDOWS) == ("text", "image")
    assert manifest.serves(MLX_DARWIN) == ("text",)
    assert manifest.spec(MLX_DARWIN).engine == "mlx-lm"
    assert manifest.spec(LLAMA_WINDOWS).mmproj == "mmproj-F16.gguf"
    args = manifest.spec(CUDA_LINUX).engine_args
    assert "--language-model-only" not in args and "--skip-mm-profiling" not in args


def test_the_0_8b_is_point_eight_and_not_rounded() -> None:
    assert load_manifest("qwen3.5-0.8b").params_b == 0.8


def test_the_2b_is_full_precision_on_every_backend() -> None:
    manifest = load_manifest("qwen3.5-2b")
    assert manifest.params_b == 2
    assert "--dtype" in (args := manifest.spec(CUDA_LINUX).engine_args)
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert manifest.spec(MLX_DARWIN).hf_repo.endswith("-bf16")
    assert manifest.spec(LLAMA_WINDOWS).file == "Qwen3.5-2B-BF16.gguf"


def test_the_9b_and_27bs_are_untouched_text_everywhere() -> None:
    for model_id in ("qwen3.5-9b", "qwen3.8-27b-4bit", "qwen3.8-27b-8bit"):
        manifest = load_manifest(model_id)
        assert manifest.modalities == ("text",)
        for kind in manifest.backends:
            assert manifest.serves(kind) == ("text",)


BASE = """
[model]
id = "demo"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = {modalities}

[backends.{backend}]
engine = "{engine}"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
{extra}
"""


def parse(
    *,
    modalities: str = '["text", "image"]',
    backend: str = "cuda-linux",
    engine: str = "vllm",
    extra: str = "",
):
    text = BASE.format(
        modalities=modalities, backend=backend, engine=engine, extra=extra
    )
    return parse_manifest(text, Path("demo.toml"), "demo")


def test_serves_defaults_to_the_models_modalities() -> None:
    assert parse().serves("cuda-linux") == ("text", "image")
    assert parse(modalities='["text"]').serves("cuda-linux") == ("text",)


def test_serves_beyond_the_weights_is_refused_by_name() -> None:
    with pytest.raises(ManifestError) as caught:
        parse(modalities='["text"]', extra='serves = ["text", "image"]')
    assert "serves_not_subset" in str(caught.value)


@pytest.mark.parametrize(
    "extra, words",
    [
        ("serves = []", "serves is empty"),
        ('serves = ["text", "text"]', "twice"),
        ('serves = ["audio"]', "'audio'"),
    ],
)
def test_serves_is_validated(extra: str, words: str) -> None:
    with pytest.raises(ManifestError) as caught:
        parse(extra=extra)
    assert words in str(caught.value)


def test_the_engine_is_chosen_from_what_the_block_serves() -> None:
    served_text = parse(backend="mlx-darwin", engine="mlx-lm", extra='serves = ["text"]')
    assert served_text.spec("mlx-darwin").engine == "mlx-lm"
    with pytest.raises(ManifestError) as caught:
        parse(backend="mlx-darwin", engine="mlx-lm")
    assert "'mlx-vlm'" in str(caught.value)


def test_the_image_flag_rules_read_the_served_set() -> None:
    parse(extra='serves = ["text"]\nengine_args = ["--language-model-only"]')
    with pytest.raises(ManifestError) as caught:
        parse(extra='engine_args = ["--language-model-only"]')
    assert "--language-model-only" in str(caught.value)


def test_a_projector_on_a_block_that_serves_no_images_is_refused() -> None:
    gguf = 'file = "demo.gguf"\nmmproj = "mmproj.gguf"\nserves = ["text"]'
    with pytest.raises(ManifestError) as caught:
        parse(backend="llama-windows", engine="llama-server", extra=gguf)
    assert "names a vision projector" in str(caught.value)
    with pytest.raises(ManifestError):
        parse(backend="llama-windows", engine="llama-server", extra='file = "demo.gguf"')


def test_params_b_is_a_number_and_not_a_bool() -> None:
    assert parse().params_b == 1
    with pytest.raises(ManifestError) as caught:
        parse_manifest(
            BASE.format(modalities='["text"]', backend="cuda-linux", engine="vllm",
                        extra="").replace("params_b = 1", "params_b = true"),
            Path("demo.toml"),
            "demo",
        )
    assert "params_b must be int or float, got bool" in str(caught.value)


def test_the_9b_vl_answers_images_on_the_mac_from_the_9b_s_own_folder() -> None:
    from crucible import decide as decide_core

    vl, text = load_manifest("qwen3.5-9b-vl"), load_manifest("qwen3.5-9b")
    spec, base = vl.spec(MLX_DARWIN), text.spec(MLX_DARWIN)
    assert spec.engine == "mlx-vlm" and base.engine == "mlx-lm"
    assert vl.serves(MLX_DARWIN) == ("text", "image")
    assert (spec.hf_repo, spec.revision) == (base.hf_repo, base.revision)
    assert spec.memory is not None and spec.memory.basis == "computed"
    assert spec.memory.weights_bytes > base.memory.weights_bytes
    assert "--width" in spec.engine_args
    decide_core.refuse_images_not_served("qwen3.5-9b-vl", vl, MLX_DARWIN, 8)


def test_a_mac_decision_with_images_on_a_text_model_names_the_9b_vl() -> None:
    from crucible import decide as decide_core
    from crucible.api.routes.decide import _image_models

    assert _image_models(MLX_DARWIN) == ["qwen3.5-9b-vl"]
    with pytest.raises(ApiError) as caught:
        decide_core.refuse_images_not_served(
            "qwen3.5-9b", load_manifest("qwen3.5-9b"), MLX_DARWIN, 3,
            lambda: _image_models(MLX_DARWIN),
        )
    assert caught.value.code == "model_text_only"
    assert caught.value.details["image_models"] == ["qwen3.5-9b-vl"]
    assert '{"type": "load-model", "model": "qwen3.5-9b-vl"}' in caught.value.message


def test_with_no_image_model_the_refusal_says_to_drop_the_images() -> None:
    from crucible import decide as decide_core

    with pytest.raises(ApiError) as caught:
        decide_core.refuse_images_not_served(
            "qwen3.5-9b", load_manifest("qwen3.5-9b"), MLX_DARWIN, 1
        )
    assert caught.value.details["image_models"] == []
    assert "without `images`" in caught.value.message


def test_the_mac_studio_decides_on_the_9b_not_the_27b() -> None:
    """It decided on the 27B at 8 bits until the goal; decide's goal is 9B, and the 9B's
    text form is taken before its vision alias (the same weights, more to hold)."""
    verdict = decide(
        BY_NAME["decide"], MLX_DARWIN, total_bytes=64 * GIB,
        desktop_allowance_bytes=3 * GIB, gpu_vendor="apple", chosen=None,
        audio_low_vram=False,
    )
    assert verdict.selected == "qwen3.5-9b"
    assert "goal 9B" in verdict.summary
