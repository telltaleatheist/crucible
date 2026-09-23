"""The decision lineup (PHASE22-DECIDE.md section 2.9).

Three things were built together and are held here:

1. A `decide` capability class, with every qwen3.8 / qwen3.5 manifest as a
   candidate and no size floor, and an EXPLICIT 9B floor on the four text classes
   that used to get theirs by accident of what was in `models/`.
2. `qwen3.5-4b` and `qwen3.5-0.8b`, pinned to the official repos.
3. `serves`: what a BACKEND serves, separate from what the weights accept.

The candidate lists of the four existing text classes are asserted EXACTLY, per
backend, as they stood before the small tiers landed: adding a 4B must not move
a floor that Owen ruled (docs/MODEL-CHOICE.md section 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible import capability
from crucible.backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from crucible.capability import BY_NAME, NINE_B_FLOOR, classes_for_model, decide
from crucible.decide import UNSTATED_ENGINE_CONCURRENCY
from crucible.errors import ApiError
from crucible.inflight import ACT_NAMES, read_act
from crucible.manifests import ManifestError, load_manifest, parse_manifest

from .conftest import FAKE_BACKEND

GIB = 1024 ** 3

TEXT_CLASSES = ("clean", "translate", "simplify", "analysis")
SMALL_TIERS = ("qwen3.5-4b", "qwen3.5-0.8b")


def ids(name: str, backend: str) -> list[str]:
    return [c.id for c in BY_NAME[name].candidates(backend)]


# ------------------------------------------------------------- the class


def test_decide_is_a_class_an_act_and_not_routable() -> None:
    entry = BY_NAME["decide"]
    assert entry.job_type == "llm"
    assert entry.plainly == "decide"
    # The door refuses every upstream id (`decide_needs_logprobs`), so a route
    # would be a class that refuses all of its own work.
    assert entry.routable is False
    assert "decide" not in capability.ROUTABLE_CLASSES
    assert "decide" in capability.SELECTABLE_CLASSES
    assert entry.min_params_b is None
    assert "decide" in ACT_NAMES
    assert read_act({"X-Crucible-Act": "decide"}) == "decide"


def test_an_unknown_act_is_still_refused() -> None:
    with pytest.raises(ApiError) as caught:
        read_act({"X-Crucible-Act": "decides"})
    assert caught.value.code == "unknown_act"


def test_decide_work_cites_the_door_and_is_not_sixteen_states() -> None:
    work = BY_NAME["decide"].work
    assert work is not None
    assert work.tokens == capability.DECIDE_STATE_TOKENS == 8192
    assert work.concurrency == 2
    assert f"{UNSTATED_ENGINE_CONCURRENCY} questions" in work.source


def test_the_nine_b_floor_is_explicit_on_the_four_text_classes() -> None:
    for name in TEXT_CLASSES:
        assert BY_NAME[name].min_params_b == NINE_B_FLOOR == 9, name


# ---------------------------------------------- exact lists, before and after


BEFORE = {
    # What each text class offered on 2026-09-23 before the 4B and 0.8B landed.
    # They must not move.
    ("clean", CUDA_LINUX): ["qwen3.5-9b"],
    ("clean", MLX_DARWIN): ["qwen3.5-9b"],
    ("clean", LLAMA_WINDOWS): ["qwen3.5-9b"],
    ("translate", CUDA_LINUX): ["qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b"],
    ("translate", MLX_DARWIN): ["qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b"],
    ("translate", LLAMA_WINDOWS): ["qwen3.8-27b-4bit", "qwen3.5-9b"],
}


@pytest.mark.parametrize("backend", [CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS])
def test_the_text_classes_offer_exactly_what_they_did(backend: str) -> None:
    assert ids("clean", backend) == BEFORE[("clean", backend)]
    for name in ("translate", "simplify", "analysis"):
        assert ids(name, backend) == BEFORE[("translate", backend)], name


@pytest.mark.parametrize("backend", [CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS])
def test_decide_offers_every_tier_best_first(backend: str) -> None:
    # The `-vl` aliases (tests/test_weights_of.py) are in this list and in no
    # text class's: same weights, served with the tower, dearer than the base.
    expected = {
        CUDA_LINUX: ["qwen3.8-27b-8bit-vl", "qwen3.8-27b-8bit",
                     "qwen3.8-27b-4bit-vl", "qwen3.5-9b-vl", "qwen3.8-27b-4bit",
                     "qwen3.5-9b", "qwen3.5-4b", "qwen3.5-0.8b"],
        MLX_DARWIN: ["qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b",
                     "qwen3.5-4b", "qwen3.5-0.8b"],
        LLAMA_WINDOWS: ["qwen3.8-27b-4bit-vl", "qwen3.8-27b-4bit",
                        "qwen3.5-9b-vl", "qwen3.5-9b", "qwen3.5-4b",
                        "qwen3.5-0.8b"],
    }[backend]
    assert ids("decide", backend) == expected


@pytest.mark.parametrize("model_id", SMALL_TIERS)
def test_a_small_tier_serves_decide_and_nothing_else(model_id: str) -> None:
    assert classes_for_model(model_id) == ("decide",)


def test_the_nine_b_and_the_27bs_keep_their_classes_and_gain_decide() -> None:
    for model_id in ("qwen3.5-9b", "qwen3.8-27b-4bit", "qwen3.8-27b-8bit"):
        classes = classes_for_model(model_id)
        assert "decide" in classes
        assert {"translate", "simplify", "analysis"} <= set(classes)
    assert "clean" in classes_for_model("qwen3.5-9b")


def test_a_floor_is_a_comparison_not_a_family(tmp_path: Path) -> None:
    """The mutation this pins: take the floor off and the 4B walks into clean."""
    source = BY_NAME["clean"].candidates
    unfloored = capability.CatalogCandidates(source.load, source.families, None)
    assert "qwen3.5-4b" in [c.id for c in unfloored(CUDA_LINUX)]
    assert "qwen3.5-4b" not in ids("clean", CUDA_LINUX)


# -------------------------------------------------------- the known-good fit


def test_the_3090ti_decides_on_the_9b_it_was_measured_on() -> None:
    """snap measured decisions on the 9B on this card (PHASE22 section 0). With
    the 27Bs refused for this work, the 9B is the best candidate that fits —
    and it is refused if the work is sized as sixteen independent states."""
    verdict = decide(
        BY_NAME["decide"],
        CUDA_LINUX,
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia",
        chosen=None,
    )
    assert verdict.enabled is True
    assert verdict.selected in ("qwen3.8-27b-4bit", "qwen3.5-9b")
    nine = next(c for c in verdict.candidates if c.id == "qwen3.5-9b")
    assert nine.need_bytes(BY_NAME["decide"].work) <= verdict.available_bytes


def test_a_six_gig_card_cannot_decide_even_on_the_0_8b() -> None:
    """The 0.8B's cuda-linux block carries a 1.90 GiB image reserve borrowed
    from the 9B as an upper bound; with it, 3 GiB of budget is not enough —
    which is the direction a carried upper bound is allowed to err."""
    verdict = decide(
        BY_NAME["decide"], CUDA_LINUX, total_bytes=6 * GIB,
        desktop_allowance_bytes=3 * GIB, gpu_vendor="nvidia", chosen=None,
    )
    assert verdict.enabled is False
    assert "qwen3.5-0.8b" in verdict.reason


# ------------------------------------------------------------ the manifests


PINS = {
    "qwen3.5-4b": {
        CUDA_LINUX: ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        MLX_DARWIN: ("mlx-community/Qwen3.5-4B-bf16", "491fdc7c087ba7fb48adcb1253f8e76d011db783"),
        LLAMA_WINDOWS: ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523"),
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
    assert manifest.params_b < NINE_B_FLOOR
    assert manifest.defaults.thinking is False
    for backend, (repo, revision) in PINS[model_id].items():
        spec = manifest.spec(backend)
        assert (spec.hf_repo, spec.revision) == (repo, revision)
    # The engine that runs it is the one that can see; the Mac's cannot.
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


def test_the_9b_and_27bs_are_untouched_text_everywhere() -> None:
    for model_id in ("qwen3.5-9b", "qwen3.8-27b-4bit", "qwen3.8-27b-8bit"):
        manifest = load_manifest(model_id)
        assert manifest.modalities == ("text",)
        for kind in manifest.backends:
            assert manifest.serves(kind) == ("text",)


# ------------------------------------------------------------------- serves


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
    """On the Mac a text+image model served text is mlx-lm's, and the same
    model served images there would be mlx-vlm's (the page server)."""
    served_text = parse(backend="mlx-darwin", engine="mlx-lm", extra='serves = ["text"]')
    assert served_text.spec("mlx-darwin").engine == "mlx-lm"
    with pytest.raises(ManifestError) as caught:
        parse(backend="mlx-darwin", engine="mlx-lm")
    assert "'mlx-vlm'" in str(caught.value)


def test_the_image_flag_rules_read_the_served_set() -> None:
    # Text-served: `--language-model-only` is allowed on image-capable weights.
    parse(extra='serves = ["text"]\nengine_args = ["--language-model-only"]')
    # Image-served: the same flag is refused.
    with pytest.raises(ManifestError) as caught:
        parse(extra='engine_args = ["--language-model-only"]')
    assert "--language-model-only" in str(caught.value)


def test_a_projector_on_a_block_that_serves_no_images_is_refused() -> None:
    gguf = 'file = "demo.gguf"\nmmproj = "mmproj.gguf"\nserves = ["text"]'
    with pytest.raises(ManifestError) as caught:
        parse(backend="llama-windows", engine="llama-server", extra=gguf)
    assert "names a vision projector" in str(caught.value)
    # And the block that DOES serve images still needs it.
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
