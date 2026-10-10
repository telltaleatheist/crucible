from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.api.proxy import chat_limit_of, chat_queue_full
from crucible.engines import ENGINES, EngineError, chat_admission, engine_load_args
from crucible.inflight import RECENT_DURATIONS, InFlight


class _FakeResident:

    def __init__(
        self, model_id: str, engine: str, engine_args: tuple[str, ...] = ()
    ) -> None:
        self.model_id = model_id
        self.engine = engine
        self.engine_args = engine_args


# The attention layers of the Qwen3.5 / 3.8 weights, as their config.json states them
# (read on the Mac Studio 2026-10-10): every 4th layer is full attention.
QWEN_CONFIGS = {
    "qwen3.5-0.8b": (24, 8), "qwen3.5-2b": (24, 8), "qwen3.5-4b": (32, 16),
    "qwen3.5-9b": (32, 16), "qwen3.8-27b-8bit": (64, 24),
}


def _weights(model_id: str) -> Path:
    import tempfile

    layers, heads = QWEN_CONFIGS[model_id]
    here = Path(tempfile.mkdtemp())
    (here / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": layers, "num_attention_heads": heads, "head_dim": 256,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                        "full_attention"] * (layers // 4),
    }}))
    return here

def _mlx_args(model_id: str = "qwen3.5-9b") -> tuple[str, ...]:
    from crucible.manifests import load_manifest

    manifest = load_manifest(model_id)
    return tuple(
        engine_load_args(
            manifest, manifest.backends["mlx-darwin"], _weights(model_id), None,
            context=manifest.context_for("mlx-darwin"),
        )
    )


class _FakeResidency:
    def __init__(self, resident: Any) -> None:
        self.resident_model = resident


def test_mlx_lm_admits_its_batch_width_plus_one_waiting() -> None:
    limit, basis = chat_admission("mlx-lm", _mlx_args("qwen3.5-9b"))
    assert limit == 17
    assert basis is not None
    assert "BatchGenerator" in basis
    assert "started with --decode-concurrency 16" in basis


def test_the_27b_is_admitted_at_its_own_narrower_batch() -> None:
    limit, basis = chat_admission("mlx-lm", _mlx_args("qwen3.8-27b-8bit"))
    assert limit == 9
    assert basis is not None and "--decode-concurrency 8" in basis


def test_the_equals_spelling_is_read_and_the_last_one_wins() -> None:
    args = ("--decode-concurrency", "4", "--decode-concurrency=6")
    assert chat_admission("mlx-lm", args)[0] == 7


def test_an_mlx_lm_record_without_the_flag_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        chat_admission("mlx-lm", ("--prompt-concurrency", "4"))
    assert "started without --decode-concurrency" in str(caught.value)


def test_a_flag_engine_that_also_states_a_constant_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["mlx-lm"], "chat_concurrency", 1)
    with pytest.raises(EngineError) as caught:
        chat_admission("mlx-lm", _mlx_args())
    assert "One number, one owner" in str(caught.value)


def test_every_mlx_lm_block_states_the_flags_its_memory_argument_uses() -> None:
    from crucible.engines.base import int_flag
    from crucible.engines.mlx_lm import REQUIRED_FLAGS
    from crucible.manifests import load_all_manifests

    blocks = [
        (model_id, manifest.backends["mlx-darwin"])
        for model_id, manifest in load_all_manifests().items()
        if "mlx-darwin" in manifest.backends
        and manifest.backends["mlx-darwin"].engine == "mlx-lm"
    ]
    assert blocks, "no mlx-lm block to check; the test would prove nothing"
    for model_id, block in blocks:
        args = block.engine_args
        for flag in REQUIRED_FLAGS:
            assert int_flag(args, flag) is not None, f"{model_id}: no {flag}"
        assert int_flag(args, "--prompt-concurrency") <= int_flag(
            args, "--decode-concurrency"
        ), model_id


def test_an_engine_that_states_no_concurrency_is_not_bounded() -> None:
    assert chat_admission("mlx-vlm", ()) == (None, None)


def test_vllm_admits_its_max_num_seqs_plus_one() -> None:
    from crucible.manifests import load_manifest

    manifest = load_manifest("qwen3.5-9b")
    args = engine_load_args(
        manifest, manifest.backends["cuda-linux"], Path("/w"), None,
        context=manifest.context_for("cuda-linux"),
    )
    limit, basis = chat_admission("vllm", args)
    assert limit == 17
    assert basis is not None and "started with --max-num-seqs 16" in basis
    with pytest.raises(EngineError) as caught:
        chat_admission("vllm", ())
    assert "started without --max-num-seqs" in str(caught.value)


def test_every_vllm_block_states_its_batch() -> None:
    from crucible.engines.base import int_flag
    from crucible.manifests import load_all_manifests

    blocks = [
        (model_id, spec)
        for model_id, manifest in load_all_manifests().items()
        for spec in manifest.backends.values()
        if spec.engine == "vllm"
    ]
    assert blocks
    for model_id, spec in blocks:
        assert int_flag(spec.engine_args, "--max-num-seqs") is not None, model_id


def test_vllm_refuses_to_start_without_its_batch(tmp_path: Path) -> None:
    from crucible.engines.vllm import VllmEngine

    engine = VllmEngine(python=tmp_path / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "weights", "m", 0, ["--dtype", "bfloat16"])
    assert str(caught.value).startswith("vllm_flags_unstated:")
    assert not (tmp_path / "e.log").exists(), "nothing was spawned"


def test_an_unknown_engine_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        chat_admission("not-an-engine", ())
    assert "unknown engine" in str(caught.value)


def test_a_concurrency_with_no_basis_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["mlx-vlm"], "chat_concurrency", 4, raising=False)
    monkeypatch.setattr(ENGINES["mlx-vlm"], "chat_concurrency_basis", None, raising=False)
    with pytest.raises(EngineError) as caught:
        chat_admission("mlx-vlm", ())
    assert "no chat_concurrency_basis" in str(caught.value)


def test_a_basis_with_no_concurrency_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["mlx-vlm"], "chat_concurrency", None, raising=False)
    monkeypatch.setattr(
        ENGINES["mlx-vlm"], "chat_concurrency_basis", "measured somewhere", raising=False
    )
    with pytest.raises(EngineError) as caught:
        chat_admission("mlx-vlm", ())
    assert "no chat_concurrency" in str(caught.value)


def test_an_empty_card_reports_no_limit_rather_than_an_unlimited_one() -> None:
    assert chat_limit_of(_FakeResidency(None)) == (None, None)
    limit, basis = chat_limit_of(
        _FakeResidency(_FakeResident("q", "mlx-lm", _mlx_args()))
    )
    assert limit == 17
    assert basis is not None
    assert basis.startswith(ENGINES["mlx-lm"].chat_concurrency_basis)


def test_activity_publishes_the_resident_engines_own_width(
    make_client: Any, auth: dict[str, str]
) -> None:
    from crucible.residency import ResidentModel

    with make_client() as client:
        client.app.state.residency._resident = ResidentModel(
            model_id="qwen3.5-9b",
            backend="mlx-darwin",
            engine="mlx-lm",
            engine_model_name="/w",
            base_url="http://127.0.0.1:1",
            port=1,
            revision="0" * 40,
            max_model_len=16384,
            memory_bytes_estimate=1,
            log_path=Path("x.log"),
            loaded_at="2026-09-24T12:00:00+00:00",
            engine_args=_mlx_args("qwen3.5-9b"),
        )
        chat = client.get("/v1/activity", headers=auth).json()["chat"]
    assert chat["max_in_flight"] == 17
    assert "--decode-concurrency 16" in chat["max_in_flight_basis"]


def test_a_server_that_has_completed_nothing_states_no_wait() -> None:
    assert InFlight().retry_after() is None


def test_the_wait_is_the_median_of_recent_completions_not_the_mean() -> None:
    flight = InFlight()
    for seconds in [1.5, 1.5, 1.5, 1.5, 600.0]:
        entry = flight.open(act=None, model="q", client=None)
        object.__setattr__(entry, "started", entry.started - seconds)
        flight.close(entry)
    assert flight.retry_after() == 2


def test_the_wait_is_floored_at_one_second() -> None:
    flight = InFlight()
    flight.close(flight.open(act=None, model="q", client=None))
    assert flight.retry_after() == 1


def test_only_the_recent_completions_are_kept() -> None:
    flight = InFlight()
    for _ in range(RECENT_DURATIONS + 25):
        flight.close(flight.open(act=None, model="q", client=None))
    assert len(flight._recent) == RECENT_DURATIONS


def test_closing_twice_records_one_duration() -> None:
    flight = InFlight()
    entry = flight.open(act=None, model="q", client=None)
    flight.close(entry)
    flight.close(entry)
    assert len(flight._recent) == 1


def _body(response: Any) -> dict[str, Any]:
    return json.loads(bytes(response.body).decode("utf-8"))


def test_the_refusal_is_503_named_and_says_nothing_was_sent() -> None:
    response = chat_queue_full(
        resident=_FakeResident("qwen3.5-9b", "mlx-lm"),
        limit=2,
        basis="one generation thread",
        wait=12,
    )
    assert response.status_code == 503
    error = _body(response)["error"]
    assert error["code"] == "chat_queue_full"
    assert "cost nothing" in error["message"]
    assert error["details"]["max_in_flight"] == 2
    assert error["details"]["retry_after"] == 12


def test_the_refusal_carries_retry_after_as_a_header() -> None:
    response = chat_queue_full(
        resident=_FakeResident("qwen3.5-9b", "mlx-lm"),
        limit=2,
        basis="one generation thread",
        wait=12,
    )
    assert response.headers["retry-after"] == "12"


def test_an_unmeasured_wait_omits_the_header_rather_than_guessing_one() -> None:
    response = chat_queue_full(
        resident=_FakeResident("qwen3.5-9b", "mlx-lm"),
        limit=2,
        basis="one generation thread",
        wait=None,
    )
    assert "retry-after" not in response.headers
    assert _body(response)["error"]["details"]["retry_after"] is None


def test_llama_server_admits_its_one_slot_plus_one_waiting() -> None:
    limit, basis = chat_admission("llama-server", ("--parallel", "1"))
    assert limit == 2
    assert basis is not None and "--parallel 1" in basis


def test_every_llama_server_block_says_the_parallel_the_class_states() -> None:
    from crucible.manifests import load_all_manifests

    blocks = [
        (f"{model_id} [{kind}]", block)
        for model_id, manifest in load_all_manifests().items()
        for kind, block in manifest.backends.items()
        if block.engine == "llama-server"
    ]
    kinds = {name.split("[")[1] for name, _ in blocks}
    assert {"llama-windows]", "cuda-linux]"} <= kinds, (
        "a backend that runs llama-server has no block to check; the test would prove nothing"
    )
    stated = ENGINES["llama-server"].chat_concurrency
    for name, block in blocks:
        args = list(block.engine_args)
        assert "--parallel" in args, f"{name}: a llama-server block states no --parallel"
        assert args[args.index("--parallel") + 1] == str(stated), name


def test_each_engine_states_whether_it_serves_a_decision() -> None:
    from crucible.engines import decide_reading

    vllm = decide_reading("vllm")
    assert vllm.served and vllm.max_logprobs == 32
    llama = decide_reading("llama-server")
    assert llama.served and llama.max_logprobs is None
    assert "server-common.cpp" in llama.basis
    mlx = decide_reading("mlx-lm")
    assert mlx.served and mlx.max_logprobs == 40
    vlm = decide_reading("mlx-vlm")
    assert vlm.served and vlm.max_logprobs == 40
    assert "top_logprobs_k" in vlm.basis and "float32" in vlm.basis


def test_vllm_is_started_with_the_cap_the_reader_clamps_to() -> None:
    from crucible.engines.vllm import DECIDE_ARGS
    from crucible.manifests import load_manifest

    manifest = load_manifest("qwen3.5-9b")
    args = engine_load_args(
        manifest, manifest.backends["cuda-linux"], __import__("pathlib").Path("/w"), None,
        context=manifest.context_for("cuda-linux"),
    )
    assert args[args.index("--max-logprobs") + 1] == str(ENGINES["vllm"].max_logprobs)
    assert args[args.index("--logprobs-mode") + 1] == "raw_logprobs"
    assert "--enable-prompt-tokens-details" in args
    assert tuple(DECIDE_ARGS) == (
        "--max-logprobs", "32", "--logprobs-mode", "raw_logprobs",
        "--enable-prompt-tokens-details",
    )


def test_vllm_constrains_json_with_llguidance_not_xgrammar() -> None:
    """xgrammar's maxLength string admits no escapes, so a capped string can hold no
    newline (B-Side's lyrics, 2026-10-05); llguidance keeps them."""
    from crucible.manifests import load_manifest

    manifest = load_manifest("qwen3.5-4b")
    args = engine_load_args(
        manifest, manifest.backends["cuda-linux"], __import__("pathlib").Path("/w"), None,
        context=manifest.context_for("cuda-linux"),
    )
    config = args[args.index("--structured-outputs-config") + 1]
    assert __import__("json").loads(config) == {"backend": "guidance"}


def test_a_decide_reading_with_no_basis_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.engines import decide_reading

    monkeypatch.setattr(ENGINES["vllm"], "decide_basis", None)
    with pytest.raises(EngineError) as caught:
        decide_reading("vllm")
    assert "no decide_basis" in str(caught.value)


def test_a_cap_on_an_engine_that_serves_nothing_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.engines import decide_reading

    monkeypatch.setattr(ENGINES["mlx-vlm"], "decide_logprobs", False)
    monkeypatch.setattr(ENGINES["mlx-vlm"], "max_logprobs", 5)
    with pytest.raises(EngineError) as caught:
        decide_reading("mlx-vlm")
    assert "serves no decision" in str(caught.value)


def test_mlx_lm_reads_prompts_in_steps_derived_from_the_models_size() -> None:
    from crucible.engines.base import int_flag
    from crucible.engines.mlx_lm import (
        MLX_LM_PREFILL_STEP, PREFILL_STEP_FLAG, attention_flops_per_position, prefill_step,
    )

    # At their manifest contexts (27B 12288, 9B 16384): the step shrinks with the
    # attention a step at the deepest position pays.
    assert int_flag(_mlx_args("qwen3.8-27b-8bit"), PREFILL_STEP_FLAG) == 234
    assert int_flag(_mlx_args("qwen3.5-9b"), PREFILL_STEP_FLAG) == 686
    assert int_flag(_mlx_args("qwen3.5-0.8b"), PREFILL_STEP_FLAG) == MLX_LM_PREFILL_STEP
    big = attention_flops_per_position(_weights("qwen3.8-27b-8bit"))
    assert big == 4 * 16 * 24 * 256, "16 of 64 layers are full attention"
    assert prefill_step(27, big, 24576) == 217, "Content Studio's 24k context"
    assert prefill_step(27, big, 131072) < prefill_step(27, big, 24576)
    assert prefill_step(27, 0.0, 1) == 256, "the budget is the 27B at 256 tokens, shallow"
    with pytest.raises(EngineError):
        prefill_step(0, big, 8)


def test_a_manifest_that_states_the_prefill_step_is_refused() -> None:
    from types import SimpleNamespace

    manifest = SimpleNamespace(path=Path("m.toml"), params_b=9)
    with pytest.raises(EngineError) as caught:
        ENGINES["mlx-lm"].model_args(
            manifest, ["--prefill-step-size", "2048"], _weights("qwen3.5-9b"), 8192
        )
    assert str(caught.value).startswith("prefill_step_stated:")


def test_weights_without_a_readable_config_are_refused_by_name(tmp_path: Path) -> None:
    from types import SimpleNamespace

    manifest = SimpleNamespace(path=Path("m.toml"), params_b=9)
    with pytest.raises(EngineError) as caught:
        ENGINES["mlx-lm"].model_args(manifest, [], tmp_path, 8192)
    assert str(caught.value).startswith("model_config_unreadable:")
