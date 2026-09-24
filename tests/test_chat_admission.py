"""A serial engine's chat door refuses instead of holding the socket open.

Foundry's clean pass died on 2026-09-20 at block 352 of 940: 12 chat completions
in flight against the Mac's proxy, a 300 s client deadline, and one request that
had still not started when the deadline passed. Nothing errored — the server
accepted every connection, reported itself healthy, and generated in the order it
felt like.

The cause is that mlx-lm serves on a `ThreadingHTTPServer` and so ACCEPTS
concurrently, while generating on one thread draining one queue. The accepting is
what makes it dangerous: a serial engine that refused the twelfth connection
would have told the client the truth immediately.

So the door now bounds chats by the engine's OWN concurrency, publishes the
number, and states a `Retry-After` it can actually justify. These tests are about
all three, and about the two things the design refuses to do: invent a number for
an engine nobody measured, and invent a wait on a server that has completed
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.api import _chat_limit_of, _chat_queue_full
from crucible.engines import ENGINES, EngineError, chat_admission
from crucible.inflight import RECENT_DURATIONS, InFlight


class _FakeResident:
    """Only the fields the refusal and the admission read."""

    def __init__(
        self, model_id: str, engine: str, engine_args: tuple[str, ...] = ()
    ) -> None:
        self.model_id = model_id
        self.engine = engine
        self.engine_args = engine_args

#: The argv the Mac's 9B is started with, composed by the real code path from
#: the real manifest: what a resident mlx-lm engine's record carries.
def _mlx_args(model_id: str = "qwen3.5-9b") -> tuple[str, ...]:
    from crucible.manifests import load_manifest
    from crucible.residency import Residency

    manifest = load_manifest(model_id)
    return tuple(
        Residency._engine_args(
            manifest, manifest.backends["mlx-darwin"], Path("/w"), None,
            context=manifest.context_for("mlx-darwin"),
        )
    )


class _FakeResidency:
    def __init__(self, resident: Any) -> None:
        self.resident_model = resident


# --------------------------------------------------------------- the limit


def test_mlx_lm_admits_its_batch_width_plus_one_waiting() -> None:
    """CORRECTED 2026-09-24: mlx-lm BATCHES, `--decode-concurrency` wide.

    It said 1 (+1 = 2) on a reading that one generation thread meant serial
    generation; that thread runs a `BatchGenerator`, and the Mac cleanup ran
    two-wide against an engine that could run thirty-two. The width is now the
    flag on the argv the engine was started with, and the plus one is the
    request ready to join the batch the moment a slot frees.
    """
    limit, basis = chat_admission("mlx-lm", _mlx_args("qwen3.5-9b"))
    assert limit == 17
    assert basis is not None
    assert "BatchGenerator" in basis
    assert "started with --decode-concurrency 16" in basis


def test_the_27b_is_admitted_at_its_own_narrower_batch() -> None:
    """PER MODEL, because the KV each sequence holds is per model: the 8-bit
    27B affords 8 in flight inside the Mac's working set, the 9B 16
    (the arithmetic is in each manifest's mlx-darwin block)."""
    limit, basis = chat_admission("mlx-lm", _mlx_args("qwen3.8-27b-8bit"))
    assert limit == 9
    assert basis is not None and "--decode-concurrency 8" in basis


def test_the_equals_spelling_is_read_and_the_last_one_wins() -> None:
    """argparse's rules, so the door and the engine read the same number."""
    args = ("--decode-concurrency", "4", "--decode-concurrency=6")
    assert chat_admission("mlx-lm", args)[0] == 7


def test_an_mlx_lm_record_without_the_flag_is_refused_by_name() -> None:
    """Misconfiguration, not weather: `start()` refuses such an argv, so a
    record that lacks it does not describe what is running."""
    with pytest.raises(EngineError) as caught:
        chat_admission("mlx-lm", ("--prompt-concurrency", "4"))
    assert "started without --decode-concurrency" in str(caught.value)


def test_a_flag_engine_that_also_states_a_constant_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One number, one owner: the argv."""
    monkeypatch.setattr(ENGINES["mlx-lm"], "chat_concurrency", 1)
    with pytest.raises(EngineError) as caught:
        chat_admission("mlx-lm", _mlx_args())
    assert "One number, one owner" in str(caught.value)


def test_every_mlx_lm_block_states_the_flags_its_memory_argument_uses() -> None:
    """House rule: no library defaults. Each block states the batch width, the
    prefill width and the prompt-cache size, and the prefill width is never
    wider than the batch (mlx-lm holds `max(decode, prompt)` sequences)."""
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
    """vLLM BATCHES. Nothing has ever measured starvation against it, and a
    number invented here would cap work nobody showed needed capping."""
    assert chat_admission("vllm", ()) == (None, None)
    assert chat_admission("mlx-vlm", ()) == (None, None)


def test_an_unknown_engine_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        chat_admission("not-an-engine", ())
    assert "unknown engine" in str(caught.value)


def test_a_concurrency_with_no_basis_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE RULE THE WHOLE DESIGN RESTS ON: a number owes its provenance.

    The same sentence `crucible/voices.py` gives a `[voice.serving]` lever with
    no `_note`. A concurrency nobody can trace is a number somebody typed, and a
    reader of `/v1/activity` would have no way to tell it from a measurement.
    """
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency", 4, raising=False)
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency_basis", None, raising=False)
    with pytest.raises(EngineError) as caught:
        chat_admission("vllm", ())
    assert "no chat_concurrency_basis" in str(caught.value)


def test_a_basis_with_no_concurrency_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leftover half of a lever somebody removed."""
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency", None, raising=False)
    monkeypatch.setattr(
        ENGINES["vllm"], "chat_concurrency_basis", "measured somewhere", raising=False
    )
    with pytest.raises(EngineError) as caught:
        chat_admission("vllm", ())
    assert "no chat_concurrency" in str(caught.value)


def test_an_empty_card_reports_no_limit_rather_than_an_unlimited_one() -> None:
    """Null is not "unlimited": the limit belongs to the ENGINE, and with
    nothing loaded there is no engine to ask."""
    assert _chat_limit_of(_FakeResidency(None)) == (None, None)
    limit, basis = _chat_limit_of(
        _FakeResidency(_FakeResident("q", "mlx-lm", _mlx_args()))
    )
    assert limit == 17
    assert basis is not None
    assert basis.startswith(ENGINES["mlx-lm"].chat_concurrency_basis)


def test_activity_publishes_the_resident_engines_own_width(
    make_client: Any, auth: dict[str, str]
) -> None:
    """THE WIRE A CLIENT SIZES ITS POOL FROM. Foundry's placement reads
    `chat.max_in_flight` after its load and uses it as the pool, so the number
    on `/v1/activity` is what raising the batch actually reaches. A resident
    record exactly as `Residency.load` writes it for the Mac's 9B, read through
    the real route."""
    from crucible.residency import ResidentModel

    with make_client() as client:
        client.app.state.residency._resident = ResidentModel(  # noqa: SLF001
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


# ----------------------------------------------------------------- the wait


def test_a_server_that_has_completed_nothing_states_no_wait() -> None:
    """`_rate_limited`'s rule applied to a number of our own: a `Retry-After` is
    measured or it is absent, never invented."""
    assert InFlight().retry_after() is None


def test_the_wait_is_the_median_of_recent_completions_not_the_mean() -> None:
    """One 27B translation among short cleanups must not tell every refused
    caller to wait ten minutes."""
    flight = InFlight()
    for seconds in [2.0, 2.0, 2.0, 2.0, 600.0]:
        entry = flight.open(act=None, model="q", client=None)
        # `Entry` is frozen, which is why this backdates the start rather than
        # sleeping: the durations under test are 2 s and 600 s.
        object.__setattr__(entry, "started", entry.started - seconds)
        flight.close(entry)
    assert flight.retry_after() == 2


def test_the_wait_is_floored_at_one_second() -> None:
    """`Retry-After: 0` reads as "immediately" and turns a refusal into a spin."""
    flight = InFlight()
    flight.close(flight.open(act=None, model="q", client=None))
    assert flight.retry_after() == 1


def test_only_the_recent_completions_are_kept() -> None:
    """A long window would answer with a model that was unloaded an hour ago."""
    flight = InFlight()
    for _ in range(RECENT_DURATIONS + 25):
        flight.close(flight.open(act=None, model="q", client=None))
    assert len(flight._recent) == RECENT_DURATIONS  # noqa: SLF001


def test_closing_twice_records_one_duration() -> None:
    """`close` is idempotent, and the streamed door really does call it twice."""
    flight = InFlight()
    entry = flight.open(act=None, model="q", client=None)
    flight.close(entry)
    flight.close(entry)
    assert len(flight._recent) == 1  # noqa: SLF001


# -------------------------------------------------------------- the refusal


def _body(response: Any) -> dict[str, Any]:
    return json.loads(bytes(response.body).decode("utf-8"))


def test_the_refusal_is_503_named_and_says_nothing_was_sent() -> None:
    """A caller has to tell "the engine is busy" from "your request ran and
    failed" - the second is not safe to repeat and this one is."""
    response = _chat_queue_full(
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
    """A body a client has to parse is not what a retrying HTTP client reads."""
    response = _chat_queue_full(
        resident=_FakeResident("qwen3.5-9b", "mlx-lm"),
        limit=2,
        basis="one generation thread",
        wait=12,
    )
    assert response.headers["retry-after"] == "12"


def test_an_unmeasured_wait_omits_the_header_rather_than_guessing_one() -> None:
    """THE FALSIFIABLE HALF. A `Retry-After` this server cannot justify would be
    a number a client paces itself by, invented here."""
    response = _chat_queue_full(
        resident=_FakeResident("qwen3.5-9b", "mlx-lm"),
        limit=2,
        basis="one generation thread",
        wait=None,
    )
    assert "retry-after" not in response.headers
    assert _body(response)["error"]["details"]["retry_after"] is None


# ------------------------------------------------ llama-server (PHASE22 2.6)


def test_llama_server_admits_its_one_slot_plus_one_waiting() -> None:
    """`--parallel 1` is one slot: mlx-lm's shape on Windows, bounded the same
    way. Before PHASE22 this door admitted everything on llama-server, and a
    decision's fan-out is exactly the load that would have found it."""
    limit, basis = chat_admission("llama-server", ("--parallel", "1"))
    assert limit == 2
    assert basis is not None and "--parallel 1" in basis


def test_every_llama_windows_block_says_the_parallel_the_class_states() -> None:
    """ONE FACT, TWO OWNERS, AND THE CHECK THAT KEEPS THEM ONE.

    The slot count is set by each manifest's `engine_args`; the admission reads
    `LlamaServerEngine.chat_concurrency`. A manifest that ever says
    `--parallel 4` must fail here, not quietly run a door bounded at 2 in front
    of an engine with four slots.
    """
    from crucible.manifests import load_all_manifests

    blocks = [
        (model_id, manifest.backends["llama-windows"])
        for model_id, manifest in load_all_manifests().items()
        if "llama-windows" in manifest.backends
    ]
    assert blocks, "no llama-windows block to check; the test would prove nothing"
    stated = ENGINES["llama-server"].chat_concurrency
    for model_id, block in blocks:
        args = list(block.engine_args)
        assert "--parallel" in args, f"{model_id}: llama-windows states no --parallel"
        assert args[args.index("--parallel") + 1] == str(stated), model_id


# --------------------------------------------- what a decision may ask (2.4)


def test_each_engine_states_whether_it_serves_a_decision() -> None:
    """Read from each engine's source at the pinned version (PHASE22 section 1),
    and every one says where."""
    from crucible.engines import decide_reading

    vllm = decide_reading("vllm")
    assert vllm.served and vllm.max_logprobs == 32
    llama = decide_reading("llama-server")
    assert llama.served and llama.max_logprobs is None
    assert "server-common.cpp" in llama.basis
    mlx = decide_reading("mlx-lm")
    assert mlx.served and mlx.max_logprobs == 40  # the env patch (envpatches: 11 -> 40)
    pages = decide_reading("mlx-vlm")
    assert not pages.served and "compute_logprobs=False" in pages.basis


def test_vllm_is_started_with_the_cap_the_reader_clamps_to() -> None:
    """The flag and the reader's number are one constant, and the flags are
    the spellings vLLM 0.29.0 has (`vllm/engine/arg_utils.py` L934-935,
    `vllm/entrypoints/launchers/cli_args.py` L132)."""
    from crucible.engines.vllm import DECIDE_ARGS
    from crucible.manifests import load_manifest
    from crucible.residency import Residency

    manifest = load_manifest("qwen3.5-9b")
    args = Residency._engine_args(
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

    monkeypatch.setattr(ENGINES["mlx-vlm"], "max_logprobs", 5)
    with pytest.raises(EngineError) as caught:
        decide_reading("mlx-vlm")
    assert "serves no decision" in str(caught.value)
