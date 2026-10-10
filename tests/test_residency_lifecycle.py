from __future__ import annotations

import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import residency as residency_module
from crucible.cardkinds import KIND_ALIGN, KIND_LLM, KIND_TTS
from crucible.engines import (
    ENGINES,
    EngineError,
    LlamaServerEngine,
    MlxLmEngine,
    MlxVlmEngine,
    VllmEngine,
    engine_load_args,
    engine_model_name,
)
from crucible.engines.vllm import DECIDE_ARGS, STRUCTURED_OUTPUTS_ARGS
from crucible.errors import JobError
from crucible.manifests import NO_DEFAULTS
from crucible.residency import (
    DyingResident,
    Occupant,
    Residency,
    ResidentAligner,
    ResidentModel,
    ResidentSeparator,
    ResidentVoice,
)

from .test_residency import holder

__all__ = ["holder"]

LOG = Path("/tmp/engine-x.log")
AT = "2026-09-27T00:00:00+00:00"


class Stoppable:
    def __init__(self, pid: int = 4242) -> None:
        self.stops = 0
        self._pid = pid

    @property
    def pids(self) -> frozenset[int]:
        return frozenset({self._pid})

    def stop(self) -> None:
        self.stops += 1


def a_voice(voice_id: str = "sigma") -> ResidentVoice:
    return ResidentVoice(
        voice_id=voice_id,
        backend="cuda-linux",
        narrator_engine="higgs-v3",
        revision="r",
        fingerprint="f",
        sample_rate=24_000,
        max_chars=None,
        memory_bytes_estimate=1,
        log_path=LOG,
        loaded_at=AT,
    )


def a_model(model_id: str = "m") -> ResidentModel:
    return ResidentModel(
        model_id=model_id,
        backend="cuda-linux",
        engine="vllm",
        engine_model_name=model_id,
        base_url="http://127.0.0.1:9",
        port=9,
        revision="r",
        max_model_len=4096,
        memory_bytes_estimate=2,
        log_path=LOG,
        loaded_at=AT,
        engine_args=("--x",),
        defaults=NO_DEFAULTS,
    )


def test_the_wire_shapes_are_the_golden_ones() -> None:
    shapes = {
        "model": a_model().to_dict(),
        "voice": a_voice().to_dict(),
        "aligner": ResidentAligner(
            aligner_id="a", backend="cuda-linux", revision="r", fingerprint="f",
            device="cuda", dtype="bfloat16", max_audio_s=300.0,
            memory_bytes_estimate=3, log_path=LOG, loaded_at=AT,
        ).to_dict(),
        "separator": ResidentSeparator(
            separator_id="s", backend="cuda-linux", model_filename="x.ckpt",
            revision="r", fingerprint="f", sample_rate=44_100, use_autocast=True,
            memory_bytes_estimate=4, log_path=LOG, loaded_at=AT,
        ).to_dict(),
        "dying": DyingResident(
            subject_id="m", kind=KIND_LLM, engine=None, session=None,
            pids=frozenset({3, 1}), since=AT, log_path=LOG,
        ).to_dict(),
    }
    log = json.dumps(str(LOG))
    assert json.dumps(shapes, sort_keys=False) == (
        '{"model": {"model": "m", "backend": "cuda-linux", "engine": "vllm", '
        '"engine_model_name": "m", "base_url": "http://127.0.0.1:9", '
        '"revision": "r", "fingerprint": '
        + json.dumps(a_model().fingerprint)
        + ', "max_model_len": 4096, "defaults": '
        + json.dumps(NO_DEFAULTS.to_dict())
        + ', "memory_bytes_estimate": 2, "log_path": '
        + log
        + ', "loaded_at": "' + AT + '", "form": null}, '
        '"voice": {"voice": "sigma", "backend": "cuda-linux", '
        '"narrator_engine": "higgs-v3", "revision": "r", "fingerprint": "f", '
        '"sample_rate": 24000, "max_chars": null, "memory_bytes_estimate": 1, '
        '"log_path": ' + log + ', "loaded_at": "' + AT + '", "reference": null}, '
        '"aligner": {"aligner": "a", "backend": "cuda-linux", "revision": "r", '
        '"fingerprint": "f", "device": "cuda", "dtype": "bfloat16", '
        '"max_audio_s": 300.0, "memory_bytes_estimate": 3, "log_path": '
        + log + ', "loaded_at": "' + AT + '"}, '
        '"separator": {"separator": "s", "backend": "cuda-linux", '
        '"model_filename": "x.ckpt", "revision": "r", "fingerprint": "f", '
        '"sample_rate": 44100, "use_autocast": true, "memory_bytes_estimate": 4, '
        '"log_path": ' + log + ', "loaded_at": "' + AT + '"}, '
        '"dying": {"kind": "llm", "id": "m", "since": "' + AT + '", "pids": [1, 3]}}'
    )


def test_occupy_publishes_what_start_returned(holder: Residency) -> None:
    engine = Stoppable()
    seen: list[str | None] = []
    said: list[str] = []

    def start() -> Occupant:
        seen.append(holder.warming)
        return Occupant(a_model(), engine=engine, base_url="http://127.0.0.1:9")

    resident = holder.occupy(KIND_LLM, "m", start, say=said.append)
    assert resident == a_model()
    assert holder.resident == resident
    assert seen == ["m"]
    assert holder.warming is None
    assert holder.owned_pids() == frozenset({4242})
    assert said[-1] == "m is resident at http://127.0.0.1:9"
    holder.unload("m")
    assert engine.stops == 1


def test_occupy_evicts_the_resident_before_starting(holder: Residency) -> None:
    first = Stoppable(1)
    holder.occupy(KIND_TTS, "sigma", lambda: Occupant(a_voice(), engine=first), say=print)
    order: list[str] = []

    def start() -> Occupant:
        order.append(f"start after {first.stops} stop(s)")
        return Occupant(a_model(), engine=Stoppable(2))

    said: list[str] = []
    holder.occupy(KIND_LLM, "m", start, say=said.append)
    assert order == ["start after 1 stop(s)"]
    assert said[0].startswith("unloading sigma")
    assert said[-1] == "m is resident"


def test_a_start_that_fails_leaves_the_card_empty_and_not_warming(
    holder: Residency,
) -> None:
    def start() -> Occupant:
        raise EngineError("would not come up")

    with pytest.raises(EngineError, match="would not come up"):
        holder.occupy(KIND_ALIGN, "a", start, say=print)
    assert holder.resident is None
    assert holder.warming is None


def test_an_occupant_that_is_not_what_was_asked_for_is_stopped(
    holder: Residency,
) -> None:
    engine = Stoppable()
    with pytest.raises(EngineError, match="instead; it was stopped"):
        holder.occupy(
            KIND_TTS, "other", lambda: Occupant(a_voice(), engine=engine), say=print
        )
    assert engine.stops == 1
    assert holder.resident is None


def test_occupy_refuses_while_another_thread_holds_the_card(
    holder: Residency,
) -> None:
    ready = threading.Event()
    done = threading.Event()

    def hold() -> None:
        with holder.claimed("stream", may_mutate=True):
            ready.set()
            done.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    ready.wait(5)
    try:
        with pytest.raises(JobError) as refusal:
            holder.occupy(KIND_LLM, "m", lambda: Occupant(a_model()), say=print)
        assert refusal.value.code == "engine_in_use"
    finally:
        done.set()
        thread.join(5)


def test_residency_knows_no_engine_and_no_job() -> None:
    source = Path(residency_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {
        ".jobs", ".engines.vllm", ".engines.llama_server", ".narratorvoices",
        ".ttsplan", ".voices", ".alignmodels", ".denoisemodels", ".jobenv",
    }
    assert not [name for name in imported if name.lstrip(".").startswith("jobs")]
    assert not imported & forbidden
    for name in ("load", "load_aligner", "load_separator", "load_voice", "_engine_args", "_start"):
        assert not hasattr(Residency, name)


def test_every_engine_is_reached_through_the_table() -> None:
    assert set(ENGINES) == {"vllm", "mlx-lm", "mlx-vlm", "llama-server"}
    here = Path("/weights/m")
    assert engine_model_name("vllm", here, "m") == "m"
    assert engine_model_name("llama-server", here, "m") == "m"
    assert engine_model_name("mlx-vlm", here, "m") == str(here)
    assert engine_model_name("mlx-lm", here, "m") == str(here.resolve())
    with pytest.raises(EngineError, match="unknown engine"):
        engine_model_name("nope", here, "m")


def _spec(engine: str, **fields: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = {
        "engine": engine, "backend": "b", "engine_args": ("--a",),
        "file": None, "mmproj": None,
    }
    return SimpleNamespace(**{**defaults, **fields})


class Plan:
    def flags(self) -> list[str]:
        return ["--plan"]


def test_each_engine_builds_its_own_argv() -> None:
    manifest = SimpleNamespace(path=Path("m.toml"), params_b=9)
    here = Path("/w")
    assert engine_load_args(
        manifest, _spec("vllm"), here, Plan(), context=8, card_args=("--card",)
    ) == [
        "--a", "--max-model-len", "8", *DECIDE_ARGS, *STRUCTURED_OUTPUTS_ARGS, "--card", "--plan",
    ]
    assert engine_load_args(
        manifest, _spec("llama-server", file="x.gguf", mmproj="p.gguf"), here, None,
        context=8,
    ) == ["-m", str(here / "x.gguf"), "--a", "--mmproj", str(here / "p.gguf"), "-c", "8"]
    assert engine_load_args(
        manifest, _spec(MlxVlmEngine.name), here, Plan(), context=8, card_args=("--c",)
    ) == ["--a", "--plan"]
    assert engine_load_args(
        manifest, _spec(MlxLmEngine.name), here, Plan(), context=8, card_args=("--c",)
    ) == ["--a", "--plan", "--prefill-step-size", "768"], "mlx-lm derives its step"
    with pytest.raises(EngineError, match="m.toml's b block names no `file`"):
        engine_load_args(manifest, _spec("llama-server"), here, None, context=8)
    assert engine_load_args(
        manifest, _spec("vllm"), here, None, context=8
    ) == VllmEngine.load_args(_spec("vllm"), here, 8, None)
    assert LlamaServerEngine.served_name(here, "m") == "m"
