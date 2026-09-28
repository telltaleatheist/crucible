from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

from crucible import procgroup
from crucible.engines import EngineError, NarratorEngine, build_voice_engine
from crucible.narratorengines import NARRATOR_ENGINES
from crucible.engines import base as engine_base
from crucible.engines.base import SubprocessEngine as BaseEngine
from crucible.engines.mlx_lm import MlxLmEngine
from crucible.engines.narrator import (
    ENGINE_VARIABLE,
    STACK_ENV_PREFIX_VARIABLE,
    env_prefix_variable_for,
    MAX_NUM_SEQS_VARIABLE,
    MEM_FRACTION_VARIABLE,
    CONTEXT_LENGTH_VARIABLE,
    HIGGS_V3_MLX_WEIGHTS_GB,
    MLX_BATCH_VARIABLE,
    MLX_CACHE_LIMIT_VARIABLE,
    MLX_MEM_BUDGET_VARIABLE,
    MLX_TIERS,
    MODULE,
    STACK_VARIABLE,
    EngineWouldNotStop,
    higgs_env_prefix,
    mlx_render_profile,
)
from crucible.engines.vllm import VllmEngine
from crucible.errors import JobCancelled
from crucible.ttsstream import STREAM_BATCH_WIDTH
from crucible.narratorvoices import (
    DOCUMENT_VARIABLE,
    MLX_MODEL_VARIABLE,
    VoicesDocument,
    write_document,
)
from crucible.voices import parse_voice
from crucible.narratorengines import NARRATOR_ENGINE_SAMPLING

from .conftest import end_process_tree
from .fake_narrator_engine import FAKE_NARRATOR, FakeNarratorEngine
from .test_voices import GOOD

BATCH_TERMINAL = frozenset({"batch_done"})

A_FUTURE_ENGINE = "an-engine-with-no-document"

A_64_GIB_MAC = 64 * 1024 * 1024 * 1024
A_16_GIB_MAC = 16 * 1024 * 1024 * 1024


def a_document(
    home: Path, weights: Path, voice_id: str = "deathstalker"
) -> VoicesDocument:
    manifest = parse_voice(
        GOOD.replace('id = "probe"', f'id = "{voice_id}"'),
        Path(f"{voice_id}.toml"),
        voice_id,
    )
    return write_document(home, manifest, manifest.spec("cuda-linux"), weights)


@pytest.fixture(autouse=True)
def brief_quit_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    return directory


@pytest.fixture
def engine(tmp_path: Path, weights: Path) -> Iterator[FakeNarratorEngine]:
    built = FakeNarratorEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "engine-deathstalker.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, weights),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    yield built
    try:
        built.stop()
    except EngineError:
        pass


def up(engine: FakeNarratorEngine, weights: Path) -> FakeNarratorEngine:
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0)
    return engine


def test_the_argv_is_narrator_serve_and_nothing_else() -> None:
    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path("/opt/env/bin/python"),
        log_path=Path("/tmp/x.log"),
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    assert built.command(Path("/weights"), "owen", 7100, []) == [
        str(Path("/opt/env/bin/python")),
        "-m",
        MODULE,
    ]
    assert "owen" not in built.command(Path("/weights"), "owen", 7100, [])
    assert built.environment()[ENGINE_VARIABLE] == A_FUTURE_ENGINE


def a_venv(tmp_path: Path, name: str = "tts-higgs-v3") -> Path:
    root = tmp_path / name
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.write_text("", encoding="utf-8")
    return python


def test_a_higgs_worker_is_told_the_stack_the_env_and_the_width(
    tmp_path: Path,
) -> None:
    python = a_venv(tmp_path)
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3",
        python,
        tmp_path / "x.log",
        serving_stack="sglang-omni",
        max_num_seqs=16,
        mem_fraction=None,
        context_length=None,
        voices=document,
        mlx_total_bytes=None,
    )
    environment = built.environment()
    assert environment[ENGINE_VARIABLE] == "higgs-v3"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment[STACK_VARIABLE] == "sglang-omni"
    assert environment[env_prefix_variable_for("sglang-omni")] == str(
        python.parent.parent)
    assert environment[MAX_NUM_SEQS_VARIABLE] == "16"
    assert environment[DOCUMENT_VARIABLE] == str(document.path)
    assert MLX_MODEL_VARIABLE not in environment


def test_a_higgs_worker_without_a_document_is_refused_by_name(
    tmp_path: Path,
) -> None:
    python = a_venv(tmp_path)
    for stack in ("sglang-omni", None):
        with pytest.raises(EngineError) as caught:
            build_voice_engine(
                "higgs-v3", python, tmp_path / "x.log",
                serving_stack=stack, max_num_seqs=16,
                mem_fraction=None, context_length=None,
                voices=None, mlx_total_bytes=None)
        assert DOCUMENT_VARIABLE in str(caught.value)
        assert "narratorvoices" in str(caught.value)


def test_a_document_for_an_engine_that_reads_none_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        NarratorEngine(
            narrator_engine=A_FUTURE_ENGINE,
            python=a_venv(tmp_path, "tts-future"),
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None,
        )
    assert "takes its weights on the load message" in str(caught.value)
    assert "'higgs-v3'" in str(caught.value)


def test_the_width_is_a_string_because_an_environment_holds_strings(
    tmp_path: Path,
) -> None:
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None)
    for name, value in built.environment().items():
        assert isinstance(value, str), name


def test_a_higgs_worker_with_no_width_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=None,
            mem_fraction=None, context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None)
    assert MAX_NUM_SEQS_VARIABLE in str(caught.value)
    assert "[voice.serving]" in str(caught.value)


def test_a_width_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=0,
            mem_fraction=None, context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None)
    assert "at least 1" in str(caught.value)


def a_venv_on_a_conda_env(tmp_path: Path) -> Path:
    conda = tmp_path / "anaconda3" / "envs" / "crucible"
    (conda / "bin").mkdir(parents=True)
    (conda / "conda-meta").mkdir()
    base = conda / "bin" / "python3.11"
    base.write_text("", encoding="utf-8")

    venv = tmp_path / ".crucible" / "envs" / "tts-higgs-v3"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {conda / 'bin'}\nexecutable = {base}\n", encoding="utf-8"
    )
    (venv / "bin" / "sgl-omni").write_text("", encoding="utf-8")
    python = venv / "bin" / "python"
    try:
        python.symlink_to(base)
    except OSError as refused:
        pytest.skip(f"this layout is a symlink and this host refused one: {refused}")
    return python


def test_the_prefix_is_the_env_the_stack_is_in_not_the_one_it_symlinks_to(
    tmp_path: Path,
) -> None:
    python = a_venv_on_a_conda_env(tmp_path)
    venv = python.parent.parent
    built = build_voice_engine(
        "higgs-v3", python, tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None)
    prefix = built.environment()[env_prefix_variable_for("sglang-omni")]
    assert prefix == str(venv)
    assert higgs_env_prefix(python, "sglang-omni") == venv
    assert str(python.resolve().parent.parent) != prefix
    assert (Path(prefix) / "bin" / "sgl-omni").is_file()


def test_a_conda_env_is_its_own_prefix(tmp_path: Path) -> None:
    conda = tmp_path / "anaconda3" / "envs" / "tts-higgs-v3"
    (conda / "bin").mkdir(parents=True)
    (conda / "conda-meta").mkdir()
    python = conda / "bin" / "python"
    python.write_text("", encoding="utf-8")
    built = build_voice_engine(
        "higgs-v3", python, tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None)
    assert built.environment()[env_prefix_variable_for("sglang-omni")] == str(conda)
    assert higgs_env_prefix(python, "sglang-omni") == conda


def test_an_interpreter_that_is_not_in_a_venv_is_refused(tmp_path: Path) -> None:
    stray = tmp_path / "not-an-env" / "bin" / "python"
    stray.parent.mkdir(parents=True)
    stray.write_text("", encoding="utf-8")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", stray, tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=16,
            mem_fraction=None, context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None)
    assert env_prefix_variable_for("sglang-omni") in str(caught.value)
    assert "pyvenv.cfg" in str(caught.value)
    assert "conda-meta" in str(caught.value)
    assert "sgl-omni" in str(caught.value)
    assert "narrator (higgs-v3)" in str(caught.value)


def test_an_arm_that_starts_no_server_is_told_none_of_the_three(
    tmp_path: Path,
) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_64_GIB_MAC)
    environment = built.environment()
    assert environment[ENGINE_VARIABLE] == "higgs-v3"
    for name in (STACK_VARIABLE, MAX_NUM_SEQS_VARIABLE,
                 *STACK_ENV_PREFIX_VARIABLE.values()):
        assert name not in environment, name
    assert environment[DOCUMENT_VARIABLE] == str(document.path)


def test_the_arm_that_starts_no_server_is_told_its_batch_width(
    tmp_path: Path,
) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_64_GIB_MAC)
    environment = built.environment()
    assert environment[MLX_BATCH_VARIABLE] == "64"
    assert environment[MLX_BATCH_VARIABLE] != str(16)


def test_a_stated_mem_fraction_and_context_length_reach_the_served_arm(
    tmp_path: Path,
) -> None:
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=4,
        mem_fraction=0.48, context_length=8192,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None).environment()
    assert environment[MEM_FRACTION_VARIABLE] == "0.48"
    assert environment[CONTEXT_LENGTH_VARIABLE] == "8192"
    assert environment[MAX_NUM_SEQS_VARIABLE] == "4"


def test_a_voice_stating_neither_sets_neither_variable(tmp_path: Path) -> None:
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None).environment()
    assert MEM_FRACTION_VARIABLE not in environment
    assert CONTEXT_LENGTH_VARIABLE not in environment


def test_both_levers_reach_the_IN_PROCESS_arm_too(tmp_path: Path) -> None:
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=0.48, context_length=8192,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC).environment()
    assert environment[MEM_FRACTION_VARIABLE] == "0.48"
    assert environment[CONTEXT_LENGTH_VARIABLE] == "8192"
    assert MAX_NUM_SEQS_VARIABLE not in environment


def test_the_width_and_the_budget_come_from_one_row(tmp_path: Path) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_64_GIB_MAC).environment()
    assert environment[MLX_BATCH_VARIABLE] == "64"
    assert environment[MLX_MEM_BUDGET_VARIABLE] == "42"
    assert environment[MLX_CACHE_LIMIT_VARIABLE] == "8"


def test_a_smaller_mac_gets_a_smaller_row(tmp_path: Path) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_16_GIB_MAC).environment()
    assert environment[MLX_BATCH_VARIABLE] == "24"
    assert environment[MLX_MEM_BUDGET_VARIABLE] == "13"
    assert environment[MLX_CACHE_LIMIT_VARIABLE] == "3"


def test_every_tier_can_hold_its_own_weights_and_cache() -> None:
    for engine, rows in MLX_TIERS.items():
        assert rows[-1].min_total_mib == 0, engine
        for row in rows:
            headroom = (
                row.mem_budget_gb - HIGGS_V3_MLX_WEIGHTS_GB - row.cache_limit_gb
            )
            assert headroom > 0, f"{engine} {row.name}"
        floors = [row.min_total_mib for row in rows]
        assert floors == sorted(floors, reverse=True), engine


def test_the_served_arm_is_not_told_the_mlx_width(tmp_path: Path) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=None)
    environment = built.environment()
    for name in (
        MLX_BATCH_VARIABLE, MLX_MEM_BUDGET_VARIABLE, MLX_CACHE_LIMIT_VARIABLE
    ):
        assert name not in environment, name
    assert environment[MAX_NUM_SEQS_VARIABLE] == "16"


def test_the_served_arm_is_refused_a_memory_figure(tmp_path: Path) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=16,
            mem_fraction=None, context_length=None, voices=document,
            mlx_total_bytes=A_64_GIB_MAC)
    assert "mlx_total_bytes" in str(caught.value)
    assert "launcher's GPU fractions" in str(caught.value)


def test_the_in_process_arm_without_a_memory_figure_is_refused_by_name(
    tmp_path: Path,
) -> None:
    document = a_document(tmp_path, tmp_path / "weights")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack=None, max_num_seqs=16,
            mem_fraction=None, context_length=None, voices=document,
            mlx_total_bytes=None)
    assert MLX_BATCH_VARIABLE in str(caught.value)
    assert MLX_MEM_BUDGET_VARIABLE in str(caught.value)
    assert "probe_unified_memory" in str(caught.value)


def test_an_unreadable_memory_figure_is_refused_rather_than_banded() -> None:
    for figure in (0, -1, 42.5, None):
        with pytest.raises(EngineError) as caught:
            mlx_render_profile("higgs-v3", figure)
        assert "probe_unified_memory" in str(caught.value)


def test_an_engine_with_no_measured_mlx_width_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        mlx_render_profile(A_FUTURE_ENGINE, A_64_GIB_MAC)
    assert A_FUTURE_ENGINE in str(caught.value)
    assert MLX_BATCH_VARIABLE in str(caught.value)


def test_an_engine_that_reads_no_higgs_vocabulary_is_told_no_mlx_width(
    tmp_path: Path,
) -> None:
    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path("/opt/env/bin/python"),
        log_path=Path("/tmp/x.log"),
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    assert MLX_BATCH_VARIABLE not in built.environment()


def test_a_stack_on_an_engine_that_has_none_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        NarratorEngine(
            narrator_engine=A_FUTURE_ENGINE,
            python=a_venv(tmp_path, "tts-future"),
            log_path=tmp_path / "x.log",
            serving_stack="sglang-omni",
            max_num_seqs=16,
            mem_fraction=None,
            context_length=None,
            voices=None,
            mlx_total_bytes=None,
        )
    assert "serving_stack='sglang-omni'" in str(caught.value)


def test_the_four_facts_have_no_defaults(tmp_path: Path) -> None:
    python = a_venv(tmp_path)
    with pytest.raises(TypeError):
        NarratorEngine(
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
        )
    with pytest.raises(TypeError):
        NarratorEngine(
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
        )
    with pytest.raises(TypeError):
        NarratorEngine(
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
        )


def test_the_engine_id_is_in_the_name_so_a_refusal_says_which(
    tmp_path: Path,
) -> None:
    built = build_voice_engine(
        "higgs-v3", Path(sys.executable), Path("/tmp/x.log"),
        serving_stack=None, max_num_seqs=None, mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC)
    assert built.name == "narrator (higgs-v3)"
    assert built.narrator_engine == "higgs-v3"


def test_an_engine_this_build_cannot_start_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v2", Path(sys.executable), Path("/tmp/x.log"),
            serving_stack=None, max_num_seqs=None,
            mem_fraction=None, context_length=None,
            voices=None, mlx_total_bytes=None)
    assert "unknown narrator engine 'higgs-v2'" in str(caught.value)
    assert "['higgs-v3']" in str(caught.value)


def test_every_engine_a_manifest_may_name_is_one_this_build_can_start() -> None:
    assert set(NARRATOR_ENGINE_SAMPLING) == set(NARRATOR_ENGINES)
    assert set(NARRATOR_ENGINE_SAMPLING) == set(STREAM_BATCH_WIDTH)


def test_there_is_no_base_url_and_saying_so_is_the_point(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.base_url
    assert "has no base url" in str(caught.value)


def test_the_http_engines_kept_both_seams(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    for cls in (VllmEngine, MlxLmEngine):
        assert cls.stdio is BaseEngine.stdio
        assert cls.attach is BaseEngine.attach
        assert cls.detach is BaseEngine.detach
    plain = BaseEngine(python=Path(sys.executable), log_path=Path("/tmp/x.log"))
    assert BaseEngine.stdio(plain, "handle") == {
        "stdin": engine_base.subprocess.DEVNULL,
        "stdout": "handle",
        "stderr": engine_base.subprocess.STDOUT,
    }
    wiring = engine.stdio("handle")
    assert wiring["stdin"] is engine_base.subprocess.PIPE
    assert wiring["stdout"] is engine_base.subprocess.PIPE
    assert wiring["stderr"] == "handle"
    assert wiring["encoding"] == "utf-8"


def test_a_stdout_ready_line_is_readiness(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_DELAY_S", "0")
    said: list[str] = []
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0, on_progress=said.append)
    assert any("is ready on fake" in message for message in said), said
    assert engine.readiness_description() == "print a ready line on stdout"


def test_a_ready_line_that_never_comes_times_out_by_name(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_NEVER", "1")
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(3.0)
    message = str(caught.value)
    assert "did not print a ready line on stdout within 3s" in message
    assert "/v1/models" not in message


def test_an_engine_that_dies_before_it_is_ready_says_so(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_EXIT_CODE", "3")
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(30.0)
    message = str(caught.value)
    assert "exited 3 before it was ready" in message
    assert "told to exit before becoming ready" in message


def test_a_load_sends_narrators_own_message_and_returns_its_answer(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    loaded = engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    assert loaded["type"] == "loaded"
    assert loaded["sampleRate"] == 24_000

    sent = transcript.read_text(encoding="utf-8").splitlines()
    assert len(sent) == 1
    message = json.loads(sent[0])
    assert message == {
        "action": "load",
        "voice": "deathstalker",
        "warm": True,
    }
    assert "modelDir" not in message
    assert "caps" not in message


def test_a_load_with_no_document_carries_the_weights_on_the_message(
    tmp_path: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    built = FakeNarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path(sys.executable),
        log_path=tmp_path / "engine-owen.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    try:
        up(built, weights)
        built.load(voice="owen", weights_dir=weights, warm=True)
    finally:
        built.stop()
    message = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])
    assert message == {
        "action": "load",
        "voice": "owen",
        "modelDir": str(weights),
        "warm": True,
    }


def test_a_voice_the_document_does_not_carry_is_refused_before_it_is_sent(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.load(voice="sigma", weights_dir=weights, warm=True)
    assert "carries no voice 'sigma'" in str(caught.value)
    assert "['deathstalker']" in str(caught.value)
    assert not transcript.exists()


def test_a_directory_the_document_disagrees_with_is_refused(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.load(
            voice="deathstalker", weights_dir=tmp_path / "elsewhere", warm=True
        )
    assert "two directories for one voice" in str(caught.value)


def test_the_fake_worker_refuses_a_model_dir_exactly_as_narrator_does(
    tmp_path: Path,
    weights: Path,
) -> None:
    built = FakeNarratorEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "engine-deathstalker.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, weights),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    try:
        up(built, weights)
        with pytest.raises(EngineError) as caught:
            for _ in built.converse(
                {
                    "action": "load",
                    "voice": "deathstalker",
                    "modelDir": str(weights),
                    "warm": True,
                },
                terminal=frozenset({"loaded"}),
                silence_timeout=10.0,
            ):
                pass
        assert "Higgs v3 load carried modelDir=" in str(caught.value)
    finally:
        built.stop()


def test_rows_retire_out_of_order_and_nothing_reorders_them(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    indices = [
        message["i"]
        for message in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": 7, "text": "one"}, {"i": 8, "text": "two"},
                          {"i": 9, "text": "three"}],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=30.0,
        )
        if message["type"] == "batch_item"
    ]
    assert indices == [9, 8, 7]


def test_a_failed_row_arrives_beside_its_successful_neighbours(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_FAIL_ROW", "1")
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    rows = {
        message["i"]: message
        for message in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": i, "text": "a sentence"} for i in (0, 1, 2)],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=30.0,
        )
        if message["type"] == "batch_item"
    }
    assert sorted(rows) == [0, 1, 2]
    assert "data" not in rows[1] and "message" in rows[1]
    assert "data" in rows[0] and "data" in rows[2]


def test_a_whole_request_refusal_ends_the_conversation(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        list(
            engine.converse(
                {"action": "nonsense"},
                terminal=BATCH_TERMINAL,
                silence_timeout=30.0,
            )
        )
    assert "refused the request" in str(caught.value)
    assert "does not know action" in str(caught.value)


def test_a_cancel_is_sent_and_the_run_is_reported_as_cancelled(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    with pytest.raises(JobCancelled):
        for _ in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": 0, "text": "a sentence"}],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=30.0,
            cancelled=lambda: True,
        ):
            pass
    assert '"action": "cancel"' in transcript.read_text(encoding="utf-8")


def test_an_engine_that_ignores_the_cancel_outlasts_its_grace_and_is_named(
    engine: FakeNarratorEngine,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.engines.narrator.CANCEL_GRACE_SECONDS", 0.5)
    monkeypatch.setenv("CRUCIBLE_FAKE_IGNORE_CANCEL", "1")
    monkeypatch.setenv("CRUCIBLE_FAKE_ROW_DELAY_MS", "80")
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    with pytest.raises(EngineWouldNotStop) as caught:
        for _ in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": index, "text": "a sentence"} for index in range(40)],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=60.0,
            cancelled=lambda: True,
        ):
            pass
    assert "was sent a cancel" in str(caught.value)
    assert "does not read that flag" in str(caught.value)


def test_a_line_on_stdout_that_is_not_a_message_is_a_refusal_naming_it(
    tmp_path: Path, weights: Path
) -> None:

    class ChattyEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            return [
                sys.executable,
                "-c",
                "import sys, time\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "sys.stdin.readline()\n"
                "print('Loading checkpoint shards:  42%', flush=True)\n"
                "time.sleep(30)\n",
            ]

    built = ChattyEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "chatty.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    try:
        built.start(weights, "deathstalker", 0, [])
        built.ready(30.0)
        with pytest.raises(EngineError) as caught:
            list(
                built.converse(
                    {"action": "load", "voice": "deathstalker"},
                    terminal=frozenset({"loaded"}),
                    silence_timeout=30.0,
                )
            )
    finally:
        try:
            built.stop()
        except EngineError:
            pass
    message = str(caught.value)
    assert "not a protocol message" in message
    assert "Loading checkpoint shards:  42%" in message


def test_silence_during_a_request_is_reported_as_silence(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        list(
            engine.converse(
                {"action": "stop"},
                terminal=frozenset({"batch_done"}),
                silence_timeout=1.0,
            )
        )
    assert "said nothing at all for 1s" in str(caught.value)


def test_an_engine_that_dies_mid_request_says_so(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    conversation = engine.converse(
        {"action": "quit"},
        terminal=frozenset({"batch_done"}),
        silence_timeout=30.0,
    )
    with pytest.raises(EngineError) as caught:
        list(conversation)
    text = str(caught.value)
    assert "in the middle of a request" in text or "closed its stdout" in text


def test_stop_asks_narrator_to_quit_before_it_signals(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    assert engine.pids
    engine.stop()
    assert engine.pids == frozenset()
    assert '{"action": "quit"}' in transcript.read_text(encoding="utf-8")


def test_a_worker_that_will_not_go_is_reported_and_never_sigkilled(
    tmp_path: Path, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BaseEngine, "sigterm_wait_seconds", 1.0)

    class DeafEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            return [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "time.sleep(120)\n",
            ]

    built = DeafEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "deaf.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    built.start(weights, "deathstalker", 0, [])
    built.ready(30.0)
    pid = next(iter(built.pids))
    try:
        if procgroup.platform_kind() == procgroup.WIN32:
            built.stop()
            assert built.pids == frozenset()
            return
        with pytest.raises(EngineError) as caught:
            built.stop()
        message = str(caught.value)
        assert "did not exit within 1s of SIGTERM" in message
        assert "does not SIGKILL" in message
        assert f"`kill {pid}`" in message
    finally:
        end_process_tree(pid)


def test_stopping_a_worker_that_ignores_sigterm_does_not_wedge_the_server(
    tmp_path: Path, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BaseEngine, "sigterm_wait_seconds", 1.0)

    class DeafEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            return [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "time.sleep(120)\n",
            ]

    built = DeafEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "deaf2.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    built.start(weights, "deathstalker", 0, [])
    built.ready(30.0)
    pid = next(iter(built.pids))
    started = time.monotonic()
    try:
        if procgroup.platform_kind() == procgroup.WIN32:
            built.stop()
        else:
            with pytest.raises(EngineError):
                built.stop()
        assert time.monotonic() - started < 10.0 + procgroup.KILL_WAIT_SECONDS
    finally:
        end_process_tree(pid)


def test_the_reader_thread_is_gone_once_the_engine_is_stopped(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    engine.stop()
    assert engine._reader is None


def test_a_started_engine_refuses_to_be_started_twice(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.start(weights, "deathstalker", 0, [])
    assert "is already running" in str(caught.value)


def test_the_fake_worker_is_the_one_the_readiness_tests_use() -> None:
    assert FAKE_NARRATOR.is_file()
    assert FAKE_NARRATOR.name == "fake_narrator.py"


def test_the_stop_budget_is_the_whole_worst_case_of_stop() -> None:
    from crucible.engines import narrator

    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path(sys.executable),
        log_path=Path("unused.log"),
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    assert built.stop_budget_seconds == (
        narrator.QUIT_GRACE_SECONDS
        + narrator.READER_JOIN_SECONDS
        + procgroup.stop_budget_seconds(BaseEngine.sigterm_wait_seconds)
        + 2
        * (
            narrator.LAUNCHED_SERVER_GRACE_SECONDS
            + narrator.LAUNCHED_SERVER_POLL_SECONDS
        )
    )
    assert built.stop_budget_seconds > narrator.QUIT_GRACE_SECONDS


def test_a_narrator_env_that_is_missing_names_the_tts_install(tmp_path: Path) -> None:
    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=tmp_path / "no-such-python",
        log_path=tmp_path / "engine.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    with pytest.raises(EngineError) as caught:
        built.start(tmp_path, "deathstalker", 0, [])
    assert "`crucible install tts`" in str(caught.value)


def test_missing_voice_weights_name_the_voices_pull(tmp_path: Path) -> None:
    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path(sys.executable),
        log_path=tmp_path / "engine.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    with pytest.raises(EngineError) as caught:
        built.start(tmp_path / "absent", "deathstalker", 0, [])
    assert "`crucible voices pull deathstalker`" in str(caught.value)


LINUX_PROC = pytest.mark.skipif(
    not Path("/proc/self/environ").is_file(),
    reason="the launched-server reap reads /proc/<pid>/environ (Linux only)",
)


def _launched_server(owner: int, *, deaf: bool) -> "subprocess.Popen[bytes]":
    import os
    import subprocess

    from crucible.engines.narrator import OWNER_MARKER_VARIABLE

    script = "import signal, time\n"
    if deaf:
        script += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    script += "print('up', flush=True)\ntime.sleep(60)\n"
    server = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, OWNER_MARKER_VARIABLE: str(owner)},
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert server.stdout is not None
    assert server.stdout.readline().strip() == b"up"
    return server


@LINUX_PROC
def test_a_server_carrying_the_owner_marker_is_found_and_reaped_on_sigterm(
    engine: FakeNarratorEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible.engines import narrator

    monkeypatch.setattr(narrator, "LAUNCHED_SERVER_GRACE_SECONDS", 3.0)
    monkeypatch.setattr(narrator, "LAUNCHED_SERVER_POLL_SECONDS", 0.1)
    owner = 999_999_001
    server = _launched_server(owner, deaf=False)
    try:
        assert narrator.processes_launched_by(owner) == frozenset({server.pid})
        engine._outlive_launched_servers(owner)
        server.wait(timeout=5)
        assert narrator.processes_launched_by(owner) == frozenset()
    finally:
        end_process_tree(server.pid)


@LINUX_PROC
def test_a_launched_server_deaf_to_sigterm_is_named_and_never_sigkilled(
    engine: FakeNarratorEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible.engines import narrator

    monkeypatch.setattr(narrator, "LAUNCHED_SERVER_GRACE_SECONDS", 1.0)
    monkeypatch.setattr(narrator, "LAUNCHED_SERVER_POLL_SECONDS", 0.1)
    owner = 999_999_002
    server = _launched_server(owner, deaf=True)
    try:
        with pytest.raises(EngineError) as caught:
            engine._outlive_launched_servers(owner)
        assert f"`kill {server.pid}`" in str(caught.value)
        assert server.poll() is None
    finally:
        end_process_tree(server.pid)
