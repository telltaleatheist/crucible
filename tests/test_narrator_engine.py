"""`crucible/engines/narrator.py` — the managed subprocess and its channel.

PHASE3-TTS.md section 4. Everything here drives `tests/fake_narrator.py`, which
speaks narrator's real JSON-lines wire, through the **real** engine: the only
thing the double replaces is the argv (`tests/fake_narrator_engine.py`).

Nothing here needs an accelerator, and nothing here is a mock of the engine. The
pipes, the reader thread, the correlation, the refusals and `stop()` are the code
that will run on the PC.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

from crucible.engines import (
    NARRATOR_ENGINES,
    EngineError,
    NarratorEngine,
    build_voice_engine,
)
from crucible.engines import base as engine_base
from crucible.engines.base import SubprocessEngine as BaseEngine
from crucible.engines.mlx_lm import MlxLmEngine
from crucible.engines.narrator import (
    ENGINE_VARIABLE,
    ENV_PREFIX_VARIABLE,
    MAX_NUM_SEQS_VARIABLE,
    MODULE,
    STACK_VARIABLE,
)
from crucible.engines.vllm import VllmEngine
from crucible.errors import JobCancelled
from crucible.narratorvoices import (
    DOCUMENT_VARIABLE,
    MLX_MODEL_VARIABLE,
    VoicesDocument,
    write_document,
)
from crucible.voices import NARRATOR_ENGINE_SAMPLING, parse_voice

from .fake_narrator_engine import FAKE_NARRATOR, FakeNarratorEngine
from .test_voices import GOOD

BATCH_TERMINAL = frozenset({"batch_done"})


def a_document(
    home: Path, weights: Path, voice_id: str = "deathstalker"
) -> VoicesDocument:
    """The voices document a load of `voice_id` from `weights` writes — the
    real writer on a real manifest, because what a `higgs-v3` engine is
    constructed with is this and nothing simpler."""
    manifest = parse_voice(
        GOOD.replace('id = "probe"', f'id = "{voice_id}"'),
        Path(f"{voice_id}.toml"),
        voice_id,
    )
    return write_document(home, manifest, manifest.spec("cuda-linux"), weights)


@pytest.fixture(autouse=True)
def brief_quit_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the wait `stop()` gives `quit` before it signals.

    Thirty seconds is right for a loaded SGLang-Omni releasing CUDA from inside
    itself. Several tests here deliberately leave a worker that is not reading
    its stdin at all, and paying that wait for each of them would put minutes on
    the suite for a number none of them is about.
    """
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    # And the readiness poll, which is two seconds because a vLLM load takes
    # minutes and polling it harder buys nothing. Every test here is up in
    # milliseconds, so the interval is the whole of its runtime.
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
        # The wire, not the server: this engine's argv is the fake worker and
        # no vllm-omni is started, so there is nothing for the three HIGGS_*
        # variables to configure. They have their own tests below.
        serving_stack=None,
        max_num_seqs=None,
        # The document IS read, by the fake exactly as by narrator: a load
        # names a voice in it or is refused.
        voices=a_document(tmp_path, weights),
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


# --------------------------------------------------------------- the spawn


def test_the_argv_is_narrator_serve_and_nothing_else() -> None:
    """narrator takes no configuration on the command line, so neither does this."""
    built = NarratorEngine(
        narrator_engine="orpheus",
        python=Path("/opt/env/bin/python"),
        log_path=Path("/tmp/x.log"),
        serving_stack=None,
        max_num_seqs=None,
        voices=None,
    )
    assert built.command(Path("/weights"), "owen", 7100, []) == [
        "/opt/env/bin/python",
        "-m",
        MODULE,
    ]
    # The voice, the weights directory and the port are deliberately absent: the
    # first two ride the `load` message, and narrator binds nothing.
    assert "owen" not in built.command(Path("/weights"), "owen", 7100, [])
    assert built.environment()[ENGINE_VARIABLE] == "orpheus"


def a_venv(tmp_path: Path, name: str = "tts-higgs-v3") -> Path:
    """A directory shaped like the env `crucible install tts` builds: a
    `bin/python` under a root carrying `pyvenv.cfg`. That file is what makes a
    venv a venv, and the engine checks for it rather than trusting the walk up
    from a binary."""
    root = tmp_path / name
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.write_text("", encoding="utf-8")
    return python


def test_a_higgs_worker_is_told_the_stack_the_env_and_the_width(
    tmp_path: Path,
) -> None:
    """The three variables narrator refuses by name, and the defect that found
    them: Crucible's first real `tts` render exited 3 before `ready` with
    `HIGGS_STACK is not set`."""
    python = a_venv(tmp_path)
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3",
        python,
        tmp_path / "x.log",
        serving_stack="vllm-omni",
        max_num_seqs=16,
        voices=document,
    )
    environment = built.environment()
    assert environment[ENGINE_VARIABLE] == "higgs-v3"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment[STACK_VARIABLE] == "vllm-omni"
    assert environment[ENV_PREFIX_VARIABLE] == str(python.parent.parent)
    assert environment[MAX_NUM_SEQS_VARIABLE] == "16"
    # The fourth thing, on both arms: where narrator resolves the voice.
    assert environment[DOCUMENT_VARIABLE] == str(document.path)
    # And NOT the MLX arm's base weights: a checkpoint voice never reads them.
    assert MLX_MODEL_VARIABLE not in environment


def test_a_higgs_worker_without_a_document_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """The keeper's finding on the Mac and the launcher's on the PC, one
    refusal: narrator resolves a Higgs v3 voice by name in the
    NARRATOR_HIGGS_VOICES document, on both arms, so an engine with none can
    load nothing — and says so before a process exists."""
    python = a_venv(tmp_path)
    for stack in ("vllm-omni", None):
        with pytest.raises(EngineError) as caught:
            build_voice_engine(
                "higgs-v3", python, tmp_path / "x.log",
                serving_stack=stack, max_num_seqs=16, voices=None)
        assert DOCUMENT_VARIABLE in str(caught.value)
        assert "narratorvoices" in str(caught.value)


def test_a_document_for_orpheus_is_refused_by_name(tmp_path: Path) -> None:
    """orpheus takes its weights on the load message and reads no
    NARRATOR_HIGGS_* variable; a document handed to it is a statement of where
    the weights are that nothing reads."""
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "orpheus", a_venv(tmp_path, "tts-orpheus"), tmp_path / "x.log",
            serving_stack=None, max_num_seqs=None,
            voices=a_document(tmp_path, tmp_path / "weights"))
    assert "orpheus takes its weights on the load message" in str(caught.value)


def test_the_width_is_a_string_because_an_environment_holds_strings(
    tmp_path: Path,
) -> None:
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="vllm-omni", max_num_seqs=16,
        voices=a_document(tmp_path, tmp_path / "weights"))
    for name, value in built.environment().items():
        assert isinstance(value, str), name


def test_a_higgs_worker_with_no_width_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="vllm-omni", max_num_seqs=None,
            voices=a_document(tmp_path, tmp_path / "weights"))
    assert MAX_NUM_SEQS_VARIABLE in str(caught.value)
    assert "[voice.serving]" in str(caught.value)


def test_a_width_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="vllm-omni", max_num_seqs=0,
            voices=a_document(tmp_path, tmp_path / "weights"))
    assert "at least 1" in str(caught.value)


def test_an_interpreter_that_is_not_in_a_venv_is_refused(tmp_path: Path) -> None:
    """`HIGGS_ENV` is the prefix narrator's launch script builds CUDA_HOME,
    PATH and the vllm-omni binary from. A prefix that is not a venv produces
    `No such file` at the end of a launch instead of here."""
    stray = tmp_path / "not-an-env" / "bin" / "python"
    stray.parent.mkdir(parents=True)
    stray.write_text("", encoding="utf-8")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", stray, tmp_path / "x.log",
            serving_stack="vllm-omni", max_num_seqs=16,
            voices=a_document(tmp_path, tmp_path / "weights"))
    assert ENV_PREFIX_VARIABLE in str(caught.value)
    assert "pyvenv.cfg" in str(caught.value)


def test_an_arm_that_starts_no_server_is_told_none_of_the_three(
    tmp_path: Path,
) -> None:
    """`mlx-darwin` renders in process (`HiggsV3MlxEngine` reads neither
    HIGGS_STACK nor HIGGS_MAX_NUM_SEQS) and `orpheus` loads vLLM 0.7.3 itself.
    Three levers read by nothing is how a Mac spawn ends up looking served.

    The DOCUMENT is not one of the three: the MLX arm reads it exactly as the
    served arm does, which is what the keeper found on the Mac."""
    for engine_id in ("higgs-v3", "orpheus"):
        document = (
            a_document(tmp_path, tmp_path / "weights") if engine_id == "higgs-v3"
            else None
        )
        built = build_voice_engine(
            engine_id, a_venv(tmp_path, f"tts-{engine_id}"), tmp_path / "x.log",
            serving_stack=None, max_num_seqs=16, voices=document)
        environment = built.environment()
        assert environment[ENGINE_VARIABLE] == engine_id
        for name in (STACK_VARIABLE, ENV_PREFIX_VARIABLE, MAX_NUM_SEQS_VARIABLE):
            assert name not in environment, (engine_id, name)
        assert (DOCUMENT_VARIABLE in environment) == (document is not None)


def test_a_stack_on_an_engine_that_has_none_is_refused(tmp_path: Path) -> None:
    """HIGGS_* is Higgs v3's vocabulary. Dropping the value in silence would
    leave the env recipe and the engine disagreeing about what that env
    starts."""
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "orpheus", a_venv(tmp_path, "tts-orpheus"), tmp_path / "x.log",
            serving_stack="vllm-omni", max_num_seqs=16, voices=None)
    assert "serving_stack='vllm-omni'" in str(caught.value)


def test_the_three_facts_have_no_defaults(tmp_path: Path) -> None:
    """A default would make FORGETTING to pass one indistinguishable from
    saying `None`, which is exactly how a worker is spawned without
    HIGGS_STACK — or, since 2026-09-14, without a voices document."""
    python = a_venv(tmp_path)
    with pytest.raises(TypeError):
        NarratorEngine(  # type: ignore[call-arg]
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
        )
    with pytest.raises(TypeError):
        NarratorEngine(  # type: ignore[call-arg]
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
        )


def test_the_engine_id_is_in_the_name_so_a_refusal_says_which(
    tmp_path: Path,
) -> None:
    built = build_voice_engine(
        "higgs-v3", Path(sys.executable), Path("/tmp/x.log"),
        serving_stack=None, max_num_seqs=None,
        voices=a_document(tmp_path, tmp_path / "weights"))
    assert built.name == "narrator (higgs-v3)"
    assert built.narrator_engine == "higgs-v3"


def test_an_engine_this_build_cannot_start_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v2", Path(sys.executable), Path("/tmp/x.log"),
            serving_stack=None, max_num_seqs=None, voices=None)
    assert "unknown narrator engine 'higgs-v2'" in str(caught.value)
    assert "['higgs-v3', 'orpheus']" in str(caught.value)


def test_every_engine_a_manifest_may_name_is_one_this_build_can_start() -> None:
    """The two tables are written in two files and must not drift.

    `crucible/voices.py` decides what a manifest's `narrator_engine` may say;
    this file decides what `build_voice_engine` will start. A voice naming an
    engine the second table lacks would pass every manifest check and fail at the
    spawn, which is the worst possible place to find out.
    """
    assert set(NARRATOR_ENGINE_SAMPLING) == set(NARRATOR_ENGINES)


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
    """`stdio()` and `attach()` are override points, not a rewrite.

    The readiness seam has this assertion in `test_engine_readiness.py`; this is
    the same claim for the second seam PHASE3-TTS.md section 4 said `start()`
    would need. vLLM and mlx-lm still get `stdin=DEVNULL` and both output streams
    in the log, byte for byte what they had.
    """
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
    # And narrator's own wiring keeps both pipes, in UTF-8.
    wiring = engine.stdio("handle")
    assert wiring["stdin"] is engine_base.subprocess.PIPE
    assert wiring["stdout"] is engine_base.subprocess.PIPE
    assert wiring["stderr"] == "handle"
    assert wiring["encoding"] == "utf-8"


# ------------------------------------------------------------- readiness


def test_a_stdout_ready_line_is_readiness(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_DELAY_S", "0")
    said: list[str] = []
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0, on_progress=said.append)
    assert any("is ready on fake" in message for message in said), said
    # No HTTP route was ever polled, and the engine's own description says so.
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
    """The commonest real failure on cuda-linux, and the reason the stderr tail
    is quoted into the error rather than pointed at."""
    monkeypatch.setenv("CRUCIBLE_FAKE_EXIT_CODE", "3")
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(30.0)
    message = str(caught.value)
    assert "exited 3 before it was ready" in message
    # The worker's own stderr, which is what explains it. This is the second seam
    # paying for itself: stderr goes to the log while stdout stays a pipe.
    assert "told to exit before becoming ready" in message


# ------------------------------------------------------------- the channel


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
    # THE VOICE ID AND `warm`, NOTHING ELSE. A `higgs-v3` load names a voice
    # in the NARRATOR_HIGGS_VOICES document; narrator refuses `modelDir` by
    # name on both arms (the launcher agent's finding on the PC and the
    # keeper's on the Mac, 2026-09-14), and the weights are the document's
    # `checkpointDir`.
    assert message == {
        "action": "load",
        "voice": "deathstalker",
        "warm": True,
    }
    assert "modelDir" not in message
    # No `caps`. See `NarratorEngine.load` on why that is a decision: narrator's
    # caps channel raises on a key it does not know; a Higgs voice's sampling
    # rides in the document instead.
    assert "caps" not in message


def test_an_orpheus_load_carries_the_weights_on_the_message(
    tmp_path: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """orpheus takes `modelDir` on the load and has no document — the shape it
    always had, kept byte for byte."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    built = FakeNarratorEngine(
        narrator_engine="orpheus",
        python=Path(sys.executable),
        log_path=tmp_path / "engine-owen.log",
        serving_stack=None,
        max_num_seqs=None,
        voices=None,
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
    """narrator's own refusal, made on this side of the pipe: a load for a voice
    the document does not carry names the file and the voices it does carry."""
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
    """Two statements of where the weights are, compared rather than trusted."""
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
    """The test double is only worth having if it fails the way the real thing
    failed. Send it the OLD message shape over the real pipes and it must
    refuse with narrator's words — which is the line the launcher agent read
    off the PC's engine log on 2026-09-14."""
    built = FakeNarratorEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "engine-deathstalker.log",
        serving_stack=None,
        max_num_seqs=None,
        voices=a_document(tmp_path, weights),
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
    """The fake retires a batch in reverse on purpose, and so does a real one.

    A consumer that relies on arrival order is wrong, so the engine must not
    quietly make it right: the iterator is arrival order, and `i` is the identity.
    """
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
    # A failure is `message` and no `data`; that is how the two are told apart.
    assert "data" not in rows[1] and "message" in rows[1]
    assert "data" in rows[0] and "data" in rows[2]


def test_a_whole_request_refusal_ends_the_conversation(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    """A top-level `error` is not a per-row failure and is not reported as one."""
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
    """Crucible asks narrator to stop rather than hanging up or killing it.

    Hanging up would leave the engine generating into nothing; killing it would
    take the voice off the card for the next job. The fake reads its stdin on the
    main thread, so it finishes the batch it is in before it sees the cancel —
    the real worker has a reader thread and aborts in flight. What this proves is
    the two things that are Crucible's: the cancel is SENT, and the run ends as
    cancelled rather than as a short success.
    """
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


def test_a_line_on_stdout_that_is_not_a_message_is_a_refusal_naming_it(
    tmp_path: Path, weights: Path
) -> None:
    """fd 1 carries the wire and nothing else — `crucible/workers.py`'s rule.

    A library that logs to stdout corrupted narrator's aligner on a 401-chunk
    book. Skipping the line would make the next protocol change a silent
    behaviour change, so it is a refusal that quotes what appeared.
    """

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
        # This engine exercises the WIRE; it starts no server, so the
        # three HIGGS_* variables have nothing to configure.
        serving_stack=None,
        max_num_seqs=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
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
                # `stop` answers `stopped`, which is not the terminal below, so
                # the conversation waits for something that will never come.
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


# ---------------------------------------------------------------- stopping


def test_stop_asks_narrator_to_quit_before_it_signals(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`quit` is narrator's own primary teardown: it unwinds the stdin loop from
    inside the process and releases the GPU through the atexit hooks."""
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
    """Owen's hardest rule in this file: a killed CUDA process wedges WSL2 until
    Windows reboots, so `stop()` gives up and says so rather than escalating."""
    monkeypatch.setattr(engine_base, "STOP_TIMEOUT_SECONDS", 1.0)

    class DeafEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            # Ignores SIGTERM AND never reads its stdin, so neither the `quit`
            # nor the signal gets it — which is the state a worker wedged in a
            # WSL dxg GPU wait is actually in.
            return [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "time.sleep(120)\n",
            ]

    built = DeafEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "deaf.log",
        # This engine exercises the WIRE; it starts no server, so the
        # three HIGGS_* variables have nothing to configure.
        serving_stack=None,
        max_num_seqs=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
    )
    built.start(weights, "deathstalker", 0, [])
    built.ready(30.0)
    pid = next(iter(built.pids))
    try:
        with pytest.raises(EngineError) as caught:
            built.stop()
        message = str(caught.value)
        assert "did not exit within 1s of SIGTERM" in message
        assert "does not SIGKILL" in message
    finally:
        # This test made the mess, so this test cleans it up. Nothing on this
        # process is holding CUDA, which is the only reason a SIGKILL is allowed
        # anywhere in this repository.
        os.kill(pid, signal.SIGKILL)


def test_stopping_a_worker_that_ignores_sigterm_does_not_wedge_the_server(
    tmp_path: Path, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal above must be a refusal and not a hang.

    `detach()` used to close stdout unconditionally, and closing a pipe another
    thread is blocked reading deadlocks on the buffered reader's own lock — in
    exactly the case that matters, a worker still alive with its stdout open. So
    `stop()` raised nothing and the server stopped instead. Measured 2026-09-13;
    this test is the one that found it.
    """
    monkeypatch.setattr(engine_base, "STOP_TIMEOUT_SECONDS", 1.0)

    class DeafEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            return [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "time.sleep(120)\n",
            ]

    built = DeafEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "deaf2.log",
        # This engine exercises the WIRE; it starts no server, so the
        # three HIGGS_* variables have nothing to configure.
        serving_stack=None,
        max_num_seqs=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
    )
    built.start(weights, "deathstalker", 0, [])
    built.ready(30.0)
    pid = next(iter(built.pids))
    started = time.monotonic()
    try:
        with pytest.raises(EngineError):
            built.stop()
        # quit grace (1s) + SIGTERM wait (1s) + the reader join, and nothing else.
        assert time.monotonic() - started < 10.0
    finally:
        os.kill(pid, signal.SIGKILL)


def test_the_reader_thread_is_gone_once_the_engine_is_stopped(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    engine.stop()
    assert engine._reader is None  # noqa: SLF001 — the thread is the thing tested


def test_a_started_engine_refuses_to_be_started_twice(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.start(weights, "deathstalker", 0, [])
    assert "is already running" in str(caught.value)


def test_the_fake_worker_is_the_one_the_readiness_tests_use() -> None:
    """One fake, driven by two builders. If this file grew its own, the render
    door and the streaming door would be built against two wires."""
    assert FAKE_NARRATOR.is_file()
    assert FAKE_NARRATOR.name == "fake_narrator.py"
