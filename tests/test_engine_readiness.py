"""How an engine says it is up is the engine's business.

PHASE3-TTS.md section 4: every engine Crucible has today proves readiness by
polling its own HTTP `/v1/models`; `narrator.serve` instead prints a
`ready{device,backend}` line on stdout. `SubprocessEngine.ready()` grew a seam for
that — `announced_ready()` and `readiness_description()` — and these tests drive
it with `tests/fake_narrator.py`, which speaks narrator's real wire, rather than
with a stub that would only prove the seam calls something.

`crucible/engines/narrator.py` is not written yet. The engine below is a TEST
DOUBLE and not a preview of it: it reads the ready line out of the engine log
because `SubprocessEngine.start()` sends stdout there, while the real one will
need stdout as a pipe (narrator's protocol is newline-delimited JSON over stdin
and stdout, and Crucible has to write to it as well as read it). That is a second
seam in `start()`, and it is deliberately not invented here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.engines import (
    ENGINES,
    EngineError,
    SubprocessEngine,
    build_engine,
    engine_model_name,
    find_free_port,
)
from crucible.engines.base import SubprocessEngine as BaseEngine
from crucible.engines.mlx_lm import MlxLmEngine
from crucible.engines.mlx_vlm import MlxVlmEngine
from crucible.engines.vllm import VllmEngine

FAKE_NARRATOR = Path(__file__).resolve().parent / "fake_narrator.py"
FAKE_MLX_VLM = Path(__file__).resolve().parent / "fake_mlx_vlm"


class ReadyLineEngine(SubprocessEngine):
    """An engine that is up when it has printed `{"type": "ready"}` on stdout."""

    name = "fake-narrator"

    #: Whether to keep the process's stdin open — see `stdio()`. A test that is
    #: about the engine DYING turns it off, so that nothing is holding it up.
    hold_stdin = True

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        # The script itself, with no shell in front of it. This used to be
        # `sh -c "sleep 60 | <python> <script>"`, which on win32 handed a
        # backslashed interpreter path to Git Bash (it arrived as
        # `C:UserstelltApp...python.exe: command not found`) and, where it did
        # run, left `sleep` and the script behind a shell that was the only pid
        # a stop could see.
        return [sys.executable, str(FAKE_NARRATOR), "--engine", "higgs-v3"]

    def stdio(self, log_handle):  # type: ignore[no-untyped-def]
        """stdin held open as a PIPE, output to the log.

        `start()` gives an engine `stdin=DEVNULL`, on which this script's
        `for line in sys.stdin` reaches EOF and the process exits cleanly the
        instant it is ready — and `ready()` checks `poll()` before it checks
        the announcement, so a process that has already exited is an exit and
        not a readiness. A pipe nobody writes to holds it open, which is the
        `stdio()` seam narrator's own engine uses for the same reason.
        """
        wiring = dict(super().stdio(log_handle))
        if self.hold_stdin:
            wiring["stdin"] = subprocess.PIPE
        return wiring

    def detach(self) -> None:
        process = self._process
        if process is not None and process.stdin is not None:
            process.stdin.close()

    def announced_ready(self) -> str | None:
        for line in self.log_tail(200).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "ready":
                return (
                    f"{self.name} is ready on {message.get('device')} "
                    f"({message.get('backend')})"
                )
        return None

    def readiness_description(self) -> str:
        return "print a ready line on stdout"


@pytest.fixture
def engine(tmp_path: Path) -> ReadyLineEngine:
    built = ReadyLineEngine(
        python=Path(sys.executable), log_path=tmp_path / "engine-probe.log"
    )
    yield built
    try:
        built.stop()
    except EngineError:
        pass


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    return directory


def test_a_stdout_ready_line_is_readiness(
    engine: ReadyLineEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No HTTP route is ever polled, and the engine still comes up."""
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_DELAY_S", "0")
    said: list[str] = []
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0, on_progress=said.append)
    assert any("is ready on fake" in message for message in said)


def test_a_slow_start_streams_warming_messages(
    engine: ReadyLineEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """narrator on cuda-linux takes about 110 s; the lane must say something."""
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_DELAY_S", "3")
    said: list[str] = []
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(60.0, on_progress=said.append)
    assert any("loading" in message for message in said), said
    assert any("is ready on fake" in message for message in said), said


def test_a_ready_line_that_never_comes_times_out_by_name(
    engine: ReadyLineEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_NEVER", "1")
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(3.0)
    message = str(caught.value)
    # The engine's own description of what it was waiting for, not /v1/models.
    assert "did not print a ready line on stdout within 3s" in message
    assert "/v1/models" not in message


def test_an_engine_that_dies_before_it_is_ready_says_so(
    engine: ReadyLineEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commonest real failure on cuda-linux: an environment fact kills the
    engine seconds in."""
    monkeypatch.setenv("CRUCIBLE_FAKE_EXIT_CODE", "3")
    engine.hold_stdin = False
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(30.0)
    message = str(caught.value)
    assert "exited 3 before it was ready" in message
    assert "told to exit before becoming ready" in message


def test_sigterm_stops_it(
    engine: ReadyLineEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_DELAY_S", "0")
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0)
    assert engine.pids
    engine.stop()
    assert engine.pids == frozenset()


class FakeMlxVlmEngine(MlxVlmEngine):
    """The REAL `MlxVlmEngine`, pointed at a fake `mlx_vlm` on PYTHONPATH.

    A subclass that overrides `environment()` and nothing else, so the command
    under test — the llm env's python running `mlx_vlm_serve.py` — is the one
    `MlxVlmEngine.command()` actually builds. A double that reimplemented
    `command()` would test the double. `tests/test_mlx_vlm_serve.py` drives
    the wire; these two are the readiness contract only.
    """

    def environment(self) -> dict[str, str]:
        return {"PYTHONPATH": str(FAKE_MLX_VLM)}


def test_mlx_vlm_is_ready_when_v1_models_answers_and_needs_no_confirm(
    tmp_path: Path,
) -> None:
    """The measured difference from mlx-lm, and the reason this class is short.

    Crucible's page server loads BEFORE it binds — the order mlx-vlm's own
    server had — so a 200 from `/v1/models` means the weights are in memory
    and the base class's poll is the honest check. mlx-lm answers that route
    while still reading gigabytes, which is why IT needs a one-token
    completion.
    """
    assert MlxVlmEngine.confirm is BaseEngine.confirm
    assert MlxLmEngine.confirm is not BaseEngine.confirm

    weights = tmp_path / "dots"
    weights.mkdir()
    engine = FakeMlxVlmEngine(
        python=Path(sys.executable), log_path=tmp_path / "engine.log"
    )
    port = find_free_port()
    # `str(weights)` and not a resolved spelling: what the engine reports is
    # what it was given, and `engine_model_name` has to agree.
    engine.start(weights, str(weights), port, ["--width", "1"])
    try:
        engine.ready(60.0)
        assert engine.base_url == f"http://127.0.0.1:{port}"
        assert engine.pids
    finally:
        engine.stop()
    assert engine.pids == frozenset()
    # The command really was the one the class builds.
    log = (tmp_path / "engine.log").read_text(encoding="utf-8", errors="replace")
    assert "mlx_vlm_serve.py --model" in log


def test_mlx_vlm_serving_something_else_is_not_a_not_yet(tmp_path: Path) -> None:
    """An engine that is up and serving the WRONG model must raise, not poll."""
    weights = tmp_path / "dots"
    weights.mkdir()
    engine = FakeMlxVlmEngine(
        python=Path(sys.executable), log_path=tmp_path / "engine.log"
    )
    engine.start(
        weights, "some-name-it-will-never-report", find_free_port(), ["--width", "1"]
    )
    try:
        with pytest.raises(EngineError) as caught:
            engine.ready(60.0)
        assert "will not proxy a model it did not ask for" in str(caught.value)
    finally:
        engine.stop()


def test_the_two_mlx_engines_name_a_model_differently_and_that_is_measured(
    tmp_path: Path,
) -> None:
    """mlx-lm resolves the path it was given and reports that; mlx-vlm stores
    the string verbatim. Resolving for mlx-vlm would introduce the very
    mismatch the resolve prevents on the other one."""
    unresolved = tmp_path / "weights" / ".." / "weights"
    (tmp_path / "weights").mkdir()
    assert engine_model_name("mlx-vlm", unresolved, "dots-ocr") == str(unresolved)
    assert engine_model_name("mlx-lm", unresolved, "dots-ocr") == str(
        unresolved.resolve()
    )
    # vLLM is told the Crucible id and answers to it, so there is no path at all.
    assert engine_model_name("vllm", unresolved, "dots-ocr") == "dots-ocr"


def test_this_build_has_four_engine_classes_for_three_backends() -> None:
    """One per (backend, class family), and fewer classes than pairs.

    cuda-linux serves text and pages with vLLM and llama-windows serves both
    with llama-server, so those two backends contribute one class each;
    mlx-darwin needs mlx-lm for one family and mlx-vlm for the other.
    """
    assert sorted(ENGINES) == ["llama-server", "mlx-lm", "mlx-vlm", "vllm"]
    built = build_engine("mlx-vlm", Path(sys.executable), Path("/tmp/x.log"))
    assert isinstance(built, MlxVlmEngine)
    with pytest.raises(EngineError) as caught:
        build_engine("mlx-vlm-2", Path(sys.executable), Path("/tmp/x.log"))
    assert "unknown engine 'mlx-vlm-2'" in str(caught.value)


def test_the_http_engines_did_not_change(
    engine: ReadyLineEngine, weights: Path
) -> None:
    """The seam is an override point, not a rewrite: vLLM, mlx-lm and mlx-vlm
    still prove readiness exactly as PHASE2-LLM.md section 3 specifies, through
    the base class's own `/v1/models` poll."""
    for cls in (VllmEngine, MlxLmEngine, MlxVlmEngine):
        assert cls.announced_ready is BaseEngine.announced_ready
        assert cls.readiness_description is BaseEngine.readiness_description
    # And the default description still names the route, so a vLLM timeout reads
    # the way it always did.
    engine.start(weights, "deathstalker", 7654, [])
    assert BaseEngine.readiness_description(engine) == (
        "answer http://127.0.0.1:7654/v1/models"
    )
