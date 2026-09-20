"""An engine log survives the reload that investigates the engine.

Found on 2026-09-20 by a BookForge session looking for what the model server was
doing during a 300 s clean-text timeout: `engine-qwen3.5-9b.log` and
`engine-thirdreich.log` were both minutes old and kilobytes long, because
`SubprocessEngine.start()` opened them `"wb"`. Reloading a voice is a start, and
reloading the voice is the one thing an operator does when an engine hangs — so
the act of investigating a hang destroyed the record of it. `crucible/workers.py`
had the same line for the same reason.

These tests are about RETENTION, not formatting: what has to hold is that run N's
bytes are still there after run N+1 begins, and that a reader can tell the two
apart. `log_tail()` is checked too, because the fix makes the file unbounded and
the old implementation read all of it to report forty lines.

THE MARKER IS SPLIT ON PURPOSE. `start()` writes the command line into the
header, so a token that appears verbatim in `command()` would be found in the log
even if the process never ran — the test would then pass against a spawn that
failed. `RUNOUT-<marker>` is assembled at runtime from `'RUN' + 'OUT-<marker>'`,
so the contiguous string exists only in the process's own stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from crucible.engines.base import LOG_TAIL_LINES, SubprocessEngine


class QuietEngine(SubprocessEngine):
    """An engine that writes one line naming its run and exits.

    Deliberately not `tests/fake_narrator.py`: nothing here is about readiness,
    and a process that exits at once makes "start it twice" a two-line test. No
    shell either — `command()` is spawned directly, so this runs the same on
    Windows, where a `sh -c` with a Windows interpreter path does not.
    """

    name = "fake-quiet"

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        marker = args[0] if args else "run"
        return [
            str(self._python),
            "-c",
            f"import sys; sys.stdout.write('RUN' + 'OUT-{marker}' + chr(10))",
        ]

    def announced_ready(self) -> str | None:
        return None

    def readiness_description(self) -> str:
        return "print a marker"


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    return directory


def _run(log_path: Path, weights: Path, marker: str) -> QuietEngine:
    """Start the double, wait for it, close the handle. Returns it for `log_tail`."""
    engine = QuietEngine(python=Path(sys.executable), log_path=log_path)
    engine.start(weights, "deathstalker", 0, [marker])
    process = engine._process  # noqa: SLF001 - the test owns this double
    assert process is not None
    assert process.wait(timeout=30) == 0, (
        "the double did not run; every assertion below would be about the "
        "command echoed in the header rather than about a real run"
    )
    return engine


def test_a_second_start_does_not_erase_the_first_run(
    tmp_path: Path, weights: Path
) -> None:
    """THE DEFECT. Reloading a hung voice used to delete the log of the hang."""
    log_path = tmp_path / "engine-probe.log"
    _run(log_path, weights, "first")._close_log()  # noqa: SLF001
    _run(log_path, weights, "second")._close_log()  # noqa: SLF001

    text = log_path.read_text(encoding="utf-8")
    assert "RUNOUT-first" in text, (
        "the first run's output was erased by the second start — this is the "
        "truncating open that made a hang impossible to investigate"
    )
    assert "RUNOUT-second" in text


def test_each_run_is_delimited_by_its_own_header(
    tmp_path: Path, weights: Path
) -> None:
    """Accumulating is only useful if a reader can tell the runs apart."""
    log_path = tmp_path / "engine-probe.log"
    _run(log_path, weights, "first")._close_log()  # noqa: SLF001
    _run(log_path, weights, "second")._close_log()  # noqa: SLF001

    text = log_path.read_text(encoding="utf-8")
    assert text.count("=== crucible fake-quiet engine, ") == 2
    # The first run opens the file; every later one is preceded by a blank line,
    # so the delimiter is visible to a person and not only to a parser.
    assert not text.startswith("\n")
    assert "\n\n=== crucible fake-quiet engine, " in text


def test_log_tail_reports_the_current_run_from_a_long_file(
    tmp_path: Path, weights: Path
) -> None:
    """`log_tail()` reads from the END, so accumulation stays cheap and correct.

    The file is padded past the read window first: an implementation that walked
    forward from byte 0, or that opened one window and gave up, would report the
    padding instead of the run.
    """
    log_path = tmp_path / "engine-probe.log"
    log_path.write_text("old noise\n" * 200_000, encoding="utf-8")
    assert log_path.stat().st_size > 64 * 1024

    engine = _run(log_path, weights, "third")
    tail = engine.log_tail()
    engine._close_log()  # noqa: SLF001

    assert "RUNOUT-third" in tail
    assert len(tail.splitlines()) <= LOG_TAIL_LINES
    # And the padding is still on disk — the tail is a READ, not a rotation.
    assert "old noise" in log_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# A tail must not cross a run boundary, and one caller ACTS on what it finds.
# --------------------------------------------------------------------------


def _log_with_two_runs(tmp_path: Path, first_body: str, second_body: str) -> Path:
    """A log holding one finished run and one live one, as appending produces."""
    log_path = tmp_path / "engine-llama.log"
    log_path.write_text(
        f"=== crucible llama-windows engine, 2026-09-20 01:00:00\n"
        f"=== llama-server.exe -m model.gguf\n"
        f"{first_body}\n"
        f"\n"
        f"=== crucible llama-windows engine, 2026-09-20 01:05:00\n"
        f"=== llama-server.exe -m model.gguf\n"
        f"{second_body}\n",
        encoding="utf-8",
    )
    return log_path


def test_a_dead_runs_fatal_line_does_not_refuse_the_next_start(
    tmp_path: Path,
) -> None:
    """THE REGRESSION APPENDING WOULD HAVE CAUSED, if the tail were not scoped.

    `LlamaServerEngine._fatal_in_log()` does not report a fatal line, it REFUSES
    the start on one. Before the tail was scoped to the current run, the log left
    behind by an engine that died of OOM would refuse every start after it — and
    the reload-after-a-hang case walks straight into that: hang, OOM in the log,
    reload, refused from then on with an error about a run that is long over.
    """
    from crucible.engines.llama_server import LlamaServerEngine

    log_path = _log_with_two_runs(
        tmp_path,
        first_body="ggml_cuda_host_malloc: CUDA error: out of memory",
        second_body="llama_model_loader: loaded meta data",
    )
    engine = LlamaServerEngine(python=Path("llama-server.exe"), log_path=log_path)

    assert engine._fatal_in_log() is None, (  # noqa: SLF001
        "the previous run's OOM was attributed to this one — an engine that is "
        "coming up fine would be refused because a dead run failed"
    )


def test_a_fatal_line_in_the_CURRENT_run_is_still_found(tmp_path: Path) -> None:
    """The scoping must not blind the check to the failure it exists for."""
    from crucible.engines.llama_server import LlamaServerEngine

    log_path = _log_with_two_runs(
        tmp_path,
        first_body="llama_model_loader: loaded meta data",
        second_body="ggml_cuda_host_malloc: CUDA error: out of memory",
    )
    engine = LlamaServerEngine(python=Path("llama-server.exe"), log_path=log_path)

    found = engine._fatal_in_log()  # noqa: SLF001
    assert found is not None
    assert "out of memory" in found[2]


def test_the_tail_stops_at_the_last_run_header(tmp_path: Path) -> None:
    """The unit underneath both: `tail_of_last_run` never crosses a boundary."""
    from crucible.logtail import tail_of_last_run

    log_path = _log_with_two_runs(
        tmp_path, first_body="OLD LINE", second_body="NEW LINE"
    )
    tail = tail_of_last_run(log_path, 40)
    assert "NEW LINE" in tail
    assert "OLD LINE" not in tail
    # The header of the run it DID read is included, so a reader knows which
    # run the lines belong to.
    assert "2026-09-20 01:05:00" in tail
    assert "2026-09-20 01:00:00" not in tail


def test_a_run_longer_than_the_window_still_reports_its_own_lines(
    tmp_path: Path,
) -> None:
    """The backwards walk widens rather than giving up at one window."""
    from crucible.logtail import TAIL_WINDOW_BYTES, tail_of_last_run

    log_path = _log_with_two_runs(
        tmp_path,
        first_body="OLD LINE",
        second_body="filler\n" * (TAIL_WINDOW_BYTES // 3) + "NEW LINE",
    )
    assert log_path.stat().st_size > TAIL_WINDOW_BYTES
    tail = tail_of_last_run(log_path, 40)
    assert "NEW LINE" in tail
    assert "OLD LINE" not in tail


def test_a_log_with_no_header_at_all_still_gives_a_tail(tmp_path: Path) -> None:
    """A log written by something else, or truncated by hand, is not an error."""
    from crucible.logtail import tail_of_last_run

    log_path = tmp_path / "engine-strange.log"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert tail_of_last_run(log_path, 2) == "two\nthree"
    assert tail_of_last_run(tmp_path / "absent.log", 2) == ""
