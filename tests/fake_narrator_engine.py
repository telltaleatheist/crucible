"""`NarratorEngine` with one line changed: the argv.

`tests/fake_engine.py` replaces vLLM with an HTTP server because the only things
Crucible needs from vLLM are a process and a route. narrator is not like that —
the protocol *is* the thing being built — so this does the opposite: it is the
**real** `crucible/engines/narrator.py`, with its real pipes, its real reader
thread, its real correlation and its real `stop()`, pointed at
`tests/fake_narrator.py` instead of `python -m narrator.serve`.

That is the whole substitution. `command()` is overridden and nothing else is, so
a test that goes green here has exercised every line of the engine that a render
on the PC will run, minus the part that needs 8.5 GB of weights and a card.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from crucible import residency as residency_module
from crucible.engines.narrator import NarratorEngine
from crucible.narratorvoices import VoicesDocument

FAKE_NARRATOR = Path(__file__).resolve().parent / "fake_narrator.py"

#: What the fake worker's machine claims to have, so the engine takes the
#: IN-PROCESS branch on every host the suite runs on.
#:
#: STATED RATHER THAN PROBED, and rather than `None`. `residency.load_voice`
#: passes the real figure only on `mlx-darwin`, so on the Windows and Linux
#: boxes this suite runs on it would arrive as `None` — and `None` on a
#: `higgs-v3` engine whose env starts no serving stack is a refusal by name
#: (the whole point of the argument). A constant here keeps every
#: residency test on the arm the fake worker actually imitates, and 64 GiB is
#: owens-mac-studio's, so the row it selects is the row the fleet runs.
FAKE_TOTAL_BYTES = 64 * 1024 * 1024 * 1024


class FakeNarratorEngine(NarratorEngine):
    """The real engine, started on the fake worker."""

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        # `sys.executable` rather than `self._python`: the env fixtures stamp a
        # non-executable placeholder there, because what they are testing is that
        # `env_status` accepts a stamped env, not that it can run one.
        return [
            sys.executable,
            str(FAKE_NARRATOR),
            "--engine",
            self.narrator_engine,
        ]


def install(monkeypatch: pytest.MonkeyPatch) -> list[FakeNarratorEngine]:
    """Point the residency's voice loads at the fake worker.

    Returns the list the built engines are appended to, so a test can reach the
    engine the server is holding — its log path, its pids — without going through
    the API for it.
    """
    built: list[FakeNarratorEngine] = []

    def build(
        narrator_engine: str,
        python: Path,
        log_path: Path,
        *,
        serving_stack: str | None,
        max_num_seqs: int | None,
        mem_fraction: float | None,
        context_length: int | None,
        voices: VoicesDocument | None,
        mlx_total_bytes: int | None,
    ) -> FakeNarratorEngine:
        # THE MACHINE'S MEMORY IS TAKEN AND REPLACED, not dropped: what
        # residency computed is right for the HOST, and this suite runs on
        # hosts that are not Macs. `FAKE_TOTAL_BYTES` puts the engine on the
        # in-process arm everywhere, which is the arm the fake worker imitates.
        #
        # THE SERVER'S OWN CONFIGURATION IS TAKEN AND DROPPED, deliberately.
        # The real `build_voice_engine` would read the interpreter's prefix
        # off disk (`higgs_env_prefix`) and emit HIGGS_STACK / HIGGS_ENV /
        # HIGGS_MAX_NUM_SEQS; the env fixtures stamp a non-executable
        # placeholder interpreter and the fake worker starts no vllm-omni, so
        # there is nothing here for those three to configure. What they DO
        # configure is asserted directly in `tests/test_narrator_engine.py`
        # against a real venv-shaped directory. The signature is mirrored so a
        # change to it fails here rather than silently passing a default.
        #
        # THE VOICES DOCUMENT IS PASSED THROUGH, because the fake worker reads
        # it exactly as narrator does: under `--engine higgs-v3` it refuses a
        # `modelDir` by name and resolves the voice in NARRATOR_HIGGS_VOICES,
        # so a residency that stopped writing the document would fail here.
        engine = FakeNarratorEngine(
            narrator_engine=narrator_engine,
            python=python,
            log_path=log_path,
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
            voices=voices,
            mlx_total_bytes=FAKE_TOTAL_BYTES,
        )
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_voice_engine", build)
    return built


def steer(monkeypatch: pytest.MonkeyPatch, **variables: Any) -> None:
    """Set the fake worker's `CRUCIBLE_FAKE_*` switches for one test.

    They reach the subprocess through `os.environ`, which is what
    `SubprocessEngine.start()` hands it, so a `monkeypatch.setenv` here is read
    by the spawned worker.
    """
    for name, value in variables.items():
        monkeypatch.setenv(f"CRUCIBLE_FAKE_{name.upper()}", str(value))


__all__: list[str] = ["FAKE_NARRATOR", "FakeNarratorEngine", "install", "steer"]
