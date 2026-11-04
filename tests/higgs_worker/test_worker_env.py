from __future__ import annotations

import importlib.util

import pytest

WORKER_MODULES = ("numpy", "num2words", "regex", "soundfile")


def test_this_interpreter_can_run_the_higgs_worker_tests() -> None:
    missing = [name for name in WORKER_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(
            f"the Higgs worker's tests were not collected: this interpreter lacks "
            f"{missing}. Run tests/higgs_worker under the tts env's python "
            "(~/.crucible/envs/tts-higgs-v3/bin/python -m pytest --noconftest "
            "tests/higgs_worker), which has them"
        )
