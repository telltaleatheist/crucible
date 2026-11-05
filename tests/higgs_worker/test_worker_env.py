from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

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


def test_ci_installs_what_the_higgs_worker_tests_need() -> None:
    """Without this a dropped package would turn these tests into one quiet skip on CI."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    extra = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]["test"]
    named = {requirement.split(">")[0].split("=")[0].strip() for requirement in extra}
    assert set(WORKER_MODULES) <= named
