from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import cli, llmconcurrency
from crucible.config import config_path, load_config
from crucible.engines import EngineError, with_concurrency
from crucible.errors import ConfigError

from .conftest import FAKE_MAC_BACKEND

MODEL = "qwen3.8-27b-8bit"


@pytest.fixture
def mac(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()


def _document(home: Path) -> dict:
    return tomllib.loads(config_path(home).read_text(encoding="utf-8"))


def test_the_setting_is_written_under_llm_and_keeps_the_token(home: Path, mac: None) -> None:
    before = load_config(home)
    llmconcurrency.set_concurrency(before, MODEL, 4)
    after = load_config(home)
    assert after.concurrency_for(MODEL) == 4
    assert after.token == before.token
    assert _document(home)["llm"] == {"concurrency": {MODEL: 4}}
    row = next(r for r in llmconcurrency.rows(after) if r["model"] == MODEL)
    assert row == {
        "model": MODEL,
        "display": row["display"],
        "manifest": 8,
        "set": 4,
        "running": None,
    }


def test_default_removes_it_and_the_manifest_width_is_no_setting(home: Path, mac: None) -> None:
    llmconcurrency.set_concurrency(load_config(home), MODEL, 4)
    llmconcurrency.set_concurrency(load_config(home), MODEL, None)
    assert "llm" not in _document(home)
    llmconcurrency.set_concurrency(load_config(home), MODEL, 8)
    assert load_config(home).concurrency_for(MODEL) is None


def test_above_the_manifest_or_below_one_is_refused(home: Path, mac: None) -> None:
    for width in (9, 0):
        with pytest.raises(ConfigError, match="concurrency_out_of_range"):
            llmconcurrency.set_concurrency(load_config(home), MODEL, width)
    with pytest.raises(ConfigError, match="concurrency_not_settable"):
        llmconcurrency.set_concurrency(load_config(home), "no-such-model", 2)


def test_a_later_install_keeps_it(home: Path, mac: None) -> None:
    from crucible.config import rewrite_config

    llmconcurrency.set_concurrency(load_config(home), MODEL, 4)
    rewrite_config(load_config(home))
    assert load_config(home).concurrency_for(MODEL) == 4


def test_the_engine_starts_with_the_set_width_in_place_of_the_manifests() -> None:
    spec = SimpleNamespace(engine="mlx-lm")
    args = ["--decode-concurrency", "8", "--prompt-concurrency", "1"]
    assert with_concurrency(spec, args, 4, MODEL) == [
        "--prompt-concurrency", "1", "--decode-concurrency", "4",
    ]
    with pytest.raises(EngineError, match="concurrency_above_manifest"):
        with_concurrency(spec, args, 9, MODEL)


def test_the_cli_sets_and_lists_it(
    home: Path, mac: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models", "concurrency", MODEL, "4"]) == 0
    assert "[llm.concurrency] qwen3.8-27b-8bit = 4" in capsys.readouterr().out
    assert cli.main(["models", "concurrency"]) == 0
    assert f"{MODEL}: 4 at once (set in config; manifest 8)" in capsys.readouterr().out
    assert cli.main(["models", "concurrency", MODEL, "default"]) == 0
    capsys.readouterr()
    assert load_config(home).concurrency_for(MODEL) is None
