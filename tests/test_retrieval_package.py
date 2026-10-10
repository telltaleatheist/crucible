"""`crucible install retrieval`: the embed and rerank verbs' optional package. It pulls the
models only where they can run and fit, and writes `[packages] retrieval = true`; where it
is not installed nothing pulls them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import cli, weights
from crucible.backend import Backend, Gpu
from crucible.config import ConfigError, load_config, rewrite_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


def _pulls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    pulled: list[str] = []

    def pull(config: Any, manifest: Any, spec: Any, **_: Any) -> Any:
        pulled.append(manifest.id)
        return SimpleNamespace(bytes=16_000_000_000, path=Path("/w") / manifest.id)

    monkeypatch.setattr(weights, "pull", pull)
    return pulled


def test_install_retrieval_pulls_both_models_and_turns_the_package_on(
    home: Path, viable: None, fake_env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    pulled = _pulls(monkeypatch)
    stepped: list[tuple[frozenset[str], tuple[str, ...]]] = []
    monkeypatch.setattr(
        cli.install, "_capability_step",
        lambda config, backend, *types: stepped.append((config.packages, types)) or 0,
    )
    assert cli.main(["install", "retrieval"]) == 0
    assert pulled == ["qwen3-embedding-8b", "qwen3-reranker-8b"]
    assert stepped == [(frozenset({"retrieval"}), ("llm",))], "decided again with the package on"
    assert load_config(home).packages == frozenset({"retrieval"})
    out = capsys.readouterr().out
    assert "package: retrieval" in out and "[packages] retrieval = true" in out


def test_install_retrieval_refuses_without_the_llm_engine_and_pulls_nothing(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    pulled = _pulls(monkeypatch)
    assert cli.main(["install", "retrieval"]) == 1
    assert pulled == []
    err = capsys.readouterr().err
    assert "crucible install llm" in err and "Nothing was pulled" in err
    assert load_config(home).packages == frozenset()


def test_install_retrieval_refuses_a_card_that_cannot_hold_it(
    home: Path, fake_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    laptop = Backend(kind="cuda-linux", platform="linux", arch="x86_64",
                     gpu=Gpu(vendor="nvidia", name="RTX 3070 Laptop", vram_bytes=8 * 1024 ** 3),
                     detail="test")
    monkeypatch.setattr(cli.common, "detect_backend", lambda: laptop)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    pulled = _pulls(monkeypatch)
    assert cli.main(["install", "retrieval"]) == 1
    assert pulled == []
    assert "does not hold the package" in capsys.readouterr().err


def test_the_packages_table_names_only_known_packages_as_booleans(home: Path, viable: None) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    config = load_config(home)
    rewrite_config(config, unowned={"packages": {"retrieval": False}})
    assert load_config(home).packages == frozenset()
    rewrite_config(load_config(home), unowned={"packages": {"retrieval": "yes"}})
    with pytest.raises(ConfigError) as caught:
        load_config(home)
    assert "[packages] retrieval" in str(caught.value)
    rewrite_config(config, unowned={"packages": {"retrieval": None, "voices": True}})
    with pytest.raises(ConfigError) as caught:
        load_config(home)
    assert "this build's packages are ['retrieval']" in str(caught.value)


def test_a_rewrite_keeps_the_packages_table(home: Path, viable: None) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    rewrite_config(load_config(home), unowned={"packages": {"retrieval": True}})
    rewrite_config(load_config(home), flags={"enable_llm": True})
    assert load_config(home).packages == frozenset({"retrieval"})


def test_the_mac_package_is_qwen_s_own_bf16_weights() -> None:
    from crucible import packages
    from crucible.manifests import load_all_manifests

    models = packages.models_of("retrieval", load_all_manifests(), FAKE_MAC_BACKEND.kind)
    assert [m.spec(FAKE_MAC_BACKEND.kind).hf_repo for m in models] == [
        "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Reranker-8B",
    ]
    assert packages.models_of("retrieval", load_all_manifests(), "llama-windows") == []
