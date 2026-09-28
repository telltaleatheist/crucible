from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible import capabilityclasses, catalog, weights
from crucible.asrmodels import ASR_LINEUP, load_asr_manifest
from crucible.config import Config, load_config
from crucible.jobs import asr as asr_job

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

BEST_FIRST = ["qwen3-asr-1.7b", "qwen3-asr-0.6b", "whisper-large-v3-turbo", "whisper-tiny"]

BEST_FIRST_FOR = {
    "cuda-linux": BEST_FIRST,
    "mlx-darwin": [
        "qwen3-asr-1.7b", "qwen3-asr-1.7b-mlx",
        "qwen3-asr-0.6b", "qwen3-asr-0.6b-mlx",
        "whisper-large-v3-turbo", "whisper-tiny",
    ],
}


@pytest.fixture
def config(make_app: Callable[..., Any], home: Path) -> Config:
    make_app(enable_asr=True)
    return load_config(home)


def _stamp(
    home: Path, subject_id: str, backend: str, hf_repo: str, revision: str
) -> Path:
    directory = home / "models" / subject_id / backend
    directory.mkdir(parents=True)
    (directory / "model.bin").write_bytes(b"\0" * 1000)
    (directory / weights.STAMP_NAME).write_text(
        json.dumps(
            {
                "family": "models",
                "id": subject_id,
                "backend": backend,
                "hf_repo": hf_repo,
                "revision": revision,
                "files": [],
                "bytes": 1000,
                "seconds": 1.0,
                "pulled": "2026-09-20T02:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.mark.parametrize("backend", ["cuda-linux", "mlx-darwin"])
def test_the_asr_class_offers_exactly_the_lineup_best_first(backend: str) -> None:
    candidates = capabilityclasses.BY_NAME["asr"].candidates(backend)
    assert [c.id for c in candidates] == BEST_FIRST_FOR[backend]
    assert set(BEST_FIRST_FOR["mlx-darwin"]) == ASR_LINEUP


def test_the_mac_runs_the_official_qwen_package_and_the_port_under_its_own_id() -> None:
    assert load_asr_manifest("qwen3-asr-1.7b").spec("mlx-darwin").engine == "qwen-asr"
    port = load_asr_manifest("qwen3-asr-1.7b-mlx")
    assert port.spec("mlx-darwin").engine == "mlx-audio"
    assert not port.supports("cuda-linux")
    assert asr_job.ENV_FOR_ENGINE["qwen-asr"] == "align"


def test_the_lineup_is_declared_in_the_build() -> None:
    declared = set(catalog.declared_ids()["model"])
    assert ASR_LINEUP <= declared


def test_a_retired_size_is_reported_and_nothing_else_is(
    config: Config, home: Path
) -> None:
    retired = _stamp(
        home, "faster-whisper-large-v3", "cuda-linux", "Systran/faster-whisper-large-v3", "e" * 40
    )
    spec = load_asr_manifest("whisper-tiny").spec("cuda-linux")
    _stamp(home, "whisper-tiny", "cuda-linux", spec.hf_repo, spec.revision)
    (home / "models" / "whisper-tiny" / "notes").mkdir()

    rows = catalog.stranded_weights(config)
    assert [(row["id"], row["backend"]) for row in rows] == [
        ("faster-whisper-large-v3", "cuda-linux")
    ]
    assert rows[0]["path"] == str(retired)


def test_a_model_this_build_does_not_declare_on_a_backend_is_stranded_there(
    config: Config, home: Path
) -> None:
    _stamp(home, "qwen3.8-27b-8bit", "cuda-linux", "Qwen/whatever", "a" * 40)
    [row] = catalog.stranded_weights(config)
    assert (row["id"], row["backend"]) == ("qwen3.8-27b-8bit", "cuda-linux")


@pytest.mark.parametrize(
    "backend, engine, hf_repo",
    [
        (FAKE_BACKEND, "faster-whisper", "dropbox-dash/faster-whisper-large-v3-turbo"),
        (FAKE_MAC_BACKEND, "mlx-whisper", "mlx-community/whisper-large-v3-turbo"),
    ],
    ids=["cuda-linux", "mlx-darwin"],
)
def test_the_provenance_names_the_conversion_that_made_the_transcript(
    make_app: Callable[..., Any], home: Path, backend: Any, engine: str, hf_repo: str
) -> None:
    make_app(enable_asr=True, backend=backend)
    job_type = asr_job.AsrJobType(load_config(home), backend, frozenset)
    spec = load_asr_manifest("whisper-large-v3-turbo").spec(backend.kind)
    assert job_type.model_provenance("whisper-large-v3-turbo") == {
        "id": "whisper-large-v3-turbo",
        "revision": spec.revision,
        "fingerprint": f"whisper-large-v3-turbo@{spec.revision}",
        "engine": engine,
        "hf_repo": hf_repo,
    }


def test_an_mlx_id_stores_in_its_official_siblings_folder(config: Config) -> None:
    for alias_id, base_id in (
        ("qwen3-asr-1.7b-mlx", "qwen3-asr-1.7b"),
        ("qwen3-asr-0.6b-mlx", "qwen3-asr-0.6b"),
    ):
        alias = load_asr_manifest(alias_id)
        base = load_asr_manifest(base_id)
        assert alias.weights_of == base_id and alias.weights_base == base
        assert weights.subject_dir(config, alias, "mlx-darwin") == weights.subject_dir(
            config, base, "mlx-darwin"
        )


def test_an_asr_alias_pinned_to_other_bytes_is_refused(tmp_path: Path) -> None:
    from crucible.asrmodels import AsrManifestError

    source = Path(load_asr_manifest("qwen3-asr-0.6b").path).parent
    (tmp_path / "qwen3-asr-0.6b.toml").write_text(
        (source / "qwen3-asr-0.6b.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    alias = (source / "qwen3-asr-0.6b-mlx.toml").read_text(encoding="utf-8")
    (tmp_path / "qwen3-asr-0.6b-mlx.toml").write_text(
        alias.replace("5eb144179a02acc5e5ba31e748d22b0cf3e303b0", "0" * 40), encoding="utf-8"
    )
    with pytest.raises(AsrManifestError, match="weights_of_pin_mismatch"):
        load_asr_manifest("qwen3-asr-0.6b-mlx", tmp_path)


def test_a_folder_left_under_an_alias_id_is_reported_stranded(
    config: Config, home: Path
) -> None:
    spec = load_asr_manifest("qwen3-asr-0.6b").spec("mlx-darwin")
    left = _stamp(home, "qwen3-asr-0.6b-mlx", "mlx-darwin", spec.hf_repo, spec.revision)
    rows = catalog.stranded_weights(config)
    assert [(r["id"], r["path"]) for r in rows] == [("qwen3-asr-0.6b-mlx", str(left))]
