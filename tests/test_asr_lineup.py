"""Owen's asr lineup ruling of 2026-09-24, held where each half of it lives.

*"the transcription job offers EXACTLY three models, which the caller picks:
Whisper large-v3-turbo, Qwen3-ASR-1.7B, and Whisper tiny"* — every other whisper
size removed, and each of the three ONE id on both backends.

- The capability class's candidates are exactly those three on cuda-linux and
  on mlx-darwin, best-first QWEN, TURBO, TINY (`crucible/capability.py` says
  why that order).
- Weights pulled under a RENAMED id are moved into the new id's folder at
  server start, and only when their stamp names the new block's exact pin
  (`jobs/asr.adopt_renamed_asr_weights`, `weights.adopt_renamed`).
- Whatever nothing owns is REPORTED by `catalog.stranded_weights`, with the
  retirement sentence where there is one, and never deleted.
- A transcript's provenance names the engine and the repo, because one id is
  now two conversions (`AsrJobType.model_provenance`).

The refusals a client sees (`unknown_model` naming the replacement) are in
`tests/test_asr_api.py`, beside the rest of the asr door's refusals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible import capability, catalog, weights
from crucible.asrmodels import (
    ASR_LINEUP,
    RENAMED_ASR_IDS,
    RETIRED_ASR_IDS,
    load_asr_manifest,
)
from crucible.config import Config, load_config
from crucible.jobs import asr as asr_job

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

BEST_FIRST = ["qwen3-asr-1.7b", "qwen3-asr-0.6b", "whisper-large-v3-turbo", "whisper-tiny"]

#: The Mac also offers each Qwen's MLX port, right after it: the official engine
#: first (Owen, 2026-09-24: *"if i want speed i can get it via mlx"*).
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
    """A finished pull under `subject_id`, as `weights.pull` writes one."""
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


def _pinned(home: Path, old_id: str, backend: str) -> Path:
    """`old_id`'s pull, stamped at exactly the pin its NEW id declares."""
    spec = load_asr_manifest(RENAMED_ASR_IDS[old_id]).spec(backend)
    return _stamp(home, old_id, backend, spec.hf_repo, spec.revision)


# ------------------------------------------------------------------ the three


@pytest.mark.parametrize("backend", ["cuda-linux", "mlx-darwin"])
def test_the_asr_class_offers_exactly_the_lineup_best_first(backend: str) -> None:
    """Qwen, turbo, tiny on both machines as the same ids; the Mac adds the
    MLX port second."""
    candidates = capability.BY_NAME["asr"].candidates(backend)
    assert [c.id for c in candidates] == BEST_FIRST_FOR[backend]
    assert set(BEST_FIRST_FOR["mlx-darwin"]) == ASR_LINEUP


def test_the_mac_runs_the_official_qwen_package_and_the_port_under_its_own_id() -> None:
    """The first live runs (2026-09-24): the official package is the default
    on the Mac, and the faster MLX port is its own id, `-mlx`."""
    assert load_asr_manifest("qwen3-asr-1.7b").spec("mlx-darwin").engine == "qwen-asr"
    port = load_asr_manifest("qwen3-asr-1.7b-mlx")
    assert port.spec("mlx-darwin").engine == "mlx-audio"
    assert not port.supports("cuda-linux")
    assert asr_job.ENV_FOR_ENGINE["qwen-asr"] == "align"


def test_no_removed_id_is_declared_anywhere_in_the_build() -> None:
    """Not as a model, and so not as a module subject, a catalog row or a
    `crucible models pull` target: the old ids are gone, not aliases."""
    declared = set(catalog.declared_ids()["model"])
    assert not declared & set(RENAMED_ASR_IDS)
    assert not declared & RETIRED_ASR_IDS
    assert ASR_LINEUP <= declared


def test_every_rename_lands_on_an_id_that_exists_on_that_backend() -> None:
    """A rename table naming a manifest the build lacks would refuse every
    server start (`adopt_renamed_asr_weights`); this keeps it red here first."""
    for old_id, new_id in RENAMED_ASR_IDS.items():
        backend = "mlx-darwin" if old_id.startswith("mlx-") else "cuda-linux"
        assert load_asr_manifest(new_id).supports(backend)


# ------------------------------------------------------------ the migration


def test_a_renamed_pull_is_moved_to_the_new_id_and_reads_as_installed(
    config: Config, home: Path
) -> None:
    _pinned(home, "faster-whisper-large-v3-turbo", "cuda-linux")
    _pinned(home, "mlx-whisper-tiny", "mlx-darwin")

    lines = asr_job.adopt_renamed_asr_weights(config)

    assert len(lines) == 2 and all(line.startswith("moved ") for line in lines)
    assert not (home / "models" / "faster-whisper-large-v3-turbo").exists()
    assert not (home / "models" / "mlx-whisper-tiny").exists()
    for model_id, backend in (
        ("whisper-large-v3-turbo", "cuda-linux"),
        ("whisper-tiny", "mlx-darwin"),
    ):
        manifest = load_asr_manifest(model_id)
        found = weights.installed(config, manifest, manifest.spec(backend))
        assert found is not None, (model_id, backend)
        assert (found.path / "model.bin").is_file()
        stamp = json.loads(
            (found.path / weights.STAMP_NAME).read_text(encoding="utf-8")
        )
        assert stamp["id"] == model_id
        assert stamp["renamed_from"] in RENAMED_ASR_IDS
    # Idempotent: nothing is left under an old id to find.
    assert asr_job.adopt_renamed_asr_weights(config) == []
    assert catalog.stranded_weights(config) == []


def test_bytes_at_another_pin_are_left_and_reported_not_moved(
    config: Config, home: Path
) -> None:
    """Moving other bytes under the new id would be the silent substitution
    `installed()` exists to refuse."""
    left = _stamp(
        home,
        "faster-whisper-tiny",
        "cuda-linux",
        "Systran/faster-whisper-tiny",
        "0" * 40,
    )
    lines = asr_job.adopt_renamed_asr_weights(config)
    assert len(lines) == 1
    assert lines[0].startswith(f"left {left}: stamped Systran/faster-whisper-tiny@000000000000")
    assert left.is_dir()
    tiny = load_asr_manifest("whisper-tiny")
    assert weights.installed(config, tiny, tiny.spec("cuda-linux")) is None

    [row] = catalog.stranded_weights(config)
    assert row["id"] == "faster-whisper-tiny"
    assert row["backend"] == "cuda-linux"
    assert row["path"] == str(left)
    assert row["bytes"] > 1000  # the model file and its stamp
    assert "renamed 'whisper-tiny' on 2026-09-24" in row["note"]


def test_a_new_id_already_pulled_leaves_the_old_copy_for_the_operator(
    config: Config, home: Path
) -> None:
    old = _pinned(home, "mlx-whisper-large-v3-turbo", "mlx-darwin")
    spec = load_asr_manifest("whisper-large-v3-turbo").spec("mlx-darwin")
    _stamp(home, "whisper-large-v3-turbo", "mlx-darwin", spec.hf_repo, spec.revision)

    [line] = asr_job.adopt_renamed_asr_weights(config)
    assert line.startswith(f"left {old}:")
    assert "already exists" in line
    assert [row["id"] for row in catalog.stranded_weights(config)] == [
        "mlx-whisper-large-v3-turbo"
    ]


def test_a_retired_size_is_reported_with_its_sentence_and_nothing_else_is(
    config: Config, home: Path
) -> None:
    """The reconciler names what nothing owns — and only that: a declared id's
    folder, and a folder not named for a backend, are none of its business."""
    retired = _stamp(
        home, "faster-whisper-large-v3", "cuda-linux", "Systran/faster-whisper-large-v3", "e" * 40
    )
    spec = load_asr_manifest("whisper-tiny").spec("cuda-linux")
    _stamp(home, "whisper-tiny", "cuda-linux", spec.hf_repo, spec.revision)
    (home / "models" / "whisper-tiny" / "notes").mkdir()

    assert asr_job.adopt_renamed_asr_weights(config) == []
    rows = catalog.stranded_weights(config)
    assert [(row["id"], row["backend"]) for row in rows] == [
        ("faster-whisper-large-v3", "cuda-linux")
    ]
    assert rows[0]["path"] == str(retired)
    assert "'faster-whisper-large-v3' was retired on 2026-09-24" in rows[0]["note"]


def test_a_model_this_build_does_not_declare_on_a_backend_is_stranded_there(
    config: Config, home: Path
) -> None:
    """Not asr-only: the Mac-only 8-bit 27B's cuda-linux folder (its block was
    removed on 2026-09-23) is the same kind of orphan, with no sentence."""
    _stamp(home, "qwen3.8-27b-8bit", "cuda-linux", "Qwen/whatever", "a" * 40)
    [row] = catalog.stranded_weights(config)
    assert (row["id"], row["backend"], row["note"]) == (
        "qwen3.8-27b-8bit",
        "cuda-linux",
        None,
    )


# ------------------------------------------------------------ the provenance


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
    """One id is two conversions since 2026-09-24, so the sidecar says which."""
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


# ------------------------------------------------------- one copy on disk


def test_an_mlx_id_stores_in_its_official_siblings_folder(config: Config) -> None:
    """2026-09-24: `qwen3-asr-0.6b-mlx` had pulled a second 1.88 GB copy of
    `qwen3-asr-0.6b`'s checkpoint. An alias's folder IS its base's."""
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
