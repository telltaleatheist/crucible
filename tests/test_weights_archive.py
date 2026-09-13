"""`weights.pull_archive` — one file out of a shared repo, verified and unpacked.

The second shape of pull, and it exists because of how a real repo is laid out
rather than because somebody wanted a second shape: every RVC model Owen has
published is a `.tar.gz` under `rvc/` in ONE HuggingFace repo alongside six others
and the XTTS weights, so a snapshot download would fetch about 800 MB to get at
80 (`crucible/rvcmodels.py`).

Nothing here reaches the network. `hf_hub_download` is replaced with a function
that writes a tarball this test built, which is what the real one does and all
this module needs from it.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from crucible import weights
from crucible.config import load_config, write_config
from crucible.rvcmodels import load_rvc_manifest

from .conftest import FAKE_BACKEND

MODEL = "deathstalker-rvc-v1"


@pytest.fixture
def config(home: Path):
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="t",
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=False,
        enable_llm=False,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=True,
        desktop_allowance_bytes=0,
    )
    return load_config(home)


def build_archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """A gzipped tar holding exactly these paths."""
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return path


def build_link_archive(tmp_path: Path) -> Path:
    path = tmp_path / "link.tar.gz"
    with tarfile.open(path, "w:gz") as handle:
        info = tarfile.TarInfo("rvc/voice_models/x/escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        handle.addfile(info)
    return path


def serve(monkeypatch: pytest.MonkeyPatch, archive: Path) -> list[dict]:
    """Replace `hf_hub_download` with one that copies `archive` into place."""
    import huggingface_hub

    calls: list[dict] = []

    def download(*, repo_id, filename, revision, local_dir, token=None, **_):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "token": token,
            }
        )
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read_bytes())
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


MEMBERS = {
    "rvc/voice_models/deathstalker_rvc_v1/deathstalker_rvc_v1.pth": b"checkpoint",
    "rvc/voice_models/deathstalker_rvc_v1/deathstalker_rvc_v1.index": b"index",
}


# ------------------------------------------------------------------ it works


def test_one_file_is_fetched_not_the_whole_repo(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_rvc_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    archive = build_archive(tmp_path, MEMBERS)
    calls = serve(monkeypatch, archive)
    # The manifest pins the digest of the REAL archive; this test built its own,
    # so the pin is swapped for that one. Everything else about the spec — the
    # repo, the revision, the path inside it — is the shipped manifest's.
    spec = replace(spec, archive_sha256=weights.sha256_of(archive))

    result = weights.pull_archive(config, manifest, spec)
    assert len(calls) == 1
    assert calls[0]["filename"] == "rvc/deathstalker_rvc_v1.tar.gz"
    assert calls[0]["revision"] == spec.revision
    # And what landed is a whole URVC_MODELS_DIR root, which is what
    # `jobs/rvc._stage_models` goes looking for.
    model_dir = result.path / "rvc" / "voice_models" / "deathstalker_rvc_v1"
    assert (model_dir / "deathstalker_rvc_v1.pth").read_bytes() == b"checkpoint"
    # The staging directory does not survive the unpack.
    assert not (result.path / ".crucible-archive").exists()


def test_the_stamp_is_the_same_shape_a_snapshot_pull_writes(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what lets `installed` read either without knowing which ran."""
    manifest = load_rvc_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    archive = build_archive(tmp_path, MEMBERS)
    serve(monkeypatch, archive)
    digest = weights.sha256_of(archive)
    spec = replace(spec, archive_sha256=digest)

    weights.pull_archive(config, manifest, spec)
    stamp = json.loads(
        (
            weights.weights_dir(config, "rvc", MODEL, FAKE_BACKEND.kind)
            / weights.STAMP_NAME
        ).read_text(encoding="utf-8")
    )
    assert stamp["hf_repo"] == spec.hf_repo
    assert stamp["revision"] == spec.revision
    assert stamp["archive_sha256"] == digest
    assert weights.installed(config, manifest, spec) is not None


def test_a_second_pull_does_not_refetch(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_rvc_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    archive = build_archive(tmp_path, MEMBERS)
    calls = serve(monkeypatch, archive)
    spec = replace(spec, archive_sha256=weights.sha256_of(archive))
    weights.pull_archive(config, manifest, spec)
    weights.pull_archive(config, manifest, spec)
    assert len(calls) == 1


# ----------------------------------------------------------------- it refuses


def test_a_digest_mismatch_refuses_and_unpacks_nothing(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated or substituted checkpoint converts a whole book into
    something subtly wrong and says nothing about it."""
    manifest = load_rvc_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, build_archive(tmp_path, MEMBERS))

    with pytest.raises(weights.WeightsError) as caught:
        weights.pull_archive(config, manifest, spec)
    message = str(caught.value)
    assert "Nothing was unpacked" in message
    assert spec.archive_sha256 in message
    target = weights.weights_dir(config, "rvc", MODEL, FAKE_BACKEND.kind)
    assert not (target / "rvc").exists()
    assert not (target / weights.STAMP_NAME).exists()
    assert weights.installed(config, manifest, spec) is None


def test_a_member_that_escapes_the_target_is_refused(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_rvc_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    archive = build_archive(tmp_path, {"../../escaped.pth": b"nope"})
    serve(monkeypatch, archive)
    spec = replace(spec, archive_sha256=weights.sha256_of(archive))
    with pytest.raises(weights.WeightsError) as caught:
        weights.pull_archive(config, manifest, spec)
    assert "written outside" in str(caught.value)


def test_a_link_member_is_refused(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weights archive is files; a link is a way to write somewhere else."""
    manifest = load_rvc_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    archive = build_link_archive(tmp_path)
    serve(monkeypatch, archive)
    spec = replace(spec, archive_sha256=weights.sha256_of(archive))
    with pytest.raises(weights.WeightsError) as caught:
        weights.pull_archive(config, manifest, spec)
    assert "contains a link" in str(caught.value)


def test_a_refusal_names_the_command_for_this_family(config) -> None:
    """`crucible models pull sigma` would send its reader to a command that
    tells them there is no such model."""
    manifest = load_rvc_manifest("sigma")
    spec = manifest.spec(FAKE_BACKEND.kind)
    with pytest.raises(weights.WeightsError) as caught:
        weights.require_installed(config, manifest, spec)
    assert "crucible rvc pull sigma" in str(caught.value)
