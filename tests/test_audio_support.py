from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible import audioweights, catalog, installonsubmit, jobenv, verdict, weights
from crucible.audiomodels import (
    AudioManifestError,
    load_all_audio_manifests,
    load_audio_manifest,
    parse_audio_manifest,
)
from crucible.capabilityclasses import BY_NAME
from crucible.config import load_config
from crucible.desktop_app.screens import JOB_TYPE_WORDS
from crucible.memorybudget import GIB
from crucible.tasks.validate import env_installed

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, configure_box, stamp_env

DIGEST = "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af"


def test_the_three_models_are_declared_with_what_they_make_and_where() -> None:
    manifests = load_all_audio_manifests()
    assert {m.id: m.kind for m in manifests.values()} == {
        "stable-audio-3-small-sfx": "sfx",
        "stable-audio-3-medium": "music",
        "yue2-3b": "song",
    }
    assert sorted(manifests["yue2-3b"].backends) == ["cuda-linux"]
    for model in ("stable-audio-3-small-sfx", "stable-audio-3-medium"):
        assert sorted(manifests[model].backends) == ["cuda-linux", "mlx-darwin"]
        assert manifests[model].spec("mlx-darwin").device == "mps"
        assert all(spec.gated for spec in manifests[model].backends.values())
    song = manifests["yue2-3b"].spec("cuda-linux")
    assert not song.gated and [c.hf_repo for c in song.companions] == ["m-a-p/YuE2-Vae"]
    for manifest in manifests.values():
        for spec in manifest.backends.values():
            assert spec.memory_basis == "declared" and spec.memory_note


def test_medium_fits_the_pc_card_with_its_desktop_allowance() -> None:
    spec = load_audio_manifest("stable-audio-3-medium").spec("cuda-linux")
    assert spec.memory_bytes_estimate <= 24 * GIB - 3 * GIB


BASE = """
[model]
id = "x"
family = "f"
display = "X"
kind = "sfx"
licence = "l"
licence_url = "https://example.invalid"
commercial_use = "no"

[backends.cuda-linux]
engine = "stable-audio-3"
hf_repo = "o/x"
revision = "ae12755283df9d62ca39a9b050a39a0b607b8c20"
gated = false
dtype = "float16"
memory_bytes_estimate = 1
memory_basis = "declared"
memory_note = "n"
files = ["model.safetensors"]
sample_rate = 44100
channels = 2
max_duration_s = 10
takes = ["duration_s"]
default_duration_s = 5
"""


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (('kind = "sfx"', 'kind = "speech"'), "is not one of"),
        (('engine = "stable-audio-3"', 'engine = "yue3"'), "does not run on cuda-linux"),
        (("default_duration_s = 5", "default_duration_s = 11"), "outside 1..10"),
        (('takes = ["duration_s"]', 'takes = ["duration_s", "steps"]'), "default_steps goes with 'steps'"),
        (('takes = ["duration_s"]', 'takes = ["duration_s", "tempo"]'), "the optional audio params"),
        (('gated = false', 'gated = "maybe"'), "gated must be bool"),
    ],
)
def test_a_manifest_that_says_the_wrong_thing_is_refused_by_name(
    change: tuple[str, str], words: str
) -> None:
    text = BASE.replace(*change)
    with pytest.raises(AudioManifestError) as caught:
        parse_audio_manifest(text, Path("x.toml"), "x")
    assert words in str(caught.value)


def test_a_companion_file_must_be_pinned_by_its_hash() -> None:
    text = BASE + """
[[backends.cuda-linux.companions]]
name = "vae"
hf_repo = "o/vae"
revision = "152733a19ad43aa67e367f9b5503ef8075bb5126"

[[backends.cuda-linux.companions.files]]
source = "model.safetensors"
target = "model.safetensors"
sha256 = "not-a-hash"
bytes = 1
"""
    with pytest.raises(AudioManifestError) as caught:
        parse_audio_manifest(text, Path("x.toml"), "x")
    assert "sha256 must be 64" in str(caught.value)


def test_each_backend_builds_one_env_per_engine_from_its_own_recipe() -> None:
    assert [spec.key for spec in jobenv.audio_envs("cuda-linux")] == ["audio-stable-audio-3", "audio-yue2"]
    assert [spec.key for spec in jobenv.audio_envs("mlx-darwin")] == ["audio-stable-audio-3"]
    for backend_kind in ("cuda-linux", "mlx-darwin"):
        for spec in jobenv.audio_envs(backend_kind):
            recipe = jobenv.recipe_for(spec)
            assert jobenv.recipe_archive_bytes(recipe) > 0
            pins = jobenv.recipe_pins(recipe)
            assert "torch" in pins and "soundfile" in pins
            assert jobenv.SMOKE_IMPORT[spec.key][backend_kind]
    cuda = jobenv.recipe_direct_references(jobenv.recipe_for(jobenv.audio_env("stable-audio-3", "cuda-linux")))
    assert cuda["flash-attn"] == DIGEST
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv.audio_env("yue2", "mlx-darwin")
    assert "the audio engines there are ['stable-audio-3']" in str(caught.value)


def test_a_wheel_url_is_pinned_by_its_digest_and_read_back_off_pips_record(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "recipes"
    (root / "audio").mkdir(parents=True)
    recipe = root / "audio" / "stable-audio-3-cuda-linux.txt"
    recipe.write_text(f"flash-attn @ https://example.invalid/fa.whl#sha256={DIGEST}\n", encoding="utf-8")
    monkeypatch.setenv(jobenv.RECIPES_DIR_ENV, str(root))
    assert jobenv.recipe_direct_references(recipe) == {"flash-attn": DIGEST}
    spec = jobenv.audio_env("stable-audio-3", "cuda-linux")
    dist = jobenv.env_dir(home, spec) / "lib" / "python3.11" / "site-packages" / "flash_attn-2.8.3.dist-info"
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps({"url": "https://example.invalid/fa.whl", "archive_info": {"hashes": {"sha256": DIGEST}}}),
        encoding="utf-8",
    )
    assert jobenv.installed_direct_references(home, spec) == {"flash-attn": DIGEST}


def test_the_audio_env_counts_as_installed_only_when_every_engine_env_is(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    first, second = jobenv.audio_envs("cuda-linux")
    stamp_env(home, first, "cuda-linux", monkeypatch)
    assert env_installed(config, FAKE_BACKEND, "audio", None) is False
    stamp_env(home, second, "cuda-linux", monkeypatch)
    assert env_installed(config, FAKE_BACKEND, "audio", None) is True


def test_install_on_submit_names_the_audio_installer_and_sums_its_recipes() -> None:
    assert installonsubmit._installer_of("audio", "/home/x/envs/audio-yue2") == "audio"
    total = sum(
        jobenv.recipe_archive_bytes(jobenv.recipe_for(spec)) for spec in jobenv.audio_envs("cuda-linux")
    )
    assert installonsubmit._env_bytes("audio", None, "cuda-linux") == total


@pytest.mark.parametrize(
    ("name", "backend_kind", "enabled", "summary"),
    [
        ("sfx", "cuda-linux", True, "can make sound effects, using stable-audio-3-small-sfx"),
        ("music", "cuda-linux", True, "can make music, using stable-audio-3-medium"),
        ("song", "cuda-linux", True, "can make songs with vocals, using yue2-3b"),
        ("sfx", "mlx-darwin", True, "can make sound effects, using stable-audio-3-small-sfx"),
        ("music", "mlx-darwin", True, "can make music, using stable-audio-3-medium"),
        ("song", "mlx-darwin", False, "cannot make songs with vocals"),
    ],
)
def test_the_capability_rows_say_what_this_host_can_make_and_with_what(
    name: str, backend_kind: str, enabled: bool, summary: str
) -> None:
    total = 24 * GIB if backend_kind == "cuda-linux" else 64 * GIB
    decided = verdict.decide(
        BY_NAME[name], backend_kind, total_bytes=total, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia" if backend_kind == "cuda-linux" else "apple", chosen=None,
    )
    assert decided.enabled is enabled, decided.reason
    assert decided.summary.startswith(summary), decided.summary
    assert decided.reason


def test_a_small_card_keeps_sound_effects_and_loses_songs() -> None:
    kwargs: dict[str, Any] = dict(total_bytes=12 * GIB, desktop_allowance_bytes=3 * GIB, gpu_vendor="nvidia", chosen=None)
    assert verdict.decide(BY_NAME["sfx"], "cuda-linux", **kwargs).enabled is True
    song = verdict.decide(BY_NAME["song"], "cuda-linux", **kwargs)
    assert song.enabled is False and song.shortfall_bytes > 0


def _stamp_main(config: Any, manifest: Any, spec: Any) -> Path:
    directory = weights.subject_dir(config, manifest, spec.backend)
    for name in spec.files:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"w")
    (directory / weights.STAMP_NAME).write_text(
        json.dumps({"hf_repo": spec.hf_repo, "revision": spec.revision, "bytes": 7, "pulled": "now"}),
        encoding="utf-8",
    )
    return directory


def test_a_song_model_is_installed_only_with_its_decoder_and_the_pull_fetches_both(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    manifest = load_audio_manifest("yue2-3b")
    spec = manifest.spec("cuda-linux")
    calls: list[str] = []

    def main_pull(config: Any, manifest: Any, spec: Any, **_: Any) -> weights.InstalledWeights:
        calls.append(spec.hf_repo)
        _stamp_main(config, manifest, spec)
        return weights.installed(config, manifest, spec)

    def files_pull(config: Any, *, hf_repo: str, revision: str, files: Any, target_root: Path, **_: Any) -> Any:
        calls.append(hf_repo)
        target_root.mkdir(parents=True, exist_ok=True)
        for entry in files:
            (target_root / entry.target).write_bytes(b"v")
        (target_root / weights.STAMP_NAME).write_text(
            json.dumps({"hf_repo": hf_repo, "revision": revision, "bytes": 1, "pulled": "now",
                        "files": [{"target": e.target} for e in files]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(weights, "pull", main_pull)
    monkeypatch.setattr(weights, "pull_files", files_pull)
    _stamp_main(config, manifest, spec)
    assert audioweights.installed(config, manifest, spec) is None
    with pytest.raises(weights.WeightsError) as caught:
        audioweights.require_installed(config, manifest, spec)
    assert "vae part(s) from m-a-p/YuE2-Vae" in str(caught.value)
    assert "`crucible models pull yue2-3b`" in str(caught.value)
    found = audioweights.pull(config, manifest, spec)
    assert calls == ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]
    assert found.path == weights.subject_dir(config, manifest, "cuda-linux")
    assert audioweights.part_dirs(found.path, spec) == {"vae": str(found.path / "vae")}


def test_the_catalog_lists_audio_models_as_models_of_the_audio_job(home: Path) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    rows = {s.id: s for s in catalog.subjects(config, FAKE_BACKEND) if s.job_type == "audio"}
    assert sorted(rows) == ["stable-audio-3-medium", "stable-audio-3-small-sfx", "yue2-3b"]
    assert all(s.kind == "model" and s.installed() is None for s in rows.values())
    mac = {s.id for s in catalog.subjects(config, FAKE_MAC_BACKEND) if s.job_type == "audio"}
    assert mac == {"stable-audio-3-medium", "stable-audio-3-small-sfx"}
    assert catalog.backends_declaring("model", "yue2-3b") == ["cuda-linux"]


def test_the_desktop_packages_screen_has_words_for_audio() -> None:
    assert JOB_TYPE_WORDS["audio"][0] == "Audio generation"
