from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import catalog, jobenv, verdict, videoweights, weights
from crucible.capabilityclasses import BY_NAME
from crucible.config import load_config
from crucible.desktop_app.screens import JOB_TYPE_WORDS
from crucible.memorybudget import GIB
from crucible.videomodels import (
    VideoManifestError,
    frames_for,
    load_all_video_manifests,
    load_video_manifest,
    parse_video_manifest,
)

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, configure_box

MODEL = "ltx-2.5-distilled"
JOBS = Path(__file__).resolve().parents[1] / "crucible" / "jobs"
NAMES = Path(__file__).resolve().parent / "data" / "ltx25_gguf_tensor_names.txt"
MANIFEST = Path(__file__).resolve().parents[1] / "crucible" / "video" / f"{MODEL}.toml"


def _sibling(name: str) -> Any:
    sys.path.insert(0, str(JOBS))
    sys.path.insert(0, str(JOBS / "video"))
    try:
        import importlib

        return importlib.import_module(name)
    finally:
        sys.path.remove(str(JOBS))
        sys.path.remove(str(JOBS / "video"))


def test_the_manifest_pins_the_lightricks_snapshot_and_the_gguf() -> None:
    manifests = load_all_video_manifests()
    assert sorted(manifests) == [MODEL]
    manifest = manifests[MODEL]
    assert not manifest.supports("mlx-darwin") and manifest.supports("cuda-linux")
    spec = manifest.spec("cuda-linux")
    assert (spec.hf_repo, spec.revision, spec.gated) == (
        "Lightricks/LTX-2.5-Diffusers", "426936f8b22dc28e4def61e515478b0b7e4a53cc", True
    )
    assert "transformer/config.json" in spec.files
    assert not any(name.startswith(("transformer/diffusion", "transformer_full/")) for name in spec.files)
    assert "text_encoder/model-00005-of-00005.safetensors" in spec.files
    assert "connectors/diffusion_pytorch_model.safetensors" not in spec.files
    companion = spec.transformer_companion
    assert (companion.hf_repo, companion.revision) == (
        "Abiray/LTX-2.5-Distilled-GGUF", "7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0"
    )
    assert companion.files[0].bytes == 18_624_033_216
    assert spec.memory_basis == "declared" and spec.memory_bytes_estimate == max(
        value for _, value in spec.stage_memory_bytes
    )
    assert "LTX-2.x Community License" in manifest.licence
    assert "US$10,000,000" in manifest.commercial_use


def test_the_declared_limits_are_the_ones_the_docs_state() -> None:
    spec = load_video_manifest(MODEL).spec("cuda-linux")
    assert (spec.max_side, spec.max_pixels, spec.max_frames, spec.fps) == (1280, 1280 * 704, 145, (24, 25))
    assert spec.video_tokens(1280, 704, 145) == spec.max_video_tokens == 16_720
    assert spec.token_ceiling("image-to-video") == 8_800 == spec.video_tokens(1280, 704, 73)
    assert (spec.steps, spec.audio_sample_rate) == (8, 48000)


@pytest.mark.parametrize(
    ("duration", "fps", "frames"),
    [(5.0, 24, 121), (6.0, 24, 145), (2.0, 24, 49), (0.1, 24, 9), (3.0, 25, 73), (4.0, 25, 97)],
)
def test_a_duration_lands_on_the_vae_frame_grid(duration: float, fps: int, frames: int) -> None:
    assert frames_for(duration, fps) == frames and (frames - 1) % 8 == 0


def _manifest_with(old: str, new: str) -> str:
    text = MANIFEST.read_text(encoding="utf-8")
    assert old in text, old
    return text.replace(old, new, 1)


@pytest.mark.parametrize(
    ("old", "new", "words"),
    [
        ("memory_bytes_estimate = 20500000000", "memory_bytes_estimate = 19000000000", "largest stage"),
        ("max_frames = 145", "max_frames = 144", "frame grid"),
        ("default_fps = 24", "default_fps = 30", "not one of fps"),
        ('revision = "426936f8b22dc28e4def61e515478b0b7e4a53cc"', 'revision = "main"', "40-character"),
        ("[backends.cuda-linux]", "[backends.mlx-darwin]", "not a video backend"),
        ('target = "LTX-2.5-Distilled-Q6_K.gguf"', 'target = "transformer.bin"', ".gguf"),
        ("max_video_tokens = 16720", "max_video_tokens = 10000", "the default clip is"),
        ("decoding = 9000000000", "sorting = 9000000000", "the stages are"),
    ],
)
def test_a_manifest_that_contradicts_itself_is_refused(old: str, new: str, words: str) -> None:
    with pytest.raises(VideoManifestError) as caught:
        parse_video_manifest(_manifest_with(old, new), MANIFEST, MODEL)
    assert words in str(caught.value)


@pytest.mark.parametrize(
    ("backend_kind", "total", "enabled", "summary"),
    [
        ("cuda-linux", 24 * GIB, True, f"can make video, using {MODEL}"),
        ("cuda-linux", 16 * GIB, False, "cannot make video"),
        ("mlx-darwin", 64 * GIB, False, "cannot make video"),
    ],
)
def test_the_capability_row_says_where_video_can_be_made(
    backend_kind: str, total: int, enabled: bool, summary: str
) -> None:
    decided = verdict.decide(
        BY_NAME["video"], backend_kind, total_bytes=total, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia" if backend_kind == "cuda-linux" else "apple", chosen=None,
    )
    assert decided.enabled is enabled, decided.reason
    assert decided.summary.startswith(summary), decided.summary
    assert decided.reason


def test_the_mac_has_no_video_env_and_says_why() -> None:
    assert [spec.key for spec in jobenv.video_envs("cuda-linux")] == ["video-ltx"]
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv.video_envs("mlx-darwin")
    assert "cuda-linux" in str(caught.value) and "only" in str(caught.value)


def test_the_recipe_pins_the_diffusers_commit_and_the_quantization_libraries() -> None:
    recipe = jobenv.recipe_for(jobenv.video_env("ltx", "cuda-linux"))
    assert recipe.name == "ltx-cuda-linux.txt"
    assert jobenv.recipe_direct_references(recipe) == {
        "diffusers": "5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2"
    }
    pins = jobenv.recipe_pins(recipe)
    assert (pins["torch"], pins["transformers"], pins["torchao"], pins["gguf"], pins["av"]) == (
        "2.14.0", "5.17.0", "0.18.0", "0.19.0", "18.1.0"
    )
    assert jobenv.recipe_archive_bytes(recipe) > 3_000_000_000
    assert jobenv.SMOKE_IMPORT["video-ltx"] == {"cuda-linux": "diffusers, gguf, torchao, av"}


def test_every_transformer_tensor_in_the_gguf_maps_to_a_diffusers_parameter() -> None:
    ltxkeys = _sibling("ltxkeys")
    names = [
        line.strip() for line in NAMES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    mapped = {name: ltxkeys.diffusers_name(name) for name in names}
    connectors = {name for name, found in mapped.items() if found is None}
    assert connectors and all("embeddings_connector" in name for name in connectors)
    top = {
        "proj_in", "audio_proj_in", "keyframes_abs_pos_embedding", "time_embed", "audio_time_embed",
        "prompt_adaln", "audio_prompt_adaln", "av_cross_attn_video_scale_shift",
        "av_cross_attn_audio_scale_shift", "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_v2a_gate", "scale_shift_table", "audio_scale_shift_table",
        "transformer_blocks", "proj_out", "audio_proj_out",
    }
    block = {
        "attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn",
        "ff", "audio_ff", "scale_shift_table", "audio_scale_shift_table", "prompt_scale_shift_table",
        "audio_prompt_scale_shift_table", "video_a2v_cross_attn_scale_shift_table",
        "audio_a2v_cross_attn_scale_shift_table",
    }
    attention = {"norm_q", "norm_k", "to_q", "to_k", "to_v", "to_out", "to_gate_logits"}
    for name, found in mapped.items():
        if found is None:
            continue
        parts = found.split(".")
        assert parts[0] in top, (name, found)
        if parts[0] == "transformer_blocks":
            assert parts[2] in block, (name, found)
            if parts[2].endswith("attn") or parts[2].endswith(("attn1", "attn2")):
                assert parts[3] in attention, (name, found)
    assert mapped["adaln_single.linear.weight"] == "time_embed.linear.weight"
    assert mapped["audio_adaln_single.linear.bias"] == "audio_time_embed.linear.bias"
    assert mapped["prompt_adaln_single.linear.weight"] == "prompt_adaln.linear.weight"
    assert mapped["audio_prompt_adaln_single.linear.weight"] == "audio_prompt_adaln.linear.weight"
    assert mapped["av_ca_a2v_gate_adaln_single.linear.weight"] == "av_cross_attn_video_a2v_gate.linear.weight"
    assert mapped["patchify_proj.weight"] == "proj_in.weight"
    assert mapped["audio_patchify_proj.bias"] == "audio_proj_in.bias"
    assert mapped["transformer_blocks.0.attn1.q_norm.weight"] == "transformer_blocks.0.attn1.norm_q.weight"
    assert (
        mapped["transformer_blocks.0.scale_shift_table_a2v_ca_video"]
        == "transformer_blocks.0.video_a2v_cross_attn_scale_shift_table"
    )
    assert len(set(mapped.values()) - {None}) == len(names) - len(connectors)


def test_the_real_worker_imports_without_torch_and_maps_a_prefixed_name() -> None:
    ltxkeys = _sibling("ltxkeys")
    assert ltxkeys.diffusers_name("model.diffusion_model.patchify_proj.weight") == "proj_in.weight"
    assert ltxkeys.diffusers_name("model.diffusion_model.video_embeddings_connector.x") is None


def _clip(videocore: Any, pcm: bytes | None) -> Any:
    frame = bytes(3 * 64 * 32)
    return videocore.Clip(64, 32, 24, [frame] * 9, 9, pcm=pcm, channels=2 if pcm else 0,
                          sample_rate=48000 if pcm else 0)


def test_the_mux_command_encodes_h264_and_aac_into_an_mp4() -> None:
    videocore = _sibling("videocore")
    command = videocore.mux_command("ffmpeg", "libopenh264", _clip(videocore, b"\0\0" * 4), "a.wav", "v.mp4")
    assert command[:1] == ["ffmpeg"] and command[-1] == "v.mp4"
    joined = " ".join(command)
    assert "-f rawvideo -pix_fmt rgb24 -s 64x32 -framerate 24 -i pipe:0 -i a.wav" in joined
    assert "-c:v libopenh264 -b:v 2000000 -pix_fmt yuv420p" in joined
    assert "-c:a aac -b:a 192k -shortest" in joined and "-movflags +faststart" in joined
    silent = videocore.mux_command("ffmpeg", "libx264", _clip(videocore, None), None, "v.mp4")
    assert "-an" in silent and "-i" in silent and silent.count("-i") == 1
    assert "-crf" in silent
    assert videocore.video_options("libopenh264", 1280, 704, 24)[3] == str(round(1280 * 704 * 24 * 0.3))


def test_the_encoder_is_picked_from_what_this_ffmpeg_has(monkeypatch: pytest.MonkeyPatch) -> None:
    videocore = _sibling("videocore")
    listing = (
        "Encoders:\n V..... = Video\n ------\n V....D libopenh264          OpenH264 H.264\n"
        " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n A....D aac                  AAC\n"
    )
    monkeypatch.setattr(
        videocore.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=listing, stderr=""),
    )
    assert videocore.pick_encoder("ffmpeg") == "libopenh264"
    monkeypatch.setattr(
        videocore.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=" A....D aac  AAC\n", stderr=""),
    )
    with pytest.raises(RuntimeError) as caught:
        videocore.pick_encoder("/opt/other/ffmpeg")
    assert "no H.264 encoder" in str(caught.value) and "libopenh264" in str(caught.value)


def test_a_real_ffmpeg_if_there_is_one_makes_an_mp4_with_sound(tmp_path: Path) -> None:
    found = shutil.which("ffmpeg")
    if found is None:
        pytest.skip("no ffmpeg on this machine's PATH; the command itself is tested above")
    videocore = _sibling("videocore")
    try:
        encoder = videocore.pick_encoder(found)
    except RuntimeError:
        pytest.skip(f"{found} has no H.264 encoder")
    output = tmp_path / "video.mp4"
    used = videocore.mux(found, _clip(videocore, b"\0\0" * 2 * 48000), str(output))
    assert used == encoder and output.read_bytes()[4:8] == b"ftyp"
    assert not (tmp_path / "video.mp4.audio.wav").exists()
    probe = shutil.which("ffprobe")
    if probe is not None:
        streams = subprocess.run(
            [probe, "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(output)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert streams == ["h264", "aac"]


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


def test_the_model_is_installed_only_with_its_gguf_and_the_pull_fetches_both(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    configure_box(home, enable_video=True)
    config = load_config(home)
    manifest = load_video_manifest(MODEL)
    spec = manifest.spec("cuda-linux")
    calls: list[tuple[str, tuple[str, ...]]] = []

    def main_pull(config: Any, manifest: Any, spec: Any, **_: Any) -> weights.InstalledWeights:
        calls.append((spec.hf_repo, spec.files))
        _stamp_main(config, manifest, spec)
        return weights.installed(config, manifest, spec)

    def files_pull(config: Any, *, hf_repo: str, revision: str, files: Any, target_root: Path, **_: Any) -> Any:
        calls.append((hf_repo, tuple(entry.sha256 for entry in files)))
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
    assert videoweights.installed(config, manifest, spec) is None
    with pytest.raises(weights.WeightsError) as caught:
        videoweights.require_installed(config, manifest, spec)
    assert "transformer-gguf part(s) from Abiray/LTX-2.5-Distilled-GGUF" in str(caught.value)
    found = videoweights.pull(config, manifest, spec)
    assert [repo for repo, _ in calls] == ["Lightricks/LTX-2.5-Diffusers", "Abiray/LTX-2.5-Distilled-GGUF"]
    assert calls[1][1] == ("ee8835ff8f11e4f59fa4be7bf31b1200172659364e724de444d790ddf4869a58",)
    assert spec.transformer_path(found.path).is_file()


def test_the_catalog_lists_the_video_model_on_the_pc_only(home: Path) -> None:
    configure_box(home, enable_video=True)
    config = load_config(home)
    rows = {s.id: s for s in catalog.subjects(config, FAKE_BACKEND) if s.job_type == "video"}
    assert sorted(rows) == [MODEL]
    assert rows[MODEL].kind == "model" and rows[MODEL].installed() is None
    assert not [s for s in catalog.subjects(config, FAKE_MAC_BACKEND) if s.job_type == "video"]
    assert catalog.backends_declaring("model", MODEL) == ["cuda-linux"]


def test_the_desktop_packages_screen_has_words_for_video() -> None:
    assert JOB_TYPE_WORDS["video"][0] == "Video generation"
