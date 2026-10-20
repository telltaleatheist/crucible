"""LTX-2.3 Dub-It: a reference video re-made so its speaker says a new line, on the Mac.

The manifest and its refusals, the reference told apart by its bytes and measured the way the
port measures it, the clip's length from the reference (never cut), the PC's refusal before
anything loads, and the request the Mac worker is sent, through the fake worker. The worker's
own lip-dub engine needs mlx and the port; a first live run on the Mac is what proves it.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible import weights
from crucible.capabilityclasses import BY_NAME
from crucible.errors import ApiError, JobError
from crucible.jobs import video as video_job
from crucible.jobs.video.params import (
    VideoParams,
    reference_size,
    settle,
    settle_lip_dub,
    snap_frames,
)
from crucible.videomodels import (
    LIP_DUB_MODE,
    VideoManifestError,
    load_clip_manifests,
    load_video_manifest,
    parse_video_manifest,
)

from .conftest import FAKE_BACKEND
from .test_video_api import PNG, events_of, mac, refusal, rows, run_job, submit  # noqa: F401
from .test_video_api import ffmpeg, ready, idle_card, transcript  # noqa: F401

MODEL = "ltx-2.3-dubit"
MANIFEST = Path(__file__).resolve().parents[1] / "crucible" / "video" / f"{MODEL}.toml"
SPEC = load_video_manifest(MODEL).spec("mlx-darwin")
LINE = 'A man in glasses speaks to the camera, warmly: "Follow me on Patreon."'
MP4 = base64.b64encode(
    b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 64
).decode("ascii")
NTSC = "30000/1001"


def _probe(frames: int = 195, rate: str = NTSC, size: tuple[int, int] = (1920, 1080)):
    def probe(reference: video_job.Reference) -> video_job.Reference:
        return video_job.measured_reference(reference, {
            "streams": [
                {"codec_type": "video", "r_frame_rate": rate, "nb_frames": str(frames),
                 "width": size[0], "height": size[1]},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "6.5395"},
        })

    return probe


# --- the manifest -------------------------------------------------------------------------


def test_the_manifest_is_a_mac_only_lip_dub_model_pinned_to_three_repos() -> None:
    manifest = load_video_manifest(MODEL)
    assert manifest.lip_dub and sorted(manifest.backends) == ["mlx-darwin"]
    assert (SPEC.engine, SPEC.hf_repo, SPEC.revision, SPEC.gated) == (
        "ltx-2-mlx", "dgrauet/ltx-2.3-mlx-q8", "6671a7572a530862d1d60ce393b5d93491e3f76b", False,
    )
    assert "transformer-distilled-1.1.safetensors" in SPEC.files
    assert not any(name.startswith(("transformer-dev", "ltx-2.3-22b-distilled-lora",
                                    "temporal_upscaler", "spatial_upscaler_x1_5"))
                   for name in SPEC.files)
    lora, text = SPEC.companions
    assert (lora.name, lora.hf_repo, lora.revision) == (
        "ic-lora", "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt", "1456334c3d69924de5083e553733b108ed1147f2"
    )
    assert lora.files[0].sha256.startswith("fc415b12cb63") and lora.files[0].bytes == 2_466_665_072
    assert (text.name, text.hf_repo) == ("text-encoder", "mlx-community/gemma-3-12b-it-4bit")
    assert {entry.target for entry in text.files} >= {
        "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    }
    assert SPEC.transformer_companion is None
    assert (SPEC.fps, SPEC.default_fps, SPEC.default_duration_s) == ((), None, None)
    assert (SPEC.steps, SPEC.refine_steps, SPEC.max_frames, SPEC.max_video_tokens) == (8, 3, 193, 8400)
    assert SPEC.memory_bytes_estimate == max(value for _, value in SPEC.stage_memory_bytes)
    assert "34.4 GB" in SPEC.memory_note and "67 GB" in SPEC.memory_note
    assert manifest.to_dict()["pipeline"] == "lip-dub"


def test_the_video_verb_never_picks_the_lip_dub_model() -> None:
    assert MODEL not in load_clip_manifests() and "ltx-2.5-distilled" in load_clip_manifests()
    assert BY_NAME["video"].candidates.load() == load_clip_manifests()


def _edited(old: str, new: str) -> str:
    text = MANIFEST.read_text(encoding="utf-8")
    assert old in text
    return text.replace(old, new, 1)


@pytest.mark.parametrize(
    ("old", "new", "said"),
    [
        ("max_frames = 193", "fps = [24]\nmax_frames = 193", "declares no ['fps']"),
        ('name = "ic-lora"', 'name = "lora"', "exactly the companions ['ic-lora', 'text-encoder']"),
        ('pipeline = "lip-dub"', 'pipeline = "dub"', "is not one of"),
        ("refine_steps = 3\n", "", "declares refine_steps"),
        ("max_frames = 193", "audio_to_video = true\nmax_frames = 193", "distilled-pipeline limit"),
    ],
)
def test_what_the_loader_refuses_in_a_lip_dub_block(old: str, new: str, said: str) -> None:
    with pytest.raises(VideoManifestError) as caught:
        parse_video_manifest(_edited(old, new), MANIFEST, MODEL)
    assert said in str(caught.value)


def test_a_lip_dub_model_on_the_pc_engine_is_refused_at_load() -> None:
    text = _edited("[backends.mlx-darwin]", "[backends.cuda-linux]").replace(
        "backends.mlx-darwin", "backends.cuda-linux"
    ).replace('engine = "ltx-2-mlx"', 'engine = "ltx"').replace(
        "mlx_cache_limit_bytes = 4000000000\n", ""
    )
    with pytest.raises(VideoManifestError) as caught:
        parse_video_manifest(text, MANIFEST, MODEL)
    assert "has no lip-dub pipeline" in str(caught.value)


# --- the reference ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "found"),
    [
        (b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00", ("reference", "MP4")),
        (b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00", ("reference", "MOV")),
        (b"\x00\x00\x00\x08wide\x00\x00\x00\x00\x00\x00\x00\x00", ("reference", "MOV")),
        (b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x1f\x42\x86\x81\x01",
         ("reference", "WebM/Matroska")),
        (b"\x89PNG\r\n\x1a\n" + b"\0" * 8, ("picture", "PNG")),
        (b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00", None),
    ],
)
def test_a_reference_video_is_told_apart_by_its_bytes(head: bytes, found) -> None:
    assert video_job.sniff(head) == found


def _measured(streams: list[dict], duration: str = "6.5395") -> video_job.Reference:
    reference = video_job.Reference("talk.mp4", Path("talk.mp4"), "MP4")
    return video_job.measured_reference(reference, {"streams": streams, "format": {"duration": duration}})


def test_the_reference_is_measured_as_the_port_measures_it() -> None:
    found = _measured([
        {"codec_type": "video", "r_frame_rate": NTSC, "nb_frames": "195", "width": 1920, "height": 1080},
        {"codec_type": "audio"},
    ])
    assert (found.frames, found.frame_rate, found.width, found.height, found.seconds) == (
        195, NTSC, 1920, 1080, 6.54,
    )
    webm = _measured([
        {"codec_type": "video", "r_frame_rate": "25/1", "width": 1280, "height": 720},
        {"codec_type": "audio"},
    ], duration="4.0")
    assert (webm.frames, webm.frame_rate) == (100, "25/1")


@pytest.mark.parametrize(
    ("streams", "code"),
    [
        ([{"codec_type": "audio"}], "video_reference_no_video"),
        ([{"codec_type": "video", "r_frame_rate": "25/1", "nb_frames": "100", "width": 64,
           "height": 64}], "video_reference_no_audio"),
        ([{"codec_type": "video", "r_frame_rate": "0/0", "width": 64, "height": 64},
          {"codec_type": "audio"}], "invalid_inputs"),
    ],
)
def test_a_reference_without_a_picture_or_a_voice_is_refused_by_name(streams, code: str) -> None:
    with pytest.raises(JobError) as caught:
        _measured(streams, duration="0")
    assert caught.value.code == code


# --- the clip -----------------------------------------------------------------------------


def _settle(params: VideoParams, frames: int = 195, rate: str = NTSC,
            size: tuple[int, int] = (1920, 1080)):
    return settle_lip_dub(params, MODEL, SPEC, 7, frames=frames, frame_rate=rate,
                          width=size[0], height=size[1])


def test_the_clip_is_the_reference_rounded_down_onto_the_grid_at_its_own_rate() -> None:
    settled = _settle(VideoParams(prompt=LINE))
    assert (settled.mode, settled.width, settled.height, settled.num_frames) == (
        LIP_DUB_MODE, 768, 448, 193,
    )
    assert settled.frame_rate == NTSC and settled.fps == pytest.approx(29.97, abs=0.001)
    assert (settled.video_tokens, settled.steps, settled.reference_strength) == (8400, 8, 1.0)
    assert snap_frames(195) == 193 and snap_frames(9) == 9 and snap_frames(8) == 1


@pytest.mark.parametrize(
    ("size", "frame"),
    [((1920, 1080), (768, 448)), ((1080, 1920), (448, 768)), ((1440, 1080), (576, 448)),
     ((1080, 1080), (576, 576))],
)
def test_the_default_frame_follows_the_reference_s_shape(size, frame) -> None:
    assert reference_size(*size, SPEC) == frame


def test_a_reference_longer_than_the_model_makes_is_refused_and_never_cut() -> None:
    with pytest.raises(ApiError) as caught:
        _settle(VideoParams(prompt=LINE), frames=300)
    error = caught.value
    assert error.code == "video_reference_too_long"
    assert "at most 193 frames (6.44 s" in error.message and "never cut short" in error.message
    assert (error.details["max_frames"], error.details["num_frames"]) == (193, 297)


def test_a_reference_under_the_shortest_clip_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        _settle(VideoParams(prompt=LINE), frames=8)
    assert caught.value.code == "video_reference_too_short"


def test_a_frame_past_the_measured_ceiling_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        _settle(VideoParams(prompt=LINE, width=1024, height=576))
    assert caught.value.code == "video_too_large"


@pytest.mark.parametrize(
    ("params", "param"),
    [
        ({"duration_s": 3}, "duration_s"),
        ({"num_frames": 97}, "num_frames"),
        ({"fps": 24}, "fps"),
        ({"audio": False}, "audio"),
        ({"negative_prompt": "blur"}, "negative_prompt"),
        ({"steps": 20}, "steps"),
    ],
)
def test_what_the_reference_decides_is_refused_as_a_param(params: dict, param: str) -> None:
    with pytest.raises(ApiError) as caught:
        _settle(VideoParams(prompt=LINE, **params))
    assert (caught.value.code, caught.value.details["param"]) == ("video_param_unsupported", param)


def test_reference_strength_is_a_lip_dub_param_only() -> None:
    assert _settle(VideoParams(prompt=LINE, reference_strength=0.6)).reference_strength == 0.6
    distilled = load_video_manifest("ltx-2.5-distilled").spec("mlx-darwin")
    with pytest.raises(ApiError) as caught:
        settle(VideoParams(prompt=LINE, reference_strength=0.6), "ltx-2.5-distilled", distilled, 1)
    assert caught.value.details["param"] == "reference_strength"
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            VideoParams(prompt=LINE, reference_strength=bad)


# --- the job ------------------------------------------------------------------------------


def _dubit_weights(home: Path) -> Path:
    directory = home / "models" / MODEL / "mlx-darwin"
    for name in SPEC.files:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"weights")
    (directory / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": SPEC.hf_repo, "revision": SPEC.revision, "bytes": 30_000_000_000,
                    "pulled": "2026-10-19T22:00:00+0000"}),
        encoding="utf-8",
    )
    for companion in SPEC.companions:
        part = directory / companion.name
        part.mkdir(parents=True, exist_ok=True)
        for entry in companion.files:
            (part / entry.target).write_bytes(b"part")
        (part / "crucible-pull.json").write_text(
            json.dumps({"hf_repo": companion.hf_repo, "revision": companion.revision,
                        "bytes": companion.bytes, "pulled": "2026-10-19T22:00:00+0000",
                        "files": [{"target": e.target} for e in companion.files]}),
            encoding="utf-8",
        )
    return directory


def test_the_pc_refuses_the_model_by_name_before_anything_loads(
    ready: TestClient, auth: dict[str, str], transcript: Path  # noqa: F811
) -> None:
    error = refusal(submit(ready, auth, model=MODEL, params={"prompt": LINE},
                           inputs={"talk.mp4": {"inline_base64": MP4}}))
    assert error["code"] == "backend_unsupported"
    assert (error["details"]["backend"], error["details"]["declared"]) == (
        FAKE_BACKEND.kind, ["mlx-darwin"],
    )
    assert rows(transcript, "load") == []


def test_a_lip_dub_job_sends_the_reference_and_reports_what_it_kept(
    mac: TestClient, auth: dict[str, str], home: Path, transcript: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _dubit_weights(home)
    monkeypatch.setattr(video_job, "probe_reference", _probe())
    _, events = run_job(mac, auth, model=MODEL, params={"prompt": LINE, "seed": 42},
                        inputs={"talk.mp4": {"inline_base64": MP4}})
    assert events[-1]["event"] == "done", events[-1]
    (load,) = rows(transcript, "load")
    assert (load["engine"], load["pipeline"]) == ("ltx-2-mlx", "lip-dub")
    assert load["lora_path"] == str(directory / "ic-lora" / "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors")
    assert load["text_encoder_dir"] == str(directory / "text-encoder")
    (made,) = rows(transcript, "generate")
    assert made["reference_path"].endswith("talk.mp4") and made["image_path"] is None
    assert (made["num_frames"], made["frame_rate"], made["reference_strength"]) == (193, NTSC, 1.0)
    assert (made["width"], made["height"], made["steps"], made["refine_steps"]) == (768, 448, 8, 3)
    assert made["fps"] == pytest.approx(30000 / 1001)
    video = events[-1]["data"]["video"]
    assert (video["mode"], video["frame_rate"], video["num_frames"]) == ("lip-dub", NTSC, 193)
    assert video["reference"] == {
        "input": "talk.mp4", "format": "MP4", "width": 1920, "height": 1080, "duration_s": 6.54,
        "frame_rate": NTSC, "fps": 29.97, "frames": 195, "frames_used": 193, "frames_unused": 2,
        "strength": 1.0, "audio_level_dbfs": {"mean": -28.4, "peak": -10.9},
    }
    assert video["lora"]["file"] == "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors"
    assert video["lora"]["strength"] == 1.0 and video["transformer"] is None
    assert set(video["stage_peak_bytes"]) == {"encoding", "denoising", "refining", "decoding"}
    assert video["soundtrack"] is None


def test_a_lip_dub_job_without_a_reference_is_refused_by_name(
    mac: TestClient, auth: dict[str, str], home: Path, transcript: Path,  # noqa: F811
) -> None:
    _dubit_weights(home)
    _, events = run_job(mac, auth, model=MODEL, params={"prompt": LINE},
                        inputs={"face.png": {"inline_base64": PNG}})
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "invalid_inputs"
    assert "exactly one input" in events[-1]["data"]["error"]["message"]
    assert rows(transcript, "load") == []


def test_a_reference_beside_another_input_is_refused(
    mac: TestClient, auth: dict[str, str], home: Path, transcript: Path,  # noqa: F811
) -> None:
    _dubit_weights(home)
    _, events = run_job(mac, auth, model=MODEL, params={"prompt": LINE},
                        inputs={"talk.mp4": {"inline_base64": MP4}, "face.png": {"inline_base64": PNG}})
    assert events[-1]["data"]["error"]["code"] == "invalid_inputs"
    assert "beside it" in events[-1]["data"]["error"]["message"]


def test_a_clip_model_refuses_a_reference_video(
    mac: TestClient, auth: dict[str, str], transcript: Path,  # noqa: F811
) -> None:
    _, events = run_job(mac, auth, inputs={"talk.mp4": {"inline_base64": MP4}})
    error = events[-1]["data"]["error"]
    assert error["code"] == "invalid_inputs" and "takes no reference video" in error["message"]
    assert rows(transcript, "load") == []


def test_a_too_long_reference_fails_by_name_before_the_worker(
    mac: TestClient, auth: dict[str, str], home: Path, transcript: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dubit_weights(home)
    monkeypatch.setattr(video_job, "probe_reference", _probe(frames=600))
    _, events = run_job(mac, auth, model=MODEL, params={"prompt": LINE},
                        inputs={"talk.mp4": {"inline_base64": MP4}})
    assert events[-1]["data"]["error"]["code"] == "video_reference_too_long"
    assert rows(transcript, "load") == []


def test_submit_refuses_a_length_for_a_lip_dub_model(
    mac: TestClient, auth: dict[str, str], home: Path,  # noqa: F811
) -> None:
    _dubit_weights(home)
    error = refusal(submit(mac, auth, model=MODEL, params={"prompt": LINE, "duration_s": 3},
                           inputs={"talk.mp4": {"inline_base64": MP4}}))
    assert (error["code"], error["details"]["param"]) == ("video_param_unsupported", "duration_s")


# --- adopting weights already on disk -----------------------------------------------------


def test_a_companion_file_already_in_place_with_its_hash_is_kept_not_fetched(
    config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    from crucible.audiomodels import CompanionFile

    here, fetched = b"the lora, moved in by hand", b"the config"
    files = [
        CompanionFile("lora.safetensors", "lora.safetensors",
                      hashlib.sha256(here).hexdigest(), len(here)),
        CompanionFile("config.json", "config.json", hashlib.sha256(fetched).hexdigest(), len(fetched)),
    ]
    root = tmp_path / "ic-lora"
    root.mkdir()
    (root / "lora.safetensors").write_bytes(here)
    (root / "config.json").write_bytes(b"stale bytes")
    asked: list[str] = []

    payloads = {"lora.safetensors": here, "config.json": fetched}

    def download(*, repo_id, filename, revision, local_dir, token=None, **_):
        asked.append(filename)
        destination = Path(local_dir) / filename
        destination.write_bytes(payloads[filename])
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    lines: list[str] = []
    result = weights.pull_files(config, hf_repo="a/b", revision="0" * 40, files=files,
                                target_root=root, label="test", on_line=lines.append)
    assert asked == ["config.json"]
    assert (root / "lora.safetensors").read_bytes() == here
    assert (root / "config.json").read_bytes() == fetched
    assert result.bytes == len(here) + len(fetched)
    assert any("kept lora.safetensors" in line for line in lines)
    asked.clear()
    weights.pull_files(config, hf_repo="a/b", revision="0" * 40, files=files,
                       target_root=root, label="test", force=True)
    assert asked == ["lora.safetensors", "config.json"]


@pytest.fixture
def config(home: Path):
    from crucible.config import load_config

    from .conftest import configure_box

    configure_box(home, enable_video=True)
    return load_config(home)
