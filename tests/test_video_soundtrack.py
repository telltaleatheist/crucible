"""A video made to a supplied soundtrack: the inputs told apart by their bytes, the clip's
length from the soundtrack, and every refusal by name. The job end to end is in
test_video_api.py; the Mac worker's frozen audio is in test_video_mac_worker.py."""

from __future__ import annotations

import pytest

from crucible.errors import ApiError
from crucible.jobs import video as video_job
from crucible.jobs.video.params import VideoParams, longest_covered, settle
from crucible.videomodels import (
    AUDIO_TO_VIDEO,
    IMAGE_AUDIO_TO_VIDEO,
    IMAGE_TO_VIDEO,
    TEXT_TO_VIDEO,
    load_video_manifest,
)

MODEL = "ltx-2.5-distilled"
MAC = load_video_manifest(MODEL).spec("mlx-darwin")
PC = load_video_manifest(MODEL).spec("cuda-linux")
PROMPT = "A woman at a kitchen table looks into the camera and says the line, smiling."


@pytest.mark.parametrize(
    ("head", "found"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"\0" * 8, ("picture", "PNG")),
        (b"\xff\xd8\xff\xe0" + b"\0" * 12, ("picture", "JPEG")),
        (b"RIFF\x10\0\0\0WEBPVP8 ", ("picture", "WebP")),
        (b"RIFF\x10\0\0\0WAVEfmt ", ("soundtrack", "WAV")),
        (b"fLaC\0\0\0\x22" + b"\0" * 8, ("soundtrack", "FLAC")),
        (b"ID3\x04\0\0\0\0\0\0" + b"\0" * 6, ("soundtrack", "MP3")),
        (b"\xff\xfb\x90\x64" + b"\0" * 12, ("soundtrack", "MP3")),
        (b"OggS\0\x02" + b"\0" * 10, ("soundtrack", "Ogg")),
        (b"\xff\xf1\x50\x80" + b"\0" * 12, None),
        (b"RIFF\x10\0\0\0AVI LIST", None),
        (b"hello, world....", None),
    ],
)
def test_an_input_is_a_picture_or_a_soundtrack_by_its_bytes(head: bytes, found) -> None:
    assert video_job.sniff(head) == found


@pytest.mark.parametrize(
    ("picture", "soundtrack", "mode"),
    [
        (False, False, TEXT_TO_VIDEO),
        (True, False, IMAGE_TO_VIDEO),
        (False, True, AUDIO_TO_VIDEO),
        (True, True, IMAGE_AUDIO_TO_VIDEO),
    ],
)
def test_the_inputs_name_the_mode(tmp_path, picture: bool, soundtrack: bool, mode: str) -> None:
    track = video_job.Soundtrack("line.wav", tmp_path / "line.wav", "WAV") if soundtrack else None
    inputs = video_job.ClipInputs(tmp_path / "face.png" if picture else None, track)
    assert inputs.mode == mode


@pytest.mark.parametrize(("seconds", "fps", "frames"), [
    (3.0, 24, 65), (2.041667, 24, 49), (2.041, 24, 41), (5.0, 25, 121), (0.3, 24, 1),
])
def test_the_longest_clip_a_soundtrack_covers_rounds_down_onto_the_grid(
    seconds: float, fps: int, frames: int
) -> None:
    assert longest_covered(seconds, fps) == frames
    if frames > 1:
        assert frames / fps <= seconds < (frames + 8) / fps


def _refused(**kwargs) -> ApiError:
    with pytest.raises(ApiError) as caught:
        settle(**kwargs)
    return caught.value


def test_with_no_length_the_soundtrack_sets_it() -> None:
    settled = settle(VideoParams(prompt=PROMPT, width=768, height=512), MODEL, MAC, 1,
                     AUDIO_TO_VIDEO, 3.0)
    assert (settled.mode, settled.num_frames, settled.duration_s, settled.soundtrack_s) == (
        AUDIO_TO_VIDEO, 65, 2.708, 3.0,
    )


def test_a_clip_shorter_than_its_soundtrack_is_made_of_the_first_part() -> None:
    settled = settle(VideoParams(prompt=PROMPT, duration_s=2), MODEL, MAC, 1, AUDIO_TO_VIDEO, 9.5)
    assert (settled.num_frames, settled.soundtrack_s) == (49, 9.5)


def test_a_clip_longer_than_its_soundtrack_is_refused_naming_the_longest_it_covers() -> None:
    error = _refused(params=VideoParams(prompt=PROMPT, num_frames=49), model=MODEL, spec=MAC,
                     seed=1, mode=AUDIO_TO_VIDEO, soundtrack_s=1.0)
    assert error.code == "video_soundtrack_too_short"
    assert "the soundtrack is 1.00 s" in error.message and "17 frames (0.71 s" in error.message
    assert "never padded or looped" in error.message
    assert (error.details["longest_frames"], error.details["num_frames"]) == (17, 49)


def test_a_soundtrack_under_the_shortest_clip_is_refused() -> None:
    error = _refused(params=VideoParams(prompt=PROMPT), model=MODEL, spec=MAC, seed=1,
                     mode=AUDIO_TO_VIDEO, soundtrack_s=0.3)
    assert error.code == "video_soundtrack_too_short"
    assert "shorter than the shortest clip, 9 frames" in error.message


def test_a_soundtrack_longer_than_the_arm_makes_is_refused_by_the_length_limit() -> None:
    error = _refused(params=VideoParams(prompt=PROMPT), model=MODEL, spec=MAC, seed=1,
                     mode=AUDIO_TO_VIDEO, soundtrack_s=30.0)
    assert error.code == "video_too_long"
    assert "came from the soundtrack (30.00 s)" in error.message
    assert "for a clip of its first part" in error.message


def test_audio_false_with_a_soundtrack_is_a_contradiction() -> None:
    error = _refused(params=VideoParams(prompt=PROMPT, audio=False), model=MODEL, spec=MAC,
                     seed=1, mode=AUDIO_TO_VIDEO, soundtrack_s=3.0)
    assert (error.code, error.details["param"]) == ("video_param_unsupported", "audio")
    assert "is the clip's sound" in error.message


@pytest.mark.parametrize("mode", [AUDIO_TO_VIDEO, IMAGE_AUDIO_TO_VIDEO])
def test_the_pc_arm_refuses_a_soundtrack_by_name(mode: str) -> None:
    error = _refused(params=VideoParams(prompt=PROMPT, width=768, height=512), model=MODEL,
                     spec=PC, seed=1, mode=mode, soundtrack_s=3.0)
    assert error.code == "audio_to_video_unsupported"
    assert "cuda-linux" in error.message and "mlx-darwin" in error.message


def test_a_picture_and_a_soundtrack_share_the_image_to_video_ceiling() -> None:
    picture = _refused(params=VideoParams(prompt=PROMPT, duration_s=6), model=MODEL, spec=MAC,
                       seed=1, mode=IMAGE_AUDIO_TO_VIDEO, soundtrack_s=7.0)
    assert picture.code == "video_too_large" and "14,080" in picture.message
    alone = settle(VideoParams(prompt=PROMPT, duration_s=6), MODEL, MAC, 1, AUDIO_TO_VIDEO, 7.0)
    assert alone.video_tokens == 16_720 == MAC.token_ceiling(AUDIO_TO_VIDEO)
    assert MAC.token_ceiling(IMAGE_AUDIO_TO_VIDEO) == MAC.token_ceiling(IMAGE_TO_VIDEO) == 14_080


def test_a_mode_and_a_soundtrack_length_come_together() -> None:
    with pytest.raises(ValueError):
        settle(VideoParams(prompt=PROMPT), MODEL, MAC, 1, AUDIO_TO_VIDEO)
    with pytest.raises(ValueError):
        settle(VideoParams(prompt=PROMPT), MODEL, MAC, 1, TEXT_TO_VIDEO, 3.0)


def test_only_the_mac_arm_declares_audio_to_video() -> None:
    assert (MAC.audio_to_video, PC.audio_to_video) == (True, False)
    assert MAC.to_dict()["audio_to_video"] is True and PC.to_dict()["audio_to_video"] is False
    assert dict(MAC.stage_memory_bytes)["audio_encoding"] < MAC.memory_bytes_estimate
