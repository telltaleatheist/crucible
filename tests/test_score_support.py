from __future__ import annotations

import sys
from pathlib import Path

import pytest

from crucible import catalog, cli, installonsubmit, jobenv, verdict
from crucible.audiomodels import load_audio_manifest
from crucible.capabilityclasses import BY_NAME
from crucible.config import load_config
from crucible.desktop_app.screens import JOB_TYPE_WORDS
from crucible.jobs import score as score_job
from crucible.jobs.score import recording
from crucible.jobs.score.params import ScoreParams
from crucible.memorybudget import GIB
from crucible.scoremodels import (
    ScoreManifestError,
    load_all_score_manifests,
    load_score_manifest,
    parse_score_manifest,
)

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, configure_box
from .score_fixtures import SHEETSAGE2_SONG, SILENT_SONG, UNKNOWN_LABEL_SONG, song

sys.path.insert(0, str(Path(score_job.__file__).resolve().parent))
import yue2native  # noqa: E402

sys.path.pop(0)

SHEETSAGE2_REVISION = "32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
MERT_SHA256 = "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d"

# SheetSage2's structure vocabulary at the pinned revision (labels_sheetsage2.py, its second
# and effective STRUCTURE_LABELS).
SHEETSAGE2_LABELS = [
    "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus",
    "interlude", "fade-out", "loop", "rap", "preshot", "irregular", "instrumental",
    "intro and verse", "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme",
    "development", "variation", "pre-outro",
]


# --- the manifest -------------------------------------------------------------------------


def test_sheetsage2_is_declared_with_its_pins_encoder_and_licence() -> None:
    manifests = load_all_score_manifests()
    assert sorted(manifests) == ["sheetsage2"]
    manifest = manifests["sheetsage2"]
    assert manifest.licence.startswith("CC BY-NC 4.0")
    assert manifest.commercial_use.startswith("non-commercial")
    assert manifest.code_licence.startswith("unstated")
    assert sorted(manifest.backends) == ["cuda-linux", "mlx-darwin"]
    for spec in manifest.backends.values():
        assert (spec.hf_repo, spec.revision, spec.gated) == ("m-a-p/SheetSage2", SHEETSAGE2_REVISION, False)
        assert spec.dtype == "float32" and spec.memory_basis == "declared"
        assert "not measured" in spec.memory_note
        assert {"model.safetensors", "modeling_sheetsage2.py", "rendering_sheetsage2.py",
                "tensors_sheetsage2.py", "config.json"} <= set(spec.files)
        encoder = spec.encoder
        assert (encoder.name, encoder.hf_repo, encoder.revision) == (
            "mert", "m-a-p/MERT-v2-FullSong", MERT_REVISION
        )
        weights = {entry.target: entry for entry in encoder.files}
        assert weights["model.safetensors"].sha256 == MERT_SHA256
        assert weights["model.safetensors"].bytes == 2_529_812_848
        # The hashes SheetSage2's loader checks the encoder's code against (BASE_CODE_HASHES).
        assert weights["modeling_mert2.py"].sha256.startswith("b1a3174e5649")
        assert weights["configuration_mert2.py"].sha256.startswith("77b53ec9d7ee")
        assert spec.verified is False and spec.unverified_note
    assert manifest.spec("cuda-linux").autocast == "bfloat16"
    assert manifest.spec("mlx-darwin").autocast is None
    assert manifest.spec("cuda-linux").device == "cuda"
    assert manifest.spec("mlx-darwin").device == "mps"
    assert "MPS is unverified" in manifest.spec("mlx-darwin").unverified_note


def test_the_ceiling_is_the_longest_song_yue2_composes() -> None:
    yue2 = load_audio_manifest("yue2-3b")
    for backend_kind, spec in load_score_manifest("sheetsage2").backends.items():
        assert spec.max_audio_s == yue2.spec(backend_kind).max_duration_s == 360


def test_it_fits_the_pc_card_beside_its_desktop_allowance() -> None:
    assert load_score_manifest("sheetsage2").spec("cuda-linux").memory_bytes_estimate <= 24 * GIB - 3 * GIB


BASE = """
[model]
id = "sheetsage2"
family = "sheetsage2"
display = "SheetSage2"
licence = "CC BY-NC 4.0"
licence_url = "https://example.com"
commercial_use = "no"
code_licence = "unstated"

[backends.cuda-linux]
engine = "sheetsage2"
hf_repo = "m-a-p/SheetSage2"
revision = "32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a"
gated = false
dtype = "float32"
{extra}
memory_bytes_estimate = 6000000000
memory_basis = "declared"
memory_note = "declared"
files = ["model.safetensors"]
max_audio_s = 360
max_audio_note = "YuE2"
verified = {verified}

[[backends.cuda-linux.companions]]
name = "{companion}"
hf_repo = "m-a-p/MERT-v2-FullSong"
revision = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"

[[backends.cuda-linux.companions.files]]
source = "model.safetensors"
target = "model.safetensors"
sha256 = "{sha}"
bytes = 1
"""


def _manifest(**changes: str):
    fields = {"extra": "", "verified": "true", "companion": "mert", "sha": MERT_SHA256, **changes}
    return parse_score_manifest(BASE.format(**fields), Path("sheetsage2.toml"), "sheetsage2")


def test_a_well_formed_manifest_parses() -> None:
    assert _manifest().spec("cuda-linux").encoder.name == "mert"


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"extra": 'dtype2 = "x"'}, "unknown key"),
        ({"extra": 'autocast = "float16"'}, "autocast 'float16'"),
        ({"verified": "false"}, "unverified_note"),
        ({"companion": "vae"}, "takes 'mert'"),
        ({"sha": "abc"}, "sha256 must be 64"),
    ],
)
def test_a_manifest_that_says_the_wrong_thing_is_refused_by_name(changes: dict, words: str) -> None:
    with pytest.raises(ScoreManifestError, match=words):
        _manifest(**changes)


# --- params ------------------------------------------------------------------------------


def test_the_task_and_prompts_follow_keep_harmony_as_yues_cover_skill_sends_them() -> None:
    melody = ScoreParams()
    assert (melody.task, melody.melody_only) == ("melody-full", True)
    assert melody.prompts == ["timestamp", "downbeat_meter", "structure", "key", "melody_full"]
    full = ScoreParams(keep_harmony=True)
    assert (full.task, full.melody_only) == ("full", False)
    assert full.prompts == ["timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full"]


# --- the recording -----------------------------------------------------------------------


def test_the_decode_is_sheetsage2s_own_format_from_the_crop() -> None:
    command = recording.decode_command(
        "/x/ffmpeg", Path("/in/song.mp3"), Path("/out/r.f32"), start_s=12.5, read_s=60.0
    )
    assert command[:5] == ["/x/ffmpeg", "-nostdin", "-v", "error", "-y"]
    assert command[command.index("-ss") + 1] == "12.500"
    assert command.index("-ss") < command.index("-i")
    for flag, value in (("-ac", "1"), ("-ar", "24000"), ("-f", "f32le"), ("-t", "60.000")):
        assert command[command.index(flag) + 1] == value
    whole = recording.decode_command(
        "/x/ffmpeg", Path("/in/song.mp3"), Path("/out/r.f32"), start_s=0.0, read_s=361.0
    )
    assert "-ss" not in whole


def test_the_one_input_is_the_recording() -> None:
    assert recording.the_one_input({"a.wav": Path("/a.wav")}) == Path("/a.wav")
    for inputs in ({}, {"a.wav": Path("/a"), "b.wav": Path("/b")}):
        with pytest.raises(Exception, match="exactly one recording"):
            recording.the_one_input(inputs)


# --- YuE2-native --------------------------------------------------------------------------


def test_every_sheetsage2_label_has_a_yue2_section() -> None:
    for label in SHEETSAGE2_LABELS:
        if label == "silence":
            continue
        assert yue2native.MAPPING[label][0] in yue2native.YUE2_SECTIONS, label
    assert set(yue2native.MAPPING) | {"silence"} == set(SHEETSAGE2_LABELS)
    assert {section for section, _ in yue2native.SILENCE_AT.values()} <= yue2native.YUE2_SECTIONS


def test_the_vendored_sections_are_the_ones_mapped_onto() -> None:
    _, instrumental, _ = yue2native.tools()
    assert set(instrumental.SECTIONS) == yue2native.YUE2_SECTIONS


def test_labels_are_mapped_merged_and_every_decision_recorded() -> None:
    text, records = yue2native.relabel(SHEETSAGE2_SONG)
    kept = [line[2:] for line in text.splitlines() if line.startswith("% ")]
    assert kept == ["intro", "verse", "chorus", "verse", "outro"]
    by_label = [(r["sheetsage2"], r["yue2"], r["kept"]) for r in records]
    assert by_label == [
        ("silence", "intro", True),
        ("verse", "verse", True),
        ("verse", "verse", False),
        ("chorus", "chorus", True),
        ("post-chorus", "chorus", False),
        ("bridge", "bridge", False),
        ("rap", "verse", True),
        ("fade-out", "outro", True),
        ("silence", "outro", False),
    ]
    assert "already the section in force" in records[2]["dropped_because"]
    assert "same group" in records[5]["dropped_because"]
    assert all(record["why"] for record in records)
    # Only the label lines change: every music line is SheetSage2's.
    music = [line for line in SHEETSAGE2_SONG.splitlines() if not line.startswith("% ")]
    assert [line for line in text.splitlines() if not line.startswith("% ")] == music


def test_silence_in_the_middle_is_an_interlude() -> None:
    text, _ = yue2native.relabel(song([
        (["verse"], "C8D8E8F8|", "Z|"),
        (["silence"], "Z|", "C,32|"),
        (["chorus"], "C8D8E8F8|", "Z|"),
    ]))
    assert [line[2:] for line in text.splitlines() if line.startswith("% ")] == ["verse", "interlude", "chorus"]


@pytest.mark.parametrize("keep_harmony", [True, False])
def test_a_sheetsage2_song_passes_yue2s_validator(keep_harmony: bool) -> None:
    native = yue2native.to_yue2(SHEETSAGE2_SONG, keep_harmony=keep_harmony)
    assert native.error is None, native.error
    assert [c["check"] for c in native.checks] == ["labels", "parse_abc", "convert_score", "validate_score"]
    _, instrumental, _ = yue2native.tools()
    instrumental.validate_score(native.instrumental)
    assert native.transfer["chords"] == ("preserved" if keep_harmony else "removed for melody cover")
    assert native.transfer["vocal_notes_after"] == 0


def test_an_unknown_label_fails_by_name_and_is_not_guessed() -> None:
    native = yue2native.to_yue2(UNKNOWN_LABEL_SONG, keep_harmony=False)
    assert native.error["code"] == "score_label_unknown"
    assert "'kazoo break'" in native.error["message"]
    assert native.score is None and native.instrumental is None


def test_a_score_yue2_cannot_take_fails_by_name_with_the_check_that_refused_it() -> None:
    native = yue2native.to_yue2(SILENT_SONG, keep_harmony=False)
    assert native.error["code"] == "score_not_yue2_native"
    assert native.checks[-1] == {
        "check": "convert_score",
        "what": "voice transfer to the instrumental form (instrumentalize.convert_score)",
        "passed": False,
        "said": "The score contains no sounding notes",
    }
    assert native.instrumental is None


def test_a_score_the_strict_parser_refuses_fails_by_name() -> None:
    broken = SHEETSAGE2_SONG.replace("C8D8E8F8|G16z16|C8D8E8F8|G16z16|", "C8D8E8|G16z16|C8D8E8F8|G16z16|")
    native = yue2native.to_yue2(broken, keep_harmony=True)
    assert native.error["code"] == "score_not_yue2_native"
    assert native.checks[-1]["check"] == "parse_abc" and native.checks[-1]["passed"] is False


# --- the env, the catalog and the rest of the build ----------------------------------------


def test_the_score_env_is_its_own_worker_env_with_sheetsage2s_pins() -> None:
    assert "score" in jobenv.WORKER_JOB_TYPES
    assert jobenv.JOB_TYPES_SERVED_BY_ENV["score"] == ("score",)
    for backend_kind in ("cuda-linux", "mlx-darwin"):
        recipe = jobenv.recipe_for(jobenv.worker_env("score", backend_kind))
        assert recipe.name == f"{backend_kind}.txt" and recipe.parent.name == "score"
        assert jobenv.recipe_archive_bytes(recipe) > 0
        pins = jobenv.recipe_pins(recipe)
        assert {name: pins[name] for name in (
            "torch", "torchaudio", "transformers", "huggingface-hub", "safetensors", "numpy",
            "scipy", "mir-eval", "pretty-midi", "mido", "setuptools",
        )} == {
            "torch": "2.8.0", "torchaudio": "2.8.0", "transformers": "4.45.2",
            "huggingface-hub": "0.36.0", "safetensors": "0.5.3", "numpy": "1.24.3",
            "scipy": "1.13.1", "mir-eval": "0.8.2", "pretty-midi": "0.2.10", "mido": "1.3.3",
            "setuptools": "78.1.1",
        }
        assert "pretty_midi" in jobenv.SMOKE_IMPORT["score"][backend_kind]
    cuda = jobenv.recipe_pins(jobenv.recipe_for(jobenv.worker_env("score", "cuda-linux")))
    mac = jobenv.recipe_pins(jobenv.recipe_for(jobenv.worker_env("score", "mlx-darwin")))
    assert "triton" in cuda and "triton" not in mac
    assert {k: v for k, v in cuda.items() if k in mac} == mac


def test_install_on_submit_names_the_score_installer_and_its_recipe() -> None:
    assert installonsubmit._installer_of("score", "/home/x/envs/score") == "score"
    recipe = jobenv.recipe_for(jobenv.worker_env("score", "cuda-linux"))
    assert installonsubmit._env_bytes("score", None, "cuda-linux") == jobenv.recipe_archive_bytes(recipe)


@pytest.mark.parametrize("backend_kind", ["cuda-linux", "mlx-darwin"])
def test_the_capability_row_says_this_host_can_transcribe_songs(backend_kind: str) -> None:
    total = 24 * GIB if backend_kind == "cuda-linux" else 64 * GIB
    decided = verdict.decide(
        BY_NAME["score"], backend_kind, total_bytes=total, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia" if backend_kind == "cuda-linux" else "apple", chosen=None,
        audio_low_vram=False,
    )
    assert decided.enabled is True, decided.reason
    assert decided.summary.startswith("can transcribe a song into a score, using sheetsage2")


def test_score_runs_in_wsl_and_never_on_windows() -> None:
    assert "score" in verdict.WSL_ONLY_JOB_TYPES


def test_the_catalog_lists_sheetsage2_as_a_model_of_the_score_job(home: Path) -> None:
    configure_box(home, enable_score=True)
    config = load_config(home)
    for backend in (FAKE_BACKEND, FAKE_MAC_BACKEND):
        rows = {s.id: s for s in catalog.subjects(config, backend) if s.job_type == "score"}
        assert sorted(rows) == ["sheetsage2"]
        subject = rows["sheetsage2"]
        assert subject.kind == "model" and subject.installed() is None
        assert subject.pull_command == "crucible models pull sheetsage2"
        assert subject.source == "hf:m-a-p/SheetSage2"
    assert catalog.backends_declaring("model", "sheetsage2") == ["cuda-linux", "mlx-darwin"]
    assert "sheetsage2" in catalog.declared_ids()["model"]


def test_the_worker_env_is_offline_and_lets_the_mac_fall_back_to_the_cpu(tmp_path: Path) -> None:
    python = tmp_path / "envs" / "score" / "bin" / "python"
    mac = score_job.worker_environment(python, "mlx-darwin")
    pc = score_job.worker_environment(python, "cuda-linux")
    assert mac["PYTORCH_ENABLE_MPS_FALLBACK"] == "1" and "PYTORCH_ENABLE_MPS_FALLBACK" not in pc
    for environment in (mac, pc):
        assert environment["HF_HUB_OFFLINE"] == "1" and environment["TRANSFORMERS_OFFLINE"] == "1"


def test_the_desktop_packages_screen_has_words_for_score() -> None:
    assert JOB_TYPE_WORDS["score"][0] == "Song transcription"


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


def test_init_records_the_score_flag(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-score"]) == 0
    assert "score:    enabled" in capsys.readouterr().out
    assert load_config(home).enable_score is True


def test_a_config_without_the_flag_reads_it_as_off(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    assert load_config(home).enable_score is False
