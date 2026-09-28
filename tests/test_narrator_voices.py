from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from crucible.narratorengines import DOCUMENT_READERS, NARRATOR_ENGINE_SAMPLING
from crucible.narratorvoices import (
    DOCUMENT_NAME,
    DOCUMENT_VARIABLE,
    MLX_MODEL_VARIABLE,
    NarratorVoicesError,
    document_path,
    take_sampling,
    voice_entry,
    write_document,
)
from crucible.voicecatalog import load_all_voices
from crucible.voicereference import ClipEntry, parse_reference, reference_path
from crucible.voices import parse_voice

from .conftest import wav_bytes
from .test_voices import GOOD

CUDA = "cuda-linux"
MLX = "mlx-darwin"

BOTH_ARMS = GOOD.replace(
    """safe_min_chars = 600
safe_max_chars = 800
""",
    "",
) + """
[voice.backends.mlx-darwin]
hf_repo = "owenmorgan/probe-higgs-v3"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 12_133_000_000
estimate_basis = "measured"
max_chars = 800
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
"""


def manifest_of(text: str, voice_id: str = "probe"):
    return parse_voice(
        text.replace('id = "probe"', f'id = "{voice_id}"'),
        Path(f"{voice_id}.toml"),
        voice_id,
    )


def test_a_checkpoint_entry_is_the_manifest_spelled_as_narrator_reads_it(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(GOOD)
    weights = tmp_path / "voices" / "probe" / CUDA
    entry = voice_entry(manifest, manifest.spec(CUDA), weights)
    assert entry == {
        "kind": "checkpoint",
        "checkpointDir": str(weights),
        "maxChars": 800,
        "safeMinChars": 600,
        "safeMaxChars": 800,
        "sampling": {"temperature": 0.8, "topP": 0.95, "topK": 50},
        "paceCharsPerSec": 16.0,
        "maxCharsPerSec": 20.8,
        "minCharsPerSec": 12.3,
    }


def test_top_k_is_a_whole_number_because_narrator_refuses_a_float_one() -> None:
    manifest = manifest_of(GOOD)
    entry = voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    assert entry["sampling"]["topK"] == 50
    assert isinstance(entry["sampling"]["topK"], int)
    assert isinstance(entry["sampling"]["temperature"], float)


def test_a_voice_declaring_no_band_writes_none(tmp_path: Path) -> None:
    manifest = manifest_of(BOTH_ARMS)
    entry = voice_entry(manifest, manifest.spec(CUDA), tmp_path)
    for key in ("safeMinChars", "safeMaxChars", "targetChars"):
        assert key not in entry


def test_a_voice_that_measured_no_cap_sends_narrator_no_maxChars(
    tmp_path: Path,
) -> None:
    uncapped = BOTH_ARMS.replace("max_chars = 800\n", "")
    manifest = manifest_of(uncapped)
    assert manifest.spec(CUDA).max_chars is None
    entry = voice_entry(manifest, manifest.spec(CUDA), tmp_path)
    assert "maxChars" not in entry
    capped = manifest_of(BOTH_ARMS)
    assert voice_entry(capped, capped.spec(CUDA), tmp_path)["maxChars"] == 800


def test_a_target_travels_when_the_manifest_declares_one() -> None:
    text = BOTH_ARMS.replace(
        "min_chars_per_sec = 12.3\n", "min_chars_per_sec = 12.3\ntarget_chars = 600\n"
    )
    manifest = manifest_of(text)
    entry = voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    assert entry["targetChars"] == 600
    assert "safeMinChars" not in entry


def test_nothing_narrator_does_not_read_is_written() -> None:
    manifest = manifest_of(GOOD)
    entry = voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    for key in (
        "maxCharsSource", "scene", "allowedControls", "maxReferenceSeconds",
        "clips", "_overrideNote",
    ):
        assert key not in entry


def test_a_sampling_lever_narrator_has_no_name_for_is_refused() -> None:
    manifest = manifest_of(GOOD)
    spec = manifest.spec(CUDA)
    odd = type(spec)(**{**spec.__dict__, "sampling": {"temperature": 0.8, "min_p": 0.1}})
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, odd, Path("/w"))
    assert "'min_p'" in str(caught.value)
    assert "topK" in str(caught.value)


def test_a_token_voice_is_narrators_default_kind_on_the_mlx_arm(
    tmp_path: Path,
) -> None:
    text = BOTH_ARMS.replace('kind = "checkpoint"', 'kind = "token"')
    manifest = manifest_of(text)
    weights = tmp_path / "voices" / "probe" / MLX
    document = write_document(tmp_path, manifest, manifest.spec(MLX), weights)
    entry = document.voices["probe"]
    assert entry["kind"] == "default"
    assert "checkpointDir" not in entry
    assert entry["maxChars"] == 800
    assert document.environment() == {
        DOCUMENT_VARIABLE: str(document.path),
        MLX_MODEL_VARIABLE: str(weights),
    }
    assert document.weights_for("probe") == weights


def test_a_token_voice_on_cuda_linux_is_refused_by_name(tmp_path: Path) -> None:
    text = BOTH_ARMS.replace('kind = "checkpoint"', 'kind = "token"')
    manifest = manifest_of(text)
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, manifest.spec(CUDA), tmp_path)
    assert "HuggingFace cache" in str(caught.value)
    assert "HIGGS_MODEL_DIR" in str(caught.value)
    assert "RULING OWED" in str(caught.value)
    assert MLX in str(caught.value)


def test_a_checkpoint_voice_on_the_mlx_arm_never_sets_the_base_weights(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(BOTH_ARMS)
    weights = tmp_path / "voices" / "probe" / MLX
    document = write_document(tmp_path, manifest, manifest.spec(MLX), weights)
    assert document.voices["probe"]["checkpointDir"] == str(weights)
    assert MLX_MODEL_VARIABLE not in document.environment()
    assert document.base_weights is None


ZEROSHOT = BOTH_ARMS.replace('kind = "checkpoint"', 'kind = "zeroshot"').replace(
    "max_chars = 800\n", 'max_chars = 800\nclips = "from-request"\n'
)


def a_clip(tmp_path: Path) -> ClipEntry:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF....WAVE")
    return ClipEntry(path=path, transcript="He had been walking.", seconds=8.4)


def test_a_zeroshot_entry_is_narrators_clips_kind_with_the_placed_clip(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(ZEROSHOT)
    weights = tmp_path / "voices" / "probe" / CUDA
    clip = a_clip(tmp_path)
    entry = voice_entry(manifest, manifest.spec(CUDA), weights, clip)
    assert entry["kind"] == "clips"
    assert entry["checkpointDir"] == str(weights)
    assert entry["clips"] == [
        {
            "path": str(clip.path),
            "transcript": "He had been walking.",
            "seconds": 8.4,
        }
    ]


def test_a_zeroshot_entry_never_calls_its_base_weights_a_checkpoint(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(ZEROSHOT)
    entry = voice_entry(
        manifest, manifest.spec(CUDA), tmp_path / "w", a_clip(tmp_path)
    )
    assert entry["kind"] == "clips" != "checkpoint"
    assert "checkpointDir" in entry, (
        "the directory is still named — it is the pinned base weights, and "
        "omitting it is how a server comes up on an HF-cache snapshot"
    )


def test_a_zeroshot_voice_with_no_clip_is_refused_by_name(tmp_path: Path) -> None:
    manifest = manifest_of(ZEROSHOT)
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, manifest.spec(CUDA), tmp_path)
    assert "zeroshot voice and no reference clip" in str(caught.value)


def test_a_clip_on_a_voice_that_is_not_zeroshot_is_refused_by_name(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(BOTH_ARMS)
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, manifest.spec(CUDA), tmp_path, a_clip(tmp_path))
    assert "reference clip was placed for it" in str(caught.value)


def test_write_document_places_the_wav_beside_the_document(tmp_path: Path) -> None:
    manifest = manifest_of(ZEROSHOT)
    reference = parse_reference(
        {"data": base64.b64encode(wav_bytes(2.0)).decode("ascii"),
         "transcript": "He had been walking.", "name": "stranger"}
    )
    document = write_document(
        tmp_path, manifest, manifest.spec(CUDA), tmp_path / "w", reference
    )
    clip = document.voices["probe"]["clips"][0]
    assert Path(clip["path"]) == reference_path(tmp_path)
    assert Path(clip["path"]).read_bytes() == reference.audio
    assert clip["seconds"] == pytest.approx(2.0)
    assert clip["transcript"] == "He had been walking."


LADDER = BOTH_ARMS + """
[[voice.takes]]

[[voice.takes]]
temperature = 0.7
reason = "measured: a different draw, not a better setting"
"""


def test_take_zero_sends_no_sampling_at_all() -> None:
    assert take_sampling(manifest_of(LADDER), 0) is None
    assert take_sampling(manifest_of(BOTH_ARMS), 0) is None


def test_a_rung_carries_only_the_keys_it_declares() -> None:
    assert take_sampling(manifest_of(LADDER), 1) == {"temperature": 0.7}


def test_a_rung_is_spelled_the_way_the_document_spells_it() -> None:
    text = LADDER.replace(
        'temperature = 0.7\nreason', 'temperature = 0.7\ntop_p = 0.9\ntop_k = 40\nreason'
    )
    assert take_sampling(manifest_of(text), 1) == {
        "temperature": 0.7, "topP": 0.9, "topK": 40,
    }
    assert isinstance(take_sampling(manifest_of(text), 1)["topK"], int)


def test_a_take_past_the_ladder_sends_no_sampling_key_at_all() -> None:
    assert take_sampling(manifest_of(LADDER), 2) is None
    assert take_sampling(manifest_of(LADDER), 1) == {"temperature": 0.7}


def test_the_document_readers_are_the_rule_the_writer_refuses_from() -> None:
    assert DOCUMENT_READERS == frozenset({"higgs-v3"})
    assert DOCUMENT_READERS >= set(NARRATOR_ENGINE_SAMPLING)


def test_the_document_is_one_file_per_server_carrying_the_loaded_voice(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(GOOD)
    weights = tmp_path / "voices" / "probe" / CUDA
    document = write_document(tmp_path, manifest, manifest.spec(CUDA), weights)
    assert document.path == tmp_path / DOCUMENT_NAME == document_path(tmp_path)
    on_disk = json.loads(document.path.read_text(encoding="utf-8"))
    assert on_disk == document.voices
    assert list(on_disk) == ["probe"]
    assert on_disk["probe"]["checkpointDir"] == str(weights)


def test_the_next_load_overwrites_the_last(tmp_path: Path) -> None:
    first = manifest_of(GOOD, "first")
    second = manifest_of(GOOD, "second")
    write_document(tmp_path, first, first.spec(CUDA), tmp_path / "a")
    document = write_document(tmp_path, second, second.spec(CUDA), tmp_path / "b")
    on_disk = json.loads(document.path.read_text(encoding="utf-8"))
    assert list(on_disk) == ["second"]
    assert on_disk["second"]["checkpointDir"] == str(tmp_path / "b")


def test_a_voice_the_document_does_not_carry_is_named_with_the_ones_it_does(
    tmp_path: Path,
) -> None:
    manifest = manifest_of(GOOD)
    document = write_document(tmp_path, manifest, manifest.spec(CUDA), tmp_path)
    with pytest.raises(NarratorVoicesError) as caught:
        document.entry("sigma")
    assert "carries no voice 'sigma'" in str(caught.value)
    assert "['probe']" in str(caught.value)
    assert str(document.path) in str(caught.value)


BAND_KEYS = ("paceCharsPerSec", "maxCharsPerSec", "minCharsPerSec")


def _assert_band_is_the_manifest(entry: dict, manifest) -> None:
    if manifest.pace.pace_chars_per_sec is None:
        for key in BAND_KEYS:
            assert key not in entry, f"{manifest.id} sends an unmeasured {key}"
        return
    assert entry["paceCharsPerSec"] == manifest.pace.pace_chars_per_sec
    assert entry["maxCharsPerSec"] == manifest.pace.max_chars_per_sec
    assert entry["minCharsPerSec"] == manifest.pace.min_chars_per_sec


def test_the_unmeasured_voices_send_narrator_no_band_at_all(tmp_path: Path) -> None:
    clip = a_clip(tmp_path)
    catalog = load_all_voices()
    for voice_id in ("higgs-default", "zeroshot"):
        manifest = catalog[voice_id]
        assert manifest.pace.pace_chars_per_sec is None, voice_id
        for backend, spec in sorted(manifest.backends.items()):
            try:
                entry = voice_entry(
                    manifest, spec, tmp_path / voice_id / backend,
                    clip if manifest.kind == "zeroshot" else None,
                )
            except NarratorVoicesError:
                continue
            for key in BAND_KEYS:
                assert key not in entry, f"{voice_id}/{backend} sends {key}"


def test_every_shipped_higgs_voice_writes_an_entry_on_the_arm_it_can(
    tmp_path: Path,
) -> None:
    refused: list[tuple[str, str]] = []
    written: list[tuple[str, str]] = []
    clip = a_clip(tmp_path)
    for voice_id, manifest in sorted(load_all_voices().items()):
        if manifest.narrator_engine not in DOCUMENT_READERS:
            continue
        for backend, spec in sorted(manifest.backends.items()):
            weights = tmp_path / "voices" / voice_id / backend
            try:
                entry = voice_entry(
                    manifest, spec, weights,
                    clip if manifest.kind == "zeroshot" else None,
                )
            except NarratorVoicesError:
                refused.append((voice_id, backend))
                continue
            written.append((voice_id, backend))
            if spec.max_chars is None:
                assert "maxChars" not in entry, voice_id
            else:
                assert entry["maxChars"] == spec.max_chars
            _assert_band_is_the_manifest(entry, manifest)
            assert entry["sampling"] == {
                "temperature": spec.sampling["temperature"],
                "topP": spec.sampling["top_p"],
                "topK": int(spec.sampling["top_k"]),
            }
            if manifest.kind in ("checkpoint", "zeroshot"):
                assert entry["checkpointDir"] == str(weights)
    assert written, "no Higgs voice in the catalog wrote an entry"
    assert ("zeroshot", CUDA) in written and ("zeroshot", MLX) in written
    for voice_id, backend in refused:
        manifest = load_all_voices()[voice_id]
        assert manifest.kind == "token" and backend == CUDA, (voice_id, backend)
