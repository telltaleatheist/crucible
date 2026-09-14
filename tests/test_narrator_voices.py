"""`crucible/narratorvoices.py` — the voices document narrator reads.

PHASE3-TTS.md section 4. Crucible's first real Higgs v3 render died at the
`load` on both arms (2026-09-14): the message carried `modelDir`, which
narrator refuses by name, and narrator then looks the voice up in a
`NARRATOR_HIGGS_VOICES` document Crucible had never written. So the document
is written from the manifest and the pulled directory at every load, and these
tests pin every field it carries to the manifest value it came from — field by
field and verbatim, because each one is a number BookForge measured and a
wrong one is a whole book packed, guarded or sampled wrong and reported as
success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.narratorvoices import (
    DOCUMENT_NAME,
    DOCUMENT_READERS,
    DOCUMENT_VARIABLE,
    MLX_MODEL_VARIABLE,
    NarratorVoicesError,
    document_path,
    voice_entry,
    write_document,
)
from crucible.voices import load_all_voices, parse_voice

from .test_voices import GOOD, as_orpheus

CUDA = "cuda-linux"
MLX = "mlx-darwin"

#: A manifest with an `mlx-darwin` block as well, and no safe band, so both
#: arms and the "declares neither" pace shape are reachable.
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


# ------------------------------------------------------------- the fields


def test_a_checkpoint_entry_is_the_manifest_spelled_as_narrator_reads_it(
    tmp_path: Path,
) -> None:
    """Every key is one `engine/higgs/config.py:load_voices` reads, in the
    catalog's camelCase, and every value is the manifest's. `checkpointDir` is
    the pulled directory: the voice's weights ARE the voice, and the server is
    started on that directory."""
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
    """The manifest loader makes every sampling value a float (`_number`);
    narrator's `_voice_sampling` refuses `topK 50.0` by name because top_k is
    a count of candidates. The count goes back to being one on the way out."""
    manifest = manifest_of(GOOD)
    entry = voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    assert entry["sampling"]["topK"] == 50
    assert isinstance(entry["sampling"]["topK"], int)
    assert isinstance(entry["sampling"]["temperature"], float)


def test_a_voice_declaring_no_band_writes_none(tmp_path: Path) -> None:
    """narrator's `_safe_band` takes each edge independently and packs to
    `maxChars` with neither; a key that is absent in the manifest is absent on
    the wire rather than written as null, which `load_voices` would read as
    'declared, and not a number'."""
    manifest = manifest_of(BOTH_ARMS)
    entry = voice_entry(manifest, manifest.spec(CUDA), tmp_path)
    for key in ("safeMinChars", "safeMaxChars", "targetChars"):
        assert key not in entry


def test_a_target_travels_when_the_manifest_declares_one() -> None:
    text = BOTH_ARMS.replace(
        "min_chars_per_sec = 12.3\n", "min_chars_per_sec = 12.3\ntarget_chars = 600\n"
    )
    manifest = manifest_of(text)
    entry = voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    assert entry["targetChars"] == 600
    assert "safeMinChars" not in entry


def test_nothing_narrator_does_not_read_is_written() -> None:
    """`maxCharsSource`, `scene`, `allowedControls`, `maxReferenceSeconds`,
    `clips`, `_overrideNote`: each is read by narrator, and each is deliberately
    absent — the module docstring says why, one by one."""
    manifest = manifest_of(GOOD)
    entry = voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    for key in (
        "maxCharsSource", "scene", "allowedControls", "maxReferenceSeconds",
        "clips", "_overrideNote",
    ):
        assert key not in entry


def test_a_sampling_lever_narrator_has_no_name_for_is_refused() -> None:
    """narrator's document takes temperature / topP / topK / repetitionPenalty
    and nothing else. A manifest cannot state a fourth (the loader replaces the
    table wholesale against the engine's three), so this reaches the writer
    only through a spec assembled in code — and is refused there rather than
    passed through to a refusal inside a started worker."""
    manifest = manifest_of(GOOD)
    spec = manifest.spec(CUDA)
    odd = type(spec)(**{**spec.__dict__, "sampling": {"temperature": 0.8, "min_p": 0.1}})
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, odd, Path("/w"))
    assert "'min_p'" in str(caught.value)
    assert "topK" in str(caught.value)


# ---------------------------------------------------------------- the kinds


def test_a_token_voice_is_narrators_default_kind_on_the_mlx_arm(
    tmp_path: Path,
) -> None:
    """Crucible's `token` is narrator's `default` — the model's own voice, no
    `checkpointDir`. On `mlx-darwin` the base weights come from
    `NARRATOR_HIGGS3_MLX_MODEL` (`model_dir = checkpoint or
    model_dir_from_env()`), so the document sets it to the pulled directory."""
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
    """narrator's served launcher serves a default voice from the HuggingFace
    cache, not from a directory Crucible names, and a server started on other
    bytes would render under this voice's fingerprint. RULING OWED on narrator's
    side; the refusal names it."""
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
    """A checkpoint voice reads `checkpointDir` and never `MODEL_ENV`; setting
    the variable would be a lever read by nothing."""
    manifest = manifest_of(BOTH_ARMS)
    weights = tmp_path / "voices" / "probe" / MLX
    document = write_document(tmp_path, manifest, manifest.spec(MLX), weights)
    assert document.voices["probe"]["checkpointDir"] == str(weights)
    assert MLX_MODEL_VARIABLE not in document.environment()
    assert document.base_weights is None


def test_a_zeroshot_voice_is_refused_by_name(tmp_path: Path) -> None:
    """The render door refuses the kind as `voice_kind_unsupported`; this is
    the same refusal at the load door, before an entry naming files Crucible
    has not checked reaches narrator."""
    text = BOTH_ARMS.replace('kind = "checkpoint"', 'kind = "zeroshot"').replace(
        "max_chars = 800\n", 'max_chars = 800\nclips = "from-request"\n'
    )
    manifest = manifest_of(text)
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, manifest.spec(CUDA), tmp_path)
    assert "zeroshot" in str(caught.value)
    assert "section 6" in str(caught.value)


def test_an_orpheus_manifest_gets_no_document() -> None:
    """orpheus takes its weights on the load message and reads no
    NARRATOR_HIGGS_* variable. The set of readers is the rule the residency
    branches on, and the writer refuses anything outside it."""
    assert DOCUMENT_READERS == frozenset({"higgs-v3"})
    manifest = manifest_of(
        as_orpheus().replace(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.6, top_p = 0.8, min_p = 0.0, "
            "repetition_penalty = 1.1 }",
        )
    )
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(manifest, manifest.spec(CUDA), Path("/w"))
    assert "'orpheus'" in str(caught.value)
    assert DOCUMENT_VARIABLE in str(caught.value)


# ----------------------------------------------------------------- the file


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
    """One voice per narrator process, one entry per document: the previous
    load's voice is exactly the stale thing a post-mortem must not find."""
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


# -------------------------------------------------------- the real catalog


def test_every_shipped_higgs_voice_writes_an_entry_on_the_arm_it_can(
    tmp_path: Path,
) -> None:
    """The catalog as shipped, through the real writer, so a manifest that
    grew a shape the writer cannot state fails here rather than at the first
    load of that voice. Which (voice, arm) pairs are refused is stated, not
    tolerated: zeroshot on either arm, a token voice on cuda-linux."""
    refused: list[tuple[str, str]] = []
    written: list[tuple[str, str]] = []
    for voice_id, manifest in sorted(load_all_voices().items()):
        if manifest.narrator_engine not in DOCUMENT_READERS:
            continue
        for backend, spec in sorted(manifest.backends.items()):
            weights = tmp_path / "voices" / voice_id / backend
            try:
                entry = voice_entry(manifest, spec, weights)
            except NarratorVoicesError:
                refused.append((voice_id, backend))
                continue
            written.append((voice_id, backend))
            assert entry["maxChars"] == spec.max_chars
            assert entry["paceCharsPerSec"] == manifest.pace.pace_chars_per_sec
            assert entry["sampling"] == {
                "temperature": spec.sampling["temperature"],
                "topP": spec.sampling["top_p"],
                "topK": int(spec.sampling["top_k"]),
            }
            if manifest.kind == "checkpoint":
                assert entry["checkpointDir"] == str(weights)
    assert written, "no Higgs voice in the catalog wrote an entry"
    for voice_id, backend in refused:
        manifest = load_all_voices()[voice_id]
        assert manifest.kind == "zeroshot" or (
            manifest.kind == "token" and backend == CUDA
        ), (voice_id, backend)
