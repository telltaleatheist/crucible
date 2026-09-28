from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import weights
from crucible.voicecatalog import load_all_voices
from crucible.voices import VoiceError


def test_zeroshot_stores_in_higgs_defaults_folder(tmp_path: Path) -> None:
    voices = load_all_voices()
    zeroshot, base = voices["zeroshot"], voices["higgs-default"]
    assert zeroshot.weights_of == "higgs-default"
    assert zeroshot.weights_base is not None and zeroshot.weights_base.id == "higgs-default"
    config = SimpleNamespace(home=tmp_path)
    for backend in ("cuda-linux", "mlx-darwin"):
        assert weights.subject_dir(config, zeroshot, backend) == weights.subject_dir(
            config, base, backend
        )


BASE_VOICE = """
[voice]
id = "probe"
display = "Probe"
kind = "checkpoint"
narrator_engine = "higgs-v3"
language = "en"
sample_rate = 24000

[voice.pace]
pace_chars_per_sec = 16.0
max_chars_per_sec = 20.8
min_chars_per_sec = 12.3
safe_min_chars = 600
safe_max_chars = 800

[voice.serving]
max_num_seqs = 16
max_num_seqs_note = "vllm-omni's own stage-0 value, and a measured ceiling at 0.35 + 0.10."

[voice.backends.cuda-linux]
hf_repo = "owenmorgan/probe-higgs-v3"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 19_000_000_000
estimate_basis = "measured"
max_chars = 800
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
"""


def test_a_voice_alias_pinned_to_the_same_bytes_shares_them(tmp_path: Path) -> None:
    (tmp_path / "probe.toml").write_text(BASE_VOICE, encoding="utf-8")
    alias = BASE_VOICE.replace('id = "probe"', 'id = "probe-two"\nweights_of = "probe"', 1)
    (tmp_path / "probe-two.toml").write_text(alias, encoding="utf-8")
    voices = load_all_voices(tmp_path)
    assert voices["probe-two"].weights_base is not None
    assert voices["probe-two"].weights_base.id == "probe"


def test_a_voice_alias_pinned_to_other_bytes_is_refused(tmp_path: Path) -> None:
    (tmp_path / "probe.toml").write_text(BASE_VOICE, encoding="utf-8")
    alias = BASE_VOICE.replace('id = "probe"', 'id = "probe-two"\nweights_of = "probe"', 1)
    alias = alias.replace(
        'revision = "0123456789abcdef0123456789abcdef01234567"',
        'revision = "' + "0" * 40 + '"',
    )
    (tmp_path / "probe-two.toml").write_text(alias, encoding="utf-8")
    with pytest.raises(VoiceError, match="weights_of_pin_mismatch"):
        load_all_voices(tmp_path)
