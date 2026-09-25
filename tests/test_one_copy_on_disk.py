"""One copy of every set of weights on disk (Owen, 2026-09-24: "lets reduce it to
a single copy of everything").

`zeroshot` and `higgs-default` sit on the Higgs base checkpoint at one pin and
had each pulled 9.3 GB of it. A voice may now say `weights_of`, like a model or
an asr model, and `zeroshot` says `higgs-default`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import weights
from crucible.voices import VoiceError, load_all_voices


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


def test_a_voice_alias_pinned_to_other_bytes_is_refused(tmp_path: Path) -> None:
    source = load_all_voices()["sigma"].path.read_text(encoding="utf-8")
    (tmp_path / "sigma.toml").write_text(source, encoding="utf-8")
    alias = source.replace('id = "sigma"', 'id = "sigma-two"\nweights_of = "sigma"', 1)
    revision_line = next(line for line in alias.splitlines() if line.startswith("revision"))
    alias = alias.replace(revision_line, 'revision = "' + "0" * 40 + '"')
    (tmp_path / "sigma-two.toml").write_text(alias, encoding="utf-8")
    with pytest.raises(VoiceError, match="weights_of_pin_mismatch"):
        load_all_voices(tmp_path)
