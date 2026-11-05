"""GET /v1/voices says how each voice's chunks join, so an assembler reads it and keeps no copy."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from crucible import voicecatalog
from crucible.jobs.tts.common import voice_edges
from crucible.narratorengines import HIGGS_V3, NARRATOR_ENGINE_EDGES

from .conftest import configure_box

WORKER = Path(__file__).resolve().parent.parent / "crucible/jobs/tts/higgs/crucible_higgs/engine/higgs"


def _constant(name: str) -> float:
    found = re.findall(rf"^{name} = ([0-9.]+)$", (WORKER / "v3_served.py").read_text(encoding="utf-8"), re.M)
    assert len(found) == 1, name
    return float(found[0])


def test_the_engine_row_is_the_workers_own_numbers() -> None:
    edges = NARRATOR_ENGINE_EDGES[HIGGS_V3]
    assert edges["fade_in_ms"] == _constant("EDGE_FADE_IN_MS")
    assert edges["fade_out_ms"] == _constant("EDGE_FADE_OUT_MS")
    engine = (WORKER / "v3_engine.py").read_text(encoding="utf-8")
    assert re.search(r"^    pads = False$", engine, re.M) and edges["pads"] is False


def test_a_voice_stating_no_fades_gets_the_engines(home: Path) -> None:
    configure_box(home, enable_tts=True)
    voice = voicecatalog.engine_voices()["higgs-default"]
    edges = voice_edges(voice, voice.spec("cuda-linux"))
    assert edges == {"pads": False, "fade_in_ms": 10.0, "fade_out_ms": 25.0}


def test_a_voice_stating_its_own_fades_gets_those(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_box(home, enable_tts=True)
    voice = voicecatalog.engine_voices()["higgs-default"]
    spec = voice.spec("cuda-linux")
    facts = {**spec.to_dict(), "edge_fade_ms": {"in": 5.0, "out": 40.0}}
    stated = type("Spec", (), {"to_dict": lambda self: facts})()
    assert voice_edges(voice, stated) == {"pads": False, "fade_in_ms": 5.0, "fade_out_ms": 40.0}
