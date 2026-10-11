"""A finished YuE2 job says, per token stage, how many tokens it wrote, whether the model
ended it or it ran to its cap, the path it ran on and how fast - so a runaway is read off
the job, not found by re-running the seed (Victoria's 6-minute song, RTX 3070, 2026-10-09:
composing ran 8960 of 9000 and the score's stage ran to its cap)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from crucible.jobs.audio import made_words, stages_at_cap

from .test_audio_instrumental import SCORE, TAGS, _load_worker

LYRICS = "[Verse]\nThe kettle sings the morning in\n"


def _timing(tokens: int, *, execution: str = "cuda_graph") -> dict[str, Any]:
    """yue2.sampling.generate_tokens' timing, every key it writes (yue2-infer at the
    commit crucible/envs/audio/yue2-*.txt pins)."""
    return {
        "seconds": 40.123, "prefill_seconds": 0.456, "ttft_seconds": 0.5,
        "output_tokens": tokens, "content_tokens": tokens - 1, "output_tps": tokens / 40.123,
        "prefix_tokens": 212, "cfg_branches": 1, "execution": execution,
        "attention": "sdpa" if execution == "eager" else "flash",
    }


class Pipe:
    """YuE2Pipeline as the worker drives it, its two token stages scripted."""

    def __init__(self, *, score_tokens: int, score_capped: bool, song_tokens: int, song_capped: bool,
                 execution: str = "cuda_graph") -> None:
        self.generation_config = SimpleNamespace(
            abc=SimpleNamespace(max_tokens=4096), semantic=SimpleNamespace(max_tokens=9000)
        )
        self._score = (score_tokens, score_capped)
        self._song = (song_tokens, song_capped)
        self._execution = execution

    def plan(self, tags, lyrics, **kwargs):
        if "abc" in kwargs:  # the instrumental's fixed score: no decode
            return SimpleNamespace(abc=kwargs["abc"], truncated=False,
                                   timing={"seconds": 0.0, "output_tokens": 0})
        tokens, capped = self._score
        return SimpleNamespace(abc=SCORE, truncated=capped, timing=_timing(tokens, execution=self._execution))

    def generate_semantic(self, plan, **kwargs):
        tokens, capped = self._song
        return SimpleNamespace(timing=_timing(tokens, execution=self._execution), truncated=capped)

    def synthesize(self, semantic, **kwargs):
        return "latents"

    def decode(self, latents):
        return "samples"


class Progress:
    asked_to_stop = False

    def __init__(self) -> None:
        self.entered: list[tuple[str, Any]] = []

    def enter(self, stage, steps=None):
        self.entered.append((stage, steps))


def _engine(monkeypatch: pytest.MonkeyPatch, pipe: Pipe, *, low_vram: bool) -> Any:
    module = _load_worker(monkeypatch)
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = pipe
    engine.low_vram = low_vram
    engine._close_stage = lambda peaks, name: peaks.__setitem__(name, 1)
    return module, engine


def _job(**overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(**{"tags": TAGS, "lyrics": LYRICS, "planning_lyrics": None, "seed": 7,
                              "cfg": 1.0, "instrumental": False, "min_duration_s": None,
                              "max_duration_s": None, "longest_s": 360,
                              "planning_resizable": False, **overrides})


def test_each_token_stage_says_how_it_ended_from_yue2s_own_account(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = Pipe(score_tokens=4096, score_capped=True, song_tokens=4321, song_capped=False)
    _, engine = _engine(monkeypatch, pipe, low_vram=True)
    progress = Progress()
    engine._stages(_job(), progress, {})

    scoring, composing = engine.decode_stages["scoring"], engine.decode_stages["composing"]
    assert list(engine.decode_stages) == ["scoring", "composing"]
    assert scoring == {
        "tokens": 4096, "cap": 4096, "ended": "cap", "execution": "cuda_graph",
        "attention": "flash", "low_vram": True, "prefix_tokens": 212, "cfg_branches": 1,
        "seconds": 40.12, "prefill_seconds": 0.46, "tokens_per_second": 102.1,
    }
    assert (composing["tokens"], composing["cap"], composing["ended"]) == (4321, 9000, "eos")
    # The caps are yue2-infer's own, read off the pipeline - not a copy of them here.
    assert progress.entered[:2] == [("scoring", 4096), ("composing", 9000)]


def test_a_song_that_ended_itself_has_no_stage_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = Pipe(score_tokens=1180, score_capped=False, song_tokens=8960, song_capped=False,
                execution="eager")
    _, engine = _engine(monkeypatch, pipe, low_vram=False)
    engine._stages(_job(), Progress(), {})
    assert {s["ended"] for s in engine.decode_stages.values()} == {"eos"}
    assert {s["execution"] for s in engine.decode_stages.values()} == {"eager"}
    assert engine.decode_stages["composing"]["tokens"] == 8960, "8960 of 9000 is close, not capped"
    assert stages_at_cap(engine.decode_stages) == []


def test_an_instrumental_reports_the_score_yue2_decoded_not_the_fixed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe(score_tokens=1500, score_capped=False, song_tokens=9000, song_capped=True)
    _, engine = _engine(monkeypatch, pipe, low_vram=False)
    engine._stages(_job(lyrics=None, planning_lyrics=LYRICS, instrumental=True), Progress(), {})
    assert engine.decode_stages["scoring"]["tokens"] == 1500
    assert engine.decode_stages["composing"]["ended"] == "cap"
    assert engine.notes["planned_score"] == SCORE


def test_a_new_song_starts_with_no_facts_from_the_last(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing(Pipe):
        def generate_semantic(self, plan, **kwargs):
            raise InterruptedError("cancelled during composing")

    _, engine = _engine(monkeypatch, Pipe(score_tokens=10, score_capped=False, song_tokens=10,
                                          song_capped=False), low_vram=False)
    engine._stages(_job(), Progress(), {})
    engine._pipe = Failing(score_tokens=20, score_capped=False, song_tokens=0, song_capped=False)
    with pytest.raises(InterruptedError):
        engine._stages(_job(), Progress(), {})
    assert list(engine.decode_stages) == ["scoring"] and engine.decode_stages["scoring"]["tokens"] == 20


def test_the_done_record_names_the_stages_at_cap_and_says_so_in_words() -> None:
    stages = {
        "scoring": {"ended": "cap", "cap": 4096},
        "composing": {"ended": "cap", "cap": 9000},
    }
    assert stages_at_cap(stages) == ["scoring", "composing"]
    assert stages_at_cap(None) is None, "an engine that decodes no tokens has nothing to cap"
    assert made_words(360.0, stages) == (
        "360.0 s of audio made; scoring at its 4096-token cap, composing at its 9000-token "
        "cap without ending (see audio.decode_stages)"
    )
    assert made_words(12.5, None) == "12.5 s of audio made"
    assert made_words(200.0, {"composing": {"ended": "eos", "cap": 9000}}) == "200.0 s of audio made"
