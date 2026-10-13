"""A YuE2 song's CFG negative branch written by the client: `negative_tags` and
`negative_lyrics` (docs/AUDIO.md "The negative branch")."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible.audiomodels import load_all_audio_manifests
from crucible.errors import ApiError
from crucible.jobs.audio import cover
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take, settle

from .test_audio_api import (  # noqa: F401  (fixtures)
    SONG,
    idle_card,
    ready,
    refusal,
    run_job,
    submit,
    transcript,
)

HERE = Path(__file__).resolve().parent
WORKER = HERE.parent / "crucible" / "jobs" / "audio" / "yue2_worker.py"
AUDIOCORE = WORKER.parent / "audiocore.py"
BALLAD = (HERE / "fixtures" / "yue2-score-ballad.abc").read_text(encoding="utf-8")
TAGS = "Instrumental, slow somber piano ballad, strings, 66 BPM"
SUNG_TAGS = "English, warm piano ballad, expressive female voice, 66 BPM"
WORDS = "[Verse]\nThe kettle sings the morning in\nThe window holds the rain\n\n[Chorus]\nStay a while\n"
PLANNING = "[Verse]\nA lantern burning by the gate\nThe evening settles on the stone\n"
NEG_TAGS = "lead vocals, female vocals, male vocals, singing, choir, spoken word, piano ballad."
NEG_LYRICS = "[Verse]\nA lantern burning by the gate\n"
# tests/fake_audio_worker.py's stock negative prefix (importing it would claim stdout).
FAKE_STOCK_NEGATIVE_TOKENS = 600


def _check(model: str, **params: Any) -> AudioParams:
    manifest = load_all_audio_manifests()[model]
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, manifest.spec("cuda-linux"))
    return parsed


def _settled(**params: Any):
    return settle(_check(SONG, **params), load_all_audio_manifests()[SONG].spec("cuda-linux"), 7)


# --- the request ---------------------------------------------------------------------------


def test_unsent_is_the_stock_negative_and_one_sent_alone_leaves_the_other_empty() -> None:
    assert _settled(tags=SUNG_TAGS, lyrics=WORDS, cfg=2.0).negative is None
    assert _settled(tags=SUNG_TAGS, lyrics=WORDS).negative is None, "cfg 1 and no negative: as before"
    assert _settled(tags=SUNG_TAGS, lyrics=WORDS, cfg=2.0, negative_tags=NEG_TAGS).negative == {
        "tags": NEG_TAGS, "lyrics": ""}
    assert _settled(tags=TAGS, instrumental=True, cfg=1.5, negative_lyrics=NEG_LYRICS).negative == {
        "tags": "", "lyrics": NEG_LYRICS}
    both = _settled(tags=TAGS, instrumental=True, cfg=2.0, negative_tags=NEG_TAGS,
                    negative_lyrics=NEG_LYRICS)
    assert both.negative == {"tags": NEG_TAGS, "lyrics": NEG_LYRICS}


@pytest.mark.parametrize(
    ("cfg", "param", "words"),
    [
        (None, "negative_tags", "yue2-3b's default cfg is 1"),
        (1.0, "negative_tags", "cfg is 1"),
        (0.5, "negative_lyrics", "cfg is 0.5"),
    ],
)
def test_a_written_negative_without_cfg_above_one_is_refused_by_name(
    cfg: float | None, param: str, words: str
) -> None:
    sent = {param: NEG_TAGS if param == "negative_tags" else NEG_LYRICS}
    if cfg is not None:
        sent["cfg"] = cfg
    with pytest.raises(ApiError) as caught:
        _check(SONG, tags=SUNG_TAGS, lyrics=WORDS, **sent)
    assert caught.value.code == "audio_param_conflict", caught.value.message
    assert words in caught.value.message and "above 1" in caught.value.message
    assert caught.value.details["param"] == param
    assert caught.value.details["cfg"] == (1.0 if cfg is None else cfg)


def test_any_yue2_song_takes_it_sung_instrumental_or_cover() -> None:
    _check(SONG, tags=SUNG_TAGS, lyrics=WORDS, cfg=2.0, negative_tags=NEG_TAGS)
    _check(SONG, tags=TAGS, instrumental=True, cfg=2.0, negative_lyrics=NEG_LYRICS)
    _check(SONG, tags=TAGS, instrumental=True, score="score.abc", cfg=2.0, negative_tags=NEG_TAGS)
    _check(SONG, tags=SUNG_TAGS, lyrics=WORDS, score="score.abc", cfg=2.0, negative_lyrics=NEG_LYRICS)


@pytest.mark.parametrize("model", ["stable-audio-3-medium", "stable-audio-3-small-sfx"])
@pytest.mark.parametrize("param", ["negative_tags", "negative_lyrics"])
def test_stable_audio_refuses_it_by_name(model: str, param: str) -> None:
    with pytest.raises(ApiError) as caught:
        _check(model, prompt="a sad piano piece", **{param: "singing"})
    assert caught.value.code == "audio_param_unsupported"
    assert caught.value.details["param"] == param
    assert "no CFG negative branch" in caught.value.message


def test_an_empty_negative_is_refused_not_read_as_stock() -> None:
    with pytest.raises(Exception, match="is empty"):
        AudioParams(tags=SUNG_TAGS, negative_tags="  ")


def test_the_manifest_lists_them_in_takes_and_an_arm_without_cfg_cannot() -> None:
    from crucible.audiomodels import AudioManifestError, _check_takes

    spec = load_all_audio_manifests()[SONG]
    for backend in ("cuda-linux", "mlx-darwin"):
        takes = spec.spec(backend).takes
        assert "negative_tags" in takes and "negative_lyrics" in takes and "cfg" in takes
    with pytest.raises(AudioManifestError, match="takes 'cfg' too"):
        _check_takes("x.toml", {"takes": ["negative_tags"]})


# --- the worker, against a yue2-infer stand-in ---------------------------------------------
#
# The protocol below is yue2-infer 0.1.6's (yue2/protocol.py) for the parts the worker
# touches: the prefixes, SongRequest and its guidance. The pipeline's generate_semantic
# builds its negative exactly as the real one does, by calling the module's
# `negative_prefix` name, and records what each branch was.

EOD, ABC_START, ABC_END, MUSIC_START = 151643, 151847, 151848, 151851
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


@dataclasses.dataclass(frozen=True)
class SongRequest:
    style: str
    lyrics: str
    cot: str = "full"
    seed: int = 831001
    abc: str | None = None
    cfg_scale: float | None = None
    id: str = "song"

    @property
    def guidance(self):
        return (1.01 if self.cot == "off" else 1.0) if self.cfg_scale is None else self.cfg_scale

    def text(self):
        return f"{INSTRUCTIONS[self.cot]}\n[Tags]\n{self.style}\n[Lyrics]\n{self.lyrics}\n"


def token_prefixes(request, tokenizer, abc_ids=None):
    base = [EOD] + tokenizer.encode(request.text())
    if abc_ids is None:
        abc_ids = tokenizer.encode(request.abc)
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def negative_prefix(request, tokenizer, abc_ids=None):
    if abc_ids is None:
        raise ValueError("Symbolic CFG must retain the exact positive-branch ABC IDs")
    return [EOD] + tokenizer.encode(INSTRUCTIONS[request.cot]) + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


class Tokenizer:
    """One token a character, so a prefix can be read back."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


STAGE = {"output_tokens": 4321, "execution": "cuda_graph", "attention": "sdpa",
         "prefix_tokens": 0, "cfg_branches": 2, "seconds": 1.0, "prefill_seconds": 0.1,
         "output_tps": 4321.0}


# yue2/pipeline.py's generate_semantic, run with that module's globals as the real one is,
# so `negative_prefix` is looked up in yue2.pipeline at call time.
PIPELINE_SOURCE = '''
class YuE2Pipeline:
    def generate_semantic(self, plan, *, sampling=None, cancelled=None, on_token=None):
        request = plan.request
        expected = token_prefixes(request, self.tokenizer, plan.abc_ids)
        if expected != plan.prefix:
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        negative = negative_prefix(request, self.tokenizer, plan.abc_ids) if request.guidance != 1 else None
        self.composed.append({"plan": plan, "negative": negative})
        if self.cancel:
            raise InterruptedError
        return SimpleNamespace(timing=STAGE, truncated=False)
'''


class FakePipe:
    """YuE2Pipeline as the worker drives it: plan (planning BALLAD when given no score),
    generate_semantic (yue2.pipeline's, the stand-in above), synthesize and decode."""

    def generate_semantic(self, plan, **kwargs):
        return sys.modules["yue2.pipeline"].YuE2Pipeline.generate_semantic(self, plan, **kwargs)

    def __init__(self, cancel: bool = False) -> None:
        self.tokenizer = Tokenizer()
        self.generation_config = SimpleNamespace(abc=SimpleNamespace(max_tokens=4096),
                                                 semantic=SimpleNamespace(max_tokens=9000),
                                                 context=24576)
        self.composed: list[dict] = []
        self.cancel = cancel

    def plan(self, tags, lyrics, *, abc=None, cot="full", seed, cfg_scale, cancelled=None, on_token=None):
        request = SongRequest(style=tags, lyrics=lyrics, cot=cot, seed=seed, abc=abc, cfg_scale=cfg_scale)
        written = BALLAD if abc is None else abc
        ids = self.tokenizer.encode(written)
        return SimpleNamespace(request=request, abc=written, abc_ids=ids,
                               prefix=token_prefixes(request, self.tokenizer, ids),
                               timing={**STAGE, "output_tokens": len(ids)}, truncated=False)

    def synthesize(self, semantic, cancelled=None):
        return "latents"

    def decode(self, latents):
        return "audio"


@pytest.fixture
def yue2(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    protocol = types.ModuleType("yue2.protocol")
    protocol.negative_prefix = negative_prefix
    protocol.token_prefixes = token_prefixes
    pipeline = types.ModuleType("yue2.pipeline")
    pipeline.__dict__.update(negative_prefix=negative_prefix, token_prefixes=token_prefixes,
                             SimpleNamespace=SimpleNamespace, STAGE=STAGE)
    exec(PIPELINE_SOURCE, pipeline.__dict__)
    package = types.ModuleType("yue2")
    package.__path__ = []
    package.protocol, package.pipeline = protocol, pipeline
    for name, module in (("yue2", package), ("yue2.protocol", protocol), ("yue2.pipeline", pipeline)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(protocol=protocol, pipeline=pipeline)


def _load_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    from crucible.jobs.audio import planning, scorelength

    workerio = types.ModuleType("workerio")
    workerio.claim_stdout = lambda: None
    workerio.send = lambda *args, **kwargs: None
    workerio.write_audio = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "workerio", workerio)
    spec = importlib.util.spec_from_file_location("audiocore_under_test", AUDIOCORE)
    audiocore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audiocore)
    siblings = {"planning": planning, "scorelength": scorelength, "audiocore": audiocore}
    workerio.load_sibling = lambda name, _file: siblings[name]
    spec = importlib.util.spec_from_file_location("yue2_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress:
    asked_to_stop = False

    def enter(self, stage, steps=None) -> None:
        pass

    def note(self, text) -> None:
        pass

    def reached(self, step, steps=None) -> None:
        pass


def _job(tmp_path: Path, **fields: Any) -> SimpleNamespace:
    base = dict(tags=SUNG_TAGS, lyrics=WORDS, planning_lyrics=None, instrumental=False, seed=7,
                cfg=2.0, min_duration_s=None, max_duration_s=None, longest_s=360,
                planning_resizable=False, cover=None, negative=None,
                output_path=str(tmp_path / "audio.flac"))
    return SimpleNamespace(**{**base, **fields})


def _render(module: Any, job: SimpleNamespace, pipe: FakePipe | None = None) -> tuple[Any, dict]:
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = pipe or FakePipe()
    engine.low_vram = False
    engine._close_stage = lambda peaks, name: None
    engine._stages(job, Progress(), {})
    (composed,) = engine._pipe.composed
    return engine, composed


def _written(plan: Any, tags: str, lyrics: str) -> list[int]:
    """The negative the client asked for: the plan's request with only style and lyrics
    replaced, over the plan's own abc ids."""
    request = dataclasses.replace(plan.request, style=tags, lyrics=lyrics)
    return token_prefixes(request, Tokenizer(), plan.abc_ids)


def _abc_ids_in(prefix: list[int]) -> list[int]:
    start = prefix.index(ABC_START)
    return prefix[start + 1:prefix.index(ABC_END, start)]


PATHS = {
    "planned": dict(),
    "instrumental": dict(tags=TAGS, lyrics=None, planning_lyrics=PLANNING, instrumental=True),
    "instrumental cover": dict(tags=TAGS, lyrics=None, instrumental=True),
    "sung cover": dict(),
}


def _path_job(tmp_path: Path, path: str, **fields: Any) -> SimpleNamespace:
    job = _job(tmp_path, **PATHS[path], **fields)
    if path.endswith("cover"):
        made = cover.prepare(BALLAD, "score.abc", instrumental=job.instrumental,
                             lyrics=None if job.instrumental else WORDS, chords=None, longest_s=360)
        job.cover = made.request()
    return job


@pytest.mark.parametrize("path", list(PATHS))
def test_the_written_negative_is_the_song_request_with_its_words_over_the_plan_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace, path: str
) -> None:
    module = _load_worker(monkeypatch)
    job = _path_job(tmp_path, path, negative={"tags": NEG_TAGS, "lyrics": NEG_LYRICS})
    engine, composed = _render(module, job)
    plan = composed["plan"]

    assert composed["negative"] == _written(plan, NEG_TAGS, NEG_LYRICS)
    text = "".join(map(chr, composed["negative"][1:composed["negative"].index(ABC_START)]))
    assert f"[Tags]\n{NEG_TAGS}\n[Lyrics]\n{NEG_LYRICS}\n" in text
    assert text.startswith(INSTRUCTIONS[plan.request.cot]), "the same cot as the song"
    assert _abc_ids_in(composed["negative"]) == plan.abc_ids == _abc_ids_in(plan.prefix), \
        "both branches compose over the plan's exact abc ids"
    assert plan.prefix == token_prefixes(plan.request, Tokenizer(), plan.abc_ids), "positive untouched"
    assert plan.request.style == job.tags and NEG_TAGS not in "".join(map(chr, plan.prefix[1:]))
    assert engine.negative_prefix_tokens == len(composed["negative"])
    assert yue2.pipeline.negative_prefix is negative_prefix, "the stock name is put back"
    if path == "instrumental":
        assert plan.request.lyrics.startswith("[Intro]") and "lantern" not in plan.request.lyrics.lower(), \
            "the negative is built for the fixed instrumental plan, after the transfer"


@pytest.mark.parametrize("path", list(PATHS))
def test_unsent_composes_against_the_stock_negative_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace, path: str
) -> None:
    module = _load_worker(monkeypatch)
    engine, composed = _render(module, _path_job(tmp_path, path))
    plan = composed["plan"]
    assert composed["negative"] == negative_prefix(plan.request, Tokenizer(), plan.abc_ids)
    assert engine.negative_prefix_tokens == len(composed["negative"])

    engine, composed = _render(module, _path_job(tmp_path, path, cfg=1.0))
    assert composed["negative"] is None and engine.negative_prefix_tokens is None


def test_tags_alone_leave_the_negative_lyrics_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    _, composed = _render(module, _job(tmp_path, negative={"tags": NEG_TAGS, "lyrics": ""}))
    assert composed["negative"] == _written(composed["plan"], NEG_TAGS, "")


def test_a_written_negative_at_cfg_one_is_refused_not_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    pipe = FakePipe()
    with pytest.raises(RuntimeError, match="no negative branch"):
        _render(module, _job(tmp_path, cfg=1.0, negative={"tags": NEG_TAGS, "lyrics": ""}), pipe)
    assert pipe.composed == []


def test_a_negative_that_leaves_the_song_no_room_is_refused_by_name_and_put_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    pipe = FakePipe()
    huge = "[Verse]\n" + "la " * 6000 + "\n"  # one token a character here: past 15576
    with pytest.raises(Exception) as raised:
        _render(module, _job(tmp_path, negative={"tags": "", "lyrics": huge}), pipe)
    assert type(raised.value).__name__ == "Refused" and raised.value.code == "negative_too_long"
    assert "leaves 15576" in raised.value.message
    assert pipe.composed == [] and yue2.pipeline.negative_prefix is negative_prefix


def test_a_cancelled_composing_pass_puts_the_stock_negative_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    with pytest.raises(InterruptedError):
        _render(module, _job(tmp_path, negative={"tags": NEG_TAGS, "lyrics": ""}), FakePipe(cancel=True))
    assert yue2.pipeline.negative_prefix is negative_prefix


def test_a_yue2_infer_that_builds_its_negative_another_way_is_refused_before_composing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)

    def generate_semantic(self, plan, **kwargs):
        self.composed.append({"plan": plan, "negative": None})
        return SimpleNamespace(timing=STAGE, truncated=False)

    monkeypatch.setattr(yue2.pipeline.YuE2Pipeline, "generate_semantic", generate_semantic)
    pipe = FakePipe()
    with pytest.raises(RuntimeError, match="builds its negative some other way"):
        _render(module, _job(tmp_path, negative={"tags": NEG_TAGS, "lyrics": ""}), pipe)
    assert pipe.composed == []


# --- through the API, with the fake worker --------------------------------------------------


def _generates(transcript: Path) -> list[dict]:
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "generate"]


def test_the_record_says_which_negative_the_song_composed_against(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    params = {"tags": TAGS, "instrumental": True, "cfg": 2.0, "seed": 3,
              "negative_tags": NEG_TAGS, "negative_lyrics": NEG_LYRICS}
    _, events = run_job(ready, auth, model=SONG, params=params)
    assert events[-1]["event"] == "done", events[-1]
    audio = events[-1]["data"]["audio"]
    assert audio["negative"] == {"tags": NEG_TAGS, "lyrics": NEG_LYRICS}
    assert audio["negative_prefix_tokens"] == FAKE_STOCK_NEGATIVE_TOKENS + len(NEG_TAGS) + len(NEG_LYRICS)
    assert _generates(transcript)[-1]["negative"] == {"tags": NEG_TAGS, "lyrics": NEG_LYRICS}

    _, events = run_job(ready, auth, model=SONG, params={**params, "negative_tags": None,
                                                         "negative_lyrics": None})
    audio = events[-1]["data"]["audio"]
    assert (audio["negative"], audio["negative_prefix_tokens"]) == (None, FAKE_STOCK_NEGATIVE_TOKENS)
    assert _generates(transcript)[-1]["negative"] is None

    _, events = run_job(ready, auth, model=SONG, params={"tags": SUNG_TAGS, "lyrics": WORDS})
    audio = events[-1]["data"]["audio"]
    assert (audio["negative"], audio["negative_prefix_tokens"]) == (None, None), "cfg 1: no branch"


def test_the_kept_request_carries_the_negative_and_sfx_records_none(
    ready: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "1")
    job_id, events = run_job(ready, auth, model=SONG, params={
        "tags": SUNG_TAGS, "lyrics": WORDS, "cfg": 2.0, "negative_lyrics": NEG_LYRICS})
    assert events[-1]["event"] == "failed"
    kept = json.loads((home / "jobs" / job_id / "request.json").read_text(encoding="utf-8"))
    assert kept["settled"]["negative"] == {"tags": "", "lyrics": NEG_LYRICS}
    assert kept["params"]["negative_lyrics"] == NEG_LYRICS

    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "0")
    _, events = run_job(ready, auth)
    audio = events[-1]["data"]["audio"]
    assert (audio["negative"], audio["negative_prefix_tokens"]) == (None, None)


def test_the_door_refuses_a_negative_at_cfg_one_and_on_stable_audio(
    ready: TestClient, auth: dict[str, str]
) -> None:
    error = refusal(submit(ready, auth, model=SONG, params={
        "tags": SUNG_TAGS, "lyrics": WORDS, "negative_tags": NEG_TAGS}))
    assert error["code"] == "audio_param_conflict" and error["details"]["param"] == "negative_tags"
    error = refusal(submit(ready, auth, params={"prompt": "door", "negative_tags": "singing"}))
    assert error["code"] == "audio_param_unsupported"
