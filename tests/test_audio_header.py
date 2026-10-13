"""A planned YuE2 song's key, meter and tempo fixed by the client: `key`, `meter`, `bpm`
(docs/AUDIO.md "Key, meter and tempo")."""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from crucible.audiomodels import load_all_audio_manifests
from crucible.errors import ApiError
from crucible.jobs.audio import cover, scoreheader
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
SCORES = sorted((HERE / "fixtures" / "yue2-scores").glob("*.abc"))
TAGS = "Instrumental, slow somber piano ballad, strings, 66 BPM"
SUNG_TAGS = "English, waltz, warm piano, expressive female voice, 76 BPM"
WORDS = "[Verse]\nThe kettle sings the morning in\nThe window holds the rain\n\n[Chorus]\nStay a while\n"
PLANNING = "[Verse]\nA lantern burning by the gate\nThe evening settles on the stone\n"


def _check(model: str, **params: Any) -> AudioParams:
    manifest = load_all_audio_manifests()[model]
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, manifest.spec("cuda-linux"))
    return parsed


def _settled(**params: Any):
    return settle(_check(SONG, **params), load_all_audio_manifests()[SONG].spec("cuda-linux"), 7)


# --- what may be fixed, and the score read back --------------------------------------------


def test_the_keys_are_the_skills_own() -> None:
    sys.path.insert(0, str(WORKER.parent / "yue2music"))
    try:
        import abc_tools
    finally:
        sys.path.remove(str(WORKER.parent / "yue2music"))
    assert scoreheader.KEYS == tuple(abc_tools.KEYS)
    for meter in ("4/4", "3/4", "6/8", "12/8", "2/2", "7/8", "16/16"):
        assert abc_tools.meter_value(scoreheader.check_meter(meter)), meter


@pytest.mark.parametrize("path", SCORES, ids=lambda p: p.stem)
def test_every_real_score_reads_back_its_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    read = scoreheader.read(text)
    assert read == {"key": lines[7][2:], "meter": lines[2][2:], "bpm": int(lines[4].split("=")[1])}
    assert read["key"] in scoreheader.KEYS, "YuE2 writes only the skill's keys"
    # The layout the fixed lines are written at: the native dialect's header.
    for name, index in scoreheader.LINE.items():
        assert lines[index][0] == {"key": "K", "meter": "M", "bpm": "Q"}[name]


def test_the_reader_takes_the_first_of_each_and_stops_at_the_body() -> None:
    assert scoreheader.read("X:1\nM:3/4\nQ:3/8=60\nK:Em\n% intro\nK:G\n") == {
        "key": "Em", "meter": "3/4", "bpm": None}
    assert scoreheader.read("X:1\nT:\n% intro\nK:G\n") == {"key": None, "meter": None, "bpm": None}
    assert scoreheader.read("") is None and scoreheader.read(None) is None


def test_the_lines_each_field_writes() -> None:
    assert scoreheader.lines(None) == {}
    assert scoreheader.lines({"key": "F#m", "meter": None, "bpm": None}) == {7: "K:F#m"}
    assert scoreheader.lines({"key": "Em", "meter": "3/4", "bpm": 76}) == {
        2: "M:3/4", 4: "Q:1/4=76", 7: "K:Em"}


# --- the request ---------------------------------------------------------------------------


@pytest.mark.parametrize(("param", "value"), [
    ("key", "Em"), ("key", "F#m"), ("key", "Bb"), ("key", "D"), ("key", "C#"),
    ("meter", "3/4"), ("meter", "6/8"), ("meter", "4/4"), ("meter", "12/8"),
    ("bpm", 76), ("bpm", 40), ("bpm", 240),
])
def test_each_alone_is_taken_and_settled(param: str, value: Any) -> None:
    settled = _settled(tags=SUNG_TAGS, lyrics=WORDS, **{param: value})
    assert settled.header == {"key": None, "meter": None, "bpm": None, param: value}


def test_none_sent_settles_no_header_and_all_three_settle_together() -> None:
    assert _settled(tags=SUNG_TAGS, lyrics=WORDS).header is None
    assert _settled(tags=TAGS, instrumental=True, key="Em", meter="3/4", bpm=76).header == {
        "key": "Em", "meter": "3/4", "bpm": 76}


@pytest.mark.parametrize(("param", "value", "words"), [
    ("key", "E minor", "not a key"),
    ("key", "em", "not a key"),
    ("key", "Edor", "no modes"),
    ("key", "Fb", "not a key"),
    ("key", "", "not a key"),
    ("meter", "3:4", "not a meter"),
    ("meter", "4/3", "not a meter"),
    ("meter", "17/8", "not a meter"),
    ("meter", "C", "not a meter"),
    ("bpm", 39, "greater than or equal to 40"),
    ("bpm", 241, "less than or equal to 240"),
    ("bpm", 76.0, "must be an integer"),
    ("bpm", "76", "must be an integer"),
    ("bpm", True, "must be an integer"),
])
def test_bad_syntax_is_refused_by_name(param: str, value: Any, words: str) -> None:
    with pytest.raises(ValidationError) as caught:
        AudioParams(tags=SUNG_TAGS, lyrics=WORDS, **{param: value})
    (error,) = caught.value.errors()
    assert error["loc"] == (param,)
    assert words in str(caught.value)


@pytest.mark.parametrize("param", ["key", "meter", "bpm"])
@pytest.mark.parametrize("instrumental", [True, False])
def test_a_cover_brings_its_own_header_so_each_is_refused_beside_a_score(
    param: str, instrumental: bool
) -> None:
    value = {"key": "Em", "meter": "3/4", "bpm": 76}[param]
    sent = dict(tags=TAGS, instrumental=True) if instrumental else dict(tags=SUNG_TAGS, lyrics=WORDS)
    with pytest.raises(ApiError) as caught:
        _check(SONG, score="score.abc", **sent, **{param: value})
    assert caught.value.code == "audio_param_conflict"
    assert caught.value.details["param"] == param
    assert "header already states" in caught.value.message


@pytest.mark.parametrize("model", ["stable-audio-3-medium", "stable-audio-3-small-sfx"])
@pytest.mark.parametrize(("param", "value"), [("key", "Em"), ("meter", "3/4"), ("bpm", 76)])
def test_stable_audio_refuses_them_by_name(model: str, param: str, value: Any) -> None:
    with pytest.raises(ApiError) as caught:
        _check(model, prompt="a sad piano piece", **{param: value})
    assert caught.value.code == "audio_param_unsupported"
    assert caught.value.details["param"] == param
    assert "writes no score" in caught.value.message


def test_the_manifest_lists_them_on_both_arms() -> None:
    spec = load_all_audio_manifests()[SONG]
    for backend in ("cuda-linux", "mlx-darwin"):
        assert {"key", "meter", "bpm"} <= set(spec.spec(backend).takes)


# --- the worker, against a yue2-infer stand-in ---------------------------------------------
#
# yue2-infer 0.1.6's scoring pass, for the parts the worker touches: pipe.plan builds the
# SymbolicPlan from the tokens yue2.sampling.generate_tokens draws, and generate_tokens
# draws each one through the module-level `distribution` - the name forced_header replaces.
# The tokenizer is one token a character; the model is scripted to write a real score
# (BALLAD) line by line: at line n, column c, it writes line n's character c, so after a
# fixed line it carries on with the score's next line - the model-written parts continue.

EOD, ABC_START, ABC_END, MUSIC_START = 151643, 151847, 151848, 151851
VOCAB = 128
END = VOCAB - 1  # the stand-in's end of score (ASCII DEL: never in a score)
INSTRUCTIONS = {
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
        return 1.0 if self.cfg_scale is None else self.cfg_scale

    def text(self):
        return f"{INSTRUCTIONS[self.cot]}\n[Tags]\n{self.style}\n[Lyrics]\n{self.lyrics}\n"


def token_prefixes(request, tokenizer, abc_ids=None):
    base = [EOD] + tokenizer.encode(request.text())
    if abc_ids is None:
        if request.abc is None:
            return base + [ABC_START]
        abc_ids = tokenizer.encode(request.abc)
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


SAMPLING_SOURCE = '''
import functools


def _no_grad(function):
    # The real generate_tokens is wrapped by torch.no_grad (functools.wraps), so its own
    # __code__ is the wrapper's; the worker's check must look through it (2026-10-13: it did
    # not, and refused every real yue2-infer).
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def distribution(logits, sampling, history, step, phase, legacy_off=False):
    scores = logits.astype(float)
    if step < sampling.min_tokens:
        scores[..., END] = -numpy.inf
    return scores


@_no_grad
def generate_tokens(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, cancelled=None, on_token=None, use_cuda_graph=True):
    history, eos = [], False
    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during " + phase)
        scores = distribution(model(prefix, history), sampling, history, step, phase, legacy_off)
        token = int(scores.argmax())
        if on_token is not None:
            on_token(phase, token)
        if token == END:
            eos = True
            break
        history.append(token)
    count = len(history) + int(eos)
    timing = {"seconds": 1.0, "prefill_seconds": 0.1, "output_tokens": count, "output_tps": count,
              "prefix_tokens": len(prefix), "cfg_branches": 1, "execution": "eager",
              "attention": "sdpa"}
    return history, timing, not eos
'''


class ScriptedModel:
    """Writes `score` line by line: at line n, column c, line n's character c (or its
    newline); END after the last line, or as soon as `stop_at` characters are written."""

    def __init__(self, score: str, stop_at: int | None = None) -> None:
        self.lines = score.splitlines(keepends=True)
        self.stop_at = stop_at
        self.prefixes: list[list[int]] = []

    def __call__(self, prefix, history) -> numpy.ndarray:
        if not history:
            self.prefixes.append(list(prefix))
        text = "".join(chr(t) for t in history)
        line, column = text.count("\n"), len(text) - (text.rfind("\n") + 1)
        logits = numpy.zeros(VOCAB)
        if line >= len(self.lines) or (self.stop_at is not None and len(history) >= self.stop_at):
            logits[END] = 10.0
        else:
            written = self.lines[line]
            logits[ord(written[column]) if column < len(written) else ord("\n")] = 10.0
        return logits


class FakePipe:
    """YuE2Pipeline as the worker drives it, with pipe.plan as yue2-infer writes it."""

    backend = "torch"

    def __init__(self, model: ScriptedModel | None = None, cancel: bool = False) -> None:
        self.tokenizer = Tokenizer()
        self.model = model or ScriptedModel(BALLAD)
        self.generation_config = SimpleNamespace(
            abc=SimpleNamespace(max_tokens=20000, min_tokens=32),
            semantic=SimpleNamespace(max_tokens=9000), context=24576)
        self.plans: list[Any] = []
        self.composed: list[Any] = []
        self.cancel = cancel
        self.distribution_during_plan: list[Any] = []

    def plan(self, tags, lyrics, *, abc=None, cot="full", seed, cfg_scale, cancelled=None, on_token=None):
        sampling = sys.modules["yue2.sampling"]
        request = SongRequest(style=tags, lyrics=lyrics, cot=cot, seed=seed, abc=abc, cfg_scale=cfg_scale)
        if abc is not None:
            ids = self.tokenizer.encode(abc)
            plan = SimpleNamespace(request=request, abc=abc, abc_ids=ids,
                                   prefix=token_prefixes(request, self.tokenizer, ids),
                                   timing={"seconds": 0.0, "output_tokens": 0}, truncated=False)
        else:
            self.distribution_during_plan.append(sampling.distribution)
            ids, timing, truncated = sampling.generate_tokens(
                self.model, token_prefixes(request, self.tokenizer),
                self.generation_config.abc, seed, "abc",
                cancelled=lambda: self.cancel or (cancelled is not None and cancelled()),
                on_token=on_token)
            plan = SimpleNamespace(request=request, abc=self.tokenizer.decode(ids), abc_ids=ids,
                                   prefix=token_prefixes(request, self.tokenizer, ids),
                                   timing=timing, truncated=truncated)
        self.plans.append(plan)
        return plan

    def generate_semantic(self, plan, **kwargs):
        # yue2-infer's own check, which a plan built any other way would fail.
        if token_prefixes(plan.request, self.tokenizer, plan.abc_ids) != plan.prefix:
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        self.composed.append(plan)
        return SimpleNamespace(timing={"seconds": 1.0, "prefill_seconds": 0.1, "output_tokens": 10,
                                       "output_tps": 10.0, "prefix_tokens": len(plan.prefix),
                                       "cfg_branches": 1, "execution": "eager", "attention": "sdpa"},
                               truncated=False)

    def synthesize(self, semantic, cancelled=None):
        return "latents"

    def decode(self, latents):
        return "audio"


@pytest.fixture
def yue2(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    protocol = types.ModuleType("yue2.protocol")
    protocol.EOD = EOD
    protocol.token_prefixes = token_prefixes
    sampling = types.ModuleType("yue2.sampling")
    # yue2.sampling imports torch; full_like is the one call the forcer makes on it.
    sampling.__dict__.update(torch=numpy, numpy=numpy, END=END)
    exec(SAMPLING_SOURCE, sampling.__dict__)
    package = types.ModuleType("yue2")
    package.__path__ = []
    package.protocol, package.sampling = protocol, sampling
    for name, module in (("yue2", package), ("yue2.protocol", protocol), ("yue2.sampling", sampling)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(protocol=protocol, sampling=sampling, stock=sampling.distribution)


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
    siblings = {"planning": planning, "scorelength": scorelength, "audiocore": audiocore,
                "scoreheader": scoreheader}
    workerio.load_sibling = lambda name, _file: siblings[name]
    spec = importlib.util.spec_from_file_location("yue2_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress:
    asked_to_stop = False

    def __init__(self) -> None:
        self.notes: list[str] = []

    def enter(self, stage, steps=None) -> None:
        pass

    def note(self, text) -> None:
        self.notes.append(text)

    def reached(self, step, steps=None) -> None:
        pass


def _job(tmp_path: Path, header: dict | None, **fields: Any) -> SimpleNamespace:
    base = dict(tags=SUNG_TAGS, lyrics=WORDS, planning_lyrics=None, instrumental=False, seed=7,
                cfg=1.0, min_duration_s=None, max_duration_s=None, longest_s=360,
                planning_resizable=False, cover=None, negative=None,
                output_path=str(tmp_path / "audio.flac"))
    full = None if header is None else {"key": None, "meter": None, "bpm": None, **header}
    return SimpleNamespace(**{**base, "header": full, **fields})


def _render(module: Any, job: SimpleNamespace, pipe: FakePipe | None = None, *,
            low_vram: bool = False) -> tuple[Any, FakePipe, Progress]:
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = pipe or FakePipe()
    engine.low_vram = low_vram
    engine._close_stage = lambda peaks, name: None
    progress = Progress()
    engine._stages(job, progress, {})
    return engine, engine._pipe, progress


def _replaced(score: str, lines: dict[int, str]) -> str:
    """`score` with its header lines `lines` replaced: what the forced pass must write."""
    written = score.splitlines(keepends=True)
    for index, line in lines.items():
        written[index] = line + "\n"
    return "".join(written)


HEADERS = {
    "key": {"key": "Em"},
    "meter": {"meter": "3/4"},
    "bpm": {"bpm": 76},
    "all three": {"key": "Em", "meter": "3/4", "bpm": 76},
}


@pytest.mark.parametrize("which", list(HEADERS))
def test_the_fixed_header_lands_in_the_score_and_the_model_writes_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace, which: str
) -> None:
    module = _load_worker(monkeypatch)
    header = HEADERS[which]
    engine, pipe, progress = _render(module, _job(tmp_path, header))
    (plan,) = pipe.plans
    fixed = scoreheader.lines({"key": None, "meter": None, "bpm": None, **header})

    assert plan.abc == _replaced(BALLAD, fixed), "only the fixed lines differ from what YuE2 wrote"
    assert plan.abc.splitlines()[8:] == BALLAD.splitlines()[8:], "the body is the model's"
    assert plan.abc.splitlines()[3] == "L:1/32", "the unit length is the model's"
    expected = {"key": "Dm", "meter": "4/4", "bpm": 66, **header}
    assert engine.score_header == expected
    # One pass: one prefix, prefilled once, and the plan is pipe.plan's own.
    assert len(pipe.model.prefixes) == 1 and pipe.model.prefixes[0][-1] == ABC_START
    assert plan.prefix == token_prefixes(plan.request, Tokenizer(), plan.abc_ids)
    assert pipe.composed == [plan], "generate_semantic's prefix check passed"
    assert engine.decode_stages["scoring"]["tokens"] == len(plan.abc_ids) + 1
    assert any("header is fixed" in note for note in progress.notes)
    assert yue2.sampling.distribution is yue2.stock, "the stock name is put back"
    assert pipe.distribution_during_plan[0] is not yue2.stock


def test_unsent_plans_exactly_as_before_with_the_stock_distribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    engine, pipe, progress = _render(module, _job(tmp_path, None))
    (plan,) = pipe.plans
    assert plan.abc == BALLAD
    assert pipe.distribution_during_plan == [yue2.stock], "nothing is replaced"
    assert engine.score_header == {"key": "Dm", "meter": "4/4", "bpm": 66}
    assert not progress.notes


def test_a_forced_token_still_goes_through_the_stock_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    seen: list[tuple[int, int]] = []
    stock = yue2.sampling.distribution

    def watched(logits, sampling, history, step, phase, legacy_off=False):
        seen.append((step, int(numpy.isfinite(logits).sum())))
        return stock(logits, sampling, history, step, phase, legacy_off)

    monkeypatch.setattr(yue2.sampling, "distribution", watched)
    _, pipe, _ = _render(module, _job(tmp_path, {"key": "F#m"}))
    k_line = pipe.plans[0].abc.index("K:F#m")
    forced_steps = range(k_line, k_line + len("K:F#m\n"))
    assert [finite for step, finite in seen if step in forced_steps] == [1] * len("K:F#m\n"), \
        "a forced step reaches the stock distribution with only its token finite"
    assert all(finite == VOCAB for step, finite in seen if step not in forced_steps)
    assert yue2.sampling.distribution is watched


def test_low_vram_scores_through_the_same_pass_with_the_same_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    """[audio] low_vram has no scoring stage of its own: the worker's `load` puts the AR half
    on the card and pipe.plan runs as ever, so the fixed header is written the same way."""
    module = _load_worker(monkeypatch)
    header = HEADERS["all three"]
    whole, whole_pipe, _ = _render(module, _job(tmp_path, header))
    halved, halved_pipe, _ = _render(module, _job(tmp_path, header), low_vram=True)
    assert halved_pipe.plans[0].abc == whole_pipe.plans[0].abc
    assert halved_pipe.plans[0].abc_ids == whole_pipe.plans[0].abc_ids
    assert halved.score_header == whole.score_header == {"key": "Em", "meter": "3/4", "bpm": 76}
    assert halved.decode_stages["scoring"]["low_vram"] is True


def test_an_instrumental_moves_the_melody_under_the_fixed_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    """The Vocal-to-Ins transfer runs on the constrained score; the fixed score YuE2 then
    renders keeps its key and tempo, and the length is read from it as for any song."""
    module = _load_worker(monkeypatch)
    job = _job(tmp_path, {"key": "Gm", "bpm": 72}, tags=TAGS, lyrics=None,
               planning_lyrics=PLANNING, instrumental=True)
    engine, pipe, _ = _render(module, job)
    planned, fixed = pipe.plans
    assert planned.abc == _replaced(BALLAD, {4: "Q:1/4=72", 7: "K:Gm"})
    assert planned.request.lyrics == PLANNING and fixed.request.abc is not None
    assert engine.notes["score_before_transfer"] == planned.abc
    assert pipe.composed == [fixed]
    assert engine.score_header == {"key": "Gm", "meter": "4/4", "bpm": 72}
    assert engine.length["score"]["bpm"] == 72, "the length is read at the fixed tempo"


def test_a_score_that_ends_before_its_k_line_is_recorded_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    pipe = FakePipe(ScriptedModel(BALLAD, stop_at=40))  # past min_tokens, inside the header
    engine, pipe, progress = _render(module, _job(tmp_path, {"meter": "3/4", "key": "Em"}), pipe)
    (plan,) = pipe.plans
    assert plan.abc.startswith("X:1\nT:\nM:3/4\n") and "K:Em" not in plan.abc
    assert len(pipe.model.prefixes) == 1, "one pass: nothing re-plans"
    assert engine.score_header["meter"] == "3/4" and engine.score_header["key"] is None
    assert any("K:Em" in note and "audio.header.score" in note for note in progress.notes)


def test_a_cancelled_scoring_pass_puts_the_stock_distribution_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    with pytest.raises(InterruptedError):
        _render(module, _job(tmp_path, {"key": "Em"}), FakePipe(cancel=True))
    assert yue2.sampling.distribution is yue2.stock


def test_a_yue2_infer_that_draws_another_way_is_refused_before_a_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    elsewhere = dict(yue2.sampling.__dict__)
    exec(SAMPLING_SOURCE.replace("scores = distribution(", "scores = sample_scores("), elsewhere)
    monkeypatch.setattr(yue2.sampling, "generate_tokens", elsewhere["generate_tokens"])
    pipe = FakePipe()
    with pytest.raises(RuntimeError, match="draws its tokens some other way"):
        _render(module, _job(tmp_path, {"key": "Em"}), pipe)
    assert pipe.model.prefixes == [] and yue2.sampling.distribution is yue2.stock

    vllm = FakePipe()
    vllm.backend = "vllm"
    with pytest.raises(RuntimeError, match="'vllm'"):
        _render(module, _job(tmp_path, {"key": "Em"}), vllm)


def test_a_cover_with_a_header_is_refused_in_the_worker_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    made = cover.prepare(BALLAD, "score.abc", instrumental=True, lyrics=None, chords=None,
                         longest_s=360)
    pipe = FakePipe()
    with pytest.raises(RuntimeError, match="refuses key, meter and bpm beside a score"):
        _render(module, _job(tmp_path, {"key": "Em"}, tags=TAGS, lyrics=None, instrumental=True,
                             cover=made.request()), pipe)
    assert pipe.plans == []


def test_a_cover_reads_back_its_own_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, yue2: SimpleNamespace
) -> None:
    module = _load_worker(monkeypatch)
    made = cover.prepare(BALLAD, "score.abc", instrumental=True, lyrics=None, chords=None,
                         longest_s=360)
    engine, pipe, _ = _render(module, _job(tmp_path, None, tags=TAGS, lyrics=None,
                                           instrumental=True, cover=made.request()))
    assert engine.score_header == {"key": "Dm", "meter": "4/4", "bpm": 66}
    assert pipe.distribution_during_plan == []


# --- through the API, with the fake worker --------------------------------------------------


def test_the_record_says_what_was_fixed_and_what_the_score_states(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    import json

    params = {"tags": SUNG_TAGS, "lyrics": WORDS, "seed": 3, "key": "Em", "meter": "3/4", "bpm": 76}
    _, events = run_job(ready, auth, model=SONG, params=params)
    assert events[-1]["event"] == "done", events[-1]
    header = events[-1]["data"]["audio"]["header"]
    assert header == {"key": "Em", "meter": "3/4", "bpm": 76,
                      "score": {"key": "Em", "meter": "3/4", "bpm": 76}}
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [r for r in rows if r.get("op") == "generate"][-1]["header"] == {
        "key": "Em", "meter": "3/4", "bpm": 76}

    _, events = run_job(ready, auth, model=SONG, params={"tags": SUNG_TAGS, "lyrics": WORDS, "bpm": 80})
    assert events[-1]["data"]["audio"]["header"] == {
        "key": None, "meter": None, "bpm": 80, "score": {"key": "C", "meter": "4/4", "bpm": 80}}

    _, events = run_job(ready, auth, model=SONG, params={"tags": SUNG_TAGS, "lyrics": WORDS})
    assert events[-1]["data"]["audio"]["header"] == {
        "key": None, "meter": None, "bpm": None, "score": {"key": "C", "meter": "4/4", "bpm": None}}
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [r for r in rows if r.get("op") == "generate"][-1]["header"] is None, "none sent: none sent on"


def test_the_kept_request_carries_the_header_and_a_sound_records_none(
    ready: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "1")
    job_id, events = run_job(ready, auth, model=SONG, params={
        "tags": SUNG_TAGS, "lyrics": WORDS, "key": "F#m"})
    assert events[-1]["event"] == "failed"
    kept = json.loads((home / "jobs" / job_id / "request.json").read_text(encoding="utf-8"))
    assert kept["settled"]["header"] == {"key": "F#m", "meter": None, "bpm": None}
    assert kept["params"]["key"] == "F#m"

    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "0")
    _, events = run_job(ready, auth)
    assert events[-1]["data"]["audio"]["header"] is None


def test_the_door_refuses_bad_syntax_a_cover_and_stable_audio(
    ready: TestClient, auth: dict[str, str]
) -> None:
    error = refusal(submit(ready, auth, model=SONG, params={
        "tags": SUNG_TAGS, "lyrics": WORDS, "key": "E minor"}))
    assert error["code"] == "invalid_params" and "key" in error["message"]
    error = refusal(submit(ready, auth, model=SONG, params={
        "tags": SUNG_TAGS, "lyrics": WORDS, "bpm": 300}))
    assert error["code"] == "invalid_params" and "bpm" in error["message"]
    error = refusal(submit(ready, auth, model=SONG, params={
        "tags": TAGS, "instrumental": True, "score": "score.abc", "meter": "3/4"}))
    assert error["code"] == "audio_param_conflict" and error["details"]["param"] == "meter"
    error = refusal(submit(ready, auth, params={"prompt": "door", "bpm": 90}))
    assert error["code"] == "audio_param_unsupported"
