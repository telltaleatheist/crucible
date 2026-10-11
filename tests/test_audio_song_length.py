"""A song's length range (docs/AUDIO.md "Song length").

YuE2 has no length input; a song lasts what its score says. Owen (2026-10-10): B-Sides sends
a range to shoot for, `min_duration_s` and `max_duration_s`; Crucible reads the score's
nominal length after the score stage and before composing. An instrumental planned from the
server's pool is re-planned (the score only) with its set grown or cut by whole sections; a
song planned from the client's words is refused `song_length_out_of_range` with the ratio
the words need. Every finished song records `score_seconds` and the range asked."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.audiomodels import load_all_audio_manifests
from crucible.errors import ApiError
from crucible.jobs.audio import planning, scorelength
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take, settle
from crucible.jobs.base import JobFailure

from .conftest import close_queue_session, open_queue_session
from .test_audio_api import SONG, TAGS, idle_card, ready, run_job, transcript
from .test_audio_instrumental import _load_worker

# test_audio_api's fixtures, imported so pytest finds them for this module's API tests.
SHARED_FIXTURES = (idle_card, ready, transcript)

HERE = Path(__file__).resolve().parent
SCORES = HERE / "fixtures" / "yue2-scores"
MEASURED = json.loads((SCORES / "measured.json").read_text(encoding="utf-8"))
REAL = sorted(name for name in MEASURED if not name.endswith("-refused.abc"))
SKILL = HERE.parent / "crucible" / "jobs" / "audio" / "yue2music"
INSTRUMENTAL_TAGS = "Instrumental, warm acoustic guitar folk, light drums, no vocals, 96 BPM"
SUNG = "[Verse]\nThe kettle sings the morning in\nThe window fogs with rain\n\n[Chorus]\nStay a while\n"


def _strict_nominal(text: str) -> float:
    sys.path.insert(0, str(SKILL))
    try:
        from abc_tools import parse_abc, report
    finally:
        sys.path.pop(0)
    return report(parse_abc(text))["nominal_duration_seconds"]


# --- the reader --------------------------------------------------------------------------


@pytest.mark.parametrize("name", REAL)
def test_the_reader_agrees_with_the_skill_on_every_real_score(name: str) -> None:
    """The nine scores YuE2 wrote on the PC (2026-10-10), which the strict parser accepts:
    the tolerant reader gives exactly its nominal_duration_seconds."""
    text = (SCORES / name).read_text(encoding="utf-8")
    assert scorelength.read(text).seconds == pytest.approx(_strict_nominal(text), abs=1e-9)


def test_the_reader_agrees_on_the_ballad_fixture_too() -> None:
    text = (HERE / "fixtures" / "yue2-score-ballad.abc").read_text(encoding="utf-8")
    assert scorelength.read(text).seconds == pytest.approx(_strict_nominal(text), abs=1e-9)


def test_the_nominal_length_predicted_the_audio_within_the_measured_band() -> None:
    """What docs/AUDIO.md states: actual over nominal 0.943 to 1.069, median 0.987."""
    ratios = sorted(
        MEASURED[name]["audio_seconds"] / scorelength.read((SCORES / name).read_text(encoding="utf-8")).seconds
        for name in REAL
    )
    assert len(ratios) == 9
    assert round(ratios[0], 3) == 0.943 and round(ratios[-1], 3) == 1.069
    assert round(ratios[4], 3) == 0.987


def test_a_score_the_strict_parser_refuses_is_still_measured_as_written() -> None:
    """lantern, seed 1000: a 4-quarter bar in 3/4 (group 9, Vocal, bar 33). The skill
    refuses it; the length is still what the notes add up to, the longest voice's."""
    text = (SCORES / "lantern-1000-refused.abc").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds meter duration"):
        _strict_nominal(text)
    reading = scorelength.read(text)
    assert reading.seconds == pytest.approx(155.647, abs=1e-3)
    assert reading.voices["Vocal"] > reading.voices["Ins"]
    assert (reading.bpm, reading.meter) == (85.0, "3/4")


def _score(body: str, header: str = "M:4/4\nL:1/8\nQ:1/4=120\n") -> str:
    return f"X:1\nT:\n{header}K:C\n{body}\n"


def test_note_lengths_rests_and_whole_bar_rests_are_timed_as_abc_writes_them() -> None:
    # At 120 quarters a minute a quarter is 0.5 s; L:1/8 makes a bare note an eighth.
    assert scorelength.read(_score("C2 D2 E2 F2|")).seconds == pytest.approx(2.0)
    assert scorelength.read(_score("C/2 C/ C// C3/2 z|")).seconds == pytest.approx(
        float((Fraction(1, 2) + Fraction(1, 2) + Fraction(1, 4) + Fraction(3, 2) + 1) / 2 * 0.5)
    )
    reading = scorelength.read(_score("Z3|C8|"))
    assert (reading.seconds, reading.bars) == (pytest.approx(8.0), 4)
    assert scorelength.read(_score("[CEG]2 [CEG]4 \"C\"!p!C2|")).seconds == pytest.approx(2.0)


def test_meter_unit_and_tempo_changes_apply_from_where_they_stand() -> None:
    body = "C8|\nM:3/4\nL:1/4\nZ|\nQ:1/4=60\nC3|"
    # 4/4 bar at 120 (2 s), a 3/4 whole-bar rest at 120 (1.5 s), three quarters at 60 (3 s).
    assert scorelength.read(_score(body)).seconds == pytest.approx(6.5)
    assert scorelength.read(_score("C8|[Q:1/4=60]C8|")).seconds == pytest.approx(6.0)
    assert scorelength.read(_score("C4|", header="M:4/4\nL:1/8\nQ:3/8=40\n")).seconds == pytest.approx(
        4 * 0.5 * 60 / 60
    )


@pytest.mark.parametrize(
    ("body", "header", "said"),
    [
        ("C8:|", "M:4/4\nL:1/8\nQ:1/4=120\n", "repeats"),
        ("|:C8|", "M:4/4\nL:1/8\nQ:1/4=120\n", "repeats"),
        ("(3CDE C6|", "M:4/4\nL:1/8\nQ:1/4=120\n", "tuplet"),
        ("C8|", "M:4/4\nL:1/8\n", "no tempo"),
        ("C8 & D8|", "M:4/4\nL:1/8\nQ:1/4=120\n", "does not time"),
        ("C8|", "M:4/4\nL:1/8\nQ:fast\n", "does not say a beat"),
    ],
)
def test_what_the_reader_cannot_time_it_refuses_by_name(body: str, header: str, said: str) -> None:
    with pytest.raises(scorelength.ScoreLengthError, match=said):
        scorelength.read(_score(body, header))


# --- sizing a pool set ---------------------------------------------------------------------


def _set(set_id: str) -> planning.PlanningSet:
    return next(entry for entry in planning.load_pool("yue2") if entry.id == set_id)


def test_every_pool_set_sizes_from_one_section_up_and_keeps_its_own_text_as_written() -> None:
    for entry in planning.load_pool("yue2"):
        sizing = planning.Sizing.of(entry.lyrics)
        assert sizing.sizes[0] == 1 and sizing.written in sizing.sizes
        assert sizing.sizes == tuple(range(1, sizing.sizes[-1] + 1))
        assert sizing.sizes[-1] > sizing.written, f"{entry.id} cannot grow"
        assert sizing.text(sizing.written) == entry.lyrics, "an unresized set is the set verbatim"
        for count in sizing.sizes:
            planning.check(sizing.text(count))


def test_a_set_grows_by_starting_its_body_over_and_shrinks_from_its_end_keeping_its_frame() -> None:
    sizing = planning.Sizing.of(_set("harbor").lyrics)
    assert sizing.structure(sizing.written) == [
        "intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "outro"]
    assert sizing.structure(2) == ["intro", "verse", "pre-chorus", "outro"]
    assert sizing.structure(8) == [
        "intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "verse",
        "pre-chorus", "outro"]
    # ember's body starts and ends on a chorus: carrying it over skips the repeat.
    ember = planning.Sizing.of(_set("ember").lyrics)
    assert ember.structure(6) == ["chorus", "verse", "chorus", "verse", "chorus", "verse"]
    assert "[Chorus]\n" in ember.text(6) and ember.text(6).endswith("\n")


def test_the_first_aim_leaves_a_set_that_fits_as_it_is() -> None:
    sizing = planning.Sizing.of(_set("lantern").lyrics)  # 16 lines, about 138 s at 8.6 s a line
    wanted = planning.LengthRange(100.0, 200.0, 360.0)
    assert planning.aim(sizing, wanted, planning.PRIOR_SECONDS_PER_LINE, set()) == sizing.written
    assert planning.aim(sizing, planning.LengthRange(None, 300.0, 360.0),
                        planning.PRIOR_SECONDS_PER_LINE, set()) == sizing.written


def test_the_aim_grows_or_cuts_the_set_into_the_window_and_never_repeats_a_size() -> None:
    sizing = planning.Sizing.of(_set("lantern").lyrics)
    long = planning.LengthRange(180.0, 270.0, 360.0)
    grown = planning.aim(sizing, long, planning.PRIOR_SECONDS_PER_LINE, set())
    low, high = long.window()
    assert grown > sizing.written and low <= sizing.lines(grown) * 8.6 <= high
    short = planning.LengthRange(None, 90.0, 360.0)
    cut = planning.aim(sizing, short, planning.PRIOR_SECONDS_PER_LINE, set())
    assert cut < sizing.written and sizing.lines(cut) * 8.6 <= 81.0
    assert planning.aim(sizing, long, 8.6, {grown}) != grown
    assert planning.aim(sizing, long, 8.6, set(sizing.sizes)) is None


def test_a_range_too_narrow_for_the_margin_is_aimed_at_its_middle() -> None:
    assert planning.LengthRange(150.0, 160.0, 360.0).window() == (155.0, 155.0)
    assert planning.LengthRange(120.0, 180.0, 360.0).window() == pytest.approx((132.0, 162.0))
    assert planning.LengthRange(120.0, None, 360.0).window() == pytest.approx((132.0, 324.0))


# --- the params ----------------------------------------------------------------------------


def _check(model: str = SONG, **params: Any) -> AudioParams:
    manifest = load_all_audio_manifests()[model]
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, manifest.spec("cuda-linux"))
    return parsed


def _refused(**params: Any) -> ApiError:
    with pytest.raises(ApiError) as caught:
        _check(**params)
    return caught.value


def test_a_range_is_settled_and_either_end_may_come_alone() -> None:
    spec = load_all_audio_manifests()[SONG].spec("cuda-linux")
    for low, high in ((120, 180), (120, None), (None, 180)):
        sent = {k: v for k, v in (("min_duration_s", low), ("max_duration_s", high)) if v is not None}
        settled = settle(_check(tags=TAGS, lyrics=SUNG, **sent), spec, 1)
        assert settled.length_range == planning.LengthRange(low, high, 360.0)
    assert settle(_check(tags=TAGS, lyrics=SUNG), spec, 1).length_range is None


@pytest.mark.parametrize(
    ("params", "code", "param"),
    [
        ({"min_duration_s": 20}, "audio_param_out_of_range", "min_duration_s"),
        ({"max_duration_s": 400}, "audio_param_out_of_range", "max_duration_s"),
        ({"min_duration_s": 180, "max_duration_s": 180}, "audio_param_conflict", "min_duration_s"),
        ({"min_duration_s": 200, "max_duration_s": 120}, "audio_param_conflict", "min_duration_s"),
    ],
)
def test_a_range_outside_the_model_or_upside_down_is_refused_by_name(
    params: dict, code: str, param: str
) -> None:
    refused = _refused(tags=TAGS, lyrics=SUNG, **params)
    assert (refused.code, refused.details["param"]) == (code, param)


def test_stable_audio_refuses_a_range_and_says_to_send_duration_s() -> None:
    for name in ("min_duration_s", "max_duration_s"):
        with pytest.raises(ApiError) as caught:
            _check("stable-audio-3-medium", prompt="a sad piano piece", **{name: 60})
        assert caught.value.code == "audio_param_unsupported"
        assert "duration_s" in caught.value.message and "exactly" in caught.value.message


def test_a_named_planning_set_is_the_starting_structure_and_recorded_as_requested() -> None:
    spec = load_all_audio_manifests()[SONG].spec("cuda-linux")
    named = settle(_check(tags=INSTRUMENTAL_TAGS, instrumental=True, planning_set="harbor"), spec, 7)
    assert named.planning_lyrics == {"source": "pool", "id": "harbor", "requested": True,
                                     "lyrics": _set("harbor").lyrics}
    picked = settle(_check(tags=INSTRUMENTAL_TAGS, instrumental=True), spec, 7)
    assert (picked.planning_lyrics["id"], picked.planning_lyrics["requested"]) == (
        planning.load_pool("yue2")[7].id, False)
    own = settle(_check(tags=INSTRUMENTAL_TAGS, instrumental=True, planning_lyrics=SUNG), spec, 7)
    assert (own.planning_lyrics["source"], own.planning_lyrics["requested"]) == ("request", None)


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"planning_set": "harbor"}, "audio_param_conflict"),
        ({"planning_set": "harbor", "instrumental": True, "planning_lyrics": SUNG}, "audio_param_conflict"),
        ({"planning_set": "harbor", "instrumental": True, "lyrics": "[Verse]\n"}, "audio_param_conflict"),
        ({"planning_set": "nowhere", "instrumental": True}, "planning_set_unknown"),
    ],
)
def test_a_planning_set_outside_its_one_use_is_refused_by_name(params: dict, code: str) -> None:
    refused = _refused(tags=TAGS, **({"lyrics": SUNG} if not params.get("instrumental") else {}), **params)
    assert refused.code == code
    if code == "planning_set_unknown":
        ids = [entry.id for entry in planning.load_pool("yue2")]
        assert refused.details["planning_sets"] == ids
        assert all(set_id in refused.message for set_id in ids)


def test_stable_audio_has_no_planning_set() -> None:
    with pytest.raises(ApiError) as caught:
        _check("stable-audio-3-medium", prompt="a sad piano piece", planning_set="harbor")
    assert caught.value.code == "audio_param_unsupported"


def test_the_playground_lists_the_pool_s_ids_for_planning_set() -> None:
    from crucible import playground

    manifest = load_all_audio_manifests()[SONG]
    fields = {f["name"]: f for f in playground.audio_fields(manifest, manifest.spec("cuda-linux"))}
    assert fields["planning_set"]["options"] == ["", *(e.id for e in planning.load_pool("yue2"))]
    assert (fields["min_duration_s"]["min"], fields["max_duration_s"]["max"]) == (30.0, 360)


# --- the worker ----------------------------------------------------------------------------


def _abc(seconds_per_line: float, lines: int, bpm: int = 120) -> str:
    """A score of `lines * seconds_per_line` seconds: whole 4/4 bars at `bpm` (2 s a bar at
    120) and a last bar of the remainder."""
    quarters = Fraction(seconds_per_line * lines).limit_denominator(1000) * bpm / 60
    bars, rest = divmod(quarters * 8, 32)
    body = "z32|" * int(bars) + (f"z{int(rest)}|" if rest else "")
    return f"X:1\nT:\nM:4/4\nL:1/32\nQ:1/4={bpm}\nK:C\nV: Vocal\n{body}\n"


class LinePipe:
    """YuE2Pipeline.plan as the worker drives it: a score lasting a fixed number of seconds
    a planning line - what the PC's scores did, 5.7 to 12.0 s a line."""

    def __init__(self, seconds_per_line: float) -> None:
        self.seconds_per_line = seconds_per_line
        self.generation_config = SimpleNamespace(
            abc=SimpleNamespace(max_tokens=4096), semantic=SimpleNamespace(max_tokens=9000))
        self.planned: list[str] = []

    def plan(self, tags, lyrics, **kwargs):
        self.planned.append(lyrics)
        lines = sum(len(words) for _, words in planning.check(lyrics))
        return Plan(abc=_abc(self.seconds_per_line, lines), truncated=False,
                    timing={"output_tokens": 40 * lines, "seconds": 1.0, "prefill_seconds": 0.1,
                            "output_tps": 40.0, "prefix_tokens": 200, "cfg_branches": 1,
                            "execution": "eager", "attention": "sdpa"})


class Plan(SimpleNamespace):
    def save(self, directory) -> None:
        Path(directory).mkdir(parents=True)
        (Path(directory) / "score.abc").write_text(self.abc, encoding="utf-8")


class Progress:
    asked_to_stop = False

    def __init__(self) -> None:
        self.notes: list[str] = []

    def enter(self, stage, steps=None) -> None:
        pass

    def note(self, text: str) -> None:
        self.notes.append(text)


def _engine(monkeypatch: pytest.MonkeyPatch, pipe: LinePipe) -> tuple[Any, Any]:
    module = _load_worker(monkeypatch)
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = pipe
    engine.low_vram = False
    engine.decode_stages = {}
    engine.length = None
    return module, engine


def _job(tmp_path: Path, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**{
        "tags": INSTRUMENTAL_TAGS, "lyrics": None, "planning_lyrics": None, "seed": 7, "cfg": 1.0,
        "instrumental": False, "min_duration_s": None, "max_duration_s": None, "longest_s": 360,
        "planning_resizable": False, "output_path": str(tmp_path / "audio.flac"), **fields,
    })


def test_with_no_range_the_song_plans_once_and_records_its_score_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, engine = _engine(monkeypatch, LinePipe(10.0))
    job = _job(tmp_path, lyrics=SUNG)
    engine._score(job, Progress())
    assert engine._pipe.planned == [SUNG]
    assert engine.length["score_seconds"] == 30.0
    assert (engine.length["min_duration_s"], engine.length["max_duration_s"]) == (None, None)
    assert engine.length["attempts"][0]["in_range"] is None
    assert engine.length["resized_planning_lyrics"] is None


def test_a_sung_song_outside_its_range_stops_before_composing_and_its_words_are_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, engine = _engine(monkeypatch, LinePipe(10.0))
    job = _job(tmp_path, lyrics=SUNG, min_duration_s=60.0, max_duration_s=90.0)
    with pytest.raises(Exception) as caught:
        engine._score(job, Progress())
    refused = caught.value
    assert refused.code == "song_length_out_of_range"
    assert engine._pipe.planned == [SUNG], "planned once, from the client's words as sent"
    details = refused.details
    assert (details["score_seconds"], details["min_duration_s"], details["max_duration_s"]) == (30.0, 60.0, 90.0)
    assert (details["direction"], details["ratio_needed"], details["ratio_to_middle"]) == ("longer", 2.0, 2.5)
    assert "longer" in refused.message and "2x" in refused.message
    assert (tmp_path / module.FAILED_PLAN_DIR / "score.abc").is_file()


def test_an_instrumental_from_the_client_s_planning_lyrics_is_refused_like_a_sung_song(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, engine = _engine(monkeypatch, LinePipe(10.0))
    job = _job(tmp_path, instrumental=True, planning_lyrics=SUNG, max_duration_s=31.0,
               min_duration_s=None)
    engine._score(job, Progress())  # 30 s: in range
    job = _job(tmp_path, instrumental=True, planning_lyrics=SUNG, max_duration_s=29.0)
    with pytest.raises(Exception) as caught:
        engine._score(job, Progress())
    assert caught.value.code == "song_length_out_of_range"
    assert caught.value.details["direction"] == "shorter"


def test_a_pool_instrumental_is_re_planned_until_its_score_lands_and_the_same_seed_plans_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model gives this request 12 s a line where the pool's prior says 8.6: the first
    size aimed at overshoots, the second is aimed with the measured rate and lands."""
    lantern = _set("lantern").lyrics
    runs = []
    for _ in range(2):
        _, engine = _engine(monkeypatch, LinePipe(12.0))
        job = _job(tmp_path, instrumental=True, planning_lyrics=lantern, planning_resizable=True,
                   min_duration_s=120.0, max_duration_s=180.0)
        progress = Progress()
        engine._score(job, progress)
        runs.append((engine.length, list(engine._pipe.planned), progress.notes))
    length, planned, notes = runs[0]
    assert runs[0] == runs[1], "deterministic: the same request plans the same attempts"
    attempts = length["attempts"]
    assert [a["in_range"] for a in attempts] == [False, True]
    assert attempts[0]["score_seconds"] == 16 * 12.0 and attempts[0]["lines"] == 16
    assert 120.0 <= attempts[1]["score_seconds"] <= 180.0
    assert attempts[1]["structure"] == planning.Sizing.of(lantern).structure(attempts[1]["body_sections"])
    assert length["score_seconds"] == attempts[1]["score_seconds"]
    assert length["resized_planning_lyrics"] == planned[-1] != lantern
    assert any("planning it again" in note for note in notes)


def test_a_pool_instrumental_that_fits_is_planned_from_its_set_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lantern = _set("lantern").lyrics
    _, engine = _engine(monkeypatch, LinePipe(8.6))
    job = _job(tmp_path, instrumental=True, planning_lyrics=lantern, planning_resizable=True,
               min_duration_s=100.0, max_duration_s=200.0)
    engine._score(job, Progress())
    assert engine._pipe.planned == [lantern]
    assert engine.length["resized_planning_lyrics"] is None


def test_a_pool_instrumental_that_never_lands_is_refused_with_every_attempt_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rate that swings each score (what a model can do) never settles inside a narrow
    range: three scores, then a refusal naming them, every plan kept."""
    module, engine = _engine(monkeypatch, LinePipe(8.6))
    swings = iter([4.0, 20.0, 4.0, 20.0])
    plan = engine._pipe.plan

    def swinging(tags, lyrics, **kwargs):
        engine._pipe.seconds_per_line = next(swings)
        return plan(tags, lyrics, **kwargs)

    engine._pipe.plan = swinging
    job = _job(tmp_path, instrumental=True, planning_lyrics=_set("lantern").lyrics,
               planning_resizable=True, min_duration_s=150.0, max_duration_s=170.0)
    with pytest.raises(Exception) as caught:
        engine._score(job, Progress())
    refused = caught.value
    assert refused.code == "instrumental_length_not_reached"
    assert len(refused.details["attempts"]) == planning.MAX_LENGTH_ATTEMPTS == 3
    assert refused.details["max_attempts"] == 3
    for number in (1, 2, 3):
        assert (tmp_path / module.FAILED_PLAN_DIR / f"attempt-{number}" / "score.abc").is_file()
    assert all(f"{a['score_seconds']} s" in refused.message for a in refused.details["attempts"])


# --- the wire ------------------------------------------------------------------------------


def test_the_card_check_script_is_valid_python() -> None:
    script = HERE.parent / "scripts" / "check-song-length.py"
    compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_a_failure_carries_its_details_only_when_it_has_some() -> None:
    plain = JobFailure("worker_failed", "it broke")
    assert plain.to_dict() == {"code": "worker_failed", "message": "it broke"}
    detailed = JobFailure("song_length_out_of_range", "too long", {"score_seconds": 212.4})
    assert JobFailure.from_dict(detailed.to_dict()) == detailed


def test_a_song_s_done_record_carries_its_score_seconds_and_the_range(
    ready, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(ready, auth, model=SONG, params={
        "tags": TAGS, "lyrics": SUNG, "seed": 3, "min_duration_s": 120, "max_duration_s": 180})
    assert events[-1]["event"] == "done", events[-1]
    length = events[-1]["data"]["audio"]["length"]
    assert (length["min_duration_s"], length["max_duration_s"], length["score_seconds"]) == (120, 180, 150.0)
    assert "resized_planning_lyrics" not in length
    sent = [row for row in map(json.loads, transcript.read_text(encoding="utf-8").splitlines())
            if row.get("op") == "generate"][-1]
    assert (sent["min_duration_s"], sent["max_duration_s"], sent["longest_s"]) == (120, 180, 360)
    assert sent["planning_resizable"] is False


def test_a_song_refused_for_its_length_fails_by_name_with_details_and_the_worker_stays(
    ready, auth: dict[str, str], transcript: Path
) -> None:
    # Held by a queue session, as B-Sides holds an album, so the model stays between songs.
    session_id = open_queue_session(ready, auth, act="song")
    job_id, events = run_job(ready, auth, model=SONG, params={
        "tags": TAGS, "lyrics": SUNG, "seed": 3, "min_duration_s": 60, "max_duration_s": 90})
    assert events[-1]["event"] == "failed", events[-1]
    error = events[-1]["data"]["error"]
    assert error["code"] == "song_length_out_of_range"
    assert (error["details"]["score_seconds"], error["details"]["max_duration_s"]) == (150.0, 90)
    record = ready.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert record["error"]["details"]["score_seconds"] == 150.0
    assert record["request"]["refused"]["code"] == "song_length_out_of_range"
    # The next song runs on the same worker: a refusal is an answer, not a crash.
    _, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": SUNG, "seed": 4})
    assert events[-1]["event"] == "done", events[-1]
    loads = [row for row in map(json.loads, transcript.read_text(encoding="utf-8").splitlines())
             if row.get("op") == "load"]
    assert len(loads) == 1
    close_queue_session(ready, auth, session_id)


def test_a_pool_instrumental_is_sent_as_resizable_and_a_named_set_is_recorded(
    ready, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(ready, auth, model=SONG, params={
        "tags": INSTRUMENTAL_TAGS, "instrumental": True, "planning_set": "meadow", "seed": 3,
        "max_duration_s": 200})
    assert events[-1]["event"] == "done", events[-1]
    planned = events[-1]["data"]["audio"]["planning_lyrics"]
    assert (planned["source"], planned["id"], planned["requested"], planned["resized"]) == (
        "pool", "meadow", True, False)
    sent = [row for row in map(json.loads, transcript.read_text(encoding="utf-8").splitlines())
            if row.get("op") == "generate"][-1]
    assert sent["planning_resizable"] is True and sent["planning_lyrics"] == _set("meadow").lyrics
