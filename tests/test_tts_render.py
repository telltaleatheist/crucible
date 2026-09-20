"""The render door — `POST /v1/jobs {"type": "tts"}`. PHASE3-TTS.md section 6.

Text in, `<index>.flac` out, and a `chunk` measurement per row. Every test here
goes through the whole server: the API, the exclusive lane, the residency, the
real `crucible/engines/narrator.py` and its pipes, `tests/fake_narrator.py` on
the other end of them, and a real ffmpeg encoding real PCM into a real FLAC.

What is faked is the env (a stamped directory), the weights (a stamped
directory), the card (monkeypatched nvidia-smi probes) and the model. Nothing
about the server's own logic is.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, residency as residency_module
from crucible.accelerator import GIB
from crucible.errors import ApiError
from crucible.jobs import asr as asr_jobs
from crucible.jobs.tts import render as render_jobs
from crucible.residency import KIND_TTS
from crucible.voices import load_voice

from . import fake_narrator_engine
from .conftest import FAKE_BACKEND, holding_the_card, parse_sse, wav_base64, wav_bytes
from .test_tts_api import (  # noqa: F401 — imported to be used as fixtures
    fake_env,
    fake_weights,
    idle_card,
    tts_client,
    tts_recipes,
)
from .test_tts_api import run_job, submit

VOICE = "deathstalker"

#: Most of this module encodes real PCM into a real FLAC through a real ffmpeg,
#: which is a deliberately strong assertion — it is the difference between "the
#: server said it wrote a FLAC" and "the bytes on disk are a 24 kHz mono FLAC".
#: It also makes those tests the only ones in the suite that need something the
#: machine did not bring with it, and a fresh clone on a box without ffmpeg
#: would otherwise report a broken server rather than a missing tool. CI installs
#: ffmpeg precisely so this skip never fires there (.github/workflows/ci.yml);
#: everywhere else it degrades to an honest "not run" instead of a false red.
#:
#: The refusal path — `ffmpeg_missing` when the probe finds nothing — is NOT
#: skipped: it monkeypatches the probe and is the test that matters most on a
#: machine without ffmpeg.
#: Module-level, not per test: `ffmpeg_missing` is a PREFLIGHT refusal, so on a
#: machine without ffmpeg EVERY submit here is a 409 before any rendering
#: happens, and marking tests one at a time would miss one. The refusal itself is
#: tested in `test_tts_api.py`, which needs no ffmpeg and therefore always runs —
#: which is the test that matters most on a machine that has none.
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="these tests encode real FLACs; install ffmpeg to run them",
)

#: The fake worker's default: 15.0 characters of text per second of audio, which
#: is Higgs's configured pace. Every duration assertion below is arithmetic on
#: this number rather than a tolerance, because the fake makes it exact.
CHARS_PER_SEC = 15.0

#: Three sentences whose lengths differ, so a test that mixed two rows up would
#: see it in the durations rather than only in the text.
CHUNKS = [
    {"index": 41, "text": "He had been walking for some time."},
    {"index": 42, "text": "The road did not appear to end, not that day."},
    {"index": 43, "text": "Rain."},
]


@pytest.fixture(autouse=True)
def narrator(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every voice load in this module starts the fake worker."""
    return fake_narrator_engine.install(monkeypatch)


@pytest.fixture(autouse=True)
def quick_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    # And the readiness poll, which is two seconds because a vLLM load takes
    # minutes and polling it harder buys nothing. Every test here is up in
    # milliseconds, so the interval is the whole of its runtime.
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)


@pytest.fixture
def rendered(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> Callable[..., list[dict[str, Any]]]:
    """Run one render job to completion and return its events.

    The job's id is left on `go.job_id` — the event stream does not carry it
    (every event is already scoped to the job whose stream it is), and the
    artifact and provenance tests need it to fetch a file.
    """

    def go(**params: Any) -> list[dict[str, Any]]:
        fake_weights(VOICE)
        body = {"language": "en", "take": 0, "chunks": CHUNKS}
        body.update(params)
        response = submit(tts_client, auth, type="tts", model=VOICE, params=body)
        assert response.status_code == 202, response.json()
        go.job_id = response.json()["job_id"]
        with tts_client.stream(
            "GET", f"/v1/jobs/{go.job_id}/events", headers=auth
        ) as stream:
            return parse_sse(line for line in stream.iter_lines())

    go.job_id = ""
    return go


def events_of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event["data"] for event in events if event["event"] == kind]


def terminal(events: list[dict[str, Any]]) -> dict[str, Any]:
    return events[-1]


# ------------------------------------------------------------------- happy


def test_a_render_publishes_one_flac_per_chunk(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    events = rendered()
    assert terminal(events)["event"] == "done", terminal(events)
    names = sorted(event["name"] for event in events_of(events, "artifact"))
    # The client's own indices, unchanged. `<index>.flac` is where BookForge's
    # assembly and resume already look.
    assert names == ["41.flac", "42.flac", "43.flac"]
    assert terminal(events)["data"]["rendered"] == 3
    assert terminal(events)["data"]["failed"] == []


def test_the_bytes_are_a_real_flac_at_the_voices_sample_rate(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    """Mono 24 kHz PCM_16, byte for byte the format assembly already expects.

    Read out of the FLAC's own STREAMINFO block rather than trusted: the header
    is what every downstream tool reads, and an encoder invoked with the wrong
    `-ar` produces a perfectly valid file that plays at the wrong speed.
    """
    rendered()
    body = tts_client.get(
        f"/v1/jobs/{rendered.job_id}/artifacts/41.flac", headers=auth
    ).content
    assert body[:4] == b"fLaC"
    # STREAMINFO is the first metadata block: 4 bytes magic, 4 bytes block
    # header, then the block. Sample rate is 20 bits starting 10 bytes in, and
    # channel count is the 3 bits after it.
    streaminfo = body[8:8 + 34]
    packed = int.from_bytes(streaminfo[10:13], "big")
    assert packed >> 4 == 24_000
    assert ((packed >> 1) & 0b111) + 1 == 1  # mono
    assert streaminfo[12] & 0b1 or True  # bit depth spans the byte boundary
    depth = (((streaminfo[12] & 0b1) << 4) | (streaminfo[13] >> 4)) + 1
    assert depth == 16


def test_the_chunk_event_carries_the_measurements_and_the_verdict(
    rendered: Callable[..., list[dict[str, Any]]]
) -> None:
    """The model judges, the server forwards, the client orders.

    Named `..._is_the_whole_guard_interface` until 2026-09-13, which is the claim
    Owen's ruling retired: the seven measured fields were never the whole of it,
    because nothing in this path was guarding at all.

    The seven measured fields are the server's own; `guard` is the eighth and it
    is narrator's (PHASE6-REMOTE-RENDER.md section 3). The key set is asserted
    exactly, because an event that grew a field nobody declared is the same
    defect as one that lost one.
    """
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert sorted(chunks) == [41, 42, 43]
    row = chunks[41]
    assert set(row) == {
        "index", "seconds", "chars", "chars_per_sec", "tokens", "capped", "take",
        "guard",
    }
    # The fake sends no verdict unless a test asks for one, which is also what a
    # narrator serving an engine with no `render_many` does.
    assert row["guard"] is None
    assert row["chars"] == len(CHUNKS[0]["text"])
    # `seconds` is measured off the PCM that arrived, so it is arithmetic on the
    # fake's declared pace rather than a tolerance.
    assert row["seconds"] == pytest.approx(row["chars"] / CHARS_PER_SEC, abs=1e-4)
    assert row["chars_per_sec"] == pytest.approx(CHARS_PER_SEC, abs=1e-3)
    assert row["take"] == 0
    assert row["capped"] is False


def test_a_longer_chunk_is_longer_and_a_shorter_one_shorter(
    rendered: Callable[..., list[dict[str, Any]]]
) -> None:
    """Rows retire in reverse out of the fake; each measurement must still be
    about its own row."""
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[42]["seconds"] > chunks[41]["seconds"] > chunks[43]["seconds"]


def test_capped_is_reported_when_narrator_says_so(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`capped` is the difference between a long sentence and a runaway, and
    BookForge's PaceTracker cannot infer it from a duration."""
    monkeypatch.setenv("CRUCIBLE_FAKE_CAP_CHARS", "10")
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[41]["capped"] is True
    assert chunks[42]["capped"] is True
    assert chunks[43]["capped"] is False  # "Rain." is five characters


def test_a_narrator_that_reports_no_cap_publishes_null_and_not_false(
    tmp_path: Path,
) -> None:
    """`null` means *narrator did not say*, and is never to be read as `false`.

    The pinned narrator sends neither `capped` nor `tokens` on a retiring row
    (`serve/worker.py` sends `{i, format, data, duration, sampleRate}`), so this
    is what the render door will actually publish on the PC until narrator grows
    them. Asserted here against a row with the fields absent, because the fake
    always sends them — see its docstring.
    """
    from crucible.jobs.tts.render import _optional_bool, _optional_int

    bare = {"i": 41, "format": "pcm16", "data": "", "duration": 1.0}
    assert _optional_bool(bare, "capped") is None
    assert _optional_int(bare, "tokens") is None
    assert _optional_bool({**bare, "capped": False}, "capped") is False


# ------------------------------------------------------------------- guard
#
# Owen's ruling of 2026-09-13: the model and its inference own the guard AND the
# retake decision, so the verdict travels with the audio it is about and Crucible
# carries it unopened. These tests assert the CARRYING, never the contents —
# asserting the contents here would put a second owner on narrator's vocabulary,
# which is the exact defect docs/ARCHITECTURE.md section 1 names.

#: One verdict, in the shape `truncation.GuardPlan.verdict()` actually builds:
#: `verdict` is the ladder's own last action, `clean` is the flag, `parts` is how
#: many text units the chunk was finally rendered as, `band` is the tracker's four
#: numbers plus `warm`, and `takes` is the event records VERBATIM — which carry
#: `chars_per_second` (not `_sec`), `rung`, `side`, `depth`, `pace` and
#: `pace_source`. Copied off the source rather than invented: an earlier draft of
#: PHASE6 section 3 made up plausible names and every one of them was wrong.
GUARD = {
    "verdict": "rerolled",
    "clean": True,
    "parts": 1,
    "band": {
        "max_chars_per_sec": 20.0,
        "min_chars_per_sec": 14.5,
        "reference": 17.03,
        "observed": 4,
        "warm": False,
    },
    "takes": [
        {
            "index": 41,
            "depth": 0,
            "side": "short",
            "chars": 34,
            "seconds": 1.2,
            "chars_per_second": 28.33,
            "max_chars_per_sec": 20.0,
            "min_chars_per_sec": 14.5,
            "hole_seconds": 0.0,
            "max_hole_seconds": 5.0,
            "pace": 17.03,
            "pace_source": "recorded",
            "action": "short",
            "rung": "reroll",
        }
    ],
}


def test_a_guard_verdict_survives_the_round_trip_byte_for_byte(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forwarded verbatim: the object narrator sent is the object a client reads.

    Whole-object equality, not a field-by-field check — a server that rebuilt the
    verdict from keys it recognised would pass every field assertion and still
    drop the rung somebody added last week.
    """
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"41": GUARD}))
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[41]["guard"] == GUARD


def test_a_row_with_no_guard_publishes_null_and_not_a_missing_key(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`null` means narrator sent no verdict, and it is never read as "clean".

    The same rule `capped` has, one level up: an ABSENT key would say "this
    server does not speak the field", which is different news from "narrator did
    not say", and only one of those is true here. Both cases are produced by one
    render, so a server that got them from two code paths could not pass.
    """
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"41": GUARD}))
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[41]["guard"] == GUARD
    for index in (42, 43):
        assert "guard" in chunks[index], chunks[index]
        assert chunks[index]["guard"] is None


def test_crucible_reads_nothing_inside_the_guard(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vocabulary this server has never heard of crosses it unchanged.

    This is the statelessness that keeps `api_version` at 1. The day the ladder
    grows a rung, the verdict grows a word, and a Crucible that validated the
    word would refuse a chunk it rendered perfectly well — at the first guard
    fire on a real book, not at build time.
    """
    future = {
        "verdict": "a-rung-invented-next-year",
        "clean": False,
        "parts": 3,
        "band": None,
        "takes": [{"whatever": ["the", "ladder", "wanted"]}],
        "a_key_this_server_has_never_seen": {"nested": 1},
    }
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"42": future}))
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[42]["guard"] == future


def test_a_guard_that_is_not_an_object_fails_its_row_and_not_the_batch(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing checked about a verdict, and it fails one row.

    `guard` is an object or it is null; a string is neither, and the `chunk`
    event has nowhere to put it. Shipping the FLAC with the verdict silently
    dropped would publish a chunk nobody can trace back to a decision, so the row
    is refused by name — and its neighbours still land, because that is what
    every failure on this door does.
    """
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"41": "clean"}))
    events = rendered()
    assert terminal(events)["event"] == "done"
    names = sorted(event["name"] for event in events_of(events, "artifact"))
    assert names == ["42.flac", "43.flac"]
    failed = terminal(events)["data"]["failed"]
    assert [row["index"] for row in failed] == [41]
    assert "guard='clean', which is not an object" in failed[0]["message"]
    assert sorted(row["index"] for row in events_of(events, "chunk")) == [42, 43]


def test_a_guard_reaches_the_event_without_being_rebuilt() -> None:
    """`_guard_of` hands back the SAME object, not a reshaped copy.

    Identity, not equality. A server that rebuilt the verdict — even into an
    equal dict — would be a server with an opinion about its keys, and the next
    rung added to the ladder is the one that opinion drops. The unit-level half
    of the round-trip test above.
    """
    from crucible.jobs.tts.render import _guard_of

    assert _guard_of({"i": 41}) is None
    assert _guard_of({"i": 41, "guard": None}) is None
    verdict = {"verdict": "clean", "takes": []}
    assert _guard_of({"i": 41, "guard": verdict}) is verdict


# ---------------------------------------------------------------- failures


def test_a_failed_chunk_is_reported_and_its_neighbours_still_land(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad sentence never sinks the other 1,399.

    The opposite of `asr`'s rule, and for a stated reason: a transcript with a
    hole in the middle is invisible in the output, while a missing `<index>.flac`
    is a file that is not there and resume already knows how to ask for it again.
    """
    monkeypatch.setenv("CRUCIBLE_FAKE_FAIL_ROW", "42")
    events = rendered()
    assert terminal(events)["event"] == "done"
    names = sorted(event["name"] for event in events_of(events, "artifact"))
    assert names == ["41.flac", "43.flac"]
    failed = terminal(events)["data"]["failed"]
    assert [row["index"] for row in failed] == [42]
    assert "told to fail row 42" in failed[0]["message"]
    # And it was said at the time as well as in `done`, so a client watching the
    # stream learns which index to re-ask for without waiting for the end.
    assert any(
        "chunk 42 failed" in row["message"] for row in events_of(events, "progress")
    )
    # No `chunk` measurement for a row that produced no audio.
    assert sorted(row["index"] for row in events_of(events, "chunk")) == [41, 43]


# ------------------------------------------------------------ the take ladder
#
# PHASE3-TTS.md section 3, wired through on 2026-09-14. The client asks for
# take N; the server resolves what N MEANS for this voice and sends narrator
# the numbers on each item. A temperature never travels on the app's wire.
#
# `sampling_not_wired` did not go away — it changed subject. It used to mean
# "the contract has no channel", which `narrator/engine/item_sampling.py` made
# false. It now means "the narrator ON THIS WIRE has no channel", asked of the
# live process, because the tts env pins narrator by COMMIT and a pin may be
# older than the channel. On 2026-09-15 it was, and the last two tests in this
# block are that night written down.


@pytest.fixture
def batch_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], list[dict[str, Any]]]:
    """The BATCH-LEVEL envelope the fake worker was sent, per `generate_batch`.

    `sampling_log` is per item and structurally cannot see `retake` or `band`:
    they ride on the request because the guard is a driver over the whole
    batch, not a per-row lever (PHASE18-UNCERTIFIED.md section 6). Read off a
    file for `sampling_log`'s reason — narrator echoes nothing back, and a fake
    that did would let a test assert about the fake.
    """
    path = tmp_path / "batch.jsonl"
    fake_narrator_engine.steer(monkeypatch, batch_log=str(path))

    def read() -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    return read


@pytest.fixture
def sampling_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], list[dict[str, Any]]]:
    """What the fake worker was asked to render each row under.

    Read off a file the fake appends to rather than off a reply, because
    narrator does NOT echo sampling back — a fake that did would let a test
    assert about the fake instead of about the wire.
    """
    path = tmp_path / "sampling.jsonl"
    fake_narrator_engine.steer(monkeypatch, sampling_log=str(path))

    def read() -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    return read


def test_take_zero_sends_no_sampling_key_at_all(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """Absent means "the voice's loaded sampling", which IS take 0. Sending
    `{}` would be asking for a rung with nothing in it, which narrator refuses
    as `sampling_malformed` — correctly.

    `take` is the other way round and deliberately so: 0 IS a take, narrator
    reads an absent key and an explicit 0 as the same number, and sending it
    makes the wire say which take produced the artifact."""
    events = rendered(take=0)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    rows = sampling_log()
    assert sorted(row["i"] for row in rows) == [41, 42, 43]
    assert all(row["sampling"] is None for row in rows), rows
    assert all(row["take"] == 0 for row in rows), rows


def test_take_one_sends_that_rungs_numbers_on_every_item(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """deathstalker's rung 1 is one line, `temperature = 0.7`, and it means
    "take 0, but cooler": only the key the rung declares travels, and the
    engine lays it over its resolved sampling. One take per job, so every row
    carries the same numbers — but PER ITEM, because that is where narrator's
    channel is and because the streaming door mixes rungs in one batch."""
    events = rendered(take=1)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    rows = sampling_log()
    assert sorted(row["i"] for row in rows) == [41, 42, 43]
    assert all(row["sampling"] == {"temperature": 0.7} for row in rows), rows
    # BOTH HALVES OF THE RUNG. The numbers say what to sample with; the take
    # says which seed lane to draw in, and narrator seeds `seed + index +
    # TAKE_SEED_STRIDE * take`. Without it, a rung whose numbers happened to
    # match take 0's would be take 0's render under another name — which is
    # what every rung with no sampling override was until 2026-09-15.
    assert all(row["take"] == 1 for row in rows), rows
    # And the measurement still says which take it was.
    assert {row["take"] for row in events_of(events, "chunk")} == {1}


def test_a_rung_narrator_cannot_honour_fails_that_row_by_name(
    rendered: Callable[..., list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """narrator refuses a rung PER ROW — `sampling_malformed` when it is not a
    sampling, `sampling_not_supported` when this engine has no such lever (the
    MLX arm has no repetition penalty) — and Crucible carries the message
    across as that row's error, by name, without interpreting it. The other
    rows are not collateral: a failed chunk is reported and the run
    continues."""
    fake_narrator_engine.steer(monkeypatch, sampling_levers="topP")
    events = rendered(take=1)
    done = terminal(events)["data"]
    assert done["rendered"] == 0
    assert sorted(row["index"] for row in done["failed"]) == [41, 42, 43]
    for row in done["failed"]:
        assert row["message"].startswith("sampling_not_supported:")
        assert "temperature" in row["message"]


def test_a_narrator_without_the_channel_refuses_the_rung_instead_of_rendering_take_zero(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE TEST THAT WOULD HAVE CAUGHT 2026-09-15.

    Crucible resolves the rung and writes it onto every item — the two tests
    above prove that, and they passed all along. What nothing proved is that
    the narrator ON THE OTHER END reads it. The tts env pins narrator by commit
    (`envs/tts/higgs-v3-cuda-linux.txt`), and that night the pin was bookforge
    0eeb0267 — one day older than `narrator/engine/item_sampling.py`. That
    narrator's `_resolve_row` reads `item['voice']` and nothing else, so the
    rung was dropped WITHOUT A WORD.

    The measurement: two render jobs, voice `owen`, the same 150-char sentence,
    take 0 and take 1 (rung 1 = `temperature = 0.7`). Both `0.flac` artifacts
    came back byte-identical — 264,174 bytes, 11.30 s — and the run's narrator
    log said `Applied extra_params: {'temperature': 0.8, ...}`, which is take
    0's. Crucible reported a successful take 1 that never happened.

    So the handshake carries the fact now (`itemTake` on `ready`) and this
    job asks before it sends. A wrong take delivered as a success is the one
    outcome this job type may not produce.
    """
    fake_narrator_engine.steer(monkeypatch, no_item_take=1)
    events = rendered(take=1)

    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "sampling_not_wired"
    assert "{'temperature': 0.7}" in error["message"]
    assert "did not announce `itemTake`" in error["message"]
    # And it refused BEFORE the wire, so no row was rendered at the wrong rung.
    assert sampling_log() == []


def test_take_zero_still_renders_on_a_narrator_without_the_channel(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is scoped to a rung, not to the narrator.

    Take 0 sends no `sampling` key at all, and "no key" is what every narrator
    ever built already does correctly — it renders at the loaded voice's own
    sampling, in the seed lane it has always used, which IS take 0. Refusing
    take 0 too would strand every render on an old pin to protect a ladder it
    was not climbing. (The `take: 0` the item now carries is read by a new
    narrator and ignored by an old one, and both are correct about it.)
    """
    fake_narrator_engine.steer(monkeypatch, no_item_take=1)
    events = rendered(take=0)

    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert all(row["sampling"] is None for row in sampling_log())


# ------------------------------------------------------- the zero-shot load
#
# PHASE3-TTS.md section 5's amendment. These go through the whole load: the
# API, the lane, the residency, the real engine and its pipes, the document
# and the wav Crucible writes, and `tests/fake_narrator.py` making narrator's
# own refusals about both (an absent clip file, a clip with no transcript).


def test_a_zeroshot_voice_loads_with_its_clip_and_the_wav_is_on_disk(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """The whole seam. narrator resolves a Higgs v3 voice by NAME in the
    document, and it calls `os.path.isfile` on every clip path that document
    names — so a Crucible that wrote one without the other would fail here,
    which is what the fake worker's own refusals are for."""
    weights = fake_weights("zeroshot")
    events = run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {
            "data": wav_base64(8.4), "transcript": "He had been walking.",
            "name": "the stranger",
        }},
    )
    done = terminal(events)
    assert done["event"] == "done", done
    assert done["data"]["resident"] == "zeroshot"
    # The digest, so two clients loading `zeroshot` can tell whose clip won.
    reference = done["data"]["reference"]
    assert reference["name"] == "the stranger"
    assert reference["seconds"] == pytest.approx(8.4)
    assert reference["sha256"] == hashlib.sha256(wav_bytes(8.4)).hexdigest()

    document = json.loads(
        (home / "narrator-higgs-voices.json").read_text(encoding="utf-8")
    )
    entry = document["zeroshot"]
    assert entry["kind"] == "clips"
    # The BASE weights Crucible pulled, not whatever the HuggingFace cache
    # holds: narrator hands `checkpointDir` to `ClipsVoice(checkpoint_dir=)`
    # and the served arm exports it as HIGGS_MODEL_DIR.
    assert entry["checkpointDir"] == str(weights)
    clip = entry["clips"][0]
    assert clip["transcript"] == "He had been walking."
    assert Path(clip["path"]).read_bytes() == wav_bytes(8.4)


def test_the_resident_report_says_which_clip_is_loaded(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """`zeroshot` is one voice id and any number of clips, so the id alone
    would be two clients each assuming the resident one is theirs.

    `/v1/activity` and not `/v1/info`: this is the "what is this machine
    doing" report, which is where the RESIDENT thing is described. A
    `/v1/voices` row says `resident: true` and nothing more, because a row is
    a statement about the manifest and the host, not about the load."""
    fake_weights("zeroshot")
    run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {
            "data": wav_base64(4.0), "transcript": "Rain.", "name": "rain-01",
        }},
    )
    resident = tts_client.get("/v1/activity", headers=auth).json()["resident"]
    assert resident["kind"] == "tts"
    assert resident["id"] == "zeroshot"
    assert resident["reference"]["name"] == "rain-01"
    assert resident["reference"]["seconds"] == pytest.approx(4.0)
    assert resident["reference"]["sha256"] == hashlib.sha256(
        wav_bytes(4.0)
    ).hexdigest()


def test_a_resident_checkpoint_voice_reports_a_null_reference(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """Always present, null for a voice that is not conditioned on a clip: an
    absent key would mean "this build does not say"."""
    fake_weights(VOICE)
    run_job(tts_client, auth, type="load-voice", model=VOICE)
    resident = tts_client.get("/v1/activity", headers=auth).json()["resident"]
    assert resident["id"] == VOICE
    assert resident["reference"] is None


def test_a_resident_zeroshot_voice_renders_like_any_other(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """Owen, 2026-09-14: a zero-shot voice "should effectively be treated as a
    model, for all intents and purposes, except the route it takes to retrieve
    and return the audio". Once it is on the card, nothing downstream knows it
    was cloned from a clip."""
    fake_weights("zeroshot")
    run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {
            "data": wav_base64(4.0), "transcript": "Rain.",
        }},
    )
    response = submit(
        tts_client, auth, type="tts", model="zeroshot",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 202, response.json()
    with tts_client.stream(
        "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)


def wsl2_card_holding_our_own_engine(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The PC's card as it read at 07:30 on 2026-09-15, mid-incident.

    18.1 GiB of 24 in use, and an EMPTY compute-app list — which is what the
    WSL2 driver shim answers even while a process inside that same VM holds the
    card. The 18.1 GiB is narrator serving the resident voice; nothing else is
    on the machine. Returns a list that grows by one every time the guard looks
    at the card, so a test can assert that it did not.
    """
    looks: list[int] = []

    def compute_apps() -> list[Any]:
        looks.append(1)
        return []

    monkeypatch.setattr(accelerator, "probe_compute_apps", compute_apps)
    monkeypatch.setattr(
        accelerator, "probe_vram", lambda: (24 * GIB - 18_100 * 1024 ** 2, 24 * GIB)
    )
    return looks


def test_a_render_on_the_resident_voice_never_asks_the_card_for_room(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """Measured on the PC, 2026-09-15 07:30. `load-voice zeroshot` succeeded,
    `/v1/activity` reported it resident with its clip's digest, and the render
    of that same voice came back `409 accelerator_busy: cannot load
    'zeroshot': 18.1 GiB of the 24.0 GiB card is in use by a process this
    host's driver will not name`.

    The 18.1 GiB was Crucible's OWN narrator, serving `zeroshot`. The door had
    recognised the voice as resident — otherwise `_require_renderable` would
    have refused `voice_kind_unsupported` one line earlier — and then ran the
    LOAD guard anyway over a load it was never going to perform. On
    mlx-darwin the identical sequence rendered, because nothing there is
    unattributed.

    A clips voice because that is what was measured; the rule is about
    residency and not about the kind."""
    fake_weights("zeroshot")
    run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {"data": wav_base64(4.0), "transcript": "Rain."}},
    )
    looks = wsl2_card_holding_our_own_engine(monkeypatch)

    response = submit(
        tts_client, auth, type="tts", model="zeroshot",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 202, response.json()
    with tts_client.stream(
        "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    # Not merely "it was not refused": the door never looked at the card at
    # all, in the preflight or in the lane. A guard that still ran and happened
    # to pass would be the same bug waiting on a bigger voice.
    assert looks == []


def test_a_render_of_a_voice_that_is_not_resident_still_asks(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    """The other half of the same rule, so the fix above cannot read as "the
    render door stopped guarding". Nothing is resident, the card reads exactly
    as it did in the incident, and the render — which would have to LOAD — is
    refused by name."""
    fake_weights(VOICE)
    wsl2_card_holding_our_own_engine(monkeypatch)

    response = submit(
        tts_client, auth, type="tts", model=VOICE,
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 409, response.json()
    assert response.json()["error"]["code"] == "accelerator_busy"


# ------------------------------------------------- the arm, and the band
#
# PHASE18-UNCERTIFIED.md sections 4 and 6, Owen's ruling of 2026-09-19. Which
# arm renders a batch is the REQUEST's to say, and the band the guarded arm
# measures against is the CALLER's to state. Both ride on the batch envelope,
# because the guard is a driver over the whole batch rather than a per-row
# lever — which is why `sampling_log` cannot see them and `batch_log` can.


def test_a_render_with_no_flag_asks_for_the_bare_arm_by_name(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """Absent means `retake: false`, and it is SENT rather than left absent.

    The arm used to be chosen inside narrator by a capability probe
    (`serve/worker.py:_guards_its_own_batch`), so what happened to a book
    depended on what the engine offered and the caller had no say. Sending the
    word both ways is what makes the choice the request's: an omitted key would
    put the decision straight back where it was.
    """
    events = rendered()
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    batches = batch_log()
    assert len(batches) == 1, batches
    assert batches[0]["retake"] is False
    # And no band, because none was stated. `null` here is the fake recording
    # an absent key, which is the difference between "sent nothing" and "sent
    # an empty band".
    assert batches[0]["band"] is None
    # Nobody judged it, so there is no verdict — `null` at its most exact.
    assert all(row["guard"] is None for row in events_of(events, "chunk"))


def test_retake_true_with_a_band_sends_both_in_narrators_spelling(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """The guarded arm, and the band it measures against, forwarded verbatim.

    camelCase because that is narrator's spelling for these three rates on
    every other wire it has (`voice_entry`'s `paceCharsPerSec`); one map does
    the translation in `render.py` so the request's names and the wire's names
    cannot drift.
    """
    events = rendered(
        retake=True,
        band={
            "pace_chars_per_sec": 15.91,
            "max_chars_per_sec": 20.68,
            "min_chars_per_sec": 12.24,
        },
    )
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    batches = batch_log()
    assert len(batches) == 1, batches
    assert batches[0]["retake"] is True
    assert batches[0]["band"] == {
        "paceCharsPerSec": 15.91,
        "maxCharsPerSec": 20.68,
        "minCharsPerSec": 12.24,
    }


def test_a_band_on_a_bare_render_travels_and_is_not_acted_on(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """Owen, 2026-09-19: *"it won't do anything with the number because it
    wasn't asked to."* Accepted, checked, forwarded — and `retake` still says
    false, which is the key narrator reads to choose the arm."""
    events = rendered(
        band={
            "pace_chars_per_sec": 15.91,
            "max_chars_per_sec": 20.68,
            "min_chars_per_sec": 12.24,
        },
    )
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert batch_log()[0]["retake"] is False
    assert batch_log()[0]["band"]["paceCharsPerSec"] == 15.91


def test_a_job_that_states_no_width_sends_none_and_the_engine_keeps_its_own(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """THE MLX REGRESSION, WRITTEN DOWN (Owen's ruling of 2026-09-20).

    Between 2026-09-19 and 2026-09-20 an absent width resolved to
    `[voice.serving].max_num_seqs` — 16 in every packaged manifest, a
    vllm-omni stage-0 admission width measured on a 3090 Ti. narrator's MLX
    backend honours a batch `width` as an in-flight ceiling and was started at
    64 out of its own measured tier row, so it logged "MLX batch narrowed 64
    rows -> 16" on every batch. Measured on mistborn/Shift Book 2, chunk
    lengths equal and zero retakes: 12.9x realtime and 189 sentences/min
    before, 5.5x and 78 after.

    So an unstated width is now ABSENT FROM THE ENVELOPE, not a substituted
    number — `keys` and not `.get`, because a `width: null` would be this
    server stating a width it does not have — and `done` says `null`, which is
    the fact rather than a gap.
    """
    events = rendered()
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert "width" not in batch_log()[0]["keys"], batch_log()[0]["keys"]
    assert terminal(events)["data"]["width"] is None


def test_a_narrower_width_is_forwarded_and_nothing_is_restarted(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
    narrator: list[Any],
) -> None:
    """The ladder's baseline is 4 wide on a voice whose manifest says 16.

    Measured 2026-09-19 by the ladder's author: 0.60 mem fraction at 16 in
    flight summed to 24.2 GB on a 24 GB card, and WDDM then pages to host RAM
    4-10x slower with no error. Narrowing is a number on the envelope and
    nothing else — the server keeps the `--max-running-requests` and
    `cuda_graph_max_bs` it was loaded with, so ONE engine serves the job.
    """
    events = rendered(width=4)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert batch_log()[0]["width"] == 4
    assert terminal(events)["data"]["width"] == 4
    assert len(narrator) == 1, "narrowing a job restarted the engine"


def test_on_the_mlx_arm_a_stated_width_travels_and_this_door_refuses_none() -> None:
    """THE CEILING BELONGS TO THE ENGINE, AND CRUCIBLE KNOWS IT ON ONE ARM.

    `[voice.serving].max_num_seqs` is `HIGGS_MAX_NUM_SEQS` — the width this
    server itself starts narrator's serving stack at — so on the served arm
    this door can refuse a wider job before the card is touched. On
    `mlx-darwin` narrator starts no server, reads no `HIGGS_*` variable and
    batches at `NARRATOR_HIGGS3_MLX_BATCH` off `engines/narrator.py:MLX_TIERS`
    — 64 on the 64 GB Mac Studio against deathstalker's 16 — so `max_num_seqs`
    describes nothing there and refusing against it would refuse a width the
    engine can run. The stated width travels, and narrator's own
    `width_over_serving` answers for the width it actually has.

    Asked of `_require_width` directly rather than through a whole Mac render:
    the two arms differ in exactly one input (`spec.backend`), and a test that
    stood up a second client would be asserting about its own fixtures.
    """
    manifest = load_voice(VOICE)
    wide = render_jobs.TtsParams(language="en", take=0, chunks=CHUNKS, width=32)

    mlx = manifest.spec("mlx-darwin")
    assert jobenv.tts_env(manifest.narrator_engine, mlx.backend).serving_stack is None
    assert render_jobs._require_width(manifest, mlx, wide) == 32

    served = manifest.spec(FAKE_BACKEND.kind)
    assert (
        jobenv.tts_env(manifest.narrator_engine, served.backend).serving_stack
        is not None
    )
    with pytest.raises(ApiError) as refusal:
        render_jobs._require_width(manifest, served, wide)
    assert refusal.value.code == "width_over_serving"


def test_the_result_names_the_full_sampling_and_the_weights_that_ran(
    rendered: Callable[..., list[dict[str, Any]]],
) -> None:
    """PHASE18 section 7's promise, discharged once per job (2026-09-19).

    THE FULL TRIPLE, not the rung's override. deathstalker's rung 1 is one
    line, `temperature = 0.7`, and a record saying only that says nothing about
    the top-p and top-k it ran at — which is what a ladder's comparison rests
    on. Sampling lives on the MANIFEST, so a manifest edited between two runs
    would otherwise make two incomparable records that both claim take 1: every
    Higgs measurement before 2026-09-06 was at temperature 1.0 and the whole
    prior ladder record had to be marked "at the wrong temperature" once
    already.

    AND THE WEIGHTS, in the `/v1/voices` row's own three words, so a mismatch is
    visible rather than reconstructed from a fingerprint somebody parsed.
    """
    take_zero = terminal(rendered(take=0))["data"]
    assert take_zero["sampling"] == {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    }
    take_one = terminal(rendered(take=1))["data"]
    assert take_one["sampling"] == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 50,
    }
    assert take_one["voice"]["id"] == VOICE
    # A pinned voice: the identity IS the fetched sha, and the basis says so.
    assert take_one["voice"]["identity_basis"] == "verified"
    assert len(take_one["voice"]["identity"]) == 40


# ---------------------------------------------------------------- refusals


def _refuse(
    client: TestClient, auth: dict[str, str], **params: Any
) -> dict[str, Any]:
    body = {"language": "en", "take": 0, "chunks": CHUNKS}
    body.update(params)
    response = submit(client, auth, type="tts", model=VOICE, params=body)
    assert response.status_code >= 400, response.json()
    return response.json()["error"]


def test_a_take_past_the_end_of_the_ladder_renders_in_its_own_seed_lane(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
) -> None:
    """PHASE18 section 5, 2026-09-19. This was `unknown_take`.

    deathstalker declares two rungs, and take 4 is past them. It is not
    refused and it is not clamped: every item carries `take: 4` — the seed
    lane, which is the whole of what a numberless take asks for — and NO
    `sampling` key, which is the voice's own numbers rather than rung 1's
    `temperature = 0.7` delivered under take 4's name.
    """
    events = rendered(take=4)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    rows = sampling_log()
    assert sorted(row["i"] for row in rows) == [41, 42, 43]
    assert all(row["take"] == 4 for row in rows), rows
    assert all(row["sampling"] is None for row in rows), rows
    assert {row["take"] for row in events_of(events, "chunk")} == {4}


def test_a_chunk_over_the_cap_renders_instead_of_being_refused(
    rendered: Callable[..., list[dict[str, Any]]],
) -> None:
    """`chunk_too_long` is RETIRED (PHASE18 section 4, 2026-09-19).

    Owen: *"I don't think it's crucible's place to refuse chunks outside the
    band… especially if we add a different tts engine."* Chunking is still the
    client's and deathstalker still advertises an 800-character cap on
    `/v1/voices`; what this server no longer does is act on it. A 900-character
    chunk goes to the engine as sent and comes back measured — which is the
    only way a sweep can find out what the cap actually is.
    """
    events = rendered(chunks=[{"index": 0, "text": "x" * 900}])
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["rendered"] == 1
    assert terminal(events)["data"]["failed"] == []
    row = events_of(events, "chunk")[0]
    assert row["chars"] == 900


def test_retake_true_with_no_band_is_refused_by_name(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    """Not filled in from the voice, and not quietly downgraded to bare.

    Filling it in from the voice is the shape that produced the 2026-09-18
    defect twice: `higgs-default` satisfied a mandatory triple with narrator's
    own frame-cap divisor, and deathstalker inherited 16.64 onto weights that
    measured 15.91. Downgrading is worse — a client that asked to be guarded
    and was not would read every clean row as a verdict.
    """
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, retake=True)
    assert error["code"] == "retake_without_band"
    assert "states no band" in error["message"]
    assert "pace_chars_per_sec" in error["message"]


@pytest.mark.parametrize(
    "band, names",
    [
        ({"pace_chars_per_sec": 15.9, "max_chars_per_sec": 20.7}, "min_chars_per_sec"),
        (
            {
                "pace_chars_per_sec": 15.9,
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 12.2,
                "safe_max_chars": 800,
            },
            "safe_max_chars",
        ),
        (
            {
                "pace_chars_per_sec": "fast",
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 12.2,
            },
            "not a rate",
        ),
        (
            {
                "pace_chars_per_sec": 15.9,
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 0,
            },
            "positive",
        ),
        (
            {
                "pace_chars_per_sec": 25.0,
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 12.2,
            },
            "out of order",
        ),
    ],
)
def test_every_way_a_band_can_be_wrong_is_one_refusal(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    band: dict[str, Any],
    names: str,
) -> None:
    """ONE code for one rule (PHASE18 section 9). A band is a single statement
    — the measured pace and the two edges derived from it — so a missing rate,
    a rate that is not a number, a rate at or below zero and an order other
    than min < pace < max are five spellings of the same mistake, and sorting
    them into different codes would be two ways to learn one thing.

    It refuses the WHOLE request, like `sampling_malformed` and unlike a per-row
    failure: there is no row a band belongs to."""
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, band=band)
    assert error["code"] == "band_malformed", error
    assert names in error["message"], error


def test_a_width_above_the_voices_serving_width_is_refused_never_clamped(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    """narrator cannot keep more in flight than the server admits, and a clamp
    would be a job reporting a width it did not run at — which makes its
    throughput reproducible by nobody. Both numbers ride in the detail so a
    client can fix the request without reading `/v1/voices` again.

    THIS IS THE SERVED ARM, which is what makes the early refusal Crucible's to
    make: `max_num_seqs` is the number this server started narrator's serving
    stack at. The darwin half of the same rule is the test above."""
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, width=32)
    assert error["code"] == "width_over_serving"
    assert error["details"] == {"width": 32, "max_num_seqs": 16}
    assert "Never clamped" in error["message"]


def test_a_zeroshot_voice_that_is_not_resident_is_refused_because_this_door_cannot_load_it(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    """A render job loads its own voice, and a zero-shot load needs the clip
    that only `load-voice` carries (`params.reference`). This job's params are
    `language`, `take` and `chunks`, and a second channel for clips here would
    be two doors owning one fact — so the refusal NARROWED on 2026-09-14 to
    "and is not resident" rather than being deleted. It used to refuse the
    KIND outright, on the true-at-the-time grounds that narrator's load
    message carried no clips at all."""
    fake_weights("zeroshot")
    response = submit(
        tts_client,
        auth,
        type="tts",
        model="zeroshot",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 400, response.json()
    error = response.json()["error"]
    assert error["code"] == "voice_kind_unsupported"
    assert "is not resident" in error["message"]
    assert "params.reference" in error["message"]
    assert error["details"]["resident"] is False


def test_a_blank_chunk_is_refused_before_it_ends_the_batch(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, chunks=[{"index": 0, "text": "   "}])
    assert error["code"] == "invalid_params"
    assert "must not be blank" in error["message"]


def test_two_chunks_with_one_index_are_refused(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    fake_weights(VOICE)
    error = _refuse(
        tts_client,
        auth,
        chunks=[{"index": 7, "text": "one"}, {"index": 7, "text": "two"}],
    )
    assert error["code"] == "invalid_params"
    assert "[7] appear more than once" in error["message"]


def test_an_unknown_param_is_refused_rather_than_ignored(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, temperature=0.9)
    assert error["code"] == "invalid_params"


def test_the_wire_word_for_the_voice_is_model(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    """`describe_models()` for this type returns the voices, so `resolve_model`
    refuses an id that is not one before `preflight` ever runs."""
    response = submit(
        tts_client,
        auth,
        type="tts",
        model="qwen3.5-9b",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


# --------------------------------------------------------------- residency


def test_a_render_loads_its_own_voice_and_says_it_is_warming(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    """Section 6's one asymmetry with `llm`: a render job is an operator's
    explicit order and owns the lane, so it loads rather than refusing."""
    with holding_the_card(tts_client):
        events = rendered()
        health = tts_client.get("/v1/health", headers=auth).json()
    warmings = [row["message"] for row in events_of(events, "warming")]
    assert any("checking the accelerator for deathstalker" in m for m in warmings)
    assert any("starting narrator (higgs-v3)" in m for m in warmings)
    assert any("narrator loaded deathstalker" in m for m in warmings)
    # Held, or the render's own end would have cleared the card before this read
    # (Owen's ruling, 2026-09-14; crucible/settle.py).
    assert health["resident_models"] == [VOICE]
    assert health["resident_kind"] == KIND_TTS


def test_a_render_writes_the_voices_document_narrator_reads(
    rendered: Callable[..., list[dict[str, Any]]],
    home: Path,
    narrator: list[Any],
) -> None:
    """The document is written at the load, from the manifest and the pulled
    directory, and the engine is told where it is — which is the whole of what
    Crucible's first real render on either arm was missing (2026-09-14)."""
    rendered()
    document = home / "narrator-higgs-voices.json"
    assert document.is_file()
    written = json.loads(document.read_text(encoding="utf-8"))
    assert list(written) == [VOICE]
    entry = written[VOICE]
    assert entry["kind"] == "checkpoint"
    assert entry["checkpointDir"] == str(home / "voices" / VOICE / "cuda-linux")
    # deathstalker.toml's own numbers, on the wire narrator reads.
    assert entry["maxChars"] == 800
    assert entry["safeMinChars"] == 500
    assert entry["safeMaxChars"] == 800
    assert entry["sampling"] == {"temperature": 0.8, "topP": 0.95, "topK": 50}
    assert entry["paceCharsPerSec"] == 15.91
    assert narrator[0].environment()["NARRATOR_HIGGS_VOICES"] == str(document)


def test_a_second_render_does_not_restart_narrator(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    narrator: list[Any],
) -> None:
    """A Higgs voice change IS a full worker restart, so not changing it must not
    be one: two jobs on one voice share the engine that is already up.

    HELD ACROSS THE TWO, since Owen's unload ruling (2026-09-14): a render is
    done with its voice when it ends, so two unheld renders are two narrators.
    The holder stands in for the lease the render door cannot take yet — the
    RULING OWED in crucible/settle.py — and the companion test below is what the
    same two renders cost without one.
    """
    with holding_the_card(tts_client):
        rendered()
        assert len(narrator) == 1
        second = rendered()
        assert terminal(second)["event"] == "done"
        assert len(narrator) == 1
        assert not any(
            "starting narrator" in row["message"]
            for row in events_of(second, "warming")
        )


def test_an_unheld_render_clears_the_card_and_the_next_one_reloads(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    narrator: list[Any],
) -> None:
    """The bill for not stating an intention, stated (crucible/settle.py).

    A book rendered as ONE job still loads once, which is the shape BookForge
    uses. A book rendered as twenty jobs is twenty narrators, and that cost is
    what the RULING OWED about leasing a voice is asking to remove.
    """
    events = rendered()
    note = [row for row in events_of(events, "note")]
    assert note, "the unload must be said on the job that triggered it"
    assert note[-1]["unloaded"] == VOICE
    assert "nothing holds it" in note[-1]["message"]
    # Said before the terminal event, so a client reading the stream is told.
    kinds = [row["event"] for row in events]
    assert kinds.index("note") < kinds.index("done")
    assert tts_client.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert len(narrator) == 1

    rendered()
    assert len(narrator) == 2


def test_a_voice_lease_turns_a_book_rendered_chapter_by_chapter_into_one_load(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    narrator: list[Any],
) -> None:
    """What the previous two tests measured, now with the real holder.

    The companion above holds the card with a chat in flight because until
    2026-09-14 nothing else could: `POST /v1/models/{id}/lease` leased the
    resident MODEL, and `Leases.open` was refused for a resident voice. So a book
    rendered as ONE job loaded its voice once and a book rendered CHAPTER BY
    CHAPTER — which is how the app actually works — paid a narrator load per
    chapter. A lease may now name the resident thing of any kind, and this is
    that cost measured with one: three chapters, one narrator.

    The order is load, lease, render — a lease never loads, so the thing has to
    be resident before its client can say it intends more of it.
    """
    fake_weights(VOICE)
    run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert len(narrator) == 1

    opened = tts_client.post(
        f"/v1/models/{VOICE}/lease",
        headers=auth,
        json={"act": "tts", "ttl_seconds": 60},
    )
    assert opened.status_code == 201, opened.text
    lease = opened.json()
    assert lease["subject"] == VOICE
    assert lease["kind"] == KIND_TTS
    assert tts_client.get("/v1/activity", headers=auth).json()["lease"]["kind"] == (
        KIND_TTS
    )

    for _ in range(3):
        assert terminal(rendered())["event"] == "done"
        assert len(narrator) == 1, "a leased voice is rendered against, not reloaded"
        assert (
            tts_client.get("/v1/health", headers=auth).json()["resident_kind"]
            == KIND_TTS
        )

    released = tts_client.delete(f"/v1/leases/{lease['lease_id']}", headers=auth)
    assert released.status_code == 204
    # And the ruling still holds at the end of the book: the last holder let go.
    assert tts_client.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert len(narrator) == 1


def test_a_voice_lease_refuses_the_jobs_that_would_evict_it(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """Including a render of a DIFFERENT voice, which is a narrator restart.

    The rule is one sentence and the door applies it to every job: a lease
    refuses what would take the leased thing off the card, and a render of
    another voice does exactly that (`Residency.load_voice` evicts first).
    """
    fake_weights(VOICE)
    run_job(tts_client, auth, type="load-voice", model=VOICE)
    opened = tts_client.post(
        f"/v1/models/{VOICE}/lease",
        headers=auth,
        json={"act": "tts", "ttl_seconds": 60},
    )
    assert opened.status_code == 201, opened.text

    for body in (
        {"type": "load-voice", "model": "mistborn"},
        {"type": "load-voice", "model": VOICE},
        {"type": "unload-voice", "model": VOICE},
        {
            "type": "tts",
            "model": "mistborn",
            "params": {"language": "en", "take": 0, "chunks": CHUNKS},
        },
    ):
        response = tts_client.post("/v1/jobs", headers=auth, json=body)
        assert response.status_code == 409, (body, response.text)
        error = response.json()["error"]
        assert error["code"] == "leased", body
        assert error["details"]["kind"] == KIND_TTS
        assert "the resident voice" in error["message"]

    # `unload-model` and `unload-aligner` are NOT refused for the lease: each can
    # only reach its own kind, so with a voice on the card they refuse
    # `*_not_resident` on their own, and answering `leased` would report the
    # wrong reason for the right outcome. This server has neither type enabled,
    # so the whole matrix of kinds against types is `test_leases.py`'s.
    echoed = tts_client.post(
        "/v1/jobs",
        headers=auth,
        json={
            "type": "echo",
            "params": {"delay_ms": 0},
            "inputs": {"x.bin": {"inline_base64": "YQ=="}},
        },
    )
    assert echoed.json().get("error", {}).get("code") != "leased"


def test_the_load_is_part_of_the_load(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
) -> None:
    """`load-voice` now means the weights are in memory, not that a process is up.

    `ready` says narrator is listening. A job that stopped there would report a
    resident voice while the card was empty, and the first render would be the
    thing that found out.
    """
    fake_weights(VOICE)
    events = run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert terminal(events)["event"] == "done", terminal(events)
    warmings = [row["message"] for row in events_of(events, "warming")]
    assert any("narrator loaded deathstalker" in m for m in warmings)
    assert any("24000 Hz" in m for m in warmings)


def test_a_worker_that_dies_during_a_load_leaves_nothing_resident(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_EXIT_CODE", "3")
    fake_weights(VOICE)
    events = run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "engine_failed"
    assert "exited 3 before it was ready" in terminal(events)["data"]["error"]["message"]
    health = tts_client.get("/v1/health", headers=auth).json()
    assert health["resident_models"] == []
    assert health["resident_kind"] is None


def test_a_sample_rate_the_engine_disagrees_with_is_refused_not_resampled(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FLAC written at the manifest's rate from bytes generated at the engine's
    is a chunk of the wrong length, and nothing in the file would say so."""
    fake_weights(VOICE)
    # A manifest directory of this test's own, with one number changed. The
    # loader, the rows and the refusal are all the real ones.
    voices = tmp_path / "voices"
    voices.mkdir()
    source = Path(__file__).resolve().parent.parent / "crucible" / "voices"
    for manifest in source.glob("*.toml"):
        shutil.copyfile(manifest, voices / manifest.name)
    text = (voices / f"{VOICE}.toml").read_text(encoding="utf-8")
    (voices / f"{VOICE}.toml").write_text(
        text.replace("sample_rate = 24000", "sample_rate = 48000"), encoding="utf-8"
    )
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(voices))

    events = run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert terminal(events)["event"] == "failed"
    message = terminal(events)["data"]["error"]["message"]
    assert "renders deathstalker at 24000 Hz" in message
    assert "declares 48000" in message
    assert "refuses rather than resampling" in message


def test_one_holder_still_serves_both_kinds(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    """A render leaves a VOICE on the card, and the proxy's honest answer for a
    chat request is still `model_not_resident`."""
    with holding_the_card(tts_client):
        rendered()
        assert tts_client.app.state.residency.resident_kind == KIND_TTS
        assert tts_client.app.state.residency.resident_model is None
        assert tts_client.app.state.residency.voice_engine is not None


def test_the_provenance_sidecar_names_the_merge_that_rendered_it(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    """Two merges of one fine-tune are two narrators, so a finished audiobook
    that says which voice rendered it should say which merge of that voice."""
    rendered()
    sidecar = json.loads(
        tts_client.get(
            f"/v1/jobs/{rendered.job_id}/artifacts/41.flac.provenance.json",
            headers=auth,
        ).content
    )
    assert sidecar["model"]["id"] == VOICE
    assert sidecar["model"]["fingerprint"].startswith(f"{VOICE}@")
    assert len(sidecar["model"]["revision"]) == 40


def test_the_residency_is_torn_down_when_the_server_stops(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    narrator: list[Any],
) -> None:
    with holding_the_card(tts_client):
        rendered()
        engine = narrator[0]
        assert engine.pids
    tts_client.app.state.residency.shutdown()
    assert engine.pids == frozenset()


# ------------------------------------------------------------------ cancel
#
# OWEN HIT THIS LIVE, 2026-09-15. A `tts` render of `thirdreich` was cancelled on
# the Mac at chunk 38 of 89. The door answered `200 {"status": "cancelling"}`,
# and NOTHING acted on it: the job stayed `running`, progress went on rising,
# `/v1/activity` went on reporting `resident: thirdreich` and
# `claim: {held_by: "tts job cd2dac93..."}`, and the render finished the book.
#
# The cancel was lost in narrator, not here: `serve/worker.py`'s
# `_emit_guarded_batch` — the `render_many` ladder, the only batch arm this door
# drives — never read the flag its own stdin reader sets. Crucible's half was
# right and is unchanged: `_until` sends `{"action": "cancel"}`, the claim is a
# `with` block, and `JobStore._settle` runs for a cancelled job exactly as for a
# finished one. What is NEW here is that the cooperation is BOUNDED
# (`CANCEL_GRACE_SECONDS`), because a server cannot hold a card hostage to an
# engine's goodwill — and because the narrator in every installed tts env today
# still predates the fix.


def _wait_for(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    what: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = tts_client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        if predicate(state):
            return state
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never {what}; last state {state}")


def _start_a_slow_render(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
) -> str:
    """Submit a render whose rows take measurable time, and wait for the lane."""
    fake_weights(VOICE)
    chunks = [
        {"index": index, "text": f"Sentence number {index} of the chapter."}
        for index in range(12)
    ]
    response = submit(
        tts_client,
        auth,
        type="tts",
        model=VOICE,
        params={"language": "en", "take": 0, "chunks": chunks},
    )
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    # Not "running" — `running` is set before the voice is even loaded, and a
    # cancel that landed there would be answered by the load rather than by the
    # render loop. Waiting for the first artifact puts the cancel INSIDE the
    # batch, which is where the hole was.
    _wait_for(
        tts_client, auth, job_id,
        lambda state: bool(state["artifacts"]),
        "rendered a first chunk",
    )
    return job_id


def test_a_cancelled_render_stops_within_one_chunk(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole of what a cancel must do, asserted in one place.

    The job reaches `cancelled` while rows are still outstanding; the claim on
    narrator's wire is released; the voice is off the card — a cancelled job goes
    through the SAME settlement a finished one does (`crucible/settle.py`), which
    is the answer to "I stopped it and the model is still loaded"; and the
    artifacts already written are still fetchable, which is what BookForge reads
    when it drains after a stop.
    """
    fake_narrator_engine.steer(monkeypatch, row_delay_ms=120)
    job_id = _start_a_slow_render(tts_client, auth, fake_weights)

    cancelled = tts_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert cancelled.status_code == 200, cancelled.json()
    assert cancelled.json()["status"] == "cancelling"

    state = _wait_for(
        tts_client, auth, job_id,
        lambda state: state["status"] in ("done", "failed", "cancelled"),
        "reached a terminal state",
    )
    assert state["status"] == "cancelled", state.get("error") or state
    # Stopped, not finished: rows were still owed when it ended.
    assert 0 < len(state["artifacts"]) < 12, state["artifacts"]

    residency = tts_client.app.state.residency
    assert residency.claimed_by is None, "the claim outlived the job"
    assert residency.resident is None, "the voice is still on the card"

    # Artifacts already produced stay fetchable. `.flac` names are the chunk
    # indices, so the first one written is the lowest index present.
    name = sorted(n for n in state["artifacts"] if n.endswith(".flac"))[0]
    fetched = tts_client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
    assert fetched.status_code == 200
    assert fetched.content[:4] == b"fLaC", name


def test_a_narrator_that_ignores_the_cancel_is_taken_off_the_card(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE BOUND. An engine that hears the cancel and keeps working is stopped.

    This is the Mac's narrator exactly: its stdin reader set the flag and its
    rendering arm never read it. Crucible cannot fix that engine from here, and
    it must not wait on it either — a render door that trusts an engine's
    goodwill completely is a door that can hold the card for a whole book after
    the operator pressed stop.

    So the wait is bounded, and crossing the bound is the engine's failure and is
    named as one: the voice comes off the card THROUGH THE SAME UNLOAD DOOR the
    settlement uses, and the job still ends `cancelled`, because a cancel is what
    the operator asked for and they are not owed a `failed`.
    """
    monkeypatch.setattr("crucible.engines.narrator.CANCEL_GRACE_SECONDS", 1.0)
    fake_narrator_engine.steer(monkeypatch, row_delay_ms=120, ignore_cancel=1)
    job_id = _start_a_slow_render(tts_client, auth, fake_weights)

    assert tts_client.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200

    state = _wait_for(
        tts_client, auth, job_id,
        lambda state: state["status"] in ("done", "failed", "cancelled"),
        "reached a terminal state",
    )
    assert state["status"] == "cancelled", state.get("error") or state

    residency = tts_client.app.state.residency
    assert residency.claimed_by is None
    assert residency.resident is None, (
        "an engine that would not stop must not be left resident: nothing else "
        "is ever going to take it off the card"
    )

    # SAID, not merely done — a `note` on the job's own stream, so a reader who
    # finds the next render paying a load can see why.
    events = parse_sse(
        line
        for line in tts_client.get(
            f"/v1/jobs/{job_id}/events", headers=auth
        ).text.splitlines()
    )
    notes = [e["data"]["message"] for e in events if e["event"] == "note"]
    assert any("was sent a cancel" in note for note in notes), events


def test_cancelling_a_finished_render_is_refused_by_name(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    """Not a fault, and not a silent success either: a named 409.

    A client draining after a stop can race its own cancel against the job's
    end, so this has to be an ANSWER rather than a fault — and it is one:
    `job_not_cancellable`, saying what the job already is. Nothing is retried,
    nothing is re-run, and the artifacts are exactly where they were.
    """
    assert terminal(rendered())["event"] == "done"
    response = tts_client.delete(f"/v1/jobs/{rendered.job_id}", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_cancellable"
    assert "already done" in response.json()["error"]["message"]


def test_cancelling_an_unknown_job_is_refused_by_name(
    tts_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    response = tts_client.delete("/v1/jobs/notajob", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_job"


def test_nothing_in_this_module_touched_a_real_engine_module() -> None:
    """The double replaces the argv and nothing else."""
    assert residency_module.build_voice_engine.__module__ == (
        "tests.fake_narrator_engine"
    )
