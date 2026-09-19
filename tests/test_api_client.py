"""`crucible api …` — the client half of the command line.

Almost everything here runs against a REAL server on a REAL socket
(tests/live_server.py), driven through `cli.main` with the argv a person would
type. That is deliberate and it is not slower than it needs to be: this module's
whole job is HTTP, SSE framing and exit codes, and a `TestClient` would let the
one thing it can get wrong — reading a streamed body off a socket a line at a
time — be faked out from under it. The `echo` job type is what carries the
end-to-end cases, because it is the one type that finishes without an
accelerator, and no test in this file touches the card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI

from crucible import cli
from crucible import apiclient
from crucible.apiclient import ClientRefusal

from .conftest import TOKEN
from .live_server import serve


@pytest.fixture
def base(make_app: Callable[..., FastAPI]) -> Iterator[str]:
    """A real Crucible on a real loopback port, with `echo` enabled."""
    with serve(make_app()) as url:
        yield url


def run(base: str, *argv: str) -> int:
    """`crucible api --url … --token … <argv>`, through the real entry point."""
    return cli.main(["api", "--url", base, "--token", TOKEN, *argv])


def lines(captured: str) -> list[Any]:
    """Every line of a JSONL stream, parsed."""
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


# ------------------------------------------------------------------ connection


def _namespace(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {"url": None, "token": None, "pairing": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_url_without_a_token_is_refused_and_never_borrows_the_local_one() -> None:
    """The credential-leak refusal. A typo'd address must not get this machine's bearer."""
    with pytest.raises(ClientRefusal) as refusal:
        apiclient.resolve(_namespace(url="http://192.168.68.20:7100"))
    assert "token_required" in str(refusal.value)


def test_a_token_without_a_url_is_refused_by_name() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        apiclient.resolve(_namespace(token="abc"))
    assert "url_required" in str(refusal.value)


def test_a_pairing_line_carries_the_address_the_name_and_the_token() -> None:
    """`crucible token --url` prints exactly this line; parsing it is pairing.py's."""
    resolved = apiclient.resolve(
        _namespace(pairing="crucible://crucible%40mac-studio@192.168.68.20:7100/#abc123")
    )
    assert resolved.url == "http://192.168.68.20:7100"
    assert resolved.name == "crucible@mac-studio"
    assert resolved.token == "abc123"
    assert resolved.source == "--pairing"


def test_a_pairing_line_cannot_be_combined_with_url_or_token() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        apiclient.resolve(
            _namespace(pairing="crucible://a@h:1/#t", url="http://elsewhere:7100")
        )
    assert "connection_overspecified" in str(refusal.value)


def test_a_line_that_is_not_a_pairing_line_is_refused_by_name() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        apiclient.resolve(_namespace(pairing="http://192.168.68.20:7100"))
    assert "pairing_line_invalid" in str(refusal.value)


# ----------------------------------------------------------------- plain reads


def test_a_read_prints_one_json_document(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "info") == 0
    document = json.loads(capsys.readouterr().out)
    assert document["server"]["name"] == "crucible@test"
    assert "echo" in document["job_types"]


def test_the_api_version_header_travels_on_every_request(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`require_api_version` refuses 426 without it, so a 200 here proves it was sent."""
    assert run(base, "health") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_the_bearer_token_is_what_the_server_checks(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The WRONG token, and the refusal that says so rather than the one for none.

    `require_auth` answers `unauthorized` to a missing header and to a wrong
    token alike, so asserting the code proves only that some header question was
    asked — a client that sent no Authorization at all passes that. The two
    MESSAGES differ ("missing Authorization header" against "bearer token is not
    this server's token"), and asserting the second is what proves the header
    was sent and carried this value. Found by mutation, 2026-09-16: deleting the
    Authorization line entirely left this test green.
    """
    assert cli.main(["api", "--url", base, "--token", "not-the-token", "info"]) == 1
    captured = capsys.readouterr()
    assert "HTTP 401" in captured.err
    refusal = json.loads(captured.err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "unauthorized"
    assert refusal["error"]["message"] == "bearer token is not this server's token"


# ------------------------------------------------------------------------ jobs


def test_submit_returns_the_job_id_and_does_not_wait(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.txt"
    source.write_text("the road did not appear to end", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo",
               "--input", f"page.txt={source}") == 0
    assert "job_id" in json.loads(capsys.readouterr().out)


def test_follow_streams_the_events_then_the_final_state(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole path: upload, submit, SSE, terminal status, artifacts on disk."""
    source = tmp_path / "in.txt"
    source.write_text("he had been walking for some time", encoding="utf-8")
    out = tmp_path / "artifacts"

    assert run(base, "job", "submit", "--type", "echo",
               "--params", '{"delay_ms": 0}',
               "--input", f"page.txt={source}",
               "--follow", "--artifacts-dir", str(out)) == 0

    printed = lines(capsys.readouterr().out)
    assert "job_id" in printed[0]
    events = [row for row in printed if "event" in row]
    assert [row["event"] for row in events][-1] == "done"
    # Strictly increasing ids are what `--since` resumes against.
    assert [row["id"] for row in events] == sorted(row["id"] for row in events)

    state = printed[-2]
    assert state["status"] == apiclient.SUCCEEDED
    assert state["artifacts"] == ["page.txt"]
    assert printed[-1]["artifacts_saved"][0]["name"] == "page.txt"
    assert (out / "page.txt").read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_a_failed_job_exits_nonzero_even_though_every_request_succeeded(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """echo with no inputs fails inside the job. A script must not read that as 0."""
    assert run(base, "job", "submit", "--type", "echo", "--follow") == 1
    printed = lines(capsys.readouterr().out)
    assert printed[-1]["status"] == "failed"
    assert printed[-1]["error"]["code"] == "no_inputs"


def test_a_server_refusal_is_printed_verbatim_with_its_code(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The code is the string a person greps this repo for; it must survive."""
    assert run(base, "job", "submit", "--type", "nosuchtype") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "unknown_job_type"


def test_artifacts_dir_without_follow_is_refused_rather_than_implying_follow(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "job", "submit", "--type", "echo",
               "--artifacts-dir", str(tmp_path)) == 1
    assert "artifacts_need_follow" in capsys.readouterr().err


def test_events_resume_after_a_last_event_id(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--since N` is the server's own Last-Event-ID — how a dropped follow comes back.

    Run against a job that has already finished, so what comes back is the
    REPLAY and nothing else: the tail of the log, starting after the id given,
    and no repeat of what was already delivered.
    """
    source = tmp_path / "in.txt"
    source.write_text("x", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo", "--params", '{"delay_ms": 0}',
               "--input", f"page.txt={source}", "--follow") == 0
    printed = lines(capsys.readouterr().out)
    job_id = printed[0]["job_id"]
    whole = [row for row in printed if "event" in row]
    assert len(whole) >= 3, "an echo job emits queued, progress and done at least"

    cut = whole[0]["id"]
    assert run(base, "job", "events", job_id, "--since", str(cut)) == 0
    resumed = [row for row in lines(capsys.readouterr().out) if "event" in row]
    assert [row["id"] for row in resumed] == [row["id"] for row in whole[1:]]


def test_params_must_be_an_object(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "job", "submit", "--type", "echo", "--params", "[1, 2]") == 1
    assert "params_not_an_object" in capsys.readouterr().err


def test_a_params_file_that_is_not_there_is_named(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "job", "submit", "--type", "echo",
               "--params", f"@{tmp_path / 'nope.json'}") == 1
    assert "--params_file_missing" in capsys.readouterr().err


def test_an_input_that_is_not_name_equals_path_is_named(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "job", "submit", "--type", "echo", "--input", "justapath") == 1
    assert "--input_malformed" in capsys.readouterr().err


def test_cancel_reports_what_the_server_answered_not_what_it_will_become(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """DELETE answers `cancelling`; a client that printed `cancelled` would be lying."""
    source = tmp_path / "in.txt"
    source.write_text("x", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo",
               "--params", '{"delay_ms": 20000}',
               "--input", f"page.txt={source}") == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    assert run(base, "job", "cancel", job_id) == 0
    assert json.loads(capsys.readouterr().out)["status"] in ("cancelling", "cancelled")


def test_an_artifact_can_be_written_to_a_named_file(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.bin"
    source.write_bytes(b"\x00\x01\x02payload")
    assert run(base, "job", "submit", "--type", "echo", "--params", '{"delay_ms": 0}',
               "--input", f"clip.bin={source}", "--follow") == 0
    job_id = lines(capsys.readouterr().out)[0]["job_id"]
    target = tmp_path / "out" / "clip.bin"
    assert run(base, "job", "artifact", job_id, "clip.bin", "--out", str(target)) == 0
    assert target.read_bytes() == source.read_bytes()
    assert json.loads(capsys.readouterr().out)["bytes"] == len(source.read_bytes())


def test_an_unknown_artifact_names_the_ones_the_job_has(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo", "--params", '{"delay_ms": 0}',
               "--input", f"page.txt={source}", "--follow") == 0
    job_id = lines(capsys.readouterr().out)[0]["job_id"]
    assert run(base, "job", "artifact", job_id, "missing.flac",
               "--out", str(tmp_path / "x")) == 1
    assert "unknown_artifact" in capsys.readouterr().err


# ---------------------------------------------------------------------- upload


def test_upload_returns_a_blob_the_next_job_can_name(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFFtest")
    assert run(base, "upload", str(source)) == 0
    blob = json.loads(capsys.readouterr().out)
    assert blob["bytes"] == 8

    assert run(base, "job", "submit", "--type", "echo", "--params", '{"delay_ms": 0}',
               "--input-blob", f"clip.wav={blob['blob_id']}", "--follow") == 0
    assert lines(capsys.readouterr().out)[-1]["status"] == apiclient.SUCCEEDED


def test_uploading_a_file_that_is_not_there_is_refused_before_any_request(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "upload", str(tmp_path / "nope.wav")) == 1
    assert "input_missing" in capsys.readouterr().err


# ------------------------------------------------------------------ the chat door


def test_chat_will_not_take_a_whole_body_and_a_shorthand_at_once(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "chat", "--body", "{}", "--model", "qwen3.5-9b") == 1
    assert "chat_overspecified" in capsys.readouterr().err


def test_chat_with_neither_form_is_refused(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "chat", "--model", "qwen3.5-9b") == 1
    assert "chat_underspecified" in capsys.readouterr().err


# ---------------------------------------------------------- the serial tts door
#
# The session's own machinery is `crucible/ttsstream.py` and is tested against a
# real narrator double in tests/test_tts_stream.py. What is tested HERE is the
# part this module owns: which frames stop the follow, and the WAV header it
# writes. The frames are canned rather than generated, so `--until` can be shown
# to stop on the right one without standing up an engine.


def _frames(*rows: dict[str, Any]) -> Callable[..., Iterator[dict[str, Any]]]:
    def fake_follow(connection: Any, path: str, *, last_event_id: int = 0):
        for index, row in enumerate(rows, start=1):
            yield {"id": index, **row}

    return fake_follow


def test_until_stops_at_that_rows_done_and_not_at_anothers(
    base: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(apiclient, "follow", _frames(
        {"event": "ready", "data": {"voice": "sigma", "sample_rate": 24000}},
        {"event": "done", "data": {"id": "r1", "seconds": 1.0}},
        {"event": "done", "data": {"id": "r2", "seconds": 1.0}},
    ))
    assert run(base, "stream", "events", "s1", "--until", "r1") == 0
    printed = lines(capsys.readouterr().out)
    assert [row["event"] for row in printed] == ["ready", "done"]
    assert printed[-1]["data"]["id"] == "r1"


def test_an_error_frame_for_the_awaited_row_exits_nonzero(
    base: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(apiclient, "follow", _frames(
        {"event": "ready", "data": {"voice": "sigma", "sample_rate": 24000}},
        {"event": "error", "data": {"id": "r1", "code": "engine_failed", "message": "no"}},
    ))
    assert run(base, "stream", "events", "s1", "--until", "r1") == 1
    assert lines(capsys.readouterr().out)[-1]["data"]["code"] == "engine_failed"


def test_the_wav_is_written_at_the_rate_the_session_reported(
    base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PHASE3-TTS section 6: the rate is the engine's, never a constant here."""
    import base64
    import wave

    pcm = b"\x00\x01" * 1200
    monkeypatch.setattr(apiclient, "follow", _frames(
        {"event": "ready", "data": {"voice": "sigma", "sample_rate": 16000}},
        {"event": "audio", "data": {"id": "r1", "seq": 0,
                                    "pcm_base64": base64.b64encode(pcm).decode()}},
        {"event": "audio", "data": {"id": "r1", "seq": 1,
                                    "pcm_base64": base64.b64encode(pcm).decode()}},
        {"event": "done", "data": {"id": "r1", "seconds": 0.15}},
    ))
    out = tmp_path / "audio"
    assert run(base, "stream", "events", "s1", "--until", "r1",
               "--audio-dir", str(out)) == 0
    with wave.open(str(out / "r1.wav"), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == len(pcm)  # two frames of 1200 samples each
    assert lines(capsys.readouterr().out)[-1]["sample_rate"] == 16000


def test_resuming_past_the_ready_frame_refuses_to_invent_a_sample_rate(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "stream", "events", "s1", "--since", "12",
               "--audio-dir", str(tmp_path)) == 1
    assert "audio_needs_the_ready_frame" in capsys.readouterr().err


def test_say_needs_text_and_will_not_take_two_sources(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "stream", "say", "s1", "--row", "r1", "--take", "0") == 1
    assert "say_needs_text" in capsys.readouterr().err
    body = tmp_path / "row.txt"
    body.write_text("hello", encoding="utf-8")
    assert run(base, "stream", "say", "s1", "--row", "r1", "--take", "0",
               "--text", "hello", "--text-file", str(body)) == 1
    assert "say_overspecified" in capsys.readouterr().err


# ------------------------------------------------------------------- transport


def test_an_address_nothing_answers_on_is_not_a_refusal(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead port is not the server saying no, and must not read like one."""
    assert cli.main(["api", "--url", "http://127.0.0.1:1", "--token", "t", "ping"]) == 1
    assert "server_unreachable" in capsys.readouterr().err


def test_a_closed_pipe_is_success_and_not_an_unreachable_server(
    base: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`crucible api info | head -4` must exit 0.

    BrokenPipeError is an OSError, so the order of the two `except` clauses
    decides this. Written the wrong way round it printed `server_unreachable:
    the local engine did not answer: [Errno 32] Broken pipe` about a server that
    had just answered — measured on the first live run, 2026-09-16.
    """
    def closed(value: Any) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(apiclient, "emit", closed)
    assert run(base, "info") == 0


# --------------------------------------------- the doors that need what is off
#
# `tts`, `llm` and the task lane are disabled on the fixture's server and
# standing any of them up means an engine. What these prove is the half this
# module owns: that the verb assembles a request the route ACCEPTS as
# well-formed, so the refusal that comes back is about the server's state and
# not about the body. A malformed body would be a 422 from pydantic instead, and
# that is what these would catch.


def test_stream_open_reaches_the_route_and_is_refused_for_the_servers_own_reason(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "stream", "open", "--voice", "sigma", "--language", "en") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "job_type_disabled"


def test_lease_open_reaches_the_route_with_both_required_fields(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing `act` or `ttl_seconds` would be a 422; nothing resident is a 409."""
    assert run(base, "lease", "open", "qwen3.5-9b", "--act", "clean", "--ttl", "600") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "not_resident"


def test_task_submit_reaches_the_route_with_the_fields_its_type_needs(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `pull` with `--kind` and `--id` is refused for the SUBJECT, not the body.

    It does NOT prove that the unset flags were left out rather than sent as
    null: measured by mutation on 2026-09-16, `TaskCreate`'s validator tests
    `getattr(...) is not None`, so an explicit `"job_type": null` and an absent
    `job_type` are the same request to it and nothing here can tell them apart.
    `cmd_task_submit` still omits them — stating a field nobody asked for is
    wrong even where it is invisible — and that is a choice this test is not the
    witness for.
    """
    assert run(base, "task", "submit", "--type", "pull",
               "--kind", "model", "--id", "no-such-model") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "unknown_subject"


def test_a_post_with_no_body_at_all_reaches_its_route(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`upstream-test` and `lease heartbeat` both POST nothing.

    `crucible/local.py` sends a literal `b"{}"` on every POST rather than an
    empty body, and it does not say why — so whether urllib's bodyless POST is
    accepted here is a question to ask rather than assume. It is: both routes
    answer their own 404, which they can only do after the request parsed.
    """
    assert run(base, "upstream-test", "no-such-upstream") == 1
    first = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert first["error"]["code"] == "unknown_upstream"

    assert run(base, "lease", "heartbeat", "no-such-lease") == 1
    second = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert second["error"]["code"] == "unknown_lease"


@pytest.fixture
def tts_base(make_app: Callable[..., FastAPI]) -> Iterator[str]:
    """The same real server with `tts` on, for the two voice-manifest verbs.

    `voice_write` and `voice_remove` check `enable_tts` before anything else, so
    against the plain `base` they would answer the same 503 whatever argv sent
    them and prove nothing about the request that was built.
    """
    with serve(make_app(enable_tts=True)) as url:
        yield url


def test_voice_write_carries_the_whole_manifest_document(
    tts_base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is the server reading the BODY — `voice_invalid` names the table.

    A document with no `voice` key is rejected by `write_home_voice`, which runs
    after `await request.json()`, so a 400 saying so is only reachable if the
    file's contents travelled and parsed as a dict.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"not_a_voice": {}}), encoding="utf-8")
    assert run(tts_base, "voice-write", "made-up",
               "--manifest", f"@{manifest}") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "voice_invalid"


def test_voice_remove_reaches_the_route_and_is_told_there_is_no_overlay(
    tts_base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`voice_not_custom` is the route's own 404, and only it says that word."""
    assert run(tts_base, "voice-remove", "made-up") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "voice_not_custom"


def test_a_params_file_is_read_and_sent(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`@file` is how a ladder's thousand chunks get past a shell's argument limit."""
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"delay_ms": 999999}), encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo",
               "--params", f"@{params}", "--follow") == 1
    # 999999 is past `EchoParams.delay_ms`'s `le=60_000`. Echo validates its
    # params inside `run()` rather than in `preflight`, so this is admitted and
    # then FAILS — and the failure names the field, which it can only do if the
    # file's contents actually reached the server.
    assert "delay_ms" in json.dumps(lines(capsys.readouterr().out)[-1])


# -------------------------------------------------------------- route coverage


#: Every route the server serves, and the argv that reaches it — or the reason
#: this CLI deliberately does not. Owen asked for a command per endpoint, and a
#: claim like that is worth nothing unless something enumerates the server and
#: checks. The map is HERE and not in the module because it is the assertion,
#: not the implementation: a route added to `crucible/api.py` with no verb makes
#: this test fail with the path in the message.
COVERED: dict[str, str] = {
    "GET /v1/ping": "api ping",
    "GET /v1/health": "api health",
    "GET /v1/info": "api info",
    "GET /v1/setup": "api setup",
    "GET /v1/capability": "api capability",
    "GET /v1/accelerator": "api accelerator",
    "GET /v1/activity": "api activity",
    "GET /v1/models": "api models",
    "GET /v1/voices": "api voices",
    "PUT /v1/voices/{voice_id}": "api voice-write",
    "DELETE /v1/voices/{voice_id}": "api voice-remove",
    "GET /v1/catalog": "api catalog",
    "DELETE /v1/catalog/{kind}/{subject_id}": "api catalog-remove",
    "GET /v1/settings": "api settings",
    "PUT /v1/settings": "api settings --patch",
    "POST /v1/settings/upstreams/{name}/test": "api upstream-test",
    "GET /v1/pairing/requests": "api pairing-requests",
    "POST /v1/pairing/decision": "api pairing-decide",
    "GET /v1/openai/models": "api openai-models",
    "GET /openai/v1/models": "api openai-models (the same handler, OpenAI's path)",
    "POST /v1/openai/chat/completions": "api chat",
    "POST /openai/v1/chat/completions": "api chat (the same handler, OpenAI's path)",
    "POST /v1/uploads": "api upload",
    "POST /v1/jobs": "api job submit",
    "GET /v1/jobs/{job_id}": "api job get",
    "DELETE /v1/jobs/{job_id}": "api job cancel",
    "GET /v1/jobs/{job_id}/events": "api job events",
    "GET /v1/jobs/{job_id}/artifacts/{name}": "api job artifact",
    "POST /v1/tasks": "api task submit",
    "GET /v1/tasks": "api task list",
    "GET /v1/tasks/{task_id}": "api task get",
    "DELETE /v1/tasks/{task_id}": "api task cancel",
    "GET /v1/tasks/{task_id}/events": "api task events",
    "POST /v1/tts/stream": "api stream open",
    "POST /v1/tts/stream/{session_id}": "api stream say / cancel / cancel-all",
    "DELETE /v1/tts/stream/{session_id}": "api stream close",
    "GET /v1/tts/stream/{session_id}/events": "api stream events",
    "POST /v1/models/{subject_id}/lease": "api lease open",
    "POST /v1/leases/{lease_id}/heartbeat": "api lease heartbeat",
    "DELETE /v1/leases/{lease_id}": "api lease release",
}

#: The routes with no verb, each with the reason. Both halves of the pairing
#: dance a REQUESTING app does, and the orchestrator's relation to its engine.
EXCLUDED: dict[str, str] = {
    "POST /v1/pairing/start": "the requesting app's half; this CLI already has a token",
    "POST /v1/pairing/poll": "the requesting app's half; this CLI already has a token",
    "GET /v1/peer": "PHASE17: the orchestrator's relation, not a client's",
    "POST /v1/peer/claim": "PHASE17: the orchestrator's relation, not a client's",
    "DELETE /v1/peer/claim": "PHASE17: the orchestrator's relation, not a client's",
}


def test_every_api_route_has_a_verb_or_a_stated_reason(
    make_app: Callable[..., FastAPI]
) -> None:
    app = make_app(enable_llm=True, enable_tts=True, enable_asr=True,
                   enable_align=True, enable_rvc=True, enable_denoise=True)
    # THE OPENAPI DOCUMENT, not `app.routes`. The first draft of this walked the
    # route list and found nothing: this FastAPI wraps every `include_router`
    # into a `_IncludedRouter` whose real routes hang off `original_router` with
    # the prefix in a separate `include_context`, so a flat `isinstance(route,
    # APIRoute)` matched only `GET /`. Measured 2026-09-16 — and the test PASSED
    # its first assertion while doing it, because an empty set is a subset of
    # everything. `app.openapi()` is the server's own published answer to "what
    # do you serve", and it is stable across the versions that private class is
    # not. The one route it omits is `GET /`, the operator page, which is
    # `include_in_schema=False` and is not an API route.
    served = {
        f"{method.upper()} {path}"
        for path, verbs in app.openapi()["paths"].items()
        for method in verbs
        if method.upper() not in ("HEAD", "OPTIONS")
    }
    assert served, "the server published no paths; the probe above is broken"

    uncovered = served - set(COVERED) - set(EXCLUDED)
    assert uncovered == set(), (
        "these routes have no `crucible api` verb and no stated reason: "
        f"{sorted(uncovered)}"
    )
    stale = (set(COVERED) | set(EXCLUDED)) - served
    assert stale == set(), (
        f"these are claimed but the server does not serve them: {sorted(stale)}"
    )
