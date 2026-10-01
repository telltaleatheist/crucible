from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI

from crucible import cli
from crucible.cli import api_cmd
from crucible.client import PAIRING_ENV, ClientRefusal, Connection

from .conftest import TOKEN
from .live_server import serve


@pytest.fixture
def base(make_app: Callable[..., FastAPI]) -> Iterator[str]:
    with serve(make_app()) as url:
        yield url


def run(base: str, *argv: str) -> int:
    return cli.main(["api", "--url", base, "--token", TOKEN, *argv])


def lines(captured: str) -> list[Any]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def _namespace(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {"url": None, "token": None, "pairing": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_url_without_a_token_is_refused_and_never_borrows_the_local_one() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        api_cmd.resolve(_namespace(url="http://192.168.68.20:7100"))
    assert "token_required" in str(refusal.value)


def test_a_token_without_a_url_is_refused_by_name() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        api_cmd.resolve(_namespace(token="abc"))
    assert "url_required" in str(refusal.value)


def test_a_pairing_line_carries_the_address_the_name_and_the_token() -> None:
    resolved = api_cmd.resolve(
        _namespace(pairing="crucible://crucible%40mac-studio@192.168.68.20:7100/#abc123")
    )
    assert resolved.url == "http://192.168.68.20:7100"
    assert resolved.name == "crucible@mac-studio"
    assert resolved.token == "abc123"
    assert resolved.source == "--pairing"


def test_a_pairing_line_cannot_be_combined_with_url_or_token() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        api_cmd.resolve(
            _namespace(pairing="crucible://a@h:1/#t", url="http://elsewhere:7100")
        )
    assert "connection_overspecified" in str(refusal.value)


def test_a_line_that_is_not_a_pairing_line_is_refused_by_name() -> None:
    with pytest.raises(ClientRefusal) as refusal:
        api_cmd.resolve(_namespace(pairing="http://192.168.68.20:7100"))
    assert "pairing_line_invalid" in str(refusal.value)


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
    assert run(base, "health") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_the_bearer_token_is_what_the_server_checks(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["api", "--url", base, "--token", "not-the-token", "info"]) == 1
    captured = capsys.readouterr()
    assert "HTTP 401 unauthorized" in captured.err
    assert "bearer token is not this server's token" in captured.err
    assert "not accepted by 127.0.0.1" in captured.err
    assert "`crucible pair <address>`" in captured.err
    assert "{" not in captured.err, "a mapped refusal is a sentence, not the JSON"


def test_an_api_version_refusal_says_which_side_is_older(
    capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import urllib.error
    from email.message import Message

    def refusal(server: int, client: int) -> urllib.error.HTTPError:
        body = json.dumps({"error": {
            "code": "api_version_mismatch",
            "message": f"client speaks API version {client}, this server speaks {server}",
            "details": {"server_api_version": server, "client_api_version": client},
        }}).encode("utf-8")
        return urllib.error.HTTPError(
            "http://kylies-pc:7100/v1/info", 426, "Upgrade Required", Message(),
            io.BytesIO(body),
        )

    remote = Connection(
        url="http://kylies-pc:7100", token="t", name="kylies-pc",
        source="--server kylies-pc",
    )
    assert api_cmd.report_http_error(refusal(server=1, client=2), remote) == 1
    older_there = capsys.readouterr().err
    assert "kylies-pc is older" in older_there
    assert "Update Crucible on kylies-pc" in older_there

    assert api_cmd.report_http_error(refusal(server=3, client=1), remote) == 1
    older_here = capsys.readouterr().err
    assert "this computer is older" in older_here
    assert "Update Crucible here" in older_here


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
    assert [row["id"] for row in events] == sorted(row["id"] for row in events)

    state = printed[-2]
    assert state["status"] == api_cmd.SUCCEEDED
    assert state["artifacts"] == ["page.txt"]
    assert printed[-1]["artifacts_saved"][0]["name"] == "page.txt"
    assert (out / "page.txt").read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_submit_with_queue_says_whether_it_waited_and_the_queue_lists_it(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.txt"
    source.write_text("a page", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo", "--queue", "600",
               "--input", f"page.txt={source}") == 0
    receipt = json.loads(capsys.readouterr().out)
    assert set(receipt) == {"job_id", "resume_id", "queued", "position"}
    assert run(base, "queue", "list") == 0
    listed = json.loads(capsys.readouterr().out)
    assert set(listed) == {"items", "depth", "limits"}


def test_the_queue_flag_takes_the_default_or_a_number() -> None:
    def body(queue: Any) -> dict[str, Any]:
        return api_cmd.job_body(argparse.Namespace(
            type="echo", model=None, params=None, resume=None, queue=queue))
    assert "queue" not in body(None)
    assert body(0)["queue"] == {}
    assert body(600)["queue"] == {"max_wait_s": 600}


def test_a_failed_job_exits_nonzero_even_though_every_request_succeeded(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "job", "submit", "--type", "echo", "--follow") == 1
    printed = lines(capsys.readouterr().out)
    assert printed[-1]["status"] == "failed"
    assert printed[-1]["error"]["code"] == "no_inputs"


def test_a_server_refusal_is_printed_verbatim_with_its_code(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
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
    source = tmp_path / "in.txt"
    source.write_text("x", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo",
               "--params", '{"delay_ms": 20000}',
               "--input", f"page.txt={source}") == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    assert run(base, "job", "cancel", job_id) == 0
    assert json.loads(capsys.readouterr().out)["status"] in ("cancelling", "cancelled")


def test_a_hold_keeps_the_job_and_a_release_removes_it(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x", encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo", "--params", '{"delay_ms": 0}',
               "--input", f"page.txt={source}", "--follow") == 0
    job_id = json.loads(capsys.readouterr().out.strip().splitlines()[0])["job_id"]
    assert run(base, "job", "hold", job_id) == 0
    held = json.loads(capsys.readouterr().out)
    assert (held["job_id"], held["held"]) == (job_id, True)
    assert run(base, "job", "release", job_id) == 0
    assert json.loads(capsys.readouterr().out) == {"released": job_id}
    assert run(base, "job", "get", job_id) == 1
    refusal = capsys.readouterr().err
    assert "job_reaped" in refusal and "released" in refusal


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
    assert lines(capsys.readouterr().out)[-1]["status"] == api_cmd.SUCCEEDED


def test_uploading_a_file_that_is_not_there_is_refused_before_any_request(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "upload", str(tmp_path / "nope.wav")) == 1
    assert "input_missing" in capsys.readouterr().err


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


def _frames(*rows: dict[str, Any]) -> Callable[..., Iterator[dict[str, Any]]]:
    def fake_follow(connection: Any, path: str, *, last_event_id: int = 0):
        for index, row in enumerate(rows, start=1):
            yield {"id": index, **row}

    return fake_follow


def test_until_stops_at_that_rows_done_and_not_at_anothers(
    base: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(api_cmd, "follow", _frames(
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
    monkeypatch.setattr(api_cmd, "follow", _frames(
        {"event": "ready", "data": {"voice": "sigma", "sample_rate": 24000}},
        {"event": "error", "data": {"id": "r1", "code": "engine_failed", "message": "no"}},
    ))
    assert run(base, "stream", "events", "s1", "--until", "r1") == 1
    assert lines(capsys.readouterr().out)[-1]["data"]["code"] == "engine_failed"


def test_the_wav_is_written_at_the_rate_the_session_reported(
    base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import base64
    import wave

    pcm = b"\x00\x01" * 1200
    monkeypatch.setattr(api_cmd, "follow", _frames(
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
        assert handle.getnframes() == len(pcm)
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


def test_an_address_nothing_answers_on_is_not_a_refusal(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["api", "--url", "http://127.0.0.1:1", "--token", "t", "ping"]) == 1
    err = capsys.readouterr().err
    assert "server_unreachable" in err
    assert "http://127.0.0.1:1" in err and "from --url/--token" in err
    assert "local engine" not in err
    assert "`crucible doctor`" in err and "`crucible lan enable`" in err


def test_an_unreachable_pairing_names_the_machine_it_dialled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(
        PAIRING_ENV, "crucible://kylies-pc@127.0.0.1:1/#not-a-real-token"
    )
    assert cli.main(["api", "ping"]) == 1
    err = capsys.readouterr().err
    assert "server_unreachable: kylies-pc at http://127.0.0.1:1" in err
    assert f"from ${PAIRING_ENV}" in err
    assert "On kylies-pc, run `crucible doctor`" in err


def test_a_closed_pipe_is_success_and_not_an_unreachable_server(
    base: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def closed(value: Any) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(api_cmd, "emit", closed)
    assert run(base, "info") == 0


def test_stream_open_reaches_the_route_and_is_refused_for_the_servers_own_reason(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "stream", "open", "--voice", "sigma", "--language", "en") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "job_type_disabled"


def test_lease_open_reaches_the_route_with_both_required_fields(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "lease", "open", "qwen3.5-9b", "--act", "clean", "--ttl", "600") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "not_resident"


def test_task_submit_reaches_the_route_with_the_fields_its_type_needs(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "task", "submit", "--type", "pull",
               "--kind", "model", "--id", "no-such-model") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "unknown_subject"


def test_a_post_with_no_body_at_all_reaches_its_route(
    base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(base, "upstream-test", "no-such-upstream") == 1
    first = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert first["error"]["code"] == "unknown_upstream"

    assert run(base, "lease", "heartbeat", "no-such-lease") == 1
    second = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert second["error"]["code"] == "unknown_lease"


@pytest.fixture
def tts_base(make_app: Callable[..., FastAPI]) -> Iterator[str]:
    with serve(make_app(enable_tts=True)) as url:
        yield url


def test_voice_write_carries_the_whole_manifest_document(
    tts_base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"not_a_voice": {}}), encoding="utf-8")
    assert run(tts_base, "voice-write", "made-up",
               "--manifest", f"@{manifest}") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "voice_invalid"


def test_voice_remove_reaches_the_route_and_is_told_a_shipped_voice_is_not_custom(
    tts_base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tts_base, "voice-remove", "zeroshot") == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "voice_not_custom"


def test_voice_remove_of_an_unknown_id_is_not_an_error(
    tts_base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tts_base, "voice-remove", "made-up") == 0
    assert json.loads(capsys.readouterr().out) == {"removed": {"manifest": "made-up"}}


def test_a_params_file_is_read_and_sent(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"delay_ms": 999999}), encoding="utf-8")
    assert run(base, "job", "submit", "--type", "echo",
               "--params", f"@{params}", "--follow") == 1
    assert "delay_ms" in json.dumps(lines(capsys.readouterr().out)[-1])


DECIDE_ARGV = (
    "decide", "--model", "qwen3.5-9b",
    "--state", "I was charged twice for March, please refund one.",
    "--choice", "team", "Which team should handle this?",
    "billing=Payment and invoice issues", "technical=Bugs and errors",
    "--score", "anger", "How frustrated is the customer?",
    "Calm,Frustrated but civil,Very angry",
    "--yesno", "urgent", "The message conveys urgency",
)

DECIDE_BODY = {
    "model": "qwen3.5-9b",
    "state": "I was charged twice for March, please refund one.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "options": {"billing": "Payment and invoice issues",
                             "technical": "Bugs and errors"}},
        "anger": {"type": "score", "instructions": "How frustrated is the customer?",
                  "levels": ["Calm", "Frustrated but civil", "Very angry"]},
        "urgent": {"type": "yesno", "instructions": "The message conveys urgency"},
    },
}


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def record(connection: Any, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"method": method, "path": path, **kwargs})
        return {}

    monkeypatch.setattr(api_cmd, "call", record)
    return calls


def test_decide_builds_the_contract_s_worked_example_from_snap_s_grammar(
    recorded: list[dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("http://127.0.0.1:1", *DECIDE_ARGV) == 0
    assert recorded == [{"method": "POST", "path": "/v1/decide",
                         "json_body": DECIDE_BODY, "extra_headers": None}]
    assert list(recorded[0]["json_body"]["questions"]) == ["team", "anger", "urgent"]

    state = tmp_path / "ticket.txt"
    state.write_text("from a file", encoding="utf-8")
    image = tmp_path / "page.png"
    image.write_bytes(b"not really a png")
    recorded.clear()
    assert run("http://127.0.0.1:1", "decide", "--model", "m", "--state", f"@{state}",
               "--yesno", "q", "is it", "--act", "analysis") == 0
    assert recorded[0]["json_body"]["state"] == "from a file"
    assert recorded[0]["extra_headers"] == {"X-Crucible-Act": "analysis"}
    recorded.clear()
    assert run("http://127.0.0.1:1", "decide", "--model", "m", "--image", str(image),
               "--yesno", "q", "is it") == 0
    assert recorded[0]["json_body"] == {
        "model": "m", "state": "", "images": ["bm90IHJlYWxseSBhIHBuZw=="],
        "questions": {"q": {"type": "yesno", "instructions": "is it"}},
    }


def test_decide_missing_travels_only_when_given(recorded: list[dict[str, Any]]) -> None:
    for word in ("report", "refuse", "sometimes"):
        recorded.clear()
        assert run("http://127.0.0.1:1", *DECIDE_ARGV, "--missing", word) == 0
        assert recorded[0]["json_body"] == {**DECIDE_BODY, "missing": word}


def test_decide_missing_report_reaches_the_door_and_an_unknown_word_is_its_400(
    make_app: Callable[..., FastAPI],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .live_server import run_job

    def probs_for(messages: list[dict[str, Any]]) -> dict[str, float]:
        text = json.dumps(messages)
        if "How frustrated" in text:
            return {"A": 0.3, "B": 0.2, "so": 0.09, "I": 0.08, "Um": 0.07,
                    "Well": 0.06, "It": 0.05, "C": 0.001}
        return {"A": 0.83, "B": 0.17}

    engine_factory(probs_for=probs_for)
    fake_weights("qwen3.5-9b")
    with serve(make_app(enable_llm=True)) as url:
        run_job(url, auth, type="load-model", model="qwen3.5-9b")
        capsys.readouterr()
        assert run(url, *DECIDE_ARGV, "--missing", "report") == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["answers"]["anger"]["missing_labels"] == ["Very angry"]
        assert printed["answers"]["anger"]["probabilities"]["Very angry"] is None
        assert printed["answers"]["team"]["missing_labels"] == []
        assert run(url, *DECIDE_ARGV, "--missing", "sometimes") == 1
    refusal, _ = json.JSONDecoder().raw_decode(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "invalid_request"
    assert ["body", "missing"] in [p["location"] for p in refusal["error"]["details"]["problems"]]


@pytest.mark.parametrize(("argv", "code"), [
    (("--yesno", "q", "is it"), "decide_needs_state"),
    (("--state", "s"), "decide_needs_a_question"),
    (("--state", "s", "--yesno", "q", "a", "--yesno", "q", "b"), "decide_question_repeated"),
    (("--state", "s", "--choice", "q", "pick", "a=1", "a=2"), "decide_option_repeated"),
    (("--state", "s", "--choice", "q", "pick", "noequals"), "--choice_malformed"),
    (("--state", "s", "--choice", "q"), "decide_choice_malformed"),
    (("--image", "no-such-image.png", "--yesno", "q", "a"), "image_missing"),
])
def test_decide_refuses_a_malformed_command_line_before_any_request(
    recorded: list[dict[str, Any]], capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...], code: str,
) -> None:
    assert run("http://127.0.0.1:1", "decide", "--model", "m", *argv) == 1
    assert code in capsys.readouterr().err
    assert recorded == []


@pytest.fixture
def llm_base(
    make_app: Callable[..., FastAPI], fake_env: Path, idle_card: None,
) -> Iterator[str]:
    with serve(make_app(enable_llm=True)) as url:
        yield url


def test_decide_reaches_the_door_and_is_refused_for_the_servers_own_reason(
    llm_base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(llm_base, *DECIDE_ARGV) == 1
    refusal = json.loads(capsys.readouterr().err.split("\n", 1)[1])
    assert refusal["error"]["code"] == "model_not_resident"


def test_decide_prints_the_door_s_answer_for_the_worked_example(
    make_app: Callable[..., FastAPI],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .live_server import run_job

    def probs_for(messages: list[dict[str, Any]]) -> dict[str, float]:
        text = json.dumps(messages)
        if "Which team" in text:
            return {"A": 0.91, "B": 0.09}
        if "How frustrated" in text:
            return {"A": 0.62, "B": 0.36, "C": 0.02}
        return {"A": 0.83, "B": 0.17}

    engine_factory(probs_for=probs_for)
    fake_weights("qwen3.5-9b")
    with serve(make_app(enable_llm=True)) as url:
        run_job(url, auth, type="load-model", model="qwen3.5-9b")
        capsys.readouterr()
        assert run(url, *DECIDE_ARGV) == 0
    printed = json.loads(capsys.readouterr().out)
    assert list(printed["answers"]) == ["team", "anger", "urgent"]
    assert printed["answers"]["team"]["choice"] == "billing"
    assert printed["answers"]["anger"]["level"] == "Calm"
    assert printed["answers"]["anger"]["score"] == pytest.approx(1.4, abs=1e-6)
    assert printed["answers"]["urgent"]["p"] == pytest.approx(0.83, abs=1e-6)
    assert printed["model"]["id"] == "qwen3.5-9b"


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
    "POST /v1/decide": "api decide",
    "POST /v1/uploads": "api upload",
    "POST /v1/jobs": "api job submit",
    "GET /v1/jobs/{job_id}": "api job get",
    "DELETE /v1/jobs/{job_id}": "api job cancel",
    "POST /v1/jobs/{job_id}/hold": "api job hold",
    "DELETE /v1/jobs/{job_id}/hold": "api job release",
    "GET /v1/jobs/{job_id}/events": "api job events",
    "GET /v1/jobs/{job_id}/artifacts/{name}": "api job artifact",
    "GET /v1/resumable": "api resumable list",
    "GET /v1/resumable/{resume_id}": "api resumable get",
    "DELETE /v1/resumable/{resume_id}": "api resumable discard",
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
    "GET /v1/queue": "api queue list",
    "DELETE /v1/queue/{job_id}": "api queue remove",
    "POST /v1/queue/{job_id}/heartbeat": "api queue heartbeat",
    "GET /v1/queue/events": "api queue events",
    "GET /v1/events": "api events",
}

EXCLUDED: dict[str, str] = {
    "POST /v1/voices/updates": "`crucible voices check-updates` posts it",
    "POST /v1/pairing/start": "the requesting app's half; this CLI already has a token",
    "POST /v1/pairing/poll": "the requesting app's half; this CLI already has a token",
    "GET /v1/peer": "PHASE17: the orchestrator's relation, not a client's",
    "POST /v1/peer/claim": "PHASE17: the orchestrator's relation, not a client's",
    "DELETE /v1/peer/claim": "PHASE17: the orchestrator's relation, not a client's",
    "GET /v1/capability/plan": "the operator page's install modal reads its sentences",
    "GET /v1/voices/{voice_id}/manifest": "the operator page's voice editor reads it",
    "GET /v1/playground": "the operator page's playground reads its forms",
}


def test_every_api_route_has_a_verb_or_a_stated_reason(
    make_app: Callable[..., FastAPI]
) -> None:
    app = make_app(enable_llm=True, enable_tts=True, enable_asr=True,
                   enable_align=True, enable_rvc=True, enable_denoise=True)
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


def test_capability_passes_a_client_size_through_for_the_server_to_judge(
    recorded: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("http://127.0.0.1:1", "capability") == 0
    assert recorded[-1]["path"] == "/v1/capability"
    assert run("http://127.0.0.1:1", "capability", "--class", "generate",
               "--context-tokens", "40960", "--concurrency", "1") == 0
    assert recorded[-1]["path"] == (
        "/v1/capability?class=generate&context_tokens=40960&concurrency=1"
    )
