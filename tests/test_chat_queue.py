from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from crucible import callqueue
from crucible.api.routes import openai as openai_routes
from crucible.inflight import InFlight
from crucible.jobs.line import Call, WaitingLine
from crucible.queuesessions import QueueSessions
from crucible.settle import Settlement

from .fake_engine import FakeEngine
from .live_server import run_job, serve
from .test_decide_api import EXAMPLE, example_probs

MODEL = "qwen3.5-9b"
SEEN_TIMEOUT = 20.0


@pytest.fixture
def chat_server(
    make_app: Callable[..., Any],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
):
    def start(**engine_options: Any):
        engines = engine_factory(**engine_options)
        fake_weights(MODEL)
        return engines, serve(make_app(enable_llm=True))

    return start


@pytest.fixture
def one_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    def single(engine: str, args: Any) -> tuple[int, str]:
        return 1, "a test allows one completion at a time"

    monkeypatch.setattr(callqueue, "chat_admission", single)
    monkeypatch.setattr(openai_routes, "chat_admission", single)


def _chat(queue: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say something."}],
        **extra,
    }
    if queue is not None:
        body["queue"] = queue
    return body


def _until(release: threading.Event) -> Callable[[dict[str, Any]], float]:
    def held(body: dict[str, Any]) -> float:
        release.wait(SEEN_TIMEOUT)
        return 0.0

    return held


def _as(auth: dict[str, str], name: str) -> dict[str, str]:
    return {**auth, "X-Crucible-Client": name}


def _in_background(call: Callable[[], httpx.Response]) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []

    def run() -> None:
        try:
            out.append(call())
        except Exception as exc:
            out.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + SEEN_TIMEOUT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"never saw {what}")


def _queue(base: str, auth: dict[str, str]) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/queue", headers=auth, timeout=30.0).json()


def _in_flight(base: str, auth: dict[str, str]) -> int:
    activity = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()
    return activity["chat"]["in_flight"]


def _post_chat(base: str, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
    return httpx.post(
        f"{base}/v1/openai/chat/completions", headers=headers, json=body, timeout=60.0
    )


def test_a_queued_chat_loads_its_model_and_is_answered(
    chat_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = chat_server()
    with server as base:
        refused = _post_chat(base, auth, _chat())
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "model_not_resident"
        assert engines == []

        answered = _post_chat(base, _as(auth, "bookforge"), _chat(queue={}))
        assert answered.status_code == 200, answered.text
        assert answered.json()["model"] == MODEL
        assert len(engines) == 1, "one engine was started for the queued chat"
        assert _queue(base, auth)["depth"] == 0


def test_a_queued_chat_waits_for_a_slot_and_is_listed(
    chat_server: Callable[..., Any], auth: dict[str, str], one_slot: None
) -> None:
    release = threading.Event()
    engines, server = chat_server(
        delay_for=_until(release)
    )
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        busy, busy_out = _in_background(lambda: _post_chat(base, auth, _chat()))
        _wait_for(lambda: _in_flight(base, auth) == 1, "the first completion in flight")

        plain = _post_chat(base, auth, _chat())
        assert plain.status_code == 503
        assert plain.json()["error"]["code"] == "chat_queue_full"

        waiting, waiting_out = _in_background(
            lambda: _post_chat(base, _as(auth, "briefcase"), _chat(queue={"max_wait_s": 60}))
        )
        _wait_for(lambda: _queue(base, auth)["depth"] == 1, "the chat in the queue")
        row = _queue(base, auth)["items"][0]
        assert row["kind"] == "call"
        assert row["type"] == "chat"
        assert row["model"] == MODEL
        assert row["client"] == "briefcase"
        assert row["job_id"].startswith("call-")
        assert row["position"] == 1
        activity = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()
        shown = [entry for entry in activity["queued"] if entry["job_id"] == row["job_id"]]
        assert len(shown) == 1 and shown[0]["kind"] == "call", activity["queued"]
        assert shown[0]["waited_s"] is not None, "the desktop Queue lists rows with waited_s"

        release.set()
        busy.join(SEEN_TIMEOUT)
        waiting.join(SEEN_TIMEOUT)
        assert busy_out[0].status_code == 200
        assert waiting_out[0].status_code == 200, waiting_out[0].text
        assert _queue(base, auth)["depth"] == 0


def test_an_operator_removes_a_waiting_chat_and_it_is_told_why(
    chat_server: Callable[..., Any], auth: dict[str, str], one_slot: None
) -> None:
    release = threading.Event()
    _, server = chat_server(delay_for=_until(release))
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        busy, _ = _in_background(lambda: _post_chat(base, auth, _chat()))
        _wait_for(lambda: _in_flight(base, auth) == 1, "the first completion in flight")
        waiting, out = _in_background(
            lambda: _post_chat(base, auth, _chat(queue={}))
        )
        _wait_for(lambda: _queue(base, auth)["depth"] == 1, "the chat in the queue")
        call_id = _queue(base, auth)["items"][0]["job_id"]

        removed = httpx.delete(f"{base}/v1/queue/{call_id}", headers=auth, timeout=30.0)
        assert removed.status_code == 200, removed.text
        waiting.join(SEEN_TIMEOUT)
        assert out[0].status_code == 409
        error = out[0].json()["error"]
        assert error["code"] == "removed_from_queue"
        assert error["details"]["reason"] == "operator"
        assert error["details"]["call_id"] == call_id

        again = httpx.delete(f"{base}/v1/queue/{call_id}", headers=auth, timeout=30.0)
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "not_queued"
        release.set()
        busy.join(SEEN_TIMEOUT)


def test_a_caller_who_leaves_leaves_the_queue(
    chat_server: Callable[..., Any], auth: dict[str, str], one_slot: None
) -> None:
    release = threading.Event()
    _, server = chat_server(delay_for=_until(release))
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        busy, _ = _in_background(lambda: _post_chat(base, auth, _chat()))
        _wait_for(lambda: _in_flight(base, auth) == 1, "the first completion in flight")
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(
                f"{base}/v1/openai/chat/completions", headers=auth,
                json=_chat(queue={}),
                timeout=httpx.Timeout(connect=10.0, read=1.0, write=10.0, pool=10.0),
            )
        _wait_for(lambda: _queue(base, auth)["depth"] == 0, "the queue empty again")
        release.set()
        busy.join(SEEN_TIMEOUT)


def test_a_malformed_queue_is_refused_before_anything_waits(
    chat_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    _, server = chat_server()
    with server as base:
        bad = _post_chat(base, auth, _chat(queue={"max_wait_s": 1}))
        assert bad.status_code == 400
        assert bad.json()["error"]["code"] == "invalid_request"
        assert _queue(base, auth)["depth"] == 0


def test_the_queue_member_never_reaches_the_engine(
    chat_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    seen: list[dict[str, Any]] = []

    def record(body: dict[str, Any]) -> float:
        seen.append(body)
        return 0.0

    _, server = chat_server(delay_for=record)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        answered = _post_chat(base, auth, _chat(queue={}))
        assert answered.status_code == 200
    assert seen and all("queue" not in body for body in seen)


def test_a_queued_decision_loads_its_model_and_is_answered(
    chat_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    _, server = chat_server(probs_for=example_probs)
    with server as base:
        refused = httpx.post(f"{base}/v1/decide", headers=auth, json=EXAMPLE, timeout=60.0)
        assert refused.status_code == 409
        answered = httpx.post(
            f"{base}/v1/decide", headers=auth, json={**EXAMPLE, "queue": {}}, timeout=60.0
        )
        assert answered.status_code == 200, answered.text
        assert set(answered.json()["answers"]) == {"team", "anger", "urgent"}


def test_a_malformed_queued_decision_is_refused_before_it_waits(
    chat_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = chat_server()
    with server as base:
        bad = {**EXAMPLE, "questions": None, "queue": {}}
        refused = httpx.post(f"{base}/v1/decide", headers=auth, json=bad, timeout=60.0)
        assert refused.status_code in (400, 422)
        assert _queue(base, auth)["depth"] == 0
        assert engines == [], "nothing was loaded for a decision that cannot run"


class _Resident:
    def __init__(self, subject: str) -> None:
        self.id = subject


class _Residency:
    claimed_by = None

    def __init__(self, subject: str) -> None:
        self.resident = _Resident(subject)


class _NoJobs:
    def occupied_by_anything_but(self, job_id: str | None) -> None:
        return None


def test_a_call_waiting_for_the_resident_model_holds_it() -> None:
    waiting = {MODEL: 2}
    settlement = Settlement(
        residency=_Residency(MODEL),  # type: ignore[arg-type]
        store=_NoJobs(),
        sessions=QueueSessions(),
        inflight=InFlight(),
        waiting_calls=lambda: waiting,
    )
    held = settlement.holder()
    assert held is not None and held.fact == "a queued call"
    waiting.clear()
    assert settlement.holder() is None
    waiting["another-model"] = 1
    assert settlement.holder() is None


class _Store:
    def __init__(self) -> None:
        self.line: Any = None

    def attach_line(self, line: Any) -> None:
        self.line = line


def test_calls_in_the_line_show_as_calls_and_count_by_model() -> None:
    asyncio.run(_calls_in_the_line())


async def _calls_in_the_line() -> None:
    line = WaitingLine(_Store(), QueueSessions())  # type: ignore[arg-type]
    first = line.join_call(Call(type="chat", model=MODEL, client="a"), 60)
    line.join_call(Call(type="decide", model=MODEL, client="b"), 60)
    line.join_call(Call(type="chat", model="other", client="c"), 60)
    assert line.calls_waiting() == {MODEL: 2, "other": 1}
    rows = line.rows()
    assert [row["kind"] for row in rows] == ["call", "call", "call"]
    assert [row["position"] for row in rows] == [1, 2, 3]
    line.remove(first.job.id, "operator", "a test removed it")
    assert first.outcome is not None and first.outcome.done()
    status, error = first.outcome.result()
    assert status == "removed" and error.code == "removed_from_queue"
    assert [row["position"] for row in line.rows()] == [1, 2]


def test_a_queued_job_that_changes_the_card_waits_for_chats_in_flight(
    chat_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    release = threading.Event()
    _, server = chat_server(delay_for=_until(release))
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        busy, busy_out = _in_background(lambda: _post_chat(base, auth, _chat()))
        _wait_for(lambda: _in_flight(base, auth) == 1, "the first completion in flight")
        submitted = httpx.post(
            f"{base}/v1/jobs", headers=auth, timeout=30.0,
            json={"type": "unload-model", "model": MODEL, "queue": {}},
        )
        assert submitted.status_code == 202, submitted.text
        assert submitted.json()["queued"] is True
        assert [row["kind"] for row in _queue(base, auth)["items"]] == ["job"]
        release.set()
        busy.join(SEEN_TIMEOUT)
        assert busy_out[0].status_code == 200
        job_id = submitted.json()["job_id"]
        _wait_for(
            lambda: httpx.get(f"{base}/v1/jobs/{job_id}", headers=auth, timeout=30.0)
            .json()["status"] == "done",
            "the unload run once the chat was answered",
        )
