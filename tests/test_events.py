from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import cli, features
from crucible import events as events_module
from crucible.api import sse
from crucible.errors import ApiError
from crucible.events import EventHub
from crucible.jobs.base import DONE
from crucible.tasks import Task

from .conftest import TOKEN
from .fake_engine import FakeEngine
from .live_server import run_job, serve
from .test_queue import body, free_the_lane, occupy_the_lane, queue_up, status, wait_for

MODEL = "qwen3.5-9b"
SEEN_TIMEOUT = 20.0


@pytest.fixture
def quick_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(events_module, "THROTTLE_S", 0.2)


class Reader:
    """GET /v1/events read on a thread, the way an app holds it open."""

    def __init__(self, base: str, auth: dict[str, str], query: str = "",
                 last_event_id: int | None = None) -> None:
        headers = dict(auth)
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        self._url = f"{base}/v1/events{query}"
        self._headers = headers
        self.events: list[dict[str, Any]] = []
        self.status: int | None = None
        self.ended = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            with httpx.stream("GET", self._url, headers=self._headers,
                              timeout=httpx.Timeout(60.0, connect=10.0)) as response:
                self.status = response.status_code
                current: dict[str, Any] = {}
                for line in response.iter_lines():
                    if line == "":
                        if current:
                            with self._lock:
                                self.events.append(current)
                            current = {}
                        continue
                    if line.startswith(":"):
                        continue
                    name, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if name == "id":
                        current["id"] = int(value)
                    elif name == "event":
                        current["event"] = value
                    elif name == "data":
                        current["data"] = json.loads(value)
        finally:
            self.ended.set()

    def seen(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)

    def names(self) -> list[str]:
        return [event["event"] for event in self.seen()]

    def wait_for(self, condition: Callable[[list[dict[str, Any]]], bool], what: str) -> None:
        deadline = time.monotonic() + SEEN_TIMEOUT
        while time.monotonic() < deadline:
            if condition(self.seen()):
                return
            time.sleep(0.02)
        raise AssertionError(f"never saw {what}; saw {self.names()}")

    def wait_for_event(self, name: str, **match: Any) -> dict[str, Any]:
        def found(seen: list[dict[str, Any]]) -> bool:
            return any(_matches(event, name, match) for event in seen)

        self.wait_for(found, f"{name} {match}")
        return next(event for event in self.seen() if _matches(event, name, match))


def _matches(event: dict[str, Any], name: str, match: dict[str, Any]) -> bool:
    return event["event"] == name and all(
        event["data"].get(key) == value for key, value in match.items()
    )


def _of_job(seen: list[dict[str, Any]], job_id: str) -> list[str]:
    return [
        event["event"] for event in seen
        if event["event"].startswith("job.") and event["data"]["job_id"] == job_id
    ]


async def _open(hub: EventHub, topics: frozenset[str], after: int | None) -> sse.Feed:
    return sse._hub_feed(hub, topics, after, lambda: {"state": "now"})


def _drain(feed: sse.Feed, cursor: int) -> tuple[list[dict[str, Any]], int]:
    taken = feed.after(cursor)
    for position, _ in taken:
        cursor = position
        feed.moved(cursor)
    return [event for _, event in taken], cursor


# --- the hub on its own ------------------------------------------------------------


def test_a_stream_opens_with_a_snapshot_and_ids_only_grow() -> None:
    async def run() -> None:
        hub = EventHub()
        feed = await _open(hub, frozenset(events_module.TOPICS), None)
        hub.publish(events_module.JOB, "job.queued", {"job_id": "a"})
        hub.publish(events_module.QUEUE, "queue.added", {"job_id": "a"})
        seen, _ = _drain(feed, 0)
        assert [event["event"] for event in seen] == ["snapshot", "job.queued", "queue.added"]
        assert seen[0]["data"]["gap"] is False and seen[0]["data"]["state"] == "now"
        ids = [event["id"] for event in seen]
        assert ids == sorted(ids) and len(set(ids)) == 3
        assert ids[0] > 10 ** 15, "ids start at the hub's birth in microseconds"
        assert "at" in seen[1]["data"]
        feed.close()

    asyncio.run(run())


def test_resume_replays_the_history_and_a_rolled_history_sends_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(events_module, "HISTORY", 5)

    async def run() -> None:
        hub = EventHub()
        for index in range(3):
            hub.publish(events_module.JOB, "job.queued", {"job_id": f"j{index}"})
        middle = hub.last_id - 1
        feed = await _open(hub, frozenset(events_module.TOPICS), middle)
        seen, _ = _drain(feed, middle)
        assert [event["data"]["job_id"] for event in seen] == ["j2"], "no snapshot on resume"
        feed.close()

        stale = hub.last_id
        for index in range(10):
            hub.publish(events_module.JOB, "job.queued", {"job_id": f"k{index}"})
        rolled = await _open(hub, frozenset(events_module.TOPICS), stale)
        seen, _ = _drain(rolled, stale)
        assert seen[0]["event"] == "snapshot" and seen[0]["data"]["gap"] is True
        assert len(seen) == 1, "the snapshot stands in for what the history lost"
        rolled.close()

        before_a_restart = await _open(hub, frozenset(events_module.TOPICS), 7)
        seen, _ = _drain(before_a_restart, 7)
        assert seen[0]["event"] == "snapshot" and seen[0]["data"]["gap"] is True
        before_a_restart.close()

    asyncio.run(run())


def test_a_slow_reader_overflows_and_never_holds_the_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(events_module, "SUBSCRIBER_LIMIT", 3)

    async def run() -> None:
        hub = EventHub()
        feed = await _open(hub, frozenset(events_module.TOPICS), None)
        started = time.monotonic()
        for index in range(50):
            hub.publish(events_module.JOB, "job.progress", {"job_id": "a", "n": index})
        assert time.monotonic() - started < 1.0
        seen, cursor = _drain(feed, 0)
        assert [event["event"] for event in seen] == ["overflow"]
        assert seen[0]["data"]["last_event_id"] == 0
        assert "Last-Event-ID" in seen[0]["data"]["message"]
        assert events_module.OVERFLOW in feed.ends
        feed.close()
        other = await _open(hub, frozenset(events_module.TOPICS), None)
        hub.publish(events_module.JOB, "job.done", {"job_id": "a"})
        assert [e["event"] for e in _drain(other, 0)[0]] == ["snapshot", "job.done"]
        other.close()

    asyncio.run(run())


def test_throttled_events_lead_coalesce_and_keep_the_last_word(quick_throttle: None) -> None:
    async def run() -> None:
        hub = EventHub()
        hub.bind(asyncio.get_running_loop())
        feed = await _open(hub, frozenset(events_module.TOPICS), None)
        for fraction in (0.1, 0.2, 0.3, 0.4):
            hub.publish_throttled(events_module.JOB, "job.progress", "job:a",
                                  {"fraction": fraction})
        hub.publish_throttled(events_module.JOB, "job.progress", "job:b", {"fraction": 0.5})
        await asyncio.sleep(0.5)
        hub.publish_throttled(events_module.JOB, "job.progress", "job:a", {"fraction": 0.4})
        await asyncio.sleep(0.3)
        seen, _ = _drain(feed, 0)
        sent = [(e["data"]["fraction"]) for e in seen if e["event"] == "job.progress"]
        assert sent == [0.1, 0.5, 0.4], "first at once, per key, then only the latest"

        hub.publish_throttled(events_module.JOB, "job.progress", "job:c", {"fraction": 0.1})
        hub.publish_throttled(events_module.JOB, "job.progress", "job:c", {"fraction": 0.9})
        hub.forget("job:c")
        await asyncio.sleep(0.4)
        later, _ = _drain(feed, seen[-1]["id"])
        assert [e["data"]["fraction"] for e in later] == [0.1], "a forgotten key says no more"
        feed.close()

    asyncio.run(run())


def test_topics_filter_and_the_server_topic_always_arrives() -> None:
    assert events_module.parse_topics(None) == frozenset(events_module.TOPICS)
    assert events_module.parse_topics("job, card") == frozenset({"job", "card", "server"})
    with pytest.raises(ApiError) as refused:
        events_module.parse_topics("jobs")
    assert refused.value.code == "unknown_topic"
    assert "job" in refused.value.message

    async def run() -> None:
        hub = EventHub()
        feed = await _open(hub, events_module.parse_topics("card"), None)
        hub.publish(events_module.JOB, "job.queued", {"job_id": "a"})
        hub.publish(events_module.CARD, "card.warming", {"subject": "m"})
        hub.stop("a test stopped it")
        hub.publish(events_module.CARD, "card.loaded", {"subject": "m"})
        seen, _ = _drain(feed, 0)
        assert [e["event"] for e in seen] == ["snapshot", "card.warming", "server.stopping"]
        assert seen[0]["data"]["topics"] == ["card", "server"]
        feed.close()
        with pytest.raises(ApiError) as closed:
            await _open(hub, frozenset(events_module.TOPICS), None)
        assert closed.value.code == "server_stopping"

    asyncio.run(run())


# --- the hooks, through a running app -------------------------------------------------


async def _open_on(client: TestClient, topics: str | None = None) -> sse.Feed:
    app = client.app
    return sse._hub_feed(
        app.state.events, events_module.parse_topics(topics), None,
        lambda: {"queue_depth": len(app.state.line)},
    )


def test_job_lifecycle_and_throttled_progress_in_order(
    client: TestClient, auth: dict[str, str]
) -> None:
    feed = client.portal.call(_open_on, client, "job")
    many = {f"in{index}.bin": {"inline_base64": "eA=="} for index in range(20)}
    answer = client.post("/v1/jobs", headers=auth, json={
        "type": "echo", "params": {"delay_ms": 5}, "inputs": many,
    })
    assert answer.status_code == 202, answer.text
    job_id = answer.json()["job_id"]
    wait_for(lambda: status(client, auth, job_id) == "done", "the echo job to finish")
    seen, _ = _drain(feed, 0)
    names = _of_job(seen, job_id)
    assert names[:2] == ["job.queued", "job.running"]
    assert names[-1] == "job.done"
    progress = names.count("job.progress")
    assert 1 <= progress <= 2, f"21 progress reports inside a second are throttled: {names}"
    done = [e for e in seen if e["event"] == "job.done"][0]["data"]
    assert done["status"] == "done" and len(done["artifacts"]) == 20
    assert done["type"] == "echo"
    feed.close()


def test_a_cancelled_and_a_removed_job_end_by_name(
    client: TestClient, auth: dict[str, str]
) -> None:
    feed = client.portal.call(_open_on, client, None)
    holder = occupy_the_lane(client, auth)
    waiting = queue_up(client, auth)["job_id"]
    client.delete(f"/v1/queue/{waiting}", headers=auth)
    free_the_lane(client, auth, holder)
    wait_for(lambda: status(client, auth, holder) == "cancelled", "the lane job to cancel")
    seen, _ = _drain(feed, 0)
    changes = [name for name in _of_job(seen, holder) if name != "job.progress"]
    assert changes == ["job.queued", "job.running", "job.cancelled"]
    assert _of_job(seen, waiting) == ["job.queued", "job.removed"]
    queued = [e for e in seen if e["event"] == "job.queued"
              and e["data"]["job_id"] == waiting][0]["data"]
    assert queued["waiting"] is True and queued["position"] == 1
    removed = [e for e in seen if e["event"] == "job.removed"][0]["data"]
    assert removed["removal"]["reason"] == "operator"
    queue = [(e["event"], e["data"]["job_id"], e["data"]["kind"])
             for e in seen if e["event"].startswith("queue.")]
    assert queue == [("queue.added", waiting, "job"), ("queue.removed", waiting, "job")]
    feed.close()


def test_a_queued_job_leaves_the_line_then_runs(
    client: TestClient, auth: dict[str, str]
) -> None:
    feed = client.portal.call(_open_on, client, "job,queue")
    holder = occupy_the_lane(client, auth)
    waiting = queue_up(client, auth)["job_id"]
    free_the_lane(client, auth, holder)
    wait_for(lambda: status(client, auth, waiting) == "done", "the queued job to run")
    seen, _ = _drain(feed, 0)
    queue = [(e["event"], e["data"]["job_id"]) for e in seen if e["event"].startswith("queue.")]
    assert queue == [("queue.added", waiting), ("queue.started", waiting)]
    assert _of_job(seen, waiting)[:2] == ["job.queued", "job.running"]
    assert _of_job(seen, waiting)[-1] == "job.done"
    feed.close()


def test_settings_writes_are_announced(client: TestClient, auth: dict[str, str]) -> None:
    feed = client.portal.call(_open_on, client, "settings")
    client.app.state.settings_history.record(act=None, client="bookforge", changed=["routes"])
    seen, _ = _drain(feed, 0)
    written = [e for e in seen if e["event"] == "settings.written"]
    assert len(written) == 1
    assert written[0]["data"]["client"] == "bookforge"
    assert written[0]["data"]["changed"] == ["routes"]
    feed.close()


def test_a_task_says_it_runs_progresses_and_ends(
    client: TestClient, quick_throttle: None
) -> None:
    feed = client.portal.call(_open_on, client, "task")
    tasks = client.app.state.tasks
    task = Task(id="t1", type="pull", request={"type": "pull", "kind": "model", "id": MODEL},
                created="now", started="now")

    def go() -> None:
        tasks.append_event(task, "started", {"type": task.type})
        tasks.append_event(task, "step", {"name": "pull", "index": 1, "total": 1})
        for done in (10, 20, 30):
            tasks.append_event(task, "progress", {"bytes_done": done, "bytes_total": 30,
                                                  "file": "w.bin"})
        tasks._finish(task, DONE, None)

    client.portal.call(_on_loop, go)
    seen, _ = _drain(feed, 0)
    names = [e["event"] for e in seen if e["event"].startswith("task.")]
    assert names == ["task.running", "task.step", "task.progress", "task.done"]
    assert seen[-1]["data"]["task_id"] == "t1" and seen[-1]["data"]["state"] == "done"
    feed.close()


async def _on_loop(call: Callable[[], None]) -> None:
    call()


def test_an_unread_stream_overflows_and_the_job_runs_on(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(events_module, "SUBSCRIBER_LIMIT", 2)
    feed = client.portal.call(_open_on, client, None)
    answer = client.post("/v1/jobs", headers=auth, json=body())
    job_id = answer.json()["job_id"]
    wait_for(lambda: status(client, auth, job_id) == "done", "the job to finish")
    seen, _ = _drain(feed, 0)
    assert [e["event"] for e in seen] == ["overflow"]
    feed.close()


def test_features_are_listed_in_info(client: TestClient, auth: dict[str, str]) -> None:
    info = client.get("/v1/info", headers=auth).json()
    assert info["features"] == sorted(features.FEATURES)
    for name in ("events", "events.topics", "queue.jobs", "queue.calls", "playground",
                 "decide.items", "tts.stream", "align", "segment", "video",
                 "image.inpaint"):
        assert name in info["features"], name
    assert not any("lease" in name for name in info["features"])
    assert all(text.strip() for text in features.FEATURES.values())


# --- the wire, through a live server ------------------------------------------------------


def test_the_cli_follows_the_stream_until_the_server_stops(
    make_app: Callable[..., Any], auth: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    exits: list[int] = []
    app = make_app()
    hub: EventHub = app.state.events
    with serve(app) as base:
        follower = threading.Thread(target=lambda: exits.append(cli.main(
            ["api", "--url", base, "--token", TOKEN, "events", "--topics", "job"]
        )), daemon=True)
        follower.start()
        wait_for(lambda: hub.subscribers == 1, "the command to open the stream")
        run_job(base, auth, type="echo", inputs={"x.bin": {"inline_base64": "eA=="}})
    follower.join(SEEN_TIMEOUT)
    assert exits == [0], "the stream ended when the server stopped, and so did the command"
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    names = [event["event"] for event in printed]
    assert names[0] == "snapshot" and names[-1] == "server.stopping"
    assert "job.done" in names
    assert hub.subscribers == 0


@pytest.fixture
def live_llm(
    make_app: Callable[..., Any],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    quick_throttle: None,
) -> Callable[..., Any]:
    def start(**engine_options: Any) -> Any:
        engine_factory(**engine_options)
        fake_weights(MODEL)
        return serve(make_app(enable_llm=True))

    return start


def _until(release: threading.Event) -> Callable[[dict[str, Any]], float]:
    def held(body: dict[str, Any]) -> float:
        release.wait(SEEN_TIMEOUT)
        return 0.0

    return held


def test_the_wire_card_and_chat_events_and_resume(
    live_llm: Callable[..., Any], auth: dict[str, str]
) -> None:
    release = threading.Event()
    with live_llm(delay_for=_until(release)) as base:
        reader = Reader(base, auth)
        reader.wait_for(lambda seen: bool(seen), "the snapshot")
        opening = reader.seen()[0]
        assert opening["event"] == "snapshot"
        assert opening["data"]["gap"] is False
        assert set(opening["data"]) >= {"activity", "queue", "tasks", "topics"}
        assert opening["data"]["activity"]["server"]["name"] == "crucible@test"

        run_job(base, auth, type="load-model", model=MODEL)
        warming = reader.wait_for_event("card.warming", subject=MODEL)
        assert warming["data"]["kind"] == "llm"
        loaded = reader.wait_for_event("card.loaded", subject=MODEL)
        assert loaded["data"]["kind"] == "llm" and isinstance(loaded["data"]["engine"], str)
        names = reader.names()
        assert names.index("card.warming") < names.index("card.warming_ended") \
            < names.index("card.loaded")

        chat = threading.Thread(target=lambda: httpx.post(
            f"{base}/v1/openai/chat/completions", headers=auth, timeout=60.0,
            json={"model": MODEL, "messages": [{"role": "user", "content": "Hi."}]},
        ), daemon=True)
        chat.start()
        reader.wait_for_event("chat.in_flight", in_flight=1)
        release.set()
        chat.join(SEEN_TIMEOUT)
        idle = reader.wait_for_event("chat.in_flight", in_flight=0)
        assert idle["data"]["by_model"] == {}
        busy = [e for e in reader.seen() if e["event"] == "chat.in_flight"
                and e["data"]["in_flight"] == 1][0]
        assert busy["data"]["by_model"] == {MODEL: 1}

        mark = loaded["id"]
        run_job(base, auth, type="unload-model", model=MODEL)
        reader.wait_for_event("card.unloading", subject=MODEL)
        reader.wait_for_event("card.unloaded", subject=MODEL)

        resumed = Reader(base, auth, last_event_id=mark)
        resumed.wait_for_event("card.unloaded", subject=MODEL)
        replay = resumed.seen()
        assert replay[0]["event"] != "snapshot", "a resume inside the history sends none"
        assert replay[0]["id"] == mark + 1
        assert [e["id"] for e in replay] == sorted(e["id"] for e in replay)
        assert [e["id"] for e in replay] == [e["id"] for e in reader.seen() if e["id"] > mark]

        stale = Reader(base, auth, last_event_id=1)
        stale.wait_for(lambda seen: bool(seen), "the stale reader's snapshot")
        assert stale.seen()[0]["event"] == "snapshot"
        assert stale.seen()[0]["data"]["gap"] is True

        filtered = Reader(base, auth, query="?topics=card")
        filtered.wait_for(lambda seen: bool(seen), "the filtered snapshot")
        run_job(base, auth, type="echo", inputs={"x.bin": {"inline_base64": "eA=="}})
        refused = httpx.get(f"{base}/v1/events?topics=jobs", headers=auth, timeout=10.0)
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "unknown_topic"

    for one in (reader, resumed, stale, filtered):
        assert one.ended.wait(SEEN_TIMEOUT), "a stream held the server's stop open"
        assert one.names()[-1] == "server.stopping"
    assert not any(name.startswith("job.") for name in filtered.names())
    assert any(name.startswith("job.") for name in reader.names())
