from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from crucible.client.connection import Connection
from crucible.desktop_app import instance, screens, theme
from crucible.desktop_app.api import ApiError, LocalApi
from crucible.desktop_app.controller import Controller
from crucible.desktop_app.progress import TaskWatch

RUNNING = {"state": "running", "name": "crucible@test", "version": "1.0.55", "backend": "cuda-linux",
           "detail": "The paired engine is answering"}

INFO = {
    "host": {"backend": "cuda-linux", "gpu": {"vendor": "nvidia", "name": "RTX 3090 Ti", "vram_bytes": 24 * 2**30}},
    "capabilities": [{"job_type": "llm"}, {"job_type": "asr"}],
    "voice_sources": {"repo": {"label": "from its repo"}, "override": {"label": "set on this machine"}},
}

CAPABILITY = {
    "total_bytes": 24 * 2**30,
    "desktop_allowance_bytes": 3 * 2**30,
    "classes": [
        {"capability": "clean", "enabled": True, "summary": "can clean up text, using qwen3.5-9b", "reason": "fits"},
        {"capability": "asr", "enabled": True, "summary": "can transcribe", "reason": "fits"},
        {"capability": "tts", "enabled": False, "summary": "", "reason": "no voice fits in 21.0 GiB"},
    ],
    "job_types": [
        {"job_type": "echo", "classes": ["echo"], "installer": None, "narrator_engines": []},
        {"job_type": "llm", "classes": ["clean"], "installer": "llm", "narrator_engines": []},
        {"job_type": "asr", "classes": ["asr"], "installer": "asr", "narrator_engines": []},
        {"job_type": "tts", "classes": ["tts"], "installer": "tts", "narrator_engines": ["higgs-v3"]},
        {"job_type": "denoise", "classes": ["denoise"], "installer": "rvc", "narrator_engines": []},
    ],
}

ACTIVITY = {
    "resident": {"kind": "llm", "id": "qwen3.5-9b", "memory_bytes_estimate": 20 * 2**30,
                 "held_by": {"who": "'bookforge' for 'decide'"}},
    "running": [{"job_id": "j1", "type": "asr", "model": "whisper", "status": "running", "position": None,
                 "progress": 0.66, "message": "transcribing 2775s of 4196s", "client": "bookforge"}],
    "queued": [{"job_id": "j2", "type": "tts", "model": None, "status": "queued", "position": 1,
                "progress": 0, "message": None, "client": None},
               {"job_id": "j3", "type": "tts", "model": "sigma", "status": "queued", "position": 1,
                "progress": 0, "message": None, "client": "briefcase", "waited_s": 125.0,
                "max_wait_s": 3600}],
    "chat": {"rows": []},
    "session": {"session_id": "ses-1", "client": "bookforge", "act": "decide", "items_run": 3,
                "in_flight": [{"kind": "job", "id": "j1"}], "stream_session": None},
}

QUEUE = {"depth": 3, "items": [
    {"position": 1, "job_id": "j3", "type": "tts", "model": "sigma", "client": "briefcase",
     "submitted": "2026-10-01T12:00:00+00:00", "waited_s": 125.0, "max_wait_s": 3600,
     "session": None, "kind": "job"},
    {"position": 2, "job_id": "call-1", "type": "chat", "model": "qwen3.5-9b", "client": "foundry",
     "submitted": "2026-10-01T12:01:00+00:00", "waited_s": 65.0, "max_wait_s": 600,
     "session": "ses-1", "kind": "call"},
    {"position": 3, "job_id": "ses-2", "type": "session", "model": None, "client": None,
     "submitted": "2026-10-01T12:02:00+00:00", "waited_s": 5.0, "max_wait_s": 600,
     "session": None, "kind": "session"},
]}

NOW = datetime(2026, 10, 1, 12, 2, 5, tzinfo=timezone.utc)

CATALOG = {"rows": [
    {"kind": "model", "id": "qwen3.5-9b", "name": "Qwen 3.5 9B", "job_type": "llm", "installed": True,
     "installed_bytes": 18 * 2**30, "expected_bytes": None, "resident": True},
    {"kind": "model", "id": "qwen3.5-4b", "name": None, "job_type": "llm", "installed": False,
     "installed_bytes": None, "expected_bytes": 8 * 2**30, "resident": False},
    {"kind": "model", "id": "whisper-tiny", "name": None, "job_type": "asr", "installed": True,
     "installed_bytes": 78 * 2**20, "expected_bytes": None, "resident": False},
    {"kind": "voice", "id": "sigma", "name": "Sigma", "job_type": "tts", "installed": True,
     "installed_bytes": 8 * 2**30, "expected_bytes": None, "resident": False},
    {"kind": "voice", "id": "mistborn", "name": "Mistborn", "job_type": "tts", "installed": False,
     "installed_bytes": None, "expected_bytes": 8 * 2**30, "resident": False},
]}

VOICES = [
    {"id": "sigma", "display": "Sigma", "narrator_engine": "higgs-v3", "language": "en", "loadable": True,
     "reason": None, "revision": "8f6b3ca2aaaa", "manifest": "override"},
    {"id": "mistborn", "display": "Mistborn", "narrator_engine": "higgs-v3", "language": "en", "loadable": True,
     "reason": None, "revision": "8a8d1bf3bbbb", "manifest": "repo"},
]

SETTINGS = {
    "upstreams": {"anthropic": {"configured": True, "key_hint": "abcd"}, "openai": {"configured": False, "key_hint": None},
                  "ollama": {"configured": False, "url": None}},
    "upstream_labels": {"anthropic": "Anthropic", "openai": "OpenAI", "ollama": "Ollama"},
    "desktop_allowance_bytes": 3 * 2**30,
    "desktop_reserve": "kept 3.0 GiB for this PC's desktop",
    "lan_advertise": ["192.168.1.5:7100"],
}

SETUP = {"urls": ["http://127.0.0.1:7100", "http://pc.example:7100"],
         "pairing": ["crucible://crucible%40test@127.0.0.1:7100/#secret-token",
                     "crucible://crucible%40test@pc.example:7100/#secret-token"],
         "backend": "cuda-linux",
         "network": {"reachable": True, "urls": ["http://pc.example:7100"],
                     "sentence": "Other devices on the network reach Crucible at http://pc.example:7100.",
                     "how": None, "command": None, "changes": None}}


class FakeApi:
    def __init__(self, docs: dict[str, Any] | None = None) -> None:
        self.docs = {
            "/v1/info": INFO, "/v1/activity": ACTIVITY, "/v1/capability": CAPABILITY,
            "/v1/catalog": CATALOG, "/v1/voices": VOICES, "/v1/tasks": {"tasks": []},
            "/v1/settings": SETTINGS, "/v1/setup": SETUP, "/v1/queue": QUEUE,
        }
        self.docs.update(docs or {})
        self.sent: list[tuple[str, str, Any]] = []
        self.answers: dict[tuple[str, str], Any] = {}
        self.frames: dict[str, list[dict]] = {}
        self.followed: list[tuple[str, int]] = []
        self.forgotten = 0

    def get(self, path: str) -> Any:
        value = self.docs.get(path)
        if isinstance(value, ApiError):
            raise value
        if value is None:
            raise ApiError("not_found", f"no {path}", 404)
        return value

    def send(self, method: str, path: str, body: Any = None) -> Any:
        self.sent.append((method, path, body))
        answer = self.answers.get((method, path))
        if isinstance(answer, ApiError):
            raise answer
        return answer

    def follow(self, path: str, last_event_id: int = 0):
        self.followed.append((path, last_event_id))
        yield from self.frames.get(path, [])

    def forget(self) -> None:
        self.forgotten += 1


class FakeHost:
    def __init__(self, status: Any = RUNNING, lan_supported: bool = True) -> None:
        self._status = status
        self.acted: list[str] = []
        self.lan: dict | None = None
        self.supported = lan_supported
        self.lan_calls: list[bool] = []
        self.opened = 0

    def status(self) -> Any:
        if isinstance(self._status, Exception):
            raise self._status
        return self._status

    def act(self, action: str) -> Any:
        self.acted.append(action)
        return RUNNING

    def lan_supported(self) -> bool:
        return self.supported

    def lan_record(self) -> dict | None:
        return self.lan

    def set_lan(self, on: bool, ask) -> dict:
        self.lan_calls.append(on)
        self.lan = {"authorities": ["192.168.1.5:7100"], "state": "configured"} if on else None
        return {"state": "configured" if on else "disabled"}

    def open_logs(self) -> Path:
        self.opened += 1
        return Path("logs")


def controller(api: FakeApi | None = None, host: FakeHost | None = None, answers: list[bool] | None = None,
               asked: list[str] | None = None) -> Controller:
    replies = list(answers if answers is not None else [True])
    questions = asked if asked is not None else []

    def ask(question: str) -> bool:
        questions.append(question)
        return replies.pop(0) if replies else True
    return Controller(api or FakeApi(), host or FakeHost(), ask, run=lambda work: work(),
                      stream=lambda work: None, wall=lambda: NOW)


def test_home_shows_the_machine_memory_what_is_loaded_and_the_running_job() -> None:
    c = controller()
    c.refresh_now("home")
    view = c.view("home")
    facts = {fact.label: fact.value for fact in view.facts}
    assert view.headline == "Crucible is running" and view.action is None
    assert facts["Version"] == "1.0.55" and facts["Machine"] == "RTX 3090 Ti"
    assert facts["Memory"] == "24.0 GB card memory, 3.0 GB kept for the desktop"
    assert facts["Loaded"].startswith("qwen3.5-9b (llm), about 20.0 GB, held by 'bookforge'")
    running, queued = view.work
    assert running.fraction == pytest.approx(0.66) and running.cancel == "j1" and running.target == "job"
    assert running.title == "asr with whisper for bookforge"
    assert queued.detail == "waiting, number 1 in line"


def test_unified_memory_is_named_as_such_on_a_mac() -> None:
    info = {"host": {"gpu": {"vendor": "apple", "name": "Apple M1 Ultra", "vram_bytes": 64 * 2**30}}}
    facts = screens.memory_facts(info, {"total_bytes": 64 * 2**30, "desktop_allowance_bytes": 0})
    assert facts[-1].value == "64.0 GB unified memory"


@pytest.mark.parametrize("state,action,label", [
    ("stopped", "start", "Start Crucible"),
    ("unreachable", "start", "Start Crucible"),
    ("broken", "repair", "Repair Crucible"),
    ("unauthorized", "restart", "Restart Crucible"),
])
def test_a_server_that_is_down_offers_the_one_action_that_fixes_it(state: str, action: str, label: str) -> None:
    c = controller(host=FakeHost({"state": state, "detail": "Engine did not answer"}))
    c.refresh_now("models")
    view = c.view("home")
    assert (view.action, view.action_label) == (action, label)
    assert view.detail == f"Engine did not answer. Click {label} to fix it."
    assert isinstance(c.view("models"), screens.HomeView)
    assert c.docs == {}


def test_a_busy_server_that_misses_a_status_read_is_slow_not_gone() -> None:
    """The PC took 2-3.5 s to answer the 3 s status ping during a long render, and every
    miss flipped the window to "Crucible is not running" and back (2026-10-02)."""
    now = [0.0]
    host = FakeHost()
    c = Controller(FakeApi(), host, lambda q: True, run=lambda w: w(), clock=lambda: now[0],
                   stream=lambda w: None, wall=lambda: NOW)
    c.refresh_now("home")
    host._status = {"state": "unreachable", "detail": "Engine did not answer: timed out",
                    "timed_out": True}
    now[0] = 10.0
    c.refresh_now("home")
    assert c.view("home").headline == "Crucible is running"
    assert "slow to answer" in c.notices["status"]
    now[0] = 12.0
    host._status = RUNNING
    c.refresh_now("home")
    assert "status" not in c.notices
    host._status = {"state": "unreachable", "detail": "timed out", "timed_out": True}
    now[0] = 20.0
    c.refresh_now("home")
    now[0] = 20.0 + 31.0
    c.refresh_now("home")
    assert c.view("home").headline == "Crucible is not running", "the budget is a budget"


def test_a_refused_connection_is_shown_down_at_once() -> None:
    host = FakeHost()
    c = controller(host=host)
    c.refresh_now("home")
    host._status = {"state": "unreachable", "detail": "Engine did not answer: refused",
                    "timed_out": False}
    c.refresh_now("home")
    assert c.view("home").headline == "Crucible is not running"


def test_no_installation_names_the_installer_to_run(monkeypatch) -> None:
    view = screens.not_installed_view(RuntimeError("crucible_not_installed: nothing at C:\\x"), "win32")
    assert "PowerShell" in view.detail and "install.ps1 | iex" in view.detail
    mac = screens.not_installed_view(RuntimeError("nothing"), "darwin")
    assert "Terminal" in mac.detail and "install.sh | sh" in mac.detail


def test_start_runs_the_host_verb_and_forgets_the_old_connection() -> None:
    api, host = FakeApi(), FakeHost({"state": "stopped", "detail": ""})
    c = controller(api, host)
    c.server("start")
    assert host.acted == ["start"] and api.forgotten == 1 and c.running()


def test_models_list_installed_first_and_never_offer_to_remove_what_is_loaded() -> None:
    c = controller()
    c.refresh_now("models")
    rows = c.view("models")
    assert [row.id for row in rows] == ["whisper-tiny", "qwen3.5-9b", "qwen3.5-4b"]
    tiny, loaded, available = rows
    assert loaded.note == "Loaded now" and not loaded.can_remove
    assert tiny.can_remove and tiny.size == "78 MB"
    assert available.can_pull and available.size == "8.0 GB download" and available.subtitle == "llm"


def test_while_a_download_runs_no_other_download_or_removal_is_offered() -> None:
    tasks = {"tasks": [{"task_id": "t1", "type": "pull", "state": "running",
                        "request": {"kind": "model", "id": "qwen3.5-4b"}}]}
    rows = screens.model_rows(CATALOG, tasks)
    pulling = next(row for row in rows if row.id == "qwen3.5-4b")
    assert pulling.note == "Downloading" and not pulling.can_pull
    assert not any(row.can_remove for row in rows)


def test_voices_join_the_catalog_and_offer_to_undo_a_local_override() -> None:
    c = controller()
    c.refresh_now("voices")
    sigma, mistborn = c.view("voices")
    assert sigma.installed and sigma.extra == ("reset",)
    assert sigma.subtitle == "higgs-v3, en, pinned to 8f6b3ca2, set on this machine"
    assert not mistborn.installed and mistborn.can_pull and mistborn.extra == ()


def test_voices_refused_by_the_server_show_its_words() -> None:
    refusal = ApiError("job_type_not_installed", "tts is not installed here. Install it under Packages", 409)
    c = controller(FakeApi({"/v1/voices": refusal}))
    c.refresh_now("voices")
    assert c.screen_errors("voices") == ["tts is not installed here. Install it under Packages (job_type_not_installed)"]


def test_packages_say_installed_installable_or_which_package_brings_them() -> None:
    c = controller()
    c.refresh_now("packages")
    rows = {row.job_type: row for row in c.view("packages")}
    assert "echo" not in rows
    assert rows["llm"].installed and rows["llm"].verdict == "Can clean up text, using qwen3.5-9b"
    assert rows["tts"].installable and rows["tts"].engines == ("higgs-v3",)
    assert rows["tts"].verdict == "No voice fits in 21.0 GiB" and rows["tts"].tone == screens.WARN
    assert not rows["denoise"].installable and rows["denoise"].note == "Comes with Voice conversion"


def test_settings_mask_secrets_but_copy_the_whole_pairing_line() -> None:
    c = controller()
    c.refresh_now("settings")
    view = c.view("settings")
    local, remote = view.pairing
    assert local.shown == "crucible://crucible%40test@127.0.0.1:7100/#••••••••"
    assert local.line.endswith("#secret-token") and local.where == "apps on this computer"
    assert remote.where == "pc.example"
    anthropic = view.upstreams[0]
    assert anthropic.shown == "•••• abcd" and anthropic.field == "key"
    assert view.upstreams[2].field == "url"
    assert view.allowance_gib == "3" and not view.lan_on


def test_lan_sharing_is_switched_through_the_host_and_reread() -> None:
    host = FakeHost()
    c = controller(host=host)
    c.set_lan(True)
    c.refresh_now("settings")
    view = c.view("settings")
    assert host.lan_calls == [True] and view.lan_on
    assert view.lan_words == "Other computers on this network can pair with it: 192.168.1.5:7100."


def test_a_mac_says_how_it_is_shared_without_offering_a_switch() -> None:
    c = controller(host=FakeHost(lan_supported=False))
    c.refresh_now("settings")
    view = c.view("settings")
    assert not view.lan_supported and view.lan_on
    assert "http://pc.example:7100" in view.lan_words


def test_an_unshared_windows_pc_says_so_on_home_and_settings_says_what_share_changes() -> None:
    from crucible.platform import lan_door

    c = controller(host=FakeHost())
    c.refresh_now("home")
    facts = {fact.label: fact.value for fact in c.view("home").facts}
    assert facts["Network"].startswith("only this computer")
    assert "Share on your network" in facts["Network"], "home names where the switch is"
    c.refresh_now("settings")
    view = c.view("settings")
    assert view.lan_supported and not view.lan_on
    assert view.lan_words.startswith("Only this computer can use Crucible.")
    for said in ("administrator", "port forward", lan_door.RULE_NAME, "Public",
                 "crucible lan disable"):
        assert said in view.lan_words, f"the Share row does not say {said!r}"


def test_a_shared_pc_that_windows_still_blocks_is_not_called_shared() -> None:
    host = FakeHost()
    host.lan = {"authorities": [], "state": "degraded"}
    c = controller(host=host)
    c.refresh_now("home")
    facts = {fact.label: fact.value for fact in c.view("home").facts}
    assert facts["Network"].startswith("shared, but Windows keeps other devices out")
    c.refresh_now("settings")
    view = c.view("settings")
    assert view.lan_on, "the rows exist, so the switch is on and offers Stop sharing"
    assert "crucible lan status" in view.lan_words


def test_a_native_windows_engine_gets_no_share_switch_and_says_what_opens_it() -> None:
    setup = {**SETUP, "backend": "llama-windows", "network": {
        "reachable": False, "urls": [], "sentence": "Only this PC can reach Crucible.",
        "how": 'set host = "0.0.0.0" under [server]', "command": None, "changes": "x"}}
    c = controller(api=FakeApi({"/v1/setup": setup}), host=FakeHost())
    c.refresh_now("settings")
    view = c.view("settings")
    assert not view.lan_supported, "Share would be refused lan_native_engine; it is not offered"
    assert view.lan_words == 'Only this PC can reach Crucible. set host = "0.0.0.0" under [server]'


def test_a_server_that_predates_the_network_report_says_so_rather_than_guessing() -> None:
    setup = {key: value for key, value in SETUP.items() if key != "network"}
    c = controller(api=FakeApi({"/v1/setup": setup}), host=FakeHost(lan_supported=False))
    c.refresh_now("settings")
    view = c.view("settings")
    assert view.lan_words == screens.PREDATES_NETWORK and not view.lan_on


def test_a_pull_asks_with_the_servers_plan_then_follows_its_progress() -> None:
    api = FakeApi({"/v1/capability/plan?subject=qwen3.5-4b": {"confirm": "Download qwen3.5-4b.\n\nDownload it?"},
                   "/v1/tasks/t9": {"state": "done"}})
    api.answers[("POST", "/v1/tasks")] = {"task_id": "t9"}
    api.frames["/v1/tasks/t9/events"] = [
        {"event": "started", "data": {"type": "pull"}},
        {"event": "step", "data": {"name": "pull model qwen3.5-4b", "index": 1, "total": 1}},
        {"event": "progress", "data": {"bytes_done": 2 * 2**30, "bytes_total": 8 * 2**30, "file": "a"}},
    ]
    asked: list[str] = []
    c = controller(api, asked=asked)
    seen: list[str] = []
    original = TaskWatch.apply

    def spy(self: TaskWatch, event: str, data: dict) -> None:
        original(self, event, data)
        seen.append(self.view().detail)
    TaskWatch.apply = spy
    try:
        c.pull("model", "qwen3.5-4b")
    finally:
        TaskWatch.apply = original
    assert asked == ["Download qwen3.5-4b.\n\nDownload it?"]
    assert api.sent == [("POST", "/v1/tasks", {"type": "pull", "kind": "model", "id": "qwen3.5-4b"})]
    assert seen[-1] == "2.0 GB of 8.0 GB"
    final = c.watch_view()
    assert final.title == "Downloading qwen3.5-4b" and final.detail == "Finished" and final.fraction == 1.0
    assert final.cancel is None and "model:qwen3.5-4b" not in c.busy


def test_saying_no_to_the_plan_sends_nothing() -> None:
    api = FakeApi({"/v1/capability/plan?subject=qwen3.5-4b": {"confirm": "Download it?"}})
    c = controller(api, answers=[False])
    c.pull("model", "qwen3.5-4b")
    assert api.sent == [] and c.watch is None


def test_a_task_that_failed_on_the_host_says_so_and_where_to_look() -> None:
    watch = TaskWatch.of({"task_id": "t1", "type": "install", "request": {"job_type": "tts"}})
    watch.settle({"state": "failed", "error": None})
    assert watch.view().detail.startswith("It failed on the server without a reason.")
    watch = TaskWatch.of({"task_id": "t2", "type": "install", "request": {"job_type": "tts"}})
    watch.apply("failed", {"code": "reload_refused", "message": "run the install again"})
    assert watch.view().detail == "run the install again (reload_refused)"


def test_a_refused_removal_shows_the_servers_message_under_that_row() -> None:
    api = FakeApi()
    api.answers[("DELETE", "/v1/catalog/model/whisper-tiny")] = ApiError(
        "subject_in_use", "whisper-tiny is named by a running job; wait for it or cancel it under Activity", 409)
    c = controller(api)
    c.remove("model", "whisper-tiny", "78 MB")
    assert c.notices["model:whisper-tiny"] == (
        "whisper-tiny is named by a running job; wait for it or cancel it under Activity (subject_in_use)")


def test_an_install_names_its_engine_and_is_watched() -> None:
    api = FakeApi({"/v1/tasks/t3": {"state": "done"}})
    api.answers[("POST", "/v1/tasks")] = {"task_id": "t3"}
    c = controller(api)
    c.install("tts", "higgs-v3")
    assert api.sent[0] == ("POST", "/v1/tasks", {"type": "install", "job_type": "tts", "narrator_engine": "higgs-v3"})
    assert c.watch_view().title == "Installing tts (higgs-v3)"


def test_a_running_task_found_on_refresh_is_followed_once() -> None:
    tasks = {"tasks": [{"task_id": "t5", "type": "install", "state": "running", "request": {"job_type": "asr"}}]}
    api = FakeApi({"/v1/tasks": tasks, "/v1/tasks/t5": {"state": "running"}})
    api.frames["/v1/tasks/t5/events"] = [{"event": "progress", "data": {"line": "Collecting torch"}}]
    c = controller(api)
    c.refresh_now("activity")
    c.refresh_now("activity")
    view = c.view("activity")
    assert view.work[0].title == "Installing asr" and view.work[0].detail == "Collecting torch"
    assert view.work[0].target == "task" and view.work[0].cancel == "t5"
    assert view.session is not None and view.session.session_id == "ses-1"
    assert view.session.title == "bookforge has Crucible to itself for decide"
    assert view.session.detail.startswith("3 requests so far; 1 in progress")


def test_the_open_queue_session_is_shown_and_an_operator_can_end_it() -> None:
    asked: list[str] = []
    api = FakeApi()
    c = controller(api, answers=[False, True], asked=asked)
    c.refresh_now("activity")
    assert c.view("activity").session.session_id == "ses-1"
    c.end_session("ses-1")
    assert api.sent == []
    c.end_session("ses-1")
    assert api.sent == [("DELETE", "/v1/queue/ses-1", None)]
    assert len(asked) == 2 and "session" in asked[0]


def test_the_queue_is_listed_apart_from_the_work_and_each_job_can_be_removed() -> None:
    asked: list[str] = []
    api = FakeApi()
    c = controller(api, answers=[False, True], asked=asked)
    c.refresh_now("activity")
    view = c.view("activity")
    assert [work.cancel for work in view.work] == ["j1", "j2"]
    job, call, session = view.queue
    assert (job.job_id, job.kind, job.title, job.detail, job.waited) == (
        "j3", "job", "1. tts with sigma", "From briefcase", "waited 2 min")
    assert (call.kind, call.title, call.detail, call.waited) == (
        "call", "2. chat with qwen3.5-9b",
        "From foundry, in the open session (it runs first)", "waited 1 min")
    assert (session.job_id, session.kind, session.title, session.waited) == (
        "ses-2", "session", "3. Session", "waited 5 s")
    c.refresh_now("home")
    home = {fact.label: fact.value for fact in c.view("home").facts}
    assert home["Queue"] == "1 job, 1 chat, 1 session waiting; see Activity"
    c.remove_queued("j3")
    assert api.sent == []
    c.remove_queued("j3")
    assert api.sent == [("DELETE", "/v1/queue/j3", None)]
    assert len(asked) == 2 and "queue" in asked[0]


def test_waited_reads_like_a_person_says_it() -> None:
    assert [screens.waited_text(s) for s in (4.2, 59, 60, 3000, 3599, 5400, None)] == [
        "4 s", "59 s", "1 min", "50 min", "1 h 0 min", "1 h 30 min", ""]


def test_cancelling_a_job_asks_first_and_a_task_does_not() -> None:
    api = FakeApi()
    asked: list[str] = []
    c = controller(api, answers=[True], asked=asked)
    c.cancel("job", "j1")
    c.cancel("task", "t1")
    assert api.sent == [("DELETE", "/v1/jobs/j1", None), ("DELETE", "/v1/tasks/t1", None)]
    assert len(asked) == 1


def test_keys_and_the_allowance_are_written_through_settings() -> None:
    api = FakeApi()
    api.answers[("PUT", "/v1/settings")] = SETTINGS
    c = controller(api)
    c.save_upstream("openai", "key", " sk-new ")
    c.save_upstream("anthropic", "key", "")
    c.save_allowance("4")
    c.save_allowance("four")
    assert api.sent == [
        ("PUT", "/v1/settings", {"upstreams": {"openai": {"key": "sk-new"}}}),
        ("PUT", "/v1/settings", {"upstreams": {"anthropic": None}}),
        ("PUT", "/v1/settings", {"desktop_allowance_bytes": 4 * 2**30}),
    ]
    assert c.notices["allowance"] == "'four' is not a number of GB; type one, like 3 (allowance_not_a_number)"


def test_one_action_at_a_time_per_row() -> None:
    held: list = []
    c = Controller(FakeApi(), FakeHost(), lambda q: True, run=held.append)
    c.remove("model", "whisper-tiny", "")
    c.remove("model", "whisper-tiny", "")
    assert len(held) == 1 and "model:whisper-tiny" in c.busy


def test_slow_documents_are_not_refetched_every_tick() -> None:
    now = [0.0]
    api = FakeApi()
    reads: list[str] = []
    real_get = api.get
    api.get = lambda path: reads.append(path) or real_get(path)
    c = Controller(api, FakeHost(), lambda q: True, run=lambda w: w(), clock=lambda: now[0],
                   stream=lambda w: None)
    c.refresh_now("home")
    c.refresh_now("home")
    assert reads.count("/v1/info") == 1 and reads.count("/v1/activity") == 1
    now[0] = 16.0
    c.refresh_now("home")
    assert reads.count("/v1/info") == 2
    assert reads.count("/v1/activity") == 1, "what the event stream covers is never polled"


def test_the_event_stream_replaces_polling_and_rereads_only_what_an_event_names() -> None:
    api = FakeApi()
    reads: list[str] = []
    real_get = api.get
    api.get = lambda path: reads.append(path) or real_get(path)
    started: list = []
    c = Controller(api, FakeHost(), lambda q: True, run=lambda w: w(), stream=started.append,
                   wall=lambda: NOW)
    c.refresh_now("activity")
    c.refresh_now("activity")
    assert len(started) == 1, "one stream for the window's life"
    snapshot_activity = {**ACTIVITY, "running": [{**ACTIVITY["running"][0], "progress": 0.1}]}
    api.frames["/v1/events"] = [
        {"id": 7, "event": "snapshot", "data": {"gap": False, "topics": [],
                                               "activity": snapshot_activity, "queue": QUEUE,
                                               "tasks": []}},
        {"id": 8, "event": "job.progress", "data": {"job_id": "j1", "fraction": 0.5,
                                                   "message": "halfway"}},
    ]
    reads.clear()
    assert c.follow_events() is True
    assert reads == [], "a snapshot and a progress tick need no read"
    assert c.view("activity").work[0].fraction == 0.5
    assert c.last_event_id == 8
    api.frames["/v1/events"] = [{"id": 9, "event": "queue.added", "data": {"kind": "session"}},
                                {"id": 10, "event": "settings.written", "data": {}}]
    c.follow_events()
    assert api.followed[-1] == ("/v1/events", 8), "a reconnect resumes after the last id"
    assert sorted(set(reads)) == ["/v1/activity", "/v1/queue"]
    reads.clear()
    c.refresh_now("settings")
    assert reads.count("/v1/settings") == 1, "settings.written is read when settings shows"
    c.refresh_now("settings")
    assert reads.count("/v1/settings") == 1


def test_a_gap_snapshot_rereads_what_no_event_covers() -> None:
    api = FakeApi()
    reads: list[str] = []
    real_get = api.get
    api.get = lambda path: reads.append(path) or real_get(path)
    c = Controller(api, FakeHost(), lambda q: True, run=lambda w: w(), stream=lambda w: None)
    c.refresh_now("home")
    api.frames["/v1/events"] = [{"id": 3, "event": "snapshot", "data": {
        "gap": True, "topics": [], "activity": ACTIVITY, "queue": QUEUE, "tasks": []}}]
    reads.clear()
    c.follow_events()
    c.refresh_now("home")
    assert "/v1/info" in reads and "/v1/activity" not in reads


def test_the_open_session_counts_down_to_its_idle_close() -> None:
    activity = {**ACTIVITY, "session": {**ACTIVITY["session"], "in_flight": [],
                                        "idle_deadline": "2026-10-01T12:04:05+00:00"}}
    c = controller(FakeApi({"/v1/activity": activity}))
    c.refresh_now("activity")
    assert "it closes in 2 min unless bookforge sends more" in c.view("activity").session.detail


class Envelope(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"error": {"code": "unknown_subject", "message": "no model called 'x'. "
                                     "`crucible models list` names the ones there are"}}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def test_the_real_client_carries_the_servers_message_and_next_step_verbatim() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Envelope)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        api = LocalApi(lambda: Connection(url=url, token="t", name="crucible@test", source="local"))
        with pytest.raises(ApiError) as caught:
            api.get("/v1/catalog")
    finally:
        server.shutdown()
    assert caught.value.code == "unknown_subject" and caught.value.status == 404
    assert caught.value.message == "no model called 'x'. `crucible models list` names the ones there are"


def test_an_engine_that_does_not_answer_names_the_command_that_says_why() -> None:
    api = LocalApi(lambda: Connection(url="http://127.0.0.1:9", token="t", name="crucible@test", source="local"))
    with pytest.raises(ApiError) as caught:
        api.get("/v1/info")
    assert caught.value.code == "server_unreachable"
    assert "`crucible local status`" in caught.value.message


def test_no_local_engine_is_an_api_error_with_its_code() -> None:
    from crucible.client.errors import ClientRefusal

    def refuse() -> Connection:
        raise ClientRefusal("no_local_engine: nothing here. Pass --url")
    with pytest.raises(ApiError) as caught:
        LocalApi(refuse).get("/v1/info")
    assert caught.value.code == "no_local_engine" and caught.value.message == "nothing here. Pass --url"


def test_one_window_per_home_and_a_second_launch_brings_it_forward(tmp_path: Path) -> None:
    first = instance.Instance(tmp_path)
    assert first.claim()
    heard: list[str] = []
    first.serve(heard.append)
    try:
        assert not instance.Instance(tmp_path).claim()
        assert instance.still_open(tmp_path)
        assert instance.signal(tmp_path, instance.FOCUS)
        deadline = time.monotonic() + 5
        while not heard and time.monotonic() < deadline:
            time.sleep(0.02)
        assert heard == ["focus"]
    finally:
        first.close()
    assert not instance.door_path(tmp_path).exists() and not instance.still_open(tmp_path)


def test_a_stranger_without_the_token_is_ignored(tmp_path: Path) -> None:
    import socket

    first = instance.Instance(tmp_path)
    first.claim()
    heard: list[str] = []
    port = first.serve(heard.append)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as link:
            link.sendall(b"wrong quit\n")
            assert link.recv(8) == b""
    finally:
        first.close()
    assert heard == []


def test_close_running_asks_the_window_to_quit_and_waits(tmp_path: Path) -> None:
    first = instance.Instance(tmp_path)
    first.claim()
    first.serve(lambda word: first.close() if word == instance.QUIT else None)
    assert instance.close_running(tmp_path, timeout=5)
    assert instance.close_running(tmp_path, timeout=5)


def test_a_door_file_that_names_nothing_is_not_a_window(tmp_path: Path) -> None:
    instance.door_path(tmp_path).write_text("garbage", encoding="ascii")
    assert not instance.signal(tmp_path, instance.FOCUS)


def test_the_theme_follows_the_override_then_the_os() -> None:
    assert theme.current("linux", {"CRUCIBLE_APP_THEME": "dark"}) is theme.DARK
    assert theme.current("linux", {"CRUCIBLE_APP_THEME": "light"}) is theme.LIGHT
    assert theme.current("linux", {}) is theme.LIGHT
    assert theme.tone_colour(theme.DARK, "bad") == theme.DARK.bad
    assert theme.tone_colour(theme.DARK, "idle") == theme.DARK.muted



def test_every_screen_draws_in_both_palettes_without_a_server(monkeypatch) -> None:
    tkinter = pytest.importorskip("tkinter")
    from crucible.desktop_app import views, window

    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display for Tk here: {exc}")
    root.withdraw()
    try:
        api = FakeApi()
        api.answers[("DELETE", "/v1/catalog/model/whisper-tiny")] = ApiError("subject_in_use", "busy; wait", 409)
        c = controller(api)
        c.remove("model", "whisper-tiny", "")
        for palette in ("light", "dark"):
            monkeypatch.setenv(theme.THEME_ENV, palette)
            app = window.App(root, c, window.Asker())
            assert app.palette.name == palette
            for screen, _title in views.SCREENS:
                c.refresh_now(screen)
                app.go(screen)
                root.update_idletasks()
                assert app.scroll.inner.winfo_children(), screen
        c.status = {"state": "stopped", "detail": ""}
        app.go("models")
        app.go("home")
    finally:
        root.destroy()


class FakeRoot:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.protocols: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self.shown = "normal"

    def protocol(self, name: str, handler: Any) -> None:
        self.protocols[name] = handler

    def createcommand(self, name: str, handler: Any) -> None:
        self.commands[name] = handler

    def state(self) -> str:
        return self.shown

    def after(self, _ms: int, _then: Any) -> None:
        self.calls.append("after")

    def attributes(self, *_args: Any) -> None:
        self.calls.append("attributes")

    def __getattr__(self, name: str) -> Any:
        return lambda *_a: self.calls.append(name)


def bare_app(monkeypatch, platform: str, home: Path | None) -> Any:
    pytest.importorskip("tkinter")
    from crucible.desktop_app import window

    monkeypatch.setattr(window.sys, "platform", platform)
    app = window.App.__new__(window.App)
    app.root, app.home = FakeRoot(), home
    app.bind_window_manager()
    return app


def test_the_close_button_hides_the_window_while_the_tray_can_bring_it_back(monkeypatch, tmp_path: Path) -> None:
    import os

    (tmp_path / "tray.pid").write_text(str(os.getpid()))
    app = bare_app(monkeypatch, "win32", tmp_path)
    app.root.protocols["WM_DELETE_WINDOW"]()
    assert app.root.calls == ["withdraw"]


def test_the_close_button_ends_a_window_no_tray_could_reopen(monkeypatch, tmp_path: Path) -> None:
    app = bare_app(monkeypatch, "win32", tmp_path)
    app.root.protocols["WM_DELETE_WINDOW"]()
    assert app.root.calls == ["destroy"]


def test_on_the_mac_the_close_button_always_hides_and_the_dock_reopens(monkeypatch, tmp_path: Path) -> None:
    from crucible.desktop_app import window

    app = bare_app(monkeypatch, "darwin", tmp_path)
    app.root.protocols["WM_DELETE_WINDOW"]()
    assert app.root.calls == ["withdraw"]
    app.root.commands[window.MAC_REOPEN]()
    assert app.root.calls[1:3] == ["deiconify", "lift"] and app.root.calls[-1] == "focus_force"


def test_cmd_q_on_the_mac_closes_the_tray_as_well_as_the_window(monkeypatch, tmp_path: Path) -> None:
    import os

    from crucible import traylife
    from crucible.desktop_app import window

    (tmp_path / "tray.pid").write_text(str(os.getpid()))
    app = bare_app(monkeypatch, "darwin", tmp_path)
    app.root.commands[window.MAC_QUIT]()
    assert traylife.close_request_path(tmp_path).read_text() == "close\n"
    assert app.root.calls == ["destroy"]


def test_windows_registers_no_mac_handlers(monkeypatch, tmp_path: Path) -> None:
    app = bare_app(monkeypatch, "win32", tmp_path)
    assert app.root.commands == {}


def test_the_show_word_brings_a_hidden_window_back_and_quit_ends_it(monkeypatch, tmp_path: Path) -> None:
    app = bare_app(monkeypatch, "win32", tmp_path)
    app.hear(instance.FOCUS)
    assert app.root.calls[:2] == ["deiconify", "lift"] and app.root.calls[-1] == "focus_force"
    app.root.calls.clear()
    app.hear(instance.QUIT)
    assert app.root.calls == ["destroy"]
    assert not traylife_close_requested(tmp_path)


def traylife_close_requested(home: Path) -> bool:
    from crucible import traylife

    return traylife.close_request_path(home).exists()


def test_a_real_tk_window_is_withdrawn_by_its_close_button_and_shown_again(monkeypatch, tmp_path: Path) -> None:
    import os

    tkinter = pytest.importorskip("tkinter")
    from crucible.desktop_app import window

    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display for Tk here: {exc}")
    (tmp_path / "tray.pid").write_text(str(os.getpid()))
    monkeypatch.setattr(window.sys, "platform", "win32")
    app = window.App.__new__(window.App)
    app.root, app.home = root, tmp_path
    try:
        app.bind_window_manager()
        root.update()
        root.eval(root.protocol("WM_DELETE_WINDOW"))
        assert root.state() == "withdrawn" and root.winfo_exists()
        app.hear(instance.FOCUS)
        root.update()
        assert root.state() == "normal"
    finally:
        root.destroy()
