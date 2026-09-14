"""Operator tasks: `POST /v1/tasks` and the four routes around it.

PHASE13-OPERATOR.md section 3.3 and 3.4. Nothing here touches the network, a
GPU or a real env: the hub is `tests/fake_hub.py` (which drives the real
`weights.pull`, stamp and all) and the installer is a fake console script that
writes a config flag, which is exactly what the real one does at the end of a
five-minute pip run.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import huggingface_hub
import pytest
from fastapi.testclient import TestClient

from crucible import jobenv, tasks, weights
from crucible.tasks import ReloadRefused

from .conftest import FAKE_BACKEND, holding_the_card, parse_sse
from .fake_hub import CHUNK, FakeHub

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
MODEL = "qwen3.5-9b"


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """Every `huggingface_hub` entry point `crucible/weights.py` calls."""
    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    return fake


@pytest.fixture
def fake_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A `crucible` console script that flips a flag and prints as it goes.

    A real script on disk, run as a real subprocess, because the thing under
    test is the streaming of its lines and the SIGTERM that cancels it — both
    of which a function call would prove nothing about. What it does NOT do is
    build a venv: the env is `crucible/jobenv.py`'s and has its own tests.
    """
    script = tmp_path / "fake-crucible"
    script.write_text(
        f"""#!{sys.executable}
import os, sys, time
from pathlib import Path
sys.path.insert(0, {REPO_ROOT!r})
from crucible.config import CAPABILITY_FLAGS, load_config, write_config

argv = sys.argv[1:]
assert argv[0] == "install", argv
job_type = argv[1]
print("fake installer: recipe " + job_type + ".txt", flush=True)

block = os.environ.get("FAKE_INSTALL_BLOCK")
if block:
    Path(block + ".started").write_text("1", encoding="utf-8")
    while Path(block).exists():
        time.sleep(0.01)

print("fake installer: collecting torch", flush=True)
if os.environ.get("FAKE_INSTALL_FAIL") == "1":
    print("fake installer: ERROR could not build a wheel", flush=True)
    raise SystemExit(3)

config = load_config()
values = {{flag: getattr(config, flag) for flag in CAPABILITY_FLAGS}}
values["enable_" + job_type] = True
write_config(
    config.home,
    name=config.name,
    host=config.host,
    port=config.port,
    token=config.token,
    backend_kind=config.backend_kind,
    desktop_allowance_bytes=config.desktop_allowance_bytes,
    **values,
)
print("fake installer: recorded in " + str(config.path), flush=True)
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    # The conda env this suite runs in HAS a real `crucible` console script
    # beside its interpreter, so `install_command()` would find it and a test
    # would build a multi-gigabyte venv. Named rather than searched for.
    monkeypatch.setattr(tasks, "install_command", lambda: str(script))
    return script


def post(client: TestClient, auth: dict[str, str], body: dict[str, Any]) -> Any:
    return client.post("/v1/tasks", headers=auth, json=body)


def admit(client: TestClient, auth: dict[str, str], body: dict[str, Any]) -> str:
    response = post(client, auth, body)
    assert response.status_code == 202, response.text
    return str(response.json()["task_id"])


def watch(client: TestClient, auth: dict[str, str], task_id: str) -> list[dict]:
    """Read the task's whole event stream, which ends at its terminal event."""
    with client.stream(
        "GET", f"/v1/tasks/{task_id}/events", headers=auth
    ) as stream:
        return parse_sse(line for line in stream.iter_lines())


def run(client: TestClient, auth: dict[str, str], body: dict[str, Any]) -> list[dict]:
    return watch(client, auth, admit(client, auth, body))


def kinds(events: list[dict]) -> list[str]:
    return [event["event"] for event in events]


def steps(events: list[dict]) -> list[str]:
    return [e["data"]["name"] for e in events if e["event"] == "step"]


# ---------------------------------------------------------------- validation


def test_a_pull_of_an_unknown_kind_is_refused_by_name(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(client, auth, {"type": "pull", "kind": "weights", "id": MODEL})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_subject"


def test_a_pull_of_an_unknown_id_names_the_catalog(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(client, auth, {"type": "pull", "kind": "model", "id": "nope"})
    assert response.status_code == 404
    body = response.json()["error"]
    assert body["code"] == "unknown_subject"
    assert "/v1/catalog" in body["message"]


def test_a_pull_of_an_installed_subject_is_refused_not_skipped(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    """3.3 writes the asymmetry down: a single pull refuses, a module skips."""
    fake_weights(MODEL)
    with make_client() as client:
        response = post(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_installed"


def test_an_install_of_a_type_with_no_installer_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(client, auth, {"type": "install", "job_type": "echo"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_job_type"


def test_tts_must_say_which_narrator_engine(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(client, auth, {"type": "install", "job_type": "tts"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "narrator_engine_required"


def test_an_unknown_narrator_engine_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(
        client, auth,
        {"type": "install", "job_type": "tts", "narrator_engine": "elevenlabs"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "narrator_engine_required"


def test_a_narrator_engine_on_a_non_tts_type_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(
        client, auth,
        {"type": "install", "job_type": "llm", "narrator_engine": "higgs-v3"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "narrator_engine_refused"


def test_an_install_of_a_built_env_is_refused_by_name(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "env_installed", lambda *a, **k: True)
    response = post(client, auth, {"type": "install", "job_type": "llm"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_type_installed"


def test_env_installed_reads_the_env_and_not_the_flag(
    home: Path, fake_env: Path, make_client: Callable[..., TestClient]
) -> None:
    """The fact is the directory, because `tts` has two envs behind one flag.

    `enable_tts` is FALSE here and `enable_llm` is false too, so a reading that
    consulted the flags would answer "neither". The llm env is on disk
    (`fake_env`) and the answer follows the disk.
    """
    from crucible.config import load_config

    with make_client(enable_llm=False, enable_tts=False):
        config = load_config(home)
    assert config.enable_llm is False
    assert tasks.env_installed(config, FAKE_BACKEND, "llm", None) is True
    assert tasks.env_installed(config, FAKE_BACKEND, "tts", "higgs-v3") is False


def test_adopt_refuses_a_config_from_another_home(
    home: Path, tmp_path: Path, make_client: Callable[..., TestClient]
) -> None:
    """Identity is refused, capability is adopted (`Config.adopt`)."""
    from crucible.config import ConfigError, load_config, write_config

    with make_client() as client:
        mine = client.app.state.config
        elsewhere = tmp_path / "another-home"
        write_config(
            elsewhere,
            name="crucible@somebody-else",
            host="127.0.0.1",
            port=7100,
            token="not-mine",
            backend_kind=FAKE_BACKEND.kind,
            enable_echo=True,
            enable_llm=True,
            enable_asr=False,
            enable_tts=False,
            enable_align=False,
            enable_rvc=False,
            desktop_allowance_bytes=0,
        )
        with pytest.raises(ConfigError, match="different server"):
            mine.adopt(load_config(elsewhere))
        assert mine.enable_llm is False


def test_nothing_but_adopt_writes_to_a_frozen_config() -> None:
    """`Config.adopt`'s docstring promises this; a grep is what keeps it true.

    `object.__setattr__` on a frozen dataclass is the one escape hatch in the
    package, and the whole argument for it is that it happens in exactly one
    named place. A second one would make `Config` mutable in practice while
    still claiming to be frozen, which is worse than not freezing it.
    """
    package = Path(__file__).resolve().parent.parent / "crucible"
    offenders = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "object.__setattr__" in path.read_text(encoding="utf-8")
    ]
    assert offenders == ["config.py"]


def test_a_module_is_validated_whole_and_names_every_problem(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(
        client,
        auth,
        {
            "type": "module",
            "module": {
                "name": "bookforge",
                "version": "1",
                "job_types": [{"type": "echo"}],
                "subjects": [
                    {"kind": "model", "id": "not-a-model"},
                    {"kind": "sorcery", "id": "x"},
                ],
            },
        },
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "invalid_module"
    problems = body["details"]["problems"]
    assert len(problems) == 3
    assert any("not-a-model" in p for p in problems)
    assert any("sorcery" in p for p in problems)


def test_a_module_with_no_version_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(
        client, auth,
        {"type": "module", "module": {"name": "x", "subjects": [], "job_types": []}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_module"


def test_a_request_carrying_another_type_s_fields_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = post(
        client, auth, {"type": "pull", "kind": "model", "id": MODEL, "job_type": "llm"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


# ----------------------------------------------------------------- the pull


def test_a_pull_reports_bytes_and_leaves_the_weights_stamped(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
    with make_client() as client:
        events = run(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
        row = next(
            r for r in client.get("/v1/catalog", headers=auth).json()["rows"]
            if r["id"] == MODEL and r["kind"] == "model"
        )
    assert kinds(events)[0] == "started"
    assert kinds(events)[-1] == "done"
    assert steps(events) == [f"pull model {MODEL}"]
    progress = [e["data"] for e in events if e["event"] == "progress"]
    assert progress, "a pull that reported no bytes reported nothing"
    assert progress[-1]["bytes_total"] == 4 * CHUNK
    assert progress[-1]["file"] == "model.safetensors"
    assert row["installed"] is True
    assert row["installed_bytes"] > 0


def test_a_pull_asks_the_hub_for_the_manifest_s_pin(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
    from crucible.manifests import load_manifest

    spec = load_manifest(MODEL).spec(FAKE_BACKEND.kind)
    with make_client() as client:
        run(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
    assert hub.asked == [(spec.hf_repo, spec.revision)]


def test_the_task_record_echoes_the_request_and_ends_done(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
    with make_client() as client:
        task_id = admit(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
        watch(client, auth, task_id)
        body = client.get(f"/v1/tasks/{task_id}", headers=auth).json()
        listed = client.get("/v1/tasks", headers=auth).json()["tasks"]
    assert body["state"] == "done"
    assert body["error"] is None
    assert body["request"] == {"type": "pull", "kind": "model", "id": MODEL}
    assert body["finished"] is not None
    assert [row["task_id"] for row in listed] == [task_id]


def test_an_unknown_task_is_a_named_404(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/v1/tasks/deadbeef", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_task"


def test_a_second_task_while_one_runs_is_refused_naming_the_first(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow = FakeHub(chunks=400, delay=0.01)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", slow.snapshot_download)
    with make_client() as client:
        first = admit(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
        assert slow.started.wait(timeout=10)
        response = post(
            client, auth, {"type": "pull", "kind": "voice", "id": "higgs-default"}
        )
        assert response.status_code == 409
        body = response.json()["error"]
        assert body["code"] == "task_busy"
        assert body["details"]["task_id"] == first
        client.delete(f"/v1/tasks/{first}", headers=auth)
        watch(client, auth, first)


def test_a_cancelled_pull_stops_and_leaves_nothing_installed(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6 does not apply: half a safetensors file is not partial work (3.3)."""
    slow = FakeHub(chunks=2000, delay=0.005)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", slow.snapshot_download)
    with make_client() as client:
        task_id = admit(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
        assert slow.started.wait(timeout=10)
        cancel = client.delete(f"/v1/tasks/{task_id}", headers=auth)
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelling"
        events = watch(client, auth, task_id)
        state = client.get(f"/v1/tasks/{task_id}", headers=auth).json()
        row = next(
            r for r in client.get("/v1/catalog", headers=auth).json()["rows"]
            if r["id"] == MODEL and r["kind"] == "model"
        )
    assert kinds(events)[-1] == "cancelled"
    assert state["state"] == "cancelled"
    assert row["installed"] is False
    assert not (home / "models" / MODEL / FAKE_BACKEND.kind).exists()


def test_cancelling_a_finished_task_is_refused_by_name(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
    with make_client() as client:
        task_id = admit(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
        watch(client, auth, task_id)
        response = client.delete(f"/v1/tasks/{task_id}", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_running"


def test_a_pull_runs_beside_a_job_because_it_is_disk_and_network(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
    """3.3: only an install is gated on the four facts."""
    with make_client() as client, holding_the_card(client):
        response = post(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
        assert response.status_code == 202
        watch(client, auth, response.json()["task_id"])


def test_a_pull_that_fails_says_so_with_the_hub_s_reason(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(**_kwargs: Any) -> None:
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)
    with make_client() as client:
        events = run(client, auth, {"type": "pull", "kind": "model", "id": MODEL})
    failed = events[-1]
    assert failed["event"] == "failed"
    assert failed["data"]["code"] == "pull_failed"
    assert "connection reset by peer" in failed["data"]["message"]


# -------------------------------------------------------------- the install


def test_an_install_streams_its_lines_reloads_and_makes_the_type_live(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    """3.4's requirement, end to end: live in `/v1/info` before `done`."""
    with make_client(enable_echo=True, enable_llm=False) as client:
        assert "load-model" not in client.get("/v1/info", headers=auth).json()["job_types"]
        events = run(client, auth, {"type": "install", "job_type": "llm"})
        info = client.get("/v1/info", headers=auth).json()
        setup = client.get("/v1/setup", headers=auth).json()

    assert kinds(events)[-1] == "done"
    assert steps(events) == ["install llm", "reload"]
    lines = [e["data"]["line"] for e in events if e["event"] == "progress"]
    assert any("collecting torch" in line for line in lines)
    reload_step = next(e for e in events if e["data"].get("name") == "reload")
    assert "load-model" in reload_step["data"]["job_types"]
    assert "load-model" in info["job_types"]
    assert setup["job_types"] == info["job_types"]


def test_the_reload_adopts_the_config_so_every_door_agrees(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    """`/v1/models` reads a flag, `POST /v1/jobs` reads the registry. Both move."""
    with make_client(enable_echo=True, enable_llm=False) as client:
        assert client.get("/v1/models", headers=auth).status_code == 400
        run(client, auth, {"type": "install", "job_type": "llm"})
        assert client.get("/v1/models", headers=auth).status_code == 200
        assert client.app.state.config.enable_llm is True


def test_an_install_that_fails_says_so_and_leaves_the_registry_alone(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_installer: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_INSTALL_FAIL", "1")
    with make_client(enable_echo=True, enable_llm=False) as client:
        events = run(client, auth, {"type": "install", "job_type": "llm"})
        info = client.get("/v1/info", headers=auth).json()
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["code"] == "install_failed"
    assert "exited 3" in events[-1]["data"]["message"]
    assert steps(events) == ["install llm"]
    assert "load-model" not in info["job_types"]


def test_an_install_is_refused_while_any_of_the_four_facts_holds_the_card(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    with make_client(enable_echo=True) as client:
        with holding_the_card(client):
            response = post(client, auth, {"type": "install", "job_type": "llm"})
        assert response.status_code == 409
        body = response.json()["error"]
        assert body["code"] == "server_busy"
        assert body["details"]["fact"] == "a chat"
        assert body["details"]["in_flight"] == 1


def test_a_lease_refusal_carries_the_lease_s_own_fields(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    """Amendment from Foundry's review: the app's row shows this verbatim."""
    with make_client(enable_echo=True) as client:
        client.app.state.leases.open(
            kind="llm",
            subject="qwen3.8-27b-4bit",
            act="translate",
            client="foundry/owens-pc",
            ttl_seconds=600,
        )
        response = post(client, auth, {"type": "install", "job_type": "llm"})
    assert response.status_code == 409
    details = response.json()["error"]["details"]
    assert details["fact"] == "a lease"
    assert details["client"] == "foundry/owens-pc"
    assert details["act"] == "translate"
    assert details["subject"] == "qwen3.8-27b-4bit"
    assert details["kind"] == "llm"
    assert "expires_at" in details and "lease_id" in details


def test_the_install_command_is_named_when_there_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(tasks, "which", lambda _name: None)
    monkeypatch.setattr(tasks, "searched_note", lambda: "(PATH searched: /bin)")
    with pytest.raises(Exception) as caught:
        tasks.install_command()
    assert getattr(caught.value, "code", None) == "install_command_missing"
    assert "/bin" in str(caught.value)


# ------------------------------------------------------- the reload's guard
#
# 3.4: the four facts gate the task at POST and are read AGAIN at the swap,
# because minutes of pip pass in between. Each of the four is proved here
# against the swap itself, which is the read the POST gate cannot cover.


def reload_of(client: TestClient) -> Callable[[], list[str]]:
    return client.app.state.tasks._reload  # the injected 3.4 swap


def test_the_swap_happens_when_nothing_holds_the_card(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_echo=True) as client:
        assert reload_of(client)() == ["echo"]


def test_the_swap_refuses_while_a_job_is_on_the_lane(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    import base64

    with make_client(enable_echo=True) as client:
        response = client.post(
            "/v1/jobs",
            headers=auth,
            json={
                "type": "echo",
                "params": {"delay_ms": 400},
                "inputs": {
                    "x.bin": {"inline_base64": base64.b64encode(b"x").decode("ascii")}
                },
            },
        )
        assert response.status_code == 202
        with pytest.raises(ReloadRefused) as caught:
            reload_of(client)()
        assert caught.value.held.fact == "a job"
        with client.stream(
            "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
        ) as stream:
            parse_sse(line for line in stream.iter_lines())


def test_the_swap_refuses_while_a_lease_is_open(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client(enable_echo=True) as client:
        client.app.state.leases.open(
            kind="llm", subject="m", act="clean", client=None, ttl_seconds=60
        )
        with pytest.raises(ReloadRefused) as caught:
            reload_of(client)()
        assert caught.value.held.fact == "a lease"


def test_the_swap_refuses_while_a_stream_claims_the_card(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client(enable_echo=True) as client:
        client.app.state.residency.claim("tts stream abc", may_mutate=False)
        with pytest.raises(ReloadRefused) as caught:
            reload_of(client)()
        assert caught.value.held.fact == "the claim"


def test_the_swap_refuses_while_a_chat_is_in_flight(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client(enable_echo=True) as client, holding_the_card(client):
        with pytest.raises(ReloadRefused) as caught:
            reload_of(client)()
        assert caught.value.held.fact == "a chat"


def test_a_holder_that_arrives_during_the_install_fails_the_task_by_name(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_installer: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window 3.4 is about: admitted at POST, held by the time pip is done.

    The env is left on disk (R6) and the task says so rather than swapping the
    registry underneath whatever arrived.
    """
    block = tmp_path / "hold-the-installer"
    block.write_text("1", encoding="utf-8")
    monkeypatch.setenv("FAKE_INSTALL_BLOCK", str(block))
    started = Path(str(block) + ".started")

    with make_client(enable_echo=True, enable_llm=False) as client:
        task_id = admit(client, auth, {"type": "install", "job_type": "llm"})
        deadline = time.monotonic() + 15
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists(), "the fake installer never started"
        with holding_the_card(client):
            block.unlink()
            events = watch(client, auth, task_id)
        info = client.get("/v1/info", headers=auth).json()

    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["code"] == "reload_refused"
    assert "a chat" in events[-1]["data"]["message"]
    assert "load-model" not in info["job_types"]


# ------------------------------------------------------------------ modules


def test_a_module_runs_its_entries_in_order_and_ends_done(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    hub: FakeHub,
    fake_installer: Path,
) -> None:
    with make_client(enable_echo=True, enable_llm=False) as client:
        events = run(
            client,
            auth,
            {
                "type": "module",
                "module": {
                    "name": "bookforge",
                    "version": "0.6.0+abc",
                    "job_types": [{"type": "llm"}],
                    "subjects": [{"kind": "model", "id": MODEL}],
                },
            },
        )
        info = client.get("/v1/info", headers=auth).json()
    assert kinds(events)[-1] == "done"
    assert steps(events) == ["install llm", f"pull model {MODEL}", "reload"]
    assert "load-model" in info["job_types"]


def test_a_module_skips_what_is_already_true(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    hub: FakeHub,
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module says what must be TRUE, so an entry already true is skipped."""
    fake_weights(MODEL)
    monkeypatch.setattr(tasks, "env_installed", lambda *a, **k: True)
    with make_client(enable_echo=True, enable_llm=True) as client:
        events = run(
            client,
            auth,
            {
                "type": "module",
                "module": {
                    "name": "bookforge",
                    "version": "1",
                    "job_types": [{"type": "llm"}],
                    "subjects": [{"kind": "model", "id": MODEL}],
                },
            },
        )
    assert kinds(events)[-1] == "done"
    skipped = [e["data"]["reason"] for e in events if e["event"] == "skipped"]
    assert len(skipped) == 3, skipped
    assert any("install llm" in reason for reason in skipped)
    assert any(MODEL in reason for reason in skipped)
    assert any("reload" in reason for reason in skipped)
    assert not [e for e in events if e["event"] == "progress"]


def test_a_module_stops_at_the_first_failure_and_keeps_what_finished(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    hub: FakeHub,
    fake_installer: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6: the pull that finished STAYS, and the failure names the step."""
    monkeypatch.setenv("FAKE_INSTALL_FAIL", "1")
    with make_client(enable_echo=True, enable_llm=False) as client:
        events = run(
            client,
            auth,
            {
                "type": "module",
                "module": {
                    "name": "bookforge",
                    "version": "1",
                    "job_types": [{"type": "llm"}],
                    "subjects": [{"kind": "model", "id": MODEL}],
                },
            },
        )
        row = next(
            r for r in client.get("/v1/catalog", headers=auth).json()["rows"]
            if r["id"] == MODEL and r["kind"] == "model"
        )
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["code"] == "install_failed"
    assert "step 1 of 3" in events[-1]["data"]["message"]
    # The install is entry 1, so the pull never ran and the module stopped.
    assert steps(events) == ["install llm"]
    assert row["installed"] is False


def test_a_module_of_pure_pulls_is_not_gated_on_the_card(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
    """It names no job type, so it swaps no registry and is a pull (3.3)."""
    with make_client(enable_echo=True) as client, holding_the_card(client):
        events = run(
            client,
            auth,
            {
                "type": "module",
                "module": {
                    "name": "bookforge",
                    "version": "1",
                    "job_types": [],
                    "subjects": [{"kind": "model", "id": MODEL}],
                },
            },
        )
    assert kinds(events)[-1] == "done"
    assert steps(events) == [f"pull model {MODEL}"]


# ------------------------------------------------------------------- access


def test_every_task_route_needs_the_token(client: TestClient) -> None:
    assert client.post("/v1/tasks", json={"type": "pull"}).status_code == 401
    assert client.get("/v1/tasks").status_code == 401
    assert client.get("/v1/tasks/x").status_code == 401
    assert client.delete("/v1/tasks/x").status_code == 401
    assert client.get("/v1/tasks/x/events").status_code == 401
