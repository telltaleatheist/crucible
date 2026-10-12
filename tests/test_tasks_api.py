from __future__ import annotations

import json
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable

import huggingface_hub
import pytest
from fastapi.testclient import TestClient

from crucible import installonsubmit, jobenv, tasks
from crucible.backend import Backend, Gpu
from crucible.capabilitystore import decide_on, record_of
from crucible.config import load_config
from crucible.memorybudget import GIB
from crucible.tasks import validate as validate_module

from .conftest import FAKE_BACKEND, holding_the_card, parse_sse
from .fake_hub import CHUNK, FakeHub

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
MODEL = "qwen3.5-9b"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    return fake


@pytest.fixture
def fake_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    script = tmp_path / "fake-crucible"
    script.write_text(
        f"""#!{sys.executable}
import os, sys, time
from pathlib import Path
sys.path.insert(0, {REPO_ROOT!r})
from crucible.config import load_config, rewrite_config

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

# AN INSTALL THAT DOWNLOADS AN INTERPRETER reports its bytes on the sentinel
# line `crucible/interpreter.py` owns. Three of them, so the test can see both
# that the first becomes a typed event and that the rest are throttled rather
# than leaking into the line stream.
from crucible.interpreter import progress_line
for done in (1000, 2000, 3000):
    print(progress_line(done, 3000, "cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"),
          flush=True)

if os.environ.get("FAKE_INSTALL_FAIL") == "1":
    print("fake installer: ERROR could not build a wheel", flush=True)
    raise SystemExit(3)

config = load_config()
# Only the named type's flag, and everything else (the capability record) as it was read.
rewrite_config(config, flags={{"enable_" + job_type: True}})
# What a test that reads envs through `built_envs` sees as built.
(config.home / "fake-built").mkdir(parents=True, exist_ok=True)
(config.home / "fake-built" / job_type).write_text("1", encoding="utf-8")
print("fake installer: recorded in " + str(config.path), flush=True)
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    if sys.platform == "win32":
        pytest.skip("the fake installer is a #! script, which Windows cannot execute")
    monkeypatch.setattr(tasks, "install_command", lambda: str(script))
    return script


def post(client: TestClient, auth: dict[str, str], body: dict[str, Any]) -> Any:
    return client.post("/v1/tasks", headers=auth, json=body)


def admit(client: TestClient, auth: dict[str, str], body: dict[str, Any]) -> str:
    response = post(client, auth, body)
    assert response.status_code == 202, response.text
    return str(response.json()["task_id"])


def watch(client: TestClient, auth: dict[str, str], task_id: str) -> list[dict]:
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


def no_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> str:
        raise AssertionError("an install of a built env must not run the installer")

    monkeypatch.setattr(tasks, "install_command", refuse)


def test_an_install_of_a_built_env_whose_type_is_off_turns_it_on(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    built_envs: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # owens-pc, 2.0.8: a video job was refused `job_type_disabled` (not_installed) with the
    # video env on disk, and the install that refusal named was refused
    # `job_type_installed`; only `crucible jobs enable video` by hand got out.
    no_installer(monkeypatch)
    built_envs("rvc")
    with make_client(
        enable_echo=True,
        enable_rvc=False,
        enable_denoise=False,
        capability=card_record(FAKE_BACKEND, 3 * GIB),
    ) as client:
        assert "rvc" not in client.get("/v1/info", headers=auth).json()["job_types"]
        events = run(client, auth, {"type": "install", "job_type": "rvc"})
        task = client.get("/v1/tasks", headers=auth).json()["tasks"][0]
        info = client.get("/v1/info", headers=auth).json()
    assert kinds(events)[-1] == "done", events
    assert steps(events) == ["install rvc", "enable rvc", "enable denoise", "reload"]
    skipped = [e["data"]["reason"] for e in events if e["event"] == "skipped"]
    assert len(skipped) == 1, skipped
    assert skipped[0].startswith("install rvc: this server already has that env")
    assert "--force" in skipped[0]
    lines = [e["data"].get("line") for e in events if e["event"] == "progress"]
    assert lines == ["[jobs] enable_rvc = true", "[jobs] enable_denoise = true"]
    reload_step = next(e for e in events if e["data"].get("name") == "reload")
    assert {"rvc", "denoise"} <= set(reload_step["data"]["job_types"])
    assert task["unmet"] == []
    config = load_config(home)
    assert (config.enable_rvc, config.enable_denoise) == (True, True)
    assert {"rvc", "denoise"} <= set(info["job_types"])


def test_an_install_of_a_built_env_whose_type_is_on_is_done_with_nothing_to_do(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    built_envs: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_installer(monkeypatch)
    built_envs("llm")
    with make_client(enable_echo=True, enable_llm=True) as client:
        response = post(client, auth, {"type": "install", "job_type": "llm"})
        assert response.status_code == 202, response.text
        events = watch(client, auth, response.json()["task_id"])
    assert kinds(events)[-1] == "done", events
    assert steps(events) == ["install llm", "enable llm", "reload"]
    skipped = [e["data"]["reason"] for e in events if e["event"] == "skipped"]
    assert len(skipped) == 3, skipped
    assert "already has that env" in skipped[0]
    assert "already on" in skipped[1]
    assert skipped[2].startswith("reload: nothing this task asks for changed")
    assert not [e for e in events if e["event"] == "progress"]


def test_an_install_of_a_built_env_the_enable_door_refuses_is_done_with_it_unmet(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    built_envs: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The enable door's own refusal, reported as a module reports it: a skip naming it
    # and a row in `unmet`. Nothing decided this card, so nothing knows it holds rvc.
    no_installer(monkeypatch)
    built_envs("rvc")
    with make_client(enable_echo=True, enable_rvc=False, enable_denoise=False) as client:
        events = run(client, auth, {"type": "install", "job_type": "rvc"})
        task = client.get("/v1/tasks", headers=auth).json()["tasks"][0]
    assert kinds(events)[-1] == "done", events
    skipped = [e["data"]["reason"] for e in events if e["event"] == "skipped"]
    assert any(reason.startswith("enable rvc: job_type_undecided: ") for reason in skipped)
    assert {row["entry"] for row in task["unmet"]} == {"enable rvc", "enable denoise"}
    assert load_config(home).enable_rvc is False


def test_a_submit_for_a_built_env_whose_type_is_off_turns_it_on(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    built_envs: Callable[..., None],
    pulled: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Install-on-submit's half of the same gap: a `denoise` job with the rvc env built
    # and `denoise` off was handed the refusal back. It now starts the module a build
    # would, whose install step skips the env and whose enable steps turn it on.
    no_installer(monkeypatch)
    built_envs("rvc")
    with make_client(
        enable_echo=True,
        enable_rvc=False,
        enable_denoise=False,
        capability=card_record(FAKE_BACKEND, 3 * GIB),
    ) as client:
        response = client.post(
            "/v1/jobs", headers=auth, json={"type": "denoise", "model": "vocals-roformer"}
        )
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "installing", error
        assert "installing the rvc environment" not in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "rvc"}]
        events = watch(client, auth, started.id)
    assert kinds(events)[-1] == "done", events
    assert "enable denoise" in steps(events)
    config = load_config(home)
    assert (config.enable_rvc, config.enable_denoise) == (True, True)


def test_env_installed_reads_the_env_and_not_the_flag(
    home: Path, fake_env: Path, make_client: Callable[..., TestClient]
) -> None:
    from crucible.config import load_config

    with make_client(enable_llm=False, enable_tts=False):
        config = load_config(home)
    assert config.enable_llm is False
    assert tasks.env_installed(config, FAKE_BACKEND, "llm", None) is True
    assert tasks.env_installed(config, FAKE_BACKEND, "tts", "higgs-v3") is False


@pytest.mark.parametrize("installed", [False, True])
def test_native_windows_install_checks_the_llama_binary_not_a_nonexistent_venv(
    home: Path, make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch,
    installed: bool,
) -> None:
    from types import SimpleNamespace

    from crucible import llamacpp
    from crucible.backend import LLAMA_WINDOWS, Backend, Gpu

    with make_client() as client:
        config = client.app.state.config
    backend = Backend(LLAMA_WINDOWS, "windows", "x86_64",
                      Gpu("nvidia", "fixture", 24 * 1024**3), "fixture")
    calls: list[str] = []
    def engine_installation(config: Any, build: str) -> Any:
        calls.append(build)
        return SimpleNamespace(path=home / "engine", bytes=1) if installed else None
    monkeypatch.setattr(llamacpp, "installed", engine_installation)
    def unexpected_recipe(*args: Any) -> Any:
        raise AssertionError("Native Windows must not look for a Python llm recipe")
    monkeypatch.setattr(jobenv, "llm_env", unexpected_recipe)
    assert tasks.env_installed(config, backend, "llm", None) is installed
    assert calls == [llamacpp.CUDA_BUILD]


def test_adopt_refuses_a_config_from_another_home(
    home: Path, tmp_path: Path, make_client: Callable[..., TestClient]
) -> None:
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
            enable_denoise=False,
            retention_days=7,
            desktop_allowance_basis="stated",
            desktop_allowance_note="",
        )
        with pytest.raises(ConfigError, match="different server"):
            mine.adopt(load_config(elsewhere))
        assert mine.enable_llm is False


def _frozen_writes(path: Path) -> list[str]:
    import ast

    found: list[str] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else function
            )
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "__setattr__"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "object"
            ):
                found.append(function)
            visit(child, name)

    visit(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    return found


def test_nothing_but_adopt_writes_to_a_frozen_config() -> None:
    package = Path(__file__).resolve().parent.parent / "crucible"
    writes = {
        path.relative_to(package).as_posix(): _frozen_writes(path)
        for path in package.rglob("*.py")
    }
    assert writes["config.py"] == ["adopt"]
    elsewhere = {
        name: functions for name, functions in writes.items()
        if name != "config.py" and functions
    }
    assert all(
        functions == ["__post_init__"] * len(functions)
        for functions in elsewhere.values()
    ), f"only a dataclass deriving its own field in __post_init__ may: {elsewhere}"


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


def test_an_install_streams_its_lines_reloads_and_makes_the_type_live(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    with make_client(enable_echo=True, enable_llm=False) as client:
        assert "load-model" not in client.get("/v1/info", headers=auth).json()["job_types"]
        events = run(client, auth, {"type": "install", "job_type": "llm"})
        info = client.get("/v1/info", headers=auth).json()
        setup = client.get("/v1/setup", headers=auth).json()

    assert kinds(events)[-1] == "done"
    assert steps(events) == ["install llm", "reload"]
    lines = [
        e["data"]["line"] for e in events
        if e["event"] == "progress" and "line" in e["data"]
    ]
    assert any("collecting torch" in line for line in lines)
    reload_step = next(e for e in events if e["data"].get("name") == "reload")
    assert "load-model" in reload_step["data"]["job_types"]
    assert "load-model" in info["job_types"]
    assert setup["job_types"] == info["job_types"]


def test_an_install_reports_its_download_in_the_shape_a_pull_does(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    with make_client(enable_echo=True, enable_llm=False) as client:
        events = run(client, auth, {"type": "install", "job_type": "llm"})

    progress = [e["data"] for e in events if e["event"] == "progress"]
    measured = [row for row in progress if "bytes_done" in row]
    assert measured, "no byte progress reached the task's event stream"
    assert measured[0] == {
        "bytes_done": 1000,
        "bytes_total": 3000,
        "file": "cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
    }
    assert all(set(row) == {"bytes_done", "bytes_total", "file"} for row in measured)
    lines = [row["line"] for row in progress if "line" in row]
    assert not [line for line in lines if line.startswith("crucible-progress")]


def test_the_reload_adopts_the_config_so_every_door_agrees(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
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


def test_an_install_runs_while_a_chat_holds_the_card(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    # Victoria's laptop, 2026-10-11: B-Sides' module was refused `server_busy` for as long
    # as her albums rendered. An install builds an env and ADDS the types it turned on, so
    # nothing on the card is in its way.
    with make_client(enable_echo=True, enable_llm=False) as client:
        with holding_the_card(client):
            events = run(client, auth, {"type": "install", "job_type": "llm"})
        info = client.get("/v1/info", headers=auth).json()
    assert kinds(events)[-1] == "done", events
    assert steps(events) == ["install llm", "reload"]
    assert "load-model" in info["job_types"]


def test_an_install_runs_while_a_queue_session_holds_the_server(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    with make_client(enable_echo=True, enable_llm=False) as client:
        opened = client.post(
            "/v1/queue/sessions",
            headers={**auth, "X-Crucible-Client": "b-sides/victoria"},
            json={"act": "translate"},
        )
        assert opened.status_code == 202, opened.text
        events = run(client, auth, {"type": "install", "job_type": "llm"})
    assert kinds(events)[-1] == "done", events


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


def test_the_take_up_adds_a_type_and_replaces_no_plugin_a_job_runs_on(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_installer: Path
) -> None:
    import base64

    with make_client(enable_echo=True, enable_llm=False) as client:
        registry = client.app.state.store.registry
        echo = registry["echo"]
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
        events = run(client, auth, {"type": "install", "job_type": "llm"})
        assert registry["echo"] is echo
        assert "load-model" in registry
        with client.stream(
            "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
        ) as stream:
            job_events = parse_sse(line for line in stream.iter_lines())
    assert kinds(events)[-1] == "done", events
    assert job_events[-1]["event"] == "done", job_events


def test_a_holder_that_arrives_during_the_install_does_not_fail_it(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_installer: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    assert events[-1]["event"] == "done", events
    assert "load-model" in info["job_types"]


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
    assert steps(events) == ["install llm", f"pull model {MODEL}", "enable llm", "reload"]
    assert "load-model" in info["job_types"]


def test_a_module_skips_what_is_already_true(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    hub: FakeHub,
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert len(skipped) == 4, skipped
    assert any("install llm" in reason for reason in skipped)
    assert any(MODEL in reason for reason in skipped)
    assert any("enable llm" in reason and "already on" in reason for reason in skipped)
    assert any("reload" in reason for reason in skipped)
    assert not [e for e in events if e["event"] == "progress"]


def test_a_module_stops_at_the_first_failure_and_keeps_what_finished(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    hub: FakeHub,
    fake_installer: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert "step 1 of 4" in events[-1]["data"]["message"]
    assert steps(events) == ["install llm"]
    assert row["installed"] is False


# Victoria's laptop: an RTX 3070 Laptop GPU, 8 GiB, its desktop on the integrated GPU. At a
# 1 GiB allowance this card holds rvc, the separator and YuE2 (under [audio] low_vram,
# which crucible/lowvram.py turns on for it), and not video or the bf16 qwen3.5-4b.
EIGHT_GB = Backend(
    kind="cuda-linux",
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3070 Laptop GPU", vram_bytes=8 * GIB),
    detail="test double",
)
LAPTOP_DESKTOP = 1 * GIB


def card_record(backend: Backend, desktop: int) -> Any:
    """The capability record `crucible capability --write` leaves on this card."""
    decisions = decide_on(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=desktop,
        gpu_vendor=backend.gpu.vendor,
        card=None,
        chosen={},
        audio_low_vram=True,
        packages=frozenset(),
    )
    return record_of(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=desktop,
        decisions=decisions,
        routes={},
    )


@pytest.fixture
def built_envs(home: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Envs read from the markers the fake installer writes, by every reader (the task's
    install step, the enable door's env check); `built_envs("rvc")` marks one built."""

    def built(config: Any, backend: Any, job_type: str, engine: Any = None) -> bool:
        return (home / "fake-built" / job_type).is_file()

    monkeypatch.setattr(tasks, "env_installed", built)
    monkeypatch.setattr(validate_module, "env_installed", built)
    monkeypatch.setattr(installonsubmit, "env_installed", built)

    def mark(*job_types: str) -> None:
        for job_type in job_types:
            (home / "fake-built").mkdir(parents=True, exist_ok=True)
            (home / "fake-built" / job_type).write_text("1", encoding="utf-8")

    return mark


@pytest.fixture
def pulled(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []

    async def fake_pull(self: Any, task: Any, subject: Any) -> None:
        seen.append((subject.kind, subject.id))

    monkeypatch.setattr(tasks.TaskStore, "_pull", fake_pull)
    return seen


def generated_for(app: str, backend: str) -> dict[str, Any]:
    """modules/<app>.module.json as gen-modules.py wrote it, posted the way an app posts
    one: entries filtered on the generated `backends` and that key stripped (validate_module
    refuses it; commit c2f7a12)."""
    generated = json.loads(
        (Path(REPO_ROOT) / "modules" / f"{app}.module.json").read_text(encoding="utf-8")
    )

    def here(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "backends"}
            for row in rows
            if backend in row["backends"]
        ]

    return {
        **generated,
        "job_types": here(generated["job_types"]),
        "subjects": here(generated["subjects"]),
    }


def test_the_b_sides_module_on_an_8_gb_card_leaves_rvc_denoise_and_yue2_ready(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    fake_installer: Path,
    built_envs: Callable[..., None],
    pulled: list[tuple[str, str]],
) -> None:
    module = generated_for("b-sides", EIGHT_GB.kind)
    assert module["job_types"] == [{"type": "rvc"}, {"type": "audio"}]
    assert module["subjects"] == [
        {"kind": "denoise", "id": "vocals-roformer"},
        {"kind": "model", "id": "yue2-3b"},
    ]
    with make_client(
        backend=EIGHT_GB, enable_echo=True, capability=card_record(EIGHT_GB, LAPTOP_DESKTOP)
    ) as client:
        events = run(client, auth, {"type": "module", "module": module})
        task = client.get("/v1/tasks", headers=auth).json()["tasks"][0]
        info = client.get("/v1/info", headers=auth).json()
    assert kinds(events)[-1] == "done", events
    assert steps(events) == [
        "install rvc",
        "install audio",
        "pull denoise vocals-roformer",
        "pull model yue2-3b",
        "enable rvc",
        "enable denoise",
        "enable audio",
        "reload",
    ]
    assert pulled == [("denoise", "vocals-roformer"), ("model", "yue2-3b")]
    lines = [e["data"].get("line") for e in events if e["event"] == "progress"]
    assert "[jobs] enable_denoise = true" in lines
    assert task["unmet"] == []
    config = load_config(home)
    assert (config.enable_rvc, config.enable_denoise, config.enable_audio) == (True, True, True)
    assert {"rvc", "denoise", "audio"} <= set(info["job_types"])


def test_what_this_card_cannot_hold_is_skipped_by_name_and_the_rest_installs(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    fake_installer: Path,
    built_envs: Callable[..., None],
    pulled: list[tuple[str, str]],
) -> None:
    module = {
        "name": "too-big-for-a-laptop",
        "version": "1",
        "job_types": [{"type": "rvc"}, {"type": "video"}],
        "subjects": [
            {"kind": "denoise", "id": "vocals-roformer"},
            {"kind": "model", "id": "qwen3.5-4b"},
        ],
    }
    with make_client(
        backend=EIGHT_GB, enable_echo=True, capability=card_record(EIGHT_GB, LAPTOP_DESKTOP)
    ) as client:
        events = run(client, auth, {"type": "module", "module": module})
        task = client.get("/v1/tasks", headers=auth).json()["tasks"][0]
    assert kinds(events)[-1] == "done", events
    skipped = [e["data"]["reason"] for e in events if e["event"] == "skipped"]
    assert any(
        reason.startswith("install video: nothing it serves can run on this card")
        for reason in skipped
    ), skipped
    assert any(
        reason.startswith("pull model qwen3.5-4b: it cannot run on this card")
        for reason in skipped
    ), skipped
    assert not (home / "fake-built" / "video").exists(), "an env this card cannot run was built"
    assert pulled == [("denoise", "vocals-roformer")]
    by_entry: dict[str, list[str]] = {}
    for row in task["unmet"]:
        by_entry.setdefault(row["entry"], []).append(row["class"])
    assert by_entry["install video"] == ["video"]
    assert "generate" in by_entry["pull model qwen3.5-4b"]
    assert set(by_entry) == {"install video", "pull model qwen3.5-4b"}
    video_row = next(row for row in task["unmet"] if row["class"] == "video")
    assert "GiB" in video_row["reason"]
    config = load_config(home)
    assert (config.enable_rvc, config.enable_denoise, config.enable_video) == (True, True, False)


def test_a_module_turns_on_the_type_that_serves_a_subject_whose_env_was_already_there(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    built_envs: Callable[..., None],
    pulled: list[tuple[str, str]],
) -> None:
    # Victoria's laptop, 2026-10-11: the rvc env was there, so the module skipped its install,
    # pulled the separator, and nothing turned `denoise` on; B-Sides had to PUT it on itself.
    built_envs("rvc")
    module = {
        "name": "b-sides",
        "version": "2.0.4",
        "job_types": [{"type": "rvc"}],
        "subjects": [{"kind": "denoise", "id": "vocals-roformer"}],
    }
    with make_client(
        enable_echo=True,
        enable_rvc=True,
        enable_denoise=False,
        capability=card_record(FAKE_BACKEND, 3 * GIB),
    ) as client:
        events = run(client, auth, {"type": "module", "module": module})
        info = client.get("/v1/info", headers=auth).json()
    assert kinds(events)[-1] == "done", events
    assert steps(events) == [
        "install rvc", "pull denoise vocals-roformer", "enable rvc", "enable denoise", "reload",
    ]
    lines = [e["data"].get("line") for e in events if e["event"] == "progress"]
    assert lines == ["[jobs] enable_denoise = true"]
    assert load_config(home).enable_denoise is True
    assert "denoise" in info["job_types"]


def test_an_enable_the_server_refuses_is_unmet_by_name_and_the_module_is_done(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    built_envs: Callable[..., None],
    pulled: list[tuple[str, str]],
) -> None:
    module = {
        "name": "separator-only",
        "version": "1",
        "job_types": [],
        "subjects": [{"kind": "denoise", "id": "vocals-roformer"}],
    }
    with make_client(enable_echo=True, capability=card_record(FAKE_BACKEND, 3 * GIB)) as client:
        events = run(client, auth, {"type": "module", "module": module})
        task = client.get("/v1/tasks", headers=auth).json()["tasks"][0]
    assert kinds(events)[-1] == "done", events
    assert pulled == [("denoise", "vocals-roformer")]
    assert [(row["class"], row["entry"]) for row in task["unmet"]] == [
        ("denoise", "enable denoise")
    ]
    assert "the env it runs in is not built here" in task["unmet"][0]["reason"]
    skipped = [e["data"]["reason"] for e in events if e["event"] == "skipped"]
    assert any(reason.startswith("enable denoise: env_not_built: ") for reason in skipped)


def test_a_module_of_pure_pulls_is_not_gated_on_the_card(
    make_client: Callable[..., TestClient], auth: dict[str, str], hub: FakeHub
) -> None:
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
    assert steps(events) == [f"pull model {MODEL}", "enable llm", "reload"]


def test_every_task_route_needs_the_token(client: TestClient) -> None:
    assert client.post("/v1/tasks", json={"type": "pull"}).status_code == 401
    assert client.get("/v1/tasks").status_code == 401
    assert client.get("/v1/tasks/x").status_code == 401
    assert client.delete("/v1/tasks/x").status_code == 401
    assert client.get("/v1/tasks/x/events").status_code == 401
