from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import controller_client, local
from crucible.host import (
    app,
    controller_door,
    installer,
    migration,
    move_policy,
    operator_stop,
    outcome,
    presence,
    wslstate,
)
from crucible.host.info import controller_info
from crucible.host.state import Distro, Engine, EngineDecision, Owner
from crucible.platform import installation
from tests.test_host import FakeCatalog, Scripted, _context


def test_the_operator_stop_is_one_persisted_state(tmp_path: Path) -> None:
    host = app.Host(_context(tmp_path, Scripted()))
    assert host.stopped_by_operator is False
    operator_stop.record(tmp_path)
    assert operator_stop.marker(tmp_path).name == "engine.stopped"
    assert host.stopped_by_operator is True
    assert host.local_status() == {"state": "stopped", "intentional": True, "detail": "starting"}
    operator_stop.restore(tmp_path, False)
    assert host.local_status()["intentional"] is False


def test_a_stopped_ownerless_engine_still_has_its_pairing_token(tmp_path: Path) -> None:
    context = _context(tmp_path, Scripted())
    (tmp_path / "pairing").write_text("crucible://engine@127.0.0.1:7100/#kept\n", encoding="utf-8")
    assert app.engine_token(context) is None
    operator_stop.record(tmp_path)
    assert app.engine_token(context) == "kept"


def _cannot(tmp_path: Path, code: str) -> None:
    outcome.write(tmp_path, state=outcome.CANNOT, code=code, sentence="s", release="1.0.0", attempts=1)


@pytest.mark.parametrize("code,kind,resumed", [
    ("virtualization_disabled", "live", "resumed: virtualization is on now"),
    ("virtualization_disabled", "no_hypervisor", None),
    (outcome.REBOOT_BUDGET_SPENT_CODE, "live", "resumed: WSL is live now"),
    (outcome.REBOOT_BUDGET_SPENT_CODE, "component_required", None),
])
def test_one_rule_table_decides_which_cannot_resumes(tmp_path, monkeypatch, code, kind, resumed) -> None:
    context = _context(tmp_path, Scripted())
    _cannot(tmp_path, code)
    live = wslstate.LiveWsl(answer=wslstate.WslAnswer(kind), features={}, signals=())
    monkeypatch.setattr(wslstate, "probe_live", lambda runner: live)
    asked: list[str] = []
    decided = move_policy.decide_engine(context, lambda why: asked.append(why) or EngineDecision.DONE)
    assert asked == ([resumed] if resumed else [])
    assert decided is (EngineDecision.DONE if resumed else EngineDecision.CANNOT)


def test_a_cannot_no_rule_names_is_never_probed(tmp_path, monkeypatch) -> None:
    context = _context(tmp_path, Scripted())
    _cannot(tmp_path, "distro_unmarked")
    monkeypatch.setattr(wslstate, "probe_live", lambda runner: pytest.fail("nothing to re-check"))
    assert move_policy.decide_engine(context, lambda why: pytest.fail(why)) is EngineDecision.CANNOT


def test_the_attempt_carries_the_walks_restarts_as_a_field() -> None:
    attempt = move_policy.MoveAttempt(number=1, restarts_before=2, rebooted=True)
    assert attempt.restarts == 2
    attempt.walk = SimpleNamespace(restarts=3)
    assert attempt.restarts == 3


def test_an_active_guest_resumes_its_cleanup_before_completing(tmp_path, monkeypatch) -> None:
    context = _context(tmp_path, Scripted())
    context.presence = presence.Presence(Distro.PRESENT, Engine.RUNNING, "guest", Owner.WSL_UNIT)
    host = app.Host(context)
    installer.record_cleanup(tmp_path, {("model", "a")})
    calls: list[str] = []
    monkeypatch.setattr(host, "resume_model_cleanup", lambda *, raise_errors: calls.append(f"resume {raise_errors}"))
    monkeypatch.setattr(installer.EngineInstall, "complete", lambda self: calls.append("complete"))
    app._sequence(context, host)(lambda event: None)
    assert calls == ["resume True", "complete"]
    assert outcome.read(tmp_path).state == outcome.DONE


def test_the_model_cleanup_keeps_its_record_until_the_guest_answers(tmp_path) -> None:
    context = _context(tmp_path, Scripted())
    windows = FakeCatalog("stopped Windows", [("model", "a")])
    guest = FakeCatalog("active guest", [("model", "a")])
    installer.record_cleanup(tmp_path, {("model", "a")})
    cleanup = migration.ModelCleanup(
        context, lock=app.threading.RLock(), windows_catalog=lambda: windows,
        guest_catalog=lambda: guest, clock=lambda: 100.0,
    )
    cleanup.running = True
    guest.unreachable = True
    cleanup.resume()
    assert cleanup.record.exists() and windows.subjects
    assert cleanup.running is False and cleanup.retry_at == 100.0 + migration.RETRY_SECONDS
    guest.unreachable = False
    cleanup.resume()
    assert not cleanup.record.exists() and windows.subjects == []


def test_the_controller_info_payload_keeps_its_shape() -> None:
    lines: list[str] = []
    body = controller_info("c@pc", Owner.WSL_UNIT, engine_url="http://127.0.0.1:7100",
                           token=lambda: None, log=lines.append)
    assert list(body) == ["server", "host", "role", "local_lifecycle_version", "job_types", "engine", "capabilities"]
    assert body["engine"] == {"name": None, "url": "http://127.0.0.1:7100", "backend": None, "owner": "wsl-unit"}
    assert body["capabilities"] == [] and body["role"] == "orchestrator"
    assert controller_info("c@pc", Owner.NONE, engine_url="u", token=lambda: "t", log=lines.append)["engine"] is None


class _Orchestrator:
    name = "crucible-orchestrator@test"

    def local_status(self) -> dict[str, object]:
        return {"state": "running"}


def _door_server(tmp_path: Path):
    from crucible.host.log import HostLog

    door = controller_door.OrchestratorDoor(
        HostLog(tmp_path / "host.log", tmp_path / "host.log.1"), lambda emit: None,
        token=lambda: "t", orchestrator=_Orchestrator(),
    )
    return controller_door.serve(door, port=0)


def _get(server, path: str, token: str | None = None) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_door_dispatches_through_one_route_table(tmp_path: Path) -> None:
    assert set(controller_door.GET_ROUTES) == {"/local/status", "/install/events", "/install", "/v1/ping", "/v1/info"}
    assert set(controller_door.POST_ROUTES) == {"/local/start", "/local/stop", "/restart", "/quit", "/install"}
    server = _door_server(tmp_path)
    try:
        assert _get(server, "/v1/ping/")[1]["role"] == "orchestrator"
        assert _get(server, "/local/status", "t") == (200, {"state": "running"})
        assert _get(server, "/local/status", "wrong")[0] == 401
        status, body = _get(server, "/nowhere")
        assert status == 404 and "this orchestrator serves" in body["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()


def test_the_old_module_names_are_the_new_modules() -> None:
    from crucible.host import door
    from crucible.host import landoor as host_landoor
    from crucible.platform import lan_door, landoor

    assert door is controller_door
    assert landoor is lan_door is host_landoor


def test_local_runs_the_tray_verbs_it_is_handed(capsys) -> None:
    ran: list[str] = []
    assert local.command(argparse.Namespace(local_action="remove-desktop"), tray_verbs=ran.append) == 0
    assert ran == ["remove-desktop"] and capsys.readouterr().out == ""
    parser = argparse.ArgumentParser()
    local.add_parser(parser.add_subparsers(dest="command"), tray_verbs=ran.append)
    args = parser.parse_args(["local", "tray"])
    assert args.func(args) == 0 and ran[-1] == "tray"


def test_the_installation_record_lives_in_platform(tmp_path: Path) -> None:
    assert local.publish_installation is installation.publish_installation
    assert installation.release_order("1.0.10", "v1.0.2") == 1
    with pytest.raises(local.LocalError, match="release_unreadable"):
        installation.release_order("latest", "1.0.0")
    with pytest.raises(local.LocalError, match="crucible local register"):
        installation.installed_control(tmp_path)


def _shutdown(tmp_path: Path, owner: str, sent: list) -> None:
    def send(url: str, **options: object) -> dict:
        sent.append((url.rsplit("/", 1)[-1], options.get("headers")))
        if url.endswith("/quit"):
            return {"quit": True}
        if url == controller_client.CONTROLLER_URL + "/v1/ping" and any(u == "quit" for u, _ in sent):
            raise urllib.error.URLError(ConnectionRefusedError())
        return {"crucible": True, "role": "orchestrator"}

    info = {"server": {"api_version": controller_client.API_VERSION, "version": "1.0.0"},
            "role": "orchestrator", "local_lifecycle_version": 1, "engine": {"owner": owner}}
    controller_client.shutdown_controller(
        tmp_path, send=send, engine_token=lambda: "t", call=lambda path, token: (info, token),
        stop_engine=lambda: sent.append(("stop-engine", None)), alive=lambda pid: False,
    )


def test_a_guest_keeps_serving_through_the_controllers_quit(tmp_path: Path) -> None:
    (tmp_path / "host.pid").write_text("4242")
    sent: list = []
    _shutdown(tmp_path, "wsl-unit", sent)
    assert ("stop-engine", None) not in sent
    assert ("quit", {controller_client.HANDOVER_HEADER: "1"}) in sent


def test_a_controller_with_no_pid_record_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(local.LocalError, match="host.pid"):
        _shutdown(tmp_path, "wsl-unit", [])


def test_the_walks_verbs_other_modules_call_are_public() -> None:
    assert callable(installer.EngineInstall.complete)
    assert installer.EngineInstall._migrate_weights is installer.EngineInstall.migrate_weights
    assert presence.PresenceWatcher._wait_for_ping is presence.PresenceWatcher.wait_for_ping


def test_the_wheel_url_is_read_from_a_named_row() -> None:
    assert wslstate.GUEST_WHEEL_URL_TEMPLATE == wslstate.state_row(wslstate.NETWORK_ROW).action_url
    assert "{release}" in wslstate.GUEST_WHEEL_URL_TEMPLATE
