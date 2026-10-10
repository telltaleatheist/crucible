"""Every key in config.toml is set from the operator page (Owen, 2026-10-09: "we should be
able to change everything in the config file from the crucible ui. for things that require
it to restart to take effect, it should ask the user if they want it to restart").

Live keys go through PUT /v1/settings and keep everything else in the file; host and port
wait for POST /v1/server/restart; [tts.<engine>] goes through its own door with the
config reader's rules; the bearer token rotates; the card is re-recorded on request.
"""
from __future__ import annotations

import json
import os
import time
import tomllib
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import API_VERSION, cli, pairing, service, settings, videodesktop
from crucible.api.routes import updating as updating_routes
from crucible.config import (
    check_bind_host,
    check_port,
    config_path,
    load_config,
    rewrite_config,
)
from crucible.errors import ConfigError
from crucible.narratorengines import HIGGS_V3

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, TOKEN
from .test_settings_api import decided

HF_TOKEN = "hf_zzzTESTzzzTOKENzzz9k2Q"


def _file(client: TestClient) -> dict[str, Any]:
    return tomllib.loads(client.app.state.config.path.read_text(encoding="utf-8"))


def _put(client: TestClient, auth: dict[str, str], patch: dict[str, Any]):
    return client.put("/v1/settings", headers=auth, json=patch)


@pytest.fixture
def box(make_client: Callable[..., TestClient]):
    with make_client(enable_llm=True, capability=decided()) as instance:
        yield instance


def test_live_keys_are_written_whole_and_keep_the_token_and_every_other_table(box, auth):
    path = box.app.state.config.path
    path.write_text(
        path.read_text(encoding="utf-8") + '\n[hf]\ntoken = "hf_keptkeptkept"\nmirror = "x"\n',
        encoding="utf-8",
    )
    response = _put(box, auth, {
        "name": "studio",
        "advertise": ["studio.example:7100"],
        "cors_origins": ["capacitor://localhost"],
        "install_on_submit": False,
        "retention_days": 3,
        "max_session_hold_s": 900,
    })
    assert response.status_code == 200, response.text
    answer = response.json()
    assert (answer["name"], answer["advertise"], answer["cors_origins"]) == (
        "studio", ["studio.example:7100"], ["capacitor://localhost"]
    )
    assert (answer["install_on_submit"], answer["retention_days"],
            answer["max_session_hold_s"]) == (False, 3, 900)
    written = _file(box)
    assert written["auth"]["token"] == TOKEN
    assert written["queue"] == {"max_session_hold_s": 900}
    assert written["hf"] == {"token": "hf_keptkeptkept", "mirror": "x"}
    assert written["capability"]["total_bytes"] == decided().total_bytes
    live = box.app.state.config
    assert (live.name, live.max_session_hold_s) == ("studio", 900)
    assert box.get("/v1/ping").json()["name"] == "studio"
    changed = box.app.state.settings_history.rows()[0]["changed"]
    assert "[server] name = studio" in changed and "[queue] max_session_hold_s = 900" in changed


def test_a_bad_value_is_refused_by_name_and_nothing_is_written(box, auth):
    before = box.app.state.config.path.read_bytes()
    for patch, field in (
        ({"retention_days": 0}, "retention_days"),
        ({"port": 70000}, "port"),
        ({"host": "http://0.0.0.0"}, "host"),
        ({"name": "two\nlines"}, "name"),
        ({"cors_origins": ["*"]}, "cors_origins"),
        ({"open_pairing": "no"}, "open_pairing"),
        ({"max_session_hold_s": -1}, "max_session_hold_s"),
        ({"name": "fine", "retention_days": -3}, "retention_days"),
    ):
        response = _put(box, auth, patch)
        assert response.status_code == 400, (patch, response.text)
        assert response.json()["error"]["details"]["field"] == field
    assert box.app.state.config.path.read_bytes() == before


def test_open_pairing_applies_to_the_next_app_that_asks(box, auth):
    start = {"client_name": "BookForge"}
    headers = {"X-Crucible-Api": str(API_VERSION)}
    assert box.post("/v1/pairing/start", headers=headers, json=start).json()[
        "approval_required"] is False
    assert _put(box, auth, {"open_pairing": False}).status_code == 200
    time.sleep(5.1)  # the pairing door refuses a second request from one address within 5 s
    assert box.post("/v1/pairing/start", headers=headers, json=start).json()[
        "approval_required"] is True
    assert _file(box)["auth"]["open_pairing"] is False


def test_host_and_port_are_written_and_wait_for_a_restart(box, auth):
    answer = _put(box, auth, {"port": 7200, "host": "0.0.0.0"}).json()
    assert (answer["host"], answer["port"]) == ("0.0.0.0", 7200)
    assert answer["bound"] == {"host": "127.0.0.1", "port": 7100}
    assert answer["restart_pending"] == ["host", "port"]
    assert _file(box)["server"]["port"] == 7200
    assert box.app.state.bind_port == 7100
    changed = box.app.state.settings_history.rows()[0]["changed"]
    assert "[server] port = 7200 (takes effect when Crucible restarts)" in changed


def test_a_pc_engine_keeps_the_port_the_windows_host_dials(
    box, auth, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(service, "in_wsl", lambda: True)
    response = _put(box, auth, {"port": 7200})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "port_fixed_by_windows_host"
    assert _put(box, auth, {"port": 7100}).status_code == 200


def test_the_hf_token_is_write_only(box, auth):
    response = _put(box, auth, {"hf_token": HF_TOKEN})
    assert response.status_code == 200
    assert HF_TOKEN not in response.text and HF_TOKEN not in box.get(
        "/v1/settings", headers=auth).text
    assert response.json()["hf"] == {
        "configured": True, "token_hint": "…9k2Q", "from_environment": False,
    }
    assert _file(box)["hf"] == {"token": HF_TOKEN}
    assert box.app.state.settings_history.rows()[0]["changed"] == ["[hf] token set"]
    assert _put(box, auth, {"hf_token": "has a space"}).status_code == 400
    answer = _put(box, auth, {"hf_token": None}).json()
    assert answer["hf"]["configured"] is False and "hf" not in _file(box)


def test_no_secret_is_in_the_settings_document(box, auth):
    text = box.get("/v1/settings", headers=auth).text
    assert TOKEN not in text
    assert json.loads(text)["token_hint"] == f"…{TOKEN[-4:]}"


def test_video_desktop_is_the_macs_and_out_of_range_is_refused(
    make_client: Callable[..., TestClient], box, auth
):
    refused = _put(box, auth, {"video_desktop": {"gpu_duty_pct": 70}})
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "video_desktop_not_here"
    assert box.get("/v1/settings", headers=auth).json()["video_desktop"] is None
    with make_client(backend=FAKE_MAC_BACKEND) as mac:
        for patch in ({"gpu_duty_pct": 150}, {"gpu_duty_pct": 5}, {"low_ram": 1},
                      {"tile_spatial": 0}, {"gpu_busy_target_pct": 0}):
            response = _put(mac, auth, {"video_desktop": patch})
            assert response.status_code == 400, patch
            assert response.json()["error"]["code"] == "video_desktop_out_of_range"
        unknown = _put(mac, auth, {"video_desktop": {"speed": 3}})
        assert unknown.json()["error"]["code"] == "video_desktop_unknown_key"
        answer = _put(mac, auth, {"video_desktop": {"gpu_duty_pct": 70, "low_ram": False}})
        assert answer.status_code == 200
        rows = {row["key"]: row for row in answer.json()["video_desktop"]["rows"]}
        assert rows["gpu_duty_pct"]["set"] == 70 and rows["gpu_duty_pct"]["default"] == 85
        assert _file(mac)["video_desktop"] == {"gpu_duty_pct": 70, "low_ram": False}
        assert _put(mac, auth, {"video_desktop": {"gpu_duty_pct": None, "low_ram": None}}
                    ).status_code == 200
        assert "video_desktop" not in _file(mac)


def test_a_value_the_engine_skips_is_shown_with_why():
    rows = {row["key"]: row for row in videodesktop.rows({"gpu_duty_pct": 400})}
    assert "the engine runs 85" in rows["gpu_duty_pct"]["problem"]
    assert rows["tile_overlap"]["problem"] is None


def test_a_job_type_is_turned_off_and_refused_on_by_name(
    make_client: Callable[..., TestClient], box, auth
):
    off = box.put("/v1/settings/jobs/llm", headers=auth, json={"enabled": False})
    assert off.status_code == 200, off.text
    rows = {row["job_type"]: row for row in off.json()["job_types"]}
    assert rows["llm"]["enabled"] is False
    assert _file(box)["jobs"]["enable_llm"] is False and _file(box)["auth"]["token"] == TOKEN
    unknown = box.put("/v1/settings/jobs/nope", headers=auth, json={"enabled": True})
    assert unknown.status_code == 404
    bad = box.put("/v1/settings/jobs/llm", headers=auth, json={"enabled": "yes"})
    assert bad.status_code == 400
    with make_client(enable_llm=False) as undecided:
        refused = undecided.put("/v1/settings/jobs/llm", headers=auth, json={"enabled": True})
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "job_type_undecided"
        assert "crucible capability --write" in refused.json()["error"]["message"]


def test_a_tts_engine_number_keeps_the_config_readers_rules(box, auth):
    path = f"/v1/settings/tts/{HIGGS_V3}"
    answer = box.put(path, headers=auth, json={"max_num_seqs": 8})
    assert answer.status_code == 200, answer.text
    row = next(r for r in answer.json()["tts_engines"] if r["engine"] == HIGGS_V3)
    assert row["max_num_seqs"] == 8 and row["resident"] is None
    assert _file(box)["tts"][HIGGS_V3]["max_num_seqs"] == 8
    for patch in ({"mem_fraction": 0.5}, {"mem_fraction": 1.5, "mem_fraction_note": "x"},
                  {"max_num_seqs": 0}, {"estimate_note": None}):
        refused = box.put(path, headers=auth, json=patch)
        assert refused.status_code == 400, patch
        assert refused.json()["error"]["code"] == "tts_lever_invalid"
    unknown = box.put(path, headers=auth, json={"speed": 2})
    assert unknown.json()["error"]["code"] == "tts_lever_unknown"
    added = box.put(path, headers=auth, json={
        "mem_fraction": 0.55, "mem_fraction_note": "measured on owens-pc"})
    assert added.status_code == 200
    assert _file(box)["tts"][HIGGS_V3]["mem_fraction"] == 0.55
    unset = box.put("/v1/settings/tts/other-engine", headers=auth, json={"max_num_seqs": 2})
    assert unset.status_code == 409
    assert unset.json()["error"]["code"] == "engine_footprint_unset"


def test_a_voice_started_with_other_numbers_is_offered_a_reload():
    row = {"engine": HIGGS_V3, "memory_bytes_estimate": 19, "max_num_seqs": 8}
    voice = SimpleNamespace(narrator_engine=HIGGS_V3, voice_id="deathstalker", levers={
        "memory_bytes_estimate": 19, "max_num_seqs": 16,
        "mem_fraction": None, "context_length": None})
    assert settings._resident_on_engine(row, voice) == {
        "voice": "deathstalker", "levers": voice.levers, "reload_needed": True,
    }
    voice.levers["max_num_seqs"] = 8
    assert settings._resident_on_engine(row, voice)["reload_needed"] is False
    assert settings._resident_on_engine({**row, "engine": "other"}, voice) is None


def test_rotating_the_token_locks_the_old_one_out_and_rewrites_the_pairing_file(box, auth):
    answer = box.post("/v1/settings/token/rotate", headers=auth)
    assert answer.status_code == 200, answer.text
    fresh = answer.json()["token"]
    assert fresh != TOKEN and _file(box)["auth"]["token"] == fresh
    assert answer.json()["pairing_file_error"] is None
    assert box.get("/v1/settings", headers=auth).status_code == 401
    renewed = {**auth, "Authorization": f"Bearer {fresh}"}
    assert box.get("/v1/settings", headers=renewed).status_code == 200
    line = pairing.read_pairing_file(box.app.state.config.home)
    assert line is not None and pairing.parse_pairing_line(line).token == fresh
    assert box.app.state.settings_history.rows()[0]["changed"] == ["[auth] token rotated"]


def test_re_measuring_the_card_records_it(make_client: Callable[..., TestClient], auth):
    with make_client(enable_llm=True) as client:
        assert "capability" not in _file(client)
        answer = client.post("/v1/capability/record", headers=auth)
        assert answer.status_code == 200, answer.text
        assert answer.json()["total_bytes"] == FAKE_BACKEND.gpu.vram_bytes
        assert _file(client)["capability"]["total_bytes"] == FAKE_BACKEND.gpu.vram_bytes
        assert _file(client)["auth"]["token"] == TOKEN
        assert client.get("/v1/capability", headers=auth).status_code == 200


# --- the restart ---------------------------------------------------------------------


class FakeServer:
    should_exit = False


def _supervised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, pid: int, port: int = 7100
) -> None:
    user = tmp_path / "user-home"
    monkeypatch.setattr(service, "user_home", lambda: user)
    unit = service.unit_path(user, service.USER_SCOPE)
    unit.parent.mkdir(parents=True)
    unit.write_text(service.systemd_unit_text(
        server_name="crucible@test", program="/opt/crucible/bin/crucible",
        crucible_home=PurePosixPath("/home/telltale/.crucible"), host="127.0.0.1", port=port,
        path_value="/usr/bin",
    ), encoding="utf-8")

    def runner(argv):
        if argv[0] == "systemctl":
            shown = f"ActiveState=active\nSubState=running\nMainPID={pid}\nUnitFileState=enabled\n"
            return service.Ran(tuple(argv), 0, shown, "")
        return service.Ran(tuple(argv), 0, "Linger=yes\n", "")

    monkeypatch.setattr(service, "subprocess_runner", runner)


def test_a_server_started_by_hand_is_not_restarted(
    box, auth, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _supervised(monkeypatch, tmp_path, pid=os.getpid() + 1)
    refused = box.post("/v1/server/restart", headers=auth)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "restart_not_supervised"
    assert "crucible service install" in refused.json()["error"]["message"]


def test_a_supervised_server_stops_to_be_started_again(
    box, auth, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _supervised(monkeypatch, tmp_path, pid=os.getpid())
    server = FakeServer()
    box.app.state.uvicorn_server = server
    answer = box.post("/v1/server/restart", headers=auth)
    assert answer.status_code == 202, answer.text
    assert answer.json()["by"] == "systemd" and answer.json()["url"] == "http://127.0.0.1:7100"
    held = box.post("/v1/jobs", headers=auth, json={"type": "echo", "params": {"text": "x"}})
    assert held.status_code == 503
    assert "to take up a settings change" in held.json()["error"]["message"]
    deadline = time.monotonic() + 5
    while not server.should_exit and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.should_exit and box.app.state.restart_asked is True


def test_a_service_bound_to_the_old_address_needs_reinstalling(
    box, auth, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _supervised(monkeypatch, tmp_path, pid=os.getpid())
    box.app.state.uvicorn_server = FakeServer()
    assert _put(box, auth, {"port": 7200}).status_code == 200
    refused = box.post("/v1/server/restart", headers=auth)
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "restart_needs_service_install"
    assert error["details"]["command"] == "crucible service install"
    assert box.app.state.restart_asked is False


def test_a_working_server_is_not_restarted_and_lets_the_hold_go(
    box, auth, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _supervised(monkeypatch, tmp_path, pid=os.getpid())
    box.app.state.uvicorn_server = FakeServer()
    monkeypatch.setattr(updating_routes, "working", lambda ctx: ["job j1 (echo) 40% done"])
    refused = box.post("/v1/server/restart", headers=auth)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "server_working"
    assert box.app.state.updating.current() is None
    assert box.app.state.restart_asked is False


def test_the_definitions_say_their_address_and_that_they_restart(tmp_path: Path):
    user = tmp_path / "user"
    unit = service.unit_path(user, service.USER_SCOPE)
    unit.parent.mkdir(parents=True)
    unit.write_text(service.systemd_unit_text(
        server_name="s", program="/bin/crucible", crucible_home=PurePosixPath("/home/t/.crucible"),
        host="0.0.0.0", port=7300, path_value="/usr/bin"), encoding="utf-8")
    assert service.read_defined(service.SYSTEMD, user) == service.Defined("0.0.0.0", 7300, True)
    plist = service.plist_path(user)
    plist.parent.mkdir(parents=True)
    plist.write_text(service.launchd_plist_text(
        program="/bin/crucible", crucible_home=tmp_path, host="127.0.0.1", port=7100,
        path_value="/usr/bin", log_path=tmp_path / "serve.log"), encoding="utf-8")
    assert service.read_defined(service.LAUNCHD, user) == service.Defined("127.0.0.1", 7100, True)
    assert service.read_defined(service.SYSTEMD, tmp_path / "nobody") is None


def test_serve_exits_for_the_service_manager_when_a_restart_was_asked(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from crucible.api import serving

    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init"]) == 0

    class Ran:
        def __init__(self, app):
            self.app = app

        def run(self):
            self.app.state.restart_asked = True

    monkeypatch.setattr(serving, "server_for", lambda app, **_: Ran(app))
    assert cli.main(["serve"]) == service.SELF_RESTART_EXIT
    assert "exiting 75" in capsys.readouterr().out


# --- the writer ----------------------------------------------------------------------


def test_rewrite_sets_keys_in_a_table_it_does_not_own_and_carries_the_rest(home: Path, box):
    config = load_config(box.app.state.config.home)
    rewrite_config(config, unowned={"queue": {"max_session_hold_s": 60, "other": 1}})
    rewrite_config(load_config(config.home), unowned={"queue": {"max_session_hold_s": None}})
    assert tomllib.loads(config_path(config.home).read_text(encoding="utf-8"))["queue"] == {
        "other": 1}
    with pytest.raises(ConfigError, match="this writer owns"):
        rewrite_config(load_config(config.home), unowned={"server": {"port": 1}})


def test_the_bind_address_checks():
    assert check_bind_host("0.0.0.0") == "0.0.0.0" and check_bind_host("::") == "::"
    assert check_bind_host("owens-mac-studio.local") == "owens-mac-studio.local"
    for bad in ("", "http://x", "x:7100", "a b", 7):
        with pytest.raises(ConfigError):
            check_bind_host(bad)
    for bad in (0, 65536, True, "7100"):
        with pytest.raises(ConfigError):
            check_port(bad)
