from __future__ import annotations

import base64
import json

import pytest

from crucible import lan
from crucible.errors import CrucibleError
from crucible.platform import lan_door as landoor
from crucible.platform.runner import RunResult

WINDOWS_ENV = {"USERPROFILE": r"C:\Users\tellt"}

FORWARD_ROW = (
    "Listen on ipv4:             Connect to ipv4:\n\n"
    "Address         Port        Address         Port\n"
    "--------------- ----------  --------------- ----------\n"
    "0.0.0.0         7100        127.0.0.1       7100\n"
)

PRIVATE = json.dumps([{"InterfaceAlias": "Ethernet 2", "NetworkCategory": "Private"}])
PUBLIC = json.dumps([{"InterfaceAlias": "Wi-Fi", "NetworkCategory": "Public"}])

ADDRESSES = ("192.168.68.100", "100.64.0.1")


def ok(stdout: str = "") -> RunResult:
    return RunResult(code=0, stdout=stdout, stderr="", failure=None)


def absent() -> RunResult:
    return RunResult(code=1, stdout="\nNo rules match the specified criteria.\n",
                     stderr="", failure=None)


class Runner:

    platform = "win32"

    def __init__(self, *, forward: bool = False, firewall: bool = False,
                 profile: str = PRIVATE, applies: bool = True,
                 addresses: tuple[str, ...] = ADDRESSES) -> None:
        self.forward = forward
        self.firewall = firewall
        self.profile = profile
        self.applies = applies
        self.addresses = list(addresses)
        self.calls: list[list[str]] = []
        self.dialled: list[str] = []
        self.env = dict(WINDOWS_ENV)

    def network(self) -> str:
        category = json.loads(self.profile)[0]["NetworkCategory"]
        rule = (
            [{"enabled": "True", "profile": "Private", "action": "Allow",
              "direction": "Inbound"}]
            if self.firewall else []
        )
        return json.dumps({
            "profiles": [
                {"index": index, "alias": f"Ethernet {index}",
                 "name": f"Network {index}", "category": category}
                for index, _ in enumerate(self.addresses, start=1)
            ],
            "addresses": [
                {"address": address, "index": index, "alias": f"Ethernet {index}"}
                for index, address in enumerate(self.addresses, start=1)
            ],
            "firewall": [
                {"profile": name, "enabled": "True", "inbound": "Block",
                 "allow_rules": "True", "local_rules": "True"}
                for name in ("Domain", "Private", "Public")
            ],
            "rule": rule,
            "elevated": False,
        })

    def get(self, url: str, *, timeout_s: float) -> int | None:
        self.dialled.append(url)
        return 200 if self.forward else None

    def run(self, argv, *, timeout_s, env=None) -> RunResult:
        self.calls.append(list(argv))
        joined = " ".join(argv)
        if "type" in joined and ".wslconfig" in joined:
            return ok("[wsl2]\nmemory=13GB\n")
        if "portproxy show" in joined:
            return ok(FORWARD_ROW if self.forward else "")
        if "advfirewall firewall show" in joined:
            return ok("Rule Name: Crucible engine (LAN)\n") if self.firewall else absent()
        if "Get-NetIPAddress" in joined:
            return ok(self.network())
        if "Get-NetConnectionProfile" in joined:
            return ok(self.profile)
        if "Start-Process" in joined:
            script = base64.b64decode(argv[-1].split("'")[-2]).decode("utf-16-le")
            if self.applies:
                if "portproxy" in script:
                    self.forward = "delete" not in script
                if "advfirewall" in script:
                    self.firewall = "delete" not in script
            return ok()
        raise AssertionError(f"unscripted call: {joined}")

    @property
    def elevations(self) -> list[list[str]]:
        return [c for c in self.calls if "Start-Process" in " ".join(c)]


class Engine:

    target = "127.0.0.1:7100"

    def __init__(self, backend: str | None = "cuda-linux") -> None:
        self.published: dict[str, list[str]] = {}
        self.verified = 0
        self.backend = backend
        self.paths: list[str] = []

    def verify(self) -> None:
        self.verified += 1

    def advertise(self, field: str, authorities: list[str]) -> None:
        self.published[field] = authorities

    def request(self, method="GET", path="settings", body=None):
        self.paths.append(path)
        if path == "info":
            host = {"platform": "linux", "arch": "x86_64"}
            if self.backend is not None:
                host["backend"] = self.backend
            return {"role": "engine", "host": host}
        return {"lan_advertise": self.published.get("lan_advertise", [])}


def _decoded(argv: list[str]) -> str:
    return base64.b64decode(argv[-1].split("'")[-2]).decode("utf-16-le")


def test_one_prompt_carries_both_rows_and_names_no_temporary_file() -> None:
    argv = lan.elevated_argv([landoor.add_argv(7100), landoor.firewall_add_argv(7100)])
    assert argv[0] == "powershell.exe"
    assert sum("Start-Process" in word for word in argv) == 1, "exactly one prompt"
    script = _decoded(argv)
    assert "interface' 'portproxy' 'add'" in script
    assert "advfirewall' 'firewall' 'add'" in script
    assert landoor.RULE_NAME in script
    assert ".cmd" not in script and ".netsh" not in script and ".ps1" not in script


def test_the_encoded_payload_is_quote_safe_by_construction() -> None:
    argv = lan.elevated_argv([["netsh", "x", "name=it's got 'quotes'"]])
    blob = argv[-1].split("'")[-2]
    assert set(blob) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    assert "'name=it''s got ''quotes'''" in _decoded(argv)


def test_this_door_refuses_to_exist_anywhere_but_windows(tmp_path, monkeypatch) -> None:
    from crucible import service

    monkeypatch.setattr(service, "in_wsl", lambda: False)
    runner = Runner()
    runner.platform = "linux"
    with pytest.raises(lan.LanError, match="lan_not_windows"):
        lan.enable(tmp_path, runner, Engine())
    assert runner.calls == [], "it did not probe a machine it cannot change"


def test_mirrored_networking_is_refused_as_nothing_to_add(tmp_path, monkeypatch) -> None:
    runner = Runner()
    monkeypatch.setattr(
        runner, "run",
        lambda argv, *, timeout_s, env=None: ok("networkingMode=mirrored\n")
        if "wslconfig" in " ".join(argv) else ok(""),
    )
    with pytest.raises(lan.LanError, match="lan_mirrored"):
        lan.enable(tmp_path, runner, Engine())


def test_a_forward_we_did_not_create_needs_adopt(tmp_path) -> None:
    runner = Runner(forward=True)
    with pytest.raises(lan.LanError, match="lan_unowned"):
        lan.enable(tmp_path, runner, Engine())
    assert not (tmp_path / lan.RECORD).exists(), "a refusal claims nothing"
    assert runner.elevations == [], "a refusal changes nothing"
    assert lan.enable(tmp_path, runner, Engine(), adopt=True)["state"] == "configured"


def test_a_machine_with_no_address_publishes_nothing(tmp_path) -> None:
    with pytest.raises(lan.LanError, match="lan_no_addresses"):
        lan.enable(tmp_path, Runner(addresses=()), Engine())


def test_a_native_windows_engine_is_refused_and_nothing_is_touched(tmp_path) -> None:
    runner, engine = Runner(), Engine(backend="llama-windows")
    with pytest.raises(lan.LanError, match="lan_native_engine"):
        lan.enable(tmp_path, runner, engine)
    assert not (tmp_path / lan.RECORD).exists(), "a refusal claims nothing"
    assert runner.calls == [], "it did not even read a machine it cannot change"


def test_the_native_refusal_says_what_opens_that_machine_instead(tmp_path) -> None:
    with pytest.raises(lan.LanError) as raised:
        lan.enable(tmp_path, Runner(), Engine(backend="llama-windows"))
    said = str(raised.value)
    assert "[server] host" in said
    assert "inbound allow for TCP 7100" in said


def test_the_wsl_engine_is_the_unchanged_path(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    assert lan.enable(tmp_path, runner, engine)["state"] == "configured"
    assert "info" in engine.paths, "it asked the engine which engine it is"
    assert runner.elevations, "and then opened the door as it always did"


def test_an_engine_that_will_not_say_its_backend_is_refused(tmp_path) -> None:
    runner = Runner()
    with pytest.raises(lan.LanError, match="lan_engine_backend_unknown"):
        lan.enable(tmp_path, runner, Engine(backend=None))
    assert runner.calls == [], "not knowing is not the same answer as wsl"


def test_enable_opens_both_rows_publishes_and_records(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    result = lan.enable(tmp_path, runner, engine)

    assert result["state"] == "configured"
    assert result["remote_reachability"] == "admitted_by_windows"
    assert result["forward_answers"] is True
    assert runner.dialled == ["http://192.168.68.100:7100/v1/ping"]
    assert result["urls"] == ["http://192.168.68.100:7100", "http://100.64.0.1:7100"]

    assert engine.published == {
        "lan_advertise": ["192.168.68.100:7100", "100.64.0.1:7100"]
    }
    assert "tailscale_advertise" not in engine.published

    record = lan.read(tmp_path)
    assert record["state"] == "configured"
    assert record["rule"] == landoor.RULE_NAME
    assert record["port"] == 7100
    assert len(runner.elevations) == 1, "both rows, one prompt"


def test_a_row_that_is_already_there_is_not_asked_for_again(tmp_path) -> None:
    runner = Runner(forward=True, firewall=True)
    lan.enable(tmp_path, runner, Engine(), adopt=True)
    assert runner.elevations == [], "nothing was missing, so nobody was prompted"


def test_only_the_missing_row_is_requested(tmp_path) -> None:
    runner = Runner(forward=True, firewall=False)
    lan.enable(tmp_path, runner, Engine(), adopt=True)
    script = _decoded(runner.elevations[0])
    assert "advfirewall" in script
    assert "portproxy" not in script, "the forward was already there"


def test_a_dismissed_prompt_is_a_refusal_and_leaves_the_record_pending(tmp_path) -> None:
    runner = Runner(applies=False)
    engine = Engine()
    with pytest.raises(lan.LanError, match="lan_verification_failed"):
        lan.enable(tmp_path, runner, engine)
    assert engine.published == {}, "nothing is advertised that is not reachable"
    assert lan.read(tmp_path)["state"] == "pending", "durable intent survives"


def test_a_forward_with_no_rule_is_not_called_open(tmp_path) -> None:
    runner = Runner(forward=True, firewall=False, applies=False)
    with pytest.raises(lan.LanError, match="lan_verification_failed"):
        lan.enable(tmp_path, runner, Engine(), adopt=True)


def test_rows_on_a_public_only_network_are_reported_degraded(tmp_path) -> None:
    runner, engine = Runner(profile=PUBLIC), Engine()
    result = lan.enable(tmp_path, runner, engine)
    assert result["state"] == "degraded"
    assert result["remote_reachability"] == "blocked_by_windows"
    assert "admits nothing here" in result["detail"]
    assert engine.published["lan_advertise"] == [], "nothing unreachable is advertised"
    assert "crucible lan enable" in result["next"]
    assert "Private" in result["next"]


def test_disable_withdraws_the_projection_before_it_removes_the_rows(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    order: list[str] = []
    engine.advertise = lambda field, authorities: order.append(f"advertise:{authorities}")
    real_run = runner.run

    def watched(argv, *, timeout_s, env=None):
        if "Start-Process" in " ".join(argv):
            order.append("netsh")
        return real_run(argv, timeout_s=timeout_s, env=env)

    runner.run = watched
    assert lan.disable(tmp_path, runner, engine)["state"] == "disabled"
    assert order == ["advertise:[]", "netsh"], "withdrawn first, then removed"
    assert not (tmp_path / lan.RECORD).exists()
    assert runner.forward is False and runner.firewall is False


def test_disable_that_does_not_remove_keeps_the_record_to_retry(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    runner.applies = False
    with pytest.raises(lan.LanError, match="lan_verification_failed"):
        lan.disable(tmp_path, runner, engine)
    assert (tmp_path / lan.RECORD).exists(), "still ours to clean up"


def test_disabling_what_was_never_enabled_says_so_and_touches_nothing(tmp_path) -> None:
    runner = Runner()
    assert lan.disable(tmp_path, runner, Engine()) == {"state": "disabled"}
    assert runner.calls == []


def test_status_is_degraded_when_the_address_moved(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    assert lan.status(tmp_path, runner, engine)["state"] == "configured"
    runner.addresses = ["192.168.68.207"]
    report = lan.status(tmp_path, runner, engine)
    assert report["state"] == "degraded"
    assert report["addresses_match"] is False
    assert "crucible lan reconcile" in report["next"]


def test_reconcile_republishes_the_new_address_without_a_prompt(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    runner.addresses = ["192.168.68.207"]
    before = len(runner.elevations)
    result = lan.reconcile(tmp_path, runner, engine=engine)
    assert result["state"] == "configured"
    assert engine.published["lan_advertise"] == ["192.168.68.207:7100"]
    assert len(runner.elevations) == before, "the rows were already there"


def test_reconcile_repairs_nothing_that_was_never_opted_into(tmp_path) -> None:
    assert lan.reconcile(tmp_path, Runner(), engine=Engine()) == {"state": "disabled"}


def test_a_corrupt_record_is_a_refusal_not_an_empty_door(tmp_path) -> None:
    (tmp_path / lan.RECORD).write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(CrucibleError, match="lan_record_invalid"):
        lan.read(tmp_path)


def test_a_failed_publish_records_that_the_port_IS_open(tmp_path) -> None:
    class Refusing(Engine):
        def advertise(self, field, authorities):
            raise lan.LanError("lan_publish_failed: this engine does not know that field")

    runner = Runner()
    with pytest.raises(lan.LanError, match="lan_publish_failed"):
        lan.enable(tmp_path, runner, Refusing())

    record = lan.read(tmp_path)
    assert record["state"] == "open", "the rows are there and the record says so"
    assert record["state"] != "pending", "understating an open port is the defect"
    assert record["authorities"], "it names what it opened"
    assert runner.forward is True and runner.firewall is True


def test_disable_cleans_up_after_a_failed_publish(tmp_path) -> None:
    class Refusing(Engine):
        def advertise(self, field, authorities):
            if getattr(self, "refuse", True):
                raise lan.LanError("lan_publish_failed: nope")

    runner, engine = Runner(), Refusing()
    with pytest.raises(lan.LanError):
        lan.enable(tmp_path, runner, engine)
    engine.refuse = False
    assert lan.disable(tmp_path, runner, engine)["state"] == "disabled"
    assert runner.forward is False and runner.firewall is False
    assert not (tmp_path / lan.RECORD).exists()


def test_inside_the_wsl_guest_it_names_windows_as_where_the_door_is(tmp_path, monkeypatch) -> None:
    from crucible import service

    monkeypatch.setattr(service, "in_wsl", lambda: True)
    runner = Runner()
    runner.platform = "linux"
    with pytest.raises(lan.LanError, match="lan_inside_wsl") as refused:
        lan.enable(tmp_path, runner, Engine())
    assert "PowerShell" in str(refused.value) and "Share on your network" in str(refused.value)
    assert "0.0.0.0" not in str(refused.value), "binding the NAT'd guest wide opens nothing"
    assert runner.calls == []


def test_a_public_network_is_named_BEFORE_the_prompt_when_nobody_can_answer(tmp_path) -> None:
    runner, said = Runner(profile=PUBLIC), []

    def say(line: str) -> None:
        said.append(f"{len(runner.elevations)}:{line}")

    lan.enable(tmp_path, runner, Engine(), say=say)
    warning = [line for line in said if "marked Public" in line]
    assert len(warning) == 1
    assert warning[0].startswith("0:"), "said before the administrator prompt, not after"
    assert "no other device can reach Crucible" in warning[0]
    assert "Network profile type: Private" in warning[0], "the Settings path to fix it by hand"
    assert "--make-private" in warning[0]


def test_a_public_network_the_person_marks_private_draws_no_warning(tmp_path) -> None:
    runner, said = Runner(profile=PUBLIC), []
    lan.enable(tmp_path, runner, Engine(), ask_private=lambda _public: True, say=said.append)
    assert not [line for line in said if "marked Public" in line]
    assert any("Set-NetConnectionProfile" in _decoded(argv) for argv in runner.elevations)


def test_a_private_network_draws_no_public_warning(tmp_path) -> None:
    said: list[str] = []
    lan.enable(tmp_path, Runner(), Engine(), say=said.append)
    assert not [line for line in said if "Public" in line]


def test_the_public_verdict_names_the_settings_path(tmp_path) -> None:
    result = lan.enable(tmp_path, Runner(profile=PUBLIC), Engine())
    assert landoor.PRIVATE_BY_HAND in result["next"]


class NativeEngine(Engine):

    def __init__(self) -> None:
        super().__init__(backend="llama-windows")

    def request(self, method="GET", path="settings", body=None):
        if path == "setup":
            return {"bind": "http://127.0.0.1:7100", "urls": ["http://127.0.0.1:7100"],
                    "config_path": r"C:\Crucible\config.toml"}
        return super().request(method, path, body)


class SilentEngine(Engine):

    def verify(self) -> None:
        raise lan.LanError("lan_engine_unreachable: GET /v1/ping: refused")


def _offer(tmp_path, engine, *, answer: bool | None, runner=None):
    runner = Runner() if runner is None else runner
    said: list[str] = []
    asked: list[str] = []

    def ask(question: str) -> bool:
        asked.append(question)
        return bool(answer)

    report = lan.offer(tmp_path, runner, engine, ask=None if answer is None else ask,
                       ask_private=None, say=said.append)
    return report, said, asked, runner


def test_offer_on_an_unshared_wsl_pc_says_what_opening_changes_and_asks(tmp_path) -> None:
    report, said, asked, runner = _offer(tmp_path, Engine(), answer=True)
    assert asked == [lan.OFFER_QUESTION]
    assert said[0].startswith("Only this PC can reach Crucible.")
    assert any(landoor.RULE_NAME in line and "administrator" in line for line in said)
    assert report["enabled"] is True and report["network"]["reachable"] is True
    assert len(runner.elevations) == 1, "the same enable, one prompt"
    assert said[-1].startswith("Other devices on the network reach Crucible at http://192.168.68.100:7100")


def test_offer_answered_no_changes_nothing_and_says_how_to_do_it_later(tmp_path) -> None:
    report, said, _asked, runner = _offer(tmp_path, Engine(), answer=False)
    assert report == {"asked": True, "enabled": False, "network": report["network"]}
    assert runner.elevations == [] and lan.read(tmp_path) is None
    assert said[-1].startswith("Nothing was changed.") and "crucible lan enable" in said[-1]


def test_offer_with_nobody_to_ask_only_tells(tmp_path) -> None:
    report, said, asked, runner = _offer(tmp_path, Engine(), answer=None)
    assert asked == [] and runner.calls == [] and report["asked"] is False
    assert report["network"]["command"] == "crucible lan enable"
    assert any("crucible lan enable" in line for line in said)


def test_offer_on_a_shared_pc_says_where_and_asks_nothing(tmp_path) -> None:
    lan.enable(tmp_path, Runner(), Engine())
    report, said, asked, _runner = _offer(tmp_path, Engine(), answer=True)
    assert asked == [] and report["network"]["reachable"] is True
    assert said == ["Other devices on the network reach Crucible at "
                    "http://192.168.68.100:7100, http://100.64.0.1:7100."]


def test_offer_before_the_linux_engine_runs_neither_asks_nor_gives_native_advice(tmp_path) -> None:
    report, said, asked, runner = _offer(tmp_path, NativeEngine(), answer=True)
    assert asked == [] and runner.calls == []
    assert said == [lan.NOT_MOVED_YET] and "0.0.0.0" not in said[0]


def test_offer_on_a_pc_that_keeps_its_windows_engine_says_how_that_engine_opens(tmp_path) -> None:
    (tmp_path / "config.toml").write_text('[orchestrator]\nwsl = "never"\n', encoding="utf-8")
    report, said, asked, _runner = _offer(tmp_path, NativeEngine(), answer=True)
    assert asked == [] and report["network"]["reachable"] is False
    assert 'host = "0.0.0.0"' in said[1] and "refuses" in said[1]
    assert report["network"]["command"] is None, "there is no one command for this engine"


def test_offer_with_the_engine_silent_says_it_cannot_tell_yet(tmp_path) -> None:
    report, said, asked, _runner = _offer(tmp_path, SilentEngine(), answer=True)
    assert asked == [] and said == [lan.ENGINE_SILENT] and report["enabled"] is False
