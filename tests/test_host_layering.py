from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import controller_client, protocol, wsl
from crucible.host import installwatch, outcome, retry
from crucible.host.state import EngineDecision, MoveState
from crucible.platform import powershell, runner

ROOT = Path(__file__).resolve().parents[1]

GENERATED_TABLE = ROOT / "crucible" / "platform" / "wsl_table.py"

HOST_MODULES_THE_NEUTRAL_ONES_MAY_LOAD = {"crucible.host", "crucible.host.errors", "crucible.host.wsl_states", "crucible.platform.wsl_table"}


def _crucible_sources(*skip: Path) -> list[Path]:
    return [path for path in (ROOT / "crucible").rglob("*.py") if path not in skip]


def test_the_neutral_modules_never_load_the_host_application() -> None:
    probe = (
        "import json, sys\n"
        "import crucible.lan, crucible.sharing, crucible.local, crucible.uninstall\n"
        "print(json.dumps(sorted(m for m in sys.modules if m == 'crucible.host' or m.startswith('crucible.host.'))))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True,
                          text=True, timeout=120, check=True)
    loaded = set(json.loads(done.stdout))
    assert "crucible.host.app" not in loaded
    assert loaded <= HOST_MODULES_THE_NEUTRAL_ONES_MAY_LOAD, loaded - HOST_MODULES_THE_NEUTRAL_ONES_MAY_LOAD


@pytest.mark.parametrize("header", [
    protocol.API_HEADER, protocol.ACT_HEADER, protocol.CLIENT_HEADER, protocol.HANDOVER_HEADER,
])
def test_protocol_is_the_one_python_source_of_each_header_name(header: str) -> None:
    spelled = [
        path.relative_to(ROOT).as_posix()
        for path in _crucible_sources(ROOT / "crucible" / "protocol.py")
        if header in path.read_text(encoding="utf-8")
    ]
    assert spelled == []


def test_the_package_reexports_the_protocol_constants() -> None:
    import crucible
    from crucible import inflight

    assert crucible.API_HEADER is protocol.API_HEADER
    assert crucible.API_VERSION == protocol.API_VERSION
    assert crucible.CLIENT_NAME_HEADER is protocol.CLIENT_HEADER
    assert inflight.ACT_HEADER is protocol.ACT_HEADER
    assert protocol.user_agent("cli") == f"crucible-cli/{crucible.VERSION}"


@pytest.mark.parametrize("spelling", [wsl.GUEST_HOME, "$HOME/.crucible", f'"{wsl.WSL_EXE}"'])
def test_wsl_is_the_one_place_the_guest_home_and_wsl_exe_are_spelled(spelling: str) -> None:
    spelled = [
        path.relative_to(ROOT).as_posix()
        for path in _crucible_sources(ROOT / "crucible" / "wsl.py", GENERATED_TABLE)
        if spelling in path.read_text(encoding="utf-8")
    ]
    assert spelled == []


def test_the_crucible_distro_is_entered_as_its_own_user_and_others_as_their_default() -> None:
    assert wsl.guest_argv(wsl.CRUCIBLE_DISTRO, ["true"])[:6] == [
        "wsl.exe", "-d", wsl.CRUCIBLE_DISTRO, "-u", wsl.GUEST_USER, "--exec"]
    assert wsl.guest_argv("Ubuntu", ["true"]) == ["wsl.exe", "-d", "Ubuntu", "--exec", "true"]
    assert wsl.root_argv("Ubuntu", ["id"]) == ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--exec", "id"]


def test_one_parser_reads_both_the_verbose_and_the_quiet_distro_list() -> None:
    verbose = "  NAME      STATE           VERSION\n* Ubuntu    Running         2\n  crucible  Stopped         2\n"
    assert wsl.parse_distro_list(verbose) == ["Ubuntu", "crucible"]
    assert wsl.parse_distro_list("Ubuntu\ncrucible\n") == ["Ubuntu", "crucible"]
    assert wsl.parse_distro_list("U\x00b\x00u\x00n\x00t\x00u\x00\n") == ["Ubuntu"]
    assert wsl.parse_distro_list("Usage: wsl.exe [Argument]\n  --install:\n") == []


def test_there_is_one_runas_builder() -> None:
    assert powershell.runas_argv(["wsl.exe", "--install"])[-1] == (
        "Start-Process -Verb RunAs -Wait -FilePath 'wsl.exe' -ArgumentList '--install'"
    )
    assert powershell.runas_argv(["x.exe"])[-1] == "Start-Process -Verb RunAs -Wait -FilePath 'x.exe'"


def test_every_controller_start_waits_the_same_deadline() -> None:
    assert retry.CONTROLLER_START_SECONDS == controller_client.START_SECONDS
    assert installwatch.CONTROLLER_START_SECONDS == controller_client.START_SECONDS


def test_a_foreign_answer_on_the_door_is_refused_before_anything_is_spawned(tmp_path: Path) -> None:
    def foreign(url: str, **_kw: object) -> dict:
        return {"crucible": True, "role": "server"}

    def spawn(_home: Path) -> None:
        pytest.fail("a controller must not be spawned over a port someone else holds")

    with pytest.raises(controller_client.LocalError, match="wrong_controller"):
        controller_client.ensure_running(
            tmp_path, up=lambda: controller_client.answering(send=foreign), spawn=spawn)


def test_the_bearer_falls_back_from_the_pairing_file_to_the_config(tmp_path: Path) -> None:
    assert controller_client.bearer(tmp_path) is None
    (tmp_path / "config.toml").write_text('[auth]\ntoken = "from-config"\n', encoding="utf-8")
    assert controller_client.bearer(tmp_path) == "from-config"
    (tmp_path / "pairing").write_text("crucible://engine@127.0.0.1:7100/#from-pairing\n", encoding="utf-8")
    assert controller_client.bearer(tmp_path) == "from-pairing"


def test_the_outcome_state_is_an_enum_and_the_file_keeps_its_strings(tmp_path: Path) -> None:
    written = outcome.write(tmp_path, state="reboot-pending", release="1.0.0", attempts=1)
    assert written.state is MoveState.REBOOT_PENDING
    assert json.loads(outcome.path(tmp_path).read_text(encoding="utf-8"))["state"] == "reboot-pending"
    assert outcome.read(tmp_path).state is MoveState.REBOOT_PENDING
    assert outcome.classify(outcome.REBOOT_BUDGET_SPENT_CODE) is MoveState.CANNOT
    assert outcome.classify(outcome.REBOOT_STILL_OWED_CODE) is MoveState.REBOOT_PENDING
    assert outcome.RESTART_BANNER_CODES == {
        "wsl_reboot_required", "wsl_reboot_still_owed", "wsl_reboot_again"}
    assert EngineDecision.of(MoveState.CANNOT) == "cannot"


def test_a_stream_that_outlives_its_budget_is_asked_to_stop_and_never_killed() -> None:
    class Child:
        pid = 4242
        asked = False

        def terminate(self) -> None:
            self.asked = True

        def kill(self) -> None:
            pytest.fail("a stream may hold the GPU through wsl.exe; it is never killed")

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired("wsl.exe", timeout)

    child = Child()
    said = runner._ask_to_stop(child, 5)
    assert child.asked
    assert "pid 4242" in said and "Nothing was force-killed" in said


def test_a_paired_engine_needs_no_label_to_be_built() -> None:
    from crucible import sharing

    assert sharing.Engine is sharing.PairedEngine
    assert sharing.PairedEngine.__init__.__defaults__ == ("sharing",)
