from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "scripts/deploy.sh"

GUEST_SECONDS = 3

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the shims are `#!/bin/bash` scripts named wsl.exe/ssh/powershell.exe; "
    "the suite runs inside WSL, where PATH resolves them",
)


def _shim(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def fleet(tmp_path: Path, before: str, want: str) -> dict[str, str]:
    state = tmp_path / "state"
    state.mkdir()
    local = tmp_path / "localappdata" / "crucible"
    local.mkdir(parents=True)
    for where in (
        state / "guest.json",
        state / "mac.json",
        local / "installation.json",
    ):
        where.write_text(json.dumps({"release": before}), encoding="utf-8")

    shims = tmp_path / "shims"
    shims.mkdir()

    _shim(
        shims / "powershell.exe",
        'echo "pc start" >> "$CRUCIBLE_TEST_STATE/order"\n'
        'if [ "$CRUCIBLE_TEST_FAIL" = "pc" ]; then\n'
        '  echo "pc: install.ps1 refused" >&2\n'
        '  echo "pc end" >> "$CRUCIBLE_TEST_STATE/order"\n'
        '  exit 1\n'
        'fi\n'
        'echo "pc: unpacking the host pack"\n'
        'echo "pc: starting crucible host"\n'
        'printf \'{"release": "%s"}\\n\' "$CRUCIBLE_TEST_RELEASE" \\\n'
        '  > "$LOCALAPPDATA/crucible/installation.json"\n'
        'if [ "$CRUCIBLE_TEST_GUEST_STUCK" != "1" ]; then\n'
        '  (\n'
        '    sleep "$CRUCIBLE_TEST_GUEST_DELAY"\n'
        '    printf \'{"release": "%s"}\\n\' "$CRUCIBLE_TEST_RELEASE" \\\n'
        '      > "$CRUCIBLE_TEST_STATE/guest.json"\n'
        '    echo "pc guest-record" >> "$CRUCIBLE_TEST_STATE/order"\n'
        '  ) >/dev/null 2>&1 &\n'
        'fi\n'
        'echo "pc end" >> "$CRUCIBLE_TEST_STATE/order"\n'
        'echo "pc: the host has it from here"\n',
    )

    _shim(
        shims / "wsl.exe",
        'last="${@: -1}"\n'
        'case "$last" in\n'
        '  *installation.json*) cat "$CRUCIBLE_TEST_STATE/guest.json" 2>/dev/null ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n',
    )
    _shim(
        shims / "ssh",
        'last="${@: -1}"\n'
        'case "$last" in\n'
        '  *installation.json*) cat "$CRUCIBLE_TEST_STATE/mac.json" 2>/dev/null ;;\n'
        '  true) exit 0 ;;\n'
        '  *)\n'
        '    echo "mac start" >> "$CRUCIBLE_TEST_STATE/order"\n'
        '    if [ "$CRUCIBLE_TEST_FAIL" = "mac" ]; then\n'
        '      echo "mac: the installer failed" >&2\n'
        '      echo "mac end" >> "$CRUCIBLE_TEST_STATE/order"\n'
        '      exit 1\n'
        '    fi\n'
        '    echo "mac: installing"\n'
        '    printf \'{"release": "%s"}\\n\' "$CRUCIBLE_TEST_RELEASE" \\\n'
        '      > "$CRUCIBLE_TEST_STATE/mac.json"\n'
        '    echo "mac end" >> "$CRUCIBLE_TEST_STATE/order"\n'
        '    echo "mac: installed $CRUCIBLE_TEST_RELEASE"\n'
        '    ;;\n'
        'esac\n',
    )
    _shim(
        shims / "curl",
        'while [ $# -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then : > "$2"; exit 0; fi\n'
        '  shift\n'
        'done\n'
        'exit 0\n',
    )

    environment = dict(os.environ)
    environment.update(
        PATH=str(shims) + os.pathsep + environment["PATH"],
        LOCALAPPDATA=str(tmp_path / "localappdata"),
        TMPDIR=str(tmp_path),
        CRUCIBLE_TEST_STATE=str(state),
        CRUCIBLE_TEST_RELEASE=want,
        CRUCIBLE_TEST_FAIL="",
        CRUCIBLE_TEST_GUEST_DELAY=str(GUEST_SECONDS),
        CRUCIBLE_TEST_GUEST_STUCK="0",
    )
    return environment


def run_deploy(
    environment: dict[str, str],
    want: str,
    *extra: str,
    answer: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(DEPLOY), "--release", want, *extra],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
        cwd=str(REPO),
        input=answer,
    )


def marks(tmp_path: Path) -> list[str]:
    path = tmp_path / "state/order"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


def test_deploy_never_drives_the_guest_itself() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "install_wsl" not in code, "deploy.sh installs into the guest again"
    for line in code.splitlines():
        if "wsl.exe" not in line:
            continue
        assert "installation.json" in line or "exit 0" in line, (
            "deploy.sh reaches into the distro for something other than its "
            "record: %s" % line.strip()
        )


def test_the_pc_is_not_done_until_both_of_its_records_name_the_release(
    tmp_path: Path,
) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pc: now runs 1.0.3" in done.stdout, done.stdout
    order = marks(tmp_path)
    assert order.index("pc end") < order.index("pc guest-record"), order
    guest = json.loads((tmp_path / "state/guest.json").read_text(encoding="utf-8"))
    assert guest["release"] == "1.0.3"


def test_a_pc_whose_guest_never_moves_is_reported_by_name(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    environment["CRUCIBLE_TEST_GUEST_STUCK"] = "1"
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "pc(reports:host:1.0.3 guest:1.0.2)" in done.stderr, done.stderr
    assert "pc: now runs" not in done.stdout, done.stdout


@pytest.mark.parametrize(
    "name, expected",
    [("wsl", "the host drives the guest"), ("windows", "one machine now, called pc")],
)
def test_the_old_two_names_are_refused_by_name(
    tmp_path: Path, name: str, expected: str
) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--only", name, "--yes")
    assert done.returncode != 0, done.stdout + done.stderr
    assert expected in done.stderr, done.stderr
    assert "--only pc" in done.stderr, done.stderr
    assert marks(tmp_path) == [], "a refused name still touched a machine"


def test_the_mac_does_not_wait_for_the_pc(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    began = time.monotonic()
    done = run_deploy(environment, "1.0.3", "--yes")
    elapsed = time.monotonic() - began
    assert done.returncode == 0, done.stdout + done.stderr
    order = marks(tmp_path)
    assert order.index("mac start") < order.index("pc guest-record"), (
        "the mac did not start until the PC was finished, which is a queue: %r" % order
    )
    assert elapsed < GUEST_SECONDS * 4, "the fleet took %.1fs" % elapsed
    for machine in ("pc", "mac"):
        assert machine + ": now runs 1.0.3" in done.stdout, done.stdout


def test_every_line_says_which_machine_it_came_from(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    for line in (
        "pc: pc: unpacking the host pack",
        "pc: pc: starting crucible host",
        "pc: pc: the host has it from here",
        "mac: mac: installing",
        "mac: mac: installed 1.0.3",
    ):
        assert line in done.stdout, done.stdout


def test_one_machine_failing_does_not_stop_the_other(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    environment["CRUCIBLE_TEST_FAIL"] = "pc"
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "pc(installer failed)" in done.stderr, done.stderr
    assert "mac: now runs 1.0.3" in done.stdout, done.stdout
    record = json.loads((tmp_path / "state/mac.json").read_text(encoding="utf-8"))
    assert record["release"] == "1.0.3"


def test_only_names_the_machines_to_touch(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--only", "pc", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pc: now runs 1.0.3" in done.stdout, done.stdout
    assert "mac:" not in done.stdout, done.stdout
    assert "timing mac" not in done.stdout, done.stdout
    assert marks(tmp_path) == ["pc start", "pc end", "pc guest-record"], marks(tmp_path)
    untouched = json.loads((tmp_path / "state/mac.json").read_text(encoding="utf-8"))
    assert untouched["release"] == "1.0.2", "the mac was installed after --only left it out"


def test_the_restart_is_confirmed_once_for_the_whole_fleet(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", answer="y\n")
    assert done.returncode == 0, done.stdout + done.stderr
    both = done.stdout + done.stderr
    assert both.count("[y/N]") == 1, both


def test_a_refused_confirmation_touches_nothing(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", answer="n\n")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "nothing was changed" in done.stdout, done.stdout
    assert marks(tmp_path) == [], marks(tmp_path)


def test_each_machine_reports_what_it_cost(tmp_path: Path) -> None:
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    timed = {}
    for line in done.stdout.splitlines():
        if line.startswith("deploy: timing "):
            _, _, machine, seconds = line.split()
            assert seconds.isdigit(), line
            timed[machine] = int(seconds)
    assert sorted(timed) == ["mac", "pc"], done.stdout


def test_await_release_polls_every_two_seconds_with_the_same_ceiling() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    start = text.index("await_release()")
    body = text[start:text.index("# ------", start)]
    assert "sleep 2" in body, "the poll interval is no longer two seconds"
    assert "-lt 30" in body, (
        "thirty attempts two seconds apart is the same sixty-second ceiling "
        "that twenty attempts three seconds apart was"
    )
