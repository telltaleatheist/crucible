"""`deploy.sh` installs two machines at once, and the PC is not one of them twice.

THE MEASUREMENT, 2026-09-18. The machines were upgraded one after another, each
with up to sixty seconds of `await_release` polling behind it, so a release whose
actual installing was about ninety seconds spent several minutes of wall-clock in
a queue. They share nothing — the PC is one box and `mac` is at the end of an
ssh — so the queue bought nothing at all.

THE OTHER HALF, ruled the same day. `wsl` and `windows` used to be two entries
here, and the `wsl` one curled `install.sh` straight into the guest. Owen:
*"windows is the driver; the thing moving wsl forward. use the established,
installed, functional system to drive the new one."* PHASE15-HOST.md 4.3 puts
the whole WSL sequence behind the host's door — "ONE implementation of the
sequence, the host's" — so a second driver from out here was the race. The PC is
one entry and one install now, and its VERDICT reads both of the machine's
records: the host's and the guest's.

THESE TESTS RUN THE REAL SCRIPT. Every other release-tooling test in this repo
is a string check over the source, and a string check cannot tell a `&` that
forks from a `&` that forks and is then immediately waited on, nor a verdict
that reads two records from one that reads one and hopes. So the fleet is faked
at the only four places `deploy.sh` touches the world — `wsl.exe`, `ssh`,
`powershell.exe` and `curl`, shimmed onto PATH. Nothing here reaches the
network, WSL, the Mac, the tray or a GPU.
"""

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

#: How long the fake host takes to carry the fake guest forward. The PC's
#: verdict must not be reached before this, and the Mac must not be held up by
#: it — those are two of the assertions below and this is the gap they need.
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
    """Two fake machines, and the environment `deploy.sh` meets them through.

    THREE records, though, because the PC has two of them: the host's on
    Windows and the engine's inside the distro. Each is a real file on disk,
    because reading them is the thing under test — faking the READING would
    fake the answer.
    """
    state = tmp_path / "state"
    state.mkdir()
    local = tmp_path / "localappdata" / "crucible"
    local.mkdir(parents=True)
    for where in (
        # `record_guest` asks wsl.exe; `read_mac` asks ssh. Both shims cat these.
        state / "guest.json",
        state / "mac.json",
        # `record_host` reads $LOCALAPPDATA/crucible/installation.json and no
        # other path, so the host's record IS that file.
        local / "installation.json",
    ):
        where.write_text(json.dumps({"release": before}), encoding="utf-8")

    shims = tmp_path / "shims"
    shims.mkdir()

    # THE FAKE install.ps1. The real one unpacks the host pack, starts
    # `crucible host` and RETURNS (PHASE15-HOST.md 4.4) — the guest is carried
    # afterwards, by the host, on its own clock. So this writes the host record
    # and returns at once, and the guest record lands later from a detached
    # subshell whose output goes to /dev/null: left on the pipe it would hold
    # the prefixer open long after the installer had finished.
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
        # `record_guest` cats the guest's installation.json, and probes the
        # distro with `exit 0` when there is nothing to read. Nothing here
        # installs: deploy.sh no longer drives the guest at all.
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
    # `install_pc` fetches install.ps1 to a file before running it. Nothing here
    # reads that file; only its existence is required.
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
    """The order file, in the order it was written."""
    path = tmp_path / "state/order"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


# ------------------------------------------------------------- the PC is one


def test_deploy_never_drives_the_guest_itself() -> None:
    """The host is the driver; a second one out here is the race.

    PHASE15-HOST.md 4.3: the WSL sequence is behind the host's own door, "ONE
    implementation of the sequence, the host's; the bootstrap is its client".
    deploy.sh runs install.ps1 and nothing else on that box — it may READ the
    guest's record, and it may not install into it.
    """
    text = DEPLOY.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "install_wsl" not in code, "deploy.sh installs into the guest again"
    # Every line that reaches the distro at all, and what it is allowed to do
    # there: read a file. An install payload, a login shell, anything that
    # fetches — that is the host's job, through its own door.
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
    """One machine, two records, one verdict.

    install.ps1 returns as soon as the host is up; the guest follows on the
    host's clock. A verdict taken from the host record alone would call the PC
    done while the engine was still on the old release — which is exactly the
    drift this script was written for (0.6.3 beside 0.6.5 on one box).
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pc: now runs 1.0.3" in done.stdout, done.stdout
    order = marks(tmp_path)
    # The installer had finished long before the verdict could be reached, so a
    # run that did not WAIT for the second record cannot have passed above.
    assert order.index("pc end") < order.index("pc guest-record"), order
    guest = json.loads((tmp_path / "state/guest.json").read_text(encoding="utf-8"))
    assert guest["release"] == "1.0.3"


def test_a_pc_whose_guest_never_moves_is_reported_by_name(tmp_path: Path) -> None:
    """AND THIS IS WHAT A REAL DEPLOY DOES TODAY, which is why it is a keeper.

    The host does not carry the guest forward on its own after install.ps1
    restarts it: `crucible/host/app.py:1175` gives the sequence to the door and
    nothing in `main()` calls it, and on a machine the guest already owns
    `app.py:1221` takes the `Owner.WSL_UNIT` branch straight to
    `walk._complete()`, which never runs `_guest_install`. So the half-upgraded
    PC is the expected state, and the only acceptable behaviour is to say so:
    by machine name, with both halves in the line, and a non-zero exit.

    This one costs the full `await_release` ceiling — sixty seconds — because
    waiting out the ceiling IS the behaviour being asserted.
    """
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
    """`--only wsl` used to be a machine and is now a mistake.

    Silently matching nothing would be a run that installs nothing and says it
    is finished, which is the failure mode this whole script exists to end.
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--only", name, "--yes")
    assert done.returncode != 0, done.stdout + done.stderr
    assert expected in done.stderr, done.stderr
    assert "--only pc" in done.stderr, done.stderr
    assert marks(tmp_path) == [], "a refused name still touched a machine"


# ----------------------------------------------------------- and two lanes


def test_the_mac_does_not_wait_for_the_pc(tmp_path: Path) -> None:
    """THE POINT OF THE PARALLELISM.

    The PC's verdict cannot land until its guest record does, seconds after its
    installer returned. In a queue the Mac could not START until then. It needs
    no clock to say so: `mac start` must appear before `pc guest-record`.
    """
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
    """Two installers writing to one terminal at once is unreadable without it.

    Including the PROGRESS lines, which are most of what an installer prints
    and which arrive interleaved with the other machine's — an unprefixed
    progress line is a line whose machine can only be guessed at.
    """
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
    """A fleet is not a transaction.

    The one that can be upgraded is, and the summary names the one that was
    not — which is why a verdict is collected per machine rather than taken
    from whichever background job finished last, a status that cannot say which
    machine it belonged to.
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    environment["CRUCIBLE_TEST_FAIL"] = "pc"
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "pc(installer failed)" in done.stderr, done.stderr
    assert "mac: now runs 1.0.3" in done.stdout, done.stdout
    record = json.loads((tmp_path / "state/mac.json").read_text(encoding="utf-8"))
    assert record["release"] == "1.0.3"


def test_only_names_the_machines_to_touch(tmp_path: Path) -> None:
    """Forking the fleet must not fork the machine nobody asked for."""
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--only", "pc", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pc: now runs 1.0.3" in done.stdout, done.stdout
    # NOT a bare search for "mac": the word is inside "machine", which this
    # script says on almost every line, so the assertion passed against the
    # very output it was written to check. The Mac's own lines are what matter.
    assert "mac:" not in done.stdout, done.stdout
    assert "timing mac" not in done.stdout, done.stdout
    assert marks(tmp_path) == ["pc start", "pc end", "pc guest-record"], marks(tmp_path)
    untouched = json.loads((tmp_path / "state/mac.json").read_text(encoding="utf-8"))
    assert untouched["release"] == "1.0.2", "the mac was installed after --only left it out"


def test_the_restart_is_confirmed_once_for_the_whole_fleet(tmp_path: Path) -> None:
    """One question, asked before the fan-out, not one per machine.

    The subshells cannot take turns at a prompt: they share one stdin, and
    whichever read first would answer for the other. So the confirmation stays
    where it has always been — ahead of the work — and is asked exactly once.
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", answer="y\n")
    assert done.returncode == 0, done.stdout + done.stderr
    both = done.stdout + done.stderr
    assert both.count("[y/N]") == 1, both


def test_a_refused_confirmation_touches_nothing(tmp_path: Path) -> None:
    """The prompt is a gate, and a gate that forks anyway is decoration."""
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", answer="n\n")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "nothing was changed" in done.stdout, done.stdout
    assert marks(tmp_path) == [], marks(tmp_path)


def test_each_machine_reports_what_it_cost(tmp_path: Path) -> None:
    """`ship.sh`'s final table has a row per machine and does not time them itself.

    deploy.sh is what knows: it forked the installs and it joined them, and a
    caller timing the whole call would only ever learn the slowest one. So the
    number is printed here, in a line shaped to be read back, and ship.sh folds
    those lines into its table rather than measuring a second time.

    WHAT IS ASSERTED IS THE SHAPE, not the magnitude: a row per machine, whole
    seconds, parseable by the `sed` in ship.sh. The magnitude cannot be asserted
    from inside WSL2, whose guest clock resyncs with its Windows host — three
    `sleep 1` calls were measured spanning 1.504s of it on 2026-09-18, which is
    what made the first version of this assertion flap.
    """
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
    """The record is published by the RUNTIME, not by the installer, so the wait
    has to stay; what it must not do is spend three seconds asking again when
    the answer arrived after one. Sixty seconds either way."""
    text = DEPLOY.read_text(encoding="utf-8")
    start = text.index("await_release()")
    body = text[start:text.index("# ------", start)]
    assert "sleep 2" in body, "the poll interval is no longer two seconds"
    assert "-lt 30" in body, (
        "thirty attempts two seconds apart is the same sixty-second ceiling "
        "that twenty attempts three seconds apart was"
    )
