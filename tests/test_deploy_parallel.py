"""`deploy.sh` installs the fleet AT ONCE, and says which machine said what.

THE MEASUREMENT, 2026-09-18. The three machines were upgraded one after
another, each with up to sixty seconds of `await_release` polling behind it, so
a release whose actual installing was about ninety seconds spent several minutes
of wall-clock in a queue. The machines share nothing — `wsl` and `windows` are
two installs on one box and `mac` is at the end of an ssh — so the queue bought
nothing at all.

THESE TESTS RUN THE REAL SCRIPT. Every other release-tooling test in this repo
is a string check over the source, and a string check cannot tell a `&` that
forks from a `&` that forks and is then immediately waited on. So the fleet is
faked at the only four places `deploy.sh` touches the world — `wsl.exe`, `ssh`,
`powershell.exe` and `curl`, shimmed onto PATH — and each shim sleeps for its
machine's own time and then writes the `installation.json` the script reads
back. Nothing here reaches the network, WSL, the Mac, the tray or a GPU.
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

#: THE SPREAD IS THE POINT. One slow machine and two instant ones: in a queue
#: the run cannot finish before the slow one plus the two reads behind it, and
#: in parallel it cannot take longer than the slow one by much. SLOW is the
#: only thing in the whole run that takes measurable time.
SLOW_SECONDS = 3

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the shims are `#!/bin/bash` scripts named wsl.exe/ssh/powershell.exe; "
    "the suite runs inside WSL, where PATH resolves them",
)


def _shim(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def fleet(tmp_path: Path, before: str, want: str) -> dict[str, str]:
    """Three fake machines, and the environment `deploy.sh` meets them through.

    Each machine's `installation.json` is a real file on disk, because that is
    the one fact `deploy.sh` reads before and after — faking the READING would
    fake the thing under test.
    """
    state = tmp_path / "state"
    state.mkdir()
    local = tmp_path / "localappdata" / "crucible"
    local.mkdir(parents=True)
    for where in (
        state / "wsl.json",
        state / "mac.json",
        # `read_windows` reads $LOCALAPPDATA/crucible/installation.json and no
        # other path, so the Windows machine's record IS that file.
        local / "installation.json",
    ):
        where.write_text(json.dumps({"release": before}), encoding="utf-8")

    shims = tmp_path / "shims"
    shims.mkdir()
    # ONE installer body, three callers. It sleeps for its machine's own time,
    # prints progress the way a real installer does — several lines, over the
    # whole of its run — and then writes the record `crucible local publish`
    # writes when the runtime comes back up.
    _shim(
        shims / "fake-install",
        'machine="$1"; record="$2"; naps="$3"\n'
        # ONE APPEND-ONLY ORDER FILE, and no clock anywhere. Three installers
        # that take 3s, 0s and 0s take three seconds in a queue and three
        # seconds at once, so duration cannot tell the two apart — but ORDER
        # can: a queue cannot start the second machine until the first has
        # finished, so "every machine started before the slow one finished" is
        # true only of a fan-out. Timestamps were the first attempt and are not
        # available here: the WSL2 guest clock resyncs with its Windows host,
        # and three `sleep 1` calls were measured spanning 1.504s of it on
        # 2026-09-18.
        'echo "$machine start" >> "$CRUCIBLE_TEST_STATE/order"\n'
        'if [ "$CRUCIBLE_TEST_FAIL" = "$machine" ]; then\n'
        '  echo "$machine: the installer failed" >&2; exit 1\n'
        'fi\n'
        'i=0\n'
        'while [ "$i" -lt "$naps" ]; do\n'
        '  echo "$machine: step $i of $naps"\n'
        '  sleep 1\n'
        '  i=$(( i + 1 ))\n'
        'done\n'
        'printf \'{"release": "%s"}\\n\' "$CRUCIBLE_TEST_RELEASE" > "$record"\n'
        'echo "$machine end" >> "$CRUCIBLE_TEST_STATE/order"\n'
        'echo "$machine: installed $CRUCIBLE_TEST_RELEASE"\n',
    )

    _shim(
        shims / "wsl.exe",
        # `read_wsl` runs `bash -c`, `install_wsl` runs `bash -lc`. The flag is
        # what tells them apart, and it is the script's own choice, not ours.
        'for a in "$@"; do\n'
        '  if [ "$a" = "-lc" ]; then\n'
        '    exec fake-install wsl "$CRUCIBLE_TEST_STATE/wsl.json" "$CRUCIBLE_TEST_SLOW"\n'
        '  fi\n'
        'done\n'
        'last="${@: -1}"\n'
        'case "$last" in\n'
        '  *installation.json*) cat "$CRUCIBLE_TEST_STATE/wsl.json" 2>/dev/null ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n',
    )
    _shim(
        shims / "ssh",
        'last="${@: -1}"\n'
        'case "$last" in\n'
        '  *installation.json*) cat "$CRUCIBLE_TEST_STATE/mac.json" 2>/dev/null ;;\n'
        '  true) exit 0 ;;\n'
        '  *) exec fake-install mac "$CRUCIBLE_TEST_STATE/mac.json" 0 ;;\n'
        'esac\n',
    )
    _shim(
        shims / "powershell.exe",
        'exec fake-install windows "$LOCALAPPDATA/crucible/installation.json" 0\n',
    )
    # `install_windows` fetches install.ps1 to a file before running it.
    # Nothing here reads that file; only its existence is required.
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
        CRUCIBLE_TEST_SLOW=str(SLOW_SECONDS),
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
        timeout=180,
        env=environment,
        cwd=str(REPO),
        input=answer,
    )


def test_the_slow_machine_does_not_delay_the_other_two(tmp_path: Path) -> None:
    """THE POINT OF THE WHOLE THING.

    `wsl` is first in the fleet and is the slow one, so in a queue nothing else
    can even START until it has finished. That ordering is the whole difference
    and it needs no clock: every machine's start must appear before the slow
    machine's end.

    The wall-clock ceiling below is a sanity bound and nothing more — 3s + 0s +
    0s is three seconds whichever way it is run — which is exactly why the
    ordering is what the assertion turns on.
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    began = time.monotonic()
    done = run_deploy(environment, "1.0.3", "--yes")
    elapsed = time.monotonic() - began
    assert done.returncode == 0, done.stdout + done.stderr
    assert elapsed < SLOW_SECONDS * 2, (
        "the fleet took %.1fs for one %ds install and two instant ones"
        % (elapsed, SLOW_SECONDS)
    )
    order = (tmp_path / "state/order").read_text(encoding="utf-8").split("\n")
    order = [line for line in order if line]
    assert sorted(order) == sorted(
        ["%s %s" % (m, w) for m in ("wsl", "windows", "mac") for w in ("start", "end")]
    ), order
    slow_finished = order.index("wsl end")
    for machine in ("windows", "mac"):
        assert order.index(machine + " start") < slow_finished, (
            "%s did not start until the slow machine had finished, which is a "
            "queue: %r" % (machine, order)
        )
    for machine in ("wsl", "windows", "mac"):
        assert machine + ": now runs 1.0.3" in done.stdout, done.stdout


def test_every_line_says_which_machine_it_came_from(tmp_path: Path) -> None:
    """Three installers writing to one terminal at once is unreadable without it.

    Including the PROGRESS lines, which are most of what an installer prints and
    which arrive interleaved with the other two machines' — an unprefixed
    progress line is a line whose machine can only be guessed at.
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    for machine in ("wsl", "windows", "mac"):
        # The installer's OWN output, prefixed — not only deploy.sh's summary.
        assert "%s: %s: installed 1.0.3" % (machine, machine) in done.stdout, done.stdout
    # And every one of the slow machine's progress lines, in order, each with
    # its own prefix rather than the first one carrying a paragraph.
    for i in range(SLOW_SECONDS):
        assert "wsl: wsl: step %d of %d" % (i, SLOW_SECONDS) in done.stdout, done.stdout


def test_one_machine_failing_does_not_stop_the_others(tmp_path: Path) -> None:
    """A fleet is not a transaction.

    The two that can be upgraded are, and the summary names the one that was
    not — which is why a verdict is collected per machine rather than taken
    from whichever background job finished last, a status that cannot say which
    machine it belonged to.
    """
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    environment["CRUCIBLE_TEST_FAIL"] = "windows"
    done = run_deploy(environment, "1.0.3", "--yes")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "windows(installer failed)" in done.stderr, done.stderr
    assert "wsl: now runs 1.0.3" in done.stdout, done.stdout
    assert "mac: now runs 1.0.3" in done.stdout, done.stdout
    record = json.loads((tmp_path / "state/mac.json").read_text(encoding="utf-8"))
    assert record["release"] == "1.0.3"


def test_only_still_names_the_machines_to_touch(tmp_path: Path) -> None:
    """Forking the fleet must not fork the machines nobody asked for."""
    environment = fleet(tmp_path, "1.0.2", "1.0.3")
    done = run_deploy(environment, "1.0.3", "--only", "wsl,mac", "--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "wsl: now runs 1.0.3" in done.stdout, done.stdout
    assert "mac: now runs 1.0.3" in done.stdout, done.stdout
    assert "windows" not in done.stdout, done.stdout
    untouched = json.loads(
        (tmp_path / "localappdata/crucible/installation.json").read_text(encoding="utf-8")
    )
    assert untouched["release"] == "1.0.2", "windows was installed after --only left it out"


def test_the_restart_is_confirmed_once_for_the_whole_fleet(tmp_path: Path) -> None:
    """One question, asked before the fan-out, not one per machine.

    Three subshells cannot take turns at a prompt: they share one stdin, and
    whichever read first would answer for the others. So the confirmation stays
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
    record = json.loads((tmp_path / "state/wsl.json").read_text(encoding="utf-8"))
    assert record["release"] == "1.0.2"


def test_each_machine_reports_what_it_cost(tmp_path: Path) -> None:
    """`ship.sh`'s final table has a row per machine and does not time them itself.

    deploy.sh is what knows: it forked the installs and it joined them, and a
    caller timing the whole call would only ever learn the slowest one. So the
    number is printed here, in a line shaped to be read back, and ship.sh folds
    those lines into its table rather than measuring a second time.

    WHAT IS ASSERTED IS THE SHAPE, not the magnitude: a row per machine, whole
    seconds, parseable by the `sed` in ship.sh. The magnitude cannot be
    asserted from inside WSL2, whose guest clock resyncs with its Windows host —
    three `sleep 1` calls were measured spanning 1.504s of it on 2026-09-18,
    which is what made the first version of this assertion flap.
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
    assert sorted(timed) == ["mac", "windows", "wsl"], done.stdout
    assert timed["wsl"] >= timed["windows"] and timed["wsl"] >= timed["mac"], (
        timed, done.stdout)


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
