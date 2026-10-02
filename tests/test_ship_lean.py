from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHIP = REPO / "scripts/ship.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the shims are `#!/bin/bash` scripts; the suite runs inside WSL",
)


def git(where: Path, *argv: str) -> str:
    return subprocess.check_output(
        ["git", *argv],
        cwd=str(where),
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
    )


def _shim(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def fake_repo(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(origin)], check=True)

    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    (work / "crucible").mkdir()
    (work / "crucible/__init__.py").write_text('VERSION = "1.0.2"\n', encoding="utf-8")
    (work / "scripts/ship.sh").write_text(SHIP.read_text(encoding="utf-8"), encoding="utf-8")
    (work / "scripts/ship.sh").chmod(0o755)

    (work / "scripts/bump.py").write_text(
        "import pathlib, sys\n"
        "here = pathlib.Path(__file__).resolve().parents[1] / 'crucible/__init__.py'\n"
        "if sys.argv[1] == '--check':\n"
        "    print(here.read_text().split(chr(34))[1]); sys.exit(0)\n"
        "major, minor, patch = (int(p) for p in "
        "here.read_text().split('\"')[1].split('.'))\n"
        "assert sys.argv[1] == 'patch', sys.argv\n"
        "here.write_text('VERSION = \"%d.%d.%d\"\\n' % (major, minor, patch + 1))\n",
        encoding="utf-8",
    )
    _shim(work / "scripts/release.sh", 'echo "release.sh: cut"\n')
    _shim(
        work / "scripts/deploy.sh",
        'printf "%s\\n" "$@" > "$SHIP_TEST_DEPLOY_ARGV"\n'
        'echo "deploy: timing wsl 7"\n'
        'echo "deploy: timing mac 4"\n',
    )
    (work / "scripts/promote_release.py").write_text("", encoding="utf-8")

    git(work, "init", "--quiet", "-b", "main")
    git(work, "config", "user.email", "keeper@example.invalid")
    git(work, "config", "user.name", "keeper")
    git(work, "add", "-A")
    git(work, "commit", "--quiet", "-m", "the tree before the release")
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "--quiet", "origin", "main")
    git(work, "fetch", "--quiet", "origin", "main")
    return work


def run_ship(work: Path, *argv: str, argv_record: Path | None = None) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["SHIP_TEST_DEPLOY_ARGV"] = str(argv_record or (work / "deploy-argv"))
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    return subprocess.run(
        ["bash", str(work / "scripts/ship.sh"), *argv],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(work),
        env=environment,
    )


GONE = ["tests.sh", "--test", "envpacks", "gh run watch"]


@pytest.mark.parametrize("dead", GONE)
def test_the_deploy_path_does_not_mention_the_suite_or_the_ci_wait(dead: str) -> None:
    assert dead not in SHIP.read_text(encoding="utf-8"), (
        "ship.sh still mentions %r; under PHASE20 7 the deploy path runs zero "
        "tests and waits for no CI" % dead
    )


@pytest.mark.parametrize("dead", GONE)
def test_the_help_does_not_offer_the_suite_or_the_ci_wait(dead: str) -> None:
    done = subprocess.run(
        ["bash", str(SHIP), "--help"], capture_output=True, text=True, timeout=60, cwd=str(REPO)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert dead not in done.stdout, done.stdout


def test_a_dry_run_prints_a_row_for_every_step_it_ran(tmp_path: Path) -> None:
    work = fake_repo(tmp_path)
    done = run_ship(work, "patch", "--dry-run")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ship: where v1.0.3 went" in done.stdout, done.stdout
    rows = re.findall(r"^  (\S.*?)\s\s+(\d+m\d\ds|\d+s)$", done.stdout, re.MULTILINE)
    named = [name for name, _ in rows]
    assert len(named) == 2, done.stdout
    assert any("tree" in name for name in named), named
    assert any("version" in name for name in named), named
    banners = [line.strip("= ") for line in done.stdout.splitlines() if line.startswith("=== ")]
    assert named == banners, (named, banners)


def test_the_deploy_step_is_timed_per_machine_by_deploy_itself(tmp_path: Path) -> None:
    work = fake_repo(tmp_path)
    done = run_ship(work, "patch", "--deploy")
    assert done.returncode == 0, done.stdout + done.stderr
    assert re.search(r"^ +the machines: wsl +7s$", done.stdout, re.MULTILINE), done.stdout
    assert re.search(r"^ +the machines: mac +4s$", done.stdout, re.MULTILINE), done.stdout
    fleet = done.stdout.index("\n  the machines ")
    assert fleet < done.stdout.index("the machines: wsl"), done.stdout


def test_deploy_is_told_not_to_ask(tmp_path: Path) -> None:
    work = fake_repo(tmp_path)
    record = tmp_path / "deploy-argv"
    done = run_ship(work, "patch", "--deploy", argv_record=record)
    assert done.returncode == 0, done.stdout + done.stderr
    argv = record.read_text(encoding="utf-8").split()
    assert argv == ["--release", "1.0.3", "--yes"], argv


def test_without_deploy_the_command_to_run_is_printed_instead(tmp_path: Path) -> None:
    work = fake_repo(tmp_path)
    done = run_ship(work, "patch")
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (tmp_path / "deploy-argv").exists() and not (work / "deploy-argv").exists()
    assert "./scripts/deploy.sh --release 1.0.3" in done.stdout, done.stdout
    assert "adopt-crucible-release.mjs 1.0.3" in done.stdout, done.stdout


def test_a_failed_step_still_prints_the_table_and_marks_the_step_it_died_in(
    tmp_path: Path,
) -> None:
    work = fake_repo(tmp_path)
    _shim(work / "scripts/deploy.sh", 'echo "deploy: the mac refused"\nexit 1\n')
    git(work, "add", "-A")
    git(work, "commit", "--quiet", "-m", "a deploy that refuses")
    git(work, "push", "--quiet", "origin", "main")
    git(work, "fetch", "--quiet", "origin", "main")

    done = run_ship(work, "patch", "--deploy")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "ship: where v1.0.3 went" in done.stdout, done.stdout
    assert re.search(r"^  the machines — FAILED +\d+s$", done.stdout, re.MULTILINE), done.stdout
    assert re.search(r"^  the tree +\d+s$", done.stdout, re.MULTILINE), done.stdout


def test_a_run_that_dies_before_it_has_a_version_still_prints_its_table(
    tmp_path: Path,
) -> None:
    work = fake_repo(tmp_path)
    (work / "uncommitted").write_text("a dirty tree\n", encoding="utf-8")

    done = run_ship(work, "patch")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "the working tree is dirty" in done.stderr, done.stderr
    assert "ship: where this run went" in done.stdout, done.stdout
    assert re.search(r"^  the tree — FAILED +\d+s$", done.stdout, re.MULTILINE), done.stdout
