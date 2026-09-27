from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_SH = REPO / "scripts/tests.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the shims are `#!/bin/bash` scripts; the suite runs inside WSL",
)

INVENTED = {
    "tests/test_leases.py": "leases",
    "tests/test_settle.py": "leases",
    "tests/test_api_client.py": "leases",
    "tests/test_release_bits.py": "crucible/__init__.py",
    "tests/test_keepers.py": "scripts/keeper-tts-live.sh",
    "tests/test_alpha.py": "alpha",
    "tests/test_beta.py": "beta",
    "tests/test_gamma.py": "gamma",
    "tests/test_delta.py": "delta",
    "tests/test_epsilon.py": "epsilon",
}


def git(where: Path, *argv: str) -> str:
    return subprocess.check_output(
        ["git", *argv],
        cwd=str(where),
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
    )


@pytest.fixture
def work(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(origin)], check=True)

    repo = tmp_path / "work"
    for relative in ("scripts", "crucible", "crucible/envs/llm", "tests", "docs",
                     ".github/workflows"):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    (repo / "scripts/tests.sh").write_text(TESTS_SH.read_text(encoding="utf-8"), encoding="utf-8")
    (repo / "scripts/tests.sh").chmod(0o755)
    (repo / "scripts/keeper-tts-live.sh").write_text("#!/bin/bash\n: live\n", encoding="utf-8")
    (repo / "crucible/__init__.py").write_text('VERSION = "1.0.2"\n', encoding="utf-8")
    (repo / "crucible/leases.py").write_text("def hold():\n    return 1\n", encoding="utf-8")
    (repo / "crucible/envs/llm/cuda-linux.txt").write_text("vllm==0.9.0\n", encoding="utf-8")
    (repo / ".github/workflows/ci.yml").write_text("name: ci\n", encoding="utf-8")
    (repo / "docs/x.md").write_text("# x\n", encoding="utf-8")
    for relative, word in INVENTED.items():
        (repo / relative).write_text("# this one is about %s\n" % word, encoding="utf-8")

    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "python").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$TESTS_SH_PYTEST_ARGV"\n', encoding="utf-8"
    )
    (shims / "python").chmod(0o755)

    git(repo, "init", "--quiet", "-b", "main")
    git(repo, "config", "user.email", "keeper@example.invalid")
    git(repo, "config", "user.name", "keeper")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", "the tree at the last release")
    git(repo, "tag", "v1.0.0")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "--quiet", "origin", "main")
    git(repo, "push", "--quiet", "origin", "v1.0.0")
    return repo


def listing(work: Path) -> tuple[str, set[str]]:
    environment = dict(os.environ)
    environment.update(
        CRUCIBLE_PYTEST_PYTHON=str(work.parent / "shims/python"),
        TESTS_SH_PYTEST_ARGV=str(work.parent / "pytest-argv"),
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
    )
    done = subprocess.run(
        ["bash", str(work / "scripts/tests.sh"), "--list"],
        capture_output=True, text=True, timeout=120, cwd=str(work), env=environment,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (work.parent / "pytest-argv").exists(), "--list ran pytest"
    would: set[str] = set()
    for line in done.stdout.splitlines():
        if line.startswith("tests: would run:"):
            would.update(line.split(":", 2)[2].split())
    return done.stdout, would


def test_a_version_bump_selects_nothing(work: Path) -> None:
    (work / "crucible/__init__.py").write_text('VERSION = "1.0.3"\n', encoding="utf-8")
    out, would = listing(work)
    assert would == set(), out
    assert "nothing" in out, out


def test_a_real_change_to_a_module_selects_the_tests_that_name_it(work: Path) -> None:
    (work / "crucible/leases.py").write_text("def hold():\n    return 2\n", encoding="utf-8")
    out, would = listing(work)
    assert would == {"tests/test_leases.py", "tests/test_settle.py", "tests/test_api_client.py"}, out


def test_a_version_bump_beside_a_real_change_still_selects_the_real_change(work: Path) -> None:
    (work / "crucible/__init__.py").write_text('VERSION = "1.0.3"\n', encoding="utf-8")
    (work / "crucible/leases.py").write_text("def hold():\n    return 2\n", encoding="utf-8")
    out, would = listing(work)
    assert would == {"tests/test_leases.py", "tests/test_settle.py", "tests/test_api_client.py"}, out


@pytest.mark.parametrize(
    "path, body",
    [
        (".github/workflows/ci.yml", "name: ci\njobs: {}\n"),
        ("docs/x.md", "# x\n\nsomething about leases\n"),
        ("crucible/envs/llm/cuda-linux.txt", "vllm==0.9.1\n"),
    ],
)
def test_a_workflow_a_doc_and_a_recipe_select_nothing(work: Path, path: str, body: str) -> None:
    (work / path).write_text(body, encoding="utf-8")
    out, would = listing(work)
    assert would == set(), out
    assert "nothing" in out, out


def test_no_live_keeper_is_ever_selected(work: Path) -> None:
    (work / "scripts/keeper-tts-live.sh").write_text("#!/bin/bash\n: live, changed\n",
                                                     encoding="utf-8")
    out, would = listing(work)
    assert would == {"tests/test_keepers.py"}, out
    for selected in would:
        assert selected.startswith("tests/test_") and selected.endswith(".py"), out


def test_the_listing_gives_one_reason_per_selected_file(work: Path) -> None:
    (work / "crucible/leases.py").write_text("def hold():\n    return 2\n", encoding="utf-8")
    out, would = listing(work)
    reasons = {}
    for line in out.splitlines():
        if "  <- " in line:
            path, reason = line.strip().split("  <- ", 1)
            assert path not in reasons, "two reasons printed for %s" % path
            reasons[path] = reason
    assert set(reasons) == would, (reasons, would)
    assert all("leases" in reason for reason in reasons.values()), reasons


def test_all_is_still_there_and_says_what_it_is_for() -> None:
    text = TESTS_SH.read_text(encoding="utf-8")
    header = text[: text.index("set -uo pipefail")]
    assert "--all" in header
    assert "debugging" in header, "the header no longer says what --all is for"


def test_the_wide_list_is_gone() -> None:
    text = TESTS_SH.read_text(encoding="utf-8")
    for dead in ["is_wide", "everything=", "no test names it"]:
        assert dead not in text, "tests.sh still carries %r" % dead
