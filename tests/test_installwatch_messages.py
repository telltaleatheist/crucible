from __future__ import annotations

import io
import sys
from datetime import datetime, timezone
from pathlib import Path

from crucible.host import installwatch

ROOT = Path(__file__).resolve().parent.parent
SINCE = datetime(2026, 10, 8, 21, 30, 55, tzinfo=timezone.utc)


def _brief(home: Path) -> str:
    clock = iter(float(n) for n in range(1000))
    out = io.StringIO()
    code = installwatch.watch(
        home, SINCE, brief=True, out=out,
        clock=lambda: next(clock), sleep=lambda seconds: None,
    )
    assert code == 0
    return out.getvalue()


def test_the_hand_over_never_promises_an_app_that_may_not_exist(tmp_path: Path) -> None:
    text = " ".join(_brief(tmp_path).split())
    assert "the app you installed from" not in text
    assert "If an app started this install, that app shows the progress." in text
    assert "there is nothing to click" in text


def test_the_hand_over_says_where_to_follow_it_and_what_to_wait_for(tmp_path: Path) -> None:
    said = _brief(tmp_path)
    command = installwatch.follow_command(tmp_path, SINCE)
    assert f"  {command}" in said.splitlines(), "the line is printed whole, unwrapped, to be copied"
    assert command.startswith(f'& "{Path(sys.executable)}" -m crucible.host.installwatch ')
    assert f'--home "{tmp_path}"' in command
    assert command.endswith("--since 2026-10-08T21:30:55Z")
    text = " ".join(said.split())
    assert 'it starts with "Done."' in text
    assert "a Windows restart" in text
    assert '"Try again"' in text


def test_the_command_it_prints_is_one_its_own_parser_reads(tmp_path: Path, monkeypatch) -> None:
    import shlex

    words = shlex.split(installwatch.follow_command(tmp_path, SINCE), posix=False)
    seen: dict[str, object] = {}
    monkeypatch.setattr(installwatch, "watch", lambda home, since, brief: seen.update(home=home, since=since, brief=brief) or 0)
    assert installwatch.main([word.strip('"') for word in words[words.index("--home"):]]) == 0
    assert seen == {"home": tmp_path, "since": SINCE, "brief": False}


def test_a_console_reader_is_told_what_to_wait_for() -> None:
    assert installwatch.WAIT_FOR in installwatch.CONSOLE_START
    assert installwatch.DONE_SENTENCE.startswith("Done.")


def test_no_sentence_claims_live_progress_where_there_is_none() -> None:
    script = (ROOT / "sdk" / "bootstrap" / "scripts" / "install.ps1").read_text(encoding="utf-8")
    nsi = (ROOT / "installer" / "windows" / "crucible.nsi").read_text(encoding="utf-8")
    for text in (script, nsi, installwatch.UNDECIDED_SENTENCE):
        assert "shows how it is going" not in text
        assert "shows how that is going" not in text
    assert "Install the WSL2 engine" not in (ROOT / "docs" / "INSTALL-UNINSTALL.md").read_text(encoding="utf-8")
