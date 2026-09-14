"""`<CRUCIBLE_HOME>/pairing` — the file an app on this machine reads.

PHASE15-HOST.md section 3.6.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from crucible import cli, pairing
from crucible.config import config_mode, load_config
from crucible.pairing import pairing_file_path


def write_pairing_file(home: Path, *, name: str, port: int, token: str) -> Path:
    """What `crucible init` does, spelled the same way it spells it.

    The ONE writer is `crucible.pairing.write_pairing_file`, which takes a
    LINE; turning (name, port, token) into the loopback line is
    `cli._write_pairing_file`'s job, and this test calls that so the file
    under test is the file the CLI writes.
    """
    return cli._write_pairing_file(home, name=name, port=port, token=token)


def test_the_file_is_one_loopback_line_with_a_trailing_newline(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    path = write_pairing_file(home, name="crucible@pc", port=7100, token="tok3n")
    assert path == pairing_file_path(home)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.count("\n") == 1
    assert text.strip() == pairing.pairing_line(
        "crucible@pc", "http://127.0.0.1:7100", "tok3n"
    )
    assert text.strip() == "crucible://crucible%40pc@127.0.0.1:7100/#tok3n"


def test_the_line_is_the_loopback_one_whatever_the_bind_is(tmp_path: Path) -> None:
    """The file answers *"an app on THIS machine wants in"*.

    A wildcard-bound server has no loopback entry in `reachable_urls` at all,
    so a file built from that list would hand a local app whichever interface
    the OS listed first.
    """
    home = tmp_path / "home"
    path = write_pairing_file(home, name="crucible@pc", port=7100, token="tok3n")
    assert "127.0.0.1" in path.read_text(encoding="utf-8")


def test_the_file_is_user_only(tmp_path: Path) -> None:
    """0600, like the config: the line carries the token in its fragment."""
    home = tmp_path / "home"
    path = write_pairing_file(home, name="crucible@pc", port=7100, token="tok3n")
    assert oct(stat.S_IMODE(path.stat().st_mode)) == "0o600"
    assert config_mode(path) == "0o600"


def test_init_writes_it_and_force_rewrites_it_with_the_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CRUCIBLE_HOME", str(home))
    monkeypatch.setattr(
        cli, "detect_backend", lambda: __import__(
            "tests.conftest", fromlist=["FAKE_BACKEND"]
        ).FAKE_BACKEND
    )
    assert cli.main(["init", "--token", "firsttoken"]) == 0
    first = pairing_file_path(home).read_text(encoding="utf-8").strip()
    assert first.endswith("#firsttoken")
    assert "pairing:" in capsys.readouterr().out

    assert cli.main(["init", "--force", "--token", "secondtoken"]) == 0
    second = pairing_file_path(home).read_text(encoding="utf-8").strip()
    assert second.endswith("#secondtoken")
    assert second != first
    # …and it agrees with the config it sits beside.
    assert load_config(home).token == "secondtoken"


def test_token_url_prints_the_same_line_the_file_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """*"`crucible token --url` prints the same"* — section 3.6, literally."""
    home = tmp_path / "home"
    monkeypatch.setenv("CRUCIBLE_HOME", str(home))
    monkeypatch.setattr(
        cli, "detect_backend", lambda: __import__(
            "tests.conftest", fromlist=["FAKE_BACKEND"]
        ).FAKE_BACKEND
    )
    cli.main(["init", "--token", "firsttoken"])
    capsys.readouterr()
    assert cli.main(["token", "--url"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert pairing_file_path(home).read_text(encoding="utf-8").strip() == printed[0]


def test_the_loopback_line_is_printed_once_on_a_loopback_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`reachable_urls` already answers with it, and it must not be doubled."""
    lines = cli._pairing_lines("crucible@pc", "127.0.0.1", 7100, "tok3n")
    assert lines == ["crucible://crucible%40pc@127.0.0.1:7100/#tok3n"]


def test_a_wildcard_bind_prints_the_loopback_line_and_then_the_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: ["192.168.68.20"])
    lines = cli._pairing_lines("crucible@pc", "0.0.0.0", 7100, "tok3n")
    assert lines == [
        "crucible://crucible%40pc@127.0.0.1:7100/#tok3n",
        "crucible://crucible%40pc@192.168.68.20:7100/#tok3n",
    ]
