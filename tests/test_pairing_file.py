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


# ------------------------------- 3.6 as amended: `crucible serve` writes it
#
# `init` and `service install` were the only writers, so a server that
# EXISTED before this phase had no pairing file and an app on its own machine
# was told there was no engine there. Measured on the Mac Studio right after
# its upgrade.


def _synced(home: Path, **overrides) -> str | None:
    """Run `cli._sync_pairing_file` over a config and read the file back."""
    from crucible.config import load_config

    config = load_config(home)
    for name, value in overrides.items():
        object.__setattr__(config, name, value)
    cli._sync_pairing_file(config)
    return pairing.read_pairing_file(home)


def _init(home: Path, monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(home))
    monkeypatch.setattr(
        cli, "detect_backend", lambda: __import__(
            "tests.conftest", fromlist=["FAKE_BACKEND"]
        ).FAKE_BACKEND
    )
    assert cli.main(["init", "--token", token]) == 0


def test_serve_writes_the_file_when_a_server_that_predates_it_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    home = tmp_path / "home"
    _init(home, monkeypatch, "firsttoken")
    capsys.readouterr()
    pairing.pairing_file_path(home).unlink()
    assert pairing.read_pairing_file(home) is None

    line = _synced(home)
    assert line is not None and line.endswith("#firsttoken")
    assert "pairing:" in capsys.readouterr().out


def test_serve_REPLACES_a_file_that_disagrees_with_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A stale line points an app at a door with the wrong key."""
    home = tmp_path / "home"
    _init(home, monkeypatch, "firsttoken")
    capsys.readouterr()
    pairing.write_pairing_file(
        home, "crucible://crucible%40pc@127.0.0.1:7100/#a-token-from-last-year"
    )
    line = _synced(home)
    assert line is not None
    assert line.endswith("#firsttoken")
    assert "a-token-from-last-year" not in line


def test_serve_writes_NOTHING_when_the_file_already_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Idempotent, and silent about it: a restart is not news."""
    home = tmp_path / "home"
    _init(home, monkeypatch, "firsttoken")
    capsys.readouterr()
    before = pairing.pairing_file_path(home).stat().st_mtime_ns
    _synced(home)
    assert pairing.pairing_file_path(home).stat().st_mtime_ns == before
    assert "pairing:" not in capsys.readouterr().out


def test_the_line_is_the_CONFIG_s_and_never_a_run_s_port_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """3.6's file answers *"an app on THIS machine wants in"*.

    A developer running `crucible serve --port 7999` for an afternoon must
    not repoint every app on the box at a server that is about to stop, so
    `cmd_serve` hands `_sync_pairing_file` the CONFIG and not the two
    overrides it resolved for this run.
    """
    home = tmp_path / "home"
    _init(home, monkeypatch, "firsttoken")
    capsys.readouterr()
    line = _synced(home)
    assert line is not None
    assert f":{7100}/" in line or ":7100/#" in line
    source = (Path(cli.__file__)).read_text(encoding="utf-8")
    assert "_sync_pairing_file(config)" in source
