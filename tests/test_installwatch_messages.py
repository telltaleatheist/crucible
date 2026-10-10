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


class _Engine:
    """What the engine answers on each read: a list of versions, None for not answering."""

    def __init__(self, answers: list[str | None]) -> None:
        self._answers = answers
        self.reads = 0

    def __call__(self, home: Path) -> installwatch.EngineAnswer:
        version = self._answers[min(self.reads, len(self._answers) - 1)]
        self.reads += 1
        return installwatch.EngineAnswer(
            version, "answering" if version is not None else "connection refused"
        )


def _await(answers: list[str | None], budget: float = 600.0) -> tuple[bool, str, _Engine]:
    tick = iter(float(n) for n in range(0, 100000, 2))
    out = io.StringIO()
    engine = _Engine(answers)
    ready = installwatch.await_release(
        ROOT, installwatch.Console(out), clock=lambda: next(tick),
        sleep=lambda seconds: None, release="1.0.200", budget=budget, read=engine,
    )
    return ready, " ".join(out.getvalue().split()), engine


def test_an_update_is_ready_only_once_the_engine_answers_on_the_new_release() -> None:
    # Victoria's laptop, 1.0.114: "ready" was printed while the tray was still
    # carrying the guest. Owen: ready is when it can actually do real work.
    ready, said, engine = _await(["1.0.199"] * 20 + [None] * 15 + ["1.0.200"])
    assert ready is True
    assert engine.reads == 36, "it waited through the old release and the restart"
    assert said.index("moving it to 1.0.200") < said.index("Crucible 1.0.200 is ready")
    assert "it answers as 1.0.199" in said, "the wait says how it stands"
    assert "not answering: connection refused" in said


def test_an_engine_that_never_reaches_the_release_is_named_after_the_stated_budget() -> None:
    ready, said, _ = _await(["1.0.199"], budget=300.0)
    assert ready is False
    assert "is ready" not in said
    assert "up to 5 minutes" in said, "the budget is stated before the wait"
    assert "after 5 minutes its Linux engine still answers as Crucible 1.0.199" in said
    assert str(ROOT / "host.log") in said


def test_the_brief_watch_of_an_update_waits_for_the_release_too(monkeypatch) -> None:
    monkeypatch.setattr(installwatch, "_owner", lambda home: installwatch.OWNER_WSL_UNIT)
    engine = _Engine([None, installwatch.VERSION])
    monkeypatch.setattr(installwatch, "engine_answer", engine)
    tick = iter(float(n) for n in range(1000))
    out = io.StringIO()
    assert installwatch.watch(ROOT, SINCE, brief=True, out=out, clock=lambda: next(tick),
                              sleep=lambda seconds: None) == 0
    text = " ".join(out.getvalue().split())
    assert f"Crucible {installwatch.VERSION} is ready" in text
    assert installwatch.APP_SENTENCE.split(".")[0] not in text, "no hand-over: it is ready"


def test_the_engine_is_read_with_the_pairing_token_and_named_when_it_will_not_answer(
    tmp_path: Path, monkeypatch
) -> None:
    from crucible import controller_client

    assert "no pairing file" in installwatch.engine_answer(tmp_path).detail
    (tmp_path / "pairing").write_text("crucible://c@127.0.0.1:7100/#tok\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def answers(url: str, **kwargs: object) -> dict:
        seen.update(url=url, **kwargs)
        return {"server": {"version": "1.0.200"}}

    monkeypatch.setattr(controller_client, "request", answers)
    answer = installwatch.engine_answer(tmp_path)
    assert answer.version == "1.0.200"
    assert seen["url"] == "http://127.0.0.1:7100/v1/info" and seen["token"] == "tok"

    def refused(url: str, **kwargs: object) -> dict:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(controller_client, "request", refused)
    answer = installwatch.engine_answer(tmp_path)
    assert answer.version is None and "connection refused" in answer.detail
