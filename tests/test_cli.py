"""The command line: init, doctor, token — with the host probe monkeypatched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urllib.parse import quote

from crucible import cli, jobenv, pairing
from crucible.config import config_path, load_config
from crucible.errors import NoViableBackend
from crucible.interfaces import InterfaceError

from .conftest import FAKE_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)


def test_init_writes_a_0600_config_with_a_token(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    path = config_path(home)
    assert path.exists()
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    config = load_config(home)
    assert config.backend_kind == "cuda-linux"
    assert config.enable_echo is True
    assert len(config.token) >= 40
    # THE TOKEN IS NOW PRINTED, inside the pairing line, and that reverses what
    # this test used to assert. PHASE13-OPERATOR.md section 3.5: `init` ends by
    # printing the lines an app's connect door takes, because Owen's rule for
    # the whole phase is that nobody types a token twice. It is printed to the
    # terminal of the person who just minted it, on the machine they are
    # sitting at — which is the one audience that already has it.
    assert config.token in capsys.readouterr().out


def test_init_refuses_to_clobber_an_existing_config(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    first = load_config(home).token
    assert cli.main(["init"]) == 1
    assert load_config(home).token == first


def test_init_force_mints_a_new_token(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    first = load_config(home).token
    assert cli.main(["init", "--force"]) == 0
    assert load_config(home).token != first


def test_init_refuses_without_a_backend(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse() -> None:
        raise NoViableBackend("no nvidia-smi on this Linux host")

    monkeypatch.setattr(cli, "detect_backend", refuse)
    assert cli.main(["init"]) == 1
    assert not config_path(home).exists()
    assert "no nvidia-smi" in capsys.readouterr().err


def test_doctor_json_is_healthy_after_init(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("crucible.cli.detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["problems"] == []
    assert report["backend"]["kind"] == "cuda-linux"
    assert report["config"]["mode"] == "0o600"
    echo = [entry for entry in report["job_types"] if entry["name"] == "echo"][0]
    assert echo == {
        "name": "echo",
        "enabled": True,
        "ready": True,
        "detail": "enabled; copies inputs to artifacts, uses no accelerator",
        "models": [],
    }


def test_doctor_is_unhealthy_without_a_config(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is False
    assert any("config" in problem for problem in report["problems"])


def test_doctor_is_unhealthy_when_the_backend_changed(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()

    from crucible.backend import Backend, Gpu

    elsewhere = Backend(
        kind="mlx-darwin",
        platform="darwin",
        arch="arm64",
        gpu=Gpu(vendor="apple", name="Apple M2 Ultra", vram_bytes=1),
        detail="test double",
    )
    monkeypatch.setattr(cli, "detect_backend", lambda: elsewhere)
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any("backend_changed" in problem for problem in report["problems"])


def test_token_needs_show_or_url(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    assert cli.main(["token"]) == 1


def test_token_url_prints_one_pairing_line_per_address(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """PHASE13-OPERATOR.md 3.5. A concrete bind is exactly one line."""
    assert cli.main(["init", "--host", "127.0.0.1", "--port", "7100"]) == 0
    capsys.readouterr()
    assert cli.main(["token", "--url"]) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    config = load_config(home)
    assert printed == [
        f"crucible://{quote(config.name, safe='')}@127.0.0.1:7100/#{config.token}"
    ]


def test_token_url_lists_every_interface_of_a_wildcard_bind(
    home: Path,
    viable: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: ["10.0.0.4", "100.64.0.3"])
    assert cli.main(["init", "--host", "0.0.0.0"]) == 0
    capsys.readouterr()
    assert cli.main(["token", "--url"]) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert [line.split("@")[-1] for line in printed] == [
        f"10.0.0.4:7100/#{load_config(home).token}",
        f"100.64.0.3:7100/#{load_config(home).token}",
    ]


def test_token_url_refuses_when_the_interfaces_cannot_be_read(
    home: Path,
    viable: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No line is better than a guessed one, and the reason is said out loud."""

    def refuse() -> list[str]:
        raise InterfaceError("getifaddrs(3) failed")

    assert cli.main(["init", "--host", "0.0.0.0"]) == 0
    monkeypatch.setattr(pairing, "ipv4_addresses", refuse)
    capsys.readouterr()
    assert cli.main(["token", "--url"]) == 1
    assert "getifaddrs" in capsys.readouterr().err


def test_service_install_ends_with_the_pairing_line(
    home: Path,
    viable: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """3.5's second line. The unit itself is `tests/test_service.py`'s."""
    from crucible import service

    assert cli.main(["init"]) == 0
    monkeypatch.setattr(
        service, "install", lambda *a, **k: ["unit:     /home/x/crucible.service"]
    )
    capsys.readouterr()
    assert cli.main(["service", "install"]) == 0
    out = capsys.readouterr().out
    assert "pairing: paste one of these" in out
    assert load_config(home).token in out


def test_init_ends_with_the_pairing_line(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "pairing:" in out
    assert f"crucible://" in out
    assert load_config(home).token in out


def test_token_show_prints_the_token(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["token", "--show"]) == 0
    assert capsys.readouterr().out.strip() == load_config(home).token


def test_token_without_a_config_is_refused(home: Path) -> None:
    assert cli.main(["token", "--show"]) == 1


# ------------------------------------------------------------------- the tts
# job type: `crucible voices`, the doctor's env rows, and `crucible install tts`.


def test_doctor_reports_one_tts_env_per_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """On cuda-linux the two engines cannot share a venv, so there are two rows —
    and neither is installed on a fresh host, which is a problem and says so."""
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["enable_tts"] is True
    assert sorted(report["tts_envs"]) == ["higgs-v3", "orpheus"]
    for engine, entry in report["tts_envs"].items():
        assert entry["installed"] is False
        assert f"envs/tts-{engine}" in entry["detail"]
        assert "crucible install tts" in entry["detail"]
    assert any("tts_env[higgs-v3]" in problem for problem in report["problems"])


def test_doctor_says_nothing_about_tts_when_it_is_off(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["tts_envs"] == {}


def test_the_two_tts_engines_share_one_env_on_the_mac(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On mlx-darwin they genuinely do, so the two rows name the same directory."""
    from .conftest import FAKE_MAC_BACKEND

    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    paths = {entry["path"] for entry in report["tts_envs"].values()}
    assert paths == {str(home / "envs" / "tts")}


def test_voices_list_names_the_pull_command(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["voices", "list", "--json"]) == 0
    listed = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert listed["deathstalker"]["installed"] is False
    assert listed["deathstalker"]["max_chars"] == 800
    assert listed["deathstalker"]["estimate_basis"] == "declared"
    assert "crucible voices pull deathstalker" in listed["deathstalker"]["detail"]


def test_voices_pull_refuses_an_unknown_voice(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["voices", "pull", "gandalf"]) == 1
    assert "no manifest for voice 'gandalf'" in capsys.readouterr().err


def test_installing_tts_needs_a_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two envs on this backend, so the command may not pick one for you."""
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "tts"]) == 1
    assert "needs --narrator-engine" in capsys.readouterr().err


def test_installing_llm_refuses_a_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "llm", "--narrator-engine", "higgs-v3"]) == 1
    assert "means nothing for 'llm'" in capsys.readouterr().err


def test_this_build_ships_a_tts_recipe_for_every_env_a_voice_can_need(
    home: Path, viable: None
) -> None:
    """One recipe per (narrator engine, backend), which is not one per backend.

    Orpheus pins `vllm==0.7.3` for its per-request logits processors and Higgs v3
    needs `vllm-omni` against a far later torch, so on cuda-linux the two cannot
    share a venv. On mlx-darwin they genuinely do, and both names resolve to the
    one `mlx-darwin` recipe.
    """
    for engine in ("higgs-v3", "orpheus"):
        assert jobenv.recipe_for(jobenv.tts_env(engine, "cuda-linux")).is_file()
    mac = {
        jobenv.recipe_for(jobenv.tts_env(engine, "mlx-darwin"))
        for engine in ("higgs-v3", "orpheus")
    }
    assert len(mac) == 1


def test_every_tts_recipe_pins_narrator_by_a_commit(home: Path, viable: None) -> None:
    """A branch name is not a pin, and `narrator==0.1.0` would let any commit in.

    narrator is not on PyPI — it is `python/narrator` in the BookForge repo,
    versioned with the app — so it is pinned by a direct reference carrying a
    40-character sha, and `crucible doctor` checks it against the commit pip
    recorded in PEP 610's direct_url.json rather than against a version.
    """
    for spec in (
        jobenv.tts_env("higgs-v3", "cuda-linux"),
        jobenv.tts_env("orpheus", "cuda-linux"),
        jobenv.tts_env("higgs-v3", "mlx-darwin"),
    ):
        recipe = jobenv.recipe_for(spec)
        references = jobenv.recipe_direct_references(recipe)
        assert list(references) == ["narrator"]
        assert len(references["narrator"]) == 40


def test_the_orpheus_recipe_pins_the_last_vllm_that_takes_a_logits_processor(
    home: Path, viable: None
) -> None:
    """0.7.3 is a hard pin, not a floor: above it the EOS boost silently stops
    applying, because V1 has no per-request logits processor at all."""
    pins = jobenv.recipe_pins(
        jobenv.recipe_for(jobenv.tts_env("orpheus", "cuda-linux"))
    )
    assert pins["vllm"] == "0.7.3"
    assert pins["torch"] == "2.5.1"
    higgs = jobenv.recipe_pins(
        jobenv.recipe_for(jobenv.tts_env("higgs-v3", "cuda-linux"))
    )
    assert higgs["vllm"] == "0.28.0"
    assert higgs["vllm-omni"] == "0.28.0"
    # And the Mac's one version of mlx-audio that can render Orpheus at all.
    mac = jobenv.recipe_pins(jobenv.recipe_for(jobenv.tts_env("orpheus", "mlx-darwin")))
    assert mac["mlx-audio"] == "0.4.8"
    assert mac["mlx-lm"] == "0.31.3"


def test_doctor_reports_the_two_site_packages_patches_by_name(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """pip cannot express an edit to somebody else's installed package.

    An env whose pins all match is reported ready, and a reader has no way to
    tell that from an env that will render every chunk with 240 ms of garbage on
    the end — so the patches are their own rows and their own problems.
    """
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    rows = {entry["id"]: entry for entry in report["narrator_patches"]}
    assert sorted(rows) == ["higgs-sentinel-filter", "vllm-negative-token-id"]
    for entry in rows.values():
        assert entry["applied"] is False
        assert entry["status"] == "no_env"
    assert any("HTTP 400" in problem for problem in report["problems"])
    assert any("240 ms of audible garbage" in problem for problem in report["problems"])


def test_doctor_runs_with_every_job_type_enabled(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every enabled type's env row, on one report, without raising.

    This is the test that was missing. `crucible doctor` crashed with a
    `TypeError` on any config with `enable_llm` on, for the whole time the
    `tts` merge was on main, and the suite stayed green because every existing
    doctor test enables `echo` alone — the one job type with no env at all. A
    command whose entire job is "tell the operator what is wrong here" was the
    one command nobody could run.

    It asserts the report is *complete and unhealthy*, not that it is healthy:
    no env is installed in a test home, so every row should be naming what is
    missing and the command that fixes it.
    """
    monkeypatch.setattr("crucible.cli.detect_backend", lambda: FAKE_BACKEND)
    # EVERY type, which is what this test is named after and what it has to keep
    # being: two more (`align`, `rvc`) landed after it was written, and a test
    # that says "every" while naming four is a test that stops covering the
    # thing it exists for the moment a fifth arrives.
    assert cli.main(
        [
            "init",
            "--enable-echo",
            "--enable-llm",
            "--enable-tts",
            "--enable-asr",
            "--enable-align",
            "--enable-rvc",
        ]
    ) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is False

    assert report["llm_env"]["installed"] is False
    assert "crucible install llm" in report["llm_env"]["detail"]
    assert set(report["tts_envs"]) == {"higgs-v3", "orpheus"}
    assert [row["job_type"] for row in report["worker_envs"]] == [
        "align", "asr", "rvc",
    ]

    enabled = {e["name"] for e in report["job_types"] if e["enabled"]}
    assert {
        "echo", "load-model", "load-voice", "tts", "asr", "align",
        "unload-aligner", "rvc",
    } <= enabled
    for problem in report["problems"]:
        assert problem, "a problem with no text is a problem nobody can act on"


def test_a_config_written_before_a_job_type_existed_still_loads(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An upgrade must not invalidate a config, and must say what it assumed.

    Found on Owen's Mac, 2026-09-13: its server had been running since before
    `asr`, `tts`, `align` and `rvc` were built, and after the upgrade `crucible
    token --show` could not read its own config to print its own token. The only
    repair on offer was `crucible init --force`, which mints a NEW token and
    breaks every client that had the old one — so requiring the flags meant
    every added job type broke every deployed server.

    Absent reads as off, which cannot switch anything on by accident, and
    `doctor` names the flags so "off because I chose that" and "off because this
    file predates it" are told apart rather than guessed at.
    """
    monkeypatch.setattr("crucible.cli.detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-echo", "--enable-llm"]) == 0
    capsys.readouterr()

    # Rewrite the config as a phase-2 one: no asr, tts, align or rvc key at all.
    path = home / "config.toml"
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines(True)
        if not any(
            line.startswith(f"{flag} =")
            for flag in ("enable_asr", "enable_tts", "enable_align", "enable_rvc")
        )
    ]
    path.write_text("".join(kept), encoding="utf-8")

    config = load_config(home)
    assert config.enable_echo is True
    assert config.enable_llm is True
    assert config.enable_asr is False
    assert config.enable_rvc is False
    assert set(config.flags_absent) == {
        "enable_asr", "enable_tts", "enable_align", "enable_rvc",
    }
    assert config.token, "the token survived, which is the whole point"

    assert cli.main(["doctor", "--json"]) in (0, 1)
    report = json.loads(capsys.readouterr().out)
    assert set(report["config"]["flags_absent"]) == {
        "enable_asr", "enable_tts", "enable_align", "enable_rvc",
    }


def test_init_takes_a_token_the_caller_minted(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--token` is for `@crucible/bootstrap`, which mints on the app's side.

    The token written is the one given, byte for byte.

    **It now appears in the printed pairing line, and that reverses what this
    test used to assert.** The old reasoning was that a bootstrapper which
    mints its own token must never have to read it back out of a log — which
    is still true, and is still why `--token` exists: the app already holds it
    and reads nothing. What changed is that the OTHER audience for this
    command's output is a person at a terminal who has just created a server
    and needs to point an app at it, and PHASE13-OPERATOR.md section 3.5 rules
    that they get one pasteable line rather than three values to transcribe.
    `token: as given` still distinguishes the two cases in the summary.
    """
    given = "bootstrap-minted-" + "x" * 30
    assert cli.main(["init", "--token", given, "--enable-echo"]) == 0
    assert load_config(home).token == given
    out = capsys.readouterr().out
    assert f"#{given}" in out, "the given token rides in the pairing line"
    assert "token:    as given" in out


def test_init_refuses_a_blank_or_spaced_token(home: Path, viable: None) -> None:
    assert cli.main(["init", "--token", "   "]) == 1
    assert cli.main(["init", "--token", "has a space"]) == 1
    assert not config_path(home).exists()
