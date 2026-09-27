from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from urllib.parse import quote

from crucible import cli, jobenv, pairing
from crucible.config import config_path, load_config
from crucible.errors import NoViableBackend
from crucible.interfaces import InterfaceError
from crucible.voices import NARRATOR_ENGINE_SAMPLING

from .conftest import FAKE_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


def test_init_writes_a_0600_config_with_a_token(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    path = config_path(home)
    assert path.exists()
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    config = load_config(home)
    assert config.backend_kind == "cuda-linux"
    assert config.enable_echo is True
    assert len(config.token) >= 40
    out = capsys.readouterr().out
    assert config.token not in out, "init names the pairing file, it never prints the token"
    assert config.token in pairing.read_pairing_file(home)
    assert "crucible token --show" in out


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

    monkeypatch.setattr(cli.common, "detect_backend", refuse)
    assert cli.main(["init"]) == 1
    assert not config_path(home).exists()
    assert "no nvidia-smi" in capsys.readouterr().err


def test_doctor_json_is_healthy_after_init(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("crucible.cli.common.detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["problems"] == []
    assert report["backend"]["kind"] == "cuda-linux"
    if os.name != "nt":
        assert report["config"]["mode"] == "0o600"
    echo = [entry for entry in report["job_types"] if entry["name"] == "echo"][0]
    assert echo == {
        "name": "echo",
        "enabled": True,
        "ready": True,
        "detail": "enabled; copies inputs to artifacts, uses no accelerator",
        "models": [],
        "awaiting_weights": False,
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
    monkeypatch.setattr(cli.common, "detect_backend", lambda: elsewhere)
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any("backend_changed" in problem for problem in report["problems"])


def test_token_needs_show_or_url(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    assert cli.main(["token"]) == 1


def test_token_url_prints_one_pairing_line_per_address(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
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
        f"127.0.0.1:7100/#{load_config(home).token}",
        f"10.0.0.4:7100/#{load_config(home).token}",
        f"100.64.0.3:7100/#{load_config(home).token}",
    ]


def test_token_url_refuses_when_the_interfaces_cannot_be_read(
    home: Path,
    viable: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

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
    from crucible import service

    assert cli.main(["init"]) == 0
    monkeypatch.setattr(
        service, "install", lambda *a, **k: ["unit:     /home/x/crucible.service"]
    )
    capsys.readouterr()
    assert cli.main(["service", "install"]) == 0
    out = capsys.readouterr().out
    assert cli.PAIRING_NOT_PRINTED in out
    assert load_config(home).token not in out


def test_init_ends_with_the_pairing_line(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "pairing:" in out
    assert "crucible://" not in out
    assert load_config(home).token not in out


def test_token_show_prints_the_token(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["token", "--show"]) == 0
    assert capsys.readouterr().out.strip() == load_config(home).token


def test_token_without_a_config_is_refused(home: Path) -> None:
    assert cli.main(["token", "--show"]) == 1


def test_doctor_reports_one_tts_env_per_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["enable_tts"] is True
    assert sorted(report["tts_envs"]) == ["higgs-v3"]
    for engine, entry in report["tts_envs"].items():
        assert entry["installed"] is False
        assert f"envs/tts-{engine}" in entry["detail"].replace("\\", "/")
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
    from .conftest import FAKE_MAC_BACKEND

    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
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
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "tts"]) == 1
    assert "needs --narrator-engine" in capsys.readouterr().err


def test_installing_tts_builds_the_env_of_the_named_narrator_engine(
    home: Path,
    viable: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    asked: list[jobenv.EnvSpec] = []

    def install_env(home_dir: Path, spec: jobenv.EnvSpec, backend_kind: str, **_: object):
        asked.append(spec)
        raise jobenv.EnvError("stopped by the test before pip ran")

    monkeypatch.setattr(jobenv, "install_env", install_env)
    assert cli.main(["install", "tts", "--narrator-engine", "higgs-v3"]) == 1
    assert "stopped by the test before pip ran" in capsys.readouterr().err
    assert asked == [jobenv.tts_env("higgs-v3", FAKE_BACKEND.kind)]


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
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        assert jobenv.recipe_for(jobenv.tts_env(engine, "cuda-linux")).is_file()
    mac = {
        jobenv.recipe_for(jobenv.tts_env(engine, "mlx-darwin"))
        for engine in sorted(NARRATOR_ENGINE_SAMPLING)
    }
    assert len(mac) == 1


def test_every_tts_recipe_pins_narrator_by_a_commit(home: Path, viable: None) -> None:
    for spec in (
        jobenv.tts_env("higgs-v3", "cuda-linux"),
        jobenv.tts_env("higgs-v3", "mlx-darwin"),
    ):
        recipe = jobenv.recipe_for(spec)
        references = jobenv.recipe_direct_references(recipe)
        assert list(references) == ["narrator"]
        assert len(references["narrator"]) == 40


def test_the_tts_recipes_pin_the_stack_each_arm_measured(
    home: Path, viable: None
) -> None:
    higgs = jobenv.recipe_pins(
        jobenv.recipe_for(jobenv.tts_env("higgs-v3", "cuda-linux"))
    )
    assert higgs["sglang-omni"] == "0.1.4"
    assert higgs["sglang"] == "0.5.18"
    assert higgs["torch"] == "2.13.0"
    assert higgs["flashinfer-python"] == "0.6.17"
    assert higgs["flashinfer-jit-cache"] == "0.6.17+cu130"
    assert "vllm" not in higgs
    assert "vllm-omni" not in higgs
    mac = jobenv.recipe_pins(jobenv.recipe_for(jobenv.tts_env("higgs-v3", "mlx-darwin")))
    assert mac["mlx-audio"] == "0.4.8"
    assert mac["mlx-lm"] == "0.31.3"


def test_doctor_runs_with_every_job_type_enabled(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("crucible.cli.common.detect_backend", lambda: FAKE_BACKEND)
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
    assert set(report["tts_envs"]) == {"higgs-v3"}
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


def test_init_takes_a_token_the_caller_minted(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    given = "bootstrap-minted-" + "x" * 30
    assert cli.main(["init", "--token", given, "--enable-echo"]) == 0
    assert load_config(home).token == given
    out = capsys.readouterr().out
    assert f"#{given}" in pairing.read_pairing_file(home), "the given token rides in the pairing line"
    assert given not in out
    assert "token:    as given" in out


def test_init_refuses_a_blank_or_spaced_token(home: Path, viable: None) -> None:
    assert cli.main(["init", "--token", "   "]) == 1
    assert cli.main(["init", "--token", "has a space"]) == 1
    assert not config_path(home).exists()


def test_every_installable_name_has_a_smoke_import() -> None:
    for backend_kind in ("cuda-linux", "mlx-darwin"):
        for job_type in cli.INSTALLABLE_JOB_TYPES:
            if job_type in jobenv.WORKER_JOB_TYPES:
                try:
                    jobenv.recipe_for(jobenv.worker_env(job_type, backend_kind))
                except jobenv.EnvError:
                    continue
                keys = [job_type]
            elif job_type == "llm":
                keys = [jobenv.llm_env(backend_kind).key]
            else:
                keys = sorted(
                    {
                        jobenv.tts_env(engine, backend_kind).key
                        for engine in NARRATOR_ENGINE_SAMPLING
                    }
                )
            for key in keys:
                assert cli.SMOKE_IMPORT.get(key, {}).get(backend_kind), (
                    f"{job_type}/{backend_kind}: the {key!r} env has no smoke import"
                )


def test_an_env_with_no_smoke_import_is_refused_rather_than_called_installed(
    tmp_path: Path,
) -> None:
    refusal = cli._smoke_import(tmp_path / "python", "llm", "rocm-linux")
    assert refusal is not None
    assert "SMOKE_IMPORT" in refusal


def test_a_failing_import_is_refused_by_name_with_the_last_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "Traceback\nImportError: libcudart.so.12: cannot open shared object file"

    monkeypatch.setattr(cli.install.subprocess, "run", lambda *a, **k: _Completed())
    refusal = cli._smoke_import(tmp_path / "python", "llm", "cuda-linux")
    assert refusal is not None
    assert refusal.startswith("env_smoke_failed:")
    assert "libcudart" in refusal


def _live_server_double(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> list[tuple[str, str, object]]:
    from crucible import apiclient

    calls: list[tuple[str, str, object]] = []
    doubled = apiclient.Connection(
        url="http://127.0.0.1:7100", token="t", name="crucible@test", source="local"
    )
    monkeypatch.setattr(cli.common, "server_here", lambda _config, _backend: doubled)

    def call(connection: object, method: str, path: str, json_body: object = None, **_: object):
        assert connection is doubled
        calls.append((method, path, json_body))
        return {"voice": None, "path": str(home / "voices" / "pins.toml")}

    monkeypatch.setattr(apiclient, "call", call)
    return calls


def test_remove_goes_through_the_server_when_one_answers(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from crucible.manifests import load_manifest

    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()

    spec = load_manifest("qwen3.5-9b").spec(FAKE_BACKEND.kind)
    directory = home / "models" / "qwen3.5-9b" / FAKE_BACKEND.kind
    directory.mkdir(parents=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps({
            "model": "qwen3.5-9b", "backend": FAKE_BACKEND.kind,
            "hf_repo": spec.hf_repo, "revision": spec.revision, "bytes": 1_000_000_000,
            "seconds": 1.0, "pulled": "2026-09-27T00:00:00+0000",
        }),
        encoding="utf-8",
    )
    calls = _live_server_double(monkeypatch, home)
    assert cli.main(["remove", "model", "qwen3.5-9b"]) == 0
    assert calls == [("DELETE", "/v1/catalog/model/qwen3.5-9b", None)]
    assert directory.exists(), "the server deletes; the CLI never reaches past it"
    out = capsys.readouterr().out
    assert "removed:  model qwen3.5-9b" in out
    assert "through:  the server at http://127.0.0.1:7100" in out


def test_remove_with_no_server_deletes_the_files_itself(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(cli.common, "server_here", lambda _config, _backend: None)
    assert cli.main(["remove", "model", "qwen3.5-9b"]) == 1
    err = capsys.readouterr().err
    assert "subject_not_installed" in err
    assert cli.main(["remove", "model", "no-such-model"]) == 1
    err = capsys.readouterr().err
    assert "subject_unknown" in err
    assert "`crucible api catalog`" in err and "`crucible catalog`" not in err


def test_voices_pin_goes_through_the_server_when_one_answers(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from crucible import voicerepo

    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    calls = _live_server_double(monkeypatch, home)
    voice = SimpleNamespace(
        display="Mistborn", kind="checkpoint", narrator_engine="higgs-v3",
        backends={"cuda-linux": None},
    )
    monkeypatch.setattr(voicerepo, "voice_for_pin", lambda _pin: voice)
    sha = "a" * 40
    assert cli.main(["voices", "pin", "mistborn", f"owenmorgan/mistborn-higgs-v3@{sha}"]) == 0
    assert calls == [(
        "PUT", "/v1/voices/mistborn",
        {"pin": {"hf_repo": "owenmorgan/mistborn-higgs-v3", "revision": sha}},
    )]
    assert not (home / "voices" / "pins.toml").exists(), "the server writes the pin"
    out = capsys.readouterr().out
    assert out.startswith(f"mistborn: owenmorgan/mistborn-higgs-v3@{sha[:12]} -> ")
    assert "Mistborn (checkpoint, higgs-v3), arms ['cuda-linux']" in out
    assert "`crucible voices pull mistborn`" in out


def test_models_list_refuses_on_a_backend_the_config_was_not_written_for(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .conftest import FAKE_MAC_BACKEND

    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    for argv in (["models", "list"], ["voices", "list"], ["rvc", "list"], ["denoise", "list"]):
        assert cli.main(argv) == 1, argv
        err = capsys.readouterr().err
        assert "backend_not_here" in err, argv
        assert "`crucible init --force`" in err, argv


def test_no_viable_backend_names_the_next_step(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()

    def refuse() -> None:
        raise NoViableBackend("no nvidia-smi on this Linux host (looked on PATH)")

    monkeypatch.setattr(cli.common, "detect_backend", refuse)
    monkeypatch.setattr(cli.common.sys, "platform", "linux")
    assert cli.main(["models", "list"]) == 1
    err = capsys.readouterr().err
    assert "no viable backend: no nvidia-smi" in err
    assert "NVIDIA driver on Windows" in err and "WSL" in err
    monkeypatch.setattr(cli.common.sys, "platform", "darwin")
    assert cli.main(["models", "list"]) == 1
    assert "Apple silicon" in capsys.readouterr().err


def test_every_doctor_problem_names_a_command_to_run(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .conftest import FAKE_MAC_BACKEND

    assert cli.main(
        ["init", "--enable-llm", "--enable-tts", "--enable-asr", "--enable-align", "--enable-rvc"]
    ) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["problems"]
    for problem in report["problems"]:
        assert "`crucible " in problem or "`chmod " in problem, problem

    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    changed = [p for p in report["problems"] if p.startswith("backend_changed")]
    assert len(changed) == 1
    assert "`crucible init --force`" in changed[0]
    assert str(config_path(home)) in changed[0]

    def refuse() -> None:
        raise NoViableBackend("no nvidia-smi on this Linux host")

    monkeypatch.setattr(cli.common, "detect_backend", refuse)
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    absent = [p for p in report["problems"] if p.startswith("no_viable_backend")]
    assert len(absent) == 1 and "driver" in absent[0]


def test_every_backticked_crucible_command_in_the_tree_exists() -> None:
    import re

    import crucible

    parser = cli.build_parser()
    top = {name for name in parser._subparsers._group_actions[0].choices}
    root = Path(crucible.__file__).parent
    pattern = re.compile(r"`crucible ([a-z][a-z-]*)")
    missing: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for word in pattern.findall(line):
                if word not in top:
                    missing.append(f"{path.relative_to(root)}:{line_number}: crucible {word}")
    assert missing == [], "\n".join(missing)
