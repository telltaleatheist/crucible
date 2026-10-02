from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from crucible import envpatches, jobenv, stallguard
from crucible.engines import EngineError, build_voice_engine
from crucible.engines.narrator import STALL_GUARD_VARIABLE
from crucible.voices import VoiceError, parse_voice, voice_document

from .conftest import configure_box
from .test_voices import GOOD

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sglang_omni_0.1.4_higgs_tts"
FILES = ("sampler.py", "model.py", "model_runner.py")
SCRIPT = envpatches.TTS_SCRIPTS_DIR / envpatches.STALL_GUARD_SCRIPT
DEFAULT_ENV = "37,0.5,20,8"
TTS_PINS = jobenv.recipe_pins(
    jobenv.recipe_for(jobenv.tts_env("higgs-v3", "cuda-linux"))
)
MAC_TTS_PINS = jobenv.recipe_pins(
    jobenv.recipe_for(jobenv.tts_env("higgs-v3", "mlx-darwin"))
)

BOC_ID = 1024
EOC_ID = 1025
N = 8
V = 1026


def stock(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")


def make_env(root: Path, version: str = "0.1.4", texts: dict[str, str] | None = None) -> Path:
    env = root / "tts-higgs-v3"
    site = env / "lib" / "python3.12" / "site-packages"
    package = site / "sglang_omni" / "models" / "higgs_tts"
    package.mkdir(parents=True)
    with open(site / "sglang_omni" / "__init__.py", "w", encoding="utf-8", newline="") as h:
        h.write(f'# SPDX\n__version__ = "{version}"\n')
    for name in FILES:
        body = stock(name) if texts is None or name not in texts else texts[name]
        with open(package / name, "w", encoding="utf-8", newline="") as handle:
            handle.write(body)
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("", encoding="utf-8")
    return env


def package_of(env: Path) -> Path:
    return env / "lib" / "python3.12" / "site-packages" / "sglang_omni" / "models" / "higgs_tts"


def run_script(env: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(env)], capture_output=True, text=True, timeout=60
    )


def texts(env: Path) -> dict[str, str]:
    return {name: (package_of(env) / name).read_text(encoding="utf-8") for name in FILES}


# ---------------------------------------------------------------- the env value


def test_the_default_writes_the_contract_example() -> None:
    assert stallguard.env_value(stallguard.DEFAULT) == DEFAULT_ENV
    assert stallguard.env_value(None) == "off"
    assert stallguard.parse_env(DEFAULT_ENV) == stallguard.DEFAULT


@pytest.mark.parametrize(
    "guard",
    [
        stallguard.DEFAULT,
        stallguard.StallGuard(frames=1, rate=0.001, max=0.001, window=1),
        stallguard.StallGuard(frames=10_000, rate=100.0, max=1000.0, window=64),
        stallguard.StallGuard(frames=50, rate=0.125, max=7.75, window=5),
    ],
)
def test_every_valid_guard_round_trips_through_the_variable(guard: Any) -> None:
    assert stallguard.parse_env(stallguard.env_value(guard)) == guard


def test_unset_and_off_are_off() -> None:
    assert stallguard.parse_env(None) is None
    assert stallguard.parse_env("off") is None
    assert stallguard.parse_env(" off\n") is None


MALFORMED = [
    "",
    "OFF",
    "on",
    "37,0.5,20",
    "37,0.5,20,8,1",
    "37, 0.5,20,8",
    "37,1e-1,20,8",
    "37,inf,20,8",
    "37,nan,20,8",
    "37.0,0.5,20,8",
    "+37,0.5,20,8",
    "-1,0.5,20,8",
    "0,0.5,20,8",
    "37,0,20,8",
    "37,0.5,0,8",
    "37,0.5,20,0",
    "37,0.5,20,65",
    "10001,0.5,20,8",
    "37,100.5,20,8",
    "37,0.5,1000.5,8",
    "37,.5,20,8",
    "37,0.5,20,8\n,",
]


@pytest.mark.parametrize("raw", MALFORMED)
def test_a_malformed_value_is_refused_naming_the_variable(raw: str) -> None:
    with pytest.raises(stallguard.StallGuardError) as caught:
        stallguard.parse_env(raw)
    assert "HIGGS_STALL_GUARD" in str(caught.value)


# ---------------------------------------------------------------- the manifest


@pytest.fixture
def a_configured_box() -> None:
    configure_box(Path(os.environ["CRUCIBLE_HOME"]))


def serving_with(extra: str) -> str:
    anchor = "[voice.backends.cuda-linux]"
    return GOOD.replace(anchor, extra.strip() + "\n\n" + anchor)


def parsed(text: str):
    return parse_voice(text, Path("probe.toml"), "probe")


def refused(text: str) -> str:
    with pytest.raises(VoiceError) as caught:
        parsed(text)
    return str(caught.value)


def test_a_higgs_voice_that_says_nothing_gets_the_default_guard(a_configured_box: None) -> None:
    serving = parsed(GOOD).serving
    assert serving is not None
    assert serving.stall_guard == stallguard.DEFAULT
    assert serving.stall_guard_basis == "default"
    assert serving.stall_guard_env == DEFAULT_ENV
    row = serving.to_dict()["stall_guard"]
    assert row == {
        "enabled": True, "frames": 37, "rate": 0.5, "max": 20.0, "window": 8,
        "env": DEFAULT_ENV, "basis": "default", "note": stallguard.DEFAULT_NOTE,
    }


def test_a_voice_may_state_its_own_numbers(a_configured_box: None) -> None:
    serving = parsed(serving_with(
        'stall_guard = { frames = 50, rate = 0.25, max = 12, window = 6 }\n'
        'stall_guard_note = "a slower reader; measured on the 10-02 ladder"'
    )).serving
    assert serving.stall_guard == stallguard.StallGuard(50, 0.25, 12.0, 6)
    assert serving.stall_guard_env == "50,0.25,12,6"
    assert serving.stall_guard_basis == "manifest"
    assert serving.to_dict()["stall_guard"]["note"].startswith("a slower reader")


def test_a_voice_may_turn_it_off(a_configured_box: None) -> None:
    serving = parsed(serving_with(
        'stall_guard = false\nstall_guard_note = "an A/B of the unguarded server"'
    )).serving
    assert serving.stall_guard is None
    assert serving.stall_guard_env == "off"
    row = serving.to_dict()["stall_guard"]
    assert row["enabled"] is False and row["env"] == "off" and row["frames"] is None


@pytest.mark.parametrize(
    "block, words",
    [
        ('stall_guard = true\nstall_guard_note = "x"', "says nothing a missing key does not"),
        ('stall_guard = 37\nstall_guard_note = "x"', "must be a table"),
        ('stall_guard = { frames = 37, rate = 0.5, max = 20 }\nstall_guard_note = "x"',
         "missing required key(s) ['window']"),
        ('stall_guard = { frames = 37, rate = 0.5, max = 20, window = 8, extra = 1 }\n'
         'stall_guard_note = "x"', "unknown key(s) ['extra']"),
        ('stall_guard = { frames = 0, rate = 0.5, max = 20, window = 8 }\n'
         'stall_guard_note = "x"', "frames must be in [1, 10000]"),
        ('stall_guard = { frames = 37, rate = 0.5, max = 20, window = 65 }\n'
         'stall_guard_note = "x"', "window must be in [1, 64]"),
        ('stall_guard = { frames = 37, rate = 0.0, max = 20, window = 8 }\n'
         'stall_guard_note = "x"', "rate must be in"),
        ('stall_guard = { frames = 37.5, rate = 0.5, max = 20, window = 8 }\n'
         'stall_guard_note = "x"', "frames must be int"),
        ('stall_guard = { frames = 37, rate = "0.5", max = 20, window = 8 }\n'
         'stall_guard_note = "x"', "rate must be a number"),
        ('stall_guard = { frames = 37, rate = 0.5, max = 20, window = 8 }',
         "carries no stall_guard_note"),
        ('stall_guard_note = "orphaned"', "states stall_guard_note and no stall_guard"),
    ],
)
def test_a_malformed_block_is_refused_by_name(
    a_configured_box: None, block: str, words: str
) -> None:
    said = refused(serving_with(block))
    assert "[voice.serving]" in said
    assert words in said


def test_the_override_survives_the_document_round_trip(a_configured_box: None) -> None:
    for block in (
        'stall_guard = false\nstall_guard_note = "A/B"',
        'stall_guard = { frames = 50, rate = 0.25, max = 12, window = 6 }\n'
        'stall_guard_note = "slow"',
    ):
        manifest = parsed(serving_with(block))
        document, _ = voice_document(manifest)
        serving = document["voice"]["serving"]
        assert "stall_guard_env" not in serving and "stall_guard_basis" not in serving
        import tomli_w
        again = parsed(tomli_w.dumps(document))
        assert again.serving == manifest.serving


def test_the_default_is_not_written_into_a_document(a_configured_box: None) -> None:
    document, _ = voice_document(parsed(GOOD))
    assert "stall_guard" not in document["voice"]["serving"]
    assert "stall_guard_note" not in document["voice"]["serving"]


# ---------------------------------------------------------------- the export


def a_venv(tmp_path: Path) -> Path:
    root = tmp_path / "tts-higgs-v3"
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.write_text("", encoding="utf-8")
    return python


class _Document:
    path = Path("voices.json")

    def environment(self) -> dict[str, str]:
        return {}

    def weights_for(self, voice: str) -> Path:
        return Path("w")


def engine_for(tmp_path: Path, served: bool, stall_guard: str | None, engine: str = "higgs-v3"):
    return build_voice_engine(
        engine, a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni" if served else None,
        max_num_seqs=16 if served else None,
        mem_fraction=None, context_length=None,
        stall_guard=stall_guard,
        voices=_Document(),
        mlx_total_bytes=None if served else 64 * 1024**3,
    )


@pytest.mark.parametrize("served", [True, False], ids=["cuda-linux", "mlx-darwin"])
@pytest.mark.parametrize("value", [DEFAULT_ENV, "off", "50,0.25,12,6"])
def test_the_variable_reaches_narrator_on_both_arms(
    tmp_path: Path, served: bool, value: str
) -> None:
    assert engine_for(tmp_path, served, value).environment()[STALL_GUARD_VARIABLE] == value


@pytest.mark.parametrize("served", [True, False], ids=["cuda-linux", "mlx-darwin"])
def test_a_higgs_engine_without_the_variable_is_refused(tmp_path: Path, served: bool) -> None:
    with pytest.raises(EngineError) as caught:
        engine_for(tmp_path, served, None)
    assert STALL_GUARD_VARIABLE in str(caught.value)


def test_a_malformed_variable_is_refused_before_anything_starts(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        engine_for(tmp_path, True, "37,0.5,20")
    assert STALL_GUARD_VARIABLE in str(caught.value)


def test_the_render_path_states_the_manifests_value(
    a_configured_box: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from crucible import engines
    from crucible.jobs.tts import common

    seen: list[str | None] = []
    monkeypatch.setattr(
        engines, "build_voice_engine", lambda *a, **k: seen.append(k["stall_guard"])
    )
    monkeypatch.setattr(common, "write_document", lambda *a, **k: _Document())
    monkeypatch.setattr(
        common.accelerator, "probe_unified_memory", lambda: (0, 64 * 1024**3)
    )
    python = a_venv(tmp_path)
    for block, expected in (
        (None, DEFAULT_ENV),
        ('stall_guard = false\nstall_guard_note = "A/B"', "off"),
        ('stall_guard = { frames = 50, rate = 0.25, max = 12, window = 6 }\n'
         'stall_guard_note = "slow"', "50,0.25,12,6"),
    ):
        manifest = parsed(GOOD if block is None else serving_with(block))
        for arm in ("cuda-linux", "mlx-darwin"):
            spec = replace(manifest.spec("cuda-linux"), backend=arm)
            seen.clear()
            common._build_narrator(
                tmp_path, manifest, spec, tmp_path, python, tmp_path / "x.log", None, None
            )
            assert seen == [expected], (arm, block)


# ---------------------------------------------------------------- the env patch


def test_the_fixtures_are_the_stock_files() -> None:
    for name in FILES:
        assert envpatches.STALL_GUARD_TAG_FAMILY not in stock(name)
    assert stock("sampler.py").count("K_MAX = 1026\n") == 1


def test_the_patch_applies_to_the_stock_files_and_keeps_snapshots(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    done = run_script(env)
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("PATCHED ") == 3
    after = texts(env)
    for name in FILES:
        assert envpatches.STALL_GUARD_TAG in after[name]
        compile(after[name], name, "exec")
        snapshot = Path(str(package_of(env) / name) + ".orig")
        assert snapshot.read_text(encoding="utf-8") == stock(name)
    rows = envpatches.check("tts", env, TTS_PINS)
    assert [row["status"] for row in rows] == ["applied"] * 3


def test_the_patch_is_idempotent(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    once = {name: (package_of(env) / name).read_bytes() for name in FILES}
    again = run_script(env)
    assert again.returncode == 0, again.stderr
    assert again.stdout.count("ALREADY_PATCHED ") == 3
    assert {name: (package_of(env) / name).read_bytes() for name in FILES} == once


def test_a_drifted_anchor_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    drifted = stock("model_runner.py").replace(
        "pool.step_count[rows_t] = model._cg_active_step_count[:n_real]",
        "pool.step_count[rows_t] = model._cg_active_step_count[:n_real].clone()",
    )
    env = make_env(tmp_path, texts={"model_runner.py": drifted})
    done = run_script(env)
    assert done.returncode == 2
    assert "ANCHOR_NOT_FOUND" in done.stderr
    assert texts(env) == {**{n: stock(n) for n in FILES}, "model_runner.py": drifted}
    assert not list(package_of(env).glob("*.orig"))
    statuses = [row["status"] for row in envpatches.check("tts", env, TTS_PINS)]
    assert statuses == ["missing"] * 3


def test_another_sglang_omni_is_refused(tmp_path: Path) -> None:
    env = make_env(tmp_path, version="0.1.5")
    done = run_script(env)
    assert done.returncode == 2
    assert "VERSION_MISMATCH" in done.stderr
    assert texts(env) == {n: stock(n) for n in FILES}


def test_a_reinstalled_package_gets_the_patch_again(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    for name in FILES:
        with open(package_of(env) / name, "w", encoding="utf-8", newline="") as handle:
            handle.write(stock(name))
    assert {r["status"] for r in envpatches.check("tts", env, TTS_PINS)} == {"missing"}
    done = run_script(env)
    assert done.returncode == 0, done.stderr
    assert {r["status"] for r in envpatches.check("tts", env, TTS_PINS)} == {"applied"}


def test_an_older_version_is_reported_stale_and_rederived_from_the_snapshot(
    tmp_path: Path,
) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    current = texts(env)
    older = envpatches.STALL_GUARD_TAG_FAMILY + "v0, envs/tts/patches/patch_sglang_omni_stall_guard.py)"
    for name in FILES:
        with open(package_of(env) / name, "w", encoding="utf-8", newline="") as handle:
            handle.write(current[name].replace(envpatches.STALL_GUARD_TAG, older))
    assert {r["status"] for r in envpatches.check("tts", env, TTS_PINS)} == {"stale"}
    done = run_script(env)
    assert done.returncode == 0, done.stderr
    assert texts(env) == current


def test_the_table_and_the_script_name_the_same_strings() -> None:
    namespace: dict[str, Any] = {"__name__": "applier"}
    exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"), namespace)
    assert namespace["TAG"] == envpatches.STALL_GUARD_TAG
    assert namespace["TAG_FAMILY"] == envpatches.STALL_GUARD_TAG_FAMILY
    assert namespace["EXPECTED_VERSION"] == TTS_PINS["sglang-omni"]
    assert {p.rel_path for p in envpatches.TTS_PATCHES} == {
        f"{namespace['PACKAGE_REL']}/{name}" for name in FILES
    }


def test_the_patch_is_selected_by_the_recipe() -> None:
    assert "sglang-omni" in TTS_PINS
    assert "sglang-omni" not in MAC_TTS_PINS
    rows = envpatches.check("tts", Path("nowhere"), MAC_TTS_PINS)
    assert {row["status"] for row in rows} == {"not_applicable"}


def test_registry_apply_runs_the_script_once_for_three_files(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    ran: list[list[str]] = []

    def runner(argv: list[str]) -> Any:
        ran.append(argv)
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)

    rows = envpatches.apply("tts", env, Path(sys.executable), TTS_PINS, runner=runner)
    assert len(ran) == 1
    assert [row["status"] for row in rows] == ["applied"] * 3


def test_install_with_nothing_to_do_still_repairs_a_missing_patch(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = jobenv.tts_env("higgs-v3", "cuda-linux")
    calls: list[str] = []
    monkeypatch.setattr(
        jobenv, "plan_install",
        lambda *a, **k: jobenv.EnvPlan(jobenv.PLAN_NOTHING, "matches"),
    )
    monkeypatch.setattr(
        jobenv.envpatches, "repair",
        lambda job_type, *a, **k: calls.append(job_type) or [],
    )
    monkeypatch.setattr(jobenv, "env_status", lambda *a, **k: "status")
    assert jobenv.install_env(home, spec, "cuda-linux") == "status"
    assert calls == ["tts"]


def test_narrator_applies_the_patch_itself_before_it_starts_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(tmp_path)
    shutil.copyfile(sys.executable, env / "bin" / "python")
    (env / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    engine = build_voice_engine(
        "higgs-v3", env / "bin" / "python", tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None, stall_guard=DEFAULT_ENV,
        voices=_Document(), mlx_total_bytes=None,
    )
    monkeypatch.setattr(
        "crucible.engines.narrator.envpatches._run_script",
        lambda argv: subprocess.run(
            [sys.executable, *argv[1:]], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=60,
        ),
    )
    started: list[bool] = []
    monkeypatch.setattr(
        "crucible.engines.base.SubprocessEngine.start",
        lambda self, *a, **k: started.append(True),
    )
    engine.start(tmp_path, "probe", 0, [])
    assert started == [True]
    assert {r["status"] for r in envpatches.check("tts", env, TTS_PINS)} == {"applied"}


def test_narrator_refuses_a_server_it_cannot_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(tmp_path, version="0.1.5")
    (env / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    engine = build_voice_engine(
        "higgs-v3", env / "bin" / "python", tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None, stall_guard=DEFAULT_ENV,
        voices=_Document(), mlx_total_bytes=None,
    )
    monkeypatch.setattr(
        "crucible.engines.narrator.envpatches._run_script",
        lambda argv: subprocess.run(
            [sys.executable, *argv[1:]], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=60,
        ),
    )
    monkeypatch.setattr(
        "crucible.engines.base.SubprocessEngine.start",
        lambda self, *a, **k: pytest.fail("started an unpatched server"),
    )
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path, "probe", 0, [])
    assert "tts_env_unpatched" in str(caught.value)
    assert "VERSION_MISMATCH" in str(caught.value)


# ---------------------------------------------------------------- the row


def test_the_voices_row_reports_the_effective_guard_and_a_put_round_trips_it(
    make_client: Any, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible.api.routes import voices as api_module

    from .test_voice_manifests_api import CUSTOM, OTHER_SHA, a_voice

    monkeypatch.setattr(api_module, "resolve_revision", lambda _config, _repo: OTHER_SHA)
    document = a_voice(revision=OTHER_SHA)
    document["voice"]["serving"]["stall_guard"] = False
    document["voice"]["serving"]["stall_guard_note"] = "an A/B of the unguarded server"
    with make_client(enable_tts=True) as client:
        answer = client.put(f"/v1/voices/{CUSTOM}", json=document, headers=auth)
        assert answer.status_code == 200, answer.text
        guard = answer.json()["voice"]["serving"]["stall_guard"]
        assert guard["enabled"] is False and guard["env"] == "off"
        assert guard["basis"] == "manifest"
        back = client.get(f"/v1/voices/{CUSTOM}/manifest", headers=auth).json()
        assert back["document"]["voice"]["serving"]["stall_guard"] is False

        del document["voice"]["serving"]["stall_guard"]
        del document["voice"]["serving"]["stall_guard_note"]
        answer = client.put(f"/v1/voices/{CUSTOM}", json=document, headers=auth)
        assert answer.status_code == 200, answer.text
        rows = {row["id"]: row for row in client.get("/v1/voices", headers=auth).json()}
        guard = rows[CUSTOM]["serving"]["stall_guard"]
        assert (guard["enabled"], guard["env"], guard["basis"]) == (True, DEFAULT_ENV, "default")
