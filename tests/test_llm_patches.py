"""The `llm` env's one patch: mlx-lm's `top_logprobs` ceiling, 11 -> 40.

PHASE22-DECIDE.md section 2.6. The applier runs here against a VERBATIM excerpt
of the installed mlx-lm 0.31.3 `mlx_lm/server.py` from the Mac Studio
(`tests/fixtures/mlx_lm_0.31.3_server_validate.py.txt` is lines 1204-1257 —
`_validate` and `validate_model_parameters` — of the file whose sha256 was
cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b, read over
`ssh mac` on 2026-09-23), so the anchor is proven against the real bytes and not
a paraphrase of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import cli, envpatches, jobenv, narratorpatches
from crucible.engines import EngineError, decide_reading
from crucible.engines.mlx_lm import MlxLmEngine

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mlx_lm_0.31.3_server_validate.py.txt"
SCRIPT = envpatches.LLM_SCRIPTS_DIR / envpatches.MLX_LM_TOP_LOGPROBS.script
PATCH = envpatches.MLX_LM_TOP_LOGPROBS
MAC_PINS = jobenv.recipe_pins(jobenv.recipe_for(jobenv.llm_env("mlx-darwin")))
CUDA_PINS = jobenv.recipe_pins(jobenv.recipe_for(jobenv.llm_env("cuda-linux")))


def pristine() -> str:
    # newline="" so the fixture's own LF bytes are what the applier meets,
    # whatever this checkout's autocrlf did to the working file.
    return FIXTURE.read_text(encoding="utf-8").replace("\r\n", "\n")


def make_env(root: Path, server_text: str | None = None) -> Path:
    """A venv-shaped directory holding `mlx_lm/server.py` and `bin/python`."""
    env = root / "llm-env"
    target = env / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "server.py"
    target.parent.mkdir(parents=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(pristine() if server_text is None else server_text)
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("", encoding="utf-8")
    return env


def server_of(env: Path) -> Path:
    return env / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "server.py"


def run_script(env: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(env)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _script_namespace() -> dict:
    """The applier's module globals, without running `main()`."""
    namespace: dict = {"__name__": "patch_mlx_lm_top_logprobs"}
    exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"), namespace)
    return namespace


# ------------------------------------------------------------------- applier


def test_the_fixture_is_the_stock_validator() -> None:
    """The excerpt holds the anchor exactly once and no patched line."""
    text = pristine()
    assert text.count(PATCH.absent_marker) == 1
    assert PATCH.marker not in text


def test_the_applier_raises_the_cap_to_40_and_keeps_a_snapshot(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    done = run_script(env)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("PATCHED ")
    text = server_of(env).read_text(encoding="utf-8")
    assert text.count(PATCH.marker) == 1
    assert PATCH.absent_marker not in text
    # Only the one line moved: everything else in the excerpt is byte-identical.
    script = _script_namespace()
    assert text.replace(script["NEW"], script["OLD"]) == pristine()
    # `.orig` is the live file as it was, kept for reference.
    assert Path(str(server_of(env)) + ".orig").read_text(encoding="utf-8") == pristine()
    [row] = narratorpatches.check(env, MAC_PINS, patches=envpatches.LLM_PATCHES)
    assert row["status"] == narratorpatches.APPLIED


def test_the_applier_is_idempotent(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    once = server_of(env).read_bytes()
    again = run_script(env)
    assert again.returncode == 0, again.stderr
    assert again.stdout.startswith("ALREADY_PATCHED ")
    assert server_of(env).read_bytes() == once


def test_a_moved_anchor_is_refused_by_name_and_touches_nothing(tmp_path: Path) -> None:
    """A newer mlx-lm that reworded the validator must be re-patched on purpose."""
    moved = pristine().replace("max_val=11, whitelist=[-1]", "max_val=20, whitelist=[-1]")
    env = make_env(tmp_path, moved)
    done = run_script(env)
    assert done.returncode == 2
    assert "ANCHOR_NOT_FOUND" in done.stderr
    assert server_of(env).read_text(encoding="utf-8") == moved
    assert not Path(str(server_of(env)) + ".orig").exists()
    [row] = narratorpatches.check(env, MAC_PINS, patches=envpatches.LLM_PATCHES)
    assert row["status"] == narratorpatches.MISSING


def test_no_server_py_is_refused_by_name(tmp_path: Path) -> None:
    done = run_script(tmp_path / "nothing-here")
    assert done.returncode != 0
    assert "NOT_FOUND" in done.stderr


def test_the_script_and_the_table_name_the_same_strings() -> None:
    """The two halves cannot drift apart in silence."""
    namespace = _script_namespace()
    assert namespace["REL"] == PATCH.rel_path
    assert namespace["MARKER"] == PATCH.marker
    assert namespace["ABSENT_MARKER"] == PATCH.absent_marker
    assert PATCH.marker in namespace["NEW"]
    assert namespace["OLD"].strip() == PATCH.absent_marker


# ----------------------------------------------------------------- registry


def test_the_tts_table_is_the_one_it_always_was() -> None:
    found = envpatches.patches_for("tts")
    assert found is not None
    assert found.patches is narratorpatches.NARRATOR_PATCHES
    assert found.scripts_dir == narratorpatches.SCRIPTS_DIR
    assert envpatches.patches_for("asr") is None
    assert envpatches.check("asr", Path("nowhere"), {}) == []


def test_the_patch_is_selected_by_the_recipe_not_by_a_backend_name(tmp_path: Path) -> None:
    """mlx-darwin pins mlx-lm; cuda-linux (vLLM) and llama-windows (no recipe) do not."""
    assert PATCH.distribution in MAC_PINS
    assert PATCH.distribution not in CUDA_PINS
    env = make_env(tmp_path)
    for pins in (CUDA_PINS, {}):
        [row] = envpatches.check("llm", env, pins)
        assert row["status"] == narratorpatches.NOT_APPLICABLE
        # And apply runs nothing there: the file is still stock.
        assert envpatches.apply("llm", env, Path(sys.executable), pins)[0]["status"] == (
            narratorpatches.NOT_APPLICABLE
        )
        assert server_of(env).read_text(encoding="utf-8") == pristine()


def test_apply_through_the_registry_patches_and_proves_it(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    said: list[str] = []
    rows = envpatches.apply("llm", env, Path(sys.executable), MAC_PINS, on_line=said.append)
    assert [row["status"] for row in rows] == [narratorpatches.APPLIED]
    assert any(line.startswith("PATCHED") for line in said)


def _llm_install_ready(home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[jobenv.EnvSpec, Path]:
    """A stamped mac llm env whose plan is `pip install -r`, with pip stubbed."""
    spec = jobenv.llm_env("mlx-darwin")
    directory = jobenv.env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("", encoding="utf-8")
    recipe = jobenv.recipe_for(spec)
    jobenv._write_stamp(
        home, spec, "mlx-darwin", recipe=recipe, python_version="3.11.16",
        seconds=1.0, references=jobenv.recipe_direct_references(recipe),
    )
    monkeypatch.setattr(
        jobenv, "plan_install",
        lambda *a, **k: jobenv.EnvPlan(jobenv.PLAN_RECIPE, "test: the recipe moved"),
    )
    monkeypatch.setattr(jobenv, "_run", lambda command, failure, on_line: None)
    # The status read after the stamp asks the env's pip, which is an empty file.
    monkeypatch.setattr(jobenv, "env_status", lambda *a, **k: "status")
    return spec, directory


def test_install_env_applies_the_llm_table_before_the_stamp(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`install_env` asks the registry by job type — `llm` now has a table."""
    spec, directory = _llm_install_ready(home, monkeypatch)
    seen: list[tuple[str, Path, dict]] = []
    monkeypatch.setattr(
        envpatches, "apply",
        lambda job_type, env_dir, python, pins, **k: seen.append((job_type, env_dir, pins)) or [],
    )
    jobenv.install_env(home, spec, "mlx-darwin")
    assert [(job, env) for job, env, _ in seen] == [("llm", directory)]
    assert seen[0][2]["mlx-lm"] == "0.31.3"


def test_an_llm_patch_that_will_not_go_in_leaves_no_new_stamp(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _llm_install_ready(home, monkeypatch)
    stamp = jobenv.stamp_path(home, spec)
    before = stamp.read_bytes()

    def refuse(*a, **k):
        raise narratorpatches.PatchError("mlx-lm-top-logprobs-40: ANCHOR_NOT_FOUND")

    monkeypatch.setattr(envpatches, "apply", refuse)
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv.install_env(home, spec, "mlx-darwin")
    assert "ANCHOR_NOT_FOUND" in str(caught.value)
    assert stamp.read_bytes() == before


# -------------------------------------------------------------- engine cap


def test_the_engine_states_40_and_the_door_reads_40() -> None:
    assert MlxLmEngine.max_logprobs == 40
    reading = decide_reading("mlx-lm")
    assert reading.served is True
    assert reading.max_logprobs == 40
    assert "mlx-lm-top-logprobs-40" in reading.basis


def test_an_unpatched_env_is_refused_at_engine_start(tmp_path: Path) -> None:
    """40 is never advertised by an engine that would answer 400 above 11."""
    env = make_env(tmp_path)
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "weights", "m", 0, [])
    message = str(caught.value)
    assert message.startswith("llm_env_unpatched:")
    assert "mlx-lm-top-logprobs-40 is missing" in message
    assert "crucible env patch llm" in message
    assert not (tmp_path / "e.log").exists(), "nothing was spawned"


def test_a_patched_env_passes_the_gate(tmp_path: Path) -> None:
    """Past the patch check, the base class's own refusals take over."""
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "no-weights", "m", 0, [])
    assert "no model directory" in str(caught.value)


# ------------------------------------------------------------------ doctor


def _mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)


def _mac_llm_env(home: Path, server_text: str) -> Path:
    env = jobenv.env_dir(home, jobenv.llm_env("mlx-darwin"))
    target = env / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "server.py"
    target.parent.mkdir(parents=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(server_text)
    return env


def _doctor(capsys: pytest.CaptureFixture[str]) -> dict:
    cli.main(["doctor", "--json"])
    return json.loads(capsys.readouterr().out)


def test_doctor_reports_an_unpatched_mac_llm_env_as_a_problem(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    _mac_llm_env(home, pristine())
    report = _doctor(capsys)
    [row] = report["llm_patches"]
    assert row["id"] == "mlx-lm-top-logprobs-40"
    assert row["status"] == "missing"
    assert any(p.startswith("llm_patch[mlx-lm-top-logprobs-40]: missing") for p in report["problems"])


def test_doctor_reports_a_patched_mac_llm_env_as_sound(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    env = _mac_llm_env(home, pristine())
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("", encoding="utf-8")
    assert run_script(env).returncode == 0
    report = _doctor(capsys)
    [row] = report["llm_patches"]
    assert row["status"] == "applied"
    assert not any("llm_patch" in p for p in report["problems"])


def test_doctor_says_not_applicable_on_cuda_linux(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    report = _doctor(capsys)
    [row] = report["llm_patches"]
    assert row["status"] == "not_applicable"
    assert not any("llm_patch" in p for p in report["problems"])


def test_doctor_does_not_repeat_a_missing_env_as_a_patch_problem(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    report = _doctor(capsys)
    [row] = report["llm_patches"]
    assert row["status"] == "no_env"
    assert not any("llm_patch" in p for p in report["problems"])


# ----------------------------------------------------------- env patch verb


def test_env_patch_llm_applies_and_exits_zero(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    env = _mac_llm_env(home, pristine())
    python = jobenv.env_python(home, jobenv.llm_env("mlx-darwin"))
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    # The env's own interpreter is an empty file here; the applier is run by
    # THIS interpreter instead, which is the only substitution.
    real = narratorpatches._run_script
    monkeypatch.setattr(
        narratorpatches, "_run_script", lambda argv: real([sys.executable, *argv[1:]])
    )
    capsys.readouterr()
    assert cli.main(["env", "patch", "llm"]) == 0
    out = capsys.readouterr().out
    assert "llm patch (mlx-lm-top-logprobs-40): applied" in out
    assert PATCH.marker in server_of(env).read_text(encoding="utf-8")


def test_env_patch_llm_refuses_by_name_when_the_anchor_moved(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    _mac_llm_env(home, pristine().replace("max_val=11,", "max_val=12,"))
    python = jobenv.env_python(home, jobenv.llm_env("mlx-darwin"))
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    real = narratorpatches._run_script
    monkeypatch.setattr(
        narratorpatches, "_run_script", lambda argv: real([sys.executable, *argv[1:]])
    )
    capsys.readouterr()
    assert cli.main(["env", "patch", "llm"]) == 1
    err = capsys.readouterr().err
    assert "env_patch_failed" in err
    assert "ANCHOR_NOT_FOUND" in err


def test_env_patch_llm_with_no_env_has_nothing_to_do(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    assert cli.main(["env", "patch", "llm"]) == 0
    assert "nothing to patch" in capsys.readouterr().out
