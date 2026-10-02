from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import cli, envpatches, jobenv
from crucible.engines import EngineError, decide_reading
from crucible.engines.mlx_lm import REQUIRED_FLAGS, MlxLmEngine

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mlx_lm_0.31.3_server_validate.py.txt"
SCRIPT = envpatches.LLM_SCRIPTS_DIR / envpatches.MLX_LM_TOP_LOGPROBS.script
PATCH = envpatches.MLX_LM_TOP_LOGPROBS
GEN_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mlx_lm_0.31.3_generate.py.txt"
GEN_SHA256 = "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f"
FP32 = envpatches.MLX_LM_FP32_LOGPROBS
FP32_SCRIPT = envpatches.LLM_SCRIPTS_DIR / FP32.script
MLX_ARGS = [
    "--decode-concurrency", "16", "--prompt-concurrency", "4", "--prompt-cache-size", "10",
]
MAC_PINS = jobenv.recipe_pins(jobenv.recipe_for(jobenv.llm_env("mlx-darwin")))
CUDA_PINS = jobenv.recipe_pins(jobenv.recipe_for(jobenv.llm_env("cuda-linux")))
ALL_SCRIPTS = tuple(
    envpatches.LLM_SCRIPTS_DIR / patch.script for patch in envpatches.LLM_PATCHES
)
ALL_IDS = [patch.id for patch in envpatches.LLM_PATCHES]
STOCK_GENERATION_THREAD = (
    "\n\nclass ResponseGenerator:\n"
    "    def __init__(self, model_provider, prompt_cache):\n"
    "        self.model_provider = model_provider\n"
    "        self._generation_thread = Thread(target=self._generate)\n"
    "        self._generation_thread.start()\n"
)

STOCK_ITEMS_ANCHORS = (
    "\n    def _generate(self):\n"
    "        while not self._stop:\n"
    "            request = None\n"
    "            # We got a request\n"
    "            if request is not None:\n"
    "                rqueue, request, args = request\n"
    "\n\nclass APIHandler(BaseHTTPRequestHandler):\n"
    "    def do_POST(self):\n"
    "        request_factories = {\n"
    '            "/v1/completions": self.handle_text_completions,\n'
    "        }\n"
)
ITEMS = envpatches.MLX_LM_DECIDE_ITEMS
ITEMS_HELPER = envpatches.MLX_LM_DECIDE_ITEMS_HELPER
ITEMS_SCRIPT = envpatches.LLM_SCRIPTS_DIR / ITEMS.script
ITEMS_HELPER_SCRIPT = envpatches.LLM_SCRIPTS_DIR / ITEMS_HELPER.script


def pristine() -> str:
    validator = FIXTURE.read_text(encoding="utf-8").replace("\r\n", "\n")
    return validator + STOCK_GENERATION_THREAD + STOCK_ITEMS_ANCHORS


def pristine_generate() -> str:
    return GEN_FIXTURE.read_text(encoding="utf-8").replace("\r\n", "\n")


def write_mlx_lm(
    package: Path,
    server_text: str,
    generate_text: str | None = None,
    version: str = "0.31.3",
) -> None:
    package.mkdir(parents=True)
    files = {
        "server.py": server_text,
        "generate.py": pristine_generate() if generate_text is None else generate_text,
        "_version.py": f'# Copyright\n\n__version__ = "{version}"\n',
    }
    for name, body in files.items():
        with open(package / name, "w", encoding="utf-8", newline="") as handle:
            handle.write(body)


def make_env(
    root: Path,
    server_text: str | None = None,
    generate_text: str | None = None,
    version: str = "0.31.3",
) -> Path:
    env = root / "llm-env"
    write_mlx_lm(
        env / "lib" / "python3.11" / "site-packages" / "mlx_lm",
        pristine() if server_text is None else server_text,
        generate_text,
        version,
    )
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("", encoding="utf-8")
    return env


def server_of(env: Path) -> Path:
    return env / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "server.py"


def generate_of(env: Path) -> Path:
    return env / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "generate.py"


def run_script(env: Path, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), str(env)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def patch_all(env: Path) -> None:
    for script in ALL_SCRIPTS:
        done = run_script(env, script)
        assert done.returncode == 0, done.stderr


def _script_namespace(script: Path = SCRIPT) -> dict:
    namespace: dict = {"__name__": script.stem}
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    return namespace


def test_the_fixture_is_the_stock_validator() -> None:
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
    script = _script_namespace()
    assert text.replace(script["NEW"], script["OLD"]) == pristine()
    assert Path(str(server_of(env)) + ".orig").read_text(encoding="utf-8") == pristine()
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(PATCH,))
    assert row["status"] == envpatches.APPLIED


def test_the_applier_is_idempotent(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    once = server_of(env).read_bytes()
    again = run_script(env)
    assert again.returncode == 0, again.stderr
    assert again.stdout.startswith("ALREADY_PATCHED ")
    assert server_of(env).read_bytes() == once


def test_a_moved_anchor_is_refused_by_name_and_touches_nothing(tmp_path: Path) -> None:
    moved = pristine().replace("max_val=11, whitelist=[-1]", "max_val=20, whitelist=[-1]")
    env = make_env(tmp_path, moved)
    done = run_script(env)
    assert done.returncode == 2
    assert "ANCHOR_NOT_FOUND" in done.stderr
    assert server_of(env).read_text(encoding="utf-8") == moved
    assert not Path(str(server_of(env)) + ".orig").exists()
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(PATCH,))
    assert row["status"] == envpatches.MISSING


def test_no_server_py_is_refused_by_name(tmp_path: Path) -> None:
    done = run_script(tmp_path / "nothing-here")
    assert done.returncode != 0
    assert "NOT_FOUND" in done.stderr


def test_the_script_and_the_table_name_the_same_strings() -> None:
    namespace = _script_namespace()
    assert namespace["REL"] == PATCH.rel_path
    assert namespace["MARKER"] == PATCH.marker
    assert namespace["ABSENT_MARKER"] == PATCH.absent_marker
    assert PATCH.marker in namespace["NEW"]
    assert namespace["OLD"].strip() == PATCH.absent_marker


def test_only_the_llm_env_carries_patches() -> None:
    assert envpatches.patched_job_types() == ("llm",)
    assert envpatches.patches_for("tts") is None
    assert envpatches.patches_for("asr") is None
    assert envpatches.check("asr", Path("nowhere"), {}) == []


def test_the_patch_is_selected_by_the_recipe_not_by_a_backend_name(tmp_path: Path) -> None:
    assert PATCH.distribution in MAC_PINS
    assert PATCH.distribution not in CUDA_PINS
    env = make_env(tmp_path)
    assert FP32.distribution in MAC_PINS and FP32.distribution not in CUDA_PINS
    for pins in (CUDA_PINS, {}):
        rows = envpatches.check("llm", env, pins)
        assert [row["status"] for row in rows] == [envpatches.NOT_APPLICABLE] * len(ALL_IDS)
        assert [
            row["status"] for row in envpatches.apply("llm", env, Path(sys.executable), pins)
        ] == [envpatches.NOT_APPLICABLE] * len(ALL_IDS)
        assert server_of(env).read_text(encoding="utf-8") == pristine()
        assert generate_of(env).read_text(encoding="utf-8") == pristine_generate()


def test_apply_through_the_registry_patches_and_proves_it(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    said: list[str] = []
    rows = envpatches.apply("llm", env, Path(sys.executable), MAC_PINS, on_line=said.append)
    assert [row["id"] for row in rows] == ALL_IDS
    assert [row["status"] for row in rows] == [envpatches.APPLIED] * len(ALL_IDS)
    assert sum(line.startswith("PATCHED") for line in said) == len(ALL_IDS)


def _llm_install_ready(home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[jobenv.EnvSpec, Path]:
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
    monkeypatch.setattr(jobenv, "env_status", lambda *a, **k: "status")
    return spec, directory


def test_install_env_applies_the_llm_table_before_the_stamp(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        raise envpatches.PatchError("mlx-lm-top-logprobs-40: ANCHOR_NOT_FOUND")

    monkeypatch.setattr(envpatches, "apply", refuse)
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv.install_env(home, spec, "mlx-darwin")
    assert "ANCHOR_NOT_FOUND" in str(caught.value)
    assert stamp.read_bytes() == before


def test_the_engine_states_40_and_the_door_reads_40() -> None:
    assert MlxLmEngine.max_logprobs == 40
    reading = decide_reading("mlx-lm")
    assert reading.served is True
    assert reading.max_logprobs == 40
    assert "mlx-lm-top-logprobs-40" in reading.basis


def test_an_unpatched_env_is_refused_at_engine_start(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "weights", "m", 0, MLX_ARGS)
    message = str(caught.value)
    assert message.startswith("llm_env_unpatched:")
    assert "mlx-lm-top-logprobs-40 is missing" in message
    assert "crucible env patch llm" in message
    assert not (tmp_path / "e.log").exists(), "nothing was spawned"


def test_an_env_without_the_fp32_patch_is_refused_at_engine_start(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "weights", "m", 0, MLX_ARGS)
    message = str(caught.value)
    assert message.startswith("llm_env_unpatched:")
    assert "mlx-lm-fp32-logprobs is missing" in message
    assert not (tmp_path / "e.log").exists(), "nothing was spawned"


def test_a_patched_env_passes_the_gate(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    patch_all(env)
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "no-weights", "m", 0, MLX_ARGS)
    assert "no model directory" in str(caught.value)


@pytest.mark.parametrize("missing", REQUIRED_FLAGS)
def test_an_argv_that_does_not_state_its_batch_is_refused_at_engine_start(
    tmp_path: Path, missing: str
) -> None:
    env = make_env(tmp_path)
    patch_all(env)
    args = list(MLX_ARGS)
    at = args.index(missing)
    del args[at : at + 2]
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "weights", "m", 0, args)
    message = str(caught.value)
    assert message.startswith("mlx_lm_flags_unstated:")
    assert missing in message
    assert not (tmp_path / "e.log").exists(), "nothing was spawned"


def test_the_generate_fixture_is_the_stock_file() -> None:
    raw = GEN_FIXTURE.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == GEN_SHA256
    text = pristine_generate()
    assert text.count(FP32.absent_marker) == 3
    assert FP32.marker not in text


def test_the_fp32_applier_patches_every_returned_site_and_keeps_a_snapshot(
    tmp_path: Path,
) -> None:
    env = make_env(tmp_path)
    done = run_script(env, FP32_SCRIPT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("PATCHED ")
    text = generate_of(env).read_text(encoding="utf-8")
    assert text.count(FP32.marker) == 1
    assert FP32.absent_marker not in text
    assert text.count("logprobs, returned = _crucible_logprobs(logits, None)") == 1
    assert text.count("logprobs, returned = _crucible_logprobs(logits, -1)") == 2
    assert "return sampled, returned.squeeze(0)" in text
    assert "return y, returned" in text
    assert "self._next_logprobs = list(returned)" in text
    assert "sampled = sampler(logprobs)" in text
    assert "y = sampler(logprobs)" in text
    assert "sampled = sample_sampler(logprobs[e : e + 1])" in text
    assert "sampled = self.fallback_sampler(logprobs)" in text
    script = _script_namespace(FP32_SCRIPT)
    undone = text
    for old, new in [(script["HELPER_ANCHOR"], script["HELPER"])] + list(script["EDITS"]):
        undone = undone.replace(new, old)
    assert undone == pristine_generate()
    assert Path(str(generate_of(env)) + ".orig").read_text(encoding="utf-8") == (
        pristine_generate()
    )
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(FP32,))
    assert row["status"] == envpatches.APPLIED
    compile(text, "generate.py", "exec")


def test_the_fp32_applier_is_idempotent(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env, FP32_SCRIPT).returncode == 0
    once = generate_of(env).read_bytes()
    again = run_script(env, FP32_SCRIPT)
    assert again.returncode == 0, again.stderr
    assert again.stdout.startswith("ALREADY_PATCHED ")
    assert generate_of(env).read_bytes() == once


def test_the_fp32_applier_refuses_another_mlx_lm_version(tmp_path: Path) -> None:
    env = make_env(tmp_path, version="0.31.4")
    done = run_script(env, FP32_SCRIPT)
    assert done.returncode == 2
    assert "VERSION_MISMATCH" in done.stderr
    assert generate_of(env).read_text(encoding="utf-8") == pristine_generate()
    assert not Path(str(generate_of(env)) + ".orig").exists()


def test_the_fp32_applier_writes_nothing_when_one_site_moved(tmp_path: Path) -> None:
    moved = pristine_generate().replace(
        "        self._next_logprobs = list(logprobs)\n",
        "        self._next_logprobs = [lp for lp in logprobs]\n",
    )
    env = make_env(tmp_path, generate_text=moved)
    done = run_script(env, FP32_SCRIPT)
    assert done.returncode == 2
    assert "ANCHOR_NOT_FOUND" in done.stderr
    assert generate_of(env).read_text(encoding="utf-8") == moved
    assert not Path(str(generate_of(env)) + ".orig").exists()
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(FP32,))
    assert row["status"] == envpatches.MISSING


def test_a_surviving_stock_site_reads_stale_not_applied(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env, FP32_SCRIPT).returncode == 0
    text = generate_of(env).read_text(encoding="utf-8")
    text += "\nlogprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)\n"
    generate_of(env).write_text(text, encoding="utf-8")
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(FP32,))
    assert row["status"] == envpatches.STALE


def test_the_fp32_script_and_the_table_name_the_same_strings() -> None:
    namespace = _script_namespace(FP32_SCRIPT)
    assert namespace["REL"] == FP32.rel_path
    assert namespace["MARKER"] == FP32.marker
    assert namespace["ABSENT_MARKER"] == FP32.absent_marker
    assert FP32.marker in namespace["HELPER"]
    assert FP32.absent_marker not in namespace["HELPER"]
    assert all(FP32.absent_marker not in new for _, new in namespace["EDITS"])
    assert namespace["EXPECTED_VERSION"] == MAC_PINS["mlx-lm"]


def _mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)


def _mac_llm_env(home: Path, server_text: str) -> Path:
    env = jobenv.env_dir(home, jobenv.llm_env("mlx-darwin"))
    write_mlx_lm(env / "lib" / "python3.11" / "site-packages" / "mlx_lm", server_text)
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
    rows = {row["id"]: row for row in report["llm_patches"]}
    assert set(rows) == set(ALL_IDS)
    assert all(row["status"] == "missing" for row in rows.values())
    assert any(p.startswith("llm_patch[mlx-lm-top-logprobs-40]: missing") for p in report["problems"])
    assert any(p.startswith("llm_patch[mlx-lm-fp32-logprobs]: missing") for p in report["problems"])


def test_doctor_reports_a_patched_mac_llm_env_as_sound(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    env = _mac_llm_env(home, pristine())
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("", encoding="utf-8")
    patch_all(env)
    report = _doctor(capsys)
    assert [row["status"] for row in report["llm_patches"]] == ["applied"] * len(ALL_IDS)
    assert not any("llm_patch" in p for p in report["problems"])


def test_doctor_says_not_applicable_on_cuda_linux(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    report = _doctor(capsys)
    assert [row["status"] for row in report["llm_patches"]] == (
        ["not_applicable"] * len(ALL_IDS)
    )
    assert not any("llm_patch" in p for p in report["problems"])


def test_doctor_does_not_repeat_a_missing_env_as_a_patch_problem(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    report = _doctor(capsys)
    assert [row["status"] for row in report["llm_patches"]] == ["no_env"] * len(ALL_IDS)
    assert not any("llm_patch" in p for p in report["problems"])


def test_env_patch_llm_applies_and_exits_zero(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    env = _mac_llm_env(home, pristine())
    python = jobenv.env_python(home, jobenv.llm_env("mlx-darwin"))
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    real = envpatches._run_script
    monkeypatch.setattr(
        envpatches, "_run_script", lambda argv: real([sys.executable, *argv[1:]])
    )
    capsys.readouterr()
    assert cli.main(["env", "patch", "llm"]) == 0
    out = capsys.readouterr().out
    assert "llm patch (mlx-lm-top-logprobs-40): applied" in out
    for patch_id in ALL_IDS:
        assert f"llm patch ({patch_id}): applied" in out
    assert PATCH.marker in server_of(env).read_text(encoding="utf-8")
    assert FP32.marker in generate_of(env).read_text(encoding="utf-8")


def test_env_patch_llm_refuses_by_name_when_the_anchor_moved(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mac(monkeypatch)
    assert cli.main(["init", "--enable-llm"]) == 0
    _mac_llm_env(home, pristine().replace("max_val=11,", "max_val=12,"))
    python = jobenv.env_python(home, jobenv.llm_env("mlx-darwin"))
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    real = envpatches._run_script
    monkeypatch.setattr(
        envpatches, "_run_script", lambda argv: real([sys.executable, *argv[1:]])
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


def helper_of(env: Path) -> Path:
    return env / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "_crucible_items.py"


def test_the_items_applier_adds_the_route_and_the_job_and_keeps_a_snapshot(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    done = run_script(env, ITEMS_SCRIPT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("PATCHED ")
    text = server_of(env).read_text(encoding="utf-8")
    assert text.count(ITEMS.marker) == 1
    assert text.count("_crucible_items.mlx_lm_serve_http(self)") == 1
    assert '"MlxLmItemsJob"' in text and "drain_batch = True" in text
    assert Path(str(server_of(env)) + ".orig").read_text(encoding="utf-8") == pristine()
    again = run_script(env, ITEMS_SCRIPT)
    assert again.stdout.startswith("ALREADY_PATCHED ")
    assert server_of(env).read_text(encoding="utf-8") == text
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(ITEMS,))
    assert row["status"] == envpatches.APPLIED


def test_the_items_applier_writes_nothing_when_an_anchor_moved(tmp_path: Path) -> None:
    moved = pristine().replace("# We got a request", "# A request arrived")
    env = make_env(tmp_path, moved)
    done = run_script(env, ITEMS_SCRIPT)
    assert done.returncode == 2 and "ANCHOR_NOT_FOUND" in done.stderr
    assert server_of(env).read_text(encoding="utf-8") == moved


def test_the_helper_is_items_forward_verbatim_and_placed_once(tmp_path: Path) -> None:
    from crucible.engines import items_forward

    env = make_env(tmp_path)
    done = run_script(env, ITEMS_HELPER_SCRIPT)
    assert done.returncode == 0 and done.stdout.startswith("PATCHED ")
    assert helper_of(env).read_bytes() == Path(items_forward.__file__).read_bytes()
    assert run_script(env, ITEMS_HELPER_SCRIPT).stdout.startswith("ALREADY_PATCHED ")
    assert ITEMS_HELPER.marker == f"ITEMS_VERSION = {items_forward.ITEMS_VERSION}"
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(ITEMS_HELPER,))
    assert row["status"] == envpatches.APPLIED


def _older_four(env: Path) -> None:
    for patch in envpatches.LLM_PATCHES:
        if patch not in envpatches.SELF_APPLIED_LLM_PATCHES:
            assert run_script(env, envpatches.LLM_SCRIPTS_DIR / patch.script).returncode == 0


def _scripts_with_this_python(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    ran: list[list[str]] = []

    def run(argv: list[str]) -> subprocess.CompletedProcess:
        ran.append(argv)
        return subprocess.run(
            [sys.executable, *argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=60, check=False,
        )

    monkeypatch.setattr(envpatches, "_run_script", run)
    return ran


def test_the_engine_applies_the_items_patches_itself_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(tmp_path)
    _older_four(env)
    ran = _scripts_with_this_python(monkeypatch)
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "no-weights", "m", 0, MLX_ARGS)
    assert "no model directory" in str(caught.value)
    assert [Path(argv[1]).name for argv in ran] == [ITEMS.script, ITEMS_HELPER.script]
    assert ITEMS.marker in server_of(env).read_text(encoding="utf-8")
    assert helper_of(env).is_file()


def test_an_items_patch_that_will_not_go_in_is_refused_by_name_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(tmp_path, pristine().replace("# We got a request", "# A request arrived"))
    _older_four(env)
    _scripts_with_this_python(monkeypatch)
    engine = MlxLmEngine(python=env / "bin" / "python", log_path=tmp_path / "e.log")
    with pytest.raises(EngineError) as caught:
        engine.start(tmp_path / "weights", "m", 0, MLX_ARGS)
    message = str(caught.value)
    assert message.startswith("llm_env_unpatched:") and "mlx-lm-decide-items" in message
    assert "crucible env patch llm" in message
    assert not (tmp_path / "e.log").exists(), "nothing was spawned"
