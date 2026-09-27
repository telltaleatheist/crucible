from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import jobenv
from crucible.jobenv import (
    BACKEND_HEADLINE_PACKAGE,
    EnvError,
    env_dir,
    env_status,
    llm_env,
    recipe_for,
    recipe_pins,
    recipes_dir,
    require_env,
    tts_env,
)
from crucible.voices import NARRATOR_ENGINE_SAMPLING

from .conftest import write_env_stamp


def tts_specs() -> list[jobenv.EnvSpec]:
    seen: dict[str, jobenv.EnvSpec] = {}
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        for backend in sorted(BACKEND_HEADLINE_PACKAGE):
            spec = tts_env(engine, backend)
            seen.setdefault(spec.recipe_name, spec)
    return list(seen.values())


def stamp_env(home: Path, backend_kind: str) -> Path:
    directory = env_dir(home, llm_env(backend_kind))
    (directory / "bin").mkdir(parents=True, exist_ok=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    write_env_stamp(home, llm_env(backend_kind), backend_kind, seconds=196.4)
    return directory


def test_this_build_ships_a_recipe_for_each_backend() -> None:
    assert sorted(p.stem for p in recipes_dir("llm").glob("*.txt")) == [
        "cuda-linux",
        "mlx-darwin",
    ]


def test_a_backend_with_no_recipe_is_refused_by_name() -> None:
    with pytest.raises(EnvError) as caught:
        recipe_for(jobenv.EnvSpec("llm", "llm", "rocm-linux", "vllm"))
    assert "no llm env recipe for 'rocm-linux'" in str(caught.value)
    assert "['cuda-linux', 'mlx-darwin']" in str(caught.value)


def test_each_recipe_pins_its_engine_exactly() -> None:
    assert recipe_pins(recipe_for(llm_env("cuda-linux")))["vllm"] == "0.29.0"
    assert recipe_pins(recipe_for(llm_env("mlx-darwin")))["mlx-lm"] == "0.31.3"


def test_every_requirement_in_every_recipe_is_pinned() -> None:
    for backend in ("cuda-linux", "mlx-darwin"):
        pins = recipe_pins(recipe_for(llm_env(backend)))
        assert len(pins) > 1, f"{backend} pins only its engine"
        for name, version in pins.items():
            assert version, f"{backend}: {name} has no version"


def test_the_engines_the_server_names_are_exactly_the_engines_with_a_recipe() -> None:
    named = sorted(NARRATOR_ENGINE_SAMPLING)
    assert named, "a build that names no narrator engine can serve no voice"
    for engine in named:
        for backend in sorted(BACKEND_HEADLINE_PACKAGE):
            recipe = recipe_for(tts_env(engine, backend))
            assert recipe.is_file(), f"{engine} on {backend}: no {recipe.name}"
    assert {spec.recipe_name for spec in tts_specs()} == {
        path.stem for path in recipes_dir("tts").glob("*.txt")
    }


def test_every_tts_recipe_pins_the_same_narrator_commit() -> None:
    shas = {
        spec.recipe_name: jobenv.recipe_direct_references(recipe_for(spec))["narrator"]
        for spec in tts_specs()
    }
    assert len(set(shas.values())) == 1, shas


def test_the_serving_stack_is_the_recipe_s_and_only_cuda_higgs_has_one() -> None:
    assert tts_env("higgs-v3", "cuda-linux").serving_stack == "sglang-omni"
    assert tts_env("higgs-v3", "mlx-darwin").serving_stack is None
    assert tts_env("not-an-engine", "cuda-linux").serving_stack is None
    assert llm_env("cuda-linux").serving_stack is None


def test_the_stack_named_is_the_stack_the_recipe_installs() -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    text = recipe_for(spec).read_text(encoding="utf-8")
    installed = {
        line.split("==")[0].strip()
        for line in text.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }
    assert spec.serving_stack == "sglang-omni"
    assert "sglang-omni" in installed
    assert "sglang" in installed
    assert "vllm-omni" not in installed
    assert "vllm" not in installed


def test_only_the_sglang_tts_env_wants_an_interpreter_of_its_own() -> None:
    assert tts_env("higgs-v3", "cuda-linux").python_version == "3.12"
    assert tts_env("higgs-v3", "mlx-darwin").python_version is None
    assert llm_env("cuda-linux").python_version is None
    assert llm_env("mlx-darwin").python_version is None


def test_an_env_wanting_another_python_downloads_it_and_never_searches_path(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobenv.sys, "version_info", (3, 11, 16))

    def which(name: str) -> str | None:
        raise AssertionError(f"interpreter_for searched PATH for {name!r}")

    monkeypatch.setattr(jobenv.shutil, "which", which)
    asked: list[tuple[Path, str, str]] = []

    def ensure(home_: Path, backend_kind: str, minor: str, **kw: object) -> Path:
        asked.append((home_, backend_kind, minor))
        return home_ / "interpreters" / "3.12.14" / "bin" / "python"

    monkeypatch.setattr(jobenv.interpreter, "ensure_interpreter", ensure)
    found = jobenv.interpreter_for(
        tts_env("higgs-v3", "cuda-linux"), home, "cuda-linux"
    )
    assert asked == [(home, "cuda-linux", "3.12")]
    assert found == str(home / "interpreters" / "3.12.14" / "bin" / "python")


def test_an_interpreter_whose_digest_does_not_match_is_refused_by_name(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobenv.sys, "version_info", (3, 11, 16))

    def ensure(home_: Path, backend_kind: str, minor: str, **kw: object) -> Path:
        raise jobenv.interpreter.InterpreterError(
            "interpreter_sha_mismatch",
            "cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz "
            "hashes 0000 and crucible/interpreter.py pins 936c",
        )

    monkeypatch.setattr(jobenv.interpreter, "ensure_interpreter", ensure)
    with pytest.raises(jobenv.interpreter.InterpreterError) as caught:
        jobenv.interpreter_for(tts_env("higgs-v3", "cuda-linux"), home, "cuda-linux")
    assert caught.value.code == "interpreter_sha_mismatch"


def test_a_spec_wanting_no_version_takes_the_servers_own_interpreter(
    home: Path,
) -> None:
    assert (
        jobenv.interpreter_for(llm_env("cuda-linux"), home, "cuda-linux")
        == jobenv.sys.executable
    )
    assert (
        jobenv.interpreter_for(tts_env("higgs-v3", "mlx-darwin"), home, "mlx-darwin")
        == jobenv.sys.executable
    )


def test_an_unpinned_requirement_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cuda-linux.txt"
    path.write_text("# a comment\nvllm==0.29.0\ntorch>=2.0\n", encoding="utf-8")
    with pytest.raises(EnvError) as caught:
        recipe_pins(path)
    assert "is not a `name==version` pin" in str(caught.value)


def test_recipe_names_are_normalised(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("huggingface_hub==1.31.0\n", encoding="utf-8")
    assert recipe_pins(path) == {"huggingface-hub": "1.31.0"}


def test_no_venv_is_not_installed(home: Path) -> None:
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "crucible install llm" in status.detail


def test_a_venv_with_no_stamp_is_not_installed(home: Path) -> None:
    directory = env_dir(home, llm_env("cuda-linux"))
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "did not finish" in status.detail


def test_an_env_built_for_another_backend_is_refused(home: Path) -> None:
    stamp_env(home, "mlx-darwin")
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "installed for backend 'mlx-darwin'" in status.detail
    assert "this host is 'cuda-linux'" in status.detail
    assert "--force" in status.detail


def test_an_env_missing_a_pinned_package_is_not_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp_env(home, "cuda-linux")
    pins = recipe_pins(recipe_for(llm_env("cuda-linux")))
    short = {name: version for name, version in pins.items() if name != "torch"}
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: short)
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "torch is absent" in status.detail


def test_an_env_at_the_wrong_version_is_not_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp_env(home, "cuda-linux")
    pins = dict(recipe_pins(recipe_for(llm_env("cuda-linux"))))
    pins["torch"] = "2.5.1"
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: pins)
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "torch is 2.5.1, recipe pins" in status.detail


def test_a_complete_env_is_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp_env(home, "cuda-linux")
    pins = recipe_pins(recipe_for(llm_env("cuda-linux")))
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: dict(pins))
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is True
    assert "vllm 0.29.0" in status.detail
    assert "python 3.11.16" in status.detail
    assert (
        require_env(home, llm_env("cuda-linux"), "cuda-linux")
        == env_dir(home, llm_env("cuda-linux")) / "bin" / "python"
    )


def test_require_env_refuses_rather_than_guessing_an_interpreter(home: Path) -> None:
    with pytest.raises(EnvError) as caught:
        require_env(home, llm_env("cuda-linux"), "cuda-linux")
    assert "crucible install llm" in str(caught.value)


def test_a_recipe_hashes_THE_SAME_whatever_line_endings_it_arrived_with(
    tmp_path: Path,
) -> None:
    import hashlib

    body = "torch==2.5.1\nvllm==0.7.3\nnumpy==1.26.4\n"
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(body.encode())
    crlf.write_bytes(body.replace("\n", "\r\n").encode())
    assert lf.read_bytes() != crlf.read_bytes(), "the two files really do differ"
    assert jobenv.recipe_sha256(lf) == jobenv.recipe_sha256(crlf)
    assert jobenv.recipe_sha256(crlf) == hashlib.sha256(body.encode()).hexdigest()


def test_a_real_edit_STILL_changes_the_recipe_hash(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    first.write_bytes(b"torch==2.5.1\r\nvllm==0.7.3\r\n")
    moved = tmp_path / "b.txt"
    moved.write_bytes(b"torch==2.6.0\r\nvllm==0.7.3\r\n")
    added = tmp_path / "c.txt"
    added.write_bytes(b"torch==2.5.1\r\nvllm==0.7.3\r\nnumpy==1.26.4\r\n")
    removed = tmp_path / "d.txt"
    removed.write_bytes(b"torch==2.5.1\r\n")
    digests = {jobenv.recipe_sha256(p) for p in (first, moved, added, removed)}
    assert len(digests) == 4


def test_the_environment_half_ignores_the_reference_lines_and_nothing_else(
    tmp_path: Path,
) -> None:
    base = "torch==2.13.0\n" + REFERENCE + "\n"
    same = tmp_path / "same.txt"
    same.write_text(base.replace(SHA, "b" * 40), encoding="utf-8")
    first = tmp_path / "first.txt"
    first.write_text(base, encoding="utf-8")
    assert jobenv.environment_sha256(first) == jobenv.environment_sha256(same)
    moved = tmp_path / "moved.txt"
    moved.write_text(base.replace("2.13.0", "2.13.1"), encoding="utf-8")
    assert jobenv.environment_sha256(first) != jobenv.environment_sha256(moved)
    assert jobenv.environment_sha256(first) != jobenv.recipe_sha256(first)


def _recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    root = tmp_path / "recipes"
    (root / "tts").mkdir(parents=True)
    path = root / "tts" / "higgs-v3-cuda-linux.txt"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("CRUCIBLE_RECIPES_DIR", str(root))
    return path


SHA = "4ebc529f30cfa205b820cf494e48fcb76ac12977"
REFERENCE = f"narrator[higgs-v3-server] @ git+https://example.invalid/x@{SHA}#subdirectory=python"


def test_a_direct_reference_is_a_pin_and_is_not_read_as_a_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(tmp_path, monkeypatch, f"{REFERENCE}\ntorch==2.13.0\n")
    assert recipe_pins(path) == {"torch": "2.13.0"}
    assert jobenv.recipe_direct_references(path) == {"narrator": SHA}


def test_a_direct_reference_without_a_commit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(
        tmp_path, monkeypatch, "narrator @ git+https://example.invalid/x@main\n"
    )
    with pytest.raises(EnvError) as caught:
        jobenv.recipe_direct_references(path)
    assert "names no commit" in str(caught.value)
    assert "a branch name is not a pin" in str(caught.value)


def test_a_line_that_is_neither_shape_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(tmp_path, monkeypatch, "torch>=2.13\n")
    with pytest.raises(EnvError) as caught:
        recipe_pins(path)
    assert "every requirement in a recipe is pinned exactly" in str(caught.value)


def test_the_commit_an_env_was_built_from_is_read_off_pips_own_record(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    packages = env_dir(home, spec) / "lib" / "python3.11" / "site-packages"
    dist = packages / "narrator-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://example.invalid/x",
                "vcs_info": {"vcs": "git", "commit_id": SHA},
            }
        ),
        encoding="utf-8",
    )
    (packages / "torch-2.13.0.dist-info").mkdir()
    assert jobenv.installed_direct_references(home, spec) == {"narrator": SHA}


def test_an_env_built_from_another_commit_is_not_ready(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    directory = env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    _recipe(tmp_path, monkeypatch, f"{REFERENCE}\ntorch==2.13.0\n")
    write_env_stamp(home, spec, "cuda-linux")
    monkeypatch.setattr(
        jobenv, "installed_packages", lambda _home, _spec: {"torch": "2.13.0"}
    )
    packages = directory / "lib" / "python3.11" / "site-packages"
    dist = packages / "narrator-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps({"vcs_info": {"vcs": "git", "commit_id": "0" * 40}}),
        encoding="utf-8",
    )
    status = env_status(home, spec, "cuda-linux")
    assert status.installed is False
    assert "narrator was installed from 0000000" in status.detail
    assert SHA in status.detail


NARRATOR_LINE = (
    "narrator[higgs-v3-server] @ git+https://github.com/telltaleatheist/bookforge"
    f"@{SHA}#subdirectory=python"
)
MOVED = "b" * 40
TTS_RECIPE = f"""# the tts env
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.13.0
{NARRATOR_LINE}
"""


def _installed_env(
    home: Path,
    spec: jobenv.EnvSpec,
    backend_kind: str,
    recipe: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    references: dict[str, str] | None = None,
) -> Path:
    directory = env_dir(home, spec)
    (directory / "bin").mkdir(parents=True, exist_ok=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    jobenv._write_stamp(
        home,
        spec,
        backend_kind,
        recipe=recipe,
        python_version="3.12.14",
        seconds=1.0,
        references=references if references is not None
        else jobenv.recipe_direct_references(recipe),
    )
    monkeypatch.setattr(
        jobenv,
        "installed_packages",
        lambda _h, _s: {**recipe_pins(recipe), spec.headline: "0.1.0"},
    )
    monkeypatch.setattr(
        jobenv,
        "installed_direct_references",
        lambda _h, _s: dict(references if references is not None
                            else jobenv.recipe_direct_references(recipe)),
    )
    return directory


def _record_runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    ran: list[list[str]] = []
    monkeypatch.setattr(
        jobenv, "_run", lambda command, failure, on_line: ran.append(list(command))
    )
    monkeypatch.setattr(
        jobenv.shutil,
        "rmtree",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rmtree: the env was deleted")),
    )
    return ran


def test_a_moved_narrator_sha_reinstalls_one_line_and_touches_nothing_else(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    recipe = _recipe(tmp_path, monkeypatch, TTS_RECIPE)
    _installed_env(
        home, spec, "cuda-linux", recipe, monkeypatch,
        references={"narrator": MOVED},
    )
    ran = _record_runs(monkeypatch)
    plan = jobenv.plan_install(home, spec, "cuda-linux")
    assert plan.action == jobenv.PLAN_REFERENCES
    assert plan.lines == (NARRATOR_LINE,)

    jobenv.install_env(home, spec, "cuda-linux")
    python = str(env_dir(home, spec) / "bin" / "python")
    assert ran == [
        [python, "-m", "pip", "install", "--no-deps", "--force-reinstall", NARRATOR_LINE]
    ]
    stamp = json.loads((env_dir(home, spec) / "crucible-env.json").read_text())
    assert stamp["direct_references"] == {"narrator": SHA}
    assert stamp["python_version"] == "3.12.14"
    monkeypatch.setattr(
        jobenv, "installed_direct_references", lambda _h, _s: {"narrator": SHA}
    )
    assert env_status(home, spec, "cuda-linux").installed is True
    assert jobenv.plan_install(home, spec, "cuda-linux").action == jobenv.PLAN_NOTHING


def test_a_moved_environment_half_pips_into_the_venv_that_is_there(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    recipe = _recipe(tmp_path, monkeypatch, TTS_RECIPE)
    _installed_env(home, spec, "cuda-linux", recipe, monkeypatch)
    recipe.write_text(TTS_RECIPE.replace("2.13.0", "2.13.1"), encoding="utf-8")
    monkeypatch.setattr(
        jobenv,
        "installed_packages",
        lambda _h, _s: {"torch": "2.13.1", "narrator": "0.1.0"},
    )
    ran = _record_runs(monkeypatch)
    monkeypatch.setattr(jobenv.envpatches, "apply", lambda *a, **k: None)
    monkeypatch.setattr(
        jobenv.envpatches, "ensure_cuda_toolkit_links", lambda *a, **k: None
    )
    plan = jobenv.plan_install(home, spec, "cuda-linux")
    assert plan.action == jobenv.PLAN_RECIPE

    jobenv.install_env(home, spec, "cuda-linux")
    python = str(env_dir(home, spec) / "bin" / "python")
    assert ran == [[python, "-m", "pip", "install", "-r", str(recipe)]]
    stamp = json.loads((env_dir(home, spec) / "crucible-env.json").read_text())
    assert stamp["environment_sha256"] == jobenv.environment_sha256(recipe)


def test_an_env_that_matches_its_recipe_runs_nothing_at_all(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    recipe = _recipe(tmp_path, monkeypatch, TTS_RECIPE)
    _installed_env(home, spec, "cuda-linux", recipe, monkeypatch)
    ran = _record_runs(monkeypatch)
    assert jobenv.plan_install(home, spec, "cuda-linux").action == jobenv.PLAN_NOTHING
    jobenv.install_env(home, spec, "cuda-linux")
    assert ran == []


def test_force_is_the_one_path_that_deletes_and_rebuilds(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = llm_env("cuda-linux")
    recipe = recipe_for(spec)
    _installed_env(home, spec, "cuda-linux", recipe, monkeypatch)
    ran: list[list[str]] = []
    monkeypatch.setattr(
        jobenv, "_run", lambda command, failure, on_line: ran.append(list(command))
    )
    removed: list[Path] = []
    monkeypatch.setattr(jobenv.shutil, "rmtree", lambda path: removed.append(Path(path)))
    monkeypatch.setattr(
        jobenv.subprocess, "run", lambda *a, **k: _Completed("3.11.16")
    )
    plan = jobenv.plan_install(home, spec, "cuda-linux", force=True)
    assert plan.action == jobenv.PLAN_BUILD

    jobenv.install_env(home, spec, "cuda-linux", force=True)
    assert removed == [env_dir(home, spec)]
    assert ran[0][1:] == ["-m", "venv", str(env_dir(home, spec))]
    assert ran[-1][1:] == ["-m", "pip", "install", "-r", str(recipe)]


class _Completed:
    def __init__(self, out: str) -> None:
        self.stdout = out
        self.returncode = 0


ENVS_ROOT = Path(jobenv.__file__).resolve().parent / "envs"


def every_recipe() -> list[Path]:
    found = sorted(ENVS_ROOT.glob("*/*.txt"))
    assert len(found) >= 10, f"only {len(found)} recipes under {ENVS_ROOT}"
    return found


def test_every_recipe_in_the_tree_states_what_it_costs() -> None:
    for path in every_recipe():
        assert jobenv.recipe_archive_bytes(path) > 0, path


def test_the_sizes_are_the_ones_phase20_measured() -> None:
    def size(*parts: str) -> int:
        return jobenv.recipe_archive_bytes(ENVS_ROOT.joinpath(*parts))

    assert size("tts", "higgs-v3-cuda-linux.txt") == 5_300_000_000
    assert size("llm", "cuda-linux.txt") == 3_300_000_000
    assert size("rvc", "cuda-linux.txt") == 3_300_000_000
    assert size("align", "cuda-linux.txt") == 2_900_000_000
    assert size("asr", "cuda-linux.txt") == 1_300_000_000
    for job_type in ("tts", "llm", "rvc", "align", "asr"):
        assert size(job_type, "mlx-darwin.txt") == 900_000_000


def test_a_recipe_with_no_size_line_is_refused_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(tmp_path, monkeypatch, "torch==2.13.0\n")
    with pytest.raises(EnvError) as caught:
        jobenv.recipe_archive_bytes(path)
    assert "recipe_unsized" in str(caught.value)


def test_a_size_with_nothing_to_cite_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(tmp_path, monkeypatch, "# archive-bytes: 900000000\ntorch==2.13.0\n")
    with pytest.raises(EnvError) as caught:
        jobenv.recipe_archive_bytes(path)
    assert "recipe_size_invalid" in str(caught.value)
    assert "names no source" in str(caught.value)


def test_two_sizes_in_one_recipe_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(
        tmp_path,
        monkeypatch,
        "# archive-bytes: 900000000  measured somewhere\n"
        "# archive-bytes: 5300000000  measured somewhere else\n"
        "torch==2.13.0\n",
    )
    with pytest.raises(EnvError) as caught:
        jobenv.recipe_archive_bytes(path)
    assert "recipe_size_invalid" in str(caught.value)
    assert "One recipe, one size, one owner" in str(caught.value)


def test_a_size_line_is_invisible_to_pip_and_to_the_pin_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(
        tmp_path,
        monkeypatch,
        "# archive-bytes: 900000000  PHASE20 section 0\ntorch==2.13.0\n",
    )
    assert recipe_pins(path) == {"torch": "2.13.0"}
    assert jobenv.recipe_direct_references(path) == {}


class _Usage:

    def __init__(self, free: int) -> None:
        self.total = free * 2
        self.used = free
        self.free = free


def _sized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int) -> Path:
    return _recipe(
        tmp_path,
        monkeypatch,
        f"# archive-bytes: {size}  PHASE20 section 0, measured 2026-09-18\n"
        "torch==2.13.0\n",
    )


def _with_free(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    monkeypatch.setattr(jobenv.shutil, "disk_usage", lambda _path: _Usage(free))


def test_less_free_than_the_recipe_states_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _sized(tmp_path, monkeypatch, 5_300_000_000)
    _with_free(monkeypatch, 4_900_000_000)
    with pytest.raises(EnvError) as caught:
        jobenv.refuse_without_room(
            job_type="tts", recipe=recipe, directory=tmp_path / "envs" / "tts"
        )
    said = str(caught.value)
    assert "env_disk" in said
    assert "'tts'" in said, "the refusal names the type"
    assert "5300000000" in said and "4900000000" in said, "and both byte counts"
    assert str(tmp_path) in said, "and the filesystem it measured"
    assert "at least" in said, "an archive size is a floor: the env unpacks larger"


def test_exactly_enough_is_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _sized(tmp_path, monkeypatch, 5_300_000_000)
    _with_free(monkeypatch, 5_300_000_000)
    jobenv.refuse_without_room(
        job_type="tts", recipe=recipe, directory=tmp_path / "envs" / "tts"
    )


def test_an_unsized_recipe_refuses_the_install_rather_than_letting_it_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(tmp_path, monkeypatch, "torch==2.13.0\n")
    _with_free(monkeypatch, 500_000_000_000)
    with pytest.raises(EnvError) as caught:
        jobenv.refuse_without_room(
            job_type="tts", recipe=path, directory=tmp_path / "envs" / "tts"
        )
    assert "recipe_unsized" in str(caught.value)


def test_the_free_space_read_is_the_one_the_venv_would_land_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[Path] = []

    def usage(path):
        asked.append(Path(path))
        return _Usage(500_000_000_000)

    recipe = _sized(tmp_path, monkeypatch, 1_000)
    monkeypatch.setattr(jobenv.shutil, "disk_usage", usage)
    jobenv.refuse_without_room(
        job_type="tts", recipe=recipe, directory=tmp_path / "a" / "b" / "c"
    )
    assert asked == [tmp_path]


def test_the_install_refuses_before_it_makes_a_venv_or_dials_a_mirror(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = llm_env("cuda-linux")
    ran: list[list[str]] = []
    monkeypatch.setattr(
        jobenv, "_run", lambda command, failure, on_line: ran.append(list(command))
    )
    _with_free(monkeypatch, 1_000_000)
    with pytest.raises(EnvError) as caught:
        jobenv.install_env(home, spec, "cuda-linux")
    assert "env_disk" in str(caught.value)
    assert ran == [], "nothing was run"
    assert not env_dir(home, spec).exists(), "and nothing was made"


FORK_SHA = "05cc3f1ba921f070e16eaf3e1f188073c31c0101"


def _worker_recipe(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cuda-linux.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_rvc_recipe_pins_a_commit_not_a_branch() -> None:
    path = recipe_for(jobenv.worker_env("rvc", "cuda-linux"))
    refs = jobenv.recipe_direct_references(path)
    assert refs["ultimate-rvc"] == FORK_SHA
    assert "ultimate-rvc" not in recipe_pins(path)


def test_both_rvc_recipes_pin_the_same_commit() -> None:
    linux = jobenv.recipe_direct_references(
        recipe_for(jobenv.worker_env("rvc", "cuda-linux"))
    )
    mac = jobenv.recipe_direct_references(
        recipe_for(jobenv.worker_env("rvc", "mlx-darwin"))
    )
    assert linux == mac


def test_the_align_recipe_is_all_version_pins() -> None:
    path = recipe_for(jobenv.worker_env("align", "cuda-linux"))
    assert jobenv.recipe_direct_references(path) == {}
    pins = recipe_pins(path)
    assert pins["qwen-asr"] == "0.0.6"
    assert pins["torch"] == "2.14.0"


def test_every_worker_recipe_that_ships_has_a_headline_package() -> None:
    for job_type in jobenv.WORKER_JOB_TYPES:
        for path in recipes_dir(job_type).glob("*.txt"):
            assert (job_type, path.stem) in jobenv.WORKER_HEADLINE_PACKAGE, path


def test_every_worker_headline_package_is_in_its_own_recipe() -> None:
    for job_type in jobenv.WORKER_JOB_TYPES:
        for path in recipes_dir(job_type).glob("*.txt"):
            headline = jobenv.worker_env(job_type, path.stem).headline
            named = set(recipe_pins(path)) | set(
                jobenv.recipe_direct_references(path)
            )
            assert headline in named, f"{path.name} does not install {headline}"


def test_a_worker_pair_nobody_decided_is_refused_by_name() -> None:
    with pytest.raises(EnvError) as caught:
        jobenv.worker_env("asr", "llama-windows")
    assert "no 'asr' env on 'llama-windows'" in str(caught.value)


def test_a_branch_is_not_a_pin(tmp_path: Path) -> None:
    path = _worker_recipe(
        tmp_path, "ultimate-rvc @ git+https://github.com/x/y@bookforge\n"
    )
    with pytest.raises(EnvError) as caught:
        jobenv.recipe_direct_references(path)
    assert "a branch name is not a pin" in str(caught.value)


def test_index_urls_and_comments_are_not_requirements(tmp_path: Path) -> None:
    path = _worker_recipe(
        tmp_path,
        "# a comment\n--extra-index-url https://example/whl\n\ntorch==2.7.0+cu128\n",
    )
    assert recipe_pins(path) == {"torch": "2.7.0+cu128"}
    assert jobenv.recipe_direct_references(path) == {}


def _stamped_worker(home: Path, job_type: str, backend: str = "cuda-linux") -> Path:
    spec = jobenv.worker_env(job_type, backend)
    recipe = recipe_for(spec)
    directory = env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "recipe": recipe.name,
                "environment_sha256": jobenv.environment_sha256(recipe),
                "direct_references": jobenv.recipe_direct_references(recipe),
                "recipe_text": jobenv.recipe_text(recipe),
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_a_matching_worker_env_names_the_commit_and_not_the_version(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stamped_worker(home, "rvc")
    spec = jobenv.worker_env("rvc", "cuda-linux")
    path = recipe_for(spec)
    monkeypatch.setattr(
        jobenv,
        "installed_packages",
        lambda _h, _s: {**recipe_pins(path), "ultimate-rvc": "0.5.11"},
    )
    monkeypatch.setattr(
        jobenv,
        "installed_direct_references",
        lambda _h, _s: dict(jobenv.recipe_direct_references(path)),
    )
    status = env_status(home, spec, "cuda-linux")
    assert status.installed is True
    assert "ultimate-rvc @ 05cc3f1ba921" in status.detail
    assert "0.5.11" not in status.detail


def test_the_wrong_commit_is_not_ready_and_says_which(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stamped_worker(home, "rvc")
    spec = jobenv.worker_env("rvc", "cuda-linux")
    path = recipe_for(spec)
    monkeypatch.setattr(jobenv, "installed_packages", lambda _h, _s: dict(recipe_pins(path)))
    monkeypatch.setattr(jobenv, "installed_direct_references", lambda _h, _s: {})
    status = env_status(home, spec, "cuda-linux")
    assert status.installed is False
    assert "ultimate-rvc was installed from no recorded commit" in status.detail
    assert FORK_SHA in status.detail


def test_a_headline_that_is_not_installed_is_a_build_bug_said_out_loud(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stamped_worker(home, "align")
    spec = jobenv.worker_env("align", "cuda-linux")
    pins = recipe_pins(recipe_for(spec))
    del pins["qwen-asr"]
    monkeypatch.setattr(jobenv, "installed_packages", lambda _h, _s: dict(pins))
    monkeypatch.setattr(jobenv, "recipe_pins", lambda _p: dict(pins))
    with pytest.raises(EnvError) as caught:
        env_status(home, spec, "cuda-linux")
    assert "the package the align env exists for" in str(caught.value)


def test_a_worker_env_is_refused_the_same_way_when_the_disk_is_short(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(
        jobenv, "_run", lambda command, failure, on_line: ran.append(list(command))
    )
    monkeypatch.setattr(jobenv.shutil, "disk_usage", lambda _p: _Usage(1_000_000))
    spec = jobenv.worker_env("align", "cuda-linux")
    with pytest.raises(EnvError) as caught:
        jobenv.install_env(home, spec, "cuda-linux")
    assert "env_disk" in str(caught.value)
    assert "'align'" in str(caught.value)
    assert ran == [], "nothing was run"
    assert not env_dir(home, spec).exists()
