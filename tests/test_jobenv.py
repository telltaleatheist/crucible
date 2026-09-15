"""A job type's env recipes and its status (PHASE2-LLM.md section 2).

None of these build a venv — that is minutes of pip. They test what the server
reads *about* an env, which is what decides whether a load is refused with
`env_missing`.

The `tts` envs are here too, because their naming rule is the interesting half:
`cuda-linux` names one venv per narrator engine and `mlx-darwin` has one for
every engine (PHASE3-TTS.md section 4), and a rule with a branch in it is a rule
that needs a test on each side.
"""

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


def tts_specs() -> list[jobenv.EnvSpec]:
    """Every `tts` env this build can be asked to install, deduplicated.

    Read off `NARRATOR_ENGINE_SAMPLING` and `BACKEND_HEADLINE_PACKAGE` rather
    than typed out, so a test cannot keep passing about an engine the server
    stopped naming — or quietly skip one it started naming.
    """
    seen: dict[str, jobenv.EnvSpec] = {}
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        for backend in sorted(BACKEND_HEADLINE_PACKAGE):
            spec = tts_env(engine, backend)
            seen.setdefault(spec.recipe_name, spec)
    return list(seen.values())


def stamp_env(home: Path, backend_kind: str) -> Path:
    """A venv directory shaped the way `crucible install llm` leaves one."""
    directory = env_dir(home, llm_env(backend_kind))
    (directory / "bin").mkdir(parents=True, exist_ok=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": f"{backend_kind}.txt",
                "python_version": "3.11.16",
                "seconds": 196.4,
            }
        ),
        encoding="utf-8",
    )
    return directory


# ------------------------------------------------------------------ recipes


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
    """A recipe with a floating requirement is not a recipe."""
    for backend in ("cuda-linux", "mlx-darwin"):
        pins = recipe_pins(recipe_for(llm_env(backend)))
        assert len(pins) > 1, f"{backend} pins only its engine"
        for name, version in pins.items():
            assert version, f"{backend}: {name} has no version"


def test_the_engines_the_server_names_are_exactly_the_engines_with_a_recipe() -> None:
    """THE DRIFT GUARD. A listed engine with no recipe is an unservable choice.

    Crucible listed `orpheus` as a narrator engine from PHASE3-TTS.md until
    Owen's ruling of 2026-09-14, and the operator page built on 2026-09-13 drew
    its `tts` engine picker straight off that list — so the page offered an
    engine the server would never serve. The defect was not the engine; it was
    that the list of engines and the recipes on disk were two answers to "what
    can this host install" with nothing comparing them (docs/ARCHITECTURE.md
    section 1).

    This is the comparison, in BOTH directions:

    - every engine the server names resolves to a recipe FILE on every backend,
      so no picker can offer a choice `crucible install tts` cannot build;
    - every recipe file under `envs/tts/` is named by one of those pairs, so a
      recipe for a deleted engine cannot linger and read as support for it.
    """
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
    """narrator is ONE package, and a recipe is a statement about bytes.

    Every `tts` recipe carries its own direct reference because each names a
    different extra, so the sha is written once per recipe — and N copies of
    one fact is the shape `docs/ARCHITECTURE.md` §1 names. A bump that lands on
    some of them gives a host whose envs speak two wires, and `crucible doctor`
    calls both installed because each matches the recipe that built it.
    """
    shas = {
        spec.recipe_name: jobenv.recipe_direct_references(recipe_for(spec))["narrator"]
        for spec in tts_specs()
    }
    assert len(set(shas.values())) == 1, shas


def test_the_serving_stack_is_the_recipe_s_and_only_cuda_higgs_has_one() -> None:
    """`HIGGS_STACK` is refused by name by narrator and has no default there.

    It is stated from the ENV SPEC rather than from the voice, because which
    server narrator can start is a property of what the recipe installed:
    `higgs-v3-cuda-linux.txt` carries `sglang-omni==0.1.4` and no vllm-omni at
    all, since Owen's ruling of 2026-09-15 ("we dont use vllm-omni. we use
    sglang. vllm-omni doesnt work for higgs").
    `None` on `mlx-darwin` is a real answer, not a gap: narrator renders in
    process there and reads none of it. So is `None` for an engine with no row
    in `CUDA_LINUX_SERVING_STACK` — one that loads its own runtime.
    """
    assert tts_env("higgs-v3", "cuda-linux").serving_stack == "sglang-omni"
    assert tts_env("higgs-v3", "mlx-darwin").serving_stack is None
    # An engine with no row in the table starts no server, and the lookup says
    # so rather than raising. This is the shape the next engine arrives in.
    assert tts_env("not-an-engine", "cuda-linux").serving_stack is None
    assert llm_env("cuda-linux").serving_stack is None


def test_the_stack_named_is_the_stack_the_recipe_installs() -> None:
    """The two copies of one fact, compared. A recipe that swapped vllm-omni
    for SGLang-Omni while this table still said `vllm-omni` would start a
    server whose requests narrator is not building — not a crash, a book
    rendered at whatever the dropped fields defaulted to.

    THIS TEST IS WHY THE FLIP IS ONE EDIT AND NOT A SEARCH. It failed the
    moment the recipe changed and passed again only when the table did."""
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
    # AND NOT A TRACE OF THE OTHER ONE. Owen's ruling is that vllm-omni does
    # not work for Higgs, so an env that carries it is an env that can render a
    # damaged book — the recipe is replaced, never kept beside a second file.
    assert "vllm-omni" not in installed
    assert "vllm" not in installed


def test_only_the_sglang_tts_env_wants_an_interpreter_of_its_own() -> None:
    """`RECIPE_PYTHON` is keyed by RECIPE because the requirement belongs to
    what is installed. sglang-omni 0.1.4 pulls torch 2.13.0+cu130 against
    python 3.12; every other env is the server's own interpreter, and `None`
    says exactly that."""
    assert tts_env("higgs-v3", "cuda-linux").python_version == "3.12"
    assert tts_env("higgs-v3", "mlx-darwin").python_version is None
    assert llm_env("cuda-linux").python_version is None
    assert llm_env("mlx-darwin").python_version is None


def test_an_env_wanting_an_interpreter_this_host_lacks_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused BEFORE `venv` runs. A venv inherits its maker's version, so a
    3.11 interpreter cannot produce a 3.12 env — it produces a 3.11 one that
    pip fails to fill several GB in, with a wheel-compatibility error naming
    neither the env nor the reason."""
    monkeypatch.setattr(jobenv.shutil, "which", lambda name: None)
    monkeypatch.setattr(jobenv.sys, "version_info", (3, 11, 16))
    with pytest.raises(EnvError) as caught:
        jobenv.interpreter_for(tts_env("higgs-v3", "cuda-linux"))
    message = str(caught.value)
    assert "python 3.12" in message
    assert "3.11" in message
    # It names the way out rather than only the problem.
    assert "uv python install 3.12" in message


def test_an_interpreter_of_the_wanted_version_on_path_is_used() -> None:
    """The second of the two sources, and the only one a host that is not
    already running 3.12 can offer."""
    spec = tts_env("higgs-v3", "cuda-linux")

    def which(name: str) -> str | None:
        return "/usr/bin/python3.12" if name == "python3.12" else None

    import unittest.mock as mock
    with mock.patch.object(jobenv.shutil, "which", which):
        with mock.patch.object(jobenv.sys, "version_info", (3, 11, 16)):
            assert jobenv.interpreter_for(spec) == "/usr/bin/python3.12"


def test_a_spec_wanting_no_version_takes_the_servers_own_interpreter() -> None:
    """Every env but one, and it is not a fallback: `None` is the answer that
    says "this env is whatever Crucible itself runs on"."""
    assert jobenv.interpreter_for(llm_env("cuda-linux")) == jobenv.sys.executable
    assert (
        jobenv.interpreter_for(tts_env("higgs-v3", "mlx-darwin"))
        == jobenv.sys.executable
    )


def test_an_unpinned_requirement_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cuda-linux.txt"
    path.write_text("# a comment\nvllm==0.29.0\ntorch>=2.0\n", encoding="utf-8")
    with pytest.raises(EnvError) as caught:
        recipe_pins(path)
    assert "is not a `name==version` pin" in str(caught.value)


def test_recipe_names_are_normalised(tmp_path: Path) -> None:
    """`huggingface_hub` in a freeze is `huggingface-hub` to pip list."""
    path = tmp_path / "x.txt"
    path.write_text("huggingface_hub==1.31.0\n", encoding="utf-8")
    assert recipe_pins(path) == {"huggingface-hub": "1.31.0"}


# ------------------------------------------------------------------- status


def test_no_venv_is_not_installed(home: Path) -> None:
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "crucible install llm" in status.detail


def test_a_venv_with_no_stamp_is_not_installed(home: Path) -> None:
    """An interrupted install must never be handed to an engine."""
    directory = env_dir(home, llm_env("cuda-linux"))
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "did not finish" in status.detail


def test_an_env_built_for_another_backend_is_refused(home: Path) -> None:
    """The wrong-backend refusal: an mlx env on a CUDA host is not an llm env."""
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
    """Drift into a different torch is a refusal, not a shrug."""
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


# ------------------------------------------------- pinning what is not on PyPI


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
    """narrator is not on PyPI, so it cannot be `name==version`.

    `pip list` reports its declared version, which does not move when the commit
    does — so a version check here would call an env built from last month's
    commit a match. The two shapes are therefore read by two functions, and
    `recipe_pins` must not report the reference as a version pin.
    """
    path = _recipe(tmp_path, monkeypatch, f"{REFERENCE}\ntorch==2.13.0\n")
    assert recipe_pins(path) == {"torch": "2.13.0"}
    assert jobenv.recipe_direct_references(path) == {"narrator": SHA}


def test_a_direct_reference_without_a_commit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch is a moving target; an env built from one cannot be said to
    match the recipe that built it."""
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
    """PEP 610: pip writes `direct_url.json` beside a distribution installed from
    a URL, and for a VCS install it carries the commit pip actually resolved."""
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
    # A package installed from an index has no such file and is not reported.
    (packages / "torch-2.13.0.dist-info").mkdir()
    assert jobenv.installed_direct_references(home, spec) == {"narrator": SHA}


def test_an_env_built_from_another_commit_is_not_ready(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    directory = env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": "cuda-linux",
                "recipe": "higgs-v3-cuda-linux.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    _recipe(tmp_path, monkeypatch, f"{REFERENCE}\ntorch==2.13.0\n")
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
