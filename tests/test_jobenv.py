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


def test_an_env_wanting_another_python_downloads_it_and_never_searches_path(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PHASE20 section 3, item 4: a recipe that names a CPython the server does
    not run gets that CPython FROM THE SAME PUBLISHER, into
    `<home>/interpreters/<version>/`. The PATH search is deleted, not kept as a
    second path — `python3.12` on PATH is a distro's or a conda's, of unknown
    provenance and unknown digest, and a venv inherits whatever it is.
    """
    monkeypatch.setattr(jobenv.sys, "version_info", (3, 11, 16))

    def which(name: str) -> str | None:  # pragma: no cover - must never run
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
    """The download is the only source, so its refusal is the env's refusal:
    it carries the publisher's name and the two digests, and no venv is made."""
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
    """Every env but one, and it is not a fallback: `None` is the answer that
    says "this env is whatever Crucible itself runs on"."""
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


# ------------------------------------------------ how a recipe is hashed
#
# MEASURED 2026-09-15: the same commit of one file hashed to `1ab85cc3…` from
# the main checkout and `cc4fda38…` from a worktree of that SAME commit, while
# `git hash-object` said both were blob `5ef53a3`. CRLF versus LF, on a machine
# with `core.autocrlf=true`. Left alone, an env on a Windows desk calls itself
# drifted from the very recipe it was installed from. These lived in
# `tests/test_envpack.py` until the packs went; the rule they hold is
# `jobenv`'s and always was.


def test_a_recipe_hashes_THE_SAME_whatever_line_endings_it_arrived_with(
    tmp_path: Path,
) -> None:
    """The half that fixes the defect."""
    import hashlib

    body = "torch==2.5.1\nvllm==0.7.3\nnumpy==1.26.4\n"
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(body.encode())
    crlf.write_bytes(body.replace("\n", "\r\n").encode())
    assert lf.read_bytes() != crlf.read_bytes(), "the two files really do differ"
    assert jobenv.recipe_sha256(lf) == jobenv.recipe_sha256(crlf)
    # And the value is the LF one, which is what every Linux and macOS machine
    # computes. A rule that agreed with neither side would call every installed
    # env drifted at once.
    assert jobenv.recipe_sha256(crlf) == hashlib.sha256(body.encode()).hexdigest()


def test_a_real_edit_STILL_changes_the_recipe_hash(tmp_path: Path) -> None:
    """The half that keeps it a hash.

    Normalising a line ENDING cannot erase what is on the line, so every edit a
    recipe can receive — a pin moved, a package added, a line removed — is still
    a different digest.
    """
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
    """THE SPLIT, in one assertion each way.

    A moved narrator sha leaves the environment half alone — that is what makes
    the 1b case one `pip install --no-deps`. A moved torch pin does not.
    """
    base = "torch==2.13.0\n" + REFERENCE + "\n"
    same = tmp_path / "same.txt"
    same.write_text(base.replace(SHA, "b" * 40), encoding="utf-8")
    first = tmp_path / "first.txt"
    first.write_text(base, encoding="utf-8")
    assert jobenv.environment_sha256(first) == jobenv.environment_sha256(same)
    moved = tmp_path / "moved.txt"
    moved.write_text(base.replace("2.13.0", "2.13.1"), encoding="utf-8")
    assert jobenv.environment_sha256(first) != jobenv.environment_sha256(moved)
    # And it is not the whole file's digest: that is the hash the two halves
    # replaced, and reusing it here would be the single answer all over again.
    assert jobenv.environment_sha256(first) != jobenv.recipe_sha256(first)


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


# --------------------------------------------------- what an install DOES now
#
# PHASE20-CODE-NOT-ENVIRONMENTS.md section 4. An env is touched only when its
# recipe moved, and then by pip INTO the existing venv — never a delete and
# rebuild, never a tarball. The two halves of a recipe move for different
# reasons and cost different amounts, so they are stamped and answered apart.


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
    """A venv on disk, stamped exactly the way `install_env` leaves one."""
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
    # `pip list` reports narrator too — it is installed from a git sha, so its
    # presence is a package and its COMMIT is the separate direct-reference
    # check below.
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
    """THE 1b CASE (design section 4). A narrator edit is one git sha in one
    line of one recipe; the 13 GB of torch and SGLang beside it did not move.
    So the environment half is compared on its own, and when only the sha
    moved the answer is one `pip install --no-deps --force-reinstall` of that
    line — not a rebuild, not a `pip install -r` that would re-resolve the lot.
    """
    spec = tts_env("higgs-v3", "cuda-linux")
    recipe = _recipe(tmp_path, monkeypatch, TTS_RECIPE)
    _installed_env(
        home, spec, "cuda-linux", recipe, monkeypatch,
        references={"narrator": MOVED},
    )
    # The recipe now pins a different commit; the stamp records the old one.
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
    # The venv's interpreter did not change, so the stamp keeps the version it
    # already recorded rather than asking a python that was never rebuilt.
    assert stamp["python_version"] == "3.12.14"
    # And once pip has actually done what it was told — `_run` is a fake here,
    # so PEP 610's record is written by this line instead — nothing is left to
    # do: the plan is empty and `crucible doctor` has no drift to report.
    monkeypatch.setattr(
        jobenv, "installed_direct_references", lambda _h, _s: {"narrator": SHA}
    )
    assert env_status(home, spec, "cuda-linux").installed is True
    assert jobenv.plan_install(home, spec, "cuda-linux").action == jobenv.PLAN_NOTHING


def test_a_moved_environment_half_pips_into_the_venv_that_is_there(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pip install -r <recipe>` into the existing venv; pip skips what is
    already satisfied, so the cost is the diff and not the env."""
    spec = tts_env("higgs-v3", "cuda-linux")
    recipe = _recipe(tmp_path, monkeypatch, TTS_RECIPE)
    _installed_env(home, spec, "cuda-linux", recipe, monkeypatch)
    # The environment half moves: a pin, not the direct reference.
    recipe.write_text(TTS_RECIPE.replace("2.13.0", "2.13.1"), encoding="utf-8")
    monkeypatch.setattr(
        jobenv,
        "installed_packages",
        lambda _h, _s: {"torch": "2.13.1", "narrator": "0.1.0"},
    )
    ran = _record_runs(monkeypatch)
    monkeypatch.setattr(jobenv.narratorpatches, "apply", lambda *a, **k: None)
    monkeypatch.setattr(
        jobenv.narratorpatches, "ensure_cuda_toolkit_links", lambda *a, **k: None
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
    """`--force` keeps its meaning for a genuinely broken env — the only
    remaining reason to throw several GB away and start again."""
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

    # The venv's python has to exist after `venv` "ran"; the fake `_run` makes
    # no directory, and the file the fixture wrote is still there.
    jobenv.install_env(home, spec, "cuda-linux", force=True)
    assert removed == [env_dir(home, spec)]
    assert ran[0][1:] == ["-m", "venv", str(env_dir(home, spec))]
    assert ran[-1][1:] == ["-m", "pip", "install", "-r", str(recipe)]


class _Completed:
    def __init__(self, out: str) -> None:
        self.stdout = out
        self.returncode = 0


# ------------------------------------------- what a recipe costs (PHASE19 2.12)
#
# PHASE20 moved the gigabytes off our releases and onto the mirrors, so a first
# install now downloads 5.3 GB of `tts` over minutes and would find out about
# the disk at the end of it. `pack_disk` cannot fire (`_guest_ready` passes no
# `required_bytes`: the move itself is ~31 MB), so the disk question lives here
# now, on the measured archive size each recipe states in its own header.

ENVS_ROOT = Path(jobenv.__file__).resolve().parent / "envs"


def every_recipe() -> list[Path]:
    """Every recipe this build ships, found rather than typed out."""
    found = sorted(ENVS_ROOT.glob("*/*.txt"))
    assert len(found) >= 10, f"only {len(found)} recipes under {ENVS_ROOT}"
    return found


def test_every_recipe_in_the_tree_states_what_it_costs() -> None:
    """A recipe with no size is one `crucible install` has to refuse, so no
    recipe in the tree may be without one."""
    for path in every_recipe():
        assert jobenv.recipe_archive_bytes(path) > 0, path


def test_the_sizes_are_the_ones_phase20_measured() -> None:
    """The five cuda numbers, off `PHASE20-CODE-NOT-ENVIRONMENTS.md` section 0:
    tts 5.3 GB, llm 3.3, rvc 3.3, align 2.9, asr 1.3. An uncited constant is
    invented, so the test cites the same table the headers do."""
    def size(*parts: str) -> int:
        return jobenv.recipe_archive_bytes(ENVS_ROOT.joinpath(*parts))

    assert size("tts", "higgs-v3-cuda-linux.txt") == 5_300_000_000
    assert size("llm", "cuda-linux.txt") == 3_300_000_000
    assert size("rvc", "cuda-linux.txt") == 3_300_000_000
    assert size("align", "cuda-linux.txt") == 2_900_000_000
    assert size("asr", "cuda-linux.txt") == 1_300_000_000
    # The mlx table row prices the five together and never apart, so each mlx
    # recipe carries the total: an overstatement, which is the only direction a
    # floor may be wrong in.
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
    """Where a number was measured is part of the number."""
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
    """It is a comment, so nothing that reads requirements trips over it."""
    path = _recipe(
        tmp_path,
        monkeypatch,
        "# archive-bytes: 900000000  PHASE20 section 0\ntorch==2.13.0\n",
    )
    assert recipe_pins(path) == {"torch": "2.13.0"}
    assert jobenv.recipe_direct_references(path) == {}


class _Usage:
    """What `shutil.disk_usage` answers, with the free space a test chose."""

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
    """No fallback: not knowing what it costs is not "probably fine"."""
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
    """`~/.crucible/envs/<key>/` does not exist yet, so the nearest existing
    parent is what can be measured — and it is on the same filesystem."""
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
    """The whole point of the guard: pip is never started.

    A fresh env is `PLAN_BUILD`, and the check sits ahead of `python -m venv`,
    ahead of `interpreter_for` (which downloads a CPython) and therefore ahead
    of every byte pip would fetch.
    """
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
