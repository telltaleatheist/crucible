from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crucible import jobenv

REF = "pkg @ git+https://example.invalid/pkg@" + "a" * 40


def _recipe(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "x-mlx-darwin.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_directive_names_a_line_of_the_same_recipe(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, f"torch==2.14.0\n# crucible: no-deps pkg\n{REF}\n")
    assert jobenv.recipe_no_deps(recipe) == (REF,)
    assert jobenv.recipe_no_deps(_recipe(tmp_path, f"torch==2.14.0\n{REF}\n")) == ()
    with pytest.raises(jobenv.EnvError, match="does not list"):
        jobenv.recipe_no_deps(_recipe(tmp_path, "torch==2.14.0\n# crucible: no-deps nothere\n"))


def test_the_named_line_is_installed_apart_without_its_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yue2-infer pins torch==2.10.0; on the Mac the recipe pins 2.14.0, and one
    `pip install -r` would refuse the pair."""
    recipe = _recipe(tmp_path, f"torch==2.14.0\nnumpy==2.2.6\n# crucible: no-deps pkg\n{REF}\n")
    ran: list[list[str]] = []
    seen_rest: list[str] = []

    def run(argv: list[str], _why: str, _on_line: Any) -> None:
        ran.append(argv)
        if "-r" in argv:
            seen_rest.append(Path(argv[argv.index("-r") + 1]).read_text(encoding="utf-8"))

    monkeypatch.setattr(jobenv, "_run", run)
    monkeypatch.setattr(jobenv.envpatches, "apply", lambda *a, **k: None)
    spec = jobenv.EnvSpec(job_type="audio", key="audio-x", recipe_name="x-mlx-darwin", headline="pkg")
    jobenv._install_recipe(spec, "mlx-darwin", Path("python"), recipe, tmp_path, None)
    assert seen_rest == ["torch==2.14.0\nnumpy==2.2.6\n"]
    assert ran[-1] == ["python", "-m", "pip", "install", "--no-deps", REF]
    assert not list(tmp_path.glob(".*.with-deps.txt")), "the split recipe is removed"
