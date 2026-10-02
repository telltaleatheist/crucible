from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from crucible import envpatches
from crucible.envpatches import (
    APPLIED,
    CUDA_TOOLKIT_LINKS,
    CUDA_TOOLKIT_REL,
    MISSING,
    NO_ENV,
    NO_FILE,
    STALE,
    PatchError,
)


def _can_symlink() -> bool:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        (directory / "target").mkdir()
        try:
            (directory / "link").symlink_to("target")
        except OSError:
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _can_symlink(),
    reason="this machine cannot create symlinks (Windows without Developer Mode)",
)


def toolkit(tmp_path: Path) -> Path:
    directory = tmp_path / "env" / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    (directory / "lib").mkdir(parents=True)
    (directory / "lib" / "libcudart.so.13").write_bytes(b"")
    return tmp_path / "env"


def rows(env: Path) -> dict[str, dict]:
    return {row["id"]: row for row in envpatches.check_cuda_toolkit_links(env)}


def test_install_creates_both_and_they_are_relative(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    envpatches.ensure_cuda_toolkit_links(env)
    site = env / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    for link_rel, target in CUDA_TOOLKIT_LINKS:
        link = site / link_rel
        assert link.is_symlink(), link_rel
        assert os.readlink(link) == target
        assert not os.path.isabs(os.readlink(link))
    assert [row["status"] for row in rows(env).values()] == [APPLIED, APPLIED]


def test_running_it_twice_is_a_no_op(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    envpatches.ensure_cuda_toolkit_links(env)
    said: list[str] = []
    envpatches.ensure_cuda_toolkit_links(env, on_line=said.append)
    assert all("already there" in line for line in said), said
    assert [row["status"] for row in rows(env).values()] == [APPLIED, APPLIED]


def test_a_link_pointing_somewhere_else_is_refused_not_replaced(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    site = env / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    (site / "lib64").symlink_to("somewhere-else")
    with pytest.raises(PatchError) as caught:
        envpatches.ensure_cuda_toolkit_links(env)
    assert "somewhere-else" in str(caught.value)
    assert os.readlink(site / "lib64") == "somewhere-else"
    assert rows(env)["cuda-toolkit-lib64"]["status"] == STALE


def test_a_real_directory_where_a_link_belongs_is_refused(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    site = env / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    (site / "lib64").mkdir()
    with pytest.raises(PatchError) as caught:
        envpatches.ensure_cuda_toolkit_links(env)
    assert "not a symlink" in str(caught.value)


def test_an_env_without_the_cuda_wheel_is_refused_by_name(tmp_path: Path) -> None:
    env = tmp_path / "env"
    (env / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    with pytest.raises(PatchError) as caught:
        envpatches.ensure_cuda_toolkit_links(env)
    assert "nvidia-cuda-runtime-cu13" in str(caught.value)
    assert [row["status"] for row in rows(env).values()] == [NO_FILE, NO_FILE]


def test_no_venv_at_all_is_its_own_answer(tmp_path: Path) -> None:
    env = tmp_path / "nothing-here"
    with pytest.raises(PatchError):
        envpatches.ensure_cuda_toolkit_links(env)
    assert [row["status"] for row in rows(env).values()] == [NO_ENV, NO_ENV]


def test_a_missing_link_is_reported_missing_with_what_breaks(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    found = rows(env)
    assert [row["status"] for row in found.values()] == [MISSING, MISSING]
    for row in found.values():
        assert row["applied"] is False
        assert "first render on a card" in row["why"]
