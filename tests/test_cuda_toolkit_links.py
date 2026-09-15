"""The two symlinks pip cannot express, and why their absence is dangerous.

flashinfer JIT-builds SGLang's attention kernels with the CUDA 13 nvcc inside
the pip wheel, and only once that directory looks like a real toolkit: `lib64`
beside `lib`, and an unsuffixed `libcudart.so`. `CUDA_HOME` points at the same
directory and `serve_higgs_sgl.sh` exports it.

WHAT MAKES THEM WORTH A SUITE. Nothing fails at install time without them — pip
is happy, the env stamps installed, and this stack has no site-packages patches
for `doctor` to report on, so the whole narrator-patches section is empty on the
one host where the one silently-missing thing lives. The failure arrives at the
first render on a card. Both were created BY HAND on owens-pc on 2026-09-15, and
`envs/tts/higgs-v3-cuda-linux.txt` claimed from that day that
`crucible/narratorpatches.py` "creates and checks them" while nothing did.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from crucible import narratorpatches
from crucible.narratorpatches import (
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
    """Whether THIS machine can make one, probed rather than assumed.

    The links are a cuda-linux fact, so the obvious guard is `os.name == "nt"` —
    but that skips the suite on a Windows box with Developer Mode on, which can
    make symlinks perfectly well and is where this was written. A capability is
    worth asking about; a platform is a proxy for it that is wrong in both
    directions.
    """
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
    """An env far enough along for the links to belong in it: a venv with the
    CUDA runtime wheel's directory unpacked and no links yet."""
    directory = tmp_path / "env" / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    (directory / "lib").mkdir(parents=True)
    (directory / "lib" / "libcudart.so.13").write_bytes(b"")
    return tmp_path / "env"


def rows(env: Path) -> dict[str, dict]:
    return {row["id"]: row for row in narratorpatches.check_cuda_toolkit_links(env)}


def test_install_creates_both_and_they_are_relative(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    narratorpatches.ensure_cuda_toolkit_links(env)
    site = env / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    for link_rel, target in CUDA_TOOLKIT_LINKS:
        link = site / link_rel
        assert link.is_symlink(), link_rel
        # RELATIVE, verbatim: an absolute target would point at the path the env
        # was built at, so moving the env or copying the guest would break it.
        assert os.readlink(link) == target
        assert not os.path.isabs(os.readlink(link))
    assert [row["status"] for row in rows(env).values()] == [APPLIED, APPLIED]


def test_running_it_twice_is_a_no_op(tmp_path: Path) -> None:
    # `install --force` re-runs the whole build; the second pass must not be an
    # error, and must not replace a link with an identical one either.
    env = toolkit(tmp_path)
    narratorpatches.ensure_cuda_toolkit_links(env)
    said: list[str] = []
    narratorpatches.ensure_cuda_toolkit_links(env, on_line=said.append)
    assert all("already there" in line for line in said), said
    assert [row["status"] for row in rows(env).values()] == [APPLIED, APPLIED]


def test_a_link_pointing_somewhere_else_is_refused_not_replaced(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    site = env / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    (site / "lib64").symlink_to("somewhere-else")
    with pytest.raises(PatchError) as caught:
        narratorpatches.ensure_cuda_toolkit_links(env)
    # Named, and it says what it found — silently overwriting would destroy the
    # evidence of whatever put it there.
    assert "somewhere-else" in str(caught.value)
    assert os.readlink(site / "lib64") == "somewhere-else"
    assert rows(env)["cuda-toolkit-lib64"]["status"] == STALE


def test_a_real_directory_where_a_link_belongs_is_refused(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    site = env / "lib" / "python3.12" / "site-packages" / CUDA_TOOLKIT_REL
    (site / "lib64").mkdir()
    with pytest.raises(PatchError) as caught:
        narratorpatches.ensure_cuda_toolkit_links(env)
    assert "not a symlink" in str(caught.value)


def test_an_env_without_the_cuda_wheel_is_refused_by_name(tmp_path: Path) -> None:
    # The cuda-linux recipe pins nvidia-cuda-runtime-cu13, so an env without the
    # directory did not install what the recipe asked for. Said plainly rather
    # than skipped, which would stamp a broken env installed.
    env = tmp_path / "env"
    (env / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    with pytest.raises(PatchError) as caught:
        narratorpatches.ensure_cuda_toolkit_links(env)
    assert "nvidia-cuda-runtime-cu13" in str(caught.value)
    assert [row["status"] for row in rows(env).values()] == [NO_FILE, NO_FILE]


def test_no_venv_at_all_is_its_own_answer(tmp_path: Path) -> None:
    env = tmp_path / "nothing-here"
    with pytest.raises(PatchError):
        narratorpatches.ensure_cuda_toolkit_links(env)
    assert [row["status"] for row in rows(env).values()] == [NO_ENV, NO_ENV]


def test_a_missing_link_is_reported_missing_with_what_breaks(tmp_path: Path) -> None:
    env = toolkit(tmp_path)
    found = rows(env)
    assert [row["status"] for row in found.values()] == [MISSING, MISSING]
    for row in found.values():
        assert row["applied"] is False
        # The row has to carry the consequence, because the consequence is the
        # only reason a reader would act on it: everything else looks fine.
        assert "first render on a card" in row["why"]
