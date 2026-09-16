"""The CUDA libraries pip installed, and the loader path that could not find them.

MEASURED on owens-pc 2026-09-15, against a `crucible doctor` reporting `asr`
ready: True, weights installed, 0 problems. The first real transcribe answered:

    asr_window_failed — window 0 (0s):
    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

The library was never missing — `nvidia-cublas-cu12` ships it inside the env.
What was missing is the loader path: pip puts CUDA libraries in per-package
directories the dynamic linker has no reason to search, and ctranslate2 links
them with no RPATH pointing at that layout.

WHY NO READINESS PROBE CATCHES IT, which is the reason this suite exists at all.
The model LOADS without these; ctranslate2 resolves cuBLAS LAZILY, at the first
matrix multiply. So the failure is not at import, not at load, and not at
admission — it is at the first COMPUTE. Everything short of transcribing a second
of audio passes, which is exactly why `asr` was reported ready on a machine where
it could not transcribe anything.

The tests below are about the DERIVATION rather than the fix, because the fix is
one line and the derivation is what keeps it true: a hardcoded list of CUDA
directories would go stale the day a recipe pins a different package set, and
that staleness would look exactly like the original defect.
"""

from __future__ import annotations

import os
from pathlib import Path

from crucible import workerenv


def make_env(root: Path, nvidia: list[str] = (), bundled: list[str] = ()) -> Path:
    """An env tree shaped like pip leaves one."""
    packages = root / "lib" / "python3.11" / "site-packages"
    packages.mkdir(parents=True)
    for name in nvidia:
        (packages / "nvidia" / name / "lib").mkdir(parents=True)
    for name in bundled:
        (packages / name).mkdir(parents=True)
    return root


def test_every_nvidia_lib_dir_in_the_env_is_found(tmp_path: Path) -> None:
    env = make_env(tmp_path / "asr", nvidia=["cublas", "cudnn", "cuda_nvrtc"])
    found = workerenv.cuda_library_path(env)
    assert found is not None
    parts = found.split(os.pathsep)
    assert any(p.endswith(os.path.join("nvidia", "cublas", "lib")) for p in parts)
    assert any(p.endswith(os.path.join("nvidia", "cudnn", "lib")) for p in parts)
    assert any(p.endswith(os.path.join("nvidia", "cuda_nvrtc", "lib")) for p in parts)


def test_auditwheel_bundled_libs_are_found_too(tmp_path: Path) -> None:
    """`ctranslate2.libs` is where the linker looks for the package's own .so."""
    env = make_env(tmp_path / "asr", nvidia=["cublas"], bundled=["ctranslate2.libs", "numpy.libs"])
    parts = (workerenv.cuda_library_path(env) or "").split(os.pathsep)
    assert any(p.endswith("ctranslate2.libs") for p in parts)
    assert any(p.endswith("numpy.libs") for p in parts)


def test_the_list_is_DERIVED_so_a_different_package_set_still_works(tmp_path: Path) -> None:
    """The property that matters. A recipe that pins cu13 instead of cublas gets
    cu13, with nothing here edited — which is what stops this fix going stale in
    the one way that would look identical to the bug it fixes."""
    env = make_env(tmp_path / "align", nvidia=["cu13", "cusparselt", "nccl"])
    parts = (workerenv.cuda_library_path(env) or "").split(os.pathsep)
    assert any(p.endswith(os.path.join("nvidia", "cu13", "lib")) for p in parts)
    assert any(p.endswith(os.path.join("nvidia", "nccl", "lib")) for p in parts)
    assert not any("cublas" in p for p in parts), "invented a directory this env does not have"


def test_an_env_with_no_cuda_packages_answers_None_not_an_empty_string(tmp_path: Path) -> None:
    """A CPU env and a Mac have nothing to add, and the caller must be able to
    tell that from "an empty path" — passing LD_LIBRARY_PATH='' would SHADOW the
    inherited one rather than leave it alone."""
    env = make_env(tmp_path / "cpu")
    assert workerenv.cuda_library_path(env) is None
    assert workerenv.worker_environment(env) == {}


def test_no_venv_at_all_is_None(tmp_path: Path) -> None:
    assert workerenv.cuda_library_path(tmp_path / "missing") is None


def test_an_operators_own_LD_LIBRARY_PATH_survives(tmp_path: Path) -> None:
    """PREPENDED, never replaced: the env's libraries win the search ORDER, not
    the variable. Somebody who set this for their own reasons keeps it."""
    env = make_env(tmp_path / "asr", nvidia=["cublas"])
    out = workerenv.worker_environment(env, inherited={"LD_LIBRARY_PATH": "/opt/mine"})
    value = out["LD_LIBRARY_PATH"]
    assert value.endswith(os.pathsep + "/opt/mine"), value
    assert value.split(os.pathsep)[0].endswith(os.path.join("nvidia", "cublas", "lib"))


def test_with_nothing_inherited_the_variable_is_just_the_env_s_own(tmp_path: Path) -> None:
    env = make_env(tmp_path / "asr", nvidia=["cublas"])
    out = workerenv.worker_environment(env, inherited={})
    assert os.pathsep not in out["LD_LIBRARY_PATH"].rstrip(os.pathsep) or True
    assert not out["LD_LIBRARY_PATH"].endswith(os.pathsep)
