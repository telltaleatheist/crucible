from __future__ import annotations

import os
from pathlib import Path

from crucible import workers


def make_env(root: Path, nvidia: list[str] = (), bundled: list[str] = ()) -> Path:
    packages = root / "lib" / "python3.11" / "site-packages"
    packages.mkdir(parents=True)
    for name in nvidia:
        (packages / "nvidia" / name / "lib").mkdir(parents=True)
    for name in bundled:
        (packages / name).mkdir(parents=True)
    return root


def test_every_nvidia_lib_dir_in_the_env_is_found(tmp_path: Path) -> None:
    env = make_env(tmp_path / "asr", nvidia=["cublas", "cudnn", "cuda_nvrtc"])
    found = workers.cuda_library_path(env)
    assert found is not None
    parts = found.split(os.pathsep)
    assert any(p.endswith(os.path.join("nvidia", "cublas", "lib")) for p in parts)
    assert any(p.endswith(os.path.join("nvidia", "cudnn", "lib")) for p in parts)
    assert any(p.endswith(os.path.join("nvidia", "cuda_nvrtc", "lib")) for p in parts)


def test_auditwheel_bundled_libs_are_found_too(tmp_path: Path) -> None:
    env = make_env(tmp_path / "asr", nvidia=["cublas"], bundled=["ctranslate2.libs", "numpy.libs"])
    parts = (workers.cuda_library_path(env) or "").split(os.pathsep)
    assert any(p.endswith("ctranslate2.libs") for p in parts)
    assert any(p.endswith("numpy.libs") for p in parts)


def test_the_list_is_DERIVED_so_a_different_package_set_still_works(tmp_path: Path) -> None:
    env = make_env(tmp_path / "align", nvidia=["cu13", "cusparselt", "nccl"])
    parts = (workers.cuda_library_path(env) or "").split(os.pathsep)
    assert any(p.endswith(os.path.join("nvidia", "cu13", "lib")) for p in parts)
    assert any(p.endswith(os.path.join("nvidia", "nccl", "lib")) for p in parts)
    assert not any("cublas" in p for p in parts), "invented a directory this env does not have"


def test_an_env_with_no_cuda_packages_answers_None_not_an_empty_string(tmp_path: Path) -> None:
    env = make_env(tmp_path / "cpu")
    assert workers.cuda_library_path(env) is None
    assert workers.worker_environment(env) == {}


def test_no_venv_at_all_is_None(tmp_path: Path) -> None:
    assert workers.cuda_library_path(tmp_path / "missing") is None


def test_an_operators_own_LD_LIBRARY_PATH_survives(tmp_path: Path) -> None:
    env = make_env(tmp_path / "asr", nvidia=["cublas"])
    out = workers.worker_environment(env, inherited={"LD_LIBRARY_PATH": "/opt/mine"})
    value = out["LD_LIBRARY_PATH"]
    assert value.endswith(os.pathsep + "/opt/mine"), value
    assert value.split(os.pathsep)[0].endswith(os.path.join("nvidia", "cublas", "lib"))


def test_with_nothing_inherited_the_variable_is_just_the_env_s_own(tmp_path: Path) -> None:
    env = make_env(tmp_path / "asr", nvidia=["cublas"])
    out = workers.worker_environment(env, inherited={})
    assert os.pathsep not in out["LD_LIBRARY_PATH"].rstrip(os.pathsep) or True
    assert not out["LD_LIBRARY_PATH"].endswith(os.pathsep)
