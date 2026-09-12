"""Backend detection: both viable hosts and the refusals."""

from __future__ import annotations

import subprocess
import sys

import pytest

from crucible import backend as backend_module
from crucible.backend import CUDA_LINUX, MLX_DARWIN, detect_backend, nvidia_smi_path
from crucible.errors import NoViableBackend


def test_cuda_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "linux")
    monkeypatch.setattr(backend_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        backend_module,
        "probe_nvidia_smi",
        lambda: ("NVIDIA GeForce RTX 3090 Ti", 25_757_220_864),
    )
    monkeypatch.setattr(backend_module, "nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")

    detected = detect_backend()
    assert detected.kind == CUDA_LINUX
    assert detected.platform == "linux"
    assert detected.gpu.vendor == "nvidia"
    assert detected.gpu.name == "NVIDIA GeForce RTX 3090 Ti"
    assert detected.gpu.vram_bytes == 25_757_220_864


def test_mlx_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "darwin")
    monkeypatch.setattr(backend_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backend_module, "probe_mlx", lambda: "0.18.0")
    monkeypatch.setattr(
        backend_module,
        "_sysctl",
        lambda name: {
            "machdep.cpu.brand_string": "Apple M2 Ultra",
            "hw.memsize": "137438953472",
        }[name],
    )

    detected = detect_backend()
    assert detected.kind == MLX_DARWIN
    assert detected.arch == "arm64"
    assert detected.gpu.vendor == "apple"
    assert detected.gpu.name == "Apple M2 Ultra"
    assert detected.gpu.vram_bytes == 137_438_953_472
    assert "mlx 0.18.0" in detected.detail


def test_darwin_intel_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "darwin")
    monkeypatch.setattr(backend_module.platform, "machine", lambda: "x86_64")
    with pytest.raises(NoViableBackend) as caught:
        detect_backend()
    assert "x86_64" in caught.value.reason
    assert "arm64" in caught.value.reason


def test_linux_without_nvidia_smi_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "linux")
    monkeypatch.setattr(backend_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(backend_module.os.path, "exists", lambda path: False)
    with pytest.raises(NoViableBackend) as caught:
        detect_backend()
    assert "nvidia-smi" in caught.value.reason
    assert backend_module.WSL_NVIDIA_SMI in caught.value.reason


def test_windows_is_never_a_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "win32")
    with pytest.raises(NoViableBackend) as caught:
        detect_backend()
    assert "WSL2" in caught.value.reason


def test_unknown_platform_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "freebsd14")
    with pytest.raises(NoViableBackend) as caught:
        detect_backend()
    assert "freebsd14" in caught.value.reason


def test_wsl_nvidia_smi_location_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under `wsl.exe --exec bash -c`, nvidia-smi is not on PATH but is at its known place."""
    monkeypatch.setattr(backend_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        backend_module.os.path, "exists", lambda path: path == backend_module.WSL_NVIDIA_SMI
    )
    monkeypatch.setattr(backend_module.os, "access", lambda path, mode: True)
    assert nvidia_smi_path() == backend_module.WSL_NVIDIA_SMI


def test_nvidia_smi_failure_names_the_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=9, stdout="", stderr="driver not loaded"
        )

    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)
    with pytest.raises(NoViableBackend) as caught:
        backend_module.probe_nvidia_smi()
    assert "exited 9" in caught.value.reason
    assert "driver not loaded" in caught.value.reason


def test_nvidia_smi_output_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 3090 Ti, 24564\n",
            stderr="",
        )

    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)
    name, vram = backend_module.probe_nvidia_smi()
    assert name == "NVIDIA GeForce RTX 3090 Ti"
    assert vram == 24564 * 1024 * 1024


def test_mlx_probe_failure_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[sys.executable],
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'mlx'",
        )

    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)
    with pytest.raises(NoViableBackend) as caught:
        backend_module.probe_mlx()
    assert "mlx" in caught.value.reason
