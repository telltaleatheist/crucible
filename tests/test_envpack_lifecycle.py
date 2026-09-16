"""Release smoke must reject an older core, without starting a service."""
import subprocess
from pathlib import Path

import pytest

from crucible import envpack


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("name,backend", [("server", "cuda-linux"), ("server", "mlx-darwin"), ("host", "llama-windows")])
def test_relocated_core_requires_lifecycle_commands(monkeypatch, tmp_path, missing, name, backend):
    target = envpack.pack_target(name, backend)
    def extract(archive, into, **kw):
        python = envpack.pack_python(into, backend)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        entry = into / "crucible.cmd" if name == "host" else into / "bin" / "crucible"
        entry.touch()
    monkeypatch.setattr(envpack, "extract_archive", extract)
    commands = []
    def run(argv, **kw):
        commands.append(argv[1:])
        assert Path(kw["cwd"]).name.startswith("crucible-smoke-")
        if "init" in argv:
            home = Path(kw["env"]["CRUCIBLE_HOME"])
            home.mkdir()
            (home / "config.toml").touch()
        return subprocess.CompletedProcess(argv, 2 if missing and "local" in argv else 0,
                                           stdout="help", stderr="unknown local" if missing else "")
    monkeypatch.setattr(envpack.subprocess, "run", run)
    if missing:
        with pytest.raises(envpack.PackError, match="lacks the lifecycle entrypoint"):
            envpack.smoke_test(target, tmp_path / "pack.tar.zst")
    else:
        envpack.smoke_test(target, tmp_path / "pack.tar.zst")
        assert ["local", "install-desktop", "--help"] in commands
        assert ["local", "shutdown", "--help"] in commands
        if backend != "cuda-linux":
            assert ["init", "--backend", backend] in commands
        else:
            assert not any("init" in args for args in commands)


@pytest.mark.parametrize("failure", ["missing-mlx", "no-config"])
def test_relocated_mac_core_must_initialize_in_an_isolated_home(monkeypatch, tmp_path, failure):
    target = envpack.pack_target("server", "mlx-darwin")
    live_home = tmp_path / "live-home"
    monkeypatch.setenv("CRUCIBLE_HOME", str(live_home))
    monkeypatch.setenv("PYTHONPATH", "unrelated-checkout")
    monkeypatch.setenv("PYTHONHOME", "unrelated-runtime")
    def extract(archive, into, **kw):
        python = envpack.pack_python(into, target.backend_kind)
        python.parent.mkdir(parents=True)
        python.touch()
        (into / "bin" / "crucible").touch()
    monkeypatch.setattr(envpack, "extract_archive", extract)
    def run(argv, **kw):
        if "init" in argv:
            environment = kw["env"]
            home = Path(environment["CRUCIBLE_HOME"])
            assert home.parent == Path(kw["cwd"])
            assert home != live_home and not home.exists()
            assert "PYTHONPATH" not in environment and "PYTHONHOME" not in environment
            return subprocess.CompletedProcess(argv, 1 if failure == "missing-mlx" else 0,
                stdout="", stderr="ModuleNotFoundError: No module named 'mlx'" if failure == "missing-mlx" else "")
        return subprocess.CompletedProcess(argv, 0, stdout="help", stderr="")
    monkeypatch.setattr(envpack.subprocess, "run", run)
    with pytest.raises(envpack.PackError, match="failed fresh-home init"):
        envpack.smoke_test(target, tmp_path / "pack.tar.zst")
    assert not live_home.exists()
