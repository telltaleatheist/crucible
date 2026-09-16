"""Release smoke must reject an older core, without starting a service."""
import subprocess
from pathlib import Path

import pytest

from crucible import envpack


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("name,backend", [("server", "cuda-linux"), ("host", "llama-windows")])
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
        assert all(args[-1] in ("--help", "--version") for args in commands)
