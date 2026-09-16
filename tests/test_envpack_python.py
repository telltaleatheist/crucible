"""Environment packs must use the interpreter their recipes require."""
from dataclasses import replace

import pytest

from crucible import envpack, jobenv


def test_higgs_uses_python312_without_changing_other_packs():
    higgs = envpack.pack_target('tts-higgs-v3', 'cuda-linux')
    pin = envpack.python_for_target(higgs)
    assert pin.python_version.startswith('3.12.')
    assert pin.asset == 'cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz'
    assert pin.sha256 == '936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22'
    for name, backend in envpack.every_pack():
        if (name, backend) != ('tts-higgs-v3', 'cuda-linux'):
            assert envpack.python_for_target(envpack.pack_target(name, backend)).python_version.startswith('3.11.')


def test_unknown_recipe_python_fails_before_any_build(monkeypatch, tmp_path):
    original = jobenv.tts_env('higgs-v3', 'cuda-linux')
    monkeypatch.setattr(jobenv, 'tts_env', lambda *args: replace(original, python_version='3.99'))
    target = envpack.pack_target('tts-higgs-v3', 'cuda-linux')
    monkeypatch.setattr(envpack, 'require_zstd_tar', lambda: None)
    monkeypatch.setattr(envpack, 'fetch_standalone_python', lambda *a, **k: pytest.fail('must not download'))
    with pytest.raises(envpack.PackError, match='requires Python 3.99'):
        envpack.build_pack(target, '0.6.1', tmp_path / 'pack')
    assert not (tmp_path / 'pack').exists()


def test_fetch_uses_the_selected_interpreter_not_backend_default(monkeypatch, tmp_path):
    pin = envpack.python_for_target(envpack.pack_target('tts-higgs-v3', 'cuda-linux'))
    urls = []
    def fetch(url, path, **kwargs):
        urls.append(url)
        path.write_bytes(b'fixture')
    monkeypatch.setattr(envpack, '_fetch', fetch)
    monkeypatch.setattr(envpack, 'sha256_of', lambda path: pin.sha256)
    archive = envpack.fetch_standalone_python('cuda-linux', tmp_path, pin=pin)
    assert urls == [pin.url]
    assert archive.name == pin.asset
