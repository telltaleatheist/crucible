"""Patch release planning/staging is CPU-only and never mutates a release."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from crucible import envpack

spec = importlib.util.spec_from_file_location('release_packs', Path(__file__).parents[1] / 'scripts/release_packs.py')
release_packs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_packs)


def manifest(tmp_path, *, name='llm', backend='mlx-darwin', payload=b'archive content'):
    target = envpack.pack_target(name, backend)
    part = envpack.part_filename(target.archive_name('0.6.0'), 0)
    (tmp_path / part).write_bytes(payload)
    entry = envpack.PackEntry(name, backend, envpack.STANDALONE_PYTHON[backend].python_version,
        len(payload), hashlib.sha256(payload).hexdigest(), (part,),
        envpack.recipe_digest(target.recipe), 1000)
    return envpack.PackManifest('0.6.0', (entry,), (tmp_path / 'envpacks.json').as_uri())


def test_runtime_never_reused_even_when_recipe_and_python_match(tmp_path):
    source = manifest(tmp_path, name='server')
    row = next(p for p in release_packs.plan(source, '0.6.1')['packs']
               if p['name'] == 'server' and p['backend'] == 'mlx-darwin')
    assert row['action'] == 'rebuild'
    with pytest.raises(ValueError, match='not reusable'):
        release_packs.stage(source, '0.6.1', 'server', 'mlx-darwin', tmp_path / 'out')


def test_unchanged_inference_staged_with_new_manifest_and_original_asset_names(tmp_path):
    source = manifest(tmp_path)
    result = release_packs.stage(source, '0.6.1', 'llm', 'mlx-darwin', tmp_path / 'out')
    staged = envpack.parse_manifest(result.read_text())
    assert staged.version == '0.6.1'
    assert staged.packs == source.packs
    assert (result.parent / source.packs[0].parts[0]).read_bytes() == b'archive content'


def test_corrupted_source_never_leaves_manifest(tmp_path):
    source = manifest(tmp_path)
    (tmp_path / source.packs[0].parts[0]).write_bytes(b'bad content')
    with pytest.raises(ValueError, match='size or SHA256'):
        release_packs.stage(source, '0.6.1', 'llm', 'mlx-darwin', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_recipe_change_cannot_reuse(tmp_path, monkeypatch):
    source = manifest(tmp_path)
    monkeypatch.setattr(envpack, 'recipe_digest', lambda path: 'changed')
    with pytest.raises(ValueError, match='not reusable'):
        release_packs.stage(source, '0.6.1', 'llm', 'mlx-darwin', tmp_path / 'out')


def test_same_version_refused(tmp_path):
    with pytest.raises(ValueError, match='new version'):
        release_packs.plan(manifest(tmp_path), '0.6.0')


def test_python_recipe_requirement_is_a_build_blocker(tmp_path, monkeypatch):
    monkeypatch.setattr(envpack, 'RECIPE_STANDALONE_PYTHON', {})
    source = manifest(tmp_path)
    row = next(p for p in release_packs.plan(source, '0.6.1')['packs']
               if p['name'] == 'tts-higgs-v3' and p['backend'] == 'cuda-linux')
    assert '3.12' in row['build_blocker']
