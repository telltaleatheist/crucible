"""Never turn a partial pack release into the recommended installer."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from crucible import envpack

spec = importlib.util.spec_from_file_location('promote_release', Path(__file__).parents[1] / 'scripts/promote_release.py')
promote_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote_release)


def complete():
    version = '0.6.1'
    rows = []
    names = [f'crucible-{version}.tar.gz', f'crucible-{version}-py3-none-any.whl',
             f'crucible-client-{version}.tgz', f'crucible-bootstrap-{version}.tgz',
             'install.sh', 'install.ps1', 'envpacks.json',
             f'crucible-rootfs-{version}.tar.zst', f'crucible-rootfs-{version}.tar.zst.sha256']
    for name, backend in envpack.every_pack():
        target = envpack.pack_target(name, backend)
        part = envpack.part_filename(target.archive_name(version), 0)
        names.append(part)
        rows.append(envpack.PackEntry(name, backend, envpack.python_for_target(target).python_version,
                    1, 'a' * 64, (part,), envpack.recipe_digest(target.recipe), 1, version))
    return envpack.PackManifest(version, tuple(rows)), [{'name': name, 'size': 1} for name in names]


def test_complete_candidate_passes_metadata_gate():
    manifest, assets = complete()
    promote_release.validate_assets(manifest, assets, '0.6.1')


@pytest.mark.parametrize('missing', ['install.ps1', 'crucible-rootfs-0.6.1.tar.zst.sha256',
                                   'crucible-env-host-llama-windows-0.6.1.tar.zst.part00'])
def test_missing_published_asset_refuses(missing):
    manifest, assets = complete()
    with pytest.raises(ValueError, match='missing'):
        promote_release.validate_assets(manifest, [a for a in assets if a['name'] != missing], '0.6.1')


def test_incomplete_manifest_refuses_even_if_parts_are_uploaded():
    manifest, assets = complete()
    with pytest.raises(ValueError, match='pack set mismatch'):
        promote_release.validate_assets(replace(manifest, packs=manifest.packs[:-1]), assets, '0.6.1')


def test_wrong_published_size_refuses():
    manifest, assets = complete()
    assets[-1]['size'] = 2
    with pytest.raises(ValueError, match='sizes differ'):
        promote_release.validate_assets(manifest, assets, '0.6.1')
