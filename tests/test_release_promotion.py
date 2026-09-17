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


# ---------------------------------------------------------------- carried packs
#
# Since `scripts/plan_packs.py`, a release BUILDS only the packs whose recipe
# changed and CARRIES the rest by reference: the row keeps naming the release
# that already holds the bytes, and no part is uploaded again. The gate below
# used to check every part against THIS release's assets, which for a carried
# row is a list the part was never going to be in — so the first release to
# reuse a pack would have been refused `missing/invalid pack part` for ten of
# thirteen packs, at the last step before promotion.


def carried(older='0.6.0'):
    """A complete 0.6.1 candidate whose `align/cuda-linux` pack is carried."""
    manifest, assets = complete()
    rows = []
    moved = None
    for entry in manifest.packs:
        if (entry.name, entry.backend) == ('align', 'cuda-linux'):
            target = envpack.pack_target(entry.name, entry.backend)
            part = envpack.part_filename(target.archive_name(older), 0)
            moved = entry.parts[0]
            entry = replace(entry, parts=(part,), release=older)
        rows.append(entry)
    assert moved is not None
    assets = [asset for asset in assets if asset['name'] != moved]
    return envpack.PackManifest(manifest.version, tuple(rows)), assets


def test_a_carried_pack_is_checked_against_the_release_it_names():
    manifest, assets = carried()
    asked = []

    def elsewhere(release):
        asked.append(release)
        entry = manifest.require('align', 'cuda-linux')
        return [{'name': entry.parts[0], 'size': 1}]

    promote_release.validate_assets(manifest, assets, '0.6.1', assets_for_release=elsewhere)
    assert asked == ['0.6.0'], 'the older release is the one asked, and asked once'


def test_a_carried_pack_missing_from_the_release_it_names_refuses():
    manifest, assets = carried()
    with pytest.raises(ValueError, match='v0.6.0'):
        promote_release.validate_assets(manifest, assets, '0.6.1',
                                        assets_for_release=lambda release: [])


def test_a_carried_row_without_a_way_to_look_it_up_refuses():
    """No lookup and a carried row is not 'assume it is fine' — it is a refusal."""
    manifest, assets = carried()
    with pytest.raises(ValueError, match='cannot verify'):
        promote_release.validate_assets(manifest, assets, '0.6.1', assets_for_release=None)


def test_a_runtime_pack_may_never_be_carried():
    """`server` embeds Crucible's own source, so a carried one is a stale server."""
    manifest, _ = complete()
    rows = tuple(replace(entry, release='0.6.0')
                 if (entry.name, entry.backend) == ('server', 'cuda-linux') else entry
                 for entry in manifest.packs)
    _, assets = complete()
    with pytest.raises(ValueError, match='runtime must be rebuilt'):
        promote_release.validate_assets(envpack.PackManifest('0.6.1', rows), assets, '0.6.1',
                                        assets_for_release=lambda release: [])
