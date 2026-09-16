"""Validate a complete candidate before explicitly promoting it to latest.

Default operation is read-only. --publish also requires the operator to attest
that fresh installs from this candidate passed; metadata cannot prove that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import VERSION, envpack  # noqa: E402

REPO = 'telltaleatheist/crucible'


def validate_assets(manifest: envpack.PackManifest, assets: list[dict], version: str) -> None:
    """The manifest is complete, matches this source, and every named byte exists."""
    if manifest.version != version:
        raise ValueError(f'manifest is {manifest.version}, expected {version}')
    by_name = {asset['name']: asset for asset in assets}
    if len(by_name) != len(assets):
        raise ValueError('duplicate asset names')
    required = {
        f'crucible-{version}.tar.gz', f'crucible-{version}-py3-none-any.whl',
        f'crucible-client-{version}.tgz', f'crucible-bootstrap-{version}.tgz',
        'install.sh', 'install.ps1', envpack.MANIFEST_NAME,
        f'crucible-rootfs-{version}.tar.zst', f'crucible-rootfs-{version}.tar.zst.sha256',
    }
    for filename in required:
        if filename not in by_name or by_name[filename]['size'] <= 0:
            raise ValueError(f'missing/empty release asset: {filename}')
    declared = set(envpack.every_pack())
    actual = {(p.name, p.backend) for p in manifest.packs}
    if declared != actual:
        raise ValueError(f'pack set mismatch; missing={declared-actual}, extra={actual-declared}')
    for entry in manifest.packs:
        target = envpack.pack_target(entry.name, entry.backend)
        if entry.recipe_sha256 != envpack.recipe_digest(target.recipe):
            raise ValueError(f'{entry.name}/{entry.backend}: recipe differs from this checkout')
        if entry.python != envpack.python_for_target(target).python_version:
            raise ValueError(f'{entry.name}/{entry.backend}: interpreter differs from this checkout')
        if not re.fullmatch(r'[a-f0-9]{64}', entry.sha256):
            raise ValueError(f'{entry.name}/{entry.backend}: invalid archive digest')
        if target.job_type is None:
            expected = tuple(envpack.part_filename(target.archive_name(version), i)
                             for i in range(len(entry.parts)))
            if entry.parts != expected:
                raise ValueError(f'{entry.name}/{entry.backend}: runtime must be rebuilt for {version}')
        total = 0
        for part in entry.parts:
            if '/' in part or '\\' in part or part not in by_name or by_name[part]['size'] <= 0:
                raise ValueError(f'missing/invalid pack part: {part}')
            total += by_name[part]['size']
        if total != entry.bytes:
            raise ValueError(f'{entry.name}/{entry.backend}: published part sizes differ from manifest')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--confirmed-install-smoke', action='store_true')
    args = parser.parse_args()
    if args.tag != f'v{VERSION}':
        parser.error(f'run from the release source: this checkout is v{VERSION}')
    if args.publish and not args.confirmed_install_smoke:
        parser.error('--publish requires --confirmed-install-smoke after fresh candidate install tests')
    metadata = json.loads(subprocess.check_output([
        'gh', 'release', 'view', args.tag, '--repo', REPO,
        '--json', 'tagName,isDraft,isPrerelease,assets',
    ], text=True))
    if metadata['isDraft'] or not metadata['isPrerelease']:
        parser.error('expected a publicly downloadable prerelease candidate, not draft/stable')
    manifest = envpack.read_manifest(envpack.manifest_url(VERSION))
    validate_assets(manifest, metadata['assets'], VERSION)
    print(f'{args.tag}: complete pack manifest, published assets and source recipes agree')
    print('Archive SHA256 checks belong to pack build/reuse and fresh-install verification; this gate checks release metadata.')
    if args.publish:
        subprocess.run(['gh', 'release', 'edit', args.tag, '--repo', REPO,
                        '--prerelease=false', '--latest=true'], check=True)
    else:
        print('Read-only check: not promoted. Fresh candidate install tests are still required.')


if __name__ == '__main__':
    main()
