from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import VERSION

REPO = 'telltaleatheist/crucible'
RELEASE_SH = Path(__file__).resolve().parents[1] / 'scripts/release.sh'

_ASSIGNMENT = re.compile(r'^([A-Z][A-Z0-9_]*)="([^"\n]*)"$', re.MULTILINE)
_REFERENCE = re.compile(r'\$([A-Z][A-Z0-9_]*)')
_ARGUMENT = re.compile(r'--[A-Za-z][A-Za-z-]*(?:=\S+)?|"\$[A-Z][A-Z0-9_]*"|\S+')
_VERSION_IN_NAME = re.compile(r'\d+\.\d+\.\d+')
_WHEEL_SUFFIX = '-py3-none-any.whl'


def _expand(value: str, assignments: dict[str, str]) -> str:
    for _ in range(10):
        expanded = _REFERENCE.sub(
            lambda hit: assignments.get(hit.group(1), hit.group(0)), value)
        if expanded == value:
            return value
        value = expanded
    raise ValueError(f'{RELEASE_SH.name}: {value!r} expands forever')


def uploaded_asset_names(version: str, release_sh: Path | None = None) -> list[str]:
    release_sh = release_sh or RELEASE_SH
    text = release_sh.read_text(encoding='utf-8')
    assignments = dict(_ASSIGNMENT.findall(text))
    assignments['VERSION'] = version
    start = text.find('gh release create')
    if start < 0:
        raise ValueError(f'{release_sh.name} never calls `gh release create`; '
                         'there is no asset list to read')
    lines: list[str] = []
    for line in text[start:].splitlines():
        lines.append(line)
        if not line.rstrip().endswith('\\'):
            break
    command = ' '.join(line.rstrip().removesuffix('\\') for line in lines)

    names: list[str] = []
    previous = ''
    positional = 0
    for argument in _ARGUMENT.findall(command):
        quoted = argument.startswith('"$')
        value_of_a_flag = previous.startswith('--') and '=' not in previous
        if not value_of_a_flag and argument not in ('gh', 'release', 'create') \
                and not argument.startswith('--'):
            positional += 1
            if quoted and positional > 1:
                variable = argument[2:-1]
                if variable not in assignments:
                    raise ValueError(
                        f'{release_sh.name} uploads {argument}, which it never assigns')
                name = Path(_expand(assignments[variable], assignments)).name
                if '$' in name:
                    raise ValueError(
                        f'{release_sh.name} uploads {argument}, which this cannot '
                        f'resolve to a filename (got {name!r})')
                names.append(name)
        previous = argument

    if not names:
        raise ValueError(f'{release_sh.name} uploads no assets at all, which cannot be right')
    if len(set(names)) != len(names):
        raise ValueError(f'{release_sh.name} uploads the same asset twice: {names}')
    return names


def assets_of_release(release: str) -> list[dict]:
    return json.loads(subprocess.check_output([
        'gh', 'release', 'view', f'v{release}', '--repo', REPO, '--json', 'assets',
    ], text=True))['assets']


def fetch_asset(release: str, name: str) -> bytes:
    return subprocess.check_output([
        'gh', 'release', 'download', f'v{release}', '--repo', REPO,
        '--pattern', name, '--output', '-',
    ])


def validate_assets(assets: list[dict], version: str, *, fetch=fetch_asset,
                    uploaded: list[str] | None = None) -> None:
    names = uploaded if uploaded is not None else uploaded_asset_names(version)
    wheels = [name for name in names if name.endswith(_WHEEL_SUFFIX)]
    if len(wheels) != 1:
        raise ValueError(f'expected exactly one wheel in the release, found {wheels}')
    wheel = wheels[0]
    wheel_sha = f'{wheel}.sha256'

    by_name = {asset['name']: asset for asset in assets}
    if len(by_name) != len(assets):
        raise ValueError('duplicate asset names')

    for filename in [*names, wheel_sha]:
        if filename not in by_name:
            raise ValueError(f'missing release asset: {filename}')
        if by_name[filename]['size'] <= 0:
            raise ValueError(f'empty release asset: {filename}')

    for name in sorted(by_name):
        found = _VERSION_IN_NAME.search(name)
        if found and found.group(0) != version:
            raise ValueError(
                f'asset {name} names version {found.group(0)}, not {version}')

    published = hashlib.sha256(fetch(version, wheel)).hexdigest()
    attested = fetch(version, wheel_sha).decode('utf-8').split()
    if not attested or not re.fullmatch(r'[a-f0-9]{64}', attested[0]):
        raise ValueError(f'{wheel_sha} does not begin with a sha256 digest')
    if attested[0] != published:
        raise ValueError(
            f'{wheel} hashes to {published} but {wheel_sha} attests {attested[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a complete release candidate, and with --publish promote it to latest.",
    )
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
    validate_assets(metadata['assets'], VERSION)
    print(f'{args.tag}: every asset scripts/release.sh uploads is published, is v{VERSION},')
    print('and the wheel matches the digest published beside it.')
    if args.publish:
        subprocess.run(['gh', 'release', 'edit', args.tag, '--repo', REPO,
                        '--prerelease=false', '--latest=true'], check=True)
    else:
        print('Read-only check: not promoted. Fresh candidate install tests are still required.')


if __name__ == '__main__':
    main()
