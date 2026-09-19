"""Validate a complete candidate before explicitly promoting it to latest.

Default operation is read-only. --publish also requires the operator to attest
that fresh installs from this candidate passed; metadata cannot prove that.

WHAT A RELEASE IS, since PHASE20-CODE-NOT-ENVIRONMENTS.md section 1: our code
and nothing that is published elsewhere -- the wheel, its digest, the sdist, the
two SDK tarballs and the two generated installers. Not an interpreter, not an
environment, not a WSL image. Those come from python-build-standalone, from PyPI
and from Canonical, each pinned where the thing that needs it lives, so there is
no per-release manifest of archives left for this to walk. What replaced that
walk is smaller and says more: the named bytes are all there, they are all this
version, and the wheel is the wheel its own digest says it is.

THE NAMES ARE NOT WRITTEN HERE. `scripts/release.sh` builds the assets and
uploads them, so it is what knows what a release carries; this reads its
`gh release create` invocation and requires exactly those. A release.sh that
gains an asset gains it here on the same commit, and one that loses an asset
cannot leave this asking for a file nobody builds -- which is the drift a second
list would have, and had: the list this replaces still demanded a rootfs and a
pack manifest that PHASE20 had already deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import VERSION  # noqa: E402

REPO = 'telltaleatheist/crucible'
RELEASE_SH = Path(__file__).resolve().parents[1] / 'scripts/release.sh'

#: `NAME="value"`, where the value is one unbroken double-quoted run. A line
#: like `REPO="$(cd ... && pwd)"` has quotes inside it and deliberately does not
#: match: it is a directory, and every asset below is read for its basename.
_ASSIGNMENT = re.compile(r'^([A-Z][A-Z0-9_]*)="([^"\n]*)"$', re.MULTILINE)
_REFERENCE = re.compile(r'\$([A-Z][A-Z0-9_]*)')
#: One argument of the invocation: a flag, a `"$VAR"`, or anything else.
_ARGUMENT = re.compile(r'--[A-Za-z][A-Za-z-]*(?:=\S+)?|"\$[A-Z][A-Z0-9_]*"|\S+')
#: A version anywhere inside an asset's name.
_VERSION_IN_NAME = re.compile(r'\d+\.\d+\.\d+')
#: The one asset every install starts from. Pure python and `py3-none-any`, so
#: one file installs on every backend.
_WHEEL_SUFFIX = '-py3-none-any.whl'


def _expand(value: str, assignments: dict[str, str]) -> str:
    """`$VAR` by `$VAR`, until nothing known is left to replace.

    Bounded, because a release.sh carrying `A="$B"` beside `B="$A"` would
    otherwise hang the release path rather than fail it.
    """
    for _ in range(10):
        expanded = _REFERENCE.sub(
            lambda hit: assignments.get(hit.group(1), hit.group(0)), value)
        if expanded == value:
            return value
        value = expanded
    raise ValueError(f'{RELEASE_SH.name}: {value!r} expands forever')


def uploaded_asset_names(version: str, release_sh: Path | None = None) -> list[str]:
    """The basenames `release.sh` hands to `gh release create`, in its order.

    WHICH ARGUMENTS ARE ASSETS is decided by gh's own contract rather than by
    any knowledge of this repo: `gh release create <tag> [files...]`, so the
    first positional is the tag, and a `"$VAR"` that follows a `--flag` is that
    flag's value. Everything else quoted is a file. Nothing here knows what
    `$SDIST` or `$WHEEL` are called, which is the entire point.
    """
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
            if quoted and positional > 1:  # the first positional is the tag
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
    """Every published asset of `v<release>`, asked of GitHub."""
    return json.loads(subprocess.check_output([
        'gh', 'release', 'view', f'v{release}', '--repo', REPO, '--json', 'assets',
    ], text=True))['assets']


def fetch_asset(release: str, name: str) -> bytes:
    """One published asset's bytes, which is the only way to check a digest."""
    return subprocess.check_output([
        'gh', 'release', 'download', f'v{release}', '--repo', REPO,
        '--pattern', name, '--output', '-',
    ])


def validate_assets(assets: list[dict], version: str, *, fetch=fetch_asset,
                    uploaded: list[str] | None = None) -> None:
    """Every named byte is published, is this version, and is what it claims.

    THE DIGEST IS A REQUIRED ASSET, not an optional one. PHASE20 section 3 has
    both installers fetch `<wheel>.sha256` and compare it before pip is allowed
    near the download: the interpreter is pinned by a digest in our source and
    our own code cannot be, so the release is what vouches for it. A release
    without that line is one neither installer can verify, whatever else it
    carries. Its name is the wheel's plus one suffix -- the same rule
    `sdk/bootstrap/src/release.ts` states as `wheelShaAssetName()`, applied to
    the wheel name release.sh gave us rather than to a second spelling of it.
    """
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

    # A LEFTOVER FROM ANOTHER RELEASE IS NOT A SPARE COPY. Promotion is what
    # makes `releases/latest` serve this version, so an asset carrying a
    # different one is an installer reaching for bytes that are not this
    # candidate's -- which is how half of a previous release becomes part of
    # the recommended one.
    for name in sorted(by_name):
        found = _VERSION_IN_NAME.search(name)
        if found and found.group(0) != version:
            raise ValueError(
                f'asset {name} names version {found.group(0)}, not {version}')

    published = hashlib.sha256(fetch(version, wheel)).hexdigest()
    # One line, the digest first, the shape `sha256sum` writes.
    attested = fetch(version, wheel_sha).decode('utf-8').split()
    if not attested or not re.fullmatch(r'[a-f0-9]{64}', attested[0]):
        raise ValueError(f'{wheel_sha} does not begin with a sha256 digest')
    if attested[0] != published:
        raise ValueError(
            f'{wheel} hashes to {published} but {wheel_sha} attests {attested[0]}')


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
