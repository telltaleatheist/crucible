from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import VERSION

REPO = 'telltaleatheist/crucible'
REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SH = REPO_ROOT / 'scripts/release.sh'
DEPLOY_SH = REPO_ROOT / 'scripts/deploy.sh'
ASSET_LIST = 'assets.json'

_ASSIGNMENT = re.compile(r'^([A-Z][A-Z0-9_]*)="([^"\n]*)"$', re.MULTILINE)
_REFERENCE = re.compile(r'\$([A-Z][A-Z0-9_]*)')
_ARGUMENT = re.compile(r'--[A-Za-z][A-Za-z-]*(?:=\S+)?|"\$[A-Z][A-Z0-9_]*"|\S+')
_VERSION_IN_NAME = re.compile(r'\d+\.\d+\.\d+')
_WHEEL_SUFFIX = '-py3-none-any.whl'
_FLEET = re.compile(r'^FLEET="([^"]*)"$', re.MULTILINE)
_READING = re.compile(r'^  (\S+)\s+(\S.*?)\s*$')


def promote_command(tag: str) -> str:
    return f'python scripts/promote_release.py --tag {tag} --publish --confirmed-install-smoke'


def write_asset_list(directory: Path, files: list[str]) -> Path:
    names = [Path(file).name for file in files]
    if len(set(names)) != len(names):
        raise ValueError(f'the release would upload the same asset name twice: {names}')
    if ASSET_LIST in names:
        raise ValueError(f'{ASSET_LIST} is the list itself, not an asset to list')
    target = Path(directory) / ASSET_LIST
    with open(target, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(names, indent=2) + '\n')
    return target


def read_asset_list(raw: bytes) -> list[str]:
    names = json.loads(raw.decode('utf-8'))
    if not isinstance(names, list) or not names \
            or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f'{ASSET_LIST} is not a list of asset names: {names!r}')
    return names


def _expand(value: str, assignments: dict[str, str], source: str) -> str:
    for _ in range(10):
        expanded = _REFERENCE.sub(
            lambda hit: assignments.get(hit.group(1), hit.group(0)), value)
        if expanded == value:
            return value
        value = expanded
    raise ValueError(f'{source}: {value!r} expands forever')


def uploaded_asset_names(version: str, release_sh: Path | None = None,
                         text: str | None = None) -> list[str]:
    source = release_sh.name if release_sh else 'release.sh'
    if text is None:
        text = (release_sh or RELEASE_SH).read_text(encoding='utf-8')
    assignments = dict(_ASSIGNMENT.findall(text))
    assignments['VERSION'] = version
    start = text.find('gh release create')
    if start < 0:
        raise ValueError(f'{source} never calls `gh release create`; '
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
                        f'{source} uploads {argument}, which it never assigns')
                name = Path(_expand(assignments[variable], assignments, source)).name
                if '$' in name:
                    raise ValueError(
                        f'{source} uploads {argument}, which this cannot '
                        f'resolve to a filename (got {name!r})')
                names.append(name)
        previous = argument

    if not names:
        raise ValueError(f'{source} uploads no assets at all, which cannot be right')
    if len(set(names)) != len(names):
        raise ValueError(f'{source} uploads the same asset twice: {names}')
    return names


def release_sh_at(tag: str) -> str:
    done = subprocess.run(['git', 'show', f'{tag}:scripts/release.sh'], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise ValueError(f'cannot read the release.sh that cut {tag} ({done.stderr.strip()}); '
                         f'fetch the tag with: git fetch --tags origin')
    return done.stdout


def fetch_asset(release: str, name: str) -> bytes:
    return subprocess.check_output([
        'gh', 'release', 'download', f'v{release}', '--repo', REPO,
        '--pattern', name, '--output', '-',
    ])


def expected_assets(assets: list[dict], version: str, *, fetch=fetch_asset,
                    release_sh_text=release_sh_at) -> list[str]:
    if any(asset['name'] == ASSET_LIST for asset in assets):
        return read_asset_list(fetch(version, ASSET_LIST))
    print(f'v{version} carries no {ASSET_LIST}: it was cut before release.sh wrote one, so '
          f'its asset list is read out of the release.sh that cut it.')
    return uploaded_asset_names(version, text=release_sh_text(f'v{version}'))


def validate_assets(assets: list[dict], version: str, *, fetch=fetch_asset,
                    uploaded: list[str] | None = None) -> None:
    names = uploaded if uploaded is not None else expected_assets(assets, version, fetch=fetch)
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


def fleet_of(deploy_sh: Path = DEPLOY_SH) -> list[str]:
    found = _FLEET.search(deploy_sh.read_text(encoding='utf-8'))
    if not found or not found.group(1).split():
        raise ValueError(f'{deploy_sh.name} names no FLEET, so there is no fleet to check')
    return found.group(1).split()


def fleet_readings(output: str, fleet: list[str]) -> dict[str, str]:
    readings: dict[str, str] = {}
    for line in output.splitlines():
        found = _READING.match(line)
        if found and found.group(1) in fleet:
            readings[found.group(1)] = found.group(2)
    return readings


def fleet_problems(readings: dict[str, str], fleet: list[str], version: str,
                   allowed: set[str]) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    waived: list[str] = []
    for machine in fleet:
        reading = readings.get(machine)
        if reading == version:
            continue
        if reading is None or 'unreachable' in reading:
            said = reading or 'deploy.sh could not ask it'
            if machine in allowed:
                waived.append(f'{machine} ({said})')
                continue
            problems.append(
                f'{machine} cannot be asked what it runs ({said}). Bring it back and '
                f're-run, or add --allow-unreachable {machine} to promote without it')
            continue
        problems.append(
            f'{machine} runs {reading}, not {version}. Install it there with: '
            f'./scripts/deploy.sh --release {version} --only {machine}')
    return problems, waived


def run_deploy_read_only(only: list[str] | None,
                         deploy_sh: Path = DEPLOY_SH) -> subprocess.CompletedProcess:
    bash = shutil.which('bash')
    if not bash:
        raise ValueError('no bash on PATH to run scripts/deploy.sh with; run this from Git Bash '
                         '(Windows) or a terminal (macOS/Linux)')
    argv = [bash, str(deploy_sh)]
    if only is not None:
        argv += ['--only', ','.join(only)]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, timeout=300,
                          stdin=subprocess.DEVNULL)


def check_fleet(version: str, allowed: set[str], fleet: list[str], *,
                run=run_deploy_read_only) -> tuple[list[str], list[str]]:
    done = run(None)
    asked = fleet
    if done.returncode != 0 and allowed:
        asked = [machine for machine in fleet if machine not in allowed]
        if asked:
            done = run(asked)
    if done.returncode != 0 and asked:
        said = (done.stderr or done.stdout).strip().splitlines()
        return [f'./scripts/deploy.sh could not read the fleet: '
                f'{said[-1] if said else f"it exited {done.returncode}"}'], []
    return fleet_problems(fleet_readings(done.stdout, asked), fleet, version, allowed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a complete release candidate, check every machine runs it, "
                    "and with --publish promote it to latest.",
    )
    parser.add_argument('--tag')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--confirmed-install-smoke', action='store_true')
    parser.add_argument('--allow-unreachable', action='append', default=[], metavar='MACHINE',
                        help='promote even though MACHINE cannot be asked what it runs')
    parser.add_argument('--print-command', action='store_true',
                        help='print the command that promotes --tag, and nothing else')
    parser.add_argument('--write-asset-list', nargs='+', metavar='PATH',
                        help=f'DIR then each FILE: write DIR/{ASSET_LIST} naming the files and '
                             f'print its name (release.sh uses this)')
    args = parser.parse_args()

    if args.write_asset_list:
        directory, *files = args.write_asset_list
        if not files:
            parser.error('--write-asset-list needs a directory and at least one file')
        try:
            print(write_asset_list(Path(directory), files).name)
        except ValueError as exc:
            parser.error(str(exc))
        return
    if not args.tag:
        parser.error(f'name the release: --tag v{VERSION}')
    if args.print_command:
        print(promote_command(args.tag))
        return
    if args.tag != f'v{VERSION}':
        parser.error(f'run from the release source: this checkout is v{VERSION}; '
                     f'git checkout {args.tag}, then {promote_command(args.tag)}')
    if args.publish and not args.confirmed_install_smoke:
        parser.error(f'--publish also needs --confirmed-install-smoke, your word that you '
                     f'installed {args.tag} and it worked: {promote_command(args.tag)}')
    fleet = fleet_of()
    unknown = sorted(set(args.allow_unreachable) - set(fleet))
    if unknown:
        parser.error(f'--allow-unreachable names {", ".join(unknown)}, which deploy.sh does not '
                     f'know; the fleet is: {" ".join(fleet)}')
    metadata = json.loads(subprocess.check_output([
        'gh', 'release', 'view', args.tag, '--repo', REPO,
        '--json', 'tagName,isDraft,isPrerelease,assets',
    ], text=True))
    if metadata['isDraft'] or not metadata['isPrerelease']:
        parser.error('expected a publicly downloadable prerelease candidate, not draft/stable')
    try:
        validate_assets(metadata['assets'], VERSION)
    except ValueError as exc:
        parser.error(f'{exc}. This candidate cannot be promoted; cut the next one with '
                     f'./scripts/ship.sh patch')
    print(f'{args.tag}: every asset release.sh uploaded is published, is v{VERSION},')
    print('and the wheel matches the digest published beside it.')

    try:
        problems, waived = check_fleet(VERSION, set(args.allow_unreachable), fleet)
    except ValueError as exc:
        problems, waived = [str(exc)], []
    for machine in waived:
        print(f'{args.tag}: not checked, because --allow-unreachable: {machine}')
    if problems:
        listed = '\n'.join(f'  {problem}' for problem in problems)
        if args.publish:
            parser.error(f'not every machine runs {args.tag}, so it was not promoted:\n{listed}')
        print(f'{args.tag}: not every machine runs it yet:\n{listed}')
    else:
        print(f'{args.tag}: every machine deploy.sh asked runs it.')

    if args.publish:
        subprocess.run(['gh', 'release', 'edit', args.tag, '--repo', REPO,
                        '--prerelease=false', '--latest=true'], check=True)
        print(f'{args.tag}: promoted; releases/latest now serves it.')
    else:
        print('Read-only check: not promoted. After a fresh install of the candidate works, run:')
        print(f'  {promote_command(args.tag)}')


if __name__ == '__main__':
    main()
