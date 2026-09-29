from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from crucible import VERSION

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/promote_release.py'
RELEASE_SH = REPO / 'scripts/release.sh'

spec = importlib.util.spec_from_file_location('promote_release', SCRIPT)
promote_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote_release)

WHEEL_BYTES = b'not really a wheel, but bytes with a digest'
WHEEL_DIGEST = hashlib.sha256(WHEEL_BYTES).hexdigest()

OLD_RELEASE_SH = '''OUT="$REPO/dist"
INSTALL_SH="$REPO/sdk/bootstrap/scripts/install.sh"
INSTALL_PS1="$REPO/sdk/bootstrap/scripts/install.ps1"
SDIST="$OUT/crucible-$VERSION.tar.gz"
WHEEL="$OUT/crucible-$VERSION-py3-none-any.whl"
TGZ="$OUT/crucible-client-$VERSION.tgz"
BOOT="$OUT/crucible-bootstrap-$VERSION.tgz"
WHEEL_SHA="$WHEEL.sha256"
gh release create "$TAG" \\
  --repo "$REPO_SLUG" \\
  --target "$HEAD_SHA" \\
  --title "$TAG" \\
  --prerelease --latest=false \\
  --generate-notes \\
  --notes "$NOTES_HEADER" \\
  "$SDIST" "$WHEEL" "$WHEEL_SHA" "$TGZ" "$BOOT" \\
  "$INSTALL_SH" "$INSTALL_PS1"
'''


def uploaded(version: str) -> list[str]:
    return [f'crucible-{version}.tar.gz', f'crucible-{version}-py3-none-any.whl',
            f'crucible-{version}-py3-none-any.whl.sha256', f'crucible-client-{version}.tgz',
            f'crucible-bootstrap-{version}.tgz', 'install.sh', 'install.ps1',
            f'crucible-setup-{version}.exe']


def complete_names(version: str) -> tuple[list[str], list[str]]:
    names = uploaded(version)
    return names, [*names, promote_release.ASSET_LIST]


def published(version: str = '1.2.3') -> tuple[list[str], list[dict]]:
    names, complete = complete_names(version)
    return names, [{'name': name, 'size': 10} for name in complete]


def fetcher(overrides: dict[str, bytes] | None = None, version: str = '1.2.3'):
    def fetch(release: str, name: str) -> bytes:
        if overrides and name in overrides:
            return overrides[name]
        if name.endswith('-py3-none-any.whl'):
            return WHEEL_BYTES
        if name.endswith('.whl.sha256'):
            return f'{WHEEL_DIGEST}  {name[:-len(".sha256")]}\n'.encode()
        if name == promote_release.ASSET_LIST:
            return json.dumps(uploaded(version)).encode()
        if name.endswith('.exe'):
            return b'MZ a windows setup'
        return b'some asset'
    return fetch


def test_release_sh_uploads_exactly_the_assets_it_lists():
    text = RELEASE_SH.read_text(encoding='utf-8')
    listed = re.search(r'^ASSETS=\(([^)]*)\)$', text, re.MULTILINE)
    assert listed, 'release.sh no longer names its assets in one ASSETS=(...) array'
    assignments = dict(promote_release._ASSIGNMENT.findall(text))
    assignments['VERSION'] = VERSION
    names = [Path(promote_release._expand(assignments[variable], assignments, 'release.sh')).name
             for variable in re.findall(r'"\$([A-Z0-9_]+)"', listed.group(1))]
    assert names == uploaded(VERSION), names
    upload = text[text.index('gh release create'):]
    upload = upload[:upload.index('\n\n')]
    assert '"${ASSETS[@]}" "$ASSET_LIST"' in upload, upload
    assert '--write-asset-list "$OUT" "${ASSETS[@]}"' in text


def test_the_asset_list_is_written_from_the_files_and_read_back(tmp_path: Path):
    files = [f'/some/dist/{name}' for name in uploaded('1.2.3')]
    written = promote_release.write_asset_list(tmp_path, files)
    assert written == tmp_path / promote_release.ASSET_LIST
    assert promote_release.read_asset_list(written.read_bytes()) == uploaded('1.2.3')
    assert b'\r\n' not in written.read_bytes()


def test_an_asset_list_naming_one_file_twice_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match='same asset name twice'):
        promote_release.write_asset_list(tmp_path, ['a/install.sh', 'b/install.sh'])


def test_a_release_carrying_its_list_is_checked_against_that_list():
    _, assets = published()
    names = promote_release.expected_assets(
        assets, '1.2.3', fetch=fetcher(),
        release_sh_text=lambda tag: pytest.fail('read release.sh for a release with a list'))
    assert names == uploaded('1.2.3')


def test_an_older_release_falls_back_to_the_release_sh_that_cut_it(capsys):
    _, assets = published()
    assets = [asset for asset in assets if asset['name'] != promote_release.ASSET_LIST]
    asked = []

    def old(tag: str) -> str:
        asked.append(tag)
        return OLD_RELEASE_SH

    names = promote_release.expected_assets(assets, '1.2.3', fetch=fetcher(),
                                            release_sh_text=old)
    assert names == [name for name in uploaded('1.2.3') if not name.endswith('.exe')]
    assert asked == ['v1.2.3']
    assert f'carries no {promote_release.ASSET_LIST}' in capsys.readouterr().out


def test_an_upload_this_cannot_resolve_is_refused_rather_than_skipped(tmp_path: Path):
    fake = tmp_path / 'release.sh'
    fake.write_text(
        'SDIST="$OUT/crucible-$VERSION.tar.gz"\n'
        'gh release create "$TAG" \\\n'
        '  --repo "$REPO_SLUG" \\\n'
        '  "$SDIST" "$MYSTERY"\n',
        encoding='utf-8')
    with pytest.raises(ValueError, match='never assigns'):
        promote_release.uploaded_asset_names('1.2.3', fake)


def test_a_release_sh_that_uploads_nothing_is_refused(tmp_path: Path):
    fake = tmp_path / 'release.sh'
    fake.write_text('gh release create "$TAG" --repo "$REPO_SLUG"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='uploads no assets'):
        promote_release.uploaded_asset_names('1.2.3', fake)


WHEEL = 'crucible-1.2.3-py3-none-any.whl'
WHEEL_SHA = WHEEL + '.sha256'


def test_a_complete_candidate_passes():
    names, assets = published()
    promote_release.validate_assets(assets, '1.2.3', fetch=fetcher(), uploaded=names)
    promote_release.validate_assets(assets, '1.2.3', fetch=fetcher())


@pytest.mark.parametrize('missing', ['install.ps1', 'install.sh',
                                     'crucible-1.2.3.tar.gz',
                                     'crucible-client-1.2.3.tgz', WHEEL])
def test_a_missing_asset_is_refused_by_name(missing: str):
    names, assets = published()
    remaining = [asset for asset in assets if asset['name'] != missing]
    assert len(remaining) == len(assets) - 1, missing
    with pytest.raises(ValueError, match=f'missing release asset: {re.escape(missing)}'):
        promote_release.validate_assets(remaining, '1.2.3', fetch=fetcher())


def test_the_digest_asset_is_required():
    names, assets = published()
    remaining = [asset for asset in assets if asset['name'] != WHEEL_SHA]
    with pytest.raises(ValueError, match=f'missing release asset: {re.escape(WHEEL_SHA)}'):
        promote_release.validate_assets(
            remaining, '1.2.3', fetch=fetcher(),
            uploaded=[name for name in names if name != WHEEL_SHA])


def test_an_empty_asset_is_refused_by_name():
    names, assets = published()
    for asset in assets:
        if asset['name'] == 'install.sh':
            asset['size'] = 0
    with pytest.raises(ValueError, match='empty release asset: install.sh'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetcher(), uploaded=names)


def test_a_wheel_that_does_not_match_its_digest_is_refused():
    names, assets = published()
    fetch = fetcher({WHEEL: b'a different wheel entirely'})
    with pytest.raises(ValueError,
                       match=f'{re.escape(WHEEL)} hashes to .* attests {WHEEL_DIGEST}'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetch, uploaded=names)


def test_a_digest_file_that_is_not_a_digest_is_refused():
    names, assets = published()
    fetch = fetcher({WHEEL_SHA: b'<html>404 Not Found</html>\n'})
    with pytest.raises(ValueError,
                       match=f'{re.escape(WHEEL_SHA)} does not begin with a sha256'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetch, uploaded=names)


def test_an_asset_from_another_release_is_refused():
    names, assets = published()
    stray = [*assets, {'name': 'crucible-0.0.9-py3-none-any.whl', 'size': 10}]
    with pytest.raises(ValueError, match='names version 0.0.9, not 1.2.3'):
        promote_release.validate_assets(stray, '1.2.3', fetch=fetcher(), uploaded=names)


def test_a_release_with_two_wheels_is_refused():
    names, _ = published()
    with pytest.raises(ValueError, match='exactly one wheel'):
        promote_release.validate_assets(
            [], '1.2.3', fetch=fetcher(),
            uploaded=[*names, 'crucible-1.2.4-py3-none-any.whl'])


def test_a_setup_that_is_not_a_windows_program_is_refused():
    names, assets = published()
    fetch = fetcher({'crucible-setup-1.2.3.exe': b'<html>not found</html>'})
    with pytest.raises(ValueError, match='crucible-setup-1.2.3.exe is not a Windows program'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetch, uploaded=names)


def test_a_setup_of_another_version_is_refused():
    names = [name for name in uploaded('1.2.3') if not name.endswith('.exe')] + ['crucible-setup-1.2.2.exe']
    with pytest.raises(ValueError, match='expected the one Windows setup crucible-setup-1.2.3.exe'):
        promote_release.check_setup(names, '1.2.3', fetcher())


def test_a_release_cut_before_the_setup_existed_still_passes():
    names, assets = published()
    names = [name for name in names if not name.endswith('.exe')]
    assets = [asset for asset in assets if not asset['name'].endswith('.exe')]
    promote_release.validate_assets(assets, '1.2.3', fetch=fetcher(), uploaded=names)


def test_the_fleet_is_the_one_deploy_sh_names():
    assert promote_release.fleet_of() == ['pc', 'mac']


DEPLOY_OUTPUT = '''deploy: what each machine runs
  pc       1.2.3
  mac      host:1.2.3 guest:unreachable
deploy: pass --release <x.y.z> to install one
'''


def test_the_readings_are_parsed_out_of_deploy_sh_read_only_output():
    assert promote_release.fleet_readings(DEPLOY_OUTPUT, ['pc', 'mac']) == {
        'pc': '1.2.3', 'mac': 'host:1.2.3 guest:unreachable'}


def test_a_machine_behind_names_the_deploy_that_fixes_it():
    problems, waived = promote_release.fleet_problems(
        {'pc': '1.2.3', 'mac': '1.2.2'}, ['pc', 'mac'], '1.2.3', set())
    assert waived == []
    assert problems == ['mac runs 1.2.2, not 1.2.3. Install it there with: '
                        './scripts/deploy.sh --release 1.2.3 --only mac']


def test_an_unreachable_machine_blocks_unless_it_is_allowed_by_name():
    readings = promote_release.fleet_readings(DEPLOY_OUTPUT, ['pc', 'mac'])
    problems, waived = promote_release.fleet_problems(readings, ['pc', 'mac'], '1.2.3', set())
    assert len(problems) == 1 and '--allow-unreachable mac' in problems[0], problems
    problems, waived = promote_release.fleet_problems(readings, ['pc', 'mac'], '1.2.3', {'mac'})
    assert problems == [] and waived == ['mac (host:1.2.3 guest:unreachable)']


def test_allowing_a_reachable_machine_does_not_excuse_it_running_the_wrong_release():
    problems, _ = promote_release.fleet_problems(
        {'pc': '1.2.3', 'mac': '1.2.2'}, ['pc', 'mac'], '1.2.3', {'mac'})
    assert problems and 'mac runs 1.2.2' in problems[0]


def test_a_deploy_that_cannot_run_for_an_allowed_machine_asks_the_rest():
    calls = []

    def run(only):
        calls.append(only)
        if only is None:
            return subprocess.CompletedProcess([], 1, '', 'deploy: ... Run this on the PC, '
                                                         'or pass --only mac\n')
        return subprocess.CompletedProcess([], 0, 'deploy: what each machine runs\n'
                                                  '  mac      1.2.3\n', '')

    problems, waived = promote_release.check_fleet('1.2.3', {'pc'}, ['pc', 'mac'], run=run)
    assert calls == [None, ['mac']]
    assert problems == [] and waived == ['pc (deploy.sh could not ask it)']


def test_a_deploy_that_cannot_run_is_a_problem_naming_why():
    def run(only):
        return subprocess.CompletedProcess([], 1, '', 'deploy: pc is read ... pass --only mac\n')

    problems, _ = promote_release.check_fleet('1.2.3', set(), ['pc', 'mac'], run=run)
    assert problems == ['./scripts/deploy.sh could not read the fleet: '
                        'deploy: pc is read ... pass --only mac']


def test_every_script_prints_the_one_promote_command():
    for script in ('ship.sh', 'deploy.sh', 'release.sh'):
        text = (REPO / 'scripts' / script).read_text(encoding='utf-8')
        assert '--print-command' in text, f'{script} does not ask promote for its command'
        assert '--confirmed-install-smoke' not in text, f'{script} spells the command itself'


def test_print_command_is_the_command():
    done = subprocess.run([sys.executable, str(SCRIPT), '--tag', 'v9.9.9', '--print-command'],
                          capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == promote_release.promote_command('v9.9.9')
    assert done.stdout.strip() == ('python scripts/promote_release.py --tag v9.9.9 '
                                   '--publish --confirmed-install-smoke')


def _shim(path: Path, body: str) -> None:
    path.write_text('#!/bin/bash\n' + body, encoding='utf-8')
    path.chmod(0o755)


def gh_shim(tmp_path: Path, *, assets: list[dict], prerelease: bool = True,
            draft: bool = False, fleet: str = VERSION, mac_down: bool = False) -> dict[str, str]:
    metadata = {'tagName': f'v{VERSION}', 'isDraft': draft,
                'isPrerelease': prerelease, 'assets': assets}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    (tmp_path / 'assets.json').write_text(json.dumps(uploaded(VERSION)), encoding='utf-8')
    record = json.dumps({'release': fleet})
    local = tmp_path / 'localappdata' / 'crucible'
    local.mkdir(parents=True)
    (local / 'installation.json').write_text(record, encoding='utf-8')
    shims = tmp_path / 'shims'
    shims.mkdir()
    _shim(shims / 'gh',
          'case "$2" in\n'
          '  view) cat "$GH_TEST_DIR/metadata.json" ;;\n'
          '  download)\n'
          '    while [ $# -gt 0 ]; do\n'
          '      if [ "$1" = "--pattern" ]; then name="$2"; fi\n'
          '      shift\n'
          '    done\n'
          '    case "$name" in\n'
          '      *.whl) printf %s "$GH_TEST_WHEEL" ;;\n'
          '      *.whl.sha256) printf "%s  x\\n" "$GH_TEST_DIGEST" ;;\n'
          '      assets.json) cat "$GH_TEST_DIR/assets.json" ;;\n'
          '      *.exe) printf "MZ a windows setup" ;;\n'
          '      *) printf "some asset" ;;\n'
          '    esac\n'
          '    ;;\n'
          '  edit) printf "%s\\n" "$@" >> "$GH_TEST_DIR/edited" ;;\n'
          '  *) echo "unexpected gh $*" >&2; exit 9 ;;\n'
          'esac\n')
    _shim(shims / 'wsl.exe',
          'case "${@: -1}" in\n'
          '  *installation.json*) printf \'%s\\n\' "$FLEET_RECORD" ;;\n'
          '  *) exit 0 ;;\n'
          'esac\n')
    _shim(shims / 'ssh',
          '[ "$FLEET_MAC_DOWN" = "1" ] && exit 255\n'
          'case "${@: -1}" in\n'
          '  *installation.json*) printf \'%s\\n\' "$FLEET_RECORD" ;;\n'
          '  *) exit 0 ;;\n'
          'esac\n')
    environment = dict(os.environ)
    environment.update(
        PATH=str(shims) + os.pathsep + environment['PATH'],
        LOCALAPPDATA=str(tmp_path / 'localappdata'),
        GH_TEST_DIR=str(tmp_path),
        GH_TEST_WHEEL=WHEEL_BYTES.decode(),
        GH_TEST_DIGEST=WHEEL_DIGEST,
        FLEET_RECORD=record,
        FLEET_MAC_DOWN='1' if mac_down else '0',
    )
    return environment


def run_promote(environment: dict[str, str], *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), '--tag', f'v{VERSION}', *argv],
                          capture_output=True, text=True, timeout=120,
                          env=environment, cwd=str(REPO))


def complete_assets() -> list[dict]:
    return [{'name': name, 'size': 10} for name in complete_names(VERSION)[1]]


@pytest.fixture
def candidate(tmp_path: Path) -> dict[str, str]:
    return gh_shim(tmp_path, assets=complete_assets())


shims_are_bash = pytest.mark.skipif(sys.platform == 'win32',
                                    reason='the gh, wsl.exe and ssh shims are `#!/bin/bash` scripts')


@shims_are_bash
def test_a_complete_candidate_on_every_machine_is_promoted(tmp_path: Path,
                                                           candidate: dict[str, str]):
    done = run_promote(candidate, '--publish', '--confirmed-install-smoke')
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'every machine deploy.sh asked runs it' in done.stdout, done.stdout
    edited = (tmp_path / 'edited').read_text(encoding='utf-8').split()
    assert '--latest=true' in edited and '--prerelease=false' in edited, edited


@shims_are_bash
def test_the_default_is_read_only(tmp_path: Path, candidate: dict[str, str]):
    done = run_promote(candidate)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'not promoted' in done.stdout, done.stdout
    assert promote_release.promote_command(f'v{VERSION}') in done.stdout, done.stdout
    assert not (tmp_path / 'edited').exists(), 'a read-only run edited the release'


@shims_are_bash
def test_publishing_without_the_attestation_is_refused(tmp_path: Path,
                                                       candidate: dict[str, str]):
    done = run_promote(candidate, '--publish')
    assert done.returncode != 0
    assert '--confirmed-install-smoke' in done.stderr, done.stderr
    assert not (tmp_path / 'edited').exists()


@shims_are_bash
def test_a_missing_installer_stops_the_whole_command(tmp_path: Path):
    environment = gh_shim(tmp_path, assets=[asset for asset in complete_assets()
                                            if asset['name'] != 'install.ps1'])
    done = run_promote(environment, '--publish', '--confirmed-install-smoke')
    assert done.returncode != 0
    assert 'missing release asset: install.ps1' in done.stderr, done.stderr
    assert not (tmp_path / 'edited').exists(), 'a refused candidate was promoted anyway'


@shims_are_bash
def test_a_stable_release_is_not_re_promoted(tmp_path: Path):
    environment = gh_shim(tmp_path, prerelease=False, assets=complete_assets())
    done = run_promote(environment, '--publish', '--confirmed-install-smoke')
    assert done.returncode != 0
    assert 'prerelease candidate' in done.stderr, done.stderr


@shims_are_bash
def test_a_machine_on_the_old_release_refuses_the_promotion(tmp_path: Path):
    environment = gh_shim(tmp_path, assets=complete_assets(), fleet='0.0.1')
    done = run_promote(environment, '--publish', '--confirmed-install-smoke')
    assert done.returncode != 0, done.stdout
    assert f'./scripts/deploy.sh --release {VERSION} --only pc' in done.stderr, done.stderr
    assert not (tmp_path / 'edited').exists(), 'promoted with a machine behind'


@shims_are_bash
def test_an_unreachable_mac_refuses_until_it_is_allowed(tmp_path: Path):
    environment = gh_shim(tmp_path, assets=complete_assets(), mac_down=True)
    done = run_promote(environment, '--publish', '--confirmed-install-smoke')
    assert done.returncode != 0, done.stdout
    assert '--allow-unreachable mac' in done.stderr, done.stderr
    assert not (tmp_path / 'edited').exists()

    allowed = run_promote(environment, '--publish', '--confirmed-install-smoke',
                          '--allow-unreachable', 'mac')
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert 'not checked, because --allow-unreachable: mac (unreachable)' in allowed.stdout
    assert (tmp_path / 'edited').exists()


def test_allow_unreachable_names_a_machine_of_the_fleet(tmp_path: Path):
    done = run_promote(dict(os.environ), '--allow-unreachable', 'wsl')
    assert done.returncode == 2
    assert 'the fleet is: pc mac' in done.stderr, done.stderr


def test_promote_imports_nothing_from_the_deleted_pack_module():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, 'body', None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    imported: set[str] = set()
    identifiers: set[str] = set()
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module.split('.')[0])
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            literals.append(node.value)

    assert 'envpack' not in imported, 'promote_release.py still imports envpack'
    for gone in ['PackManifest', 'every_pack', 'read_manifest', 'part_filename',
                 'pack_target', 'recipe_digest', 'manifest_url', 'MANIFEST_NAME']:
        assert gone not in identifiers, f'promote_release.py still calls {gone}'
    for literal in literals:
        for gone in ['rootfs', 'envpacks.json', '.tar.zst']:
            assert gone not in literal, f'promote_release.py still names {gone}: {literal!r}'
