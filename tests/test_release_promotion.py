from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from crucible import VERSION

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/promote_release.py'

spec = importlib.util.spec_from_file_location('promote_release', SCRIPT)
promote_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote_release)

WHEEL_BYTES = b'not really a wheel, but bytes with a digest'
WHEEL_DIGEST = hashlib.sha256(WHEEL_BYTES).hexdigest()


def complete_names(version: str) -> tuple[list[str], list[str]]:
    uploaded = promote_release.uploaded_asset_names(version)
    wheel = next(name for name in uploaded if name.endswith('-py3-none-any.whl'))
    return uploaded, list(dict.fromkeys([*uploaded, f'{wheel}.sha256']))


def published(version: str = '1.2.3') -> tuple[list[str], list[dict]]:
    uploaded, names = complete_names(version)
    return uploaded, [{'name': name, 'size': 10} for name in names]


def fetcher(overrides: dict[str, bytes] | None = None):
    def fetch(version: str, name: str) -> bytes:
        if overrides and name in overrides:
            return overrides[name]
        if name.endswith('-py3-none-any.whl'):
            return WHEEL_BYTES
        if name.endswith('.whl.sha256'):
            return f'{WHEEL_DIGEST}  {name[:-len(".sha256")]}\n'.encode()
        return b'some asset'
    return fetch


def test_the_expected_assets_are_read_out_of_release_sh():
    names = promote_release.uploaded_asset_names(VERSION)
    assert names, names
    assert [name for name in names if name.endswith('-py3-none-any.whl')] == [
        f'crucible-{VERSION}-py3-none-any.whl'], names
    assert 'install.sh' in names and 'install.ps1' in names, names
    assert not any('/' in name for name in names), names
    for name in names:
        assert VERSION in name or name in ('install.sh', 'install.ps1'), name


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
    uploaded, assets = published()
    promote_release.validate_assets(assets, '1.2.3', fetch=fetcher(), uploaded=uploaded)


@pytest.mark.parametrize('missing', ['install.ps1', 'install.sh',
                                     'crucible-1.2.3.tar.gz',
                                     'crucible-client-1.2.3.tgz', WHEEL])
def test_a_missing_asset_is_refused_by_name(missing: str):
    uploaded, assets = published()
    remaining = [asset for asset in assets if asset['name'] != missing]
    assert len(remaining) == len(assets) - 1, missing
    with pytest.raises(ValueError, match=f'missing release asset: {re.escape(missing)}'):
        promote_release.validate_assets(remaining, '1.2.3', fetch=fetcher(),
                                        uploaded=uploaded)


def test_the_digest_asset_is_required():
    uploaded, assets = published()
    remaining = [asset for asset in assets if asset['name'] != WHEEL_SHA]
    with pytest.raises(ValueError, match=f'missing release asset: {re.escape(WHEEL_SHA)}'):
        promote_release.validate_assets(
            remaining, '1.2.3', fetch=fetcher(),
            uploaded=[name for name in uploaded if name != WHEEL_SHA])


def test_an_empty_asset_is_refused_by_name():
    uploaded, assets = published()
    for asset in assets:
        if asset['name'] == 'install.sh':
            asset['size'] = 0
    with pytest.raises(ValueError, match='empty release asset: install.sh'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetcher(), uploaded=uploaded)


def test_a_wheel_that_does_not_match_its_digest_is_refused():
    uploaded, assets = published()
    fetch = fetcher({WHEEL: b'a different wheel entirely'})
    with pytest.raises(ValueError,
                       match=f'{re.escape(WHEEL)} hashes to .* attests {WHEEL_DIGEST}'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetch, uploaded=uploaded)


def test_a_digest_file_that_is_not_a_digest_is_refused():
    uploaded, assets = published()
    fetch = fetcher({WHEEL_SHA: b'<html>404 Not Found</html>\n'})
    with pytest.raises(ValueError,
                       match=f'{re.escape(WHEEL_SHA)} does not begin with a sha256'):
        promote_release.validate_assets(assets, '1.2.3', fetch=fetch, uploaded=uploaded)


def test_an_asset_from_another_release_is_refused():
    uploaded, assets = published()
    stray = [*assets, {'name': 'crucible-0.0.9-py3-none-any.whl', 'size': 10}]
    with pytest.raises(ValueError, match='names version 0.0.9, not 1.2.3'):
        promote_release.validate_assets(stray, '1.2.3', fetch=fetcher(), uploaded=uploaded)


def test_a_release_with_two_wheels_is_refused():
    uploaded, _ = published()
    with pytest.raises(ValueError, match='exactly one wheel'):
        promote_release.validate_assets(
            [], '1.2.3', fetch=fetcher(),
            uploaded=[*uploaded, 'crucible-1.2.4-py3-none-any.whl'])


def gh_shim(tmp_path: Path, *, assets: list[dict], prerelease: bool = True,
            draft: bool = False) -> dict[str, str]:
    metadata = {'tagName': f'v{VERSION}', 'isDraft': draft,
                'isPrerelease': prerelease, 'assets': assets}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    shims = tmp_path / 'shims'
    shims.mkdir()
    gh = shims / 'gh'
    gh.write_text(
        '#!/bin/bash\n'
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
        '      *) printf "some asset" ;;\n'
        '    esac\n'
        '    ;;\n'
        '  edit) printf "%s\\n" "$@" >> "$GH_TEST_DIR/edited" ;;\n'
        '  *) echo "unexpected gh $*" >&2; exit 9 ;;\n'
        'esac\n',
        encoding='utf-8')
    gh.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        PATH=str(shims) + os.pathsep + environment['PATH'],
        GH_TEST_DIR=str(tmp_path),
        GH_TEST_WHEEL=WHEEL_BYTES.decode(),
        GH_TEST_DIGEST=WHEEL_DIGEST,
    )
    return environment


def run_promote(environment: dict[str, str], *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), '--tag', f'v{VERSION}', *argv],
                          capture_output=True, text=True, timeout=120,
                          env=environment, cwd=str(REPO))


@pytest.fixture
def candidate(tmp_path: Path) -> dict[str, str]:
    _, complete = complete_names(VERSION)
    return gh_shim(tmp_path, assets=[{'name': name, 'size': 10} for name in complete])


@pytest.mark.skipif(sys.platform == 'win32', reason='the gh shim is a `#!/bin/bash` script')
def test_a_complete_candidate_is_promoted(tmp_path: Path, candidate: dict[str, str]):
    done = run_promote(candidate, '--publish', '--confirmed-install-smoke')
    assert done.returncode == 0, done.stdout + done.stderr
    edited = (tmp_path / 'edited').read_text(encoding='utf-8').split()
    assert '--latest=true' in edited and '--prerelease=false' in edited, edited


@pytest.mark.skipif(sys.platform == 'win32', reason='the gh shim is a `#!/bin/bash` script')
def test_the_default_is_read_only(tmp_path: Path, candidate: dict[str, str]):
    done = run_promote(candidate)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'not promoted' in done.stdout, done.stdout
    assert not (tmp_path / 'edited').exists(), 'a read-only run edited the release'


@pytest.mark.skipif(sys.platform == 'win32', reason='the gh shim is a `#!/bin/bash` script')
def test_publishing_without_the_attestation_is_refused(tmp_path: Path,
                                                       candidate: dict[str, str]):
    done = run_promote(candidate, '--publish')
    assert done.returncode != 0
    assert '--confirmed-install-smoke' in done.stderr, done.stderr
    assert not (tmp_path / 'edited').exists()


@pytest.mark.skipif(sys.platform == 'win32', reason='the gh shim is a `#!/bin/bash` script')
def test_a_missing_installer_stops_the_whole_command(tmp_path: Path):
    _, complete = complete_names(VERSION)
    environment = gh_shim(tmp_path, assets=[{'name': name, 'size': 10}
                                            for name in complete if name != 'install.ps1'])
    done = run_promote(environment, '--publish', '--confirmed-install-smoke')
    assert done.returncode != 0
    assert 'missing release asset: install.ps1' in done.stderr, done.stderr
    assert not (tmp_path / 'edited').exists(), 'a refused candidate was promoted anyway'


@pytest.mark.skipif(sys.platform == 'win32', reason='the gh shim is a `#!/bin/bash` script')
def test_a_stable_release_is_not_re_promoted(tmp_path: Path):
    _, complete = complete_names(VERSION)
    environment = gh_shim(tmp_path, prerelease=False,
                          assets=[{'name': name, 'size': 10} for name in complete])
    done = run_promote(environment, '--publish', '--confirmed-install-smoke')
    assert done.returncode != 0
    assert 'prerelease candidate' in done.stderr, done.stderr


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
