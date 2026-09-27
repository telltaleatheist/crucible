from pathlib import Path
import re
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]

SHIP = REPO / 'scripts/ship.sh'
TESTS = REPO / 'scripts/tests.sh'
DEPLOY = REPO / 'scripts/deploy.sh'
BUMP = REPO / 'scripts/bump.py'


@pytest.mark.parametrize('script', [SHIP, TESTS, DEPLOY, BUMP])
def test_it_exists(script):
    assert script.is_file(), f'{script.relative_to(REPO)} is gone'


@pytest.mark.parametrize('script', [SHIP, TESTS, DEPLOY])
def test_it_is_executable(script):
    relative = script.relative_to(REPO).as_posix()
    listed = subprocess.check_output(['git', 'ls-files', '-s', '--', relative],
                                     cwd=REPO, text=True).strip()
    assert listed, f'{relative} is not tracked by git'
    mode = listed.split()[0]
    assert mode == '100755', (
        f'{relative} is mode {mode}; it will not be executable on Linux. '
        f'Fix with: git update-index --chmod=+x {relative}')


def test_ship_only_names_scripts_that_exist():
    text = SHIP.read_text(encoding='utf-8')
    named = set(re.findall(r'scripts/[A-Za-z0-9_.-]+', text))
    assert named, 'ship.sh names no scripts at all, which cannot be right'
    for relative in sorted(named):
        assert (REPO / relative).is_file(), f'ship.sh names {relative}, which does not exist'


def test_ship_runs_the_steps_in_the_order_it_documents():
    text = SHIP.read_text(encoding='utf-8')
    header = text[:text.index('set -euo pipefail')]
    order = re.findall(r'^#\s+\d+\.\s.*?\((scripts/[A-Za-z0-9_.-]+)', header, re.MULTILINE)
    assert order, 'the header no longer numbers its steps with the script each runs'
    positions = []
    for name in order:
        positions.append(text.rindex(name))
    assert positions == sorted(positions), (
        f'ship.sh calls its steps out of order: {list(zip(order, positions))}')


def test_ship_never_passes_the_install_attestation_itself():
    text = SHIP.read_text(encoding='utf-8')
    for line in text.splitlines():
        if '--confirmed-install-smoke' not in line:
            continue
        stripped = line.strip()
        assert stripped.startswith('#') or stripped.startswith('echo '), (
            f'ship.sh does something other than print the attestation flag: {stripped}')


def test_deploy_reads_the_same_file_wherever_it_lives():
    text = DEPLOY.read_text(encoding='utf-8')
    records = re.findall(r'^(record_\w+|read_mac)\(\) \{(.*?)^\}', text,
                         re.MULTILINE | re.DOTALL)
    assert sorted(name for name, _ in records) == ['read_mac', 'record_guest', 'record_host'], (
        [name for name, _ in records])
    for name, body in records:
        assert 'installation.json' in body, f'{name} does not read installation.json'
        assert 'parse_release' in body, f'{name} does not use the shared parser'


def test_the_pc_verdict_is_taken_from_both_of_its_records():
    text = DEPLOY.read_text(encoding='utf-8')
    start = text.index('read_pc() {')
    body = text[start:text.index('\n}', start)]
    for half in ['record_host', 'record_guest']:
        assert half in body, f'read_pc no longer consults {half}'
    assert 'host:$host guest:$guest' in body, (
        'a PC whose two records disagree must print both halves, or the '
        'summary cannot say which one is behind')


def test_deploy_never_reports_an_unreachable_machine_as_current():
    text = DEPLOY.read_text(encoding='utf-8')
    assert 'unreachable' in text
    after_prompt = text.split('the work', 1)[-1]
    assert 'unreachable' in after_prompt, (
        'deploy.sh notices an unreachable machine but does not carry it into the result')


def test_the_selector_says_which_files_no_test_names():
    text = TESTS.read_text(encoding='utf-8')
    assert 'unnamed="$unnamed $path"' in text, (
        'tests.sh no longer collects the files no test names')
    assert 'no test in tests/ names:$unnamed' in text, (
        'tests.sh collects them and does not print them, which is the silent '
        'skip this was written against')


RELEASE = REPO / 'scripts/release.sh'


def test_the_release_uploads_our_code_and_nothing_that_is_published_elsewhere():
    text = RELEASE.read_text(encoding='utf-8')
    upload = text[text.index('gh release create'):]
    upload = upload[:upload.index('\n\n')]
    assert '"$SDIST" "$WHEEL" "$WHEEL_SHA" "$TGZ" "$BOOT"' in upload, upload
    assert '"$INSTALL_SH" "$INSTALL_PS1"' in upload, upload
    for gone in ('envpacks', 'rootfs', 'part0', '.tar.zst'):
        assert gone not in upload, f'the release still uploads {gone!r}'


def test_the_wheel_ships_its_own_digest_because_the_installers_check_it():
    text = RELEASE.read_text(encoding='utf-8')
    assert 'WHEEL_SHA="$WHEEL.sha256"' in text
    assert 'sha256sum' in text and 'shasum -a 256' in text, (
        'macOS has no sha256sum, and a release is cut from whichever desk is free'
    )


def test_the_release_dispatches_no_workflow_and_there_is_none_to_dispatch():
    text = RELEASE.read_text(encoding='utf-8')
    assert 'gh workflow run' not in text
    for workflow in ('envpacks.yml', 'release-core.yml', 'release-preflight.yml'):
        assert not (REPO / '.github/workflows' / workflow).exists(), (
            f'{workflow} is back; PHASE20 section 6 deleted what it built'
        )

README = REPO / 'README.md'


def release_section() -> str:
    text = README.read_text(encoding='utf-8')
    start = text.index('## Releases')
    return text[start:text.index('\n## ', start + 1)]


def test_the_readme_names_every_step_of_the_release_path():
    section = release_section()
    for script in ['scripts/ship.sh', 'scripts/bump.py', 'scripts/tests.sh',
                   'scripts/release.sh', 'scripts/deploy.sh', 'scripts/promote_release.py']:
        assert script in section, f'the README no longer tells anyone about {script}'


def test_the_readme_says_what_a_release_carries_and_what_it_does_not():
    section = release_section()
    assert 'A release carries CODE' in section
    assert 'crucible-<ver>-py3-none-any.whl.sha256' in section
    assert 'python-build-standalone' in section
    assert 'Canonical' in section
    for gone in ('envpacks.json', 'WSL rootfs', 'carried by reference'):
        assert gone not in section, f'the README still describes {gone!r}'


def test_the_readme_says_a_deploy_runs_no_tests():
    section = release_section()
    assert 'no CI to wait for' in section
    assert 'scripts/tests.sh' in section, 'the branch still has a suite, and it is still named'
    assert 'a deploy runs none' in section


def test_the_readme_says_cutting_and_promoting_are_two_jobs():
    section = release_section()
    assert 'two jobs' in section
    assert '--confirmed-install-smoke' in section


def test_the_selector_discards_a_name_that_matches_almost_everything():
    text = TESTS.read_text(encoding='utf-8')
    assert 'narrows nothing' in text, 'the uninformative-name guard is gone'
    assert re.search(r'total \* \d+ / \d+', text), 'the guard no longer compares against the total'


def test_the_selector_never_searches_on_a_directory_name():
    text = TESTS.read_text(encoding='utf-8')
    assert 'basename "$(dirname' not in text, (
        'a directory is being made into a search term again')
    assert '$parent' not in text, 'the directory candidate is back'


def test_the_selector_fetches_tags_before_deciding_what_is_new():
    text = TESTS.read_text(encoding='utf-8')
    fetch = text.index('git fetch --tags')
    describe = text.index('git describe --tags')
    assert fetch < describe, 'tests.sh asks what the last tag is before fetching the tags'
    assert 'run_all' in text[fetch:describe], (
        'a failed tag fetch must widen to the whole suite, not proceed on a stale tag')


def test_deploy_waits_for_the_record_rather_than_reading_it_once():
    text = DEPLOY.read_text(encoding='utf-8')
    assert 'await_release()' in text, 'the bounded wait for the record is gone'
    assert 'after="$(await_release' in text, 'the after-check reads the record directly again'


def test_ship_does_not_gate_on_a_dry_run_it_has_made_impossible():
    text = SHIP.read_text(encoding='utf-8')
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('echo '):
            continue
        assert 'release.sh --dry-run' not in stripped, (
            f'ship.sh runs a gate that a bumped tree can never pass: {stripped}')

INSTALL_PS1 = REPO / 'sdk/bootstrap/scripts/install.ps1'
INSTALL_SH = REPO / 'sdk/bootstrap/scripts/install.sh'


def test_windows_unpacks_with_the_tar_windows_guarantees():
    text = INSTALL_PS1.read_text(encoding='utf-8')
    body = '{}'.format(chr(10)).join(
        line for line in text.splitlines() if not line.lstrip().startswith('#')
    )
    assert 'Join-Path $env:SystemRoot "System32' in body, (
        'the Windows installer must name the tar Windows guarantees'
    )
    assert '& tar.exe' not in body, (
        'a PATH-resolved tar.exe is back; it is the caller shell that decides '
        'which one that is'
    )
    assert body.count('& $Tar') >= 1, 'the unpack has to use the named tar'
    assert 'zstd' not in body, 'nothing this installer fetches is zstd'


def test_the_mac_install_runs_under_the_accounts_own_login_shell():
    text = DEPLOY.read_text(encoding='utf-8')
    start = text.index('install_mac()')
    body = text[start:text.index('install_pc()')]
    code = '{}'.format(chr(10)).join(
        line for line in body.splitlines() if not line.lstrip().startswith('#')
    )
    assert '$SHELL' in code, (
        'the mac install must ask the account which shell it uses'
    )
    assert '-lc' in code, 'and run it as a LOGIN shell, or the profile is not read'
    assert 'bash -lc' not in code, (
        'naming bash guesses at a shell this account does not use'
    )


def test_every_installer_probes_for_the_tools_it_actually_runs():
    text = INSTALL_SH.read_text(encoding='utf-8')
    assert 'for t in curl tar;' in text, (
        'the POSIX installer probes curl and tar by name; if that list moved, '
        'this test is the place to say what it moved to'
    )
    assert 'zstd' not in text, 'nothing this installer fetches is zstd'
    assert 'tar -xzf' in text, 'and what it probed for is what it runs'

def test_no_installer_is_piped_straight_into_a_shell():
    text = DEPLOY.read_text(encoding='utf-8')
    code = '{}'.format(chr(10)).join(
        line for line in text.splitlines() if not line.lstrip().startswith('#')
    )
    assert '| sh -s' not in code and '| sh ' not in code, (
        'an installer piped into a shell cannot report a truncated download'
    )
    assert 'sh -n \"$f\"' in code, (
        'the fetched script is parsed before it is run'
    )


def test_the_mac_payload_is_quoted_for_its_extra_shell():
    text = DEPLOY.read_text(encoding='utf-8')
    assert 'shquote()' in text, 'the extra parse needs a quoter'
    start = text.index('install_mac()')
    body = text[start:text.index('install_pc()')]
    assert 'shquote' in body, 'the mac payload must go through it'


def test_deploy_can_reinstall_a_machine_that_already_names_the_release():
    text = DEPLOY.read_text(encoding='utf-8')
    assert '--force' in text
    assert 'force=0' in text, 'and it must default to off'


CI = REPO / '.github/workflows/ci.yml'

SDK_PACKAGES = sorted(
    path.parent.relative_to(REPO).as_posix()
    for path in (REPO / 'sdk').glob('*/package.json')
)


def test_there_are_sdk_packages_to_run_at_all():
    assert SDK_PACKAGES == ['sdk/bootstrap', 'sdk/ts'], SDK_PACKAGES


@pytest.mark.parametrize('package', SDK_PACKAGES)
def test_ci_runs_the_tests_of_every_sdk_package(package):
    text = CI.read_text(encoding='utf-8')
    assert f'working-directory: {package}' in text, (
        f'.github/workflows/ci.yml never enters {package}')
    steps = [
        f'working-directory: {package}\n        run: npm test',
        f'run: npm test\n        working-directory: {package}',
    ]
    assert any(step in text for step in steps), (
        f'ci.yml enters {package} but never runs `npm test` there')


def test_ci_runs_npm_test_once_per_sdk_package():
    text = CI.read_text(encoding='utf-8')
    assert text.count('run: npm test') == len(SDK_PACKAGES), (
        f'ci.yml runs `npm test` {text.count("run: npm test")} time(s) for '
        f'{len(SDK_PACKAGES)} SDK package(s)')


def test_ci_builds_the_client_before_installing_the_bootstrap():
    text = CI.read_text(encoding='utf-8')
    build_client = text.index('working-directory: sdk/ts\n        run: npm run build')
    bootstrap = text.index('working-directory: sdk/bootstrap')
    assert build_client < bootstrap, (
        'ci.yml installs sdk/bootstrap before sdk/ts has been built')
