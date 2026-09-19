"""The release path is one door, and every script it names must exist.

`scripts/ship.sh` runs and prints a sequence of other scripts. A renamed or
deleted one is not found until somebody is midway through a release, which is
the worst moment to discover it — the tag may already exist. These are cheap
string checks, and cheap is the point: they run in milliseconds and they fail
the moment the chain is broken.
"""
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
    """Asked of git, not of the filesystem.

    This repo is worked on from Windows, where `core.filemode` is false and
    `os.stat` reports no executable bit for any file — so a filesystem check
    here fails on every script, including the ones that have always worked. The
    mode that decides whether `./scripts/ship.sh` runs on Linux and in CI is the
    one in git's index, and that is the same fact on every machine.
    """
    relative = script.relative_to(REPO).as_posix()
    listed = subprocess.check_output(['git', 'ls-files', '-s', '--', relative],
                                     cwd=REPO, text=True).strip()
    assert listed, f'{relative} is not tracked by git'
    mode = listed.split()[0]
    assert mode == '100755', (
        f'{relative} is mode {mode}; it will not be executable on Linux. '
        f'Fix with: git update-index --chmod=+x {relative}')


def test_ship_only_names_scripts_that_exist():
    """Every `scripts/<something>` ship.sh mentions is a file in this checkout."""
    text = SHIP.read_text(encoding='utf-8')
    named = set(re.findall(r'scripts/[A-Za-z0-9_.-]+', text))
    assert named, 'ship.sh names no scripts at all, which cannot be right'
    for relative in sorted(named):
        assert (REPO / relative).is_file(), f'ship.sh names {relative}, which does not exist'


def test_ship_runs_the_steps_in_the_order_it_documents():
    """The header lists the steps; the body must call them in that order."""
    text = SHIP.read_text(encoding='utf-8')
    order = ['scripts/bump.py', 'scripts/tests.sh', 'scripts/release.sh', 'scripts/deploy.sh',
             'scripts/promote_release.py']
    positions = []
    for name in order:
        # The LAST mention, so a header that lists them all does not decide this.
        positions.append(text.rindex(name))
    assert positions == sorted(positions), (
        f'ship.sh calls its steps out of order: {list(zip(order, positions))}')


def test_ship_never_passes_the_install_attestation_itself():
    """`--confirmed-install-smoke` asserts a human installed the candidate.

    A script that passes it is a script that attests to something it cannot
    know. ship.sh may PRINT the command; it may not run it.
    """
    text = SHIP.read_text(encoding='utf-8')
    for line in text.splitlines():
        if '--confirmed-install-smoke' not in line:
            continue
        stripped = line.strip()
        assert stripped.startswith('#') or stripped.startswith('echo '), (
            f'ship.sh does something other than print the attestation flag: {stripped}')


def test_deploy_reads_the_same_file_on_every_machine():
    """installation.json is the one fact with one meaning on all three platforms."""
    text = DEPLOY.read_text(encoding='utf-8')
    readers = re.findall(r'^read_(\w+)\(\) \{(.*?)^\}', text, re.MULTILINE | re.DOTALL)
    assert len(readers) == 3, f'expected three machines, found {[name for name, _ in readers]}'
    for name, body in readers:
        assert 'installation.json' in body, f'read_{name} does not read installation.json'
        assert 'parse_release' in body, f'read_{name} does not use the shared parser'


def test_deploy_never_reports_an_unreachable_machine_as_current():
    """A machine that could not be asked has an UNKNOWN version, not a matching one."""
    text = DEPLOY.read_text(encoding='utf-8')
    assert 'unreachable' in text
    # The word must reach the failure summary, not only the report.
    after_prompt = text.split('the work', 1)[-1]
    assert 'unreachable' in after_prompt, (
        'deploy.sh notices an unreachable machine but does not carry it into the result')


def test_the_selector_widens_when_it_does_not_understand_a_file():
    """The safe direction is MORE tests; a silent skip is the failure mode."""
    text = TESTS.read_text(encoding='utf-8')
    assert 'no test names it' in text
    assert re.search(r'everything="\$everything .*no test names it', text), (
        'tests.sh no longer routes an unrecognised file to the whole suite')


def test_the_selector_treats_its_own_widening_list_as_wide():
    """conftest, the fakes and the workflows must each still select everything."""
    text = TESTS.read_text(encoding='utf-8')
    wide = text.split('is_wide()', 1)[1].split('}', 1)[0]
    for needed in ['conftest.py', 'fake_*.py', '.github/*', 'crucible/app.py']:
        assert needed in wide, f'{needed} is no longer a whole-suite trigger'


def test_the_two_files_every_release_touches_are_not_unconditionally_wide():
    """`crucible/__init__.py` and `pyproject.toml` both carry the version.

    If either is wide, every release runs the whole suite no matter what the
    release contains — the selector switches itself off exactly when it is being
    asked to work. They are allowed through only when the diff is the version
    line alone, and that condition must still be attached to both of them.
    """
    text = TESTS.read_text(encoding='utf-8')
    wide = text.split('is_wide()', 1)[1].split('}', 1)[0]
    conditional = [line for line in wide.splitlines() if 'version_line_only' in line]
    assert len(conditional) == 1, 'the version-line exception is not a single rule any more'
    for carrier in ['crucible/__init__.py', 'pyproject.toml']:
        assert carrier in conditional[0], f'{carrier} no longer goes through version_line_only'
        assert f'{carrier})' in conditional[0] or f'{carrier}|' in conditional[0]


# ------------------------------------------------------- what a release IS
#
# PHASE20-CODE-NOT-ENVIRONMENTS.md section 1, in the two places a person reads
# it: the script that cuts one, and the README section they read first. Prose
# outliving the code it describes is the failure this repo keeps finding.

RELEASE = REPO / 'scripts/release.sh'


def test_the_release_uploads_our_code_and_nothing_that_is_published_elsewhere():
    """The asset list, exactly. A release that carried an interpreter or an
    environment again would be the 190-MB-per-tag shape PHASE20 deleted, and
    the only place that can be seen is the `gh release create` line."""
    text = RELEASE.read_text(encoding='utf-8')
    upload = text[text.index('gh release create'):]
    upload = upload[:upload.index('\n\n')]
    assert '"$SDIST" "$WHEEL" "$WHEEL_SHA" "$TGZ" "$BOOT"' in upload, upload
    assert '"$INSTALL_SH" "$INSTALL_PS1"' in upload, upload
    for gone in ('envpacks', 'rootfs', 'part0', '.tar.zst'):
        assert gone not in upload, f'the release still uploads {gone!r}'


def test_the_wheel_ships_its_own_digest_because_the_installers_check_it():
    """The one thing a release publishes about ITS OWN bytes that an installer
    can check: the interpreter is pinned by a digest in our source, and our
    code cannot be, so the sidecar is what vouches for the wheel."""
    text = RELEASE.read_text(encoding='utf-8')
    assert 'WHEEL_SHA="$WHEEL.sha256"' in text
    assert 'sha256sum' in text and 'shasum -a 256' in text, (
        'macOS has no sha256sum, and a release is cut from whichever desk is free'
    )


def test_the_release_dispatches_no_workflow_and_there_is_none_to_dispatch():
    """`release.sh` used to `gh workflow run envpacks.yml` after the cut, and
    then a person waited 13 minutes per backend for archives of somebody
    else's bytes. Both halves are gone: the dispatch, and the workflow."""
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
    """PHASE20 section 1, in the section a person reads before cutting one.

    This test used to hold the opposite claim honest — "the manifest does not
    name only its own assets", which was true of the pack world. There is no
    manifest now and nothing carried by reference, so what it holds is the
    ruling that replaced both.
    """
    section = release_section()
    assert 'A release carries CODE' in section
    assert 'crucible-<ver>-py3-none-any.whl.sha256' in section
    assert 'python-build-standalone' in section
    assert 'Canonical' in section
    for gone in ('envpacks.json', 'WSL rootfs', 'carried by reference'):
        assert gone not in section, f'the README still describes {gone!r}'


def test_the_readme_says_a_deploy_runs_no_tests():
    """PHASE20 section 7. 25 of the 39 minutes before the first byte of the
    2026-09-18 cutover moved were suites that had already passed at merge."""
    section = release_section()
    assert 'no CI to wait for' in section
    assert 'scripts/tests.sh' in section, 'the branch still has a suite, and it is still named'
    assert 'a deploy runs none' in section


def test_the_readme_says_cutting_and_promoting_are_two_jobs():
    """Six promotions were missed in a row; the README is where that is now said."""
    section = release_section()
    assert 'two jobs' in section
    assert '--confirmed-install-smoke' in section


def test_the_selector_discards_a_name_that_matches_almost_everything():
    """`scripts/tests.sh` has the stem "tests", which every test file contains.

    Selecting on it returned all 75 files with a confident per-file reason for
    each — a selector that cannot tell "this really is everything" from "my
    search term was useless" is worse than none, because it looks like it
    worked. The threshold is what tells those apart.
    """
    text = TESTS.read_text(encoding='utf-8')
    assert 'narrows nothing' in text, 'the uninformative-name guard is gone'
    assert re.search(r'total \* \d+ / \d+', text), 'the guard no longer compares against the total'


def test_the_selector_only_uses_a_directory_name_inside_the_package():
    """Outside `crucible/`, a directory is a place, not a subject.

    "scripts" appears in twenty-three test files and says nothing about any of
    them; "voices" is what a test means when it reads `crucible/voices/*.toml`.
    """
    text = TESTS.read_text(encoding='utf-8')
    assert 'case "$path" in crucible/*/*) candidates="$candidates $parent" ;; esac' in text, (
        'the directory candidate is no longer restricted to package data directories')


def test_the_selector_fetches_tags_before_deciding_what_is_new():
    """Tags are created server-side by `gh`, so a checkout does not have them.

    Measured 2026-09-17: local tags stopped at v0.6.0 while the remote was at
    v0.6.7, so `git describe` named a tag seven releases old and "since the last
    release" silently meant "since seven releases ago".
    """
    text = TESTS.read_text(encoding='utf-8')
    fetch = text.index('git fetch --tags')
    describe = text.index('git describe --tags')
    assert fetch < describe, 'tests.sh asks what the last tag is before fetching the tags'
    assert 'run_all' in text[fetch:describe], (
        'a failed tag fetch must widen to the whole suite, not proceed on a stale tag')


def test_deploy_waits_for_the_record_rather_than_reading_it_once():
    """`installation.json` is published when the RUNTIME starts, not by the installer.

    On Windows especially — install.ps1 unpacks the host pack, launches the host
    and returns — the file can still hold the old release the instant the
    installer exits successfully. Reading it once is reading a race.
    """
    text = DEPLOY.read_text(encoding='utf-8')
    assert 'await_release()' in text, 'the bounded wait for the record is gone'
    assert 'after="$(await_release' in text, 'the after-check reads the record directly again'


def test_ship_does_not_gate_on_a_dry_run_it_has_made_impossible():
    """`release.sh --dry-run` refuses a dirty tree and an unpushed HEAD.

    A just-bumped working tree is both, so calling it between the bump and the
    commit refuses every single time — which is what the first version of
    ship.sh did, and what its first rehearsal caught. The cut's own run is the
    gate: it checks everything and builds every asset before creating anything.
    """
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
    """One tool, named once, never resolved through PATH.

    `tar` resolved through PATH found Git Bash's GNU tar 1.32 on a Windows 11
    machine whose System32 bsdtar was fine, and the 0.6.8 deploy refused it
    (measured 2026-09-17).

    THE ZSTD CHECK WENT WITH THE PACKS. python-build-standalone publishes gzip,
    which every tar reads, so a probe for a format nothing downloads would
    refuse a machine over a capability it does not need.
    """
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
    """Asked, not named.

    A bare `ssh host cmd` gets PATH=/usr/bin:/bin:/usr/sbin:/sbin on macOS, and
    the installer then probes for curl/tar/zstd in an environment its owner
    never uses. `bash -lc` does not fix it either: this account is zsh and its
    Homebrew line is in ~/.zprofile, which bash does not read.
    """
    text = DEPLOY.read_text(encoding='utf-8')
    start = text.index('install_mac()')
    body = text[start:text.index('install_windows()')]
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
    """The probe list and the tools the script runs must not drift apart.

    It was `curl tar zstd` while the installer fetched 8 GB environment
    archives. PHASE20 deleted those; python-build-standalone publishes gzip,
    which every tar reads, so `zstd` became a tool the probe demanded and
    nothing used — a refusal that costs somebody an afternoon over nothing.
    """
    text = INSTALL_SH.read_text(encoding='utf-8')
    assert 'for t in curl tar;' in text, (
        'the POSIX installer probes curl and tar by name; if that list moved, '
        'this test is the place to say what it moved to'
    )
    assert 'zstd' not in text, 'nothing this installer fetches is zstd'
    assert 'tar -xzf' in text, 'and what it probed for is what it runs'

def test_no_installer_is_piped_straight_into_a_shell():
    """`curl | sh` throws curl's exit status away.

    The remote shell runs whatever arrived and its own status is all `set -e`
    can see, so a transfer cut in half is an installer that runs half. Measured
    2026-09-17 installing 0.6.11 into WSL: `sh: 352: Syntax error: Unterminated
    quoted string` from a published install.sh that is byte-identical to the
    repo's and passes `sh -n`.
    """
    text = DEPLOY.read_text(encoding='utf-8')
    code = '{}'.format(chr(10)).join(
        line for line in text.splitlines() if not line.lstrip().startswith('#')
    )
    assert '| sh -s' not in code and '| sh ' not in code, (
        'an installer piped into a shell cannot report a truncated download'
    )
    # NOT `'sh -n' in code`: that matches `ssh -n`, which has been in this
    # file all along, so the assertion passed against the very version it
    # was written to catch.
    assert 'sh -n \"$f\"' in code, (
        'the fetched script is parsed before it is run'
    )


def test_the_mac_payload_is_quoted_for_its_extra_shell():
    """`ssh host '"$SHELL" -lc <payload>'` is parsed once BEFORE $SHELL sees it.

    WSL's `--exec bash -lc <payload>` passes an argv element and nothing
    re-reads it. The Mac has one more parse, so a payload containing double
    quotes ends the string early - measured while building this: WSL took the
    same payload and the Mac answered `no such file or directory`.
    """
    text = DEPLOY.read_text(encoding='utf-8')
    assert 'shquote()' in text, 'the extra parse needs a quoter'
    start = text.index('install_mac()')
    body = text[start:text.index('install_windows()')]
    assert 'shquote' in body, 'the mac payload must go through it'


def test_deploy_can_reinstall_a_machine_that_already_names_the_release():
    """A record is written PARTWAY through an install.

    So a run that died after `local-register` leaves a machine claiming the
    version with its later steps never run - and the retry then skipped it as
    already done (measured 2026-09-17). --force is the way to say otherwise.
    """
    text = DEPLOY.read_text(encoding='utf-8')
    assert '--force' in text
    assert 'force=0' in text, 'and it must default to off'


# ------------------------------------------------------------------- the CI
#
# A guard that has gone red and cannot say so is not a guard. `sdk/bootstrap`
# had 278 tests and five of them had been failing since the install scripts
# were rewritten — nothing ran them on a push, so nothing said. `sdk/ts` was
# installed, built and packed by the `sdk` job and its 300-odd tests were never
# run either. These hold the door open once it has been opened.

CI = REPO / '.github/workflows/ci.yml'

#: Every npm package in this repo whose tests CI must run. Derived from the
#: tree rather than typed twice: a third SDK added beside these two gets the
#: same treatment or fails here, which is the whole reason the list is not a
#: literal.
SDK_PACKAGES = sorted(
    path.parent.relative_to(REPO).as_posix()
    for path in (REPO / 'sdk').glob('*/package.json')
)


def test_there_are_sdk_packages_to_run_at_all():
    """An empty list would make every assertion below vacuously true."""
    assert SDK_PACKAGES == ['sdk/bootstrap', 'sdk/ts'], SDK_PACKAGES


@pytest.mark.parametrize('package', SDK_PACKAGES)
def test_ci_runs_the_tests_of_every_sdk_package(package):
    """A `run: npm test` attached to THIS package's directory.

    The directory alone proves nothing: `sdk/ts` was entered three times — to
    install, to build and to pack — and ran no test in any of them. A step is
    two lines, so the pairing is what is asserted.
    """
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
    """One `npm test` per package, counted, because one for both is one untested.

    The defect this replaces was not "no tests anywhere" — it was a job that
    did enough with `sdk/ts` to look thorough. A count is what tells a job that
    runs both from a job that runs the one somebody remembered.
    """
    text = CI.read_text(encoding='utf-8')
    assert text.count('run: npm test') == len(SDK_PACKAGES), (
        f'ci.yml runs `npm test` {text.count("run: npm test")} time(s) for '
        f'{len(SDK_PACKAGES)} SDK package(s)')


def test_ci_builds_the_client_before_installing_the_bootstrap():
    """`sdk/bootstrap` depends on `@crucible/client` as `file:../ts`.

    npm links the directory rather than packing it, so bootstrap's compile
    reads `sdk/ts/dist/esm/index.d.ts` off disk. A job that installed bootstrap
    before building the client would fail on a missing types file, which reads
    as a broken SDK rather than as a step in the wrong order.
    """
    text = CI.read_text(encoding='utf-8')
    build_client = text.index('working-directory: sdk/ts\n        run: npm run build')
    bootstrap = text.index('working-directory: sdk/bootstrap')
    assert build_client < bootstrap, (
        'ci.yml installs sdk/bootstrap before sdk/ts has been built')
