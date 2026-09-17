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


# ------------------------------------------------------------------ the README
#
# The release section of the README is the first thing a person reads before
# cutting one, and prose outliving the code it describes is the failure this
# repo keeps finding. It claimed a manifest "names the actual uploaded parts"
# for a day after pack reuse made that false. These hold the replacement.

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


def test_the_readme_does_not_promise_a_self_contained_manifest():
    """The sentence that pack reuse falsified, kept falsified."""
    section = release_section()
    assert 'names the actual uploaded parts' not in section
    assert 'carried by reference' in section
    assert 'RESOLVES' in section


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
