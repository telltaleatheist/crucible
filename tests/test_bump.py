import importlib.util
import re
from pathlib import Path

import pytest

from crucible import VERSION

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bump', REPO / 'scripts/bump.py')
bump = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bump)


def test_every_anchor_still_matches_its_file_exactly_once():
    for relative, pattern, description in bump.PLACES:
        text = (REPO / relative).read_text(encoding='utf-8')
        found = re.findall(pattern, text, re.MULTILINE)
        assert len(found) == 1, f'{relative}: {len(found)} matches for {description}'


def test_the_seven_agree_and_are_this_checkout():
    assert bump.read_current() == VERSION


@pytest.mark.parametrize('caller', ['scripts/release.sh', 'scripts/ship.sh',
                                    '.github/workflows/ci.yml'])
def test_every_version_check_is_bump_check_and_none_is_a_copy(caller):
    text = (REPO / caller).read_text(encoding='utf-8')
    assert 'scripts/bump.py --check' in text, f'{caller} does not ask bump.py --check'
    for copy in ('SDK_VERSION', 'BOOTSTRAP_VERSION', "package.json').version",
                 's/^VERSION = '):
        assert copy not in text, f'{caller} reads a version place itself: {copy!r}'
    assert len(bump.PLACES) == 7


def places_in(tmp_path: Path, monkeypatch, version: str = "1.2.3", capture=None) -> Path:
    for relative, _, _ in bump.PLACES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text((REPO / relative).read_text(encoding='utf-8'), encoding='utf-8')
    monkeypatch.setattr(bump, 'REPO', tmp_path)
    bump.write_places(version)
    if capture is not None:
        capture.readouterr()
    return tmp_path


def run_main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr('sys.argv', ['bump.py', *argv])
    try:
        return bump.main()
    except SystemExit as exc:
        return exc.code


def test_check_prints_the_one_version(tmp_path, monkeypatch, capsys):
    places_in(tmp_path, monkeypatch, capture=capsys)
    assert run_main(monkeypatch, '--check') == 0
    assert capsys.readouterr().out.strip() == '1.2.3'


def test_check_names_every_place_when_they_disagree(tmp_path, monkeypatch, capsys):
    root = places_in(tmp_path, monkeypatch, capture=capsys)
    version_ts = root / 'sdk/ts/src/version.ts'
    version_ts.write_text(version_ts.read_text(encoding='utf-8').replace('1.2.3', '1.2.2'),
                          encoding='utf-8')
    assert run_main(monkeypatch, '--check') == 1
    said = capsys.readouterr()
    assert said.out == ''
    for relative, _, _ in bump.PLACES:
        assert relative in said.err, said.err
    assert '1.2.2  sdk/ts/src/version.ts' in said.err, said.err
    assert 'python scripts/bump.py --align' in said.err, said.err


def test_align_sets_every_place_to_the_canonical_one(tmp_path, monkeypatch, capsys):
    root = places_in(tmp_path, monkeypatch)
    pyproject = root / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text(encoding='utf-8').replace('"1.2.3"', '"0.0.1"'),
                         encoding='utf-8')
    monkeypatch.setattr(bump, 'GENERATORS', [])
    monkeypatch.setattr(bump.subprocess, 'check_output', lambda *a, **k: '')
    assert run_main(monkeypatch, '--align') == 0
    assert bump.read_current() == '1.2.3'


def test_check_changes_nothing_and_takes_no_version(tmp_path, monkeypatch):
    root = places_in(tmp_path, monkeypatch)
    before = {relative: (root / relative).read_bytes() for relative, _, _ in bump.PLACES}
    assert run_main(monkeypatch, '--check', 'patch') == 2
    assert run_main(monkeypatch, '--check') == 0
    assert before == {relative: (root / relative).read_bytes() for relative, _, _ in bump.PLACES}


def test_a_bump_writes_lf_on_every_platform(tmp_path, monkeypatch):
    root = places_in(tmp_path, monkeypatch)
    for relative, _, _ in bump.PLACES:
        assert b"\r\n" not in (root / relative).read_bytes(), relative


@pytest.mark.parametrize('current,wanted,expected', [
    ('0.6.7', 'patch', '0.6.8'),
    ('0.6.7', 'minor', '0.7.0'),
    ('0.6.7', 'major', '1.0.0'),
    ('0.6.7', '0.9.1', '0.9.1'),
    ('0.9.9', 'patch', '0.9.10'),
])
def test_arithmetic(current, wanted, expected):
    assert bump.next_version(current, wanted) == expected


@pytest.mark.parametrize('wanted', ['0.6.7', '0.6.6', '0.5.0', 'latest', '1.0', 'v0.7.0'])
def test_a_version_never_goes_backwards_or_sideways(wanted):
    with pytest.raises(SystemExit):
        bump.next_version('0.6.7', wanted)


def test_a_measurement_is_not_a_version_place():
    prose = ['docs/history/PHASE14-ENVPACKS.md']
    bumped = {relative for relative, _, _ in bump.PLACES}
    for relative in prose:
        assert relative not in bumped
        text = (REPO / relative).read_text(encoding='utf-8')
        assert re.search(r'\d+\.\d+\.\d+', text), (
            f'{relative} no longer names any version; if the measurement moved, '
            'move this guard with it rather than deleting it')
