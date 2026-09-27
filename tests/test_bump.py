import importlib.util
from pathlib import Path
import re

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


def test_the_places_are_the_ones_release_sh_refuses_over():
    text = (REPO / 'scripts/release.sh').read_text(encoding='utf-8')
    for relative, _, _ in bump.PLACES:
        assert relative in text, f'{relative} is bumped but release.sh never checks it'
    assert len(bump.PLACES) == 7


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
    prose = ['docs/PHASE14-ENVPACKS.md']
    bumped = {relative for relative, _, _ in bump.PLACES}
    for relative in prose:
        assert relative not in bumped
        text = (REPO / relative).read_text(encoding='utf-8')
        assert re.search(r'\d+\.\d+\.\d+', text), (
            f'{relative} no longer names any version; if the measurement moved, '
            'move this guard with it rather than deleting it')
