"""`write`: seams onto silence, and the refusal that stops an empty VTT shipping.

`snap_boundaries` is conservative by construction and each rule below is one of
those constructions — a snap that could move a seam further than its window, or
past a neighbour, would create the drift the whole stage exists to remove.

The last test is the one that matters most and is not about geometry at all: a
run that matched nothing must FAIL rather than write two lines that look like a
successful transcript.
"""

from __future__ import annotations

import pytest

from crucible.jobs.alignlongform import cues as C


def test_a_seam_moves_to_the_middle_of_a_nearby_silence() -> None:
    starts = [0.0, 10.0]
    ends = [10.0, 20.0]
    silences = [(9.5, 10.5)]
    ns, ne, stats = C.snap_boundaries(starts, ends, silences, window=1.0)
    assert ne[0] == pytest.approx(10.0)  # already the midpoint
    assert ns[1] == ne[0], "the seam is ONE time shared by two cues"
    assert stats.considered == 1


def test_an_off_centre_silence_pulls_the_seam_to_its_middle() -> None:
    ns, ne, stats = C.snap_boundaries(
        [0.0, 10.0], [10.0, 20.0], [(10.2, 10.8)], window=1.0
    )
    assert ne[0] == pytest.approx(10.5)
    assert ns[1] == pytest.approx(10.5)
    assert stats.snapped == 1


def test_a_long_silence_pulls_only_to_the_WINDOW_edge_not_its_own_centre() -> None:
    """A chapter gap is a huge silence; its centre may be a minute away.

    Clipping the candidate to the window is what stops a seam being dragged
    across a chapter — the snap can never move further than `window`.
    """
    ns, ne, _ = C.snap_boundaries(
        [0.0, 10.0], [10.0, 80.0], [(10.0, 70.0)], window=1.0
    )
    assert ne[0] <= 11.0 + 1e-9, "a snap moved a seam further than its own window"
    assert ne[0] == pytest.approx(10.5)  # midpoint of the CLIPPED overlap


def test_a_silence_outside_the_window_is_not_a_candidate() -> None:
    ns, ne, stats = C.snap_boundaries(
        [0.0, 10.0], [10.0, 20.0], [(50.0, 51.0)], window=1.0
    )
    assert (ns, ne) == ([0.0, 10.0], [10.0, 20.0])
    assert stats.snapped == 0


def test_cues_that_do_not_share_a_seam_are_left_alone() -> None:
    """A gap between cues is deliberate — a fallback retraction or a length cap."""
    ns, ne, stats = C.snap_boundaries(
        [0.0, 12.0], [10.0, 20.0], [(9.8, 10.4)], window=1.0
    )
    assert stats.considered == 0
    assert (ns, ne) == ([0.0, 12.0], [10.0, 20.0])


def test_no_silences_or_no_window_is_a_no_op() -> None:
    assert C.snap_boundaries([0.0], [1.0], [], window=1.0)[2].snapped == 0
    assert C.snap_boundaries([0.0], [1.0], [(0.4, 0.6)], window=0)[2].snapped == 0


def test_timestamps_are_vtt_shaped() -> None:
    assert C.timestamp(0) == "00:00:00.000"
    assert C.timestamp(61.5) == "00:01:01.500"
    assert C.timestamp(3661.25) == "01:01:01.250"


def test_the_vtt_orders_cues_by_time_and_numbers_them_from_one() -> None:
    text = C.write_vtt([
        C.Cue(10.0, 12.0, "second"),
        C.Cue(1.0, 2.0, "first"),
    ])
    lines = text.splitlines()
    assert lines[0] == "WEBVTT"
    assert "1" in lines and "00:00:01.000 --> 00:00:02.000" in lines
    assert lines.index("first") < lines.index("second")


def test_a_heading_keeps_the_note_the_extractor_earned() -> None:
    text = C.write_vtt([C.Cue(0.0, 1.0, "Chapter One", kind="heading")])
    assert "NOTE heading" in text


def test_an_empty_result_is_REFUSED_rather_than_written_as_a_bare_webvtt() -> None:
    """The one that is not about geometry.

    Two lines of `WEBVTT` is a run that aligned nothing and reported success,
    and nothing downstream can tell it from a book with no speech.
    """
    with pytest.raises(C.NoCues) as caught:
        C.write_vtt([])
    assert "not a transcript" in str(caught.value)
    assert "indistinguishable from success" in str(caught.value)
