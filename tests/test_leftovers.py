from __future__ import annotations

import os
import re
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import VERSION, protocol
from crucible.api.responses import VoiceInfo
from crucible.api.sse import TERMINAL_EVENTS
from crucible.cli import build_parser
from crucible.jobs.base import TERMINAL_STATES as JOB_TERMINAL_STATES
from crucible.jobs.tts.common import VOICE_ROW_FIELDS, voice_row
from crucible.tasks.states import TERMINAL_STATES as TASK_TERMINAL_STATES
from crucible.voicecard import CardError, export_manifest
from crucible.voicecatalog import load_voice
from crucible.voicerepo import REPO_MANIFEST_NAME, parse_repo_manifest
from crucible.voices import (
    MANIFEST_ENGINE,
    MANIFEST_OVERRIDE,
    MANIFEST_REPO,
    check_pace,
)

from .conftest import configure_box

REPO = Path(__file__).resolve().parent.parent
APP_JS = REPO / "crucible" / "ui" / "app.js"
SDK_TYPES = REPO / "sdk" / "ts" / "src" / "types.ts"


@pytest.fixture
def tts_client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client(enable_tts=True) as instance:
        yield instance


def percentile_mistborn():
    configure_box(Path(os.environ["CRUCIBLE_HOME"]))
    voice = load_voice("mistborn")
    return replace(voice, pace=replace(voice.pace, max_chars_per_sec=20.0))


def export(voice, edges: str | None = None) -> str:
    text, _dropped = export_manifest(
        voice,
        pace_basis="measured",
        measured_from="a percentile ladder",
        inherited_from=None,
        max_chars_basis="measured",
        uncertified=False,
        edges=edges,
    )
    return text


def test_an_asymmetric_band_names_the_flag_that_exports_it() -> None:
    with pytest.raises(CardError) as caught:
        export(percentile_mistborn())
    said = str(caught.value)
    assert "crucible voices export mistborn --edges percentile" in said
    assert "by hand" not in said


def test_edges_percentile_is_written_and_reads_back() -> None:
    text = export(percentile_mistborn(), edges="percentile")
    assert 'edges              = "percentile"' in text
    repo = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    assert repo.pace["edges"] == "percentile"
    check_pace("mistborn [voice.pace]", repo.pace)


def test_the_export_verb_takes_edges() -> None:
    args = build_parser().parse_args(
        ["voices", "export", "mistborn", "--pace-basis", "measured", "--edges", "percentile"]
    )
    assert args.edges == "percentile"


def test_no_refusal_in_the_package_says_by_hand() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "crucible").rglob("*.py")
        if "by hand" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_the_voice_row_has_exactly_voice_infos_fields() -> None:
    assert list(VOICE_ROW_FIELDS) == list(VoiceInfo.model_fields)


def test_a_voice_row_refuses_a_field_voice_info_does_not_have() -> None:
    with pytest.raises(KeyError):
        voice_row(id="x", colour="red")


def test_every_voices_row_validates_as_voice_info(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    rows = tts_client.get("/v1/voices", headers=auth).json()
    assert rows
    for row in rows:
        assert set(row) == set(VoiceInfo.model_fields)


def test_info_carries_the_terminal_states(client: TestClient, auth: dict[str, str]) -> None:
    body = client.get("/v1/info", headers=auth).json()
    assert body["terminal_states"] == {
        "jobs": sorted(JOB_TERMINAL_STATES),
        "tasks": sorted(TASK_TERMINAL_STATES),
    }


def test_info_names_every_voice_source(client: TestClient, auth: dict[str, str]) -> None:
    sources = client.get("/v1/info", headers=auth).json()["voice_sources"]
    assert set(sources) == {MANIFEST_REPO, MANIFEST_OVERRIDE, MANIFEST_ENGINE}
    assert all(set(entry) == {"label", "tone"} for entry in sources.values())


def test_every_service_command_parses_with_the_real_cli(
    client: TestClient, auth: dict[str, str]
) -> None:
    commands = client.get("/v1/info", headers=auth).json()["service_commands"]
    assert commands
    for entry in commands:
        argv = shlex.split(entry["command"])
        assert argv[0] == "crucible"
        build_parser().parse_args(argv[1:])


def test_the_console_keeps_no_copy_of_the_info_tables() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    for name in ("TERMINAL", "VOICE_SOURCE", "SERVICE_COMMANDS"):
        assert not re.search(r"\bvar " + name + r"\b", script), name
    for field in ("terminal_states", "voice_sources", "service_commands"):
        assert "state.info." + field in script, field


def test_the_sdk_spells_the_same_terminal_states() -> None:
    types = SDK_TYPES.read_text(encoding="utf-8")

    def spelled(pattern: str) -> set[str]:
        found = re.search(pattern, types)
        assert found, pattern
        return set(re.findall(r"'([a-z]+)'", found.group(1)))

    assert spelled(r"TASK_TERMINAL_STATES = \[([^\]]*)\]") == set(TASK_TERMINAL_STATES)
    assert spelled(r"TERMINAL_EVENTS = \[([^\]]*)\]") == set(TERMINAL_EVENTS)
    job_states = spelled(r"export type JobState = ([^;]*);")
    assert set(JOB_TERMINAL_STATES) <= job_states


def test_the_user_agent_takes_its_version_from_the_caller() -> None:
    assert protocol.user_agent("cli", VERSION) == f"crucible-cli/{VERSION}"
    assert "import" not in Path(protocol.__file__).read_text(encoding="utf-8").split(
        "def user_agent", 1
    )[1].split("def ", 1)[0]
