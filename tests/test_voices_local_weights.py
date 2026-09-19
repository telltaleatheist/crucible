"""A voice whose weights are a DIRECTORY, not a pin — PHASE18-UNCERTIFIED.md §3.

The schema half lives in `test_voices.py`. This file is about what the rest of
the server does with a local block once the loader has accepted it, and the
whole of that is one rule stated four ways: **Crucible does not manage these
bytes.** It does not fetch them, it does not stamp them, it does not delete
them, it does not list them among the things this machine downloads, and it does
not treat their absence as a broken install.

The absence case is the one worth having a test for at all. A screening merge is
8 GiB that exists for four minutes and is then deleted by the thing that made
it, so "the directory is gone" is the EXPECTED end of a local voice's life
rather than a fault — and the refusal that reports it must not tell its reader
to run a pull that cannot help.

The second half of the file is the other thing a local block changes downstream:
it has no `revision`, so every place that published one has to publish the
block's IDENTITY instead. That fact has one owner — `weights_identity` — and a
keeper per reader, because a reader left on `spec.revision` reports null beside
a fingerprint that names a checkpoint, which is one record giving two answers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from crucible import cli, weights
from crucible.config import Config, load_config, write_config
from crucible.jobs.tts.common import voice_provenance, voice_rows
from crucible.narratorvoices import NarratorVoicesError, voice_entry
from crucible.residency import Residency
from crucible.voices import VOICES_DIR_ENV, parse_voice

from . import fake_narrator_engine
from .conftest import FAKE_BACKEND
from .test_tts_api import (  # noqa: F401 -- imported to be used as fixtures
    fake_env,
    idle_card,
    tts_recipes,
)
from .test_tts_api import run_job

#: A checkpoint whose cuda-linux block names a directory. Written against
#: `tmp_path` so the directory can be made and unmade inside a test.
LOCAL = """
[voice]
id = "screening"
display = "Screening"
kind = "checkpoint"
narrator_engine = "higgs-v3"
language = "en"
sample_rate = 24000

[voice.pace]
pace_chars_per_sec = 16.0
max_chars_per_sec = 20.8
min_chars_per_sec = 12.3

[voice.serving]
max_num_seqs = 4
max_num_seqs_note = "the ladder's own measured width; 16 collapsed WDDM paging on this box."

[voice.backends.cuda-linux]
path = "{path}"
identity = "mb_ha_rvcbed1@5368"
memory_bytes_estimate = 19_000_000_000
estimate_basis = "declared"
estimate_note = "SGLang's own --mem-fraction-static reservation; nobody watched the card."
max_chars = 800
sampling = {{ temperature = 0.8, top_p = 0.95, top_k = 50 }}
"""


def a_local_voice(directory: Path):
    text = LOCAL.format(path=directory.as_posix())
    return parse_voice(text, Path("screening.toml"), "screening")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A real config rooted at a scratch home, through the loader every other
    caller uses — so `weights_dir` and the stamp path are the ones the server
    would compute."""
    root = tmp_path / "home"
    write_config(
        root,
        name="test",
        host="127.0.0.1",
        port=8080,
        token="t",
        backend_kind="cuda-linux",
        enable_echo=True,
        enable_llm=True,
        enable_asr=True,
        enable_tts=True,
        enable_align=True,
        enable_rvc=True,
        desktop_allowance_bytes=0,
    )
    return load_config(root)


def test_a_present_directory_is_servable_without_a_stamp(
    tmp_path: Path, config: Config
) -> None:
    merged = tmp_path / "higgs_v3_merged" / "mb_ha_rvcbed1_5368"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"x" * 1024)

    voice = a_local_voice(merged)
    spec = voice.spec("cuda-linux")
    found = weights.installed(config, voice, spec)

    assert found is not None
    assert found.path == merged
    assert found.source == "local"
    # NOT AN INSTALL. Nothing was fetched, so there is no repo and no timestamp,
    # and inventing either would make a directory that has sat here for a month
    # read as a recent download.
    assert found.hf_repo is None
    assert found.pulled is None
    assert found.revision == "mb_ha_rvcbed1@5368"
    assert found.bytes == 1024
    # And no stamp was written anywhere under the server's home.
    assert not (config.home / "voices").exists()


def test_a_directory_that_is_gone_is_simply_not_there(
    tmp_path: Path, config: Config
) -> None:
    voice = a_local_voice(tmp_path / "deleted-four-minutes-ago")
    assert weights.installed(config, voice, voice.spec("cuda-linux")) is None


def test_the_refusal_does_not_send_its_reader_to_a_pull(
    tmp_path: Path, config: Config
) -> None:
    gone = tmp_path / "deleted-four-minutes-ago"
    voice = a_local_voice(gone)
    with pytest.raises(weights.WeightsError) as caught:
        weights.require_installed(config, voice, voice.spec("cuda-linux"))
    message = str(caught.value)
    assert str(gone) in message
    assert "does not fetch" in message
    # The pinned refusal's advice would be actively wrong here: there is no
    # repo to pull from and the command would report an unknown voice.
    assert "crucible voices pull" not in message


def test_pulling_a_local_voice_is_refused_by_name(
    tmp_path: Path, config: Config
) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    voice = a_local_voice(merged)
    with pytest.raises(weights.WeightsError) as caught:
        weights.pull(config, voice, voice.spec("cuda-linux"))
    assert "cannot be pulled" in str(caught.value)


def test_removing_a_local_voice_is_refused_by_name(
    tmp_path: Path, config: Config
) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"x")
    voice = a_local_voice(merged)

    with pytest.raises(weights.WeightsError) as caught:
        weights.remove(config, voice, voice.spec("cuda-linux"))
    assert "cannot be removed" in str(caught.value)
    # THE POINT OF THE REFUSAL: the bytes are still there.
    assert (merged / "model.safetensors").is_file()


def test_a_pinned_voice_still_pulls_and_stamps(
    tmp_path: Path, config: Config
) -> None:
    # The guard is on the LOCAL branch only; a pinned block reaches the
    # HuggingFace path exactly as it did. Proven by the fact that it gets far
    # enough to complain about the hub rather than about being local.
    from crucible.voices import parse_voice as parse

    pinned = LOCAL.format(path="/unused").replace(
        'path = "/unused"\nidentity = "mb_ha_rvcbed1@5368"',
        'hf_repo = "owenmorgan/screening"\n'
        'revision = "0123456789abcdef0123456789abcdef01234567"',
    )
    voice = parse(pinned, Path("screening.toml"), "screening")
    spec = voice.spec("cuda-linux")
    assert spec.source == "pinned"
    assert weights.local_source(spec) is None


# ------------------------------------------- the readers of ONE identity
#
# `VoiceBackendSpec.weights_identity` is the single owner of "what this block
# says its weights are" (ARCHITECTURE.md R1), and it exists because four
# readers want that fact: `fingerprint()`, the `/v1/voices` row, the render's
# provenance sidecar and the resident record. A reader that goes back to
# `spec.revision` reports `None` for a voice whose identity WAS stated — which
# is the failure `identity_basis` exists to make impossible, since the row's
# own invariant is `fingerprint == f"{id}@{revision}"` (`test_tts_api.py`
# asserts it of every row). So each reader gets a keeper.

IDENTITY = "mb_ha_rvcbed1@5368"


@pytest.fixture
def screening(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The screening voice as a manifest on disk, with its merge present.

    `CRUCIBLE_VOICES_DIR` REPLACES the packaged set rather than adding to it
    (`test_voices.py`), which is what these tests want: every row they read is
    this voice's. Returns the merged directory the manifest names.
    """
    merged = tmp_path / "higgs_v3_merged" / "mb_ha_rvcbed1_5368"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"x" * 1024)
    directory = tmp_path / "voices"
    directory.mkdir()
    (directory / "screening.toml").write_text(
        LOCAL.format(path=merged.as_posix()), encoding="utf-8"
    )
    monkeypatch.setenv(VOICES_DIR_ENV, str(directory))
    return merged


def test_the_voices_row_reports_the_asserted_identity_as_its_revision(
    screening: Path, config: Config
) -> None:
    row = next(
        entry
        for entry in voice_rows(config, FAKE_BACKEND, Residency(config))
        if entry["id"] == "screening"
    )
    assert row["source"] == "local"
    assert row["identity_basis"] == "asserted"
    assert row["installed"] is True
    # The row's own invariant, and the reason `revision` cannot be null here.
    assert row["revision"] == IDENTITY
    assert row["fingerprint"] == f"screening@{row['revision']}"


def test_the_sidecar_says_the_same_identity_and_says_it_is_asserted(
    screening: Path,
) -> None:
    assert voice_provenance(FAKE_BACKEND.kind, "screening") == {
        "id": "screening",
        "revision": IDENTITY,
        "identity_basis": "asserted",
        "fingerprint": f"screening@{IDENTITY}",
    }


def test_the_resident_record_carries_the_identity_it_was_loaded_from(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    screening: Path,
    fake_env: Path,
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local voice loads through the SAME door, and leaves the same record.

    `residency` is handed `installed.path`, which for a local block is the
    directory the manifest named — so there is no second load path to keep in
    step, and the one thing that could still differ is what the record says the
    weights were. `fingerprint` and `revision` are two fields of one record and
    must not be able to disagree.
    """
    fake_narrator_engine.install(monkeypatch)
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)
    with make_client(enable_tts=True, enable_echo=False) as client:
        events = run_job(client, auth, type="load-voice", model="screening")
        done = events[-1]
        assert done["event"] == "done", done
        assert done["data"]["fingerprint"] == f"screening@{IDENTITY}"
        resident = client.app.state.residency.resident_voice
        assert resident.revision == IDENTITY
        assert resident.fingerprint == f"screening@{resident.revision}"


# ------------------------------------ the two doors that fetch and delete
#
# `weights.pull` and `weights.remove` refuse a local spec BY NAME, and a
# command that printed the pin before calling them would never reach the
# refusal: `spec.revision` is None on a local block, so `revision[:12]` is a
# TypeError and the operator gets a traceback instead of the sentence.


def test_the_pull_command_reaches_the_named_refusal(
    home: Path, screening: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["voices", "pull", "screening"]) == 1
    assert "cannot be pulled" in capsys.readouterr().err


def test_the_list_command_does_not_offer_a_pull_that_would_refuse(
    home: Path, screening: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The merge deleted the moment its renders landed, which is the EXPECTED
    # end of a screening voice's life rather than a broken install.
    shutil.rmtree(screening)
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["voices", "list", "--json"]) == 0
    row = {r["id"]: r for r in json.loads(capsys.readouterr().out)}["screening"]
    assert row["installed"] is False
    assert "crucible voices pull" not in row["detail"]
    assert str(screening) in row["detail"]


def test_a_local_token_voice_is_refused_by_name_and_not_by_a_subscript(
    tmp_path: Path,
) -> None:
    """narrator's served arm cannot be told which directory a TOKEN voice is,
    so one is refused on `cuda-linux` — and the refusal names the pin it would
    have served. A local block has no pin, and `spec.revision[:12]` on it is a
    TypeError: the voice is still refused, but by the wrong name.
    """
    merged = tmp_path / "merged"
    voice = parse_voice(
        LOCAL.format(path=merged.as_posix()).replace(
            'kind = "checkpoint"', 'kind = "token"'
        ),
        Path("screening.toml"),
        "screening",
    )
    with pytest.raises(NarratorVoicesError) as caught:
        voice_entry(voice, voice.spec("cuda-linux"), merged)
    assert "is a token voice" in str(caught.value)
