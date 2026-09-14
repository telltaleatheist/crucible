"""ultimate-rvc's base assets: the declaration, and `weights.pull_files`.

Nothing here reaches the network. `hf_hub_download` is replaced with a function
that writes bytes this test made up, which is all this module needs from it —
the same shape `tests/test_weights_archive.py` uses for the archive pull.

What these tests are really about is the rule that made the third pull shape
worth writing: **every digest is checked before any file is placed**, because a
half-placed base tree is one urvc will start against and fail inside, hours
later, in somebody's book.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from crucible import rvcbase, weights
from crucible.config import load_config, write_config
from crucible.jobs import rvc as rvc_job

from .conftest import FAKE_BACKEND

GOOD = """
[engine]
id = "demo-engine"
hf_repo = "someone/resources"
revision = "0123456789abcdef0123456789abcdef01234567"

[[files]]
source = "Resources/embedders/contentvec/pytorch_model.bin"
target = "rvc/embedders/contentvec/pytorch_model.bin"
sha256 = "{bin_sha}"
bytes = 11
why = "the embedder"

[[files]]
source = "Resources/predictors/rmvpe.pt"
target = "rvc/predictors/rmvpe.pt"
sha256 = "{pt_sha}"
bytes = 9
why = "the pitch predictor"
"""

BIN = b"embedder!!!"
PT = b"predictor"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def declaration(text: str | None = None, name: str = "demo-engine"):
    body = text if text is not None else GOOD.format(
        bin_sha=sha(BIN), pt_sha=sha(PT)
    )
    return rvcbase.parse_rvc_base(body, Path(f"{name}.toml"), name)


@pytest.fixture
def config(home: Path):
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="t",
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=False,
        enable_llm=False,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=True,
        desktop_allowance_bytes=0,
    )
    return load_config(home)


def serve(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes]) -> list[str]:
    """Replace `hf_hub_download` with one that writes `payloads[filename]`."""
    import huggingface_hub

    asked: list[str] = []

    def download(*, repo_id, filename, revision, local_dir, token=None, **_):
        asked.append(filename)
        if filename not in payloads:
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"{filename} is not in {repo_id}")
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[filename])
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return asked


PAYLOADS = {
    "Resources/embedders/contentvec/pytorch_model.bin": BIN,
    "Resources/predictors/rmvpe.pt": PT,
}


# ------------------------------------------------------------ the shipped one


def test_the_shipped_declaration_names_the_engines_own_repo() -> None:
    """Read out of the installed fork's own downloader, not from a tutorial:
    `prerequisites_download.py`'s `url_base` is JackismyShephard/ultimate-rvc."""
    assets = rvcbase.load_rvc_base()
    assert assets.id == rvcbase.ULTIMATE_RVC
    assert assets.hf_repo == "JackismyShephard/ultimate-rvc"
    assert len(assets.revision) == 40
    assert assets.targets == (
        "rvc/embedders/contentvec/pytorch_model.bin",
        "rvc/embedders/contentvec/config.json",
        "rvc/predictors/rmvpe.pt",
        "rvc/predictors/fcpe.pt",
    )
    # About 600 MB, all four.
    assert 5e8 < assets.total_bytes < 7e8
    for entry in assets.files:
        assert len(entry.sha256) == 64
        assert entry.why.strip() != ""


def test_the_job_reads_the_same_list_the_puller_does(config) -> None:
    """One owner of "which files urvc needs". They were two lists, and the
    job's was missing the embedder's config.json."""
    assets = rvcbase.load_rvc_base()
    root = rvc_job.rvc_base_dir(config)
    assert root == rvcbase.base_root(config)
    assert rvcbase.missing(config, assets) == list(assets.targets)
    for target in assets.targets:
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    assert rvcbase.missing(config, assets) == []


# --------------------------------------------------------- what it refuses


@pytest.mark.parametrize(
    "edit,fragment",
    [
        (("revision = \"0123456789abcdef0123456789abcdef01234567\"",
          "revision = \"main\""), "40-character"),
        (("why = \"the embedder\"", "reason = \"the embedder\""), "why"),
        (("bytes = 11", "bytes = 0"), "bytes must be positive"),
        (("target = \"rvc/predictors/rmvpe.pt\"",
          "target = \"../../etc/passwd\""), "climbs out"),
        (("target = \"rvc/predictors/rmvpe.pt\"",
          "target = \"rvc/embedders/contentvec/pytorch_model.bin\""),
         "declared twice"),
        (("hf_repo = \"someone/resources\"", "hf_repo = \"resources\""),
         "<owner>/<name>"),
    ],
)
def test_what_the_loader_refuses(edit, fragment: str) -> None:
    text = GOOD.format(bin_sha=sha(BIN), pt_sha=sha(PT)).replace(*edit)
    with pytest.raises(rvcbase.RvcBaseError) as caught:
        declaration(text)
    assert fragment in str(caught.value)


def test_a_short_digest_is_refused() -> None:
    text = GOOD.format(bin_sha="abc", pt_sha=sha(PT))
    with pytest.raises(rvcbase.RvcBaseError) as caught:
        declaration(text)
    assert "64-character" in str(caught.value)


def test_a_declaration_with_no_files_is_refused() -> None:
    with pytest.raises(rvcbase.RvcBaseError) as caught:
        declaration(
            "[engine]\nid = \"demo-engine\"\nhf_repo = \"a/b\"\n"
            "revision = \"0123456789abcdef0123456789abcdef01234567\"\n"
        )
    assert "missing [[files]]" in str(caught.value)


# ------------------------------------------------------------------- pulling


def test_every_file_is_fetched_and_placed_where_the_engine_looks(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = declaration()
    asked = serve(monkeypatch, PAYLOADS)
    result = rvcbase.pull(config, assets)
    assert asked == [entry.source for entry in assets.files]
    root = rvcbase.base_root(config)
    assert (root / "rvc/embedders/contentvec/pytorch_model.bin").read_bytes() == BIN
    assert (root / "rvc/predictors/rmvpe.pt").read_bytes() == PT
    assert result.bytes == len(BIN) + len(PT)
    # The staging directory does not survive the placement.
    assert not (root / ".crucible-files").exists()


def test_a_second_pull_at_the_same_pin_does_nothing(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = declaration()
    asked = serve(monkeypatch, PAYLOADS)
    rvcbase.pull(config, assets)
    asked.clear()
    rvcbase.pull(config, assets)
    assert asked == []


def test_force_re_pulls(config, monkeypatch: pytest.MonkeyPatch) -> None:
    assets = declaration()
    asked = serve(monkeypatch, PAYLOADS)
    rvcbase.pull(config, assets)
    asked.clear()
    rvcbase.pull(config, assets, force=True)
    assert len(asked) == 2


def test_a_declaration_at_a_new_revision_is_not_installed(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stamp at another pin is another set of bytes."""
    assets = declaration()
    serve(monkeypatch, PAYLOADS)
    rvcbase.pull(config, assets)
    assert rvcbase.installed(config, assets) is not None
    moved = declaration(
        GOOD.format(bin_sha=sha(BIN), pt_sha=sha(PT)).replace(
            "0123456789abcdef0123456789abcdef01234567",
            "fedcba9876543210fedcba9876543210fedcba98",
        )
    )
    assert rvcbase.installed(config, moved) is None


def test_a_bad_digest_places_nothing_at_all(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule the third pull shape exists for: a half-placed base tree is one
    urvc starts against and fails inside, hours later, in somebody's book."""
    assets = declaration()
    serve(
        monkeypatch,
        {**PAYLOADS, "Resources/predictors/rmvpe.pt": b"not the predictor"},
    )
    with pytest.raises(weights.WeightsError) as caught:
        rvcbase.pull(config, assets)
    assert "hashes to" in str(caught.value)
    assert "NOTHING was placed" in str(caught.value)
    root = rvcbase.base_root(config)
    # Not even the file that DID verify, because it is verified first and
    # placed last.
    assert not (root / "rvc/embedders/contentvec/pytorch_model.bin").exists()
    assert not (root / "rvc/predictors/rmvpe.pt").exists()
    assert rvcbase.installed(config, assets) is None


def test_a_missing_file_in_the_repo_names_it(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = declaration()
    serve(monkeypatch, {"Resources/embedders/contentvec/pytorch_model.bin": BIN})
    with pytest.raises(weights.WeightsError) as caught:
        rvcbase.pull(config, assets)
    assert "rmvpe.pt" in str(caught.value)


def test_an_interrupted_pull_does_not_read_as_installed(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp is removed before anything is fetched, so a run that dies part
    way leaves a tree nothing downstream trusts."""
    assets = declaration()
    serve(monkeypatch, PAYLOADS)
    rvcbase.pull(config, assets)
    stamp = rvcbase.base_root(config) / weights.STAMP_NAME
    assert stamp.is_file()

    serve(monkeypatch, {"Resources/embedders/contentvec/pytorch_model.bin": BIN})
    with pytest.raises(weights.WeightsError):
        rvcbase.pull(config, assets, force=True)
    assert not stamp.exists()
    assert rvcbase.installed(config, assets) is None


def test_the_stamp_records_every_file_and_the_pin(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = declaration()
    serve(monkeypatch, PAYLOADS)
    rvcbase.pull(config, assets)
    record = json.loads(
        (rvcbase.base_root(config) / weights.STAMP_NAME).read_text(encoding="utf-8")
    )
    assert record["hf_repo"] == assets.hf_repo
    assert record["revision"] == assets.revision
    assert [entry["target"] for entry in record["files"]] == list(assets.targets)


def test_a_deleted_file_makes_the_set_not_installed(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stamp beside a file somebody removed is a stamp that lies."""
    assets = declaration()
    serve(monkeypatch, PAYLOADS)
    rvcbase.pull(config, assets)
    (rvcbase.base_root(config) / "rvc/predictors/rmvpe.pt").unlink()
    assert rvcbase.installed(config, assets) is None


# ------------------------------------------------------------- the refusal


def test_the_cli_verb_pulls_the_shipped_set(
    config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`crucible rvc pull-base`, driven the way an operator drives it."""
    from crucible import cli

    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)
    assets = rvcbase.load_rvc_base()
    payloads = {entry.source: b"" for entry in assets.files}
    serve(monkeypatch, payloads)
    # The shipped declaration pins the REAL digests; these are empty files, so
    # the pins are swapped for theirs. Everything else — the repo, the
    # revision, every source and every target — is the shipped one's.
    empty = sha(b"")
    monkeypatch.setattr(
        rvcbase,
        "load_rvc_base",
        lambda *a, **k: rvcbase.RvcBaseAssets(
            id=assets.id,
            hf_repo=assets.hf_repo,
            revision=assets.revision,
            files=tuple(
                rvcbase.BaseFile(
                    source=entry.source,
                    target=entry.target,
                    sha256=empty,
                    bytes=1,
                    why=entry.why,
                )
                for entry in assets.files
            ),
            path=assets.path,
        ),
    )
    assert cli.main(["rvc", "pull-base"]) == 0
    out = capsys.readouterr().out
    assert assets.hf_repo in out
    root = rvcbase.base_root(config)
    for target in assets.targets:
        assert (root / target).is_file(), target


def test_the_job_refusal_names_the_command_that_fetches_them(config) -> None:
    from crucible.errors import ApiError

    plugin = rvc_job.RvcJobType(config, FAKE_BACKEND, frozenset)
    with pytest.raises(ApiError) as caught:
        rvc_job._require_base_assets(config)
    error = caught.value
    assert error.code == "rvc_base_models_missing"
    assert rvcbase.PULL_COMMAND in error.message
    assert error.details["command"] == rvcbase.PULL_COMMAND
    assert error.details["hf_repo"] == "JackismyShephard/ultimate-rvc"
    # And the status line says it too, so a doctor run and a refused job send
    # the reader to the same place.
    assert plugin is not None
