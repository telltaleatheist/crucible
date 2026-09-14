"""Denoise manifests and `crucible denoise pull`.

Strict for `crucible/asrmodels.py`'s reason. A separator run with the wrong
checkpoint produces audio that sounds nearly right, and the manifest is the only
place that says which weights made it.

Nothing here reaches the network: `hf_hub_download` is replaced with a function
that writes bytes this test made up, which is `tests/test_rvcbase.py`'s fake hub
and is all these need from the hub. What the second half is really about is the
thing that makes the command worth having — **the puller and the job read the
same function for the layout**, so the two files land under exactly the names
audio-separator resolves by and a doctor run says so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from crucible import denoisemodels, weights
from crucible.config import load_config, write_config
from crucible.denoisemodels import (
    DenoiseManifestError,
    load_all_denoise_manifests,
    load_denoise_manifest,
    parse_denoise_manifest,
)
from crucible.jobs import denoise as denoise_job

from .conftest import FAKE_BACKEND

SHA = "7c1c39191edc34e942ca7f2346ce6b6c0e1208a5f76349ffce6f696bd12910de"
OTHER_SHA = "5d7d83b2e9d232da60941b717b0abdc345155d45cff3f79715cdb2790ba18c36"

GOOD = f"""
[model]
id = "demo-denoise"
display = "Demo Denoiser"
model_filename = "demo_denoise.ckpt"
config_filename = "demo_denoise_config.yaml"
primary_stem = "dry"
sample_rate = 44100

[backends.cuda-linux]
engine = "audio-separator"
hf_repo = "someone/Resources"
revision = "0123456789abcdef0123456789abcdef01234567"
model_path = "models/demo_denoise.ckpt"
model_sha256 = "{SHA}"
model_bytes = 913097300
config_path = "models/config_demo.yaml"
config_sha256 = "{OTHER_SHA}"
config_bytes = 1621
memory_bytes_estimate = 2523710036
"""


def parse(text: str, name: str = "demo-denoise"):
    return parse_denoise_manifest(text, Path(f"{name}.toml"), name)


def test_a_complete_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.id == "demo-denoise"
    assert manifest.primary_stem == "dry"
    assert manifest.sample_rate == 44100
    spec = manifest.spec("cuda-linux")
    assert spec.model_sha256 == SHA
    assert spec.model_bytes == 913097300
    assert spec.config_bytes == 1621
    assert spec.total_bytes == 913097300 + 1621
    assert manifest.supports("cuda-linux")
    assert not manifest.supports("mlx-darwin")


def test_the_shipped_manifest_names_both_filenames_and_a_real_pin() -> None:
    """The names audio-separator resolves by are NOT the paths they come from,
    and the manifest carries both halves."""
    manifest = load_denoise_manifest("denoise-roformer")
    assert manifest.model_filename.endswith(".ckpt")
    assert manifest.config_filename.endswith("_config.yaml")
    for kind in ("cuda-linux", "mlx-darwin"):
        spec = manifest.spec(kind)
        assert spec.model_path != manifest.model_filename
        assert spec.config_path.rsplit("/", 1)[-1] != manifest.config_filename
        assert len(spec.revision) == 40
        assert len(spec.model_sha256) == 64
        assert len(spec.config_sha256) == 64
        # Both sizes are the HuggingFace API's, read at the pinned revision on
        # 2026-09-13: 913,097,300 for the checkpoint and 1,621 for the YAML.
        assert spec.model_bytes == 913_097_300
        assert spec.config_bytes == 1_621


def test_every_shipped_manifest_loads() -> None:
    manifests = load_all_denoise_manifests()
    assert "denoise-roformer" in manifests


@pytest.mark.parametrize(
    "bad,fragment",
    [
        (GOOD.replace("primary_stem", "primary_stemm"), "primary_stemm"),
        (GOOD.replace('revision = "0123456789abcdef0123456789abcdef01234567"',
                      'revision = "main"'), "40-character"),
        (GOOD.replace(f'model_sha256 = "{SHA}"', 'model_sha256 = "abc"'),
         "sha256"),
        (GOOD.replace("model_bytes = 913097300", "model_bytes = 0"),
         "model_bytes"),
        (GOOD.replace("config_bytes = 1621", "config_bytes = 0"),
         "config_bytes"),
        (GOOD.replace("sample_rate = 44100", "sample_rate = 0"), "sample_rate"),
        (GOOD.replace('engine = "audio-separator"', 'engine = "vllm"'),
         "does not denoise"),
        (GOOD.replace("[backends.cuda-linux]", "[backends.windows]"),
         "not a denoise backend"),
    ],
)
def test_what_the_loader_refuses(bad: str, fragment: str) -> None:
    with pytest.raises(DenoiseManifestError) as caught:
        parse(bad)
    assert fragment in str(caught.value)


@pytest.mark.parametrize(
    "value",
    ["../../etc/passwd", "sub/dir/model.ckpt", ".hidden.ckpt"],
)
def test_a_filename_that_is_a_path_is_refused(value: str) -> None:
    """These are written into one flat directory an engine then reads BY NAME."""
    with pytest.raises(DenoiseManifestError) as caught:
        parse(GOOD.replace('model_filename = "demo_denoise.ckpt"',
                           f'model_filename = "{value}"'))
    assert "plain filename" in str(caught.value)


def test_an_id_that_disagrees_with_its_filename_is_refused() -> None:
    with pytest.raises(DenoiseManifestError) as caught:
        parse(GOOD, name="something-else")
    assert "the same thing" in str(caught.value)


# ------------------------------------------------------------------- pulling

CKPT = b"not really 913 MB"
YAML = b"chunk_size: 352800"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


#: The shipped manifest with this test's digests in place of the real ones, so
#: the pull can be driven end to end against files a test can write. Everything
#: else — the repo, the revision, both source paths, both target filenames — is
#: the shipped manifest's, because those are what the command has to get right.
def shipped_with_test_digests(model_id: str = "denoise-roformer"):
    manifest = load_denoise_manifest(model_id)
    backends = {
        kind: denoisemodels.DenoiseBackendSpec(
            backend=spec.backend,
            engine=spec.engine,
            hf_repo=spec.hf_repo,
            revision=spec.revision,
            model_path=spec.model_path,
            model_sha256=sha(CKPT),
            model_bytes=len(CKPT),
            config_path=spec.config_path,
            config_sha256=sha(YAML),
            config_bytes=len(YAML),
            memory_bytes_estimate=spec.memory_bytes_estimate,
        )
        for kind, spec in manifest.backends.items()
    }
    return denoisemodels.DenoiseManifest(
        id=manifest.id,
        display=manifest.display,
        model_filename=manifest.model_filename,
        config_filename=manifest.config_filename,
        primary_stem=manifest.primary_stem,
        sample_rate=manifest.sample_rate,
        backends=backends,
        path=manifest.path,
    )


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
        enable_rvc=False,
        enable_denoise=True,
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


def payloads_for(manifest, spec) -> dict[str, bytes]:
    return {spec.model_path: CKPT, spec.config_path: YAML}


def test_the_puller_and_the_job_read_the_same_layout(config) -> None:
    """One owner of "where the separator's files live". They were two
    definitions, and two definitions are two that can disagree."""
    assert denoisemodels.denoise_models_root(config.home) == (
        denoise_job.denoise_models_dir(config)
    )
    assert denoise_job.denoise_models_dir_for(config.home).name == "denoise-models"


def test_the_file_list_is_the_manifest_s_two_names() -> None:
    manifest = load_denoise_manifest("denoise-roformer")
    spec = manifest.spec(FAKE_BACKEND.kind)
    files = denoisemodels.model_files(manifest, spec)
    assert [entry.source for entry in files] == [spec.model_path, spec.config_path]
    # What they are CALLED when they land is not what they are called upstream.
    assert [entry.target for entry in files] == [
        manifest.model_filename,
        manifest.config_filename,
    ]
    # The config's upstream name is not the name it lands under, which is the
    # whole reason both halves are declared.
    assert files[1].target != files[1].source.rsplit("/", 1)[-1]
    for entry in files:
        assert entry.why.strip() != ""


def test_a_pull_places_both_files_where_the_job_looks(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    asked = serve(monkeypatch, payloads_for(manifest, spec))
    assert denoisemodels.missing(config.home, manifest) == [
        manifest.model_filename,
        manifest.config_filename,
    ]

    result = denoisemodels.pull(config, manifest, spec)

    assert asked == [spec.model_path, spec.config_path]
    root = denoisemodels.denoise_models_root(config.home)
    assert (root / manifest.model_filename).read_bytes() == CKPT
    assert (root / manifest.config_filename).read_bytes() == YAML
    assert result.bytes == len(CKPT) + len(YAML)
    assert denoisemodels.missing(config.home, manifest) == []
    # The staging directory does not survive the placement.
    assert not (root / ".crucible-files").exists()


def test_a_second_pull_at_the_same_pin_does_nothing(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    asked = serve(monkeypatch, payloads_for(manifest, spec))
    denoisemodels.pull(config, manifest, spec)
    asked.clear()
    denoisemodels.pull(config, manifest, spec)
    assert asked == []


def test_force_re_pulls(config, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    asked = serve(monkeypatch, payloads_for(manifest, spec))
    denoisemodels.pull(config, manifest, spec)
    asked.clear()
    denoisemodels.pull(config, manifest, spec, force=True)
    assert asked == [spec.model_path, spec.config_path]


def test_a_bad_digest_places_nothing_at_all(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint beside somebody else's config is a separator that loads and
    produces audio which is subtly wrong and says so nowhere."""
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, {spec.model_path: CKPT, spec.config_path: b"another config"})
    with pytest.raises(weights.WeightsError) as caught:
        denoisemodels.pull(config, manifest, spec)
    assert "hashes to" in str(caught.value)
    assert "NOTHING was placed" in str(caught.value)
    root = denoisemodels.denoise_models_root(config.home)
    # Not even the checkpoint, which DID verify: verified first, placed last.
    assert not (root / manifest.model_filename).exists()
    assert denoisemodels.installed(config.home, manifest, spec) is None


def test_each_model_gets_its_own_stamp(config, monkeypatch: pytest.MonkeyPatch) -> None:
    """One flat directory, one stamp per set. A single `crucible-pull.json` at
    the root would be overwritten by the second model's pull and would then
    report the first as never installed."""
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, payloads_for(manifest, spec))
    denoisemodels.pull(config, manifest, spec)
    root = denoisemodels.denoise_models_root(config.home)
    assert not (root / weights.STAMP_NAME).exists()
    stamp = root / f"crucible-pull-{manifest.id}.json"
    record = json.loads(stamp.read_text(encoding="utf-8"))
    assert record["hf_repo"] == spec.hf_repo
    assert record["revision"] == spec.revision
    assert [entry["target"] for entry in record["files"]] == [
        manifest.model_filename,
        manifest.config_filename,
    ]


def test_a_deleted_file_makes_the_set_not_installed(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stamp beside a file somebody removed is a stamp that lies."""
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, payloads_for(manifest, spec))
    denoisemodels.pull(config, manifest, spec)
    assert denoisemodels.installed(config.home, manifest, spec) is not None
    (denoisemodels.denoise_models_root(config.home) / manifest.config_filename).unlink()
    assert denoisemodels.installed(config.home, manifest, spec) is None


# --------------------------------------------------------------------- the CLI


@pytest.fixture
def cli_backend(monkeypatch: pytest.MonkeyPatch):
    from crucible import cli

    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)
    return cli


def test_denoise_list_says_what_is_here_and_what_is_not(
    config, cli_backend, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    assert cli_backend.main(["denoise", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in rows] == ["denoise-roformer"]
    assert rows[0]["installed"] is False
    assert rows[0]["present"] is False
    assert denoisemodels.PULL_COMMAND in rows[0]["detail"]

    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, payloads_for(manifest, spec))
    monkeypatch.setattr(
        denoisemodels, "load_denoise_manifest", lambda *a, **k: manifest
    )
    monkeypatch.setattr(
        denoisemodels,
        "load_all_denoise_manifests",
        lambda *a, **k: {manifest.id: manifest},
    )
    assert cli_backend.main(["denoise", "pull", "denoise-roformer"]) == 0
    capsys.readouterr()

    assert cli_backend.main(["denoise", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["installed"] is True
    assert rows[0]["missing"] == []


def test_denoise_list_prints_a_row_per_model(config, cli_backend, capsys) -> None:
    assert cli_backend.main(["denoise", "list"]) == 0
    out = capsys.readouterr().out
    assert "denoise-roformer" in out
    assert "not pulled" in out


def test_denoise_pull_refuses_an_unknown_id(config, cli_backend, capsys) -> None:
    assert cli_backend.main(["denoise", "pull", "no-such-model"]) == 1
    err = capsys.readouterr().err
    assert "no-such-model" in err
    assert "denoise-roformer" in err


def test_denoise_pull_refuses_a_backend_with_no_block(
    config, cli_backend, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    manifest = shipped_with_test_digests()
    only_mac = denoisemodels.DenoiseManifest(
        id=manifest.id,
        display=manifest.display,
        model_filename=manifest.model_filename,
        config_filename=manifest.config_filename,
        primary_stem=manifest.primary_stem,
        sample_rate=manifest.sample_rate,
        backends={"mlx-darwin": manifest.backends["mlx-darwin"]},
        path=manifest.path,
    )
    monkeypatch.setattr(
        denoisemodels, "load_denoise_manifest", lambda *a, **k: only_mac
    )
    assert cli_backend.main(["denoise", "pull", manifest.id]) == 1
    err = capsys.readouterr().err
    assert FAKE_BACKEND.kind in err
    assert "mlx-darwin" in err


def test_denoise_pull_refuses_a_digest_mismatch(
    config, cli_backend, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, {spec.model_path: b"substituted", spec.config_path: YAML})
    monkeypatch.setattr(
        denoisemodels, "load_denoise_manifest", lambda *a, **k: manifest
    )
    assert cli_backend.main(["denoise", "pull", manifest.id]) == 1
    assert "hashes to" in capsys.readouterr().err


def test_denoise_pull_says_so_when_the_file_is_not_there_afterwards(
    config, cli_backend, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`rvc pull-base`'s rule: a pull that finished and left the expected file
    absent is said out loud, never reported as success."""
    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, payloads_for(manifest, spec))
    monkeypatch.setattr(
        denoisemodels, "load_denoise_manifest", lambda *a, **k: manifest
    )

    real_pull = denoisemodels.pull

    def pull_then_lose_a_file(*args, **kwargs):
        result = real_pull(*args, **kwargs)
        (denoisemodels.denoise_models_root(config.home) / manifest.config_filename).unlink()
        return result

    monkeypatch.setattr(denoisemodels, "pull", pull_then_lose_a_file)
    assert cli_backend.main(["denoise", "pull", manifest.id]) == 1
    err = capsys.readouterr().err
    assert "the pull finished but" in err
    assert manifest.config_filename in err


# ------------------------------------------------- what the doctor then says


@pytest.fixture
def rvc_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/rvc` whose python is this interpreter.

    `rvc`, not `denoise`: this type has no env of its own. Without it `check()`
    stops at the env and never reaches the question these tests are about,
    which is whether the checkpoint is there.
    """
    import sys

    from crucible import workerenv

    directory = workerenv.worker_env_dir(home, "rvc")
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").symlink_to(sys.executable)
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "job_type": "rvc",
                "backend": FAKE_BACKEND.kind,
                "recipe": f"{FAKE_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    recipe = workerenv.recipe_for("rvc", FAKE_BACKEND.kind)
    pins = workerenv.recipe_pins(recipe)
    refs = workerenv.recipe_direct_refs(recipe)
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    monkeypatch.setattr(
        workerenv, "installed_direct_refs", lambda _home, _type: dict(refs)
    )
    return directory


def denoise_row(capsys: pytest.CaptureFixture[str], cli) -> dict:
    """`crucible doctor --json`'s row for the denoise job type."""
    cli.main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    return next(
        entry for entry in report["job_types"] if entry["name"] == "denoise"
    )


def test_the_doctor_line_flips_once_the_pull_has_run(
    config,
    cli_backend,
    monkeypatch: pytest.MonkeyPatch,
    rvc_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole point of the command, read where the operator reads it.

    `job denoise: NOT READY — no separator checkpoint is in ~/.crucible/
    denoise-models` is the line both machines printed before this verb existed.
    It flips because the puller writes into the tree `check()` reads, and that
    is one function rather than two that agree.
    """
    before = denoise_row(capsys, cli_backend)
    assert before["ready"] is False
    assert denoisemodels.PULL_COMMAND in before["detail"]
    assert "denoise-roformer" in before["detail"]

    manifest = shipped_with_test_digests()
    spec = manifest.spec(FAKE_BACKEND.kind)
    serve(monkeypatch, payloads_for(manifest, spec))
    denoisemodels.pull(config, manifest, spec)
    capsys.readouterr()

    after = denoise_row(capsys, cli_backend)
    assert after["ready"] is True
    assert "installed: ['denoise-roformer']" in after["detail"]
    # And `/v1/info`'s row for the same model names the same bytes: the
    # revision the puller pinned and the path it fetched. One model, one
    # description — a client never reconciles two.
    row = next(row for row in after["models"] if row["id"] == "denoise-roformer")
    shipped = load_denoise_manifest("denoise-roformer").spec(FAKE_BACKEND.kind)
    assert row["revision"] == shipped.revision
    assert row["source"] == f"{shipped.hf_repo}:{shipped.model_path}"


def test_the_refusal_names_the_command_that_fetches_them(config) -> None:
    from crucible.errors import ApiError

    manifest = load_denoise_manifest("denoise-roformer")
    with pytest.raises(ApiError) as caught:
        denoise_job._require_model_files(config, manifest)
    error = caught.value
    assert error.code == "denoise_model_missing"
    assert error.details["command"] == f"{denoisemodels.PULL_COMMAND} {manifest.id}"
    assert error.details["command"] in error.message
    assert error.details["hf_repo"] == "Politrees/UVR_resources"
