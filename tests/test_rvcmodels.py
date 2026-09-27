from __future__ import annotations

from pathlib import Path

import pytest

from crucible.backend import CUDA_LINUX, MLX_DARWIN
from crucible.rvcmodels import (
    RVC_BACKEND_ENGINES,
    RvcManifestError,
    load_all_rvc_manifests,
    load_rvc_manifest,
    parse_rvc_manifest,
)

REPO = "owenmorgan/owen-morgan-bookforge"
REVISION = "dd3504ae68cc8f042a20d7ed8a84238615d398e3"
SHA = "c9d3fea887c4875da5c2848e929f7d891059bc2b40219e22a85e28a61ff1e777"

GOOD = f"""
[model]
id = "deathstalker-rvc-v1"
display = "Deathstalker RVC v1"
model_name = "deathstalker_rvc_v1"
has_index = true

[backends.cuda-linux]
engine = "ultimate-rvc"
hf_repo = "{REPO}"
revision = "{REVISION}"
archive = "rvc/deathstalker_rvc_v1.tar.gz"
archive_sha256 = "{SHA}"
archive_bytes = 184681783
memory_bytes_estimate = 2684354560
"""


def parse(text: str, model_id: str = "deathstalker-rvc-v1"):
    return parse_rvc_manifest(text, Path(f"{model_id}.toml"), model_id)


def test_every_published_model_has_a_manifest() -> None:
    assert sorted(load_all_rvc_manifests()) == [
        "deathstalker-rvc-v1",
        "deathstalker-rvc-v3",
        "girlfriend",
        "mistborn-rvc-v1",
        "owen-morgan",
        "sigma",
        "us-female-1",
    ]


def test_every_shipped_manifest_names_one_archive_in_the_shared_repo() -> None:
    for manifest in load_all_rvc_manifests().values():
        for backend in (CUDA_LINUX, MLX_DARWIN):
            spec = manifest.spec(backend)
            assert spec.hf_repo == REPO
            assert spec.revision == REVISION
            assert spec.archive.startswith("rvc/")
            assert spec.archive.endswith(".tar.gz")
            assert len(spec.archive_sha256) == 64


def test_the_model_name_is_not_derivable_from_the_id() -> None:
    names = {m.id: m.model_name for m in load_all_rvc_manifests().values()}
    assert names["sigma"] == "Sigma Male Narrator"
    assert names["us-female-1"] == "US_Female_1"
    assert names["owen-morgan"] == "Owen Morgan"


def test_every_published_model_ships_an_index() -> None:
    assert all(m.has_index for m in load_all_rvc_manifests().values())


def test_rvc_weights_live_in_their_own_namespace() -> None:
    assert load_rvc_manifest("sigma").weights_family == "rvc"


def test_the_mac_is_a_real_backend_here_unlike_asr() -> None:
    assert sorted(RVC_BACKEND_ENGINES) == sorted([CUDA_LINUX, MLX_DARWIN])
    assert load_rvc_manifest("sigma").supports(MLX_DARWIN)


def test_a_good_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.model_name == "deathstalker_rvc_v1"
    assert manifest.has_index is True
    assert manifest.spec(CUDA_LINUX).archive_bytes == 184681783


def test_an_unknown_key_is_refused_not_ignored() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace("has_index = true", "has_index = true\nindex_rate = 0.5"))
    assert "unknown key(s) ['index_rate']" in str(caught.value)


def test_a_digest_that_is_not_a_sha256_is_refused() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace(SHA, "deadbeef"))
    assert "64 lower-case hex characters" in str(caught.value)


def test_a_branch_name_is_not_a_pin() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace(REVISION, "main"))
    assert "branch names are not pins" in str(caught.value)


@pytest.mark.parametrize(
    "archive",
    [
        "/etc/passwd.tar.gz",
        "../../escape.tar.gz",
        "rvc/../../escape.tar.gz",
        "rvc/model.zip",
    ],
)
def test_an_archive_that_escapes_or_is_not_a_tarball_is_refused(archive: str) -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace("rvc/deathstalker_rvc_v1.tar.gz", archive))
    assert "repo-relative" in str(caught.value)


def test_a_model_name_with_a_separator_is_refused() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace('"deathstalker_rvc_v1"\nhas_index', '"a/b"\nhas_index'))
    assert "single folder name" in str(caught.value)


def test_has_index_must_be_a_bool_not_a_string() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace("has_index = true", 'has_index = "true"'))
    assert "has_index must be bool" in str(caught.value)


def test_the_id_and_the_filename_are_the_same_thing() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse_rvc_manifest(GOOD, Path("deathstalker.toml"), "deathstalker")
    assert "the id and the filename are the same thing" in str(caught.value)


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(RvcManifestError) as caught:
        parse(GOOD.replace("[backends.cuda-linux]", "[backends.rocm-linux]"))
    assert "not an rvc backend" in str(caught.value)


def test_a_missing_manifest_lists_what_ships(tmp_path: Path) -> None:
    with pytest.raises(RvcManifestError) as caught:
        load_rvc_manifest("nobody", tmp_path)
    assert "this build ships []" in str(caught.value)
