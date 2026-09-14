"""`DELETE /v1/catalog/{kind}/{id}` — PHASE15-HOST.md 3.5a.

THE DOOR THE WEIGHTS RULE NEEDS. 3.5: a subject is never stored twice on one
machine, so when the WSL guest has its own copy the Windows one goes. The
host must never reach into `crucible/weights.py`'s layout from outside to do
that — a layout with two owners is ARCHITECTURE.md R1's shape — so the server
that owns the disk owns the deletion and this is how it is asked.

What these tests hold to is the ORDER of the refusals and the fact that
nothing is deleted when one fires. The order is the job door's: what is wrong
with the REQUEST first (an unknown kind or id is true whatever this server is
doing), then what is wrong with this server's STATE. A caller who misspelled
a subject id and was told "it is in use" would fix the wrong thing.

`tests/test_host.py`'s migration tests drive the OTHER side of the same door
through `crucible/host/catalog.py`, against a fake server; these drive the
real route. The two agree on the codes, and `subject_in_use` in particular,
because the migration's retry turns on that exact string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import weights
from crucible.manifests import load_manifest

from .conftest import FAKE_BACKEND


def catalog_rows(client: TestClient, auth: dict[str, str]) -> list[dict[str, Any]]:
    response = client.get("/v1/catalog", headers=auth)
    assert response.status_code == 200, response.text
    return response.json()["rows"]


def row_for(
    client: TestClient, auth: dict[str, str], kind: str, subject_id: str
) -> dict[str, Any]:
    found = [
        row
        for row in catalog_rows(client, auth)
        if row["kind"] == kind and row["id"] == subject_id
    ]
    assert len(found) == 1, f"{kind}/{subject_id} appears {len(found)} times"
    return found[0]


# ------------------------------------------------------------ what is refused


def test_an_unknown_kind_is_subject_unknown_and_names_the_kinds(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.delete("/v1/catalog/weights/qwen3.5-9b", headers=auth)
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "subject_unknown"
    assert "model" in error["message"] and "engine" in error["message"]


def test_an_unknown_id_is_subject_unknown_and_points_at_the_catalog(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.delete("/v1/catalog/model/no-such-model", headers=auth)
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "subject_unknown"
    assert "GET /v1/catalog" in error["message"]
    assert error["details"] == {"kind": "model", "id": "no-such-model"}


def test_a_subject_that_is_not_installed_is_REFUSED_not_answered_204(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A caller told "done" about a subject that was never there would
    believe a migration had deleted something."""
    assert row_for(client, auth, "model", "qwen3.5-9b")["installed"] is False
    response = client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "subject_not_installed"


def test_the_request_is_checked_BEFORE_the_state(
    client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    """A misspelled id is a misspelled id whatever this server is doing."""
    fake_weights("qwen3.5-9b")
    # Installed and NOT in use, so the only thing wrong with this request is
    # the id — and that is what comes back, not "not installed".
    response = client.delete("/v1/catalog/model/qwen3.5-9C", headers=auth)
    assert response.json()["error"]["code"] == "subject_unknown"


# --------------------------------------------------------------- what is done


def test_removing_an_installed_model_takes_its_files_and_answers_204(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    directory = fake_weights("qwen3.5-9b")
    (directory / "model.safetensors").write_bytes(b"x" * 64)
    assert row_for(client, auth, "model", "qwen3.5-9b")["installed"] is True

    response = client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    assert response.status_code == 204, response.text
    assert response.content == b""
    assert not directory.exists()
    # And the catalog says so, which is what the host polls.
    assert row_for(client, auth, "model", "qwen3.5-9b")["installed"] is False


def test_the_subject_s_own_directory_goes_when_it_is_empty(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    """3.5a: *"and the subject's directory if it is then empty"*."""
    directory = fake_weights("qwen3.5-9b")
    subject_dir = directory.parent
    client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    assert not subject_dir.exists()
    # But never the tree this module owns.
    assert (home / "models").is_dir()


def test_another_backend_s_copy_of_the_same_subject_is_untouched(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    """A machine that ran one backend yesterday and another today has two.

    3.5's migration deletes ONE of them, which is only true if this door
    deletes one of them.
    """
    fake_weights("qwen3.5-9b")
    other = home / "models" / "qwen3.5-9b" / "mlx-darwin"
    other.mkdir(parents=True, exist_ok=True)
    (other / "weights.safetensors").write_bytes(b"not this backend's")

    assert client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth).status_code == 204
    assert other.is_dir()
    assert (other / "weights.safetensors").is_file()


# ------------------------------------------------------------- subject_in_use


def test_a_resident_model_cannot_be_removed_and_details_say_who(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    """Deleting the files under a running engine leaves it serving nothing."""
    directory = fake_weights("qwen3.5-9b")
    residency = client.app.state.residency  # type: ignore[attr-defined]

    class Resident:
        id = "qwen3.5-9b"
        kind = "llm"

    residency._resident = Resident()  # noqa: SLF001 - the state under test
    try:
        response = client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    finally:
        residency._resident = None  # noqa: SLF001
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "subject_in_use"
    # `details.who` is pinned by 3.5a and read verbatim by
    # `crucible/host/catalog.py`'s `CatalogRefusal`.
    assert "who" in error["details"]
    assert "card" in error["details"]["who"]
    assert directory.exists(), "nothing is deleted when the door refuses"


def test_a_leased_subject_cannot_be_removed_and_the_receipt_travels(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    directory = fake_weights("qwen3.5-9b")
    # Opened through `Leases` directly rather than through the route, because
    # the route requires the subject to be RESIDENT first (a lease promises
    # not to move what is on the card) and this test is about the OTHER of
    # the three holds. Going through the route would prove the resident check
    # a second time and this one not at all.
    client.app.state.leases.open(  # type: ignore[attr-defined]
        kind="llm",
        subject="qwen3.5-9b",
        act="clean",
        client="foundry/1",
        ttl_seconds=60,
    )

    response = client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "subject_in_use"
    assert error["details"]["fact"] == "lease"
    assert "foundry/1" in error["details"]["who"]
    # The RECEIPT travels, so a caller can name the holder verbatim rather
    # than re-deriving it — the same shape `POST /lease` hands back.
    assert error["details"]["subject"] == "qwen3.5-9b"
    assert error["details"]["act"] == "clean"
    assert directory.exists(), "nothing is deleted when the door refuses"


# ------------------------------------------------------------- the record


def test_the_removal_is_recorded_in_activity_with_the_act(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    """3.5a: *"recorded in `/v1/activity` with the act"*."""
    fake_weights("qwen3.5-9b")
    assert (
        client.delete(
            "/v1/catalog/model/qwen3.5-9b",
            headers={**auth, "X-Crucible-Act": "clean", "User-Agent": "the-host/1"},
        ).status_code
        == 204
    )
    activity = client.get("/v1/activity", headers=auth).json()
    rows = activity["catalog"]["removals"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "model"
    assert rows[0]["id"] == "qwen3.5-9b"
    assert rows[0]["act"] == "clean"
    assert "the-host/1" in rows[0]["client"]
    assert rows[0]["bytes_freed"] == 19_306_310_880


def test_the_record_is_newest_first_and_never_holds_a_path(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights("qwen3.5-9b")
    fake_weights("qwen3.8-27b-4bit")
    client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    client.delete("/v1/catalog/model/qwen3.8-27b-4bit", headers=auth)
    rows = client.get("/v1/activity", headers=auth).json()["catalog"]["removals"]
    assert [row["id"] for row in rows] == ["qwen3.8-27b-4bit", "qwen3.5-9b"]
    assert all("path" not in row for row in rows)


# ------------------------------------------------ the kinds that are not models


def test_a_denoise_separator_takes_its_own_files_and_leaves_the_others(
    client: TestClient, auth: dict[str, str], home: Path
) -> None:
    """One flat directory holds every separator (`crucible/denoisemodels.py`).

    Removing the DIRECTORY would remove somebody else's model, so the set
    goes and the directory stays — which is the whole reason `remove_files`
    exists beside `remove`.
    """
    from crucible import denoisemodels

    manifest = next(iter(denoisemodels.load_all_denoise_manifests().values()))
    spec = manifest.spec(FAKE_BACKEND.kind)
    root = denoisemodels.denoise_models_root(home)
    root.mkdir(parents=True, exist_ok=True)
    for name in (manifest.model_filename, manifest.config_filename):
        (root / name).write_bytes(b"x" * 8)
    (root / "somebody-elses.ckpt").write_bytes(b"not mine")
    (root / denoisemodels.stamp_name(manifest)).write_text(
        json.dumps(
            {
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "bytes": 24,
                "pulled": "2026-09-14T00:00:00+0000",
                "files": [
                    {"target": manifest.model_filename},
                    {"target": manifest.config_filename},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert row_for(client, auth, "denoise", manifest.id)["installed"] is True

    assert (
        client.delete(
            f"/v1/catalog/denoise/{manifest.id}", headers=auth
        ).status_code
        == 204
    )
    assert not (root / manifest.model_filename).exists()
    assert (root / "somebody-elses.ckpt").is_file()
    assert root.is_dir()


def test_every_kind_the_catalog_lists_has_a_remove(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Not a table here: the subjects themselves are asked.

    A sixth kind added to `catalog.subjects()` without a `remove` would be a
    row the page draws a button on and the door cannot serve.
    """
    from crucible import catalog as catalog_module
    from crucible.backend import Backend, Gpu

    live = client.app.state.config  # type: ignore[attr-defined]
    for subject in catalog_module.subjects(live, FAKE_BACKEND):
        assert callable(subject.remove), f"{subject.kind}/{subject.id}"

    windows = Backend(
        kind="llama-windows",
        platform="windows",
        arch="AMD64",
        gpu=Gpu(vendor="nvidia", name="RTX 3090 Ti", vram_bytes=24 * 1024**3),
        detail="llama.cpp cuda build",
    )
    for subject in catalog_module.subjects(live, windows):
        assert callable(subject.remove), f"{subject.kind}/{subject.id}"


def test_a_removal_that_fails_is_subject_remove_failed_with_the_path(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = fake_weights("qwen3.5-9b")

    def refuse(path: Path) -> None:
        raise weights.RemoveFailed(path, f"{path} is held open by something")

    monkeypatch.setattr(weights, "_remove", refuse)
    response = client.delete("/v1/catalog/model/qwen3.5-9b", headers=auth)
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "subject_remove_failed"
    assert error["details"]["path"] == str(directory)
    assert directory.exists()
