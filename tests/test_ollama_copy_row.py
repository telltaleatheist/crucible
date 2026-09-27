from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from crucible import ollamastore
from crucible.backend import Backend, Gpu
from crucible.config import load_config, write_config
from crucible.jobs.llm import model_rows
from crucible.residency import Residency

DIGEST = "sha256:" + "ab" * 32


def a_store(root: Path, tag: str, payload: bytes = b"GGUF-bytes") -> None:
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    name, _, label = tag.partition(":")
    folder = root / "manifests" / "registry.ollama.ai" / "library" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / label).write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "layers": [
                    {
                        "mediaType": ollamastore.MODEL_MEDIA_TYPE,
                        "digest": DIGEST,
                        "size": len(payload),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "blobs" / DIGEST.replace(":", "-", 1)).write_bytes(payload)


def rows(backend_kind: str, store: Path | None) -> dict[str, dict[str, Any]]:
    home = Path(tempfile.mkdtemp(prefix="crucible-ocopy-"))
    os.environ["CRUCIBLE_HOME"] = str(home)
    if store is None:
        os.environ.pop(ollamastore.STORE_ENV, None)
        os.environ[ollamastore.STORE_ENV] = str(home / "no-ollama-here")
    else:
        os.environ[ollamastore.STORE_ENV] = str(store)
    backend = Backend(
        kind=backend_kind,
        platform="win32" if backend_kind == "llama-windows" else "linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="RTX 3090 Ti", vram_bytes=25_757_220_864),
        detail="test double",
    )
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="not-minted",
        backend_kind=backend.kind,
        enable_echo=True,
        enable_llm=True,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        desktop_allowance_bytes=3 * 1024 ** 3,
        enable_denoise=False,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
    )
    config = load_config(home)
    return {row["id"]: row for row in model_rows(config, backend, Residency(config))}


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "ollama"
    a_store(root, "qwen3.5:9b-bf16")
    return root


def test_a_copy_that_is_there_is_reported(store: Path) -> None:
    found = rows("llama-windows", store)["qwen3.5-9b"]["ollama_copy"]
    assert found is not None
    assert found["tag"] == "qwen3.5:9b-bf16"
    assert found["provenance"].startswith("ollama:qwen3.5:9b-bf16@sha256:")
    assert Path(found["path"]).is_file()


def test_the_row_says_it_is_not_the_file_the_block_pins(store: Path) -> None:
    found = rows("llama-windows", store)["qwen3.5-9b"]["ollama_copy"]
    assert found["same_file_as_the_pin"] is False
    assert "unsloth" not in found["provenance"]


def test_only_llama_windows_is_told_about_it(store: Path) -> None:
    for kind in ("cuda-linux", "mlx-darwin"):
        for row in rows(kind, store).values():
            assert row["ollama_copy"] is None, (kind, row["id"])


def test_a_tag_that_was_never_pulled_is_simply_absent(tmp_path: Path) -> None:
    root = tmp_path / "ollama"
    a_store(root, "qwen3.8:27b-24g")
    found = rows("llama-windows", root)
    assert found["qwen3.8-27b-4bit"]["ollama_copy"] is None
    assert found["qwen3.5-9b"]["ollama_copy"] is None


def test_no_ollama_at_all_is_quiet(tmp_path: Path) -> None:
    for row in rows("llama-windows", None).values():
        assert row["ollama_copy"] is None


def test_a_model_with_no_local_table_claims_nothing(store: Path) -> None:
    assert rows("llama-windows", store)["qwen3.8-27b-8bit"]["ollama_copy"] is None
