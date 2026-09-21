"""Crucible's page server for `mlx-darwin`, driven through the real engine class.

The process under test is the one `MlxVlmEngine.command()` builds — the llm
env's python running `crucible/engines/mlx_vlm_serve.py` — with a fake
`mlx_vlm` on PYTHONPATH (`tests/fake_mlx_vlm`) that reproduces the call shapes
of mlx-vlm 0.7.1 and "reads" every page as `page <h>x<w> row <i>`. What is
proven here is everything around the model: the readiness contract, the wire,
the refusals, the width and the same-shape rule, `finish_reason` and `usage`,
and that a failed batch costs its rows and not the thread. What is NOT proven
here is that a page is read — that was measured on the Mac Studio on
2026-09-21 and is recorded in the server's module docstring.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from crucible import pages
from crucible.engines import EngineError, find_free_port
from crucible.engines.mlx_vlm import SERVE_SCRIPT, MlxVlmEngine
from crucible.engines import mlx_vlm_serve

FAKE = Path(__file__).resolve().parent / "fake_mlx_vlm"


def png(width: int, height: int) -> bytes:
    """A real PNG, because the server decodes with real Pillow."""
    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (255, 255, 255, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def page_body(model: str, width: int = 100, height: int = 160, **overrides: Any) -> dict:
    """`crucible.pages.request_body`'s shape, which is the one the server is for."""
    body = pages.request_body(pages.data_uri(png(width, height)), model=model)
    body.update(overrides)
    return body


class ServedFake(MlxVlmEngine):
    """The REAL engine class, pointed at the fake package; nothing else differs."""

    def __init__(self, python: Path, log_path: Path, batches: Path | None = None) -> None:
        super().__init__(python, log_path)
        self._batches = batches

    def environment(self) -> dict[str, str]:
        environment = {"PYTHONPATH": str(FAKE)}
        if self._batches is not None:
            environment["CRUCIBLE_FAKE_MLX_VLM_BATCHES"] = str(self._batches)
        return environment


@pytest.fixture
def served(tmp_path: Path):
    """A ready engine at width 2, and its weights dir (the served name)."""
    weights = tmp_path / "dots"
    weights.mkdir()
    batches = tmp_path / "batches.jsonl"
    engine = ServedFake(Path(sys.executable), tmp_path / "engine.log", batches)
    engine.start(weights, str(weights), find_free_port(), ["--width", "2"])
    try:
        engine.ready(60.0)
        yield engine, str(weights), batches
    finally:
        engine.stop()


def post(engine: MlxVlmEngine, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{engine.base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# ------------------------------------------------------------- the contract


def test_the_command_is_the_shipped_script_from_the_env_python(tmp_path: Path) -> None:
    engine = MlxVlmEngine(Path("/env/bin/python"), tmp_path / "x.log")
    command = engine.command(Path("/w/dots"), "/w/dots", 4321, ["--width", "2"])
    assert command[:2] == ["/env/bin/python", str(SERVE_SCRIPT)]
    assert SERVE_SCRIPT.is_file(), "the server ships inside the package"
    assert command[2:] == ["--model", "/w/dots", "--host", "127.0.0.1", "--port", "4321", "--width", "2"]


def test_a_manifest_that_forgot_the_width_is_refused_before_a_spawn(tmp_path: Path) -> None:
    """Point 3: the width is the manifest's, and a missing one is named."""
    engine = MlxVlmEngine(Path(sys.executable), tmp_path / "x.log")
    with pytest.raises(EngineError) as caught:
        engine.command(tmp_path, str(tmp_path), 1, [])
    assert "--width" in str(caught.value)
    assert "models/dots-ocr.toml" in str(caught.value)


def test_ready_means_loaded_and_the_name_is_verbatim(served, tmp_path: Path) -> None:
    """Points 1 and 2: /v1/models answers only after load, with the dir as given."""
    engine, name, _ = served
    with urllib.request.urlopen(f"{engine.base_url}/v1/models", timeout=5) as response:
        listed = json.loads(response.read())["data"]
    assert [entry["id"] for entry in listed] == [name]
    log = (tmp_path / "engine.log").read_text(encoding="utf-8", errors="replace")
    assert "mlx_vlm_serve.py --model" in log
    assert "loaded" in log
    assert "width 2" in log


def test_models_refuses_until_the_weights_are_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake sleeps in load(); the socket must not exist meanwhile."""
    monkeypatch.setenv("CRUCIBLE_FAKE_MLX_VLM_LOAD_S", "2")
    weights = tmp_path / "dots"
    weights.mkdir()
    engine = ServedFake(Path(sys.executable), tmp_path / "engine.log")
    port = find_free_port()
    engine.start(weights, str(weights), port, ["--width", "1"])
    try:
        assert engine.announced_ready() is None, "answered before the load finished"
        engine.ready(60.0)
        assert engine.announced_ready() is not None
    finally:
        engine.stop()
    assert engine.pids == frozenset()


# ------------------------------------------------------------------ the wire


def test_a_page_comes_back_as_a_chat_completion_with_usage(served) -> None:
    engine, name, _ = served
    status, document = post(engine, page_body(name, width=100, height=160))
    assert status == 200, document
    choice = document["choices"][0]
    assert choice["message"]["content"] == "page 160x100 row 0"
    assert choice["finish_reason"] == "stop"
    assert document["model"] == name
    assert document["usage"] == {
        # 34 + (160 * 100) // 196, the fake's stated rule.
        "prompt_tokens": 34 + (160 * 100) // 196,
        "completion_tokens": len("page 160x100 row 0"),
        "total_tokens": 34 + (160 * 100) // 196 + len("page 160x100 row 0"),
    }
    assert not pages.was_truncated(choice)


def test_a_page_cut_off_at_max_tokens_says_length(served) -> None:
    """THE FIELD THE CONTRACT NEEDS. `batch_generate` throws it away; the
    reader keeps it, and `pages.was_truncated` reads it."""
    engine, name, _ = served
    status, document = post(engine, page_body(name, max_tokens=5))
    assert status == 200, document
    choice = document["choices"][0]
    assert choice["message"]["content"] == "page "
    assert choice["finish_reason"] == "length"
    assert document["usage"]["completion_tokens"] == 5
    assert pages.was_truncated(choice)


def test_an_rgba_png_is_read_as_rgb(served) -> None:
    """The Mac's finding: a PNG that decodes to RGBA trips the image processor,
    so every image is converted. The fake reports the shape it received."""
    engine, name, _ = served
    status, document = post(engine, page_body(name, width=64, height=80))
    assert status == 200
    assert document["choices"][0]["message"]["content"] == "page 80x64 row 0"


# ------------------------------------------------------------- the refusals


@pytest.mark.parametrize(
    "override,code",
    [
        ({"temperature": 0.5}, "not_greedy"),
        ({"top_p": 0.9}, "not_greedy"),
        ({"n": 2}, "one_choice"),
        ({"stream": True}, "no_streaming"),
        ({"top_k": 40}, "unknown_field"),
        ({"max_tokens": 0}, "max_tokens_required"),
    ],
)
def test_a_request_that_is_not_a_page_request_is_refused_by_name(served, override, code) -> None:
    engine, name, _ = served
    status, document = post(engine, page_body(name, **override))
    assert status == 400, document
    assert document["error"]["code"] == code


def test_the_wrong_model_is_a_404(served) -> None:
    engine, name, _ = served
    status, document = post(engine, page_body("some-other-model"))
    assert status == 404
    assert document["error"]["code"] == "model_not_found"
    assert name in document["error"]["message"]


def test_two_images_or_no_text_is_refused(served) -> None:
    engine, name, _ = served
    body = page_body(name)
    body["messages"][0]["content"].append(dict(body["messages"][0]["content"][0]))
    status, document = post(engine, body)
    assert status == 400
    assert document["error"]["code"] == "content_parts"


def test_a_fetchable_url_is_refused_this_reader_fetches_nothing(served) -> None:
    engine, name, _ = served
    body = page_body(name)
    body["messages"][0]["content"][0]["image_url"]["url"] = "https://example.com/page.png"
    status, document = post(engine, body)
    assert status == 400
    assert document["error"]["code"] == "not_a_data_uri"


# ---------------------------------------------------------------- batching


def _read_many(engine: MlxVlmEngine, bodies: list[dict]) -> list[tuple[int, dict]]:
    results: list[Any] = [None] * len(bodies)

    def one(index: int) -> None:
        results[index] = post(engine, bodies[index])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_concurrent_pages_are_batched_to_the_width_and_never_wider(served) -> None:
    """Six same-shape pages at width 2: no batch wider than 2, and at least one
    that IS 2 — a server that quietly read them one at a time would pass the
    first assertion and fail the second."""
    engine, name, batches = served
    results = _read_many(engine, [page_body(name, width=100, height=160) for _ in range(6)])
    assert all(status == 200 for status, _ in results), results
    contents = sorted(r["choices"][0]["message"]["content"] for _, r in results)
    # Every page answered from ITS OWN row of its batch.
    assert all(c.startswith("page 160x100 row ") for c in contents)
    recorded = [json.loads(line) for line in batches.read_text().splitlines()]
    assert sum(b["rows"] for b in recorded) == 6
    assert max(b["rows"] for b in recorded) <= 2
    assert any(b["rows"] == 2 for b in recorded), recorded
    # Prefill and completion batch are BOTH the row count (the fake asserts
    # they agree; this reads that they are the batch and not the width).
    assert all(b["batch_size"] == b["rows"] for b in recorded), recorded


def test_pages_of_two_shapes_never_share_a_batch(served) -> None:
    """Upstream prefills with `pad_to_uniform_size=False`; the fake's
    `prepare_inputs` asserts on a mixed batch, so a wrong grouping is a 500."""
    engine, name, batches = served
    bodies = [page_body(name, width=100, height=160), page_body(name, width=120, height=160)] * 2
    results = _read_many(engine, bodies)
    assert all(status == 200 for status, _ in results), results
    for line in batches.read_text().splitlines():
        shapes = json.loads(line)["shapes"]
        assert len({tuple(s) for s in shapes}) == 1, shapes


def test_a_failed_batch_fails_its_rows_and_the_next_page_still_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_MLX_VLM_RAISE_ON", "77")
    weights = tmp_path / "dots"
    weights.mkdir()
    engine = ServedFake(Path(sys.executable), tmp_path / "engine.log")
    engine.start(weights, str(weights), find_free_port(), ["--width", "2"])
    try:
        engine.ready(60.0)
        status, document = post(engine, page_body(str(weights), width=50, height=77))
        assert status == 500
        assert document["error"]["code"] == "engine_error"
        assert "77 tall" in document["error"]["message"]
        status, document = post(engine, page_body(str(weights), width=50, height=78))
        assert status == 200, document
        assert document["choices"][0]["message"]["content"] == "page 78x50 row 0"
    finally:
        engine.stop()


# ------------------------------------------- the parser, without a process


def test_parse_request_reads_exactly_the_published_shape() -> None:
    image, prompt, max_tokens = mlx_vlm_serve.parse_request(page_body("m", width=30, height=40), "m")
    assert (image.height, image.width) == (40, 30)
    assert image.mode == "RGB"
    assert prompt == pages.DOTS_PROMPT
    assert max_tokens == pages.MAX_TOKENS


def test_the_request_shape_the_server_reads_is_the_one_pages_publishes() -> None:
    """One owner. If `crucible/pages.py` ever adds a field to the page request,
    this server must learn it, and this is the test that says so."""
    published = set(pages.request_body("data:image/png;base64,AA==", model="m"))
    assert published <= mlx_vlm_serve.KNOWN_FIELDS, published - mlx_vlm_serve.KNOWN_FIELDS


def test_a_missing_width_stops_the_process_by_name(tmp_path: Path) -> None:
    """argparse's refusal, so the server never defaults it either."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SERVE_SCRIPT), "--model", str(tmp_path), "--host", "127.0.0.1", "--port", "1"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(FAKE)},
    )
    assert result.returncode != 0
    assert "--width" in result.stderr
