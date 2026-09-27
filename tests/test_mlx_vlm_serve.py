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
    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (255, 255, 255, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def page_body(model: str, width: int = 100, height: int = 160, **overrides: Any) -> dict:
    body = pages.request_body(pages.data_uri(png(width, height)), model=model)
    body.update(overrides)
    return body


class ServedFake(MlxVlmEngine):

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


def test_the_command_is_the_shipped_script_from_the_env_python(tmp_path: Path) -> None:
    engine = MlxVlmEngine(Path("/env/bin/python"), tmp_path / "x.log")
    command = engine.command(Path("/w/dots"), "/w/dots", 4321, ["--width", "2"])
    assert command[:2] == [str(Path("/env/bin/python")), str(SERVE_SCRIPT)]
    assert SERVE_SCRIPT.is_file(), "the server ships inside the package"
    assert command[2:] == [
        "--model", str(Path("/w/dots")), "--host", "127.0.0.1", "--port", "4321",
        "--width", "2",
    ]


def test_a_manifest_that_forgot_the_width_is_refused_before_a_spawn(tmp_path: Path) -> None:
    engine = MlxVlmEngine(Path(sys.executable), tmp_path / "x.log")
    with pytest.raises(EngineError) as caught:
        engine.command(tmp_path, str(tmp_path), 1, [])
    assert "--width" in str(caught.value)
    assert "models/dots-ocr.toml" in str(caught.value)


def test_ready_means_loaded_and_the_name_is_verbatim(served, tmp_path: Path) -> None:
    engine, name, _ = served
    with urllib.request.urlopen(f"{engine.base_url}/v1/models", timeout=5) as response:
        listed = json.loads(response.read())["data"]
    assert [entry["id"] for entry in listed] == [name]
    log = (tmp_path / "engine.log").read_text(encoding="utf-8", errors="replace")
    assert "mlx_vlm_serve.py --model" in log
    assert "loaded" in log
    assert "width 2" in log


def test_models_refuses_until_the_weights_are_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_a_page_comes_back_as_a_chat_completion_with_usage(served) -> None:
    engine, name, _ = served
    status, document = post(engine, page_body(name, width=112, height=168))
    assert status == 200, document
    choice = document["choices"][0]
    assert choice["message"]["content"] == "page 168x112 row 0"
    assert choice["finish_reason"] == "stop"
    assert document["model"] == name
    assert document["usage"] == {
        "prompt_tokens": 34 + (168 * 112) // 196,
        "completion_tokens": len("page 168x112 row 0"),
        "total_tokens": 34 + (168 * 112) // 196 + len("page 168x112 row 0"),
    }
    assert not pages.was_truncated(choice)


def test_a_page_cut_off_at_max_tokens_says_length(served) -> None:
    engine, name, _ = served
    status, document = post(engine, page_body(name, max_tokens=5))
    assert status == 200, document
    choice = document["choices"][0]
    assert choice["message"]["content"] == "page "
    assert choice["finish_reason"] == "length"
    assert document["usage"]["completion_tokens"] == 5
    assert pages.was_truncated(choice)


def test_an_rgba_png_is_read_as_rgb(served) -> None:
    engine, name, _ = served
    status, document = post(engine, page_body(name, width=56, height=84))
    assert status == 200
    assert document["choices"][0]["message"]["content"] == "page 84x56 row 0"


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
    assert repr(name) in document["error"]["message"]


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
    engine, name, batches = served
    results = _read_many(engine, [page_body(name, width=112, height=168) for _ in range(6)])
    assert all(status == 200 for status, _ in results), results
    contents = sorted(r["choices"][0]["message"]["content"] for _, r in results)
    assert all(c.startswith("page 168x112 row ") for c in contents)
    recorded = [json.loads(line) for line in batches.read_text().splitlines()]
    assert sum(b["rows"] for b in recorded) == 6
    assert max(b["rows"] for b in recorded) <= 2
    assert any(b["rows"] == 2 for b in recorded), recorded
    assert all(b["batch_size"] == b["rows"] for b in recorded), recorded


def test_two_raw_sizes_on_one_grid_share_a_batch_and_two_grids_do_not() -> None:
    import threading

    taken: list[list[tuple[int, ...]]] = []
    gate = threading.Event()
    first_started = threading.Event()

    class HeldReader:
        def grid_of(self, image):
            return image

        def read_batch(self, jobs):
            taken.append([job.grid for job in jobs])
            if len(taken) == 1:
                first_started.set()
                gate.wait(10)
            return [
                mlx_vlm_serve.Row(text="x", finish_reason="stop", prompt_tokens=1, completion_tokens=1)
                for _ in jobs
            ]

    logged: list[str] = []
    batcher = mlx_vlm_serve.Batcher(HeldReader(), width=3, log=logged.append)
    batcher.submit(mlx_vlm_serve.Job(image=None, prompt="p", max_tokens=1, grid=(0,)))
    assert first_started.wait(10)
    grid_a, grid_b = (1, 90, 52), (1, 160, 92)
    waiting = [
        mlx_vlm_serve.Job(image=None, prompt="p", max_tokens=1, grid=grid_a),
        mlx_vlm_serve.Job(image=None, prompt="p", max_tokens=1, grid=grid_b),
        mlx_vlm_serve.Job(image=None, prompt="p", max_tokens=1, grid=grid_a),
        mlx_vlm_serve.Job(image=None, prompt="p", max_tokens=1, grid=grid_a),
        mlx_vlm_serve.Job(image=None, prompt="p", max_tokens=1, grid=grid_a),
    ]
    for job in waiting:
        batcher.submit(job)
    gate.set()
    for job in waiting:
        assert job.done.wait(10)
    assert taken[1] == [grid_a, grid_a, grid_a], taken
    assert taken[2] == [grid_b], taken
    assert taken[3] == [grid_a], taken
    assert all("describing it failed" in line for line in logged), logged


def test_the_grid_key_comes_from_the_processor_and_absorbs_pixel_jitter(served) -> None:
    engine, name, batches = served
    bodies = [page_body(name, width=100, height=168), page_body(name, width=110, height=168)]
    results = _read_many(engine, bodies)
    assert all(status == 200 for status, _ in results), results
    assert {r["choices"][0]["message"]["content"][:14] for _, r in results} == {"page 168x112 r"}
    status, document = post(engine, page_body(name, width=130, height=168))
    assert status == 200
    assert document["choices"][0]["message"]["content"].startswith("page 168x140 row")


def test_a_failed_batch_fails_its_rows_and_the_next_page_still_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_MLX_VLM_RAISE_ON", "84")
    weights = tmp_path / "dots"
    weights.mkdir()
    engine = ServedFake(Path(sys.executable), tmp_path / "engine.log")
    engine.start(weights, str(weights), find_free_port(), ["--width", "2"])
    try:
        engine.ready(60.0)
        status, document = post(engine, page_body(str(weights), width=56, height=77))
        assert status == 500
        assert document["error"]["code"] == "engine_error"
        assert "84 tall" in document["error"]["message"]
        status, document = post(engine, page_body(str(weights), width=56, height=112))
        assert status == 200, document
        assert document["choices"][0]["message"]["content"] == "page 112x56 row 0"
    finally:
        engine.stop()


def test_parse_request_reads_exactly_the_published_shape() -> None:
    image, prompt, max_tokens = mlx_vlm_serve.parse_request(page_body("m", width=30, height=40), "m")
    assert (image.height, image.width) == (40, 30)
    assert image.mode == "RGB"
    assert prompt == pages.DOTS_PROMPT
    assert max_tokens == pages.MAX_TOKENS


def test_the_request_shape_the_server_reads_is_the_one_pages_publishes() -> None:
    published = set(pages.request_body("data:image/png;base64,AA==", model="m"))
    assert published <= mlx_vlm_serve.KNOWN_FIELDS, published - mlx_vlm_serve.KNOWN_FIELDS


def test_a_missing_width_stops_the_process_by_name(tmp_path: Path) -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SERVE_SCRIPT), "--model", str(tmp_path), "--host", "127.0.0.1", "--port", "1"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(FAKE)},
    )
    assert result.returncode != 0
    assert "--width" in result.stderr
