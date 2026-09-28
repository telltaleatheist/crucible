from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from crucible import decide as decide_core
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

    def __init__(
        self, python: Path, log_path: Path, batches: Path | None = None, model_type: str | None = None
    ) -> None:
        super().__init__(python, log_path)
        self._batches = batches
        self._model_type = model_type

    def environment(self) -> dict[str, str]:
        environment = {"PYTHONPATH": str(FAKE)}
        if self._batches is not None:
            environment["CRUCIBLE_FAKE_MLX_VLM_BATCHES"] = str(self._batches)
        if self._model_type is not None:
            environment["CRUCIBLE_FAKE_MLX_VLM_MODEL_TYPE"] = self._model_type
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


def coloured_png(width: int, height: int, colour: tuple[int, int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


THREE_FRAMES = (
    (64, 36, (255, 0, 0)),
    (48, 48, (0, 255, 0)),
    (32, 80, (0, 0, 255)),
)


def frames() -> list[str]:
    return [coloured_png(w, h, colour) for w, h, colour in THREE_FRAMES]


YESNO = decide_core.YesNoQuestion(type="yesno", instructions="A face is visible")


def question_body(model: str, images: list[str], k: int | None) -> dict:
    item = decide_core.plan("face", YESNO)
    messages = decide_core.question_messages("", images, item)
    return decide_core.request_body(model, messages, k)


def recorded(batches: Path) -> list[dict]:
    return [json.loads(line) for line in batches.read_text().splitlines()]


def test_a_question_returns_top_logprobs_in_the_openai_chat_shape(served) -> None:
    engine, name, batches = served
    status, document = post(engine, question_body(name, frames()[:1], 5))
    assert status == 200, document
    choice = document["choices"][0]
    assert choice["message"]["content"] == "A"
    assert choice["finish_reason"] == "length"
    (step,) = choice["logprobs"]["content"]
    assert step["token"] == "A" and step["logprob"] == pytest.approx(math.log(0.6))
    assert [entry["token"] for entry in step["top_logprobs"]] == ["A", "B", "C", "D", "E"]
    assert step["top_logprobs"][1] == {"token": "B", "logprob": pytest.approx(math.log(0.3)), "bytes": [66]}
    assert document["usage"]["prompt_tokens"] > 0
    (question,) = [entry for entry in recorded(batches) if entry.get("question")]
    assert question["compute_logprobs"] is True and question["top_logprobs_k"] == 5
    assert question["max_tokens"] == 1
    assert question["logits_dtype"] == "float32"
    assert "thinking=False" in question["prompt"]


def test_the_prime_asks_for_no_logprobs_and_gets_none(served) -> None:
    engine, name, batches = served
    body = decide_core.request_body(name, decide_core.prime_messages("state", frames()[:1]), None)
    status, document = post(engine, body)
    assert status == 200, document
    assert "logprobs" not in document["choices"][0]
    (question,) = [entry for entry in recorded(batches) if entry.get("question")]
    assert question["compute_logprobs"] is False and question["top_logprobs_k"] == 0
    assert question["logits_dtype"] == "bfloat16"


def test_three_images_in_one_question_reach_the_model_intact_and_in_order(served) -> None:
    engine, name, batches = served
    status, document = post(engine, question_body(name, frames(), 4))
    assert status == 200, document
    (question,) = [entry for entry in recorded(batches) if entry.get("question")]
    assert question["images"] == [[w, h, list(colour)] for w, h, colour in THREE_FRAMES]
    assert question["prompt"].startswith("Q|images=3|")
    assert "system:" + decide_core.SYSTEM_PROMPT in question["prompt"]
    assert decide_core.IMAGES_NOTE in question["prompt"]


def test_a_text_only_question_carries_no_images(served) -> None:
    engine, name, batches = served
    body = question_body(name, [], 2)
    status, document = post(engine, body)
    assert status == 200, document
    (question,) = [entry for entry in recorded(batches) if entry.get("question")]
    assert question["images"] == [] and question["prompt"].startswith("Q|images=0|")


def _too_many(body: dict) -> dict:
    body["top_logprobs"] = mlx_vlm_serve.MAX_TOP_LOGPROBS + 1
    return body


def _no_logprobs_flag(body: dict) -> dict:
    body.pop("logprobs")
    return body


def _nine_images(body: dict) -> dict:
    image = body["messages"][-1]["content"][0]
    body["messages"][-1]["content"] = [image] * 9 + body["messages"][-1]["content"][1:]
    return body


def _video_kwarg(body: dict) -> dict:
    body["chat_template_kwargs"] = {"enable_thinking": False, "video": "x"}
    return body


def _image_in_system(body: dict) -> dict:
    image = body["messages"][-1]["content"][0]
    body["messages"][0]["content"] = [image, {"type": "text", "text": "s"}]
    return body


def _warm(body: dict) -> dict:
    body["temperature"] = 0.7
    return body


@pytest.mark.parametrize(
    "spoil,code",
    [
        (_too_many, "too_many_top_logprobs"),
        (_no_logprobs_flag, "top_logprobs_without_logprobs"),
        (_nine_images, "too_many_images"),
        (_video_kwarg, "unknown_template_kwarg"),
        (_image_in_system, "content_parts"),
        (_warm, "not_greedy"),
    ],
)
def test_a_question_the_server_cannot_answer_is_refused_by_name(served, spoil, code) -> None:
    engine, name, batches = served
    status, document = post(engine, spoil(question_body(name, frames()[:1], 5)))
    assert status == 400, document
    assert document["error"]["code"] == code
    assert not batches.exists() or not recorded(batches)


def test_the_cap_the_server_refuses_past_is_named_in_the_refusal(served) -> None:
    engine, name, _ = served
    status, document = post(engine, _too_many(question_body(name, frames()[:1], 5)))
    assert status == 400
    assert str(mlx_vlm_serve.MAX_TOP_LOGPROBS) in document["error"]["message"]


def test_the_engine_states_the_cap_the_server_enforces() -> None:
    from crucible.engines import decide_reading
    from crucible.engines.mlx_lm import MlxLmEngine

    assert MlxVlmEngine.decide_logprobs is True
    assert MlxVlmEngine.max_logprobs == mlx_vlm_serve.MAX_TOP_LOGPROBS == MlxLmEngine.max_logprobs
    assert decide_reading("mlx-vlm").max_logprobs == mlx_vlm_serve.MAX_TOP_LOGPROBS
    assert mlx_vlm_serve.MAX_IMAGES == decide_core.MAX_IMAGES


def test_a_page_body_is_still_a_page_and_its_reply_carries_no_logprobs(served) -> None:
    engine, name, batches = served
    assert mlx_vlm_serve.is_page_request(page_body(name))
    assert not mlx_vlm_serve.is_page_request(question_body(name, frames()[:1], 3))
    assert not mlx_vlm_serve.is_page_request(question_body(name, frames()[:1], None))
    status, document = post(engine, page_body(name, width=112, height=168))
    assert status == 200
    assert set(document["choices"][0]) == {"index", "message", "finish_reason"}
    assert [entry.get("question") for entry in recorded(batches)] == [None]


def test_questions_over_the_same_images_reuse_the_vision_features(tmp_path: Path) -> None:
    weights = tmp_path / "qwen"
    weights.mkdir()
    batches = tmp_path / "batches.jsonl"
    engine = ServedFake(Path(sys.executable), tmp_path / "engine.log", batches, model_type="qwen3_5")
    engine.start(weights, str(weights), find_free_port(), ["--width", "1"])
    try:
        engine.ready(60.0)
        images = frames()
        for k in (None, 4, 4):
            status, document = post(engine, question_body(str(weights), images, k))
            assert status == 200, document
        status, _ = post(engine, question_body(str(weights), images[:1], 4))
        assert status == 200
    finally:
        engine.stop()
    seen = [entry for entry in recorded(batches) if "vision_key" in entry]
    assert [entry["cached"] for entry in seen] == [False, True, True, False]
    assert len({entry["vision_key"] for entry in seen}) == 2


def test_a_decision_about_three_images_is_answered_end_to_end_on_mlx_darwin(served) -> None:
    import asyncio
    from types import SimpleNamespace

    from crucible.engines import decide_reading
    from crucible.manifests import load_manifest

    engine, name, batches = served
    manifest = load_manifest("qwen3.5-9b-vl")
    decide_core.refuse_images_not_served("qwen3.5-9b-vl", manifest, "mlx-darwin", 3)
    request = decide_core.DecideRequest(
        model="qwen3.5-9b-vl",
        state="",
        images=frames(),
        questions={
            "face": {"type": "yesno", "instructions": "Any of these frames shows a face"},
            "expressive": {"type": "score", "instructions": "How expressive is the face?",
                           "levels": ["1", "2", "3", "4", "5"]},
            "kind": {"type": "choice", "instructions": "What is shown?",
                     "options": {"desktop": "a desktop UI", "video": "video footage"}},
        },
    )
    resident = SimpleNamespace(
        engine=manifest.spec("mlx-darwin").engine, engine_model_name=name,
        model_id="qwen3.5-9b-vl", revision=manifest.spec("mlx-darwin").revision,
        fingerprint="qwen3.5-9b-vl@" + manifest.spec("mlx-darwin").revision,
    )

    async def over_http(body: dict) -> dict:
        status, document = await asyncio.to_thread(post, engine, body)
        assert status == 200, document
        return document

    answered = asyncio.run(decide_core.decide_on_engine(
        over_http, resident, request, decide_core.plan_all(request),
        max_logprobs=decide_reading(resident.engine).max_logprobs, concurrency=4,
    ))
    assert answered.engine == "mlx-vlm"
    assert answered.tokens.images == 3
    assert answered.answers["face"].p == pytest.approx(0.6 / 0.9)
    expressive = answered.answers["expressive"]
    mass = 0.6 + 0.3 + 0.05 + 0.02 + 0.01
    assert expressive.label_mass == pytest.approx(mass)
    assert expressive.score == pytest.approx((0.6 + 0.6 + 0.15 + 0.08 + 0.05) / mass)
    assert answered.answers["kind"].choice == "desktop"
    asked = [entry for entry in recorded(batches) if entry.get("question")]
    assert len(asked) == 4
    assert all(entry["images"] == [[w, h, list(c)] for w, h, c in THREE_FRAMES] for entry in asked)
    assert sorted(entry["top_logprobs_k"] for entry in asked) == [0, 6, 6, 9]


class _ItemsReader:
    def __init__(self, refuse: bool = False) -> None:
        self.jobs: list[Any] = []
        self._refuse = refuse

    def items(self, job: Any) -> dict:
        self.jobs.append(job)
        if self._refuse:
            raise mlx_vlm_serve.ITEMS.ItemsRefusal(
                400, "item_too_long", "item 1 is 2000 tokens", {"item": 1, "tokens": 2000, "max_tokens": 1024})
        return {"object": "crucible.items", "shared_tokens": 9, "item_tokens": [1] * len(job.ask.questions),
                "slots": [{"top_logprobs": [{"token": "A", "logprob": -0.1}]} for _ in job.ask.questions]}


def _items_server(reader: _ItemsReader) -> tuple[Any, str]:
    from http.server import ThreadingHTTPServer

    batcher = mlx_vlm_serve.Batcher(reader, 1, lambda line: None)
    handler = type("ItemsHandler", (mlx_vlm_serve._Handler,), {"served": "/w", "batcher": batcher})
    server = ThreadingHTTPServer(("127.0.0.1", find_free_port()), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post_items(base: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + mlx_vlm_serve.ITEMS.ITEMS_PATH, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _items_body(images: list[str]) -> dict:
    from crucible import decide_items

    return decide_items.items_body(
        "/w", decide_items.open_messages("", images), ["Question: a?", "Question: b?"], 5, 4000)


def test_the_items_route_reads_one_job_with_its_images_and_answers_the_document() -> None:
    reader = _ItemsReader()
    server, base = _items_server(reader)
    try:
        status, document = _post_items(base, _items_body(frames()[:1]))
    finally:
        server.shutdown()
    assert status == 200, document
    assert document["object"] == "crucible.items" and len(document["slots"]) == 2
    (job,) = reader.jobs
    assert job.ask.questions == ["Question: a?", "Question: b?"]
    assert [image.size for image in job.images] == [(64, 36)]
    assert job.image_key is not None


def test_an_items_refusal_from_the_reader_is_a_400_with_its_details() -> None:
    server, base = _items_server(_ItemsReader(refuse=True))
    try:
        status, document = _post_items(base, _items_body([]))
    finally:
        server.shutdown()
    assert status == 400
    assert document["error"]["code"] == "item_too_long"
    assert document["error"]["details"] == {"item": 1, "tokens": 2000, "max_tokens": 1024}


def test_an_items_request_it_cannot_read_is_refused_before_the_reader() -> None:
    reader = _ItemsReader()
    server, base = _items_server(reader)
    try:
        status, document = _post_items(base, {**_items_body([]), "model": "other"})
        closed, closed_doc = _post_items(base, {**_items_body([]), "messages": [{"role": "user", "content": "x"}]})
    finally:
        server.shutdown()
    assert status == 404 and document["error"]["code"] == "model_not_found"
    assert closed == 400 and closed_doc["error"]["code"] == "open_user_turn"
    assert reader.jobs == []


def test_the_served_script_loads_items_forward_by_path_when_run_alone() -> None:
    probe = (
        "import sys\n"
        "import importlib.util as u\n"
        f"spec = u.spec_from_file_location('serve', r'{SERVE_SCRIPT}')\n"
        "m = u.module_from_spec(spec); sys.modules['serve'] = m; spec.loader.exec_module(m)\n"
        "print(m.ITEMS.ITEMS_PATH, m.ITEMS.__name__)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["/v1/crucible/items", "crucible_items_forward"]
