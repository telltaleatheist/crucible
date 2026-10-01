from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from crucible import decide as decide_core
from crucible import uploads, voicerepo
from crucible.admission import AdmissionContext, AdmittedJob, JobRequest, Refusal, admit
from crucible.api import context as context_module
from crucible.api import sse
from crucible.api.schemas import JobInput
from crucible.errors import ApiError
from crucible.events import EventHub
from crucible.jobs import disabled_error

API_DIR = Path(__file__).resolve().parent.parent / "crucible" / "api"


def admission_context(app: FastAPI) -> AdmissionContext:
    state = app.state
    return AdmissionContext(
        config=state.config,
        store=state.store,
        residency=state.residency,
        sessions=state.sessions,
        installs=state.installs,
        decide_here=lambda job_type: ApiError(409, "job_type_disabled", job_type),
    )


def inline(payload: bytes) -> dict[str, JobInput]:
    return {"x.bin": JobInput(inline_base64=base64.b64encode(payload).decode("ascii"))}


def test_admission_admits_an_echo_job_without_http(client: TestClient) -> None:
    ctx = admission_context(client.app)
    outcome = client.portal.call(
        admit, JobRequest(type="echo", inputs=inline(b"hello"), client="unit"), ctx
    )
    assert isinstance(outcome, AdmittedJob)
    receipt = outcome.receipt()
    assert set(receipt) == {"job_id", "resume_id"}
    job = ctx.store.get(receipt["job_id"])
    assert job.client == "unit"
    assert (job.inputs_dir / "x.bin").read_bytes() == b"hello"


def test_admission_refuses_by_name_instead_of_raising(client: TestClient) -> None:
    ctx = admission_context(client.app)
    unknown = client.portal.call(admit, JobRequest(type="sorcery"), ctx)
    assert isinstance(unknown, Refusal)
    assert unknown.error.status_code in (400, 404)

    upstream = client.portal.call(
        admit, JobRequest(type="echo", model="openai/gpt-x", inputs=inline(b"x")), ctx
    )
    assert isinstance(upstream, Refusal)
    assert upstream.error.code == "upstream_never_resident"


def test_a_bad_input_discards_the_job_it_created(client: TestClient) -> None:
    ctx = admission_context(client.app)
    before = len(ctx.store.queued())
    bad = {"x.bin": JobInput(blob_id="no-such-blob")}
    outcome = client.portal.call(admit, JobRequest(type="echo", inputs=bad), ctx)
    assert isinstance(outcome, Refusal)
    assert outcome.error.code == "unknown_blob"
    assert len(ctx.store.queued()) == before
    assert ctx.store.running is None


def test_the_upload_store_writes_the_sidecar_the_digest_reads(tmp_path: Path) -> None:
    payload = b"a" * (uploads.UPLOAD_CHUNK + 5)
    chunks = [payload[: uploads.UPLOAD_CHUNK], payload[uploads.UPLOAD_CHUNK:], b""]

    async def read(size: int) -> bytes:
        assert size == uploads.UPLOAD_CHUNK
        return chunks.pop(0)

    blob = asyncio.run(uploads.store_upload(tmp_path, read, "voice.wav"))
    assert blob.receipt() == {
        "blob_id": blob.blob_id,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert uploads.blob_path(tmp_path, blob.blob_id).read_bytes() == payload
    assert uploads.recorded_sha256(tmp_path, blob.blob_id, len(payload)) == blob.sha256
    assert uploads.recorded_sha256(tmp_path, blob.blob_id, len(payload) - 1) is None


def test_decide_on_engine_runs_with_an_injected_post() -> None:
    sent: list[dict[str, Any]] = []

    async def post(body: dict[str, Any]) -> Any:
        sent.append(body)
        return {
            "usage": {"prompt_tokens": 12},
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "A", "logprob": math.log(0.75)},
                                    {"token": "B", "logprob": math.log(0.25)},
                                ]
                            }
                        ]
                    }
                }
            ],
        }

    request = decide_core.DecideRequest(
        model="m", state="it is raining", questions={"wet": {"type": "yesno", "instructions": "Is it wet?"}}
    )
    resident = SimpleNamespace(
        engine="vllm", engine_model_name="m", model_id="m", revision="r", fingerprint="f"
    )
    answered = asyncio.run(
        decide_core.decide_on_engine(
            post, resident, request, decide_core.plan_all(request),
            max_logprobs=None, concurrency=1,
        )
    )
    assert len(sent) == 1
    assert sent[0]["model"] == "m"
    assert answered.answers["wet"].p == pytest.approx(0.75)
    assert answered.tokens.per_question == {"wet": 12}


def test_the_decide_refusals_live_in_the_domain() -> None:
    resident = SimpleNamespace(engine="vllm", model_id="m")
    request = decide_core.DecideRequest(
        model="m",
        state="s",
        questions={"q": {"type": "choice", "instructions": "i", "options": {"a": "1", "b": "2", "c": "3"}}},
    )
    plans = decide_core.plan_all(request)
    with pytest.raises(ApiError) as capped:
        decide_core.refuse_unreadable_labels(
            resident, SimpleNamespace(served=True, basis="b", max_logprobs=2), plans
        )
    assert capped.value.code == "decide_not_served"
    assert capped.value.details["max_logprobs"] == 2
    decide_core.refuse_unreadable_labels(
        resident, SimpleNamespace(served=True, basis="b", max_logprobs=None), plans
    )

    text_only = SimpleNamespace(serves=lambda kind: ("text",), modalities=("text",))
    with pytest.raises(ApiError) as images:
        decide_core.refuse_images_not_served("m", text_only, "cuda-linux", 2)
    assert images.value.code == "model_text_only"


def test_repin_is_a_domain_function_with_the_hub_injected(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    with pytest.raises(ApiError) as not_a_table:
        voicerepo.repin(None, "v", "owner/repo")
    assert not_a_table.value.code == "voice_invalid"
    with pytest.raises(ApiError) as extra:
        voicerepo.repin(None, "v", {"hf_repo": "o/r", "display": "x"})
    assert "display" in extra.value.message

    asked: list[str] = []
    monkeypatch.setattr(voicerepo, "voice_for_pin", lambda pin: None)
    written = voicerepo.repin(
        None, "v", {"hf_repo": "o/r"}, resolve=lambda _c, repo: asked.append(repo) or "c" * 40
    )
    assert asked == ["o/r"]
    assert written.revision == "c" * 40


def test_pinned_backends_resolves_only_what_is_missing() -> None:
    block = {
        "backends": {
            "cuda-linux": {"hf_repo": "o/r"},
            "mlx-darwin": {"hf_repo": "o/r", "revision": "d" * 40},
            "local": {"path": "/x", "revision": None},
        }
    }
    pinned = voicerepo.pinned_backends(None, block, resolve=lambda _c, _r: "e" * 40)
    assert pinned["backends"]["cuda-linux"]["revision"] == "e" * 40
    assert pinned["backends"]["mlx-darwin"]["revision"] == "d" * 40
    assert "revision" not in pinned["backends"]["local"]


def test_every_error_envelope_is_the_api_error_body(
    client: TestClient, auth: dict[str, str]
) -> None:
    missing = client.get("/v1/no-such-route", headers=auth)
    assert missing.status_code == 404
    assert missing.json() == ApiError(404, "not_found", "Not Found").body()

    invalid = client.post("/v1/jobs", json={"type": 3}, headers=auth)
    assert invalid.status_code == 400
    body = invalid.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "invalid_request"
    assert set(body["error"]) == {"code", "message", "details"}


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/v1/voices", None),
        ("GET", "/v1/voices/deathstalker/manifest", None),
        ("PUT", "/v1/voices/deathstalker", {"pin": {"hf_repo": "o/r"}}),
        ("DELETE", "/v1/voices/deathstalker", None),
        ("POST", "/v1/tts/stream", {"voice": "deathstalker", "language": "en"}),
    ],
)
def test_one_tts_gate_guards_every_voice_door(
    client: TestClient, auth: dict[str, str], method: str, path: str, body: Any
) -> None:
    refused = disabled_error("tts", client.app.state.config)
    response = client.request(method, path, json=body, headers=auth)
    assert response.status_code == refused.status_code, response.text
    assert response.json() == refused.body()


def test_routes_reach_services_through_the_typed_context_only() -> None:
    for source in sorted((API_DIR / "routes").glob("*.py")) + [
        API_DIR / "upstream.py",
        API_DIR / "sse.py",
        API_DIR / "proxy.py",
    ]:
        text = source.read_text(encoding="utf-8")
        assert "app.state" not in text, source.name
        assert not re.search(r"\blive: Config\b", text), source.name


def test_the_context_has_a_typed_reader_for_every_published_service(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app()
    names = [field.name for field in context_module.Services.__dataclass_fields__.values()]
    for name in names:
        assert isinstance(getattr(context_module.AppContext, name), property), name
        assert hasattr(app.state, name), name


def test_a_service_swapped_on_app_state_is_what_the_routes_see(
    client: TestClient, auth: dict[str, str]
) -> None:
    class Nobody:
        def current(self) -> None:
            return None

    real = client.app.state.sessions
    client.app.state.sessions = Nobody()
    try:
        response = client.get("/v1/activity", headers=auth)
        assert response.status_code == 200
        assert response.json()["session"] is None
    finally:
        client.app.state.sessions = real


def test_no_header_name_is_spelled_in_the_api_package() -> None:
    for source in sorted(API_DIR.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        assert "X-Crucible" not in text and "x-crucible" not in text, source.name


def imported_names(text: str) -> list[str]:
    names: list[str] = []
    for grouped in re.findall(r"^from \.[\w.]* import \(([^)]*)\)", text, re.MULTILINE):
        names += grouped.split(",")
    for single in re.findall(r"^from \.[\w.]* import ([^(\n]+)$", text, re.MULTILINE):
        names += single.split(",")
    return [name.split(" as ")[0].strip() for name in names if name.strip()]


def test_no_shared_name_crosses_a_module_with_an_underscore() -> None:
    for source in sorted(API_DIR.rglob("*.py")):
        if source.name in ("__init__.py", "inputs.py", "upstream.py"):
            continue
        private = [n for n in imported_names(source.read_text(encoding="utf-8")) if n.startswith("_")]
        assert private == [], f"{source.name} imports {private}"


def test_one_events_after_loop_serves_a_job_and_ends_at_its_terminal_event() -> None:
    events = [
        {"id": 1, "event": "progress", "data": {"fraction": 0.5}},
        {"id": 2, "event": "done", "data": {}},
        {"id": 3, "event": "never", "data": {}},
    ]
    closed: list[bool] = []
    feed = sse.Feed(
        after=lambda index: [(i + 1, e) for i, e in enumerate(events) if i >= index],
        waiter=asyncio.Event(),
        ends=sse.TERMINAL_EVENTS,
        moved=lambda _: None,
        close=lambda: closed.append(True),
    )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(events=EventHub())))

    async def collect() -> list[str]:
        return [frame async for frame in sse.events_after(request, lambda: feed, 0)]

    frames = asyncio.run(collect())
    assert frames == [sse.format_event(events[0]), sse.format_event(events[1])]
    assert closed == [True]


def test_the_decide_route_posts_through_the_injected_engine_call(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from crucible.residency import Residency

    resident = SimpleNamespace(
        model_id="m", engine="vllm", engine_args=["--max-num-seqs", "4"], engine_model_name="served-m",
        base_url="http://engine.invalid", log_path="/tmp/engine.log",
        revision="r", fingerprint="f",
    )
    monkeypatch.setattr(Residency, "resident_model", property(lambda self: resident))
    monkeypatch.setattr(Residency, "engine_exit_code", property(lambda self: None))
    seen: list[dict[str, Any]] = []

    def engine(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        top = [{"token": "A", "logprob": math.log(0.9)}, {"token": "B", "logprob": math.log(0.1)}]
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 7},
                "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
            },
        )

    real = client.app.state.http
    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    try:
        response = client.post(
            "/v1/decide",
            json={"model": "m", "state": "s", "questions": {"q": {"type": "yesno", "instructions": "i"}}},
            headers=auth,
        )
    finally:
        client.app.state.http = real
    assert response.status_code == 200, response.text
    assert seen[0]["model"] == "served-m"
    assert response.json()["answers"]["q"]["p"] == pytest.approx(0.9)
