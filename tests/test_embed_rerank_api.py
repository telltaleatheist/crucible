"""The embed and rerank doors end to end: native and compatible routes, the package, the
model's name on every answer, against a resident engine faked on the wire (httpx
MockTransport), as tests/test_api_thin.py fakes the decision door's."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import verdict
from crucible.api import create_app
from crucible.capabilitystore import record_of
from crucible.config import load_config, rewrite_config
from crucible.errors import ApiError
from crucible.manifests import NO_DEFAULTS, load_manifest
from crucible.residency import Residency

from .conftest import FAKE_BACKEND, configure_box
from .fake_engine import fake_prompt_ids, fake_token_logprob

EMBED = "qwen3-embedding-8b"
RERANK = "qwen3-reranker-8b"
EOS = "<|endoftext|>"


def _record(packages: frozenset[str]) -> Any:
    decisions = verdict.decide_all(
        FAKE_BACKEND.kind,
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * 1024 ** 3,
        gpu_vendor=FAKE_BACKEND.gpu.vendor,
        chosen={},
        audio_low_vram=False,
        packages=packages,
    )
    return record_of(
        FAKE_BACKEND.kind,
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * 1024 ** 3,
        decisions=decisions,
        routes={},
    )


def _resident(model: str, engine: str, args: list[str]) -> SimpleNamespace:
    spec = load_manifest(model).spec(FAKE_BACKEND.kind)
    return SimpleNamespace(
        model_id=model, engine=engine, engine_args=args, engine_model_name="served",
        base_url="http://engine.invalid", log_path="/tmp/engine.log", revision=spec.revision,
        fingerprint=f"{model}@{spec.revision}", form=None, max_model_len=8192,
        defaults=NO_DEFAULTS,
    )


LLAMA = ["--parallel", "1"]
VLLM = ["--max-num-seqs", "4"]


def _engine(seen: list[tuple[str, dict]]) -> Callable[[httpx.Request], httpx.Response]:
    """llama-server's /tokenize, /v1/embeddings and /completion, and vLLM's /tokenize and
    prompt-logprobs chat, each answering from the request alone."""

    def answer(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content)
        seen.append((path, body))
        if path == "/tokenize" and "content" in body:
            text = body["content"]
            ids = [ord(c) for c in text.removesuffix(EOS)] + ([151643] if text.endswith(EOS) else [])
            return httpx.Response(200, json={"tokens": ids})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": fake_prompt_ids(body)})
        if path == "/v1/embeddings":
            data = [
                {"index": i, "object": "embedding", "embedding": [float(len(ids)), 1.0] + [0.0] * 4094}
                for i, ids in enumerate(body["input"])
            ]
            return httpx.Response(200, json={"object": "list", "data": data})
        if path == "/completion":
            prompt = list(body["prompt"])
            forced = [int(t) for t in body["grammar"].removeprefix("root ::= ")
                      .replace("<[", "").replace("]>", "").split()]
            return httpx.Response(200, json={
                "tokens": forced,
                "completion_probabilities": [
                    {"id": t, "logprob": fake_token_logprob(prompt + forced[:i], t)}
                    for i, t in enumerate(forced)
                ],
                "timings": {"cache_n": len(prompt) - 2},
            })
        assert path == "/v1/chat/completions", path
        ids = fake_prompt_ids(body)
        entries: list[Any] = [None] + [
            {str(ids[p]): {"logprob": fake_token_logprob(ids[:p], ids[p])}} for p in range(1, len(ids))
        ]
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": len(ids), "prompt_tokens_details": {"cached_tokens": 0}},
            "prompt_logprobs": entries, "prompt_token_ids": ids,
        })

    return answer


@pytest.fixture
def served(
    home: Path, fake_env: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., tuple[TestClient, list]]]:
    clients: list[TestClient] = []

    def start(resident: SimpleNamespace, *, package: bool = True) -> tuple[TestClient, list]:
        packages = frozenset({"retrieval"}) if package else frozenset()
        configure_box(home, enable_llm=True, capability=_record(packages))
        if package:
            rewrite_config(load_config(home), unowned={"packages": {"retrieval": True}})
        monkeypatch.setattr(Residency, "resident_model", property(lambda self: resident))
        monkeypatch.setattr(Residency, "engine_exit_code", property(lambda self: None))
        client = TestClient(create_app(load_config(home), FAKE_BACKEND))
        client.__enter__()
        clients.append(client)
        seen: list[tuple[str, dict]] = []
        client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(_engine(seen)))
        return client, seen

    yield start
    for client in clients:
        client.__exit__(None, None, None)


def _unit(vector: list[float]) -> float:
    return math.fsum(v * v for v in vector)


def test_embed_answers_unit_vectors_and_names_what_wrote_them(
    served: Any, auth: dict[str, str]
) -> None:
    client, seen = served(_resident(EMBED, "llama-server", LLAMA))
    response = client.post("/v1/embed", headers=auth, json={
        "inputs": ["what is it", "a"], "input_type": "query", "instruction": "Find it",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "crucible.embeddings" and body["dimensions"] == 4096
    model = body["model"]
    assert model["id"] == EMBED and model["file"] == "qwen3-embedding-8b-bf16.gguf"
    assert model["fingerprint"].startswith(f"{EMBED}@") and model["fingerprint"].endswith(":e1")
    assert all(_unit(v) == pytest.approx(1.0) for v in body["embeddings"])
    assert body["instruction"] == "Find it" and body["input_type"] == "query"
    assert body["timing_ms"]["queued"] is not None
    tokenized = [b["content"] for p, b in seen if p == "/tokenize"]
    assert tokenized == [f"Instruct: Find it\nQuery:what is it{EOS}", f"Instruct: Find it\nQuery:a{EOS}"]


def test_a_stored_fingerprint_is_served_exactly_or_refused(served: Any, auth: dict[str, str]) -> None:
    client, seen = served(_resident(EMBED, "llama-server", LLAMA))
    first = client.post("/v1/embed", headers=auth, json={"inputs": ["a"], "input_type": "document"})
    fingerprint = first.json()["model"]["fingerprint"]
    again = client.post("/v1/embed", headers=auth, json={
        "inputs": ["b"], "input_type": "document", "fingerprint": fingerprint,
    })
    assert again.status_code == 200, again.text
    before = len(seen)
    stale = client.post("/v1/embed", headers=auth, json={
        "inputs": ["b"], "input_type": "document", "fingerprint": fingerprint[:-1] + "0",
    })
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "fingerprint_mismatch"
    assert len(seen) == before, "nothing was embedded"


def test_the_openai_shape_on_both_mounts(served: Any, auth: dict[str, str]) -> None:
    client, _ = served(_resident(EMBED, "llama-server", LLAMA))
    for path in ("/v1/openai/embeddings", "/openai/v1/embeddings"):
        response = client.post(path, headers=auth, json={"input": "hello", "encoding_format": "base64"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list" and body["model"] == EMBED
        (row,) = body["data"]
        assert row["object"] == "embedding" and row["index"] == 0 and isinstance(row["embedding"], str)
        assert body["crucible"]["input_type"] == "document" and body["crucible"]["dimensions"] == 4096
        assert body["usage"]["prompt_tokens"] == len("hello") + 1


@pytest.mark.parametrize(
    "request_body,status,code",
    [
        ({"inputs": ["a"], "input_type": "query", "model": "qwen3.5-9b"}, 400, "model_not_for_verb"),
        ({"inputs": ["a"], "input_type": "document", "instruction": "x"}, 400, "instruction_not_taken"),
        ({"inputs": ["a"], "input_type": "query", "dimensions": 8}, 400, "dimensions_not_supported"),
        ({"inputs": ["a"]}, 400, "invalid_request"),
    ],
)
def test_embed_refuses_by_name(
    served: Any, auth: dict[str, str], request_body: dict, status: int, code: str
) -> None:
    client, seen = served(_resident(EMBED, "llama-server", LLAMA))
    response = client.post("/v1/embed", headers=auth, json=request_body)
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert seen == []


def test_without_the_package_embed_is_refused_naming_the_install(
    served: Any, auth: dict[str, str]
) -> None:
    client, seen = served(_resident(EMBED, "llama-server", LLAMA), package=False)
    for body in ({"inputs": ["a"], "input_type": "query"},
                 {"inputs": ["a"], "input_type": "query", "model": EMBED}):
        response = client.post("/v1/embed", headers=auth, json=body)
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "package_not_installed"
        assert "crucible install retrieval" in error["message"]
    assert seen == []


def test_rerank_scores_every_document_with_the_reranker_s_own_prompt(
    served: Any, auth: dict[str, str]
) -> None:
    client, seen = served(_resident(RERANK, "llama-server", LLAMA))
    response = client.post("/v1/rerank", headers=auth, json={
        "query": "capital of China?", "documents": ["Beijing.", "Paris.", "A cat."],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"]["id"] == RERANK and body["model"]["template"] == "model"
    assert len(body["scores"]) == 3 and all(0.0 < s < 1.0 for s in body["scores"])
    assert [r["index"] for r in body["results"]] == sorted(
        range(3), key=lambda i: (-body["scores"][i], i)
    )
    assert "/apply-template" not in [p for p, _ in seen], "the prompt form: no chat template"
    first = next(b["content"] for p, b in seen if p == "/tokenize")
    assert first.startswith("<|im_start|>system\nJudge whether the Document")
    assert first.endswith("<Document>: Beijing.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_the_compatible_rerank_is_cohere_shaped(served: Any, auth: dict[str, str]) -> None:
    client, _ = served(_resident(RERANK, "llama-server", LLAMA))
    response = client.post("/openai/v1/rerank", headers=auth, json={
        "query": "q", "documents": ["one", {"text": "two"}], "top_n": 1, "return_documents": True,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    (result,) = body["results"]
    assert set(result) == {"index", "relevance_score", "document"}
    assert result["document"]["text"] == ["one", "two"][result["index"]]
    assert result["relevance_score"] == max(body["crucible"]["scores"])
    assert body["model"] == RERANK and body["crucible"]["model"]["fingerprint"]


def test_a_decide_model_reranks_without_the_package(served: Any, auth: dict[str, str]) -> None:
    client, seen = served(_resident("qwen3.5-9b", "vllm", VLLM), package=False)
    refused = client.post("/v1/rerank", headers=auth, json={"query": "q", "documents": ["d"]})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "package_not_installed"
    assert "name a decide model" in refused.json()["error"]["message"]
    response = client.post("/v1/rerank", headers=auth, json={
        "query": "q", "documents": ["d1", "d2"], "model": "qwen3.5-9b",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"]["template"] == "crucible-general-1"
    scored = [b for p, b in seen if p == "/v1/chat/completions"]
    assert len(scored) == 4 and all(b["prompt_logprobs"] == 0 for b in scored)
    assert scored[0]["messages"][-1] == {"role": "assistant", "content": "yes"}


def test_an_embedding_model_is_refused_at_the_chat_door(served: Any, auth: dict[str, str]) -> None:
    client, seen = served(_resident(EMBED, "llama-server", LLAMA))
    response = client.post("/v1/openai/chat/completions", headers=auth, json={
        "model": EMBED, "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 400 and response.json()["error"]["code"] == "model_embeds_only"
    assert seen == []


def test_the_models_and_the_info_say_who_serves_the_verbs(served: Any, auth: dict[str, str]) -> None:
    client, _ = served(_resident(EMBED, "llama-server", LLAMA))
    rows = {row["id"]: row for row in client.get("/v1/models", headers=auth).json()}
    embed_row = rows[EMBED]
    assert embed_row["verbs"] == ["embed"] and embed_row["package"] == "retrieval"
    assert embed_row["package_installed"] is True
    assert embed_row["embed"]["dimensions"] == 4096 and embed_row["embed"]["dimensions_range"] == [32, 4096]
    assert embed_row["embed"]["max_inputs"] == 256 and embed_row["embed"]["max_input_tokens"] == 8192
    assert rows[RERANK]["rerank"]["template"] == "model"
    assert "rerank" in rows["qwen3.5-9b"]["verbs"] and rows["qwen3.5-9b"]["package"] is None
    assert rows["qwen3.5-9b"]["rerank"]["template"] == "crucible-general-1"
    info = client.get("/v1/info", headers=auth).json()
    assert {"embed", "rerank", "embed.openai", "rerank.compat"} <= set(info["features"])
    assert info["verbs"]["embed"]["registered"] == EMBED and info["verbs"]["embed"]["available"]
    assert info["verbs"]["rerank"]["models"][0] == RERANK


def test_a_load_of_a_package_model_is_refused_where_the_package_is_not_installed(
    home: Path, fake_env: Path
) -> None:
    from crucible.jobs.llm import _require_loadable

    configure_box(home, enable_llm=True)
    with pytest.raises(ApiError) as caught:
        _require_loadable(load_config(home), FAKE_BACKEND, EMBED)
    assert caught.value.code == "package_not_installed"
