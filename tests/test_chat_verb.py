"""Verb-addressed chat with a size ceiling (docs/VERB-SIZING.md rule 8, phase 2's door): a
chat, a queue session and a load may name a verb and a `max_params_b` ceiling instead of
a model. The ceiling caps this server's own pick for the verb and never raises it (Owen,
2026-10-11). Every pick here is the catalog's estimate against each card's budget."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import verbmodel, verdict
from crucible.api import create_app
from crucible.backend import Backend, Gpu
from crucible.capabilitystore import record_of
from crucible.config import load_config
from crucible.errors import ApiError
from crucible.fit import forget_cached_catalogs
from crucible.manifests import load_manifest
from crucible.residency import Residency

from .conftest import FAKE_BACKEND, configure_box

GIB = 1024 ** 3

# The three hosts of the B-Sides case: the PC (24 GiB, its stated 3 GiB reserve), the Mac
# Studio (64 GiB unified, the 25% reserve) and Victoria's laptop (8 GiB, the scaled 1 GiB).
HOSTS = {
    "pc": ("cuda-linux", 24 * GIB, 3 * GIB),
    "mac": ("mlx-darwin", 64 * GIB, 16 * GIB),
    "victoria": ("cuda-linux", 8 * GIB, GIB),
}


def _host(
    tmp_path: Path, name: str, chosen: dict[str, str] | None = None, recorded: bool = True
) -> tuple[SimpleNamespace, Backend]:
    kind, vram, allowance = HOSTS[name]
    backend = Backend(kind=kind, platform="linux" if kind == "cuda-linux" else "darwin",
                      arch="x86_64" if kind == "cuda-linux" else "arm64",
                      gpu=Gpu(vendor="nvidia" if kind == "cuda-linux" else "apple",
                              name="card", vram_bytes=vram), detail="test")
    forget_cached_catalogs()
    decisions = verdict.decide_all(
        kind, total_bytes=vram, desktop_allowance_bytes=allowance,
        gpu_vendor=backend.gpu.vendor, chosen=chosen or {}, audio_low_vram=False,
        packages=frozenset(),
    )
    record = verdict.record(kind, total_bytes=vram, desktop_allowance_bytes=allowance,
                            decisions=decisions, routes={})
    config = SimpleNamespace(
        packages=frozenset(), capability=record if recorded else None, home=tmp_path,
        desktop_allowance_bytes=allowance, audio_low_vram=False,
        local_model=lambda verb: (chosen or {}).get(verb),
    )
    return config, backend


def _resolve(verb: str, config: Any, backend: Any, **request: Any) -> verbmodel.VerbModel:
    request = {"model": None, "form": None, "max_params_b": None, "resident": None, **request}
    return verbmodel.resolve(verb, config, backend, **request)


# --- the B-Sides case: the biggest chat model at most 4B --------------------------------


@pytest.mark.parametrize("host,registered,picked,chosen_by", [
    ("pc", "qwen3.8-27b-4bit", "qwen3.5-4b", "max_params_b"),
    ("mac", "qwen3.8-27b-8bit", "qwen3.5-4b", "max_params_b"),
    ("victoria", "qwen3.5-4b-8bit", "qwen3.5-4b-8bit", "registered"),
])
def test_each_host_serves_a_4b_ceiling_with_its_biggest_4b(
    tmp_path: Path, host: str, registered: str, picked: str, chosen_by: str
) -> None:
    config, backend = _host(tmp_path, host)
    assert _resolve("generate", config, backend).model == registered
    chosen = _resolve("generate", config, backend, max_params_b=4)
    assert (chosen.model, chosen.chosen_by) == (picked, chosen_by)
    bits = load_manifest(chosen.model).spec(backend.kind).bits
    assert bits == (8 if host == "victoria" else 16), (
        "16-bit wherever the 4B fits whole; the 8-bit only where it does not"
    )


def test_a_pick_at_or_below_the_ceiling_is_kept_and_never_raised(tmp_path: Path) -> None:
    # Owen 2026-10-11: "victoria's is 8 bit 4b ... that's the one chat will always use".
    config, backend = _host(tmp_path, "victoria")
    for ceiling in (4, 9, 27, 100):
        chosen = _resolve("generate", config, backend, max_params_b=ceiling)
        assert (chosen.model, chosen.chosen_by) == ("qwen3.5-4b-8bit", "registered")


def test_a_settings_choice_below_the_ceiling_is_kept(tmp_path: Path) -> None:
    config, backend = _host(tmp_path, "pc", chosen={"generate": "qwen3.5-2b"})
    chosen = _resolve("generate", config, backend, max_params_b=9)
    assert (chosen.model, chosen.chosen_by) == ("qwen3.5-2b", "registered"), (
        "a server configured smaller than the ceiling keeps its model; the ceiling never "
        "selects the 9B past it"
    )


def test_a_settings_choice_above_the_ceiling_is_replaced_by_the_biggest_under_it(
    tmp_path: Path,
) -> None:
    config, backend = _host(tmp_path, "pc", chosen={"generate": "qwen3.5-9b"})
    assert _resolve("generate", config, backend).model == "qwen3.5-9b"
    chosen = _resolve("generate", config, backend, max_params_b=4)
    assert (chosen.model, chosen.chosen_by) == ("qwen3.5-4b", "max_params_b")


def test_the_ceiling_refusals(tmp_path: Path) -> None:
    config, backend = _host(tmp_path, "pc")
    with pytest.raises(ApiError) as caught:
        _resolve("generate", config, backend, max_params_b=0.5)
    assert caught.value.code == "nothing_fits_ceiling" and caught.value.status_code == 409
    with pytest.raises(ApiError) as caught:
        _resolve("generate", config, backend, model="qwen3.5-4b", max_params_b=4)
    assert caught.value.code == "model_with_ceiling" and caught.value.status_code == 400
    unrecorded, backend = _host(tmp_path, "pc", recorded=False)
    with pytest.raises(ApiError) as caught:
        _resolve("generate", unrecorded, backend, max_params_b=4)
    assert caught.value.code == "capability_undecided", (
        "a ceiling caps this server's pick, so a server with none says so"
    )


def test_the_verb_checks_name_what_is_wrong() -> None:
    for verb, code in (("lyrics", "unknown_verb"), ("tts", "verb_not_sized")):
        with pytest.raises(ApiError) as caught:
            verbmodel.refuse_unsized_verb(verb, "this request's verb")
        assert caught.value.code == code
    verbmodel.refuse_unsized_verb("decide", "this request's verb")
    for verb, code in (("lyrics", "unknown_verb"), ("embed", "verb_not_for_chat")):
        with pytest.raises(ApiError) as caught:
            verbmodel.refuse_unchattable_verb(verb)
        assert caught.value.code == code
    verbmodel.refuse_unchattable_verb("translate")


def test_a_raw_body_s_verb_and_ceiling_are_taken_off_it() -> None:
    body: dict[str, Any] = {"verb": "generate", "max_params_b": 4, "messages": []}
    assert verbmodel.take_verb(body) == "generate" and verbmodel.take_ceiling(body) == 4.0
    assert body == {"messages": []}
    assert verbmodel.take_verb({"verb": None}) is None
    for bad in ({"max_params_b": 0}, {"max_params_b": True}, {"max_params_b": "4"}):
        with pytest.raises(ApiError) as caught:
            verbmodel.take_ceiling(dict(bad))
        assert caught.value.code == "invalid_request"
    with pytest.raises(ApiError):
        verbmodel.take_verb({"verb": ""})


# --- the doors, end to end, against a resident engine faked on the wire ------------------


def _record() -> Any:
    decisions = verdict.decide_all(
        FAKE_BACKEND.kind, total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * GIB, gpu_vendor="nvidia", chosen={},
        audio_low_vram=False, packages=frozenset(),
    )
    return record_of(FAKE_BACKEND.kind, total_bytes=FAKE_BACKEND.gpu.vram_bytes,
                     desktop_allowance_bytes=3 * GIB, decisions=decisions, routes={})


def _resident(model: str) -> SimpleNamespace:
    spec = load_manifest(model).spec(FAKE_BACKEND.kind)
    return SimpleNamespace(
        model_id=model, engine=spec.engine, engine_args=list(spec.engine_args),
        engine_model_name="served", base_url="http://engine.invalid",
        log_path="/tmp/engine.log", revision=spec.revision,
        fingerprint=f"{model}@{spec.revision}", form=None, max_model_len=8192,
        defaults=load_manifest(model).defaults,
    )


@pytest.fixture
def served(
    home: Path, fake_env: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[str], tuple[TestClient, list]]]:
    clients: list[TestClient] = []

    def start(model: str) -> tuple[TestClient, list]:
        configure_box(home, enable_llm=True, capability=_record())
        resident = _resident(model)
        monkeypatch.setattr(Residency, "resident_model", property(lambda self: resident))
        monkeypatch.setattr(Residency, "engine_exit_code", property(lambda self: None))
        client = TestClient(create_app(load_config(home), FAKE_BACKEND))
        client.__enter__()
        clients.append(client)
        seen: list[dict] = []

        def engine(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(200, json={
                "id": "c1", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": '{"a":1}'}}],
            })

        client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(engine))
        return client, seen

    yield start
    for client in clients:
        client.__exit__(None, None, None)


SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "a", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}},
}


def test_a_chat_with_a_ceiling_and_no_model_is_served_exactly_as_one_naming_it(
    served: Any, auth: dict[str, str]
) -> None:
    client, seen = served("qwen3.5-4b")
    messages = [{"role": "user", "content": "hi"}]
    for path in ("/v1/openai/chat/completions", "/openai/v1/chat/completions"):
        by_ceiling = client.post(path, headers=auth, json={
            "max_params_b": 4, "messages": messages, "response_format": SCHEMA,
            "json_whitespace": "compact",
        })
        assert by_ceiling.status_code == 200, by_ceiling.text
        assert by_ceiling.json()["model"] == "qwen3.5-4b"
        by_name = client.post(path, headers=auth, json={
            "model": "qwen3.5-4b", "messages": messages, "response_format": SCHEMA,
            "json_whitespace": "compact",
        })
        assert by_name.status_code == 200 and by_name.json()["model"] == "qwen3.5-4b"
        assert seen[-2] == seen[-1], "the engine is sent the same body either way"
        assert "verb" not in seen[-1] and "max_params_b" not in seen[-1]
        assert seen[-1]["model"] == "served"
    null_model = client.post("/v1/openai/chat/completions", headers=auth, json={
        "model": None, "verb": "generate", "max_params_b": 4, "messages": messages,
    })
    assert null_model.status_code == 200 and null_model.json()["model"] == "qwen3.5-4b"


def test_a_chat_with_a_ceiling_whose_model_is_not_resident_is_refused_naming_it(
    served: Any, auth: dict[str, str]
) -> None:
    client, seen = served("qwen3.5-9b")
    response = client.post("/v1/openai/chat/completions", headers=auth, json={
        "max_params_b": 4, "messages": [{"role": "user", "content": "hi"}], "queue": False,
    })
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "model_not_resident" and "qwen3.5-4b" in error["message"]
    assert seen == []


def test_the_chat_door_refuses_a_mixed_or_unknown_choice_by_name(
    served: Any, auth: dict[str, str]
) -> None:
    client, seen = served("qwen3.5-4b")
    messages = [{"role": "user", "content": "hi"}]
    for body, code in (
        ({"model": "qwen3.5-4b", "max_params_b": 4}, "model_with_ceiling"),
        ({"verb": "lyrics"}, "unknown_verb"),
        ({"verb": "embed"}, "verb_not_for_chat"),
        ({"max_params_b": 0.5}, "nothing_fits_ceiling"),
        ({"max_params_b": "4"}, "invalid_request"),
        ({"model": ""}, "model_required"),
        ({"form": "bf16", "max_params_b": 4}, "form_without_model"),
        ({"model": "qwen3.5-4b-bside", "verb": "generate"}, "model_not_for_verb"),
    ):
        response = client.post("/v1/openai/chat/completions", headers=auth,
                               json={**body, "messages": messages})
        assert response.json()["error"]["code"] == code, (body, response.text)
    assert seen == []


def test_a_verb_with_no_ceiling_is_served_by_the_registered_model(
    served: Any, auth: dict[str, str]
) -> None:
    client, _ = served("qwen3.8-27b-4bit")
    response = client.post("/v1/openai/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "qwen3.8-27b-4bit"


def test_a_session_opens_on_the_ceiling_s_model_and_its_ticket_names_it(
    served: Any, auth: dict[str, str]
) -> None:
    client, _ = served("qwen3.5-4b")
    refused = client.post("/v1/queue/sessions", headers=auth,
                          json={"act": "tts", "max_params_b": 4})
    assert refused.json()["error"]["code"] == "verb_not_sized"
    mixed = client.post("/v1/queue/sessions", headers=auth,
                        json={"act": "generate", "model": "qwen3.5-4b", "max_params_b": 4})
    assert mixed.json()["error"]["code"] == "model_with_ceiling"
    ticket = client.post("/v1/queue/sessions", headers=auth,
                         json={"act": "generate", "max_params_b": 4})
    assert ticket.status_code == 202, ticket.text
    assert ticket.json()["model"] == "qwen3.5-4b"
    state = client.get(f"/v1/queue/sessions/{ticket.json()['session_id']}", headers=auth).json()
    assert state["model"] == "qwen3.5-4b"
    client.delete(f"/v1/queue/sessions/{ticket.json()['session_id']}", headers=auth)


def test_the_info_lists_the_chat_verbs_and_the_feature(served: Any, auth: dict[str, str]) -> None:
    client, _ = served("qwen3.5-4b")
    info = client.get("/v1/info", headers=auth).json()
    assert "chat.verbs" in info["features"]
    generate = info["verbs"]["generate"]
    assert generate["route"] == "POST /v1/openai/chat/completions"
    assert generate["registered"] == "qwen3.8-27b-4bit" and generate["available"]
    assert generate["limits"] == {} and "qwen3.5-4b" in generate["models"]
    assert set(info["verbs"]) >= {"embed", "rerank", "generate", "clean", "translate",
                                  "simplify", "analysis"}


def test_a_load_names_a_verb_and_a_ceiling_in_place_of_a_model(
    home: Path, fake_env: Path
) -> None:
    from crucible.jobs import resolve_model
    from crucible.jobs.llm import LoadModelJobType

    configure_box(home, enable_llm=True, capability=_record())
    residency = SimpleNamespace(resident_model=None, is_resident=lambda kind, model: False)
    loads = LoadModelJobType(load_config(home), FAKE_BACKEND, residency)
    assert loads.choose_model(None, {"max_params_b": 4}) == "qwen3.5-4b"
    assert loads.choose_model(None, {"verb": "decide"}) == "qwen3.5-9b"
    assert loads.choose_model("qwen3.5-2b", {}) == "qwen3.5-2b"
    for model, params, code in (
        (None, {}, "model_required"),
        ("qwen3.5-4b", {"max_params_b": 4}, "model_with_ceiling"),
        (None, {"verb": "tts"}, "verb_not_sized"),
        (None, {"verb": "generate", "max_params_b": 0.5}, "nothing_fits_ceiling"),
        (None, {"max_params_b": 4, "form": "bf16"}, "form_without_model"),
    ):
        with pytest.raises(ApiError) as caught:
            loads.choose_model(model, params)
        assert caught.value.code == code, (model, params)
    assert resolve_model(loads, None, {"verb": "generate", "max_params_b": 4}) == "qwen3.5-4b"
