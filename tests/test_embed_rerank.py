"""The embed and rerank verbs, their pure pieces: the manifests' formats, the pick and the
package, the engine routes against fakes (llama-server's /tokenize and /v1/embeddings and
forced tokens, the Mac's items route), the vector and the score."""

from __future__ import annotations

import asyncio
import base64
import math
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import capabilityclasses, embed, manifests, rerank, verbspec, verdict
from crucible.decide_likelihood import (
    Scoring,
    score_groups_on_forced_tokens,
    score_groups_on_items,
)
from crucible.embed import EmbedRequest
from crucible.engines import embed_reading, items_forward, likelihood_reading
from crucible.engines.items_forward import ItemsRefusal
from crucible.engines.llama_server import LlamaServerEngine
from crucible.errors import ApiError
from crucible.fit import forget_cached_catalogs
from crucible.manifests import ManifestError, load_manifest
from crucible.rerank import RerankRequest

from .fake_engine import fake_token_logprob
from .test_decide_likelihood import LikelyTokenizer, likely_mlx  # noqa: F401 - a fixture

EMBED = "qwen3-embedding-8b"
RERANK = "qwen3-reranker-8b"
GIB = 1024 ** 3

QWEN_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# --- the manifests' formats ----------------------------------------------------------


def test_a_template_fills_each_slot_once_and_never_reads_a_value_for_slots() -> None:
    assert verbspec.fill("a {text} b", {"text": "{instruction} {x}"}) == "a {instruction} {x} b"


@pytest.mark.parametrize(
    "change,fragment",
    [
        ({"query": "Q:{txt}"}, "names slot(s) ['txt']"),
        ({"query": "Q:{text} {"}, "brace that is not a slot"),
        ({"document": "{text}{text}"}, "more than once"),
        ({"document": "nothing"}, "no {text} slot"),
        ({"min_dimensions": 4096}, "min_dimensions is 4096"),
        ({"pooling": "mean"}, "pooling 'mean'"),
        ({"query": "Q:{text}"}, "default_instruction is stated"),
    ],
)
def test_a_malformed_embed_table_is_refused_by_name(change: dict, fragment: str) -> None:
    table = {
        "dimensions": 4096, "min_dimensions": 32, "pooling": "last",
        "query": "Instruct: {instruction}\nQuery:{text}", "document": "{text}",
        "default_instruction": "d", "source": "s", **change,
    }
    with pytest.raises(verbspec.VerbSpecError) as caught:
        verbspec.parse_embed(table, "m.toml [embed]")
    assert fragment in str(caught.value)


def test_the_shipped_embedding_model_writes_qwen3_s_own_format() -> None:
    spec = load_manifest(EMBED).embed
    assert spec is not None and spec.dimensions == 4096 and spec.dimensions_allowed() == (32, 4096)
    assert spec.render("query", "What is the capital of China?", None) == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the "
        "query\nQuery:What is the capital of China?<|endoftext|>"
    ), "the model card's get_detailed_instruct, with the EOS its tokenizer appends"
    assert spec.render("query", "x", "Find podcasts") == "Instruct: Find podcasts\nQuery:x<|endoftext|>"
    assert spec.render("document", "The capital is Beijing.", None) == (
        "The capital is Beijing.<|endoftext|>"
    )
    assert spec.takes_instruction("query") and not spec.takes_instruction("document")


def test_the_shipped_reranker_writes_the_model_card_s_prompt_exactly() -> None:
    spec = load_manifest(RERANK).rerank
    assert spec is not None
    body = RerankRequest(query="What is the capital of China?", documents=["Beijing is."])
    scoring = rerank.scoring(body, load_manifest(RERANK))
    whole = scoring.text(0)
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    reference = (
        QWEN_RERANK_PREFIX
        + f"<Instruct>: {instruction}\n<Query>: What is the capital of China?\n"
        "<Document>: Beijing is."
        + QWEN_RERANK_SUFFIX
    )
    assert whole == reference, "prefix + format_instruction + suffix, as the model card writes them"
    assert scoring.candidates == [["yes", "no"]]
    assert scoring.prompt is not None and "Beijing" not in scoring.prompt, (
        "the instruction and the query are the shared prompt; the document is its own part"
    )


def test_a_model_is_in_the_retrieval_package_by_having_a_verb_table() -> None:
    assert load_manifest(EMBED).package == "retrieval"
    assert load_manifest(RERANK).package == "retrieval"
    assert load_manifest("qwen3.5-9b").package is None


def test_the_ladder_is_the_8b_alone_at_bf16_on_the_pc_and_the_mac() -> None:
    for model in (EMBED, RERANK):
        manifest = load_manifest(model)
        assert sorted(manifest.backends) == ["cuda-linux", "mlx-darwin"]
        for kind in manifest.backends:
            spec = manifest.spec(kind)
            assert spec.bits == 16 and not spec.forms
        assert manifest.spec("cuda-linux").engine == "llama-server"
        assert "bf16" in (manifest.spec("cuda-linux").file or "")
        assert manifest.spec("mlx-darwin").engine == "mlx-lm"
    catalog = manifests.load_all_manifests()
    assert sorted(m for m in catalog if catalog[m].package) == [EMBED, RERANK]


def test_a_manifest_with_both_verb_tables_is_refused(tmp_path: Path) -> None:
    text = (manifests.manifests_dir() / f"{EMBED}.toml").read_text(encoding="utf-8")
    rerank_table = (manifests.manifests_dir() / f"{RERANK}.toml").read_text(encoding="utf-8")
    start = rerank_table.index("[rerank]")
    end = rerank_table.index("[backends.cuda-linux]")
    both = text + "\n" + rerank_table[start:end].split("\n# ")[0]
    path = tmp_path / f"{EMBED}.toml"
    path.write_text(both, encoding="utf-8")
    with pytest.raises(ManifestError) as caught:
        manifests.parse_manifest(both, path, EMBED)
    assert "[embed] and [rerank] both" in str(caught.value)


# --- the pick and the package ---------------------------------------------------------


def _decide(name: str, kind: str, total: int, allowance: int, packages: frozenset[str],
            chosen: str | None = None) -> verdict.Decision:
    forget_cached_catalogs()
    return verdict.decide_capabilities(
        capabilityclasses.BY_NAME[name], kind, total_bytes=total,
        desktop_allowance_bytes=allowance, gpu_vendor="nvidia", chosen=chosen,
        audio_low_vram=False, packages=packages,
    )


@pytest.mark.parametrize("kind,total,allowance", [
    ("cuda-linux", 24 * GIB, 3 * GIB),
    ("mlx-darwin", 64 * GIB, 16 * GIB),
])
@pytest.mark.parametrize("verb,model", [("embed", EMBED), ("rerank", RERANK)])
def test_with_the_package_each_verb_picks_its_8b(kind: str, total: int, allowance: int,
                                                 verb: str, model: str) -> None:
    decision = _decide(verb, kind, total, allowance, frozenset({"retrieval"}))
    assert decision.enabled and decision.selected == model
    assert "goal 8B" in decision.summary


@pytest.mark.parametrize("verb", ["embed", "rerank"])
def test_without_the_package_each_verb_is_refused_naming_the_install(verb: str) -> None:
    decision = _decide(verb, "cuda-linux", 24 * GIB, 3 * GIB, frozenset())
    assert not decision.enabled and decision.selected == ""
    assert "`crucible install retrieval`" in decision.reason
    assert "retrieval package is not installed" in decision.summary


def test_rerank_on_a_decide_model_chosen_in_settings_needs_no_package() -> None:
    decision = _decide("rerank", "cuda-linux", 24 * GIB, 3 * GIB, frozenset(), chosen="qwen3.5-9b")
    assert decision.enabled and decision.selected == "qwen3.5-9b" and decision.chosen


def test_the_rerank_lineup_ranks_the_reranker_before_any_decide_model() -> None:
    entry = capabilityclasses.BY_NAME["rerank"]
    assert entry.candidates is not None
    ranked = entry.pick_order(entry.candidates("cuda-linux"))
    assert ranked[0].id == RERANK
    assert all(c.family != "qwen3-reranker" for c in ranked[1:])
    assert all(c.params_b <= 8 for c in ranked), "the goal caps the automatic pick"


def test_an_8_gib_card_with_the_package_cannot_embed_and_says_why() -> None:
    decision = _decide("embed", "cuda-linux", 8 * GIB, GIB, frozenset({"retrieval"}))
    assert not decision.enabled and "qwen3-embedding-8b" in decision.reason


# --- the request's precedence: model, ceiling, Settings, the pick ----------------------


def _host(tmp_path: Path, vram: int, packages: frozenset[str], chosen: dict[str, str] | None = None,
          recorded: bool = True) -> tuple[SimpleNamespace, SimpleNamespace]:
    from crucible.backend import Backend, Gpu

    backend = Backend(kind="cuda-linux", platform="linux", arch="x86_64",
                      gpu=Gpu(vendor="nvidia", name="card", vram_bytes=vram), detail="test")
    allowance = 3 * GIB if vram > 16 * GIB else GIB
    forget_cached_catalogs()
    decisions = verdict.decide_all(
        "cuda-linux", total_bytes=vram, desktop_allowance_bytes=allowance, gpu_vendor="nvidia",
        chosen=chosen or {}, audio_low_vram=False, packages=packages,
    )
    record = verdict.record("cuda-linux", total_bytes=vram, desktop_allowance_bytes=allowance,
                            decisions=decisions, routes={})
    config = SimpleNamespace(
        packages=packages, capability=record if recorded else None, home=tmp_path,
        desktop_allowance_bytes=allowance, audio_low_vram=False,
        local_model=lambda verb: (chosen or {}).get(verb),
    )
    return config, backend


def _resolve(verb: str, config: Any, backend: Any, **request: Any) -> Any:
    from crucible import verbmodel

    request = {"model": None, "form": None, "max_params_b": None, "resident": None, **request}
    return verbmodel.resolve(verb, config, backend, **request)


def test_a_model_that_serves_another_verb_is_refused_naming_those_that_do(tmp_path: Path) -> None:
    config, backend = _host(tmp_path, 24 * GIB, frozenset({"retrieval"}))
    with pytest.raises(ApiError) as caught:
        _resolve("embed", config, backend, model="qwen3.5-9b")
    assert caught.value.code == "model_not_for_verb" and caught.value.details["models"] == [EMBED]


def test_a_named_model_that_does_not_fit_is_refused_unless_a_person_chose_it_or_it_is_loaded(
    tmp_path: Path,
) -> None:
    packages = frozenset({"retrieval"})
    config, backend = _host(tmp_path, 8 * GIB, packages)
    with pytest.raises(ApiError) as caught:
        _resolve("embed", config, backend, model=EMBED)
    assert caught.value.code == "model_does_not_fit" and caught.value.status_code == 409
    assert _resolve("embed", config, backend, model=EMBED, resident=EMBED).model == EMBED
    chosen, backend = _host(tmp_path, 8 * GIB, packages, chosen={"embed": EMBED})
    assert _resolve("embed", chosen, backend, model=EMBED).chosen_by == "model"


def test_a_ceiling_picks_the_biggest_that_fits_at_or_below_it(tmp_path: Path) -> None:
    config, backend = _host(tmp_path, 24 * GIB, frozenset({"retrieval"}))
    assert _resolve("rerank", config, backend).model == RERANK
    under = _resolve("rerank", config, backend, max_params_b=4)
    assert under.model == "qwen3.5-4b" and under.chosen_by == "max_params_b", (
        "below the reranker's 8B the lineup goes on to the decide models"
    )
    with pytest.raises(ApiError) as caught:
        _resolve("embed", config, backend, max_params_b=4)
    assert caught.value.code == "nothing_fits_ceiling"


def test_the_registered_model_and_the_package_refusals(tmp_path: Path) -> None:
    config, backend = _host(tmp_path, 24 * GIB, frozenset({"retrieval"}))
    assert _resolve("embed", config, backend).model == EMBED
    bare, backend = _host(tmp_path, 24 * GIB, frozenset())
    for verb in ("embed", "rerank"):
        with pytest.raises(ApiError) as caught:
            _resolve(verb, bare, backend)
        assert caught.value.code == "package_not_installed"
    with pytest.raises(ApiError) as caught:
        _resolve("rerank", bare, backend, model=RERANK)
    assert caught.value.code == "package_not_installed"
    assert _resolve("rerank", bare, backend, model="qwen3.5-9b").model == "qwen3.5-9b"
    chosen, backend = _host(tmp_path, 24 * GIB, frozenset(), chosen={"rerank": "qwen3.5-9b"})
    assert _resolve("rerank", chosen, backend).model == "qwen3.5-9b", "Settings chose a decide model"
    unrecorded, backend = _host(tmp_path, 24 * GIB, frozenset({"retrieval"}), recorded=False)
    with pytest.raises(ApiError) as caught:
        _resolve("embed", unrecorded, backend)
    assert caught.value.code == "capability_undecided"
    with pytest.raises(ApiError) as caught:
        _resolve("embed", config, backend, form="f16")
    assert caught.value.code == "form_without_model"


# --- the engines' readings ------------------------------------------------------------


def test_each_engine_states_how_it_writes_vectors_and_scores_a_rendered_prompt() -> None:
    assert embed_reading("llama-server").route == "openai-embeddings"
    assert embed_reading("mlx-lm").route == "items"
    assert embed_reading("vllm").route is None and "llama-server" in embed_reading("vllm").basis
    assert embed_reading("mlx-vlm").route is None
    assert likelihood_reading("llama-server").prompt is True
    assert likelihood_reading("mlx-lm").prompt is True
    assert likelihood_reading("vllm").prompt is False
    assert likelihood_reading("mlx-vlm").prompt is False


def test_llama_server_derives_the_embedding_flags_from_the_manifest() -> None:
    manifest = load_manifest(EMBED)
    args = LlamaServerEngine.model_args(manifest, ["--parallel", "1"], Path("/w"), 8192)
    assert args == ["--parallel", "1", "--embedding", "--pooling", "last"]
    assert LlamaServerEngine.model_args(load_manifest(RERANK), ["-c", "1"], Path("/w"), 1) == ["-c", "1"]
    with pytest.raises(Exception) as caught:
        LlamaServerEngine.model_args(manifest, ["--embedding"], Path("/w"), 8192)
    assert "embed_flags_stated" in str(caught.value)


# --- the vector -----------------------------------------------------------------------


def _resident(engine: str = "llama-server", max_model_len: int = 8192) -> SimpleNamespace:
    spec = load_manifest(EMBED).spec("cuda-linux")
    return SimpleNamespace(
        model_id=EMBED, engine=engine, engine_model_name="served", revision=spec.revision,
        form=None, max_model_len=max_model_len,
    )


def _llama_embeddings(sent: list[tuple[str, dict]], spoil: Any = None) -> Any:
    async def call(path: str, wire: dict) -> Any:
        sent.append((path, wire))
        if path == "/tokenize":
            assert wire["add_special"] is False and wire["parse_special"] is True
            text = wire["content"]
            assert text.endswith("<|endoftext|>")
            return {"tokens": [ord(c) for c in text[: -len("<|endoftext|>")]] + [151643]}
        assert path == "/v1/embeddings"
        data = [
            {"index": i, "object": "embedding",
             "embedding": [float(len(ids)), 3.0, 4.0] + [0.0] * 4093}
            for i, ids in enumerate(wire["input"])
        ]
        reply = {"object": "list", "data": data, "usage": {"prompt_tokens": 1}}
        return reply if spoil is None else spoil(reply)

    return call


def _embed(body: EmbedRequest, sent: list, spoil: Any = None, **resident: Any) -> embed.EmbedResponse:
    manifest = load_manifest(EMBED)
    assert manifest.embed is not None
    model = embed.EmbedModel(
        id=EMBED, revision="r", file="f", form=None, bits=16, engine="llama-server",
        engine_build="b", scheme=1, fingerprint="fp", dimensions=4096,
    )
    return asyncio.run(embed.embed_on_engine(
        _llama_embeddings(sent, spoil), _resident(**resident), "openai-embeddings", body,
        manifest.embed, model,
    ))


def test_llama_server_is_sent_token_ids_it_never_adds_to_and_normalises_nothing() -> None:
    sent: list[tuple[str, dict]] = []
    body = EmbedRequest(inputs=["ab", "c"], input_type="document")
    answer = _embed(body, sent)
    assert [path for path, _ in sent] == ["/tokenize", "/tokenize", "/v1/embeddings"]
    wire = sent[-1][1]
    assert wire["input"] == [[97, 98, 151643], [99, 151643]] and wire["embd_normalize"] == -1
    assert answer.tokens.per_input == [3, 2] and answer.tokens.total == 5
    first = answer.embeddings[0]
    assert len(first) == 4096 and math.fsum(v * v for v in first) == pytest.approx(1.0)
    assert first[:3] == pytest.approx([3 / math.sqrt(34), 3 / math.sqrt(34), 4 / math.sqrt(34)])
    assert answer.instruction is None and answer.input_type == "document"


def test_a_shorter_vector_is_the_prefix_normalised_again() -> None:
    body = EmbedRequest(inputs=["abcd"], input_type="query", dimensions=2, instruction="Find it")
    answer = _embed(body, [])
    assert answer.dimensions == 2 and answer.instruction == "Find it"
    length = len("Instruct: Find it\nQuery:abcd") + 1
    assert answer.embeddings[0] == pytest.approx([length / math.hypot(length, 3), 3 / math.hypot(length, 3)])


def test_an_input_past_the_context_is_refused_before_anything_is_embedded() -> None:
    sent: list[tuple[str, dict]] = []
    with pytest.raises(ApiError) as caught:
        _embed(EmbedRequest(inputs=["ok", "x" * 20], input_type="document"), sent, max_model_len=10)
    assert caught.value.code == "embed_input_too_long"
    assert caught.value.details == {"input": 1, "tokens": 21, "max_tokens": 10}
    assert all(path == "/tokenize" for path, _ in sent)


@pytest.mark.parametrize(
    "spoil,fragment",
    [
        (lambda r: {**r, "data": r["data"][:-1]}, "a list of 2 embeddings"),
        (lambda r: {**r, "data": list(reversed(r["data"]))}, "is not embedding 0"),
        (lambda r: {**r, "data": [{**d, "embedding": [0.0] * 4096} for d in r["data"]]}, "no length"),
        (lambda r: {**r, "data": [{**d, "embedding": [1.0] * 10} for d in r["data"]]}, "writes 4096"),
    ],
)
def test_an_engine_reply_that_is_not_the_vectors_asked_for_is_engine_error(spoil: Any, fragment: str) -> None:
    with pytest.raises(ApiError) as caught:
        _embed(EmbedRequest(inputs=["a", "b"], input_type="document"), [], spoil)
    assert caught.value.code == "engine_error" and fragment in caught.value.message


def test_the_encodings_are_float32_and_float16_little_endian_base64() -> None:
    vectors = [[0.6, 0.8]]
    (f32,) = embed.encoded(vectors, "base64")
    assert struct.unpack("<2f", base64.b64decode(f32)) == pytest.approx((0.6, 0.8))
    (f16,) = embed.encoded(vectors, "base64_float16")
    assert len(base64.b64decode(f16)) == 4
    assert struct.unpack("<2e", base64.b64decode(f16)) == pytest.approx((0.6, 0.8), abs=1e-3)


@pytest.mark.parametrize(
    "body,code",
    [
        ({"input_type": "document", "instruction": "x"}, "instruction_not_taken"),
        ({"input_type": "query", "dimensions": 16}, "dimensions_not_supported"),
        ({"input_type": "query", "dimensions": 4097}, "dimensions_not_supported"),
    ],
)
def test_what_the_model_s_format_says_no_to_is_refused_by_name(body: dict, code: str) -> None:
    spec = load_manifest(EMBED).embed
    assert spec is not None
    with pytest.raises(ApiError) as caught:
        embed.refuse_unfit_request(EmbedRequest(inputs=["a"], **body), spec, EMBED)
    assert caught.value.code == code


def test_the_fingerprint_names_everything_that_changes_a_float() -> None:
    resident = _resident()
    identity = embed.identity(resident, load_manifest(EMBED), "cuda-linux")
    assert identity.file == "qwen3-embedding-8b-bf16.gguf" and identity.bits == 16
    assert identity.engine_build.startswith("llama-server-b10970")
    assert identity.fingerprint == (
        f"{EMBED}@{resident.revision[:12]}:qwen3-embedding-8b-bf16.gguf:"
        f"{identity.engine_build}:e{embed.EMBED_SCHEME}"
    )
    mac = embed.engine_build("mlx-lm", "mlx-darwin")
    assert mac.startswith("mlx-lm-0.31.3+mlx-")
    embed.refuse_other_fingerprint(identity.fingerprint, identity)
    with pytest.raises(ApiError) as caught:
        embed.refuse_other_fingerprint(identity.fingerprint[:-1] + "0", identity)
    assert caught.value.code == "fingerprint_mismatch" and caught.value.status_code == 409


def test_a_fingerprint_names_the_model_and_conflicts_are_refused() -> None:
    catalog = manifests.load_all_manifests()
    body = EmbedRequest(inputs=["a"], input_type="query", fingerprint=f"{EMBED}@abc:file:b:e1")
    assert embed.named_by_fingerprint(body, catalog, "cuda-linux") == embed.Named(EMBED, None)
    with pytest.raises(ApiError) as caught:
        embed.named_by_fingerprint(
            EmbedRequest(inputs=["a"], input_type="query", fingerprint="nope@x:y:z:e1"),
            catalog, "cuda-linux",
        )
    assert caught.value.code == "fingerprint_unknown"
    with pytest.raises(ApiError) as caught:
        embed.named_by_fingerprint(
            EmbedRequest(inputs=["a"], input_type="query", model="qwen3.5-9b",
                         fingerprint=f"{EMBED}@abc:file:b:e1"),
            catalog, "cuda-linux",
        )
    assert caught.value.code == "fingerprint_conflict"


def test_an_embedding_model_is_refused_at_the_chat_and_decide_doors() -> None:
    with pytest.raises(ApiError) as caught:
        embed.refuse_vectors_only(EMBED, "chat")
    assert caught.value.code == "model_embeds_only"
    embed.refuse_vectors_only(RERANK, "chat")
    embed.refuse_vectors_only("an-upstream/model", "chat")


# --- the Mac's items route: vectors and the prompt form --------------------------------


class _Picked(list):
    def astype(self, dtype: Any) -> "_Picked":
        return self

    def tolist(self) -> list:
        return [list(row) for row in self]


class _Hidden:
    """Each row's context so far; read at (rows, lasts), a vector that says which tokens
    the position had seen: [tokens seen, sum of them, the token there]."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def __getitem__(self, index: Any) -> _Picked:
        rows, lasts = index
        return _Picked(
            [float(last + 1), float(sum(self.rows[r][: last + 1])), float(self.rows[r][last])]
            for r, last in zip(rows, lasts)
        )


@pytest.fixture
def embed_mlx(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    core = types.ModuleType("mlx.core")
    core.array = lambda value: value
    core.arange = lambda n: list(range(n))
    core.eval = lambda *args: None
    core.clear_cache = lambda: None
    core.float32 = "float32"
    mlx = types.ModuleType("mlx")
    mlx.core = core
    cache_module = types.ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = lambda model: []
    for name, module in {
        "mlx": mlx, "mlx.core": core, "mlx_lm": types.ModuleType("mlx_lm"),
        "mlx_lm.models": types.ModuleType("mlx_lm.models"), "mlx_lm.models.cache": cache_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    forwards: list[tuple[int, int]] = []

    def inner(ids: list[list[int]], cache: Any) -> _Hidden:
        forwards.append((len(ids), len(ids[0])))
        return _Hidden(ids)

    model = SimpleNamespace(model=inner, lm_head=lambda h: h)
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: [ord(c) for c in text],
    )
    assert not hasattr(tokenizer, "apply_chat_template")
    return SimpleNamespace(forwards=forwards, provider=SimpleNamespace(
        load=lambda *a: (model, tokenizer),
        cli_args=SimpleNamespace(prefill_step_size=2048),
    ))


def _embed_job(inputs: list[str], cap: int = 100) -> items_forward.MlxLmItemsJob:
    return items_forward.MlxLmItemsJob(items_forward.parse_request(
        {"model": "w", "inputs": inputs, "max_input_tokens": cap}, ("w",)
    ))


def test_the_mac_reads_each_vector_at_the_input_s_own_last_token(embed_mlx: Any) -> None:
    document = items_forward.mlx_lm_answer(embed_mlx.provider, _embed_job(["abc", "z", "hi"]))
    assert document["object"] == "crucible.embeddings" and document["dimensions"] == 3
    for text, row in zip(["abc", "z", "hi"], document["data"]):
        ids = [ord(c) for c in text]
        assert row["tokens"] == len(ids)
        assert row["embedding"] == [float(len(ids)), float(sum(ids)), float(ids[-1])], (
            f"{text!r} is read at its own last token, never the padding after it"
        )
    assert embed_mlx.forwards == [(3, 3)], "every input a row of one right-padded forward"


def test_the_mac_refuses_an_input_past_the_context_before_any_forward(embed_mlx: Any) -> None:
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.mlx_lm_answer(embed_mlx.provider, _embed_job(["ok", "x" * 9], cap=5))
    assert caught.value.code == "embed_input_too_long"
    assert caught.value.details == {"input": 1, "tokens": 9, "max_tokens": 5}
    assert embed_mlx.forwards == []


def test_an_embed_request_the_route_cannot_read_is_refused_by_name() -> None:
    for body, code in [
        ({"model": "w", "inputs": [], "max_input_tokens": 5}, "bad_inputs"),
        ({"model": "w", "inputs": ["a"], "max_input_tokens": 5, "messages": []}, "unknown_field"),
        ({"model": "w", "inputs": ["a"]}, "bad_max_input_tokens"),
    ]:
        with pytest.raises(ItemsRefusal) as caught:
            items_forward.parse_request(body, ("w",))
        assert caught.value.code == code


class _PromptTokenizer(LikelyTokenizer):
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_the_prompt_form_reads_the_query_once_and_each_document_s_tail_once(
    likely_mlx: Any,  # noqa: F811
) -> None:
    model, _ = likely_mlx.provider.load()
    likely_mlx.provider.load = lambda *a: (model, _PromptTokenizer())
    job = items_forward.MlxLmItemsJob(items_forward.parse_request({
        "model": "w", "prompt": "PQ|",
        "candidates": [{"question": "d1>", "texts": ["y", "n"]},
                       {"question": "doc2>", "texts": ["y", "n"]}],
        "max_prompt_tokens": 100, "max_item_tokens": 100, "max_candidate_tokens": 10,
    }, ("w",)))
    document = items_forward.mlx_lm_answer(likely_mlx.provider, job)
    assert likely_mlx.model.forwards == [(1, 3), (1, 3), (1, 5)], (
        "the prompt (the query) once as the state, then each document's tail once; a "
        "one-token reply needs no forward of its own"
    )
    for question, group in zip(["d1>", "doc2>"], document["groups"]):
        whole = [ord(c) for c in "PQ|" + question]
        assert group["boundary"] == len(whole)
        for text, row in zip("yn", group["candidates"]):
            assert row["logprobs"] == pytest.approx([fake_token_logprob(whole, ord(text))])
    assert [g["read_tokens"] for g in document["groups"]] == [3 + 3, 5], (
        "the query read once, with the first document, then each document's own tail"
    )

    # Through the door: each document's yes and no are each its 6- or 8-token prompt, as
    # llama-server is sent them (28), and all but the 11 tokens read are cached (the
    # Mac's 2026-10-10 rerank reported its query once per document and 0 cached).
    async def call(path: str, wire: dict) -> Any:
        assert path == items_forward.ITEMS_PATH
        return document

    scoring = Scoring(messages=None, prompt="PQ|", questions=["d1>", "doc2>"],
                      candidates=[["y", "n"], ["y", "n"]])
    resident = SimpleNamespace(engine="mlx-lm", engine_model_name="w", max_model_len=8192)
    groups = asyncio.run(score_groups_on_items(call, resident, scoring))
    model = rerank.RerankModel(id=RERANK, revision="r", file=None, form=None, engine="mlx-lm",
                               engine_build="b", template="model", fingerprint="f")
    tokens = rerank.answer(groups, model, "i", 0.0).tokens
    assert (tokens.total, tokens.cached) == (6 * 2 + 8 * 2, 6 * 2 + 8 * 2 - 11)
    assert tokens.per_document == [6, 8]


def test_a_likelihood_request_is_the_chat_form_or_the_prompt_form_never_both() -> None:
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.parse_request({
            "model": "w", "prompt": "p", "messages": [{"role": "user", "content": ""}],
            "candidates": [{"question": "q", "texts": ["a"]}],
            "max_prompt_tokens": 1, "max_item_tokens": 1, "max_candidate_tokens": 1,
        }, ("w",))
    assert caught.value.code == "prompt_and_messages"


# --- the score ------------------------------------------------------------------------


def _llama_forced(sent: list[tuple[str, dict]]) -> Any:
    async def call(path: str, wire: dict) -> Any:
        sent.append((path, wire))
        if path == "/tokenize":
            assert wire["add_special"] is False and wire["parse_special"] is True
            return {"tokens": [ord(c) for c in wire["content"]]}
        assert path == "/completion"
        prompt, grammar = list(wire["prompt"]), wire["grammar"]
        forced = [int(t) for t in grammar.removeprefix("root ::= ").replace("<[", "").replace("]>", "").split()]
        return {
            "tokens": forced,
            "completion_probabilities": [
                {"id": t, "logprob": fake_token_logprob(prompt + forced[:i], t)}
                for i, t in enumerate(forced)
            ],
            "timings": {"cache_n": 0},
        }

    return call


def test_llama_server_scores_the_prompt_form_with_no_template_and_nothing_added() -> None:
    sent: list[tuple[str, dict]] = []
    scoring = Scoring(messages=None, prompt="P:", questions=["a>", "bb>"],
                      candidates=[["y", "n"], ["y", "n"]])
    resident = SimpleNamespace(engine="llama-server", max_model_len=8192)
    groups = asyncio.run(score_groups_on_forced_tokens(_llama_forced(sent), resident, scoring))
    paths = [path for path, _ in sent]
    assert "/apply-template" not in paths
    assert paths == ["/tokenize"] * 6 + ["/completion"] * 4, "every check before any forward"
    completions = [wire for path, wire in sent if path == "/completion"]
    assert [wire["prompt"] for wire in completions] == [
        [ord(c) for c in "P:a>"]] * 2 + [[ord(c) for c in "P:bb>"]] * 2, (
        "documents in turn, each document's yes and no over the same cached context"
    )
    for group, question in zip(groups, ["a>", "bb>"]):
        context = [ord(c) for c in "P:" + question]
        yes, no = (fake_token_logprob(context, ord(t)) for t in "yn")
        assert rerank.relevance(group) == pytest.approx(math.exp(yes) / (math.exp(yes) + math.exp(no)))


def test_llama_server_reports_every_prompt_it_was_sent_and_what_its_slot_reused() -> None:
    sent: list[tuple[str, dict]] = []
    plain = _llama_forced(sent)
    last: list[int] = []

    async def call(path: str, wire: dict) -> Any:
        data = await plain(path, wire)
        if path == "/completion":
            prompt = list(wire["prompt"])
            common = 0
            while common < min(len(prompt), len(last)) and prompt[common] == last[common]:
                common += 1
            # The slot re-reads at least the prompt's last token, as llama-server does.
            data["timings"] = {"cache_n": min(common, len(prompt) - 1)}
            last[:] = prompt
        return data

    scoring = Scoring(messages=None, prompt="P:", questions=["a>", "bb>"],
                      candidates=[["y", "n"], ["y", "n"]])
    resident = SimpleNamespace(engine="llama-server", max_model_len=8192)
    groups = asyncio.run(score_groups_on_forced_tokens(call, resident, scoring))
    assert [g.prompt_tokens for g in groups] == [4 * 2, 5 * 2]
    assert [g.timing.cached_tokens for g in groups] == [0 + 3, 2 + 4], (
        "the first prompt read whole; its no reuses 3; the next document reuses the "
        "shared 'P:' and then 4 of its own 5"
    )


def test_relevance_is_p_yes_against_p_no_and_results_sort_most_relevant_first() -> None:
    def group(yes: float, no: float) -> Any:
        return SimpleNamespace(rows=[[yes], [no]], context_tokens=10, prompt_tokens=20,
                               timing=SimpleNamespace(cached_tokens=5))

    groups = [group(math.log(0.2), math.log(0.6)), group(math.log(0.9), math.log(0.05)),
              group(math.log(0.5), math.log(0.5))]
    model = rerank.RerankModel(id=RERANK, revision="r", file=None, form=None, engine="e",
                               engine_build="b", template="model", fingerprint="f")
    answer = rerank.answer(groups, model, "i", 0.0)
    assert answer.scores == pytest.approx([0.25, 0.9 / 0.95, 0.5])
    assert [r.index for r in answer.results] == [1, 2, 0]
    assert answer.tokens.total == 60 and answer.tokens.cached == 15


def test_a_decide_model_is_judged_with_crucible_s_general_template_in_the_chat_form() -> None:
    body = RerankRequest(query="q?", documents=["d1", "d2"], instruction="Find answers")
    scoring = rerank.scoring(body, load_manifest("qwen3.5-9b"))
    assert scoring.prompt is None and scoring.messages is not None
    system = scoring.messages[0]["content"]
    assert system.endswith("<Instruct>: Find answers\n<Query>: q?"), "the shared part is the system turn"
    assert scoring.messages[1] == {"role": "user", "content": ""}
    assert scoring.questions == ["<Document>: d1", "<Document>: d2"]


def test_a_reranker_on_an_engine_that_scores_chat_turns_only_is_refused() -> None:
    reading = likelihood_reading("vllm")
    with pytest.raises(ApiError) as caught:
        rerank.refuse_unscorable(RERANK, load_manifest(RERANK), "vllm", reading)
    assert caught.value.code == "rerank_unsupported_on_engine"
    rerank.refuse_unscorable("qwen3.5-9b", load_manifest("qwen3.5-9b"), "vllm", reading)


def test_a_refusal_about_a_group_names_the_document() -> None:
    error = rerank.by_document(ApiError(400, "item_prompt_too_long", "too long", {"group": 3}))
    assert error.details["document"] == 3 and error.message.startswith("document 3:")
