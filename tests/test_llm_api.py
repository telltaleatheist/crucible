"""`load-model`, `unload-model`, `/v1/models` and the OpenAI proxy.

No GPU and no 19 GB of weights — the env, the weights and the engine are stood
up by the fixtures in conftest.py exactly as the real code paths read them. What
is *not* faked is any of the server's own logic: the preflight refusals, the
exclusive lane, the event stream and the proxy are exactly what runs on the PC.

The one thing this file cannot ask is what happens when the caller goes away
mid-request; `TestClient` has no such state. That is tests/test_proxy_disconnect.py.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator
from crucible.accelerator import GIB, ComputeApp
from crucible.jobs.llm import residency as residency_module
from crucible.manifests import load_manifest

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse
from .fake_engine import ANSWER, DELTAS, TOOL_CALL, FakeEngine

MODEL = "qwen3.5-9b"
#: The page reader, which sorts first by id and so leads every listing.
PAGE_MODEL = "dots-ocr"
BIG_MODEL = "qwen3.8-27b"
#: The same 27B at 4 bits: the one that does fit Owen's card.
SMALL_BIG_MODEL = "qwen3.8-27b-4bit"


# ------------------------------------------------------------------ fixtures


#: A card big enough for the 27B, so residency can be tested with two real
#: manifests rather than a contrived pair.
ROOMY_BACKEND = replace(
    FAKE_BACKEND,
    gpu=replace(FAKE_BACKEND.gpu, name="NVIDIA H100 80GB HBM3", vram_bytes=80 * GIB),
)


@pytest.fixture
def roomy_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (80 * GIB, 80 * GIB))


@pytest.fixture
def roomy_client(
    make_client: Callable[..., TestClient], fake_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_llm=True, backend=ROOMY_BACKEND) as client:
        yield client


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> list[FakeEngine]:
    """Every engine the residency builds, in order, so a test can inspect them."""
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    return built


@pytest.fixture
def llm_client(
    make_client: Callable[..., TestClient], fake_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_llm=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    return client.post("/v1/jobs", headers=auth, json=body)


def _stream_chunks(text: str) -> list[dict[str, Any]]:
    """Every `chat.completion.chunk` of a streamed completion, in order.

    Asserts the stream ended on OpenAI's terminator on the way past, because a
    chunk list read out of a truncated stream would quietly be a shorter one.
    """
    lines = [line for line in text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]", lines[-3:]
    return [json.loads(line[len("data: ") :]) for line in lines[:-1]]


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    """Submit a job and read its whole event stream. Fails loudly if refused."""
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream(
        "GET", f"/v1/jobs/{job_id}/events", headers=auth
    ) as stream:
        return parse_sse(line for line in stream.iter_lines())


# -------------------------------------------------------------- /v1/models


def test_models_lists_every_manifest_with_its_standing(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.get("/v1/models", headers=auth)
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}
    assert [row["id"] for row in response.json()] == [
        PAGE_MODEL, MODEL, BIG_MODEL, SMALL_BIG_MODEL,
    ]
    row = rows[MODEL]
    assert row["family"] == "qwen3.5"
    assert row["params_b"] == 9
    # The pin for *this* host's backend, verbatim from the manifest: a client
    # that records what it talked to records the same sha the puller used.
    assert row["revision"] == load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert row["backend_supported"] is True
    assert row["installed"] is False
    assert row["resident"] is False
    assert row["loadable"] is False
    assert row["context_default"] == 12288
    assert row["max_model_len"] == 12288
    assert row["memory_bytes_estimate"] > 0
    assert "no weights at" in row["reason"]


def test_models_says_loadable_once_the_weights_are_there(
    llm_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(MODEL)
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[MODEL]["installed"] is True
    assert rows[MODEL]["loadable"] is True
    assert "reason" not in rows[MODEL]
    # The 27B has a manifest and is supported here; it is simply not pulled.
    assert rows[BIG_MODEL]["installed"] is False


def test_models_is_refused_when_llm_is_off(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_llm=False) as client:
        response = client.get("/v1/models", headers=auth)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "job_type_disabled"


def test_info_gains_an_llm_capability(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    capabilities = llm_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert "llm" in by_type
    assert [row["id"] for row in by_type["llm"]["models"]] == [
        PAGE_MODEL, MODEL, BIG_MODEL, SMALL_BIG_MODEL,
    ]
    # The two things you can actually POST are listed as themselves.
    assert "load-model" in by_type
    assert "unload-model" in by_type


def test_the_llm_capability_rows_are_the_models_rows(
    llm_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    """One shape, one producer: `/info`'s llm rows *are* `/v1/models`' rows."""
    fake_weights(MODEL)
    models = llm_client.get("/v1/models", headers=auth).json()
    capabilities = llm_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == models
    for row in models:
        assert row["revision"] == load_manifest(row["id"]).spec(
            FAKE_BACKEND.kind
        ).revision


def test_a_model_this_backend_cannot_serve_has_no_revision(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`revision` is the pin for *this* backend, so a model with no block here
    reports null — never the other backend's sha, and never an empty string that
    would read as a pin."""
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        rows = client.get("/v1/models", headers=auth).json()
        capabilities = client.get("/v1/info", headers=auth).json()["capabilities"]
    assert [row["id"] for row in rows] == ["mac-only"]
    assert rows[0]["backend_supported"] is False
    assert rows[0]["revision"] is None
    assert rows[0]["memory_bytes_estimate"] is None
    # And no max_model_len either: this host would not serve it at any context.
    # `context_default` still answers, because the model's own number is a fact
    # about the model rather than about a backend block that is not there.
    assert rows[0]["max_model_len"] is None
    assert rows[0]["context_default"] == 4096
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == rows


# ---------------------------------------------------------------- fingerprint


def test_every_row_carries_the_fingerprint_a_client_records(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    """`<id>@<revision>` — the id alone does not identify bytes.

    Foundry hashes the served model id into its cleanup cache key and BookForge
    stamps it into a book's OPF (CLIENT-SURFACES.md section 6.5). The server
    spells the fingerprint out rather than leaving each client to assemble one,
    because two clients inventing two spellings is two names for one set of
    weights.
    """
    rows = llm_client.get("/v1/models", headers=auth).json()
    for row in rows:
        assert row["fingerprint"] == f"{row['id']}@{row['revision']}"
    # By id, not by position: the rows are sorted by manifest stem, so which one
    # is first changes the moment a manifest is added — `dots-ocr` took the slot
    # from `qwen3.5-9b` the day page reading landed.
    row = next(r for r in rows if r["id"] == MODEL)
    assert row["fingerprint"] == (
        f"{MODEL}@{load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision}"
    )


def test_a_model_this_backend_cannot_serve_has_no_fingerprint(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null, not the bare id: an unpinned fingerprint would look like a pin."""
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        row = client.get("/v1/models", headers=auth).json()[0]
    assert row["revision"] is None
    assert row["fingerprint"] is None


def test_the_openai_listing_names_the_weights_the_engine_actually_read(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """That entry describes the engine, so its pin is the loaded one."""
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    revision = load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert entry["revision"] == revision
    assert entry["fingerprint"] == f"{MODEL}@{revision}"


def test_a_provenance_sidecar_names_the_revision_it_was_served_at(
    make_app: Callable[..., Any],
    fake_env: Path,
) -> None:
    """The bug: `revision: null` on every artifact Crucible had ever written.

    The queue holds a model id and nothing that could turn it into a revision, so
    it wrote null and a sidecar named a model while declining to say which one.
    The job type knows; it is asked.

    No `llm` job produces artifacts today, so this reads the document the way the
    artifact writer does rather than fetching a file. That is the point of fixing
    it now: `tts` and `vlm-pages` are the ones that will write it into a book.
    """
    store = make_app(enable_llm=True).state.store
    spec = load_manifest(MODEL).spec(FAKE_BACKEND.kind)

    job = store.create("load-model", MODEL, {})
    assert store.provenance(job)["model"] == {
        "id": MODEL,
        "revision": spec.revision,
        "fingerprint": f"{MODEL}@{spec.revision}",
    }

    # A model-less job type still says `model: null`, which is the honest shape.
    assert store.provenance(store.create("echo", None, {}))["model"] is None


def test_a_provenance_sidecar_names_THIS_host_s_pin(
    make_app: Callable[..., Any],
    fake_env: Path,
) -> None:
    """The same model at two shas, because the weights differ per backend.

    `qwen3.5-9b` is one Crucible id over two HuggingFace repos — Qwen's own on
    cuda-linux, the bf16 conversion on mlx-darwin. A record that named the id
    without the host's pin would say the same thing about two different sets of
    bytes, which is exactly what the fingerprint exists to prevent.
    """
    mac_store = make_app(enable_llm=True, backend=FAKE_MAC_BACKEND).state.store
    mac = mac_store.provenance(mac_store.create("load-model", MODEL, {}))["model"]
    pc_store = make_app(enable_llm=True).state.store
    pc = pc_store.provenance(pc_store.create("load-model", MODEL, {}))["model"]

    assert mac["id"] == pc["id"] == MODEL
    assert mac["revision"] == load_manifest(MODEL).spec(FAKE_MAC_BACKEND.kind).revision
    assert pc["revision"] == load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert mac["fingerprint"] != pc["fingerprint"]


# ------------------------------------------------------------- max_model_len


def test_the_openai_listing_reports_the_context_the_engine_was_started_with(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """`/v1/openai/models` is the door Foundry reads, so it carries the number.

    Without it `capFor` has no clamp at all and the request goes out unsized
    (CLIENT-SURFACES.md section 6.1).
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    assert entry["id"] == MODEL
    assert entry["max_model_len"] == 12288
    # The same number vLLM was handed as --max-model-len.
    assert engines[0].args[-2:] == ["--max-model-len", "12288"]


def test_max_model_len_follows_the_engine_and_context_default_follows_the_manifest(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The one moment the two fields disagree, and which one is which.

    A manifest edited while its engine is up: `context_default` is the manifest's
    intent and moves with the file, `max_model_len` is what is being served and
    stays with the engine. Re-deriving `max_model_len` from the manifest would
    have this row promise a 32768-token context to a client talking to an engine
    that was started at 8192 — and the client would size a request against it and
    be refused by the engine.
    """
    fixture = tmp_path / "models"
    fixture.mkdir()
    manifest = fixture / "shifty.toml"

    def write(context: int) -> None:
        manifest.write_text(
            f"""
[model]
id = "shifty"
family = "demo"
params_b = 1
context_default = {context}
modalities = ["text"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
            encoding="utf-8",
        )

    write(8192)
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        spec = load_manifest("shifty", fixture).spec(FAKE_BACKEND.kind)
        weights_dir = home / "models" / "shifty" / FAKE_BACKEND.kind
        weights_dir.mkdir(parents=True)
        (weights_dir / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "model": "shifty",
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 3_000_000_000,
                    "seconds": 1.0,
                    "pulled": "2026-09-12T19:00:00+0000",
                }
            ),
            encoding="utf-8",
        )
        events = run_job(client, auth, type="load-model", model="shifty")
        assert events[-1]["event"] == "done", events[-1]
        assert engines[0].args == ["--max-model-len", "8192"]

        write(32768)  # somebody edits the manifest with the engine still up
        row = client.get("/v1/models", headers=auth).json()[0]
        assert row["resident"] is True
        assert row["context_default"] == 32768, "the manifest's intent moved"
        assert row["max_model_len"] == 8192, "what is being served did not"
        entry = client.get("/v1/openai/models", headers=auth).json()["data"][0]
        assert entry["max_model_len"] == 8192


# -------------------------------------------------- the refusals before queuing


def test_an_unknown_model_is_refused_by_name(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(llm_client, auth, type="load-model", model="llama9000")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


def test_a_model_with_no_block_for_this_backend_is_backend_unsupported(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        response = submit(client, auth, type="load-model", model="mac-only")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "backend_unsupported"
    assert "mlx-darwin" in error["message"]


def test_a_missing_env_is_env_missing(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(MODEL)
    with make_client(enable_llm=True) as client:  # no fake_env fixture here
        response = submit(client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "env_missing"
    assert "crucible install llm" in error["message"]


def test_missing_weights_are_model_not_installed(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_installed"
    assert "crucible models pull qwen3.5-9b" in error["message"]


def test_weights_at_the_wrong_revision_are_not_installed(
    llm_client: TestClient,
    auth: dict[str, str],
    home: Path,
) -> None:
    """A manifest that moved must not serve yesterday's bytes under today's id."""
    directory = home / "models" / MODEL / FAKE_BACKEND.kind
    directory.mkdir(parents=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "backend": FAKE_BACKEND.kind,
                "hf_repo": "Qwen/Qwen3.5-9B",
                "revision": "f" * 40,
                "bytes": 1,
                "pulled": "2026-01-01T00:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_installed"
    assert "now pins" in response.json()["error"]["message"]


def test_a_busy_card_is_refused_before_the_job_exists(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    monkeypatch.setattr(
        accelerator,
        "probe_compute_apps",
        lambda: [ComputeApp(pid=12769, name="sgl-omni", used_bytes=17 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (7 * GIB, 24 * GIB))
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "accelerator_busy"
    assert "sgl-omni" in error["message"]
    # Refused *before* queuing: nothing was created.
    assert llm_client.get("/v1/health", headers=auth).json()["queue_depth"] == 0


def test_the_27b_on_this_card_is_insufficient_memory(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(BIG_MODEL)
    response = submit(llm_client, auth, type="load-model", model=BIG_MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == 56_368_313_144
    assert error["details"]["total_bytes"] == FAKE_BACKEND.gpu.vram_bytes


def test_a_model_too_big_for_the_card_is_refused_before_the_download(
    llm_client: TestClient, auth: dict[str, str], idle_card: None
) -> None:
    """52.5 GiB on a 24 GiB card is not a "pull 55 GB first" problem.

    The weights are deliberately NOT stamped here: a refusal that says
    `model_not_installed` would send somebody off to download 55 GB for a model
    that can never load on this host.
    """
    response = submit(llm_client, auth, type="load-model", model=BIG_MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "insufficient_memory"
    assert "ever" in error["message"]
    assert "52.5 GiB" in error["message"]
    assert "24.0 GiB in total" in error["message"]
    assert "NVIDIA GeForce RTX 3090 Ti" in error["message"]


def test_models_says_why_the_27b_is_not_loadable_here(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[BIG_MODEL]["loadable"] is False
    assert "52.5 GiB" in rows[BIG_MODEL]["reason"]
    assert "24.0 GiB in total" in rows[BIG_MODEL]["reason"]


def test_the_4bit_27b_is_loadable_on_this_card_where_the_bf16_27b_is_not(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    """The whole point of the third manifest, on Owen's own card.

    `llm_client` is the RTX 3090 Ti: 24 GiB in total. Two manifests for the same
    27B — same family, same params_b — and the host answers differently about
    each, because the answer is arithmetic about the weights each one points at
    and not about the model's name.
    """
    fake_weights(SMALL_BIG_MODEL)
    fake_weights(BIG_MODEL)
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}

    small = rows[SMALL_BIG_MODEL]
    assert small["family"] == rows[BIG_MODEL]["family"] == "qwen3.8"
    assert small["params_b"] == rows[BIG_MODEL]["params_b"] == 27
    assert small["backend_supported"] is True
    assert small["installed"] is True
    assert small["loadable"] is True
    assert "reason" not in small
    # THIS HOST's context, not the model's. The model wants Owen's
    # `qwen3.8:27b-24g` 98304 and gets it on mlx-darwin; on a 24 GB card that is
    # 7.9 GiB of KV the card does not have, so the cuda-linux block carries its
    # own 16384 and that is what vLLM is given as --max-model-len.
    assert load_manifest(SMALL_BIG_MODEL).context_default == 98304
    assert small["context_default"] == 16384
    # Nothing is resident, so what this host WOULD serve it at is the whole
    # answer, and the two fields agree.
    assert small["max_model_len"] == 16384
    assert small["memory_bytes_estimate"] == 21_633_171_456
    assert small["revision"] == (
        load_manifest(SMALL_BIG_MODEL).spec(FAKE_BACKEND.kind).revision
    )

    assert rows[BIG_MODEL]["loadable"] is False
    assert "52.5 GiB" in rows[BIG_MODEL]["reason"]

    # And the refusal the listing predicts is the refusal the load makes.
    response = submit(llm_client, auth, type="load-model", model=BIG_MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "insufficient_memory"


def test_the_4bit_27b_actually_loads_on_a_free_24_gib_card(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
    engines: list[FakeEngine],
) -> None:
    """Not merely "the card is big enough" — the guard lets it through.

    `loadable` in `/v1/models` compares the estimate against the card's TOTAL, so
    it cannot answer "right now". This does: an empty 24 GiB card, and the live
    guard passes the 23.3 GiB estimate. The margin is 0.7 GiB, which is why the
    `idle_card` fixture — 22 GiB free, the Windows desktop holding the rest — is
    deliberately not used here. On Owen's real card, with his desktop up, this
    load is expected to be tight; see the manifest's comment.
    """
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(
        accelerator,
        "probe_vram",
        lambda: (FAKE_BACKEND.gpu.vram_bytes, FAKE_BACKEND.gpu.vram_bytes),
    )
    fake_weights(SMALL_BIG_MODEL)
    events = run_job(llm_client, auth, type="load-model", model=SMALL_BIG_MODEL)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["resident"] == SMALL_BIG_MODEL
    assert len(engines) == 1
    # vLLM is told the context the manifest promises, and nothing forces a dtype:
    # compressed-tensors W4A16 carries its own, and `dtype auto` is what it wants.
    # `--max-model-len` is the cuda-linux block's own 16384, not the model's
    # 98304: on a 24 GB card 98304 of KV is 7.9 GiB that is not there.
    assert engines[0].args == [
        "--gpu-memory-utilization", "0.86",
        "--max-num-seqs", "16",
        "--skip-mm-profiling",
        "--max-model-len", "16384",
    ]


def test_unknown_params_are_refused(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(MODEL)
    response = submit(
        llm_client, auth, type="load-model", model=MODEL, params={"eager": True}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


# ------------------------------------------------------ the load/unload machine


def test_a_load_warms_then_reports_the_resident_model(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    events = run_job(llm_client, auth, type="load-model", model=MODEL)
    kinds = [event["event"] for event in events]

    assert kinds[0] == "queued"
    assert kinds[-1] == "done"
    assert "warming" in kinds
    # Several warming events, streamed from the engine's readiness (section 5).
    assert kinds.count("warming") >= 3
    # Event ids are strictly increasing from 1, as DESIGN.md section 4 requires.
    assert [event["id"] for event in events] == list(range(1, len(events) + 1))

    warmings = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any("checking the accelerator" in m for m in warmings)
    assert any("22.0 GiB free" in m for m in warmings)
    assert any("fake engine warming" in m for m in warmings)
    assert any("is resident at" in m for m in warmings)

    assert events[-1]["data"]["resident"] == MODEL
    assert len(engines) == 1


def test_health_and_models_see_the_resident_model(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)

    health = llm_client.get("/v1/health", headers=auth).json()
    assert health["resident_models"] == [MODEL]
    assert health["status"] == "ok"

    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[MODEL]["resident"] is True
    assert rows[BIG_MODEL]["resident"] is False

    listed = llm_client.get("/v1/openai/models", headers=auth).json()
    assert listed["object"] == "list"
    assert [entry["id"] for entry in listed["data"]] == [MODEL]


def test_loading_a_second_model_unloads_the_first(
    roomy_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    roomy_card: None,
    engines: list[FakeEngine],
) -> None:
    """Phase 2 residency rule: one resident model at a time (section 3)."""
    llm_client = roomy_client
    fake_weights(MODEL)
    fake_weights(BIG_MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    events = run_job(llm_client, auth, type="load-model", model=BIG_MODEL)

    messages = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any("unloading qwen3.5-9b" in m for m in messages)
    assert events[-1]["data"]["resident"] == BIG_MODEL
    assert len(engines) == 2
    assert engines[0].stopped is True
    assert engines[1].stopped is False
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == [
        BIG_MODEL
    ]


def test_unload_frees_the_engine(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    events = run_job(llm_client, auth, type="unload-model", model=MODEL)

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["resident"] is None
    assert engines[0].stopped is True
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert llm_client.get("/v1/openai/models", headers=auth).json()["data"] == []


def test_unloading_a_model_that_is_not_resident_is_refused(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    response = submit(llm_client, auth, type="unload-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert "no model is" in error["message"]

    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = submit(llm_client, auth, type="unload-model", model=BIG_MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["details"]["resident"] == MODEL


def test_health_says_warming_while_a_load_is_in_flight(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PHASE2-LLM.md section 5: `/health` reports `warming` while a load runs.

    The engine is held inside `ready()` so the server is looked at with a load
    genuinely in flight, rather than racing a sleep. The job's event stream is
    deliberately NOT open while `/health` is read: nesting a request inside a
    TestClient stream answers from a stale view of the store.
    """
    hold = threading.Event()
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path, hold=hold)
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    fake_weights(MODEL)

    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if built and built[0].warming_started.wait(timeout=0.1):
                break
            time.sleep(0.05)  # the engine is not built yet; let the lane run
        assert built and built[0].warming_started.is_set(), "the lane never started"

        health = llm_client.get("/v1/health", headers=auth).json()
        assert health["status"] == "warming", health
        assert health["queue_depth"] == 1, health
        assert health["resident_models"] == [], health
        assert (
            llm_client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"]
            == "running"
        )
    finally:
        hold.set()

    with llm_client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done"
    assert llm_client.get("/v1/health", headers=auth).json() == {
        "status": "ok",
        "queue_depth": 0,
        "resident_models": [MODEL],
    }


def test_an_engine_that_never_becomes_ready_fails_the_job(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(
            python, log_path, fail_ready="vllm exited 1 before it was ready"
        )
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    fake_weights(MODEL)
    events = run_job(llm_client, auth, type="load-model", model=MODEL)
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "engine_failed"
    assert "exited 1" in events[-1]["data"]["error"]["message"]
    # Nothing is left resident and the half-started engine was stopped.
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert built[0].stopped is True


# ------------------------------------------------------------------- the proxy


def test_the_proxy_refuses_a_model_that_is_not_resident(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert "no model is" in error["message"]
    assert "never loads a model to answer a chat request" in error["message"]
    assert error["details"] == {"requested": MODEL, "resident": None}


def test_the_proxy_names_the_resident_model_in_the_409(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert "'qwen3.5-9b' is" in error["message"]
    assert error["details"] == {"requested": "gpt-4", "resident": MODEL}
    # And nothing was loaded to satisfy it.
    assert len(engines) == 1


def test_a_non_streamed_completion_is_passed_through(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is Crucible?"}],
            "temperature": 0.2,
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert body["usage"]["total_tokens"] == 12
    # The sampling the client asked for reached the engine unchanged.
    sent = engines[0].last_request
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 32
    assert sent["messages"] == [{"role": "user", "content": "What is Crucible?"}]


def test_a_streamed_completion_keeps_its_sse_framing(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is Crucible?"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    chunks = _stream_chunks(text)
    deltas = [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "content" in chunk["choices"][0]["delta"]
    ]
    assert deltas == DELTAS
    assert "".join(deltas) == ANSWER
    # The closing frame carries no delta, only the reason the engine stopped.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # Every frame is terminated by a blank line, as SSE requires.
    assert text.endswith("data: [DONE]\n\n")


@pytest.fixture
def engines_under_a_path_name(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> list[FakeEngine]:
    """Engines that answer to the weights directory, the way mlx-lm does.

    The other `engines` fixture makes the engine's name and the Crucible id the
    same string, which is true of vLLM (`--served-model-name`) and hides the
    substitution entirely. This is the other backend's shape.
    """
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: str(Path(model_dir).resolve()),
    )
    return built


def test_a_completion_comes_back_naming_crucible_s_id_not_the_engine_s(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines_under_a_path_name: list[FakeEngine],
) -> None:
    """The proxy substitutes `model` on the way in; it undoes it on the way out."""
    weights = fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    # The engine was asked by the name it answers to...
    assert engines_under_a_path_name[0].last_request["model"] == str(weights.resolve())
    # ...and the client is answered by the name it asked with.
    assert response.json()["model"] == MODEL
    # Everything else is the engine's own body, untouched.
    assert response.json()["choices"][0]["message"]["content"] == ANSWER


def test_every_streamed_chunk_names_crucible_s_id(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines_under_a_path_name: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    chunks = _stream_chunks(text)
    # Every frame, the closing one included — the relabelling reaches all of them.
    assert [chunk["model"] for chunk in chunks] == [MODEL] * (len(DELTAS) + 1)
    # The framing and the content survived the relabelling.
    assert [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "content" in chunk["choices"][0]["delta"]
    ] == DELTAS
    assert text.endswith("data: [DONE]\n\n")


# ------------------------------------------- the constrained transport survives

#: A Foundry analyze verdict, as `askConstrained` builds it (CLIENT-SURFACES.md
#: section 6.2), plus every other knob the OpenAI dialect defines that a client
#: might one day send. The point of the extra fields is that "verbatim" is a rule
#: about the whole body and not about the four keys Crucible happens to know:
#: `logit_bias` and `top_logprobs` are here precisely because nothing in either
#: app sends them today, so nothing in the proxy has ever been taught about them.
CONSTRAINED_BODY: dict[str, Any] = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Does the passage support the claim?"}],
    "temperature": 0,
    "max_tokens": 128,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "verdict",
            "schema": {
                "type": "object",
                "properties": {
                    "supported": {"type": "boolean"},
                    "quote": {"type": "string", "maxLength": 200},
                },
                "required": ["supported", "quote"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    "chat_template_kwargs": {"enable_thinking": False},
    "logprobs": True,
    "top_logprobs": 5,
    "seed": 1729,
    "stop": ["\n\n", "</answer>"],
    "logit_bias": {"15496": -100},
}


def _post_raw(
    client: TestClient, auth: dict[str, str], body: dict[str, Any]
) -> tuple[bytes, Any]:
    """POST a chat body as exact bytes, so byte identity is a question you can ask."""
    payload = json.dumps(body).encode("utf-8")
    response = client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        content=payload,
    )
    return payload, response


def test_a_constrained_body_reaches_the_engine_byte_for_byte(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """On vLLM there is nothing to substitute, so nothing is re-encoded.

    `response_format.json_schema.schema` is a grammar the guided-decoding backend
    compiles. Crucible round-tripping it through `json.loads`/`json.dumps` would
    be a re-encoding of somebody else's document on the way past — harmless until
    the day it is not. Here the engine reads the client's own bytes.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    payload, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)

    assert response.status_code == 200, response.text
    assert engines[0].last_request_bytes == payload


def test_only_the_model_field_changes_when_the_engine_answers_to_a_path(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines_under_a_path_name: list[FakeEngine],
) -> None:
    """The other backend's shape: one field substituted, nothing else touched.

    mlx-lm has no `--served-model-name`, so `model` has to change and the
    document is re-serialised. Everything else — the schema, `strict`, the
    template kwargs, the knobs Crucible has never heard of — arrives with the
    same value in the same position.
    """
    weights = fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)
    assert response.status_code == 200, response.text

    sent = engines_under_a_path_name[0].last_request
    assert sent == {**CONSTRAINED_BODY, "model": str(weights.resolve())}
    # Order too: `model` keeps the place it held, so nothing is appended or
    # shuffled on the way through.
    assert list(sent) == list(CONSTRAINED_BODY)


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
def test_finish_reason_comes_back_untouched(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    reason: str,
) -> None:
    """Foundry turns `length` into a degradation rather than a wrong answer.

    A proxy that normalised or dropped the field would turn a caught truncation
    into silent corruption (CLIENT-SURFACES.md section 6.2), so the engine's own
    word is what comes back — including on a `tool_calls` completion, whose
    `content` is null and whose answer is not text at all.
    """
    engine_factory(finish_reason=reason)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == reason
    if reason == "tool_calls":
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"] == [TOOL_CALL]


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
def test_a_streamed_finish_reason_comes_back_untouched(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    reason: str,
) -> None:
    """The same rule on the streaming half, where it lives in the closing frame."""
    engine_factory(finish_reason=reason)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    chunks = _stream_chunks(text)
    assert chunks[-1]["choices"][0]["finish_reason"] == reason
    # And it is the ONLY frame that names one: the deltas keep their null.
    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks[:-1]] == [
        None
    ] * len(DELTAS)


def test_an_engine_s_own_400_is_relayed_rather_than_rewritten(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
    """A schema the engine will not compile is the engine's refusal to explain.

    Rewrapped in Crucible's `{"error": {"code", "message"}}` envelope it would
    read as the server refusing, and the message naming the part of the grammar
    to fix would be gone.
    """
    engine_factory(reject_response_format=True)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "BadRequestError"
    assert "prefixItems" in body["message"]
    # Not Crucible's envelope: this refusal is not Crucible's to make.
    assert "error" not in body


def test_an_engine_s_own_400_is_relayed_on_a_streamed_request_too(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
    """The stream is opened before anything is returned, so a refusal is a refusal.

    It must never come back as a 200 whose stream turns out to be an error.
    """
    engine_factory(reject_response_format=True)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, {**CONSTRAINED_BODY, "stream": True})

    assert response.status_code == 400
    assert response.json()["type"] == "BadRequestError"


def test_the_proxy_requires_a_model(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_required"


def test_the_proxy_needs_auth(llm_client: TestClient) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers={"X-Crucible-Api": "1"},
        json={"model": MODEL, "messages": []},
    )
    assert response.status_code == 401


def test_a_body_that_is_not_json_is_refused(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        content=b"not json",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
