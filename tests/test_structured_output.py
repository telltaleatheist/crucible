"""Structured output: every engine states which constraints it enforces, the chat door
refuses the rest by name, and mlx-lm's patch reads a request the way vLLM does.

The grammar itself (llguidance over mlx) is tests/test_structured_mlx.py, which needs
llguidance, mlx and mlx-lm and skips without them."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import envpatches, jobenv
from crucible.engines import ENGINES, structured_output_reading
from crucible.engines import structured_mlx
from crucible.errors import ApiError
from crucible.structured import GRAMMAR_FIELDS, constrained_fields, refuse_unenforced_constraint

from .conftest import FAKE_MAC_BACKEND
from .fake_engine import FakeEngine
from .test_llm_api import (
    MODEL,
    engines,  # noqa: F401 - a fixture this module uses
    fake_env,  # noqa: F401 - a fixture this module uses
    fake_weights,  # noqa: F401 - a fixture this module uses
    idle_card,  # noqa: F401 - a fixture this module uses
    llm_client,  # noqa: F401 - a fixture this module uses
    run_job,
)
from .test_llm_patches import (
    FIXTURE,
    generate_of,
    make_env,
    pristine,
    pristine_generate,
    run_script,
    server_of,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "v", "schema": SCHEMA, "strict": True}}
USER = [{"role": "user", "content": "is it?"}]
PATCH = envpatches.MLX_LM_STRUCTURED_OUTPUT
BATCH = envpatches.MLX_LM_STRUCTURED_OUTPUT_BATCH
HELPER = envpatches.MLX_LM_STRUCTURED_OUTPUT_HELPER
SCRIPT = envpatches.LLM_SCRIPTS_DIR / PATCH.script
HELPER_SCRIPT = envpatches.LLM_SCRIPTS_DIR / HELPER.script
MAC_PINS = jobenv.recipe_pins(jobenv.recipe_for(jobenv.llm_env("mlx-darwin")))
VL_MODEL = "qwen3.5-9b-vl"


def refused(call: Callable[[], Any]) -> ApiError:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value


def door(engine: str, body: dict[str, Any]) -> None:
    reading = structured_output_reading(engine, "cuda-linux")
    refuse_unenforced_constraint(
        engine=engine,
        model_id=MODEL,
        formats=reading.formats,
        fields=reading.fields,
        basis=reading.basis,
        body=body,
    )


# --- what each engine states -----------------------------------------------------


def test_every_engine_states_what_it_enforces_and_where_it_read_it() -> None:
    for name in ENGINES:
        reading = structured_output_reading(name, "cuda-linux")
        assert reading.basis.strip(), name


def test_the_text_engines_enforce_a_json_schema_and_the_page_reader_none() -> None:
    for name in ("vllm", "llama-server", "mlx-lm"):
        reading = structured_output_reading(name, "cuda-linux")
        assert {"json_object", "json_schema"} <= reading.formats, name
    assert not structured_output_reading("mlx-vlm", "mlx-darwin").served


def test_mlx_lm_states_what_its_patch_reads() -> None:
    reading = structured_output_reading("mlx-lm", "mlx-darwin")
    assert reading.formats == frozenset(structured_mlx.RESPONSE_FORMATS)
    assert reading.fields == frozenset({"structured_outputs"})
    assert set(structured_mlx.UNSERVED_FIELDS) == set(GRAMMAR_FIELDS) - reading.fields


# --- the door ---------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["vllm", "llama-server", "mlx-lm"])
def test_a_json_schema_passes_where_it_is_enforced(engine: str) -> None:
    door(engine, {"response_format": JSON_SCHEMA})
    door(engine, {"response_format": {"type": "json_object"}})


@pytest.mark.parametrize("engine", list(ENGINES))
def test_a_text_format_is_no_constraint_anywhere(engine: str) -> None:
    door(engine, {"response_format": {"type": "text"}})
    door(engine, {})


def test_a_schema_on_mlx_vlm_is_refused_by_name() -> None:
    error = refused(lambda: door("mlx-vlm", {"response_format": JSON_SCHEMA}))
    assert error.status_code == 400 and error.code == "structured_output_not_served"
    assert error.details["fields"] == ["response_format type 'json_schema'"]
    assert error.details["enforced"] == []
    assert "mlx_vlm_serve.py" in str(error)


@pytest.mark.parametrize(
    ("engine", "field"),
    [
        ("vllm", "guided_json"),
        ("vllm", "grammar"),
        ("llama-server", "structured_outputs"),
        ("mlx-lm", "guided_json"),
        ("mlx-lm", "grammar"),
        ("mlx-lm", "json_schema"),
    ],
)
def test_a_field_the_engine_does_not_read_is_refused(engine: str, field: str) -> None:
    error = refused(lambda: door(engine, {field: SCHEMA}))
    assert error.code == "structured_output_not_served"
    assert error.details["fields"] == [field]
    assert error.details["engine"] == engine


def test_a_format_type_the_engine_does_not_read_is_refused() -> None:
    door("vllm", {"response_format": {"type": "structural_tag"}})
    error = refused(lambda: door("mlx-lm", {"response_format": {"type": "structural_tag"}}))
    assert error.details["fields"] == ["response_format type 'structural_tag'"]


def test_constrained_fields_lists_the_grammar_fields_then_response_format() -> None:
    body = {"response_format": JSON_SCHEMA, "grammar": "root ::= x", "guided_json": SCHEMA}
    assert constrained_fields(body) == ["guided_json", "grammar", "response_format"]
    assert constrained_fields({"response_format": {"type": "text"}}) == []


def load(client: TestClient, auth: dict[str, str]) -> None:
    run_job(client, auth, type="load-model", model=MODEL)


def test_the_door_refuses_a_guided_field_on_vllm_and_sends_nothing(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    before = engines[0].last_request
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": USER, "guided_json": SCHEMA},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "structured_output_not_served"
    assert error["details"]["engine"] == "vllm" and error["details"]["fields"] == ["guided_json"]
    assert engines[0].last_request == before


def test_the_door_sends_structured_outputs_to_vllm(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": USER, "structured_outputs": {"choice": ["a", "b"]}},
    )
    assert response.status_code == 200, response.json()
    assert engines[0].last_request["structured_outputs"] == {"choice": ["a", "b"]}


def test_a_schema_for_a_mac_page_model_is_refused_before_it_waits(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
) -> None:
    with make_client(backend=FAKE_MAC_BACKEND, enable_llm=True) as client:
        response = client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": VL_MODEL, "messages": USER, "response_format": JSON_SCHEMA},
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "structured_output_not_served"
    assert error["details"]["model"] == VL_MODEL and error["details"]["engine"] == "mlx-vlm"


def test_a_guided_field_for_a_mac_text_model_is_refused_before_it_waits(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
) -> None:
    with make_client(backend=FAKE_MAC_BACKEND, enable_llm=True) as client:
        response = client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": MODEL, "messages": USER, "guided_json": SCHEMA},
        )
    assert response.status_code == 400
    assert response.json()["error"]["details"]["engine"] == "mlx-lm"


# --- what mlx-lm's patch reads (no llguidance needed) -----------------------------


def asked(body: dict[str, Any]) -> Any:
    return structured_mlx.asked_for(body)


def refusal(body: dict[str, Any]) -> structured_mlx.GrammarRefusal:
    with pytest.raises(structured_mlx.GrammarRefusal) as caught:
        structured_mlx.asked_for(body)
    return caught.value


def test_the_patch_reads_what_vllm_reads() -> None:
    assert asked({}) is None
    assert asked({"response_format": {"type": "text"}}) is None
    assert asked({"response_format": JSON_SCHEMA}) == ("json", SCHEMA, "response_format json_schema")
    assert asked({"response_format": {"type": "json_object"}})[0] == "json_object"
    assert asked({"structured_outputs": {"json": SCHEMA}})[:2] == ("json", SCHEMA)
    assert asked({"structured_outputs": {"json": '{"type": "object"}'}})[1] == {"type": "object"}
    assert asked({"structured_outputs": {"regex": "[a-c]+"}})[:2] == ("regex", "[a-c]+")
    assert asked({"structured_outputs": {"choice": ["a", "b"]}})[:2] == ("choice", ["a", "b"])
    assert asked({"structured_outputs": {"json_object": True}})[0] == "json_object"
    assert asked({"structured_outputs": {"regex": "x", "disable_any_whitespace": False}})


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"guided_json": SCHEMA}, "structured_output_not_served"),
        ({"grammar": "root ::= x"}, "structured_output_not_served"),
        ({"response_format": {"type": "structural_tag"}}, "structured_output_not_served"),
        ({"structured_outputs": {"structural_tag": "{}"}}, "structured_output_not_served"),
        ({"structured_outputs": {"regex": "x", "whitespace_pattern": " "}}, "structured_output_not_served"),
        ({"structured_outputs": {"regex": "x", "disable_any_whitespace": True}}, "structured_output_not_served"),
        ({"response_format": {"type": "json_schema", "json_schema": {"name": "v"}}}, "invalid_response_format"),
        ({"response_format": {"type": "json_schema"}}, "invalid_response_format"),
        ({"response_format": "json"}, "invalid_response_format"),
        ({"structured_outputs": {}}, "invalid_structured_outputs"),
        ({"structured_outputs": {"regex": "a", "choice": ["a"]}}, "invalid_structured_outputs"),
        ({"structured_outputs": {"choice": []}}, "invalid_structured_outputs"),
        ({"structured_outputs": {"json": "{not json"}}, "invalid_structured_outputs"),
        (
            {"response_format": {"type": "json_object"}, "structured_outputs": {"regex": "a"}},
            "invalid_request",
        ),
    ],
)
def test_the_patch_refuses_by_name_what_it_cannot_enforce(body: dict[str, Any], code: str) -> None:
    error = refusal(body)
    assert error.status == 400 and error.code == code


# --- the env patch ----------------------------------------------------------------


def test_the_mac_recipe_installs_llguidance() -> None:
    assert MAC_PINS["llguidance"] == "1.8.0"


def test_the_applier_patches_server_and_generate_and_keeps_snapshots(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    done = run_script(env, SCRIPT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("PATCHED ") == 2
    server = server_of(env).read_text(encoding="utf-8")
    generate = generate_of(env).read_text(encoding="utf-8")
    assert server.count(PATCH.marker) == 1 and PATCH.absent_marker not in server
    assert server.count("_make_logits_processors(args, tokenizer)") == 3
    assert "crucible_constraint=self.crucible_constraint," in server
    assert "_crucible_grammar.answer_failure(self, failed)" in server
    assert generate.count(BATCH.marker) == 1 and BATCH.absent_marker not in generate
    assert Path(str(server_of(env)) + ".orig").read_text(encoding="utf-8") == pristine()
    assert Path(str(generate_of(env)) + ".orig").read_text(encoding="utf-8") == pristine_generate()
    py_compile.compile(str(server_of(env)), doraise=True)
    py_compile.compile(str(generate_of(env)), doraise=True)
    rows = envpatches.check_patches(env, MAC_PINS, patches=(PATCH, BATCH))
    assert [row["status"] for row in rows] == [envpatches.APPLIED] * 2


def test_the_applier_is_idempotent(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env, SCRIPT).returncode == 0
    once = (server_of(env).read_bytes(), generate_of(env).read_bytes())
    again = run_script(env, SCRIPT)
    assert again.returncode == 0 and again.stdout.count("ALREADY_PATCHED ") == 2
    assert (server_of(env).read_bytes(), generate_of(env).read_bytes()) == once


def test_a_moved_anchor_in_one_file_writes_neither(tmp_path: Path) -> None:
    moved = pristine_generate().replace(
        "for processor in self.logits_processors[e]:", "for proc in self.logits_processors[e]:"
    )
    env = make_env(tmp_path, generate_text=moved)
    done = run_script(env, SCRIPT)
    assert done.returncode == 2 and "ANCHOR_NOT_FOUND" in done.stderr
    assert server_of(env).read_text(encoding="utf-8") == pristine()
    assert generate_of(env).read_text(encoding="utf-8") == moved


def test_another_mlx_lm_version_is_refused(tmp_path: Path) -> None:
    env = make_env(tmp_path, version="0.32.0")
    done = run_script(env, SCRIPT)
    assert done.returncode == 2 and "VERSION_MISMATCH" in done.stderr
    assert server_of(env).read_text(encoding="utf-8") == pristine()


def test_the_applier_goes_in_after_every_other_llm_patch(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    for patch in envpatches.LLM_PATCHES:
        done = run_script(env, envpatches.LLM_SCRIPTS_DIR / patch.script)
        assert done.returncode == 0, (patch.id, done.stderr)
    rows = envpatches.check_patches(env, MAC_PINS, patches=envpatches.LLM_PATCHES)
    assert {row["id"]: row["status"] for row in rows} == {
        patch.id: envpatches.APPLIED for patch in envpatches.LLM_PATCHES
    }
    py_compile.compile(str(server_of(env)), doraise=True)


def test_the_script_and_the_table_name_the_same_strings() -> None:
    namespace: dict = {"__name__": SCRIPT.stem}
    exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"), namespace)
    assert namespace["REL"] == PATCH.rel_path and namespace["GENERATE_REL"] == BATCH.rel_path
    assert namespace["MARKER"] == PATCH.marker
    assert namespace["ABSENT_MARKER"] == PATCH.absent_marker
    assert namespace["GENERATE_MARKER"] == BATCH.marker
    assert namespace["GENERATE_ABSENT_MARKER"] == BATCH.absent_marker
    assert PATCH.script == BATCH.script == SCRIPT.name
    assert FIXTURE.name == "mlx_lm_0.31.3_server.py.txt"


def test_the_helper_is_structured_mlx_verbatim_and_placed_once(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    done = run_script(env, HELPER_SCRIPT)
    assert done.returncode == 0 and done.stdout.startswith("PATCHED ")
    placed = server_of(env).with_name("_crucible_grammar.py")
    assert placed.read_bytes() == Path(structured_mlx.__file__).read_bytes()
    assert run_script(env, HELPER_SCRIPT).stdout.startswith("ALREADY_PATCHED ")
    assert HELPER.marker == f"GRAMMAR_VERSION = {structured_mlx.GRAMMAR_VERSION}"
    [row] = envpatches.check_patches(env, MAC_PINS, patches=(HELPER,))
    assert row["status"] == envpatches.APPLIED


def test_the_helper_imports_nothing_from_crucible() -> None:
    source = Path(structured_mlx.__file__).read_text(encoding="utf-8")
    assert "from crucible" not in source and "import crucible" not in source
    assert "from ." not in source and "from .." not in source
    done = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1])", structured_mlx.__file__],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
