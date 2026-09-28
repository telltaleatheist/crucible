from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.errors as hub_errors
import pytest

from crucible import jobenv, manifests, service, settings, weights
from crucible.api.routes import catalog as catalog_routes
from crucible.errors import ApiError, JobError
from crucible.jobs.alignlongform import coarse
from crucible.jobs.tts import render
from crucible.manifests import ManifestError

ROOT = Path(__file__).resolve().parent.parent / "crucible"

WHOLE_FILES = (
    "manifests.py",
    "settings.py",
    "weights.py",
    "api/routes/activity.py",
    "api/routes/catalog.py",
    "jobs/rvc/worker.py",
    "jobs/asr/worker.py",
    "jobs/asr/mlx_worker.py",
    "jobs/asr/qwen_worker.py",
    "jobs/tts/render.py",
    "jobs/alignlongform/jobtype.py",
    "jobs/alignlongform/coarse.py",
)

NAMED_FUNCTIONS = (
    ("jobenv.py", "env_status"),
    ("jobenv.py", "install_env"),
    ("jobenv.py", "plan_env"),
    ("jobenv.py", "recipe_index_urls"),
    ("service.py", "status"),
    ("service.py", "install"),
    ("lan.py", "enable"),
)

MAX_LINES = 60
MAX_DEPTH = 3
BLOCKS = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith, ast.Match)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _depth(node, level=0):
    deepest = level
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            continue
        deepest = max(deepest, _depth(child, level + isinstance(child, BLOCKS)))
    return deepest


def _functions(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _too_long_or_deep(relative, nodes):
    return [
        f"{relative}:{n.lineno} {n.name} is {n.end_lineno - n.lineno + 1} lines, depth {_depth(n)}"
        for n in nodes
        if n.end_lineno - n.lineno + 1 > MAX_LINES or _depth(n) > MAX_DEPTH
    ]


@pytest.mark.parametrize("relative", WHOLE_FILES)
def test_no_function_in_a_split_file_is_long_or_deep(relative):
    assert _too_long_or_deep(relative, _functions(relative)) == []


@pytest.mark.parametrize("relative,name", NAMED_FUNCTIONS)
def test_each_named_split_function_is_short_and_shallow(relative, name):
    nodes = [n for n in _functions(relative) if n.name == name and n.col_offset == 0]
    assert len(nodes) == 1
    assert _too_long_or_deep(relative, nodes) == []


def _in_a_fresh_interpreter(source):
    ran = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, cwd=ROOT.parent
    )
    if ran.returncode == 3:
        pytest.skip(ran.stderr.strip())
    assert ran.returncode == 0, ran.stderr


SHA = "0123456789abcdef0123456789abcdef01234567"


def test_the_document_validator_names_an_unknown_top_level_table():
    with pytest.raises(ManifestError, match=r"unknown top-level table\(s\) \['extra'\]"):
        manifests._check_document({"model": {}, "backends": {}, "extra": {}}, Path("m.toml"))


def test_the_memory_validator_refuses_terms_that_disagree_with_the_estimate():
    block = {
        "memory_bytes_estimate": 1000,
        "memory": {
            "weights_bytes": 500,
            "overhead_bytes": 0,
            "kv_bytes_per_token": 1,
            "basis": "measured",
            "measured_at_context": 100,
        },
    }
    with pytest.raises(ManifestError, match="parts-against-whole"):
        manifests._parse_memory("m.toml [backends.cuda-linux]", block, 100)
    block["memory_bytes_estimate"] = 600
    terms = manifests._parse_memory("m.toml [backends.cuda-linux]", block, 100)
    assert terms.bytes_for(context=100, concurrency=1) == 600


def test_the_local_validator_builds_the_form_its_kind_names():
    table = {
        "kind": "gguf",
        "download_bytes": 10,
        "needs_bytes": 20,
        "needs_basis": "measured",
        "hf_repo": "owner/name",
        "revision": SHA,
        "file": "m-Q8_0.gguf",
    }
    local = manifests._parse_local(table, Path("m.toml"), ("text",))
    assert isinstance(local, manifests.GgufLocal)
    assert local.mmproj is None
    with pytest.raises(ManifestError, match="names a vision projector"):
        manifests._parse_local({**table, "mmproj": "p-Q8_0.gguf"}, Path("m.toml"), ("text",))


def test_the_defaults_validator_coerces_numbers_and_refuses_bools_for_them():
    defaults = manifests._parse_defaults({"temperature": 1, "top_k": 5}, Path("m.toml"))
    assert defaults.temperature == 1.0 and isinstance(defaults.temperature, float)
    with pytest.raises(ManifestError, match="temperature must be a number, got bool"):
        manifests._parse_defaults({"temperature": True}, Path("m.toml"))


def test_each_settings_section_has_one_resolver_in_patch_order():
    assert [key for key, _ in settings.SECTION_RESOLVERS] == [
        "upstreams",
        "routes",
        "desktop_allowance_bytes",
        "local_models",
        "tailscale_advertise",
        "lan_advertise",
    ]
    assert {key for key, _ in settings.SECTION_RESOLVERS} == settings.PATCH_KEYS


def test_the_route_resolver_refuses_a_value_that_names_no_upstream():
    resolved = SimpleNamespace(routes={}, changed=[])
    with pytest.raises(ApiError) as caught:
        settings._resolve_route(resolved, "clean", "nowhere/model")
    assert caught.value.code == "route_bad_model"
    settings._resolve_route(resolved, "clean", "openai/gpt")
    assert resolved.routes == {"clean": "openai/gpt"}
    assert resolved.changed == ["routes.clean = openai/gpt"]


def test_env_drift_names_every_wrong_pin_and_reference():
    assert jobenv._pin_drift({"a": "1", "b": "2"}, {"a": "1"}) == [
        "b is absent, recipe pins 2"
    ]
    assert jobenv._reference_drift({"n": "x" * 40}, {"n": "y" * 40}) == [
        f"n was installed from {'y' * 40}, recipe pins {'x' * 40}"
    ]


def test_an_env_with_no_venv_is_reported_once_with_the_install_command(tmp_path):
    spec = jobenv.EnvSpec(job_type="asr", key="asr", recipe_name="cuda-linux", headline="faster-whisper")
    status = jobenv.env_status(tmp_path, spec, "cuda-linux")
    assert not status.installed
    assert status.packages == {}
    assert status.python_version is None
    assert status.detail.endswith("run `crucible install asr`")


def test_launchd_state_reads_the_pid_from_launchctl_list():
    listing = f"42\t0\t{service.LAUNCHD_LABEL}\n"
    runner = lambda argv: SimpleNamespace(ok=True, stdout=listing, argv=list(argv))
    running, pid, detail, linger = service._launchd_state(runner)
    assert (running, pid, linger) == (True, 42, None)
    assert detail == f"agent {service.LAUNCHD_LABEL} is loaded and running as pid 42"


def _hub_error(kind):
    error = kind.__new__(kind)
    Exception.__init__(error, "boom")
    return error


def test_a_gated_repo_is_named_before_it_is_called_missing():
    wanted = weights.HubFile(
        repo="org/repo", revision="a" * 40, filename="w.bin", pinned_by="x.toml", names="a path"
    )
    config = SimpleNamespace(path="/c/config.toml", name="c")
    gated = weights.download_error(
        _hub_error(hub_errors.GatedRepoError), hub_errors, config, wanted
    )
    assert "is gated" in str(gated)
    entry = weights.download_error(
        _hub_error(hub_errors.EntryNotFoundError), hub_errors, config, wanted
    )
    assert "has no file 'w.bin'; x.toml names a path" in str(entry)
    other = weights.download_error(ValueError("no"), hub_errors, config, wanted)
    assert str(other) == "pulling org/repo:w.bin failed: ValueError: no"


def test_a_task_names_a_subject_directly_or_through_its_module():
    direct = SimpleNamespace(request={"kind": "model", "id": "m"})
    nested = SimpleNamespace(
        request={"module": {"subjects": [{"kind": "voice", "id": "v"}, "junk"]}}
    )
    assert catalog_routes._task_names(direct, "model", "m")
    assert catalog_routes._task_names(nested, "voice", "v")
    assert not catalog_routes._task_names(nested, "voice", "w")
    assert not catalog_routes._task_names(SimpleNamespace(request={"module": None}), "voice", "v")


def test_the_anchor_chain_keeps_the_longest_run_that_moves_forward_in_both_axes() -> None:
    cands = [(0, 5), (1, 2), (1, 9), (2, 12), (3, 1)]
    assert coarse._anchor_chain(cands) == [(0, 5), (1, 9), (2, 12)]


def test_the_batch_tally_refuses_a_row_answered_twice_and_skips_quiet_messages() -> None:
    tally = render._BatchTally(expected={0, 1}, total=2)
    assert tally.claim({"type": "batch_done"}) is None
    assert tally.claim({"type": "stopped"}) is None
    assert tally.claim({"type": "batch_item", "i": 0}) == 0
    tally.record(0, None)
    assert tally.row_line(0, None) == "1 of 2 chunk(s) rendered"
    with pytest.raises(JobError, match="twice"):
        tally.claim({"type": "batch_item", "i": 0})
    with pytest.raises(JobError, match="unanswered"):
        tally.require_complete()


def test_a_band_rate_that_is_a_bool_is_malformed_not_one() -> None:
    with pytest.raises(ApiError) as caught:
        render._band_rate({"pace_chars_per_sec": True}, "pace_chars_per_sec")
    assert caught.value.code == "band_malformed"


FASTER_WHISPER_REQUEST = """
from crucible.jobs.asr import worker
request = {
    "model_dir": "m", "ffmpeg": "f", "audio": "a", "device": "cpu",
    "compute_type": "int8", "vad_filter": False, "word_timestamps": True,
    "window_s": 30, "overlap_s": 2, "language": None, "initial_prompt": None,
    "speech": None,
}
params = worker.parse_request(request)
assert params["language"] is None and params["window_s"] == 30
del request["language"]
try:
    worker.parse_request(request)
except KeyError as exc:
    assert "'language'" in str(exc)
else:
    raise AssertionError("a request with no language was accepted")
"""

RVC_BATCH = """
import sys
try:
    import scipy
    import soundfile
except ImportError as exc:
    print(f"the rvc worker needs {exc.name}", file=sys.stderr)
    sys.exit(3)
from crucible.jobs.rvc import worker
batch = worker._Batch(2, 10.0)
assert batch.fits(50.0)
batch.take(0, 50.0)
assert not batch.fits(1.0)
small = worker._Batch(2, 10.0)
small.take(0, 1.0)
small.take(1, 1.0)
assert not small.fits(1.0)
"""


def test_the_faster_whisper_request_parses_to_one_record_and_names_a_missing_language():
    _in_a_fresh_interpreter(FASTER_WHISPER_REQUEST)


def test_an_rvc_batch_always_takes_its_first_piece_and_then_stops_at_either_limit():
    _in_a_fresh_interpreter(RVC_BATCH)


def test_the_qwen_split_region_is_the_whole_source_when_null_and_refused_when_empty():
    from crucible.jobs.asr import qwen_worker

    rate = qwen_worker.SAMPLE_RATE
    assert qwen_worker._region_bounds(None, 10 * rate, "a.wav") == (0, 10 * rate)
    assert qwen_worker._region_bounds([1.0, 2.0], 10 * rate, "a.wav") == (rate, 2 * rate)
    with pytest.raises(ValueError, match="holds no audio"):
        qwen_worker._region_bounds([5.0, 5.0], 10 * rate, "a.wav")
