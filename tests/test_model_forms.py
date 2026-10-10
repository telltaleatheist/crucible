"""A model's FORMS: one id, the same weights in more than one precision, and each host
serving the best form its card holds (docs/FITS-AND-THE-CARD.md section 8).

Owen, 2026-10-10: qwen3.5-4b-bside is full precision on the Mac and this PC, and 8-bit on
Victoria's laptop; B-Sides names the model and never the form. A calling app MAY name a form
("crucible handles serving the model, with more granular controls available from the
calling app if they want it").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from crucible import manifests, weights
from crucible.backend import CUDA_LINUX, Backend, Gpu
from crucible.cli import doctor
from crucible.config import load_config
from crucible.errors import ApiError
from crucible.formrequest import take_form
from crucible.jobs.llm import _require_loadable, model_rows
from crucible.manifests import (
    HostFit,
    ManifestError,
    UnknownForm,
    host_fit_of,
    load_manifest,
    parse_manifest,
    pick_form,
)
from crucible.memorybudget import GIB
from crucible.residency import Residency, serves_model

from .conftest import FAKE_BACKEND, configure_box
from .fake_engine import FakeEngine
from .fake_hub import FakeHub
from .live_server import run_job, serve

BSIDE = "qwen3.5-4b-bside"

# The two cards, with the numbers the repo already holds for them.
# owens-pc-wsl: RTX 3090 Ti, 24,564 MiB (docs/MEASUREMENTS.md "The machines"; the test
# backend's figure), its desktop reserve at the 3 GiB cap (docs/VERB-SIZING.md 1b).
PC = HostFit(
    backend_kind=CUDA_LINUX,
    card="NVIDIA GeForce RTX 3090 Ti",
    total_bytes=25_757_220_864,
    desktop_allowance_bytes=3 * GIB,
)
# Victoria's laptop: RTX 3070 Laptop, 8 GiB, its desktop measured at 1 GiB
# (docs/VERB-SIZING.md 1b; config.CARD_DESKTOP_ALLOWANCE_FRACTION's comment).
VICTORIA = HostFit(
    backend_kind=CUDA_LINUX,
    card="NVIDIA GeForce RTX 3070 Laptop GPU",
    total_bytes=8 * GIB,
    desktop_allowance_bytes=1 * GIB,
)


def _backend(fit: HostFit) -> Backend:
    return Backend(
        kind=fit.backend_kind,
        platform="linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name=fit.card, vram_bytes=fit.total_bytes),
        detail="test",
    )


# ---- the picks -------------------------------------------------------------------------------


def test_the_pc_takes_bf16_and_victoria_s_laptop_takes_q8_0() -> None:
    block = load_manifest(BSIDE).block(CUDA_LINUX)
    assert block.form_names == ("bf16", "q8_0"), "best first"

    pc = pick_form(BSIDE, block, PC)
    assert PC.available_bytes == 22_535_995_392  # 20.99 GiB
    assert block.form_named("bf16", BSIDE).memory_bytes_estimate == 9_952_490_976  # 9.27 GiB
    assert (pc.form.name, pc.fits) == ("bf16", True)
    assert pc.fitting == ("bf16", "q8_0")
    assert "bf16 needs 9.27 GiB" in pc.reason and "20.99 GiB" in pc.reason

    laptop = pick_form(BSIDE, block, VICTORIA)
    assert VICTORIA.available_bytes == 7 * GIB
    assert block.form_named("q8_0", BSIDE).memory_bytes_estimate == 5_897_450_784  # 5.49 GiB
    assert (laptop.form.name, laptop.fits) == ("q8_0", True)
    assert laptop.fitting == ("q8_0",)
    assert "the best form that fits" in laptop.reason
    assert "bf16 needs 9.27 GiB, which does not fit" in laptop.reason


def test_the_mac_block_is_bf16_and_has_no_forms() -> None:
    manifest = load_manifest(BSIDE)
    assert manifest.form_pick("mlx-darwin") is None
    spec = manifest.spec("mlx-darwin")
    assert (spec.bits, spec.form, spec.forms) == (16, None, ())


def test_the_bf16_form_is_the_uploaded_file_at_the_shared_pin() -> None:
    manifest = load_manifest(BSIDE)
    bf16 = manifest.spec(CUDA_LINUX, "bf16")
    q8 = manifest.spec(CUDA_LINUX, "q8_0")
    assert bf16.file == "qwen3.5-4b-bside-BF16.gguf"
    assert q8.file == "qwen3.5-4b-bside-Q8_0.gguf"
    assert bf16.revision == q8.revision == "2a9bc5673485439ecff9fcd55eb81743b1fc0c2f"
    assert bf16.memory is not None and bf16.memory.weights_bytes == 8_665_620_064
    assert q8.memory is not None and q8.memory.weights_bytes == 4_610_579_872
    for spec in (bf16, q8):
        assert spec.memory is not None
        assert (spec.memory.kv_bytes_per_token, spec.memory.overhead_bytes) == (32_768, 750_000_000)
        assert spec.memory.basis == "computed"
        # Same id, same served name, same engine and arguments: only the file differs.
        assert spec.engine_args == ("--parallel", "1", "--n-gpu-layers", "all")


def test_the_pick_moves_only_with_the_card_or_the_allowance() -> None:
    block = load_manifest(BSIDE).block(CUDA_LINUX)
    # A 16 GiB card holds bf16 with a 3 GiB reserve, and loses it only if the reserve grows
    # past what is left over.
    sixteen = HostFit(CUDA_LINUX, "16 GiB", 16 * GIB, 3 * GIB)
    assert pick_form(BSIDE, block, sixteen).form.name == "bf16"
    crowded = HostFit(CUDA_LINUX, "16 GiB", 16 * GIB, 7 * GIB)
    assert pick_form(BSIDE, block, crowded).form.name == "q8_0"
    assert pick_form(BSIDE, block, sixteen) == pick_form(BSIDE, block, sixteen)


def test_when_nothing_fits_the_smallest_is_tried_and_the_reason_has_the_numbers() -> None:
    block = load_manifest(BSIDE).block(CUDA_LINUX)
    tiny = HostFit(CUDA_LINUX, "a 6 GiB card", 6 * GIB, 1 * GIB)
    pick = pick_form(BSIDE, block, tiny)
    assert (pick.form.name, pick.fits, pick.fitting) == ("q8_0", False, ())
    assert pick.reason.startswith("no form fits: bf16 needs 9.27 GiB, q8_0 needs 5.49 GiB")
    assert "q8_0, the smallest, is tried" in pick.reason


def test_with_no_card_known_the_first_form_is_read_and_says_so() -> None:
    manifest = load_manifest(BSIDE)
    pick = manifest.form_pick(CUDA_LINUX)
    assert pick is not None and (pick.form.name, pick.fits) == ("bf16", None)
    assert "no card is known here" in pick.reason


def test_a_registered_host_picks_for_the_whole_process(make_app: Callable[..., Any]) -> None:
    """create_app registers its host: every reader with no host of its own (a catalog row,
    a candidate, a descriptor) reads the form this card takes."""
    make_app(enable_llm=True, backend=_backend(VICTORIA), desktop_allowance_bytes=1 * GIB)
    assert load_manifest(BSIDE).spec(CUDA_LINUX).form == "q8_0"
    make_app(enable_llm=True, backend=_backend(PC), desktop_allowance_bytes=3 * GIB)
    assert load_manifest(BSIDE).spec(CUDA_LINUX).form == "bf16"


def test_an_explicit_host_wins_over_the_registered_one() -> None:
    manifests.use_host_fit(lambda: PC)
    manifest = load_manifest(BSIDE)
    assert manifest.spec(CUDA_LINUX).form == "bf16"
    assert manifest.spec(CUDA_LINUX, host=VICTORIA).form == "q8_0"
    assert manifest.spec(CUDA_LINUX, "bf16", host=VICTORIA).form == "bf16", "a named form is that form"


def test_an_unknown_form_is_refused_by_name_with_the_forms_it_has() -> None:
    manifest = load_manifest(BSIDE)
    with pytest.raises(UnknownForm) as refused:
        manifest.spec(CUDA_LINUX, "q4_k_m")
    assert refused.value.forms == ("bf16", "q8_0")
    assert "['bf16', 'q8_0']" in str(refused.value)
    with pytest.raises(UnknownForm) as single:
        load_manifest("qwen3.5-9b").spec(CUDA_LINUX, "bf16")
    assert single.value.forms == ()


# ---- the manifest shape ----------------------------------------------------------------------

_HEAD = '''
[model]
id = "demo-4b"
family = "demo"
params_b = 4
context_default = 8192
trained_context = 32768
modalities = ["text"]

[backends.cuda-linux]
engine = "llama-server"
hf_repo = "someone/demo-4b-gguf"
revision = "0123456789abcdef0123456789abcdef01234567"
engine_args = ["--parallel", "1"]
'''

_BF16 = '''
[[backends.cuda-linux.forms]]
name = "bf16"
bits = 16
file = "demo-4b-BF16.gguf"
memory_bytes_estimate = 9_000_000_000
'''

_Q8 = '''
[[backends.cuda-linux.forms]]
name = "q8_0"
bits = 8
file = "demo-4b-Q8_0.gguf"
memory_bytes_estimate = 5_000_000_000
'''


def _parse(text: str, tmp_path: Path) -> Any:
    return parse_manifest(text, tmp_path / "demo-4b.toml", "demo-4b")


def test_a_block_with_forms_parses_and_a_block_without_is_unchanged(tmp_path: Path) -> None:
    manifest = _parse(_HEAD + _BF16 + _Q8, tmp_path)
    block = manifest.block(CUDA_LINUX)
    assert block.form_names == ("bf16", "q8_0")
    assert (block.file, block.bits, block.memory) == (None, None, None)
    assert block.to_dict()["forms"][1]["file"] == "demo-4b-Q8_0.gguf"
    assert load_manifest("qwen3.5-9b").block(CUDA_LINUX).forms == ()


@pytest.mark.parametrize(
    ("text", "said"),
    [
        (_HEAD + _BF16, "forms lists 1 form(s)"),
        (_HEAD + _Q8 + _BF16, "forms are best first"),
        (_HEAD + 'file = "x-Q8_0.gguf"\n' + _BF16 + _Q8, "stated on a block that has forms"),
        (_HEAD + "memory_bytes_estimate = 1\n" + _BF16 + _Q8, "stated on a block that has forms"),
        (_HEAD + _BF16 + _Q8.replace('"q8_0"', '"bf16"'), "two forms state the same name"),
        (_HEAD + _BF16 + _Q8.replace("bits = 8", "bits = 4"), "says 8-bit"),
        (_HEAD + _BF16 + _Q8.replace('name = "q8_0"', 'name = "Q8"'), "must be lower-case"),
        (_HEAD + _BF16 + _Q8 + "colour = 1\n", "unknown key(s) ['colour']"),
        (_HEAD + _BF16 + _Q8.replace("demo-4b-Q8_0.gguf", "demo-4b-Q2_K.gguf"), "no less than 4"),
    ],
)
def test_a_malformed_forms_block_is_refused_by_name(
    text: str, said: str, tmp_path: Path
) -> None:
    with pytest.raises(ManifestError) as refused:
        _parse(text, tmp_path)
    assert said in str(refused.value)


def test_forms_on_a_whole_repo_block_are_refused(tmp_path: Path) -> None:
    text = (_HEAD + _BF16 + _Q8).replace('engine = "llama-server"', 'engine = "vllm"')
    with pytest.raises(ManifestError) as refused:
        _parse(text, tmp_path)
    assert "llama-server" in str(refused.value)


def test_a_form_s_memory_terms_must_agree_with_its_estimate(tmp_path: Path) -> None:
    terms = '''
[backends.cuda-linux.forms.memory]
weights_bytes = 4_000_000_000
overhead_bytes = 500_000_000
kv_bytes_per_token = 32_768
basis = "computed"
measured_at_context = 8192
'''
    good = _parse(_HEAD + _BF16 + _Q8 + terms.replace("4_000_000_000", "4_231_564_544"), tmp_path)
    assert good.spec(CUDA_LINUX, "q8_0").memory is not None
    with pytest.raises(ManifestError) as refused:
        _parse(_HEAD + _BF16 + _Q8 + terms.replace("4_000_000_000", "2_000_000_000"), tmp_path)
    assert "memory_bytes_estimate says 5000000000" in str(refused.value)


# ---- pulling a second form -------------------------------------------------------------------


def _config(home: Path, fit: HostFit) -> Any:
    configure_box(
        home, enable_llm=True, backend=_backend(fit), desktop_allowance_bytes=fit.desktop_allowance_bytes
    )
    return load_config(home)


def _pull(config: Any, form: str, monkeypatch: pytest.MonkeyPatch, hub: FakeHub | None = None) -> Any:
    hub = hub if hub is not None else FakeHub(chunks=1)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub.snapshot_download, raising=False)
    manifest = load_manifest(BSIDE)
    return weights.pull(config, manifest, manifest.spec(CUDA_LINUX, form))


def test_pulling_the_second_form_keeps_the_first_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "home", PC)
    manifest = load_manifest(BSIDE)
    q8, bf16 = manifest.spec(CUDA_LINUX, "q8_0"), manifest.spec(CUDA_LINUX, "bf16")
    _pull(config, "q8_0", monkeypatch)
    assert weights.installed(config, manifest, bf16) is None

    hub = FakeHub(chunks=1)
    _pull(config, "bf16", monkeypatch, hub)
    assert hub.allowed == [["qwen3.5-4b-bside-BF16.gguf"]], "only the new form's file is fetched"
    assert weights.installed(config, manifest, bf16) is not None
    assert weights.installed(config, manifest, q8) is not None, "the first form stays installed"
    stamp = json.loads((weights.subject_dir(config, manifest, CUDA_LINUX) / "crucible-pull.json").read_text())
    assert stamp["files"] == ["qwen3.5-4b-bside-BF16.gguf", "qwen3.5-4b-bside-Q8_0.gguf"]


def test_a_failed_pull_of_a_second_form_leaves_the_first_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "home", PC)
    manifest = load_manifest(BSIDE)
    _pull(config, "q8_0", monkeypatch)
    hub = FakeHub(chunks=1)
    hub.absent.add("qwen3.5-4b-bside-BF16.gguf")
    with pytest.raises(weights.WeightsError, match="BF16"):
        _pull(config, "bf16", monkeypatch, hub)
    assert weights.installed(config, manifest, manifest.spec(CUDA_LINUX, "q8_0")) is not None


# ---- a load ------------------------------------------------------------------------------------


@pytest.fixture
def llama_ready(fake_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crucible.accelerator.refuse_if_card_lacks", lambda **_: None)


def test_victoria_s_laptop_loads_q8_0_when_no_form_is_named(
    home: Path, llama_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(home, VICTORIA)
    _pull(config, "q8_0", monkeypatch)
    _, spec, _ = _require_loadable(config, _backend(VICTORIA), BSIDE)
    assert (spec.form, spec.file) == ("q8_0", "qwen3.5-4b-bside-Q8_0.gguf")


def test_the_pc_with_only_q8_0_pulled_says_it_needs_a_pull_and_never_loads_the_smaller(
    home: Path, llama_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(home, PC)
    _pull(config, "q8_0", monkeypatch)
    with pytest.raises(ApiError) as refused:
        _require_loadable(config, _backend(PC), BSIDE)
    error = refused.value
    assert error.code == "model_not_installed", "install-on-submit pulls the form this card takes"
    assert "bf16 form, the one this card takes" in error.message
    assert "q8_0 is installed, and is not loaded in its place" in error.message
    assert "`crucible models pull qwen3.5-4b-bside`" in error.message
    assert error.details is not None
    assert (error.details["form"], error.details["installed_forms"]) == ("bf16", ["q8_0"])


def test_a_named_form_is_loaded_and_a_named_form_not_pulled_is_refused_with_its_command(
    home: Path, llama_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(home, PC)
    _pull(config, "q8_0", monkeypatch)
    _, spec, _ = _require_loadable(config, _backend(PC), BSIDE, "q8_0")
    assert spec.form == "q8_0"

    laptop = _config(home / "laptop", VICTORIA)
    _pull(laptop, "q8_0", monkeypatch)
    with pytest.raises(ApiError) as larger:
        _require_loadable(laptop, _backend(VICTORIA), BSIDE, "bf16")
    assert larger.value.code == "insufficient_memory", "a named form is tried, and this one never fits"
    assert "9.3 GiB" in larger.value.message and "8.0 GiB in total" in larger.value.message

    with pytest.raises(ApiError) as unknown:
        _require_loadable(config, _backend(PC), BSIDE, "q4")
    assert unknown.value.code == "unknown_form"
    assert unknown.value.details == {"model": BSIDE, "form": "q4", "forms": ["bf16", "q8_0"]}


def test_a_named_form_this_card_does_not_take_is_refused_when_not_pulled(
    home: Path, llama_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(home, PC)
    _pull(config, "bf16", monkeypatch)
    with pytest.raises(ApiError) as refused:
        _require_loadable(config, _backend(PC), BSIDE, "q8_0")
    assert refused.value.code == "form_not_installed", "not the form this card takes: no auto-pull"
    assert "`crucible models pull qwen3.5-4b-bside --form q8_0`" in refused.value.message
    assert "bf16 is installed" in refused.value.message


def test_the_models_row_lists_each_form_and_which_this_card_takes(
    home: Path, llama_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(home, PC)
    _pull(config, "q8_0", monkeypatch)
    rows = {row["id"]: row for row in model_rows(config, _backend(PC), Residency(config))}
    row = rows[BSIDE]
    assert row["form"] == "bf16" and "20.99 GiB" in row["form_reason"]
    assert [(f["name"], f["fits"], f["installed"], f["picked"]) for f in row["forms"]] == [
        ("bf16", True, False, True),
        ("q8_0", True, True, False),
    ]
    assert row["installed"] is False and row["loadable"] is False
    assert "q8_0 is installed, and is not what this card takes" in row["reason"]
    assert row["memory_bytes_estimate"] == 9_952_490_976, "the row's numbers are the pick's"
    assert rows["qwen3.5-9b"]["forms"] is None


# ---- doctor ------------------------------------------------------------------------------------


def test_doctor_says_when_a_host_holds_another_form_than_the_one_it_takes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(home, PC)
    host = doctor.Host(home, _backend(PC), None, config, None, None)
    assert doctor.check_model_forms(host).findings == ()

    _pull(config, "q8_0", monkeypatch)
    section = doctor.check_model_forms(host)
    [finding] = section.findings
    assert finding.code == "model_form_not_installed"
    assert finding.fix == "crucible models pull qwen3.5-4b-bside"
    assert "takes the bf16 form" in finding.message and "q8_0 is installed instead" in finding.message
    [entry] = section.facts["model_forms"]
    assert (entry["form"], entry["installed_forms"]) == ("bf16", ["q8_0"])

    laptop = _config(home / "laptop", VICTORIA)
    _pull(laptop, "q8_0", monkeypatch)
    on_laptop = doctor.Host(home / "laptop", _backend(VICTORIA), None, laptop, None, None)
    assert doctor.check_model_forms(on_laptop).findings == ()


# ---- a chat that names a form ------------------------------------------------------------------


def test_take_form_takes_it_off_the_body_and_refuses_a_non_string() -> None:
    body: dict[str, Any] = {"model": BSIDE, "form": "q8_0"}
    assert take_form(body) == "q8_0" and "form" not in body
    assert take_form({"model": BSIDE}) is None
    with pytest.raises(ApiError) as refused:
        take_form({"form": ""})
    assert refused.value.code == "invalid_request"


def test_a_resident_form_serves_a_call_naming_no_form_and_only_its_own_name() -> None:
    class Resident:
        model_id = BSIDE
        form = "q8_0"

    assert serves_model(Resident(), BSIDE)
    assert serves_model(Resident(), BSIDE, "q8_0")
    assert not serves_model(Resident(), BSIDE, "bf16")
    assert not serves_model(Resident(), "qwen3.5-9b")
    assert not serves_model(None, BSIDE)


def _place_both_forms(home: Path) -> None:
    manifest = load_manifest(BSIDE)
    block = manifest.block(CUDA_LINUX)
    directory = home / "models" / BSIDE / CUDA_LINUX
    directory.mkdir(parents=True, exist_ok=True)
    for name in block.form_names:
        (directory / block.form_named(name, BSIDE).file).write_bytes(b"GGUF")
    (directory / "crucible-pull.json").write_text(json.dumps({
        "model": BSIDE, "backend": CUDA_LINUX, "hf_repo": block.hf_repo,
        "revision": block.revision,
        "files": [block.form_named(name, BSIDE).file for name in block.form_names], "bytes": 8,
        "seconds": 1.0, "pulled": "2026-10-10T00:00:00+0000",
    }), encoding="utf-8")


def _chat(base: str, auth: dict[str, str], **extra: Any) -> httpx.Response:
    return httpx.post(
        f"{base}/v1/openai/chat/completions",
        headers=auth,
        json={"model": BSIDE, "messages": [{"role": "user", "content": "[describe] x"}], **extra},
        timeout=60.0,
    )


def _resident_form(base: str, auth: dict[str, str]) -> Any:
    listed = httpx.get(f"{base}/v1/openai/models", headers=auth, timeout=30.0).json()["data"]
    return [row.get("form") for row in listed if row["id"] == BSIDE]


def _file_of(engine: FakeEngine) -> str:
    return Path(engine.args[engine.args.index("-m") + 1]).name


def test_a_chat_naming_a_form_gets_it_and_one_naming_none_takes_what_is_resident(
    home: Path,
    make_app: Callable[..., Any],
    fake_env: Path,
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.accelerator.refuse_if_card_lacks", lambda **_: None)
    engines = engine_factory()
    _place_both_forms(home)
    with serve(make_app(enable_llm=True)) as base:
        unknown = _chat(base, auth, form="q4")
        assert unknown.status_code == 400
        assert unknown.json()["error"]["code"] == "unknown_form"
        assert engines == [], "an unknown form is refused before anything loads"

        named = _chat(base, auth, form="q8_0")
        assert named.status_code == 200, named.text
        assert _file_of(engines[-1]) == "qwen3.5-4b-bside-Q8_0.gguf"
        last = engines[-1].last_request
        assert last is not None and "form" not in last, "form is Crucible's, never the engine's"

        plain = _chat(base, auth)
        assert plain.status_code == 200, plain.text
        assert _file_of(engines[-1]) == "qwen3.5-4b-bside-BF16.gguf", (
            "loaded for a call naming no form: the form this card takes"
        )

        run_job(base, auth, type="load-model", model=BSIDE, params={"form": "q8_0"})
        built = len(engines)
        refused = _chat(base, auth, form="bf16", queue=False)
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "model_not_resident"
        assert refused.json()["error"]["details"]["resident_form"] == "q8_0"

        resident = _chat(base, auth)
        assert resident.status_code == 200, resident.text
        assert len(engines) == built, "a call naming no form takes the form on the card"
        assert _file_of(engines[-1]) == "qwen3.5-4b-bside-Q8_0.gguf"

        other = _chat(base, auth, form="bf16")
        assert other.status_code == 200, other.text
        assert len(engines) == built + 1, "another form of the same id is a reload"
        assert _file_of(engines[-1]) == "qwen3.5-4b-bside-BF16.gguf"


def test_load_model_takes_params_form(
    home: Path,
    make_app: Callable[..., Any],
    fake_env: Path,
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.accelerator.refuse_if_card_lacks", lambda **_: None)
    engines_built = engine_factory()
    _place_both_forms(home)
    with serve(make_app(enable_llm=True)) as base:
        run_job(base, auth, type="load-model", model=BSIDE, params={"form": "q8_0"})
        assert _resident_form(base, auth) == ["q8_0"]
        run_job(base, auth, type="load-model", model=BSIDE)
        assert _resident_form(base, auth) == ["bf16"], "no form: the one this card takes"
        rows = {row["id"]: row for row in httpx.get(f"{base}/v1/models", headers=auth, timeout=30.0).json()}
        assert [f["resident"] for f in rows[BSIDE]["forms"]] == [True, False]
        assert len(engines_built) == 2
        refused = httpx.post(
            f"{base}/v1/jobs", headers=auth,
            json={"type": "load-model", "model": BSIDE, "params": {"form": "q4"}}, timeout=30.0,
        )
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "unknown_form"


def test_fake_backend_is_the_pc_s_card() -> None:
    assert FAKE_BACKEND.gpu.vram_bytes == PC.total_bytes
    assert host_fit_of(type("C", (), {"desktop_allowance_bytes": 3 * GIB})(), FAKE_BACKEND) == HostFit(
        CUDA_LINUX, FAKE_BACKEND.gpu.name, PC.total_bytes, 3 * GIB
    )


def test_a_decision_naming_an_unknown_form_is_refused_before_it_waits(
    make_client: Callable[..., Any], auth: dict[str, str]
) -> None:
    with make_client(enable_llm=True) as client:
        refused = client.post(
            "/v1/decide",
            headers=auth,
            json={
                "model": BSIDE,
                "form": "q4",
                "state": "a song",
                "questions": {"sad": {"type": "yesno", "instructions": "Is it sad?"}},
            },
        )
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "unknown_form"
    assert refused.json()["error"]["details"]["forms"] == ["bf16", "q8_0"]
