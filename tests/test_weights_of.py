"""One copy on disk, two fit rows in the catalog (PHASE22-DECIDE.md section 2.9).

Owen, 2026-09-23: *"One copy on disk, two fit rows in the catalog — i think this
is a fine way to do it."* A model manifest may declare `[model] weights_of =
"<base id>"`: an ALIAS, whose weights are its base's download in its base's
folder, and which owns on disk only the files its block names beyond the
base's. These tests hold every rule of that by the name it is refused with,
drive the real store against `tests/fake_hub.py`, and read the real rows.

Nothing here touches the network, a card or an engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import huggingface_hub
import pytest
from fastapi.testclient import TestClient

from crucible import weights
from crucible.backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN, Backend, Gpu
from crucible.capability import BY_NAME, classes_for_model
from crucible.config import Config
from crucible.manifests import (
    ManifestError,
    aliases_of,
    load_all_manifests,
    load_manifest,
    manifests_dir,
)

from .fake_hub import CHUNK, FakeHub

GIB = 1024 ** 3
BASE = "qwen3.5-9b"
ALIAS = "qwen3.5-9b-vl"
#: TWO since 2026-09-23. `qwen3.8-27b-8bit-vl` was served on cuda-linux alone
#: and went with its base's cuda-linux arm (Owen: *"we shouldnt have an 8 bit
#: 27b on here. waste of space, wont fit in the gpu"*); the test below that
#: rebuilds it shows the loader would refuse it by name if it came back.
ALIASES = {
    "qwen3.5-9b-vl": "qwen3.5-9b",
    "qwen3.8-27b-4bit-vl": "qwen3.8-27b-4bit",
}
MMPROJ = "mmproj-F16.gguf"


# ------------------------------------------------------------------ fixtures


def _config(home: Path, backend_kind: str) -> Config:
    home.mkdir(parents=True, exist_ok=True)
    return Config(
        path=home / "config.toml",
        home=home,
        name="crucible@weights-of",
        host="127.0.0.1",
        port=7101,
        token="t",
        backend_kind=backend_kind,
        enable_echo=True,
        enable_llm=True,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=False,
        desktop_allowance_bytes=3 * GIB,
        capability=None,
    )


def _backend(kind: str) -> Backend:
    return Backend(
        kind=kind,
        platform="win32" if kind == LLAMA_WINDOWS else "linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="RTX 3090 Ti", vram_bytes=25_757_220_864),
        detail="test double",
    )


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub(chunks=2)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    return fake


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    """A models/ tree holding the shipped 9B, for aliases written by hand."""
    root = tmp_path / "models"
    root.mkdir()
    shipped = manifests_dir()
    for model_id in (BASE, "qwen3.5-4b"):
        (root / f"{model_id}.toml").write_text(
            (shipped / f"{model_id}.toml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return root


def _alias_text(**overrides: Any) -> str:
    """The shipped 9B-vl, with lines swapped out. Keys are TOML lines."""
    text = (manifests_dir() / f"{ALIAS}.toml").read_text(encoding="utf-8")
    for old, new in overrides.items():
        assert old in text, old
        text = text.replace(old, new, 1)
    return text


def _write_alias(root: Path, text: str, model_id: str = ALIAS) -> None:
    (root / f"{model_id}.toml").write_text(text, encoding="utf-8")


def _refusal(root: Path, model_id: str = ALIAS) -> str:
    with pytest.raises(ManifestError) as caught:
        load_manifest(model_id, root)
    return str(caught.value)


# ------------------------------------------------------------ the loader


def test_a_well_formed_alias_loads_with_its_base_attached(catalog_dir: Path) -> None:
    _write_alias(catalog_dir, _alias_text())
    alias = load_manifest(ALIAS, catalog_dir)
    assert alias.weights_of == BASE
    assert alias.weights_base is not None and alias.weights_base.id == BASE
    assert alias.store_id == BASE
    # A base names no base and owns its own folder.
    base = load_manifest(BASE, catalog_dir)
    assert base.weights_of is None and base.weights_base is None
    assert base.store_id == BASE
    assert aliases_of(base) == (alias,)
    assert aliases_of(alias) == ()


def test_an_alias_of_a_model_the_catalog_does_not_ship_is_weights_of_unknown(
    catalog_dir: Path,
) -> None:
    _write_alias(
        catalog_dir,
        _alias_text(**{'weights_of = "qwen3.5-9b"': 'weights_of = "qwen3.5-10b"'}),
    )
    message = _refusal(catalog_dir)
    assert "weights_of_unknown" in message
    assert "qwen3.5-10b" in message


def test_an_alias_of_an_alias_is_weights_of_chain(catalog_dir: Path) -> None:
    _write_alias(catalog_dir, _alias_text())
    second = _alias_text(
        **{
            'id = "qwen3.5-9b-vl"': 'id = "qwen3.5-9b-vl2"',
            'weights_of = "qwen3.5-9b"': 'weights_of = "qwen3.5-9b-vl"',
        }
    )
    _write_alias(catalog_dir, second, "qwen3.5-9b-vl2")
    message = _refusal(catalog_dir, "qwen3.5-9b-vl2")
    assert "weights_of_chain" in message
    assert "qwen3.5-9b-vl" in message


def test_a_base_aliased_to_something_else_is_weights_of_chain(
    catalog_dir: Path,
) -> None:
    """The mirror: the BASE declares a `weights_of`, so the alias that names it
    names a folder with two owners. Also closes a cycle without recursing."""
    base = catalog_dir / f"{BASE}.toml"
    text = base.read_text(encoding="utf-8")
    # An alias carries no [local] (`weights_of_local`), so the base loses its
    # own before it is made one — otherwise that refusal would answer first.
    text = text[: text.index("[local]\n")] + text[text.index("[backends.cuda-linux]"):]
    text = text.replace(
        'id = "qwen3.5-9b"\n', 'id = "qwen3.5-9b"\nweights_of = "qwen3.5-4b"\n', 1
    )
    base.write_text(text, encoding="utf-8")
    _write_alias(catalog_dir, _alias_text())
    message = _refusal(catalog_dir)
    assert "weights_of_chain" in message


def test_an_alias_of_itself_is_weights_of_chain(catalog_dir: Path) -> None:
    _write_alias(
        catalog_dir,
        _alias_text(**{'weights_of = "qwen3.5-9b"': 'weights_of = "qwen3.5-9b-vl"'}),
    )
    assert "weights_of_chain" in _refusal(catalog_dir)


@pytest.mark.parametrize(
    "old, new, named",
    [
        (
            'revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"',
            'revision = "0000000000000000000000000000000000000000"',
            "revision",
        ),
        ('hf_repo = "Qwen/Qwen3.5-9B"', 'hf_repo = "Qwen/Qwen3.5-9B-Base"', "hf_repo"),
        (
            'file = "Qwen3.5-9B-Q8_0.gguf"',
            'file = "Qwen3.5-9B-Q4_K_M.gguf"',
            "file",
        ),
    ],
)
def test_a_different_pin_on_a_shared_backend_is_weights_of_pin_mismatch(
    catalog_dir: Path, old: str, new: str, named: str
) -> None:
    _write_alias(catalog_dir, _alias_text(**{old: new}))
    message = _refusal(catalog_dir)
    assert "weights_of_pin_mismatch" in message
    assert named in message


@pytest.mark.parametrize(
    "old, new, named",
    [
        ("params_b = 9\n", "params_b = 8\n", "params_b"),
        ('family = "qwen3.5"', 'family = "qwen3.6"', "family"),
        ("trained_context = 262144", "trained_context = 131072", "trained_context"),
        ("[defaults]\nthinking = false", "[defaults]\nthinking = true", "defaults"),
    ],
)
def test_a_different_shared_fact_is_weights_of_fact_mismatch(
    catalog_dir: Path, old: str, new: str, named: str
) -> None:
    _write_alias(catalog_dir, _alias_text(**{old: new}))
    message = _refusal(catalog_dir)
    assert "weights_of_fact_mismatch" in message
    assert named in message


def test_an_alias_with_no_defaults_when_its_base_has_them_is_a_fact_mismatch(
    catalog_dir: Path,
) -> None:
    _write_alias(catalog_dir, _alias_text(**{"[defaults]\nthinking = false\n": ""}))
    assert "weights_of_fact_mismatch" in _refusal(catalog_dir)


def test_what_the_alias_serves_is_its_own(catalog_dir: Path) -> None:
    """modalities, serves, context_default, display, description, engine_args
    and memory differ from the base's — the point of the alias — and load."""
    _write_alias(catalog_dir, _alias_text())
    alias = load_manifest(ALIAS, catalog_dir)
    base = load_manifest(BASE, catalog_dir)
    assert alias.modalities != base.modalities
    assert alias.serves(CUDA_LINUX) != base.serves(CUDA_LINUX)
    assert alias.display != base.display
    assert alias.spec(CUDA_LINUX).engine_args != base.spec(CUDA_LINUX).engine_args
    assert alias.spec(CUDA_LINUX).memory != base.spec(CUDA_LINUX).memory


def test_a_backend_the_base_does_not_declare_is_weights_of_backend_missing(
    catalog_dir: Path,
) -> None:
    base = catalog_dir / f"{BASE}.toml"
    text = base.read_text(encoding="utf-8")
    cut = text.index("# ---------------------------------------------------------------------------\n# PHASE15-HOST.md sections 0 and 3.10")
    base.write_text(text[:cut], encoding="utf-8")
    assert "llama-windows" not in load_manifest(BASE, catalog_dir).backends
    _write_alias(catalog_dir, _alias_text())
    message = _refusal(catalog_dir)
    assert "weights_of_backend_missing" in message
    assert "llama-windows" in message


def test_an_8bit_27b_vision_alias_on_cuda_linux_is_weights_of_backend_missing(
    catalog_dir: Path,
) -> None:
    """The retired `qwen3.8-27b-8bit-vl`, rebuilt as it shipped until
    2026-09-23: one cuda-linux block over the FP8 repo. Its base has no
    cuda-linux block any more (Mac only, by Owen's ruling), so there is no
    download for the alias to share and the loader says so by name."""
    shipped = manifests_dir()
    (catalog_dir / "qwen3.8-27b-8bit.toml").write_text(
        (shipped / "qwen3.8-27b-8bit.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_alias(
        catalog_dir,
        """
[model]
id = "qwen3.8-27b-8bit-vl"
weights_of = "qwen3.8-27b-8bit"
family = "qwen3.8"
params_b = 27
trained_context = 262144
context_default = 12288
modalities = ["text", "image"]
display = "Qwen 3.8 · 27B (8-bit) · with vision"
description = "retired"

[backends.cuda-linux]
engine = "vllm"
hf_repo = "Qwen/Qwen3.8-27B-FP8"
revision = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
serves = ["text", "image"]
memory_bytes_estimate = 50_725_919_915
engine_args = []
""",
        model_id="qwen3.8-27b-8bit-vl",
    )
    message = _refusal(catalog_dir, "qwen3.8-27b-8bit-vl")
    assert "weights_of_backend_missing" in message
    assert "cuda-linux" in message


def test_an_alias_with_a_local_form_is_weights_of_local(catalog_dir: Path) -> None:
    _write_alias(
        catalog_dir,
        _alias_text()
        + '\n[local]\nkind = "ollama"\ntag = "qwen3.5:9b-bf16"\n'
        "download_bytes = 1\nneeds_bytes = 2\nneeds_basis = \"declared\"\n",
    )
    assert "weights_of_local" in _refusal(catalog_dir)


# ------------------------------------------------------------- the two aliases


@pytest.mark.parametrize("alias_id, base_id", sorted(ALIASES.items()))
def test_the_vision_forms_load_and_agree_with_their_bases(
    alias_id: str, base_id: str
) -> None:
    alias = load_manifest(alias_id)
    base = load_manifest(base_id)
    assert alias.weights_of == base_id
    for fact in ("family", "params_b", "trained_context", "defaults"):
        assert getattr(alias, fact) == getattr(base, fact), fact
    assert alias.modalities == ("text", "image")
    assert alias.display == f"{base.display} · with vision"
    assert alias.local is None
    # No mlx-darwin block: mlx-lm serves text, and the base covers the Mac.
    assert MLX_DARWIN not in alias.backends
    for kind, spec in alias.backends.items():
        base_spec = base.spec(kind)
        assert (spec.hf_repo, spec.revision, spec.file) == (
            base_spec.hf_repo, base_spec.revision, base_spec.file,
        )
        assert spec.serves == ("text", "image")
        if kind == CUDA_LINUX:
            assert "--language-model-only" not in spec.engine_args
            assert "--skip-mm-profiling" not in spec.engine_args
            flag = spec.engine_args.index("--limit-mm-per-prompt")
            assert json.loads(spec.engine_args[flag + 1]) == {"image": 8, "video": 0}
            # The image reserve, 1.90 GiB, is what the intercept gains.
            assert (
                spec.memory.overhead_bytes - base_spec.memory.overhead_bytes
                == 2_040_109_466
            )
        if kind == LLAMA_WINDOWS:
            assert spec.mmproj == MMPROJ
            assert alias.extra_files(kind) == (MMPROJ,)
        else:
            assert alias.extra_files(kind) == ()


def test_the_vl_aliases_carry_the_tower_their_text_bases_do_not() -> None:
    """The 9B's calibrated weights are text-only, so the tower is added. The
    27B-4bit's base term is text-only too since 2026-09-23 (its 17.68 GiB card
    figure less the 921_460_192 B tower `--language-model-only` no longer
    loads), so its alias adds that tower back. (The 8-bit's alias is gone with
    its base's cuda-linux arm, 2026-09-23.)"""
    nine = load_manifest(ALIAS).spec(CUDA_LINUX).memory
    assert nine.weights_bytes == load_manifest(BASE).spec(CUDA_LINUX).memory.weights_bytes + 912_020_960
    four = load_manifest("qwen3.8-27b-4bit-vl")
    assert (
        four.spec(CUDA_LINUX).memory.weights_bytes
        == four.weights_base.spec(CUDA_LINUX).memory.weights_bytes + 921_460_192
    )


def test_the_nine_b_vl_leaves_about_1700_tokens_on_the_3090ti() -> None:
    """PHASE22 section 7.3's ~2,000, to the byte, against the fit budget."""
    terms = load_manifest(ALIAS).spec(CUDA_LINUX).memory
    budget = 25_757_220_864 - 3 * GIB
    assert terms.max_context(available_bytes=budget, concurrency=1) == 1_700
    terms = load_manifest("qwen3.8-27b-4bit-vl").spec(CUDA_LINUX).memory
    assert terms.max_context(available_bytes=budget, concurrency=1) == 0


def test_decide_lists_the_aliases_and_the_text_classes_do_not() -> None:
    expected_decide = {
        # No 8-bit 27B, text or vision: Mac only since 2026-09-23.
        CUDA_LINUX: [
            "qwen3.8-27b-4bit-vl", "qwen3.5-9b-vl", "qwen3.8-27b-4bit",
            "qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b",
        ],
        MLX_DARWIN: [
            "qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b", "qwen3.5-4b",
            "qwen3.5-2b", "qwen3.5-0.8b",
        ],
        LLAMA_WINDOWS: [
            "qwen3.8-27b-4bit-vl", "qwen3.8-27b-4bit", "qwen3.5-9b-vl",
            "qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b",
        ],
    }
    text = {
        "clean": {k: ["qwen3.5-9b"] for k in (CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS)},
        "translate": {
            CUDA_LINUX: ["qwen3.8-27b-4bit", "qwen3.5-9b"],
            MLX_DARWIN: ["qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b"],
            LLAMA_WINDOWS: ["qwen3.8-27b-4bit", "qwen3.5-9b"],
        },
    }
    for backend, ids in expected_decide.items():
        assert [c.id for c in BY_NAME["decide"].candidates(backend)] == ids, backend
        assert [c.id for c in BY_NAME["clean"].candidates(backend)] == text["clean"][backend]
        for name in ("translate", "simplify", "analysis"):
            assert (
                [c.id for c in BY_NAME[name].candidates(backend)]
                == text["translate"][backend]
            ), (name, backend)
    for alias_id in ALIASES:
        assert classes_for_model(alias_id) == ("decide",)


def test_the_small_tiers_are_untouched() -> None:
    for model_id in ("qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b"):
        manifest = load_manifest(model_id)
        assert manifest.weights_of is None
        assert aliases_of(manifest) == ()
        assert classes_for_model(model_id) == ("decide",)


def test_every_alias_the_catalog_ships_is_one_of_the_two() -> None:
    shipped = {
        m.id: m.weights_of for m in load_all_manifests().values() if m.weights_of
    }
    assert shipped == ALIASES


# ------------------------------------------------------------ the store


def test_an_alias_s_folder_is_its_base_s(tmp_path: Path) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    for kind in (CUDA_LINUX, LLAMA_WINDOWS):
        assert weights.subject_dir(config, alias, kind) == weights.subject_dir(
            config, base, kind
        ) == config.home / "models" / BASE / kind


def test_pulling_the_alias_pulls_the_base_then_only_its_own_file(
    tmp_path: Path, hub: FakeHub
) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    found = weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    # The base as the base (its one GGUF), then ONLY the projector.
    assert hub.allowed == [["Qwen3.5-9B-Q8_0.gguf"], [MMPROJ]]
    assert found.path == config.home / "models" / BASE / LLAMA_WINDOWS
    # The alias owns the projector's bytes and nothing else.
    assert found.bytes == 2 * CHUNK
    assert weights.installed(config, base, base.spec(LLAMA_WINDOWS)) is not None
    assert not (config.home / "models" / ALIAS).exists()
    record = json.loads(
        weights.alias_record_path(config, alias, LLAMA_WINDOWS).read_text("utf-8")
    )
    assert record["weights_of"] == BASE and record["files"] == [MMPROJ]
    # A second pull of either is a no-op: nothing is downloaded twice.
    weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    weights.pull(config, base, base.spec(LLAMA_WINDOWS))
    assert len(hub.allowed) == 2


def test_with_the_base_already_pulled_the_alias_fetches_only_its_file(
    tmp_path: Path, hub: FakeHub
) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    weights.pull(config, base, base.spec(LLAMA_WINDOWS))
    # A base-only pull leaves the alias NOT installed, honestly.
    assert weights.installed(config, alias, alias.spec(LLAMA_WINDOWS)) is None
    with pytest.raises(weights.WeightsError, match=MMPROJ):
        weights.require_installed(config, alias, alias.spec(LLAMA_WINDOWS))
    hub.allowed.clear()
    weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    assert hub.allowed == [[MMPROJ]]


def test_on_a_whole_repo_backend_the_alias_downloads_nothing_of_its_own(
    tmp_path: Path, hub: FakeHub
) -> None:
    config = _config(tmp_path / "home", CUDA_LINUX)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    found = weights.pull(config, alias, alias.spec(CUDA_LINUX))
    assert hub.allowed == [None]
    assert found.bytes == 0
    assert found.path == weights.subject_dir(config, base, CUDA_LINUX)


def test_an_alias_with_no_base_installed_says_so(tmp_path: Path) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias = load_manifest(ALIAS)
    with pytest.raises(weights.WeightsError) as caught:
        weights.require_installed(config, alias, alias.spec(LLAMA_WINDOWS))
    assert f"shares the weights of {BASE!r}" in str(caught.value)
    assert f"crucible models pull {ALIAS}" in str(caught.value)


def test_removing_the_base_while_the_alias_holds_it_is_weights_shared(
    tmp_path: Path, hub: FakeHub
) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    with pytest.raises(weights.WeightsShared) as caught:
        weights.remove(config, base, base.spec(LLAMA_WINDOWS))
    assert caught.value.code == "weights_shared"
    assert caught.value.aliases == (ALIAS,)
    assert ALIAS in str(caught.value)
    # A forced re-pull of the base empties the folder: the same refusal.
    with pytest.raises(weights.WeightsShared):
        weights.pull(config, base, base.spec(LLAMA_WINDOWS), force=True)
    # Nothing went.
    assert weights.installed(config, alias, alias.spec(LLAMA_WINDOWS)) is not None


def test_removing_the_alias_takes_only_its_file_then_the_base_may_go(
    tmp_path: Path, hub: FakeHub
) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    folder = weights.subject_dir(config, base, LLAMA_WINDOWS)
    gone = weights.remove(config, alias, alias.spec(LLAMA_WINDOWS))
    assert gone == folder
    assert not (folder / MMPROJ).exists()
    assert not weights.alias_record_path(config, alias, LLAMA_WINDOWS).exists()
    assert (folder / "Qwen3.5-9B-Q8_0.gguf").is_file()
    assert weights.installed(config, base, base.spec(LLAMA_WINDOWS)) is not None
    assert weights.installed(config, alias, alias.spec(LLAMA_WINDOWS)) is None
    weights.remove(config, base, base.spec(LLAMA_WINDOWS))
    assert not folder.exists()


def test_on_cuda_linux_a_pulled_alias_also_holds_the_base(
    tmp_path: Path, hub: FakeHub
) -> None:
    """The record is what makes "an alias exists here" a fact on a backend
    where it owns no file; without it the base could never be refused."""
    config = _config(tmp_path / "home", CUDA_LINUX)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    weights.pull(config, base, base.spec(CUDA_LINUX))
    # Installed by the ruling's definition (base stamp + no extras), but never
    # pulled as itself: removing the base is not refused.
    assert weights.installed(config, alias, alias.spec(CUDA_LINUX)) is not None
    assert weights.aliases_holding(config, base, CUDA_LINUX) == ()
    weights.pull(config, alias, alias.spec(CUDA_LINUX))
    assert weights.aliases_holding(config, base, CUDA_LINUX) == (ALIAS,)
    with pytest.raises(weights.WeightsShared):
        weights.remove(config, base, base.spec(CUDA_LINUX))
    weights.remove(config, alias, alias.spec(CUDA_LINUX))
    weights.remove(config, base, base.spec(CUDA_LINUX))


def test_a_cancelled_projector_pull_leaves_the_base_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: FakeHub
) -> None:
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    alias, base = load_manifest(ALIAS), load_manifest(BASE)
    weights.pull(config, base, base.spec(LLAMA_WINDOWS))

    def cancel(done: int, total: int | None, name: str) -> None:
        raise weights.PullCancelled("the caller cancelled")

    with pytest.raises(weights.PullCancelled):
        weights.pull(config, alias, alias.spec(LLAMA_WINDOWS), on_progress=cancel)
    folder = weights.subject_dir(config, base, LLAMA_WINDOWS)
    assert not (folder / MMPROJ).exists()
    assert weights.installed(config, base, base.spec(LLAMA_WINDOWS)) is not None


# ------------------------------------------------------------- the rows


def _rows_through_the_api(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    backend: Backend,
    path: str,
) -> Any:
    with make_client(enable_llm=True, backend=backend) as client:
        response = client.get(path, headers=auth)
        assert response.status_code == 200, response.text
        return response.json()


def test_the_catalog_counts_the_download_once_on_the_base(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    hub: FakeHub,
) -> None:
    backend = _backend(LLAMA_WINDOWS)
    config = _config(home, LLAMA_WINDOWS)
    alias = load_manifest(ALIAS)
    base = load_manifest(BASE)
    weights.pull(config, base, base.spec(LLAMA_WINDOWS))

    rows = {
        (r["kind"], r["id"]): r
        for r in _rows_through_the_api(make_client, auth, backend, "/v1/catalog")["rows"]
    }
    base_row, alias_row = rows[("model", BASE)], rows[("model", ALIAS)]
    assert base_row["shares_weights_of"] is None
    assert base_row["missing_files"] is None
    assert alias_row["shares_weights_of"] == BASE
    # Base-only: the alias is not installed, and its row says which file.
    assert alias_row["installed"] is False
    assert alias_row["missing_files"] == [MMPROJ]
    # A projector's size is not declared by any manifest, so null, not a guess.
    assert alias_row["expected_bytes"] is None

    weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    rows = {
        (r["kind"], r["id"]): r
        for r in _rows_through_the_api(make_client, auth, backend, "/v1/catalog")["rows"]
    }
    base_row, alias_row = rows[("model", BASE)], rows[("model", ALIAS)]
    assert alias_row["installed"] is True
    assert alias_row["missing_files"] == []
    stamp = json.loads(
        (weights.subject_dir(config, base, LLAMA_WINDOWS) / weights.STAMP_NAME)
        .read_text("utf-8")
    )
    assert base_row["installed_bytes"] == stamp["bytes"]
    # ONCE: the alias's row counts only the projector.
    assert alias_row["installed_bytes"] == 2 * CHUNK


def test_a_no_extras_alias_expects_zero_bytes_of_its_own(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    rows = _rows_through_the_api(make_client, auth, _backend(CUDA_LINUX), "/v1/catalog")
    alias_row = next(r for r in rows["rows"] if r["id"] == ALIAS)
    assert alias_row["expected_bytes"] == 0
    assert alias_row["missing_files"] == []
    assert alias_row["shares_weights_of"] == BASE


def test_models_and_info_rows_carry_weights_of(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_env: Path
) -> None:
    models = {
        r["id"]: r
        for r in _rows_through_the_api(
            make_client, auth, _backend(CUDA_LINUX), "/v1/models"
        )
    }
    for model_id, row in models.items():
        assert row["weights_of"] == ALIASES.get(model_id), model_id
    info = _rows_through_the_api(make_client, auth, _backend(CUDA_LINUX), "/v1/info")
    llm = next(c for c in info["capabilities"] if c["job_type"] == "llm")
    assert {r["id"]: r["weights_of"] for r in llm["models"]} == {
        model_id: ALIASES.get(model_id) for model_id in models
    }
    # A not-installed alias's reason names the shared download, not a folder.
    assert f"shares the weights of {BASE!r}" in models[ALIAS]["reason"]


def test_an_alias_never_reads_as_held_in_the_ollama_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store reuse is base-only: Ollama's blob is the text GGUF with no
    projector, and the alias has no [local] to name a tag with."""
    from crucible.jobs.llm import model_rows
    from crucible.residency import Residency

    from .test_ollama_copy_row import a_store

    from crucible import ollamastore

    store = tmp_path / "ollama"
    a_store(store, "qwen3.5:9b-bf16")
    monkeypatch.setenv(ollamastore.STORE_ENV, str(store))
    config = _config(tmp_path / "home", LLAMA_WINDOWS)
    rows = {
        row["id"]: row
        for row in model_rows(config, _backend(LLAMA_WINDOWS), Residency(config))
    }
    assert rows[BASE]["ollama_copy"] is not None
    assert rows[ALIAS]["ollama_copy"] is None
    assert rows[ALIAS]["installed"] is False


def test_the_api_refuses_the_base_by_name_and_names_the_alias(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    hub: FakeHub,
) -> None:
    config = _config(home, CUDA_LINUX)
    alias = load_manifest(ALIAS)
    weights.pull(config, alias, alias.spec(CUDA_LINUX))
    with make_client(enable_llm=True) as client:
        response = client.delete(f"/v1/catalog/model/{BASE}", headers=auth)
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "weights_shared"
        assert error["details"]["aliases"] == [ALIAS]
        assert ALIAS in error["message"]
        # The alias goes (only what is its own), then the base may.
        assert client.delete(f"/v1/catalog/model/{ALIAS}", headers=auth).status_code == 204
        assert client.delete(f"/v1/catalog/model/{BASE}", headers=auth).status_code == 204


def test_an_alias_holding_the_card_holds_its_base_even_unpulled(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    hub: FakeHub,
) -> None:
    """On cuda-linux the alias is installed by the base's pull alone, with no
    record of its own — so `weights_shared` cannot see it. What CAN is the
    hold: a lease (or a residency) on the alias is a run reading the base's
    folder, and the door refuses `subject_in_use`, naming the alias."""
    config = _config(home, CUDA_LINUX)
    base = load_manifest(BASE)
    weights.pull(config, base, base.spec(CUDA_LINUX))
    with make_client(enable_llm=True) as client:
        client.app.state.leases.open(  # type: ignore[attr-defined]
            kind="llm", subject=ALIAS, act="decide", client="foundry/1",
            ttl_seconds=60,
        )
        response = client.delete(f"/v1/catalog/model/{BASE}", headers=auth)
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "subject_in_use"
    assert error["details"]["fact"] == "lease"
    assert ALIAS in error["details"]["who"]
    assert weights.installed(config, base, base.spec(CUDA_LINUX)) is not None


def test_the_cli_refuses_the_base_by_name(
    home: Path,
    hub: FakeHub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from crucible import cli

    config = _config(home, CUDA_LINUX)
    alias = load_manifest(ALIAS)
    weights.pull(config, alias, alias.spec(CUDA_LINUX))
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "detect_backend", lambda: _backend(CUDA_LINUX))

    class Args:
        kind = "model"
        id = BASE
        json = False

    assert cli.cmd_remove(Args()) != 0
    err = capsys.readouterr().err
    assert "weights_shared" in err and ALIAS in err
    Args.id = ALIAS
    assert cli.cmd_remove(Args()) == 0
    Args.id = BASE
    assert cli.cmd_remove(Args()) == 0


def test_the_windows_migration_removes_the_alias_before_its_base(
    tmp_path: Path, hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`qwen3.5-9b` sorts first, meets `weights_shared`, and goes on the next
    round once the alias has — the real catalog, the real store."""
    from crucible.host import installer
    from crucible.host.catalog import StoppedWindowsCatalog

    from .test_host import FakeCatalog, migration

    home = tmp_path / "home"
    config = _config(home, LLAMA_WINDOWS)
    alias = load_manifest(ALIAS)
    weights.pull(config, alias, alias.spec(LLAMA_WINDOWS))
    keys = {("model", BASE), ("model", ALIAS)}
    # The controller journals the cleanup before it builds the stopped
    # catalog (tests/test_host_migration_faults.py does the same).
    installer.record_cleanup(home, keys)
    stopped = StoppedWindowsCatalog(
        config, _backend(LLAMA_WINDOWS), installer.cleanup_subjects(home)
    )
    assert {row.key for row in stopped.installed_subjects()} == keys
    guest = FakeCatalog("guest", sorted(keys))
    events: list[installer.Event] = []
    migration(stopped, guest, events, home)._migrate_weights(allow_pull=False)
    assert stopped.installed_subjects() == []
    assert not weights.subject_dir(config, alias, LLAMA_WINDOWS).exists()
