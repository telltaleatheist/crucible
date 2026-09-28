from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import (
    capabilityclasses,
    capabilitywords,
    classnames,
    config,
    enginespec,
    manifests,
    modules,
    narratorengines,
    tomltable,
    upstreamrecord,
    voices,
    weights,
)
from crucible.alignmodels import load_all_align_manifests
from crucible.asrmodels import load_all_asr_manifests
from crucible.denoisemodels import load_all_denoise_manifests
from crucible.errors import CrucibleError
from crucible.rvcmodels import load_all_rvc_manifests

REPO = Path(__file__).resolve().parents[1]


def _loaded_after(module: str) -> set[str]:
    probe = (
        "import sys, importlib; "
        f"importlib.import_module({module!r}); "
        "print('\\n'.join(sorted(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(out.split())


def test_config_imports_only_leaves() -> None:
    loaded = _loaded_after("crucible.config")
    for heavy in (
        "crucible.voices",
        "crucible.verdict",
        "crucible.manifests",
        "crucible.upstreams",
        "crucible.engines",
        "httpx",
    ):
        assert heavy not in loaded, heavy


@pytest.mark.parametrize(
    "module", ["crucible.verdict", "crucible.installplan", "crucible.capabilityquery"]
)
def test_capability_loads_no_engine_adapter_and_no_decide_door(module: str) -> None:
    loaded = _loaded_after(module)
    assert "crucible.engines" not in loaded
    assert "crucible.decide" not in loaded
    assert "pydantic" not in loaded


def test_weights_reaches_no_manifest_module() -> None:
    loaded = _loaded_after("crucible.weights")
    assert "crucible.voices" not in loaded
    assert "crucible.manifests" not in loaded


REMOVED_RE_EXPORTS = {
    "crucible.config": (
        "desktop_reserve_words",
        "declared_tts_footprints",
        "TTS_ESTIMATE_BASES",
        "DESKTOP_BASIS_MEASURED",
    ),
    "crucible.voices": ("_parse", "load_all_voices", "load_voice", "unserved_pins"),
    "crucible.voicerepo": ("_parse_pins",),
    "crucible.decide": ("UNSTATED_ENGINE_CONCURRENCY",),
    "crucible.engines.base": ("str_flag",),
    "crucible.engines.vllm": ("bf16_fallback", "card_args", "dtype_of", "ENGINE_NAME"),
    "crucible.ttsplan": ("HIGGS_ENGINE",),
    "crucible.upstreams": ("UPSTREAM_NAMES", "blank", "split_model", "settings_entry"),
}


@pytest.mark.parametrize("module", sorted(REMOVED_RE_EXPORTS))
def test_the_old_import_spots_no_longer_answer(module: str) -> None:
    loaded = importlib.import_module(module)
    for name in REMOVED_RE_EXPORTS[module]:
        assert not hasattr(loaded, name), f"{module}.{name} is back"


def test_higgs_v3_is_written_once() -> None:
    assert narratorengines.HIGGS_V3 == "higgs-v3"
    assert narratorengines.NARRATOR_ENGINES == frozenset(
        narratorengines.NARRATOR_ENGINE_SAMPLING
    )
    for footprint in narratorengines.declared_tts_footprints("cuda-linux"):
        assert footprint.engine in narratorengines.NARRATOR_ENGINES
    offenders = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "crucible").rglob("*.py")
        if path.name != "narratorengines.py"
        and '"higgs-v3"' in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], offenders


def test_classnames_agree_with_the_class_catalogue() -> None:
    assert classnames.CLASS_NAMES == tuple(entry.name for entry in capabilityclasses.CLASSES)
    assert set(classnames.ROUTABLE_CLASSES) <= set(classnames.SELECTABLE_CLASSES)


def test_a_config_route_for_a_class_that_is_not_routable_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(config.ConfigError, match="route_not_routable"):
        config._route_records({"routes": {"decide": "openai/gpt"}}, ())
    with pytest.raises(config.ConfigError, match="local_model_not_selectable"):
        config._local_model_records({"local_models": {"echo": "x"}})


@pytest.mark.parametrize(
    ("spec", "declared", "stated"),
    [
        (SimpleNamespace(engine="vllm", dtype="bfloat16"), "bfloat16", "bfloat16"),
        (
            SimpleNamespace(engine="vllm", engine_args=("--dtype", "bfloat16")),
            "bfloat16",
            "bfloat16",
        ),
        (
            SimpleNamespace(engine="vllm", engine_args=("--dtype=float16",)),
            "float16",
            "float16",
        ),
        (SimpleNamespace(engine="vllm", engine_args=()), None, "auto"),
        (SimpleNamespace(engine="vllm", dtype="auto"), None, "auto"),
        (SimpleNamespace(engine="mlx-lm", dtype="bfloat16"), "bfloat16", "bfloat16"),
    ],
)
def test_one_rule_says_which_dtype_a_spec_states(
    spec: object, declared: str | None, stated: str
) -> None:
    assert enginespec.declared_dtype(spec) == declared
    assert enginespec.stated_dtype(spec) == stated


def test_a_bf16_vllm_block_falls_back_to_fp16_only_where_bf16_is_absent() -> None:
    spec = SimpleNamespace(engine="vllm", engine_args=("--dtype", "bfloat16"))
    no_bf16 = SimpleNamespace(has=lambda feature: False)
    has_all = SimpleNamespace(has=lambda feature: True)
    assert enginespec.bf16_fallback(spec) == "float16"
    assert enginespec.run_dtype(spec, no_bf16) == "float16"
    assert enginespec.run_dtype(spec, has_all) == "bfloat16"
    assert enginespec.card_args(spec, no_bf16) == (
        "--dtype",
        "float16",
        "--enforce-eager",
    )
    mlx = SimpleNamespace(engine="mlx-lm", dtype="bfloat16")
    assert enginespec.bf16_fallback(mlx) is None
    assert enginespec.card_needs(mlx) == ()


class _TableError(CrucibleError):
    ...


def test_check_table_refuses_unknown_missing_and_mistyped_keys() -> None:
    required = {"id": str, "n": int}
    with pytest.raises(_TableError, match=r"unknown key\(s\) \['x'\]"):
        tomltable.check_table("t", {"id": "a", "n": 1, "x": 0}, required, error=_TableError)
    with pytest.raises(_TableError, match=r"missing required key\(s\) \['n'\]"):
        tomltable.check_table("t", {"id": "a"}, required, error=_TableError)
    with pytest.raises(_TableError, match="n must be int, got bool"):
        tomltable.check_table("t", {"id": "a", "n": True}, required, error=_TableError)
    tomltable.check_table("t", {"id": "a", "n": 1}, required, error=_TableError)


def test_a_voice_id_is_at_most_64_characters_everywhere() -> None:
    assert tomltable.VOICE_ID_PATTERN.match("a" * 64)
    assert not tomltable.VOICE_ID_PATTERN.match("a" * 65)
    with pytest.raises(voices.VoiceError, match="at most 64"):
        voices.home_voice_path("a" * 65)
    from crucible.voicerepo import parse_pins

    text = f'["{"a" * 65}"]\nhf_repo = "o/n"\nrevision = "{"0" * 40}"\n'
    with pytest.raises(voices.VoiceError, match="at most 64"):
        parse_pins(text, Path("pins.toml"))


def test_the_upstream_offer_names_every_upstream() -> None:
    for name in upstreamrecord.UPSTREAM_NAMES:
        assert upstreamrecord.UPSTREAM_DISPLAY[name] in capabilitywords.UPSTREAM_OFFER
    assert "add an API key" in capabilitywords.UPSTREAM_OFFER


def test_every_weights_subject_names_its_pull_command_and_its_aliases() -> None:
    kinds = {
        "crucible models pull": (
            list(manifests.load_all_manifests().values())
            + list(load_all_asr_manifests().values())
            + list(load_all_align_manifests().values())
        ),
        "crucible rvc pull": list(load_all_rvc_manifests().values()),
        "crucible denoise pull": list(load_all_denoise_manifests().values()),
    }
    for command, found in kinds.items():
        assert found, command
        for manifest in found:
            assert manifest.pull_command == f"{command} {manifest.id}"
            assert isinstance(manifest.aliases(), tuple)
            assert isinstance(manifest, weights.WeightsSubject)
            assert manifest.weights_family in weights.FAMILY_NOUNS
    for manifest in manifests.load_all_manifests().values():
        assert manifest.aliases() == manifests.aliases_of(manifest)


def test_a_denoise_model_not_installed_names_its_own_pull_command(
    tmp_path: Path,
) -> None:
    separator = next(iter(load_all_denoise_manifests().values()))
    backend = sorted(separator.backends)[0]
    home = SimpleNamespace(home=tmp_path)
    with pytest.raises(weights.WeightsError, match=separator.pull_command):
        weights.require_installed(home, separator, separator.spec(backend))


def test_a_class_is_checked_once_and_resolved_from_the_same_table() -> None:
    served = capabilityclasses.models_by_class()
    assert modules.check_class("clean") == served["clean"]
    for model_id in sorted(set().union(*served.values())):
        assert set(capabilityclasses.classes_for_model(model_id)) == {
            name for name, ids in served.items() if model_id in ids
        }
    with pytest.raises(modules.ModuleError, match=r"\[\[subjects\]\]"):
        modules.check_class("tts")
    with pytest.raises(modules.ModuleError, match="not a capability class"):
        modules.check_class("transcribe")
