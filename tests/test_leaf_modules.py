from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import (
    capability,
    capabilityrecord,
    classnames,
    config,
    decide,
    enginespec,
    manifests,
    modules,
    narratorengines,
    tomltable,
    upstreamrecord,
    upstreams,
    voices,
    weights,
)
from crucible.alignmodels import load_all_align_manifests
from crucible.asrmodels import load_all_asr_manifests
from crucible.denoisemodels import load_all_denoise_manifests
from crucible.engines import vllm
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
        "crucible.capability",
        "crucible.manifests",
        "crucible.upstreams",
        "crucible.engines",
        "httpx",
    ):
        assert heavy not in loaded, heavy


def test_capability_loads_no_engine_adapter_and_no_decide_door() -> None:
    loaded = _loaded_after("crucible.capability")
    assert "crucible.engines" not in loaded
    assert "crucible.decide" not in loaded
    assert "pydantic" not in loaded


def test_weights_reaches_no_manifest_module() -> None:
    loaded = _loaded_after("crucible.weights")
    assert "crucible.voices" not in loaded
    assert "crucible.manifests" not in loaded


def test_the_old_import_spots_hand_back_the_leaf_s_own_objects() -> None:
    assert config.CapabilityRecord is capabilityrecord.CapabilityRecord
    assert config.CapabilityRow is capabilityrecord.CapabilityRow
    assert config.desktop_reserve_words is capabilityrecord.desktop_reserve_words
    assert config.EngineFootprint is narratorengines.EngineFootprint
    assert config.declared_tts_footprints is narratorengines.declared_tts_footprints
    assert config.TTS_ESTIMATE_BASES is narratorengines.ESTIMATE_BASES
    assert voices.ESTIMATE_BASES is narratorengines.ESTIMATE_BASES
    assert voices.NARRATOR_ENGINE_SAMPLING is narratorengines.NARRATOR_ENGINE_SAMPLING
    assert capability.ROUTABLE_CLASSES is classnames.ROUTABLE_CLASSES
    assert capability.SELECTABLE_CLASSES is classnames.SELECTABLE_CLASSES
    assert decide.UNSTATED_ENGINE_CONCURRENCY == enginespec.UNSTATED_ENGINE_CONCURRENCY
    assert vllm.bf16_fallback is enginespec.bf16_fallback
    assert vllm.card_args is enginespec.card_args
    assert vllm.dtype_of is enginespec.dtype_of
    assert manifests.check_table is tomltable.check_table
    assert upstreams.UpstreamRecord is upstreamrecord.UpstreamRecord
    assert upstreams.UPSTREAM_NAMES is upstreamrecord.UPSTREAM_NAMES


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
    assert offenders in ([], ["crucible/jobenv.py"]), offenders


def test_classnames_agree_with_the_class_catalogue() -> None:
    assert classnames.CLASS_NAMES == tuple(entry.name for entry in capability.CLASSES)
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
    from crucible.voicerepo import _parse_pins

    text = f'["{"a" * 65}"]\nhf_repo = "o/n"\nrevision = "{"0" * 40}"\n'
    with pytest.raises(voices.VoiceError, match="at most 64"):
        _parse_pins(text, Path("pins.toml"))


def test_the_upstream_offer_names_every_upstream() -> None:
    for name in upstreamrecord.UPSTREAM_NAMES:
        assert upstreamrecord.UPSTREAM_DISPLAY[name] in capability.UPSTREAM_OFFER
    assert "add an API key" in capability.UPSTREAM_OFFER


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
    served = capability.models_by_class()
    assert modules.check_class("clean") == served["clean"]
    for model_id in sorted(set().union(*served.values())):
        assert set(capability.classes_for_model(model_id)) == {
            name for name, ids in served.items() if model_id in ids
        }
    with pytest.raises(modules.ModuleError, match=r"\[\[subjects\]\]"):
        modules.check_class("tts")
    with pytest.raises(modules.ModuleError, match="not a capability class"):
        modules.check_class("transcribe")
