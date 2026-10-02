from __future__ import annotations

import ast
from pathlib import Path

import pytest

from crucible import catalog, errors, voicecatalog, voicerepo, voices
from crucible.errors import CrucibleError, EngineError
from crucible.narratorengines import HIGGS_V3, EngineFootprint
from crucible.narratorvoices import NarratorVoicesError
from crucible.voicerepo import REPO_MANIFEST_NAME, Pin, merge, parse_repo_manifest

from .test_voice_repo_manifest import GOOD as REPO_GOOD

CRUCIBLE = Path(voices.__file__).resolve().parent


def _imported_by(module: str) -> set[str]:
    tree = ast.parse((CRUCIBLE / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module is None:
                found.update(alias.name for alias in node.names)
            else:
                found.add(node.module.split(".")[0])
    return found


def test_voices_and_voicerepo_never_import_the_catalog_above_them() -> None:
    assert not {"voicerepo", "voicecatalog", "catalog"} & _imported_by("voices")
    assert not {"voicecatalog", "catalog"} & _imported_by("voicerepo")
    assert {"voices", "voicerepo"} <= _imported_by("voicecatalog")


def test_the_old_names_on_voices_are_the_catalog_s_functions() -> None:
    assert voicecatalog.load_all_voices is voicecatalog.load_all_voices
    assert voicecatalog.load_voice is voicecatalog.load_voice
    assert voicecatalog.unserved_pins is voicecatalog.unserved_pins
    assert voicecatalog.voice_aliases_of is voicecatalog.voice_aliases_of
    assert voicerepo.parse_pins is voicerepo.parse_pins
    with pytest.raises(AttributeError):
        voices.__getattr__("no_such_name")


def test_a_base_voice_names_the_aliases_its_catalog_resolved() -> None:
    resolved = voicecatalog.resolve_weights_of(voicecatalog.engine_voices())
    base = resolved["higgs-default"]
    alias = resolved["zeroshot"]
    assert [a.id for a in base.aliases()] == ["zeroshot"]
    assert alias.aliases() == ()
    assert alias.weights_base is not None and alias.weights_base.id == "higgs-default"


def test_the_declared_voices_are_the_packaged_pins_and_the_engine_rows() -> None:
    declared = voicecatalog.declared_voice_ids()
    assert set(voicerepo.packaged_pins()) <= declared
    assert {"higgs-default", "zeroshot"} <= declared
    assert catalog.declared_ids()["voice"] == sorted(declared)


def test_a_shipped_voice_declares_its_own_backends_and_an_unknown_one_none() -> None:
    assert catalog.backends_declaring("voice", "zeroshot") == sorted(
        voicecatalog.engine_voices()["zeroshot"].backends
    )
    assert catalog.backends_declaring("voice", "nobody") == []


def test_a_packaged_pin_declares_the_arms_of_its_cached_repo_manifest(
    tmp_path: Path,
) -> None:
    voice_id, pin = sorted(voicerepo.packaged_pins().items())[0]
    pin = voicerepo.settled(tmp_path, pin)
    cached = (
        tmp_path
        / voicerepo.MANIFEST_CACHE_DIRNAME
        / pin.hf_repo.replace("/", "--")
        / pin.revision
        / REPO_MANIFEST_NAME
    )
    cached.parent.mkdir(parents=True)
    cached.write_text(REPO_GOOD, encoding="utf-8")
    assert voicecatalog.declared_voice_backends(voice_id, tmp_path) == [
        "cuda-linux",
        "mlx-darwin",
    ]


def test_merge_takes_the_machine_facts_from_the_footprint_itself() -> None:
    repo = parse_repo_manifest(REPO_GOOD, Path(REPO_MANIFEST_NAME))
    assert set(repo.voice) == {
        "display", "kind", "narrator_engine", "language", "sample_rate",
    }
    footprint = EngineFootprint(
        engine=HIGGS_V3,
        memory_bytes_estimate=19_000_000_000,
        estimate_basis="declared",
        estimate_note="the server's reservation",
        max_num_seqs=16,
        max_num_seqs_note="stage 0's width",
        mem_fraction=0.5,
        mem_fraction_note="measured under the card",
        context_length=8192,
        context_length_note="holds a long rung",
    )
    pin = Pin(id="probe", hf_repo="owner/name", revision="a" * 40, path=Path("pins.toml"))
    voice = merge(repo, pin, footprint)
    assert voice.serving is not None
    assert voice.serving.to_document() == {
        key: value
        for key, value in footprint.to_dict().items()
        if key in voices.SERVING_KEYS
    }
    assert voice.serving.stall_guard_basis == "default"
    assert voice.serving.stall_guard_env == "37,1,20,16"
    for spec in voice.backends.values():
        assert (spec.memory_bytes_estimate, spec.estimate_basis, spec.estimate_note) == (
            19_000_000_000, "declared", "the server's reservation",
        )
        assert (spec.hf_repo, spec.revision) == ("owner/name", "a" * 40)


def test_the_engine_error_lives_in_errors_and_the_engines_package_re_exports_it() -> None:
    assert errors.EngineError is EngineError
    assert issubclass(EngineError, CrucibleError)
    assert issubclass(NarratorVoicesError, EngineError)
    assert "engines" not in _imported_by("narratorvoices")
