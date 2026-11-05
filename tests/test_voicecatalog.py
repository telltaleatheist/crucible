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
from .conftest import configure_box


@pytest.fixture
def sized_box(home: Path) -> Path:
    """A box whose config states the [tts.higgs-v3] footprint: the engine's own Higgs rows
    take their serving width from it, so without one they are refused, not served."""
    configure_box(home, enable_tts=True)
    return home


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


def test_a_base_voice_names_the_aliases_its_catalog_resolved(sized_box: Path) -> None:
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


def test_a_shipped_voice_declares_its_own_backends_and_an_unknown_one_none(
    sized_box: Path,
) -> None:
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


def test_every_higgs_voice_takes_its_serving_width_from_the_box(home: Path) -> None:
    """One owner: the box's [tts.higgs-v3] footprint sizes the engine's own rows exactly as it
    sizes every pinned voice. base.toml carried 16 while owens-pc ran 12 (2026-11-04)."""
    from dataclasses import replace as _replace

    from crucible.narratorengines import HIGGS_V3, declared_tts_footprints

    footprint = {f.engine: f for f in declared_tts_footprints("cuda-linux")}[HIGGS_V3]
    configure_box(
        home, enable_tts=True,
        tts_engines=(_replace(footprint, max_num_seqs=7, max_num_seqs_note="seven, for the test"),),
    )
    engine = voicecatalog.engine_voices()
    assert {"higgs-default", "zeroshot"} <= set(engine)
    for voice_id in ("higgs-default", "zeroshot"):
        serving = engine[voice_id].serving
        assert serving is not None and serving.max_num_seqs == 7, voice_id
        assert serving.max_num_seqs_note == "seven, for the test"


def test_a_box_with_no_higgs_footprint_lists_the_engine_rows_unsized(home: Path) -> None:
    """Listed (the catalog and capability ask what the engine ships, which does not depend
    on this box), with no serving width: nothing is sized from a number the box never said."""
    configure_box(home, tts_engines=())
    engine = voicecatalog.engine_voices()
    assert {"higgs-default", "zeroshot"} <= set(engine)
    assert engine["higgs-default"].serving is None
    assert engine["zeroshot"].serving is None


def test_an_unsized_engine_row_is_refused_at_load_by_name(home: Path) -> None:
    from crucible.config import load_config
    from crucible.errors import ApiError
    from crucible.jobs.tts.common import require_loadable

    from .conftest import FAKE_BACKEND

    configure_box(home, enable_tts=True, tts_engines=())
    with pytest.raises(ApiError) as caught:
        require_loadable(load_config(home), FAKE_BACKEND, "higgs-default")
    assert caught.value.code == "engine_footprint_unset"
    assert "[tts.higgs-v3]" in caught.value.message


def test_a_base_row_that_states_its_own_width_is_refused(tmp_path: Path) -> None:
    from crucible.voices import VoiceError, parse_engine_base

    path = tmp_path / "base.toml"
    text = (
        '[voices.x]\nid = "x"\ndisplay = "x"\nkind = "token"\nnarrator_engine = "higgs-v3"\n'
        '[voices.x.serving]\nmax_num_seqs = 16\nmax_num_seqs_note = "n"\n'
    )
    with pytest.raises(VoiceError, match="second owner"):
        parse_engine_base(text, path, "higgs-v3", None)
