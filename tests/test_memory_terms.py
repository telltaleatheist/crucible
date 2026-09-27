from __future__ import annotations

import pytest

from crucible.manifests import (
    MEMORY_BASES,
    MEMORY_TERMS_TOLERANCE,
    ManifestError,
    MemoryTerms,
    load_all_manifests,
    parse_manifest,
)

MINIMAL = """
[model]
id = "probe"
family = "probe"
params_b = 9
context_default = 16384
trained_context = 262144
modalities = ["text"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "probe/probe"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 11_027_502_048
engine_args = []

[backends.cuda-linux.memory]
weights_bytes = 9_527_502_048
overhead_bytes = 963_129_088
kv_bytes_per_token = 32_768
basis = "declared"
measured_at_context = 16384
"""


def parse(text: str):
    from pathlib import Path

    return parse_manifest(text, Path("probe.toml"), "probe")


def test_the_minimal_manifest_parses_and_carries_its_terms() -> None:
    spec = parse(MINIMAL).backends["cuda-linux"]
    assert spec.memory is not None
    assert spec.memory.kv_bytes_per_token == 32_768
    assert spec.memory.basis == "declared"
    assert spec.memory.bytes_for(context=16384, concurrency=1) == (
        spec.memory_bytes_estimate
    )


def test_terms_that_do_not_add_up_to_the_estimate_are_refused() -> None:
    deformed = MINIMAL.replace(
        "weights_bytes = 9_527_502_048", "weights_bytes = 5_527_502_048"
    )
    with pytest.raises(ManifestError) as refusal:
        parse(deformed)
    message = str(refusal.value)
    assert "memory_bytes_estimate says" in message
    assert "11027502048" in message.replace("_", "")
    assert "apart" in message


def test_a_gb_for_gib_slip_is_caught() -> None:
    deformed = MINIMAL.replace(
        "weights_bytes = 9_527_502_048", "weights_bytes = 8_871_244_000"
    )
    with pytest.raises(ManifestError, match="apart"):
        parse(deformed)


def test_drift_inside_the_tolerance_is_allowed() -> None:
    nudged = MINIMAL.replace(
        "weights_bytes = 9_527_502_048", "weights_bytes = 9_727_502_048"
    )
    spec = parse(nudged).backends["cuda-linux"]
    assert spec.memory is not None
    drift = (
        spec.memory.bytes_for(context=16384, concurrency=1)
        - spec.memory_bytes_estimate
    ) / spec.memory_bytes_estimate
    assert 0 < drift < MEMORY_TERMS_TOLERANCE


def test_terms_measured_at_another_context_are_refused() -> None:
    deformed = MINIMAL.replace("measured_at_context = 16384", "measured_at_context = 8192")
    with pytest.raises(ManifestError) as refusal:
        parse(deformed)
    assert "only true at the context it was taken at" in str(refusal.value)


def test_an_unknown_basis_is_refused() -> None:
    deformed = MINIMAL.replace('basis = "declared"', 'basis = "vibes"')
    with pytest.raises(ManifestError, match="basis"):
        parse(deformed)


def test_a_missing_term_is_refused() -> None:
    deformed = MINIMAL.replace("kv_bytes_per_token = 32_768\n", "")
    with pytest.raises(ManifestError, match="kv_bytes_per_token"):
        parse(deformed)


def test_a_zero_overhead_is_allowed_and_a_negative_one_is_not() -> None:
    allowed = MINIMAL.replace("overhead_bytes = 963_129_088", "overhead_bytes = 0")
    allowed = allowed.replace(
        "memory_bytes_estimate = 11_027_502_048",
        "memory_bytes_estimate = 10_064_372_960",
    )
    assert parse(allowed).backends["cuda-linux"].memory.overhead_bytes == 0

    with pytest.raises(ManifestError, match="cannot be negative"):
        parse(MINIMAL.replace("overhead_bytes = 963_129_088", "overhead_bytes = -1"))


def terms() -> MemoryTerms:
    return MemoryTerms(
        weights_bytes=18_983_441_367,
        overhead_bytes=1_546_188_226,
        kv_bytes_per_token=86_251,
        basis="measured",
        measured_at_context=16384,
    )


def test_the_same_bytes_buy_a_short_context_often_or_a_long_one_once() -> None:
    assert terms().bytes_for(context=4096, concurrency=4) == terms().bytes_for(
        context=16384, concurrency=1
    )


def test_max_context_is_the_arithmetic_run_backwards() -> None:
    subject = terms()
    budget = 22_548_578_304
    ceiling = subject.max_context(available_bytes=budget, concurrency=1)
    assert subject.bytes_for(context=ceiling, concurrency=1) <= budget
    assert subject.bytes_for(context=ceiling + 1, concurrency=1) > budget


def test_max_context_halves_when_twice_as_much_is_in_flight() -> None:
    subject = terms()
    budget = 22_548_578_304
    one = subject.max_context(available_bytes=budget, concurrency=1)
    two = subject.max_context(available_bytes=budget, concurrency=2)
    assert two == one // 2


def test_a_card_that_cannot_hold_the_weights_affords_no_context_at_all() -> None:
    assert terms().max_context(available_bytes=4 * 1024 ** 3, concurrency=1) == 0


def test_a_nonsensical_shape_raises_rather_than_returning_a_number() -> None:
    with pytest.raises(ValueError, match="context must be positive"):
        terms().bytes_for(context=0, concurrency=1)
    with pytest.raises(ValueError, match="concurrency must be positive"):
        terms().bytes_for(context=4096, concurrency=0)
    with pytest.raises(ValueError, match="concurrency must be positive"):
        terms().max_context(available_bytes=1, concurrency=0)


def test_every_shipped_block_with_terms_agrees_with_its_own_estimate() -> None:
    checked = 0
    for model_id, manifest in load_all_manifests().items():
        for kind, spec in manifest.backends.items():
            if spec.memory is None:
                continue
            checked += 1
            served = manifest.context_for(kind)
            assert spec.memory.measured_at_context == served, (model_id, kind)
            drift = abs(
                spec.memory.bytes_for(context=served, concurrency=1)
                - spec.memory_bytes_estimate
            ) / spec.memory_bytes_estimate
            assert drift <= MEMORY_TERMS_TOLERANCE, (model_id, kind, drift)
            assert spec.memory.basis in MEMORY_BASES
    assert checked >= 7, f"only {checked} blocks carry terms; the promotion regressed"


def test_the_one_measured_row_is_still_measured() -> None:
    spec = load_all_manifests()["qwen3.8-27b-4bit"].backends["cuda-linux"]
    assert spec.memory is not None
    assert spec.memory.basis == "measured"
    assert spec.memory.kv_bytes_per_token == 86_251
    assert spec.memory.kv_bytes_per_token > 65_536
