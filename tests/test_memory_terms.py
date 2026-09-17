"""`[backends.<kind>.memory]` — the estimate taken apart into its terms.

docs/FITS-AND-THE-CARD.md. The collapsed `memory_bytes_estimate` answers one
question, at one context, and every capability class is then made to ask that
question whether or not it is the one it has: translate sends a paragraph at a
time, batched and independent, and gets refused for a KV cache it will never
fill.

These tests hold the two things that make the split safe rather than merely
convenient. First, that the terms and the collapsed estimate are COMPARED — two
readings of one thing, parts against whole, and a manifest that lets them drift
is refused by name (ARCHITECTURE.md R1: every defect in that audit was a fact
with two owners and nothing checking them). Second, that the arithmetic the terms
exist for actually works: the same bytes bought as a short context many times
over, or a long one once.
"""

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

#: A whole manifest with one backend block, for the parser tests to deform. The
#: terms add to exactly `memory_bytes_estimate` at 16384, so any drift a test
#: sees is the drift that test introduced.
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
    """The fixture the deformations below start from is itself valid.

    Stated as its own test because every other test here asserts a REFUSAL, and
    a fixture that was broken for some unrelated reason would make all of them
    pass for the wrong reason.
    """
    spec = parse(MINIMAL).backends["cuda-linux"]
    assert spec.memory is not None
    assert spec.memory.kv_bytes_per_token == 32_768
    assert spec.memory.basis == "declared"
    assert spec.memory.bytes_for(context=16384, concurrency=1) == (
        spec.memory_bytes_estimate
    )


def test_terms_that_do_not_add_up_to_the_estimate_are_refused() -> None:
    """The whole point of allowing two owners is that they are compared."""
    deformed = MINIMAL.replace(
        "weights_bytes = 9_527_502_048", "weights_bytes = 5_527_502_048"
    )
    with pytest.raises(ManifestError) as refusal:
        parse(deformed)
    message = str(refusal.value)
    assert "memory_bytes_estimate says" in message
    # BOTH numbers in the sentence, because a refusal that says only "they
    # disagree" leaves the reader to go and compute the two it is talking about.
    assert "11027502048" in message.replace("_", "")
    assert "apart" in message


def test_a_gb_for_gib_slip_is_caught() -> None:
    """The mistake the tolerance was actually sized for.

    17.66 GiB written as 17.66 GB is 7.4% low — inside anything loose and
    outside 5%, which is why 5% is the number.
    """
    deformed = MINIMAL.replace(
        "weights_bytes = 9_527_502_048", "weights_bytes = 8_871_244_000"
    )
    with pytest.raises(ManifestError, match="apart"):
        parse(deformed)


def test_drift_inside_the_tolerance_is_allowed() -> None:
    """Parts measured against a whole measured at a peak are allowed to differ.

    The 27B-4bit block in the real catalog sits 1.4% out for exactly this
    reason, and refusing it would force somebody to round a measurement until
    the parser stopped complaining.
    """
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
    """An estimate is only true at the context it was taken at.

    The defect this catches is real and shipped: a `llama-windows` block with no
    context of its own inherited 98304 — a number `qwen3.8-27b-4bit.toml` calls
    "a Mac fact" — where KV is 6.44 GB against a declared 1.5 GB allowance.
    """
    deformed = MINIMAL.replace("measured_at_context = 16384", "measured_at_context = 8192")
    with pytest.raises(ManifestError) as refusal:
        parse(deformed)
    assert "only true at the context it was taken at" in str(refusal.value)


def test_an_unknown_basis_is_refused() -> None:
    deformed = MINIMAL.replace('basis = "declared"', 'basis = "vibes"')
    with pytest.raises(ManifestError, match="basis"):
        parse(deformed)


def test_a_missing_term_is_refused() -> None:
    """Not defaulted. A term left out is a term somebody supplies from elsewhere."""
    deformed = MINIMAL.replace("kv_bytes_per_token = 32_768\n", "")
    with pytest.raises(ManifestError, match="kv_bytes_per_token"):
        parse(deformed)


def test_a_zero_overhead_is_allowed_and_a_negative_one_is_not() -> None:
    """Zero says "nobody accounted for this", which both 27B blocks mean."""
    allowed = MINIMAL.replace("overhead_bytes = 963_129_088", "overhead_bytes = 0")
    allowed = allowed.replace(
        "memory_bytes_estimate = 11_027_502_048",
        "memory_bytes_estimate = 10_064_372_960",
    )
    assert parse(allowed).backends["cuda-linux"].memory.overhead_bytes == 0

    with pytest.raises(ManifestError, match="cannot be negative"):
        parse(MINIMAL.replace("overhead_bytes = 963_129_088", "overhead_bytes = -1"))


# --------------------------------------------------------------- the arithmetic


def terms() -> MemoryTerms:
    return MemoryTerms(
        weights_bytes=18_983_441_367,
        overhead_bytes=1_546_188_226,
        kv_bytes_per_token=86_251,
        basis="measured",
        measured_at_context=16384,
    )


def test_the_same_bytes_buy_a_short_context_often_or_a_long_one_once() -> None:
    """The reframe, in one assertion.

    Four batched 4096-token translate blocks in flight cost exactly what one
    16384-token request costs. That equality is why a class must be allowed to
    state its own working context instead of inheriting the model's.
    """
    assert terms().bytes_for(context=4096, concurrency=4) == terms().bytes_for(
        context=16384, concurrency=1
    )


def test_max_context_is_the_arithmetic_run_backwards() -> None:
    """What a client should READ instead of guessing a chunk size."""
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
    """Zero, and it must stay a different answer from "your request is too long".

    A model that does not fit at any length is a refusal about the MODEL; a
    request past the ceiling is a refusal about the REQUEST. Rounding the first
    into the second sends somebody off to shorten a paragraph that was never the
    problem.
    """
    assert terms().max_context(available_bytes=4 * 1024 ** 3, concurrency=1) == 0


def test_a_nonsensical_shape_raises_rather_than_returning_a_number() -> None:
    with pytest.raises(ValueError, match="context must be positive"):
        terms().bytes_for(context=0, concurrency=1)
    with pytest.raises(ValueError, match="concurrency must be positive"):
        terms().bytes_for(context=4096, concurrency=0)
    with pytest.raises(ValueError, match="concurrency must be positive"):
        terms().max_context(available_bytes=1, concurrency=0)


# ----------------------------------------------------------- the real catalog


def test_every_shipped_block_with_terms_agrees_with_its_own_estimate() -> None:
    """The catalog itself, not a fixture.

    `parse_manifest` already refuses a disagreement, so this cannot fail while
    the manifests load — which is the point: it states what loading PROVES, so a
    later change that loosens the parser is visible as this test going quiet
    rather than as nothing at all.
    """
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
    """`qwen3.8-27b-4bit` on cuda-linux is the only block a card has answered.

    Named here because it is the row every other row is calibrated against, and
    a change that quietly demoted it to `computed` would take the catalog's only
    evidence that the architecture's KV arithmetic runs 24% light.
    """
    spec = load_all_manifests()["qwen3.8-27b-4bit"].backends["cuda-linux"]
    assert spec.memory is not None
    assert spec.memory.basis == "measured"
    assert spec.memory.kv_bytes_per_token == 86_251
    # 65_536 is what config.json says; the card said otherwise.
    assert spec.memory.kv_bytes_per_token > 65_536
