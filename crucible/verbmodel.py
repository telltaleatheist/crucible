"""Which model serves a verb's request: the sizing ruling's precedence
(docs/VERB-SIZING.md section 1a.3), applied at the door of a verb that names no engine.

First that applies: an exact `model` in the request, then a `max_params_b` ceiling in the
request, then the user's per-verb choice in Settings, then Crucible's own pick. The last
two are what this server registered (its capability record's row, which holds the
Settings choice), so the door reads the row; a ceiling decides again, against this card,
with the goal lowered to it. A named model the estimate says does not fit is refused
with the verb's own pick named (only a model the USER configured in Settings is tried
past the estimate, section 1a.4). A model in an optional package this server has not
installed is refused by name, whichever way it was reached.

Every refusal here is the caller's or the operator's to repair, so each one is by name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .capabilityclasses import BY_NAME, CapabilityClass, Goal
from .cardfacts import card_for
from .errors import ApiError
from .fit import Candidate
from .manifests import ManifestError, UnknownForm, load_manifest
from .memorybudget import available_bytes, gib_text
from .packages import package_not_installed
from .verdict import decide_capabilities

MODEL = "model"
CEILING = "max_params_b"
REGISTERED = "registered"


@dataclass(frozen=True)
class VerbModel:
    """The model a request is served by, and which rule chose it."""

    model: str
    form: str | None
    chosen_by: str
    """`model` (the request named it), `max_params_b` (the request's ceiling picked it)
    or `registered` (this server's record: the Settings choice or Crucible's pick)."""


def _candidate_ids(entry: CapabilityClass, backend_kind: str) -> list[str]:
    assert entry.candidates is not None
    return [candidate.id for candidate in entry.candidates(backend_kind)]


def _refuse_foreign(entry: CapabilityClass, model: str, backend_kind: str) -> None:
    served = _candidate_ids(entry, backend_kind)
    if model in served:
        return
    raise ApiError(
        400,
        "model_not_for_verb",
        f"{model!r} does not serve {entry.name} on this {backend_kind} server; the models "
        f"that do are {served}",
        {"model": model, "verb": entry.name, "models": served},
    )


def _refuse_uninstalled_package(entry: CapabilityClass, config: Any, model: str) -> None:
    try:
        package = load_manifest(model).package
    except ManifestError:
        return
    if package is not None and package not in config.packages:
        raise package_not_installed(model, package, f"{entry.plainly} with it")


def _budget(config: Any, backend: Any) -> int:
    return available_bytes(backend.gpu.vram_bytes, config.desktop_allowance_bytes)


def _refuse_too_big(
    entry: CapabilityClass, config: Any, backend: Any, model: str, form: str | None
) -> None:
    """A named model whose estimate does not fit this card is refused naming the verb's
    own pick, unless it is the model the user chose for the verb in Settings."""
    if config.local_model(entry.name) == model:
        return
    manifest = load_manifest(model)
    try:
        candidate = Candidate.of(manifest, backend.kind, form)
    except UnknownForm as exc:
        raise ApiError(400, exc.code, str(exc), exc.details()) from None
    budget = _budget(config, backend)
    if candidate.holds(entry.work, budget):
        return
    row = None if config.capability is None else config.capability.row(entry.name)
    pick = row.selected if row is not None and row.enabled else None
    need = candidate.need_bytes(entry.work)
    raise ApiError(
        409,
        "model_does_not_fit",
        f"{model!r} needs {gib_text(need)} for {entry.name} and this server's card gives "
        f"a model {gib_text(budget)}, so it is not tried for a request; "
        + (
            f"{entry.name} here is served by {pick!r} (send no model, or that one)"
            if pick
            else f"nothing is registered for {entry.name} here"
        )
        + ". A model a person chooses for the verb in Settings is tried past the estimate",
        {"model": model, "need_bytes": need, "available_bytes": budget, "registered": pick},
    )


def _ceiling_pick(
    entry: CapabilityClass, config: Any, backend: Any, ceiling: float
) -> str:
    """The verb's automatic pick on this card with its goal lowered to the request's
    ceiling (rule 8): the biggest that fits at or below it."""
    assert entry.goal is not None and entry.candidates is not None
    sizes = sorted({c.params_b for c in entry.candidates(backend.kind) if c.params_b is not None})
    if not sizes or sizes[0] > ceiling:
        raise ApiError(
            409,
            "nothing_fits_ceiling",
            f"nothing serves {entry.name} at or below {ceiling:g}B on this server: the "
            + (
                f"smallest model that does is {sizes[0]:g}B"
                if sizes
                else f"build ships no model for it on {backend.kind}"
            ),
            {"verb": entry.name, "max_params_b": ceiling},
        )
    lowered = replace(
        entry,
        goal=Goal(
            params_b=min(ceiling, entry.goal.params_b),
            source=f"the request's max_params_b, {ceiling:g}",
        ),
    )
    decision = decide_capabilities(
        lowered,
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        gpu_vendor=backend.gpu.vendor,
        chosen=None,
        audio_low_vram=config.audio_low_vram,
        card=card_for(config.home, backend.gpu),
        packages=config.packages,
    )
    if not decision.enabled:
        raise ApiError(
            409,
            "nothing_fits_ceiling",
            f"nothing serves {entry.name} at or below {ceiling:g}B on this server: "
            f"{decision.reason}",
            {"verb": entry.name, "max_params_b": ceiling},
        )
    return decision.selected


def _registered(entry: CapabilityClass, config: Any) -> str:
    record = config.capability
    row = None if record is None else record.row(entry.name)
    if entry.package is not None and entry.package not in config.packages and (
        row is None or not row.enabled
    ):
        raise ApiError(
            409,
            "package_not_installed",
            f"this server cannot {entry.plainly}: its models are the optional "
            f"{entry.package} package, which it has not installed. `crucible install "
            f"{entry.package}` on the server pulls them and turns it on"
            + (
                "; meanwhile a request may name a decide model (`model`) to rerank with"
                if entry.name == "rerank"
                else ""
            ),
            {"verb": entry.name, "package": entry.package},
        )
    if row is None:
        raise ApiError(
            503,
            "capability_undecided",
            f"this request names no model, and this server has registered nothing for "
            f"{entry.name}: its capability record "
            + ("does not exist" if record is None else f"predates the {entry.name} class")
            + ". Run `crucible capability --write`, or name a `model`",
            {"capability": entry.name},
        )
    if not row.enabled:
        raise ApiError(
            409,
            "capability_disabled",
            f"this request names no model, and this server cannot {entry.plainly}: "
            f"{row.reason}",
            {"capability": entry.name, "shortfall_bytes": row.shortfall_bytes},
        )
    return row.selected


def resolve(
    verb: str,
    config: Any,
    backend: Any,
    *,
    model: str | None,
    form: str | None,
    max_params_b: float | None,
    resident: str | None,
) -> VerbModel:
    """The model this request is served by, every refusal made before anything waits.
    `resident` is the model on the card now: one that is loaded is held, whatever its
    estimate says."""
    entry = BY_NAME[verb]
    if model is not None:
        _refuse_foreign(entry, model, backend.kind)
        _refuse_uninstalled_package(entry, config, model)
        if model != resident:
            _refuse_too_big(entry, config, backend, model, form)
        return VerbModel(model=model, form=form, chosen_by=MODEL)
    if form is not None:
        raise ApiError(
            400,
            "form_without_model",
            "`form` names a form of a named `model`; send the model with it, or neither "
            "to be served by what this server registered",
            {"form": form},
        )
    if max_params_b is not None:
        picked = _ceiling_pick(entry, config, backend, max_params_b)
        _refuse_uninstalled_package(entry, config, picked)
        return VerbModel(model=picked, form=None, chosen_by=CEILING)
    picked = _registered(entry, config)
    _refuse_uninstalled_package(entry, config, picked)
    return VerbModel(model=picked, form=None, chosen_by=REGISTERED)


__all__ = ["CEILING", "MODEL", "REGISTERED", "VerbModel", "resolve"]
