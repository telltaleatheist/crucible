"""The `form` member of a chat or decision: which form of a model to serve it with.

A model whose backend block states FORMS (docs/FITS-AND-THE-CARD.md section 8) is served
as the best form this host's card holds; a caller that names only the model gets that.
`form` is the granular control on top (Owen, 2026-10-10: "crucible handles serving the
model, with more granular controls available from the calling app if they want it"):

- omitted: whichever form is resident answers, and a load made for the call loads the
  form this card takes;
- named: that form answers. Another form of the same id on the card is a reload, as
  another model is; the load is the one `load-model` makes with `params.form`;
- a name the model does not have is refused `unknown_form`, listing the forms it has,
  before the call waits or anything is loaded.

`form` is Crucible's, never the engine's: it is taken off the body before the body is
forwarded.
"""

from __future__ import annotations

from typing import Any

from .errors import ApiError
from .manifests import ManifestError, UnknownForm, load_manifest

FORM_KEY = "form"


def take_form(body: dict[str, Any]) -> str | None:
    """The body's `form`, taken off it; None when it is absent."""
    if FORM_KEY not in body:
        return None
    form = body.pop(FORM_KEY)
    if not isinstance(form, str) or form == "":
        raise ApiError(
            400,
            "invalid_request",
            f'"form" names a form of the model, a non-empty string (GET /v1/models lists '
            f"each model's forms); got {form!r}. Leave it out to take the form this "
            "server's card holds",
            {"form": form},
        )
    return form


def refuse_unknown_form(model: str, form: str | None, backend_kind: str) -> None:
    """Refuse a form the model does not have, by name, before anything waits or loads.
    A model with no manifest, or no block here, is left to the door's own refusal."""
    if form is None:
        return
    try:
        manifest = load_manifest(model)
    except ManifestError:
        return
    if not manifest.supports(backend_kind):
        return
    try:
        manifest.spec(backend_kind, form)
    except UnknownForm as exc:
        raise ApiError(400, exc.code, str(exc), exc.details()) from None


def refuse_upstream_form(model: str) -> None:
    raise ApiError(
        400,
        UnknownForm.code,
        f"{model!r} is an upstream model; forms are a Crucible model's, chosen by its "
        "manifest on this server. Leave \"form\" out",
        {"model": model},
    )


__all__ = ["FORM_KEY", "refuse_unknown_form", "refuse_upstream_form", "take_form"]
