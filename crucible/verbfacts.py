"""What an app's picker reads about the embed and rerank verbs: which models serve them,
what each model's vectors or scores are, and the per-call limits, on `GET /v1/models`
(each row's `verbs`, `embed`, `rerank`, `package`) and `GET /v1/info` (`verbs`)."""

from __future__ import annotations

from typing import Any

from .capabilityclasses import BY_NAME, models_by_class
from .embed import MAX_EMBED_INPUTS
from .rerank import GENERAL_INSTRUCTION, GENERAL_TEMPLATE, MAX_DOCUMENTS

VERBS: tuple[str, ...] = ("embed", "rerank")


def model_facts(
    manifest: Any, served_by: dict[str, set[str]], served_context: int | None, packages: frozenset[str]
) -> dict[str, Any]:
    """A `GET /v1/models` row's verb facts: the classes the model serves, its package,
    and for an embedding or rerank model what it writes and how much a call may carry
    (the served context: the resident engine's, else what a load starts it at)."""
    verbs = sorted(name for name, ids in served_by.items() if manifest.id in ids)
    embed = None
    if manifest.embed is not None:
        embed = {
            **manifest.embed.to_dict(),
            "max_inputs": MAX_EMBED_INPUTS,
            "max_input_tokens": served_context,
        }
    rerank = None
    if manifest.rerank is not None:
        rerank = {**manifest.rerank.to_dict(), "max_documents": MAX_DOCUMENTS,
                  "max_tokens": served_context}
    elif "rerank" in verbs:
        rerank = {
            "template": GENERAL_TEMPLATE,
            "default_instruction": GENERAL_INSTRUCTION,
            "max_documents": MAX_DOCUMENTS,
            "max_tokens": served_context,
        }
    package = manifest.package
    return {
        "verbs": verbs,
        "package": package,
        "package_installed": None if package is None else package in packages,
        "embed": embed,
        "rerank": rerank,
    }


def _lineup(entry: Any, backend_kind: str) -> list[str]:
    """The models a request may name for the verb here, in the order the automatic pick
    ranks them, then those above its goal (a request may still name one that fits)."""
    found = entry.candidates(backend_kind)
    ranked = [c.id for c in entry.pick_order(found)]
    return ranked + [c.id for c in found if c.id not in ranked]


def info(config: Any, backend_kind: str) -> dict[str, Any]:
    """`GET /v1/info`'s `verbs`: per verb, whether this server serves it now and with
    what, its package, the models a request may name, and the per-call limits."""
    record = config.capability
    found: dict[str, Any] = {}
    for verb in VERBS:
        entry = BY_NAME[verb]
        assert entry.candidates is not None
        row = None if record is None else record.row(verb)
        found[verb] = {
            "route": f"POST /v1/{verb}",
            "available": bool(row is not None and row.enabled),
            "registered": row.selected if row is not None and row.enabled else None,
            "reason": None if row is None else row.reason,
            "package": entry.package,
            "package_installed": entry.package is None or entry.package in config.packages,
            "models": _lineup(entry, backend_kind),
            "limits": (
                {"max_inputs": MAX_EMBED_INPUTS}
                if verb == "embed"
                else {"max_documents": MAX_DOCUMENTS}
            ),
        }
    return found


def served_by_class() -> dict[str, set[str]]:
    return models_by_class()


__all__ = ["VERBS", "info", "model_facts", "served_by_class"]
