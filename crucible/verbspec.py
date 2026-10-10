"""What a model manifest says about the two verbs whose input format belongs to the model:
`embed` ([embed]) and `rerank` ([rerank]) (docs/VERB-SIZING.md section 9).

The format is a fact about the weights, so it lives in the model's manifest and nowhere
else (model identity belongs to Crucible; docs/ARCHITECTURE.md R1): an app sends a query or
a document and never the prompt around it. Templates name their slots `{text}`,
`{instruction}`, `{query}` and `{document}`; a slot is filled once, with the value as sent,
and the value is never read for slots of its own, so a brace in a document is a brace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import CrucibleError
from .tomltable import check_table

TEXT = "text"
INSTRUCTION = "instruction"
QUERY = "query"
DOCUMENT = "document"

SLOT = re.compile(r"\{([a-z_]*)\}")
BRACES = re.compile(r"[{}]")

INPUT_TYPES: tuple[str, ...] = ("query", "document")

EMBED_POOLINGS: frozenset[str] = frozenset({"last"})
"""Pooling a Crucible embedding reads: the hidden state at the input's last token (the
pooling every engine here is started with, and the one Qwen3-Embedding is trained for)."""


class VerbSpecError(CrucibleError):
    ...


def fill(template: str, values: dict[str, str]) -> str:
    """`template` with each `{slot}` replaced by its value, the values never scanned for
    slots themselves."""
    parts: list[str] = []
    at = 0
    for found in SLOT.finditer(template):
        parts.append(template[at:found.start()])
        parts.append(values[found.group(1)])
        at = found.end()
    parts.append(template[at:])
    return "".join(parts)


def _slots(where: str, key: str, template: str, allowed: frozenset[str]) -> list[str]:
    named = SLOT.findall(template)
    unknown = sorted(set(named) - allowed)
    if unknown:
        raise VerbSpecError(
            f"{where}: {key} names slot(s) {unknown}; it may name {sorted(allowed)}"
        )
    if BRACES.search(SLOT.sub("", template)):
        raise VerbSpecError(
            f"{where}: {key} has a brace that is not a slot; a template's braces are its "
            f"slots ({sorted(allowed)}) and nothing else"
        )
    twice = sorted({name for name in named if named.count(name) > 1})
    if twice:
        raise VerbSpecError(f"{where}: {key} names {twice} more than once")
    return named


def _needs(where: str, key: str, named: list[str], slot: str) -> None:
    if slot not in named:
        raise VerbSpecError(f"{where}: {key} has no {{{slot}}} slot; it is where the {slot} goes")


def _non_empty(where: str, table: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in table and not table[key].strip():
            raise VerbSpecError(
                f"{where}: {key} is empty; a fact nobody wrote is said by omitting the key"
            )


@dataclass(frozen=True)
class EmbedSpec:
    """How a model's input is written and its vector read."""

    dimensions: int
    """The vector the model writes, in floats."""
    min_dimensions: int | None
    """The smallest prefix of the vector the model is trained to be used at (Matryoshka
    representation learning): a request's `dimensions` may be any size from this to
    `dimensions`, and the prefix is normalised again. None: the model is not trained for
    it, and a request's `dimensions` other than `dimensions` is refused."""
    pooling: str
    """Which hidden state is the vector: `last`, the input's last token."""
    query: str
    """A query as the model reads it: `{text}`, and `{instruction}` where the model takes
    one."""
    document: str
    """A document as the model reads it: `{text}`."""
    default_instruction: str | None
    """The instruction a query is written with when the request states none; None when
    `query` takes none."""
    source: str
    """Where the format was read."""

    def takes_instruction(self, input_type: str) -> bool:
        return input_type == "query" and INSTRUCTION in SLOT.findall(self.query)

    def render(self, input_type: str, text: str, instruction: str | None) -> str:
        if input_type == "document":
            return fill(self.document, {TEXT: text})
        if not self.takes_instruction(input_type):
            return fill(self.query, {TEXT: text})
        chosen = instruction if instruction is not None else self.default_instruction
        assert chosen is not None
        return fill(self.query, {TEXT: text, INSTRUCTION: chosen})

    def dimensions_allowed(self) -> tuple[int, int]:
        return (self.dimensions if self.min_dimensions is None else self.min_dimensions,
                self.dimensions)

    def to_dict(self) -> dict[str, Any]:
        low, high = self.dimensions_allowed()
        return {
            "dimensions": self.dimensions,
            "dimensions_range": [low, high],
            "matryoshka": self.min_dimensions is not None,
            "pooling": self.pooling,
            "normalized": True,
            "input_types": list(INPUT_TYPES),
            "query_takes_instruction": self.takes_instruction("query"),
            "default_instruction": self.default_instruction,
            "query_template": self.query,
            "document_template": self.document,
            "source": self.source,
        }


@dataclass(frozen=True)
class RerankSpec:
    """A dedicated reranker's own prompt: the part every document shares (the instruction
    and the query), the part each document adds, and the two replies whose probabilities
    are compared."""

    prefix: str
    """`{instruction}` and `{query}`: read once for every document of a request."""
    document: str
    """`{document}`, and everything after it up to where the model's reply opens."""
    yes: str
    no: str
    default_instruction: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": "model",
            "prefix_template": self.prefix,
            "document_template": self.document,
            "yes": self.yes,
            "no": self.no,
            "default_instruction": self.default_instruction,
            "source": self.source,
        }


_EMBED_REQUIRED: dict[str, Any] = {
    "dimensions": int,
    "pooling": str,
    "query": str,
    "document": str,
    "source": str,
}
_EMBED_OPTIONAL: dict[str, Any] = {
    "min_dimensions": int,
    "default_instruction": str,
}


def _embed_dimensions(where: str, table: dict[str, Any]) -> tuple[int, int | None]:
    dimensions = table["dimensions"]
    if dimensions < 1:
        raise VerbSpecError(f"{where}: dimensions must be positive, got {dimensions}")
    low = table.get("min_dimensions")
    if low is not None and not 1 <= low < dimensions:
        raise VerbSpecError(
            f"{where}: min_dimensions is {low}; it is the smallest prefix the model is "
            f"trained for, from 1 to below dimensions ({dimensions}). A model not "
            "trained for prefixes states no min_dimensions"
        )
    return dimensions, low


def parse_embed(table: Any, where: str) -> EmbedSpec:
    if not isinstance(table, dict):
        raise VerbSpecError(f"{where}: must be a table")
    check_table(where, table, _EMBED_REQUIRED, _EMBED_OPTIONAL, error=VerbSpecError)
    _non_empty(where, table, ("query", "document", "source", "default_instruction"))
    dimensions, low = _embed_dimensions(where, table)
    if table["pooling"] not in EMBED_POOLINGS:
        raise VerbSpecError(
            f"{where}: pooling {table['pooling']!r}; Crucible reads {sorted(EMBED_POOLINGS)}"
        )
    query = _slots(where, "query", table["query"], frozenset({TEXT, INSTRUCTION}))
    _needs(where, "query", query, TEXT)
    document = _slots(where, "document", table["document"], frozenset({TEXT}))
    _needs(where, "document", document, TEXT)
    takes = INSTRUCTION in query
    default = table.get("default_instruction")
    if takes and default is None:
        raise VerbSpecError(
            f"{where}: query takes an {{instruction}} and there is no "
            "default_instruction, the one a query is written with when a request states none"
        )
    if not takes and default is not None:
        raise VerbSpecError(
            f"{where}: default_instruction is stated and query takes no {{instruction}}"
        )
    return EmbedSpec(
        dimensions=dimensions,
        min_dimensions=low,
        pooling=table["pooling"],
        query=table["query"],
        document=table["document"],
        default_instruction=default,
        source=table["source"],
    )


_RERANK_REQUIRED: dict[str, Any] = {
    "prefix": str,
    "document": str,
    "yes": str,
    "no": str,
    "default_instruction": str,
    "source": str,
}


def parse_rerank(table: Any, where: str) -> RerankSpec:
    if not isinstance(table, dict):
        raise VerbSpecError(f"{where}: must be a table")
    check_table(where, table, _RERANK_REQUIRED, error=VerbSpecError)
    _non_empty(where, table, tuple(_RERANK_REQUIRED))
    prefix = _slots(where, "prefix", table["prefix"], frozenset({INSTRUCTION, QUERY}))
    _needs(where, "prefix", prefix, QUERY)
    _needs(where, "prefix", prefix, INSTRUCTION)
    document = _slots(where, "document", table["document"], frozenset({DOCUMENT}))
    _needs(where, "document", document, DOCUMENT)
    if table["yes"] == table["no"]:
        raise VerbSpecError(f"{where}: yes and no are both {table['yes']!r}")
    return RerankSpec(
        prefix=table["prefix"],
        document=table["document"],
        yes=table["yes"],
        no=table["no"],
        default_instruction=table["default_instruction"],
        source=table["source"],
    )


__all__ = [
    "EMBED_POOLINGS",
    "EmbedSpec",
    "INPUT_TYPES",
    "RerankSpec",
    "VerbSpecError",
    "fill",
    "parse_embed",
    "parse_rerank",
]
