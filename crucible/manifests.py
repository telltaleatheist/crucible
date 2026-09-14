"""Model manifests — `models/<id>.toml` (PHASE2-LLM.md section 1).

One file per Crucible model id. The id is stable across backends; the weights
differ per backend, so each manifest carries one `[backends.<kind>]` block per
backend it can be served on.

Validation is strict on purpose. An unknown key is a refusal, not a warning: a
manifest with `memory_bytes_estimat` in it must not quietly load with no estimate
and let the guard wave a 27B onto a 24 GB card.

Where the manifests live
------------------------
`models/` sits beside the `crucible` package in the checkout, exactly as the
contract writes it. `manifests_dir()` resolves it there, and honours
`$CRUCIBLE_MODELS_DIR` so a test can point at a fixture directory. If neither
exists the loader refuses by name; it never falls back to "no models".
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError

MODELS_DIR_ENV = "CRUCIBLE_MODELS_DIR"

#: Which engine each backend is allowed to name. A manifest that pairs them any
#: other way is a manifest bug, not a runtime decision.
BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "vllm",
    MLX_DARWIN: "mlx-lm",
}

#: What a client may put in a chat request's content parts for this model
#: (PHASE3-VLM.md section 2). It is a statement about what Crucible OFFERS on
#: this server, not about what the checkpoint could theoretically do:
#: `qwen3.5-9b` has a vision tower and is served text-only here, and says
#: `["text"]` because a client that sends it a page gets an engine error rather
#: than a reading.
MODALITIES: frozenset[str] = frozenset({"text", "image"})

#: vLLM's flag for "do not budget for an image while profiling". It does not make
#: an engine text-only; it only stops it RESERVING for one image of the maximum
#: feature size, which on `qwen3.5-9b` was MEASURED at 1.90 GiB of the budget and
#: is the difference between a KV pool and no KV pool at all
#: (models/qwen3.5-9b.toml). That manifest's comment has always carried the
#: condition — "if the `llm` proxy is ever given image input, this line must come
#: out and the utilisation be measured again" — and `modalities` is the first
#: thing in the repo that can say when that day has arrived. So the loader now
#: refuses the pair rather than trusting a reviewer to remember, because the
#: failure it prevents is silent: an image-capable model started with this flag
#: loads, serves, and then meets a real page with no reservation behind it.
SKIP_MM_PROFILING = "--skip-mm-profiling"

#: What a `[defaults]` table may state, and the type each key takes.
#:
#: **Only keys an engine actually honours.** Every one of these reaches vLLM's
#: and mlx-lm's OpenAI chat surface under this exact name — `temperature`,
#: `top_p`, `top_k`, `max_tokens` and `repetition_penalty` are sampling fields
#: both accept, and `thinking` is the one that is not a wire field at all: it
#: becomes `chat_template_kwargs: {"enable_thinking": …}`, which PHASE2-LLM.md
#: section 5 already records as read per request by both engines.
#:
#: A key outside this table is a REFUSAL naming it, for the reason every other
#: manifest key is: a `[defaults]` block with `temperture` in it must not load
#: with no temperature and leave a reader believing one was set. And a knob no
#: engine reads would be worse than useless — it would be a number in a file
#: that looks like it is doing something.
#:
#: The float-typed keys accept an int too, because TOML reads `temperature = 0`
#: as an integer and "zero temperature" is exactly the value a page reader
#: wants. They are stored as floats.
DEFAULTS_KEYS: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "max_tokens": int,
    "repetition_penalty": float,
    "thinking": bool,
}

#: Which of those are sampling fields sent under their own name, in the order a
#: row reports them. `thinking` is deliberately absent: it travels inside
#: `chat_template_kwargs` and is applied by `crucible/sampling.py`.
DEFAULTS_WIRE_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
)

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "params_b": int,
    "context_default": int,
    # Required, not defaulted to `["text"]`, for the reason every other key in
    # this table is required: a manifest that forgets it must not quietly load as
    # text-only and have a page reader refused at request time with an error
    # about content parts, several layers away from the file that was wrong.
    "modalities": list,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "memory_bytes_estimate": int,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    "engine_args": list,
    # A context this backend can actually hold, when the model's own number is
    # not one it can. `[model] context_default` is what the model is FOR; this is
    # what a particular accelerator has room for, and the two are allowed to
    # disagree — `qwen3.8-27b-4bit` wants Owen's 98304 and gets it on 64 GB of
    # unified memory, while 98304 of its KV is 7.9 GiB the 3090 Ti does not have
    # once the weights are down. Absent means "the model's number"; it is never
    # a silent default.
    "context_default": int,
}

def fingerprint(model_id: str, revision: str) -> str:
    """`qwen3.5-9b@<sha>` — how a model's identity is written down.

    The bare id is not enough to identify bytes. Foundry hashes the served model
    id into its cleanup cache key and BookForge stamps it into a book's OPF
    (CLIENT-SURFACES.md section 6.5), so "what cleaned this book" has to name the
    pin as well as the model, or a manifest that moves to a new revision goes on
    answering from a cache built by the old one.
    """
    return f"{model_id}@{revision}"


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class ManifestError(CrucibleError):
    """A manifest is missing, unreadable, or does not say what it must say."""


@dataclass(frozen=True)
class BackendSpec:
    """One `[backends.<kind>]` block."""

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    engine_args: tuple[str, ...]
    #: This backend's own context, or None to use the model's.
    context_default: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "engine_args": list(self.engine_args),
            "context_default": self.context_default,
        }


@dataclass(frozen=True)
class ModelDefaults:
    """`[defaults]` — what this model is answered with when a request is silent.

    PHASE2-LLM.md section 9. Every field is `None` when the manifest did not
    state it, and `None` means **the engine's own default**, never a value
    Crucible picked: a server that filled one in would be substituting its guess
    for the engine's measurement, and nothing in the answer would say so.

    The precedence is one line and it is the whole contract: **a field the
    request STATES wins; a field the request omits takes the manifest's; a field
    neither states is the engine's.** `crucible/sampling.py` is the one place
    that is applied, and it reports which of the three each effective value came
    from, because a default that cannot be seen is a default nobody can debug.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    thinking: bool | None = None

    def stated(self) -> dict[str, Any]:
        """Only the keys this manifest actually states, in `DEFAULTS_KEYS` order.

        The distinction this method exists for: an absent key and a key set to
        the engine's own value are different manifests, and only the second one
        is a decision somebody made.
        """
        values: dict[str, Any] = {}
        for key in DEFAULTS_KEYS:
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        return values

    def to_dict(self) -> dict[str, Any]:
        """All six keys, `null` where this manifest states nothing.

        Never the `stated()` subset: a row whose keys come and go would make
        "this model has no default temperature" and "this build predates the
        field" read the same, which is the mistake `[jobs] enable_*` had to be
        rescued from in `crucible/config.py`.
        """
        return {key: getattr(self, key) for key in DEFAULTS_KEYS}


#: A manifest that states nothing. Shared rather than rebuilt, and it is what
#: every manifest without a `[defaults]` table carries — so the application code
#: has no "is there a defaults table" branch at all.
NO_DEFAULTS = ModelDefaults()


@dataclass(frozen=True)
class ModelManifest:
    id: str
    family: str
    params_b: int
    context_default: int
    #: In the order the manifest wrote them, so a row reads the way the file does.
    modalities: tuple[str, ...]
    backends: dict[str, BackendSpec]
    path: Path
    #: `[defaults]`, or `NO_DEFAULTS` when the manifest has no such table. A
    #: MODEL-level fact and not a per-backend one: the reason a model wants
    #: `thinking = false` is what the model does with a prompt, which is the
    #: same on both cards. If a backend ever needs its own, it needs its own
    #: argument first.
    defaults: ModelDefaults = NO_DEFAULTS

    #: Which subtree of `~/.crucible/` this thing's weights live under. A model id
    #: and a voice id are separate namespaces and must not be able to collide on
    #: disk — see `crucible/weights.py`.
    weights_family = "models"

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def context_for(self, backend_kind: str) -> int:
        """The context THIS backend serves: its own, or the model's.

        Everything that names a context — vLLM's `--max-model-len`, the resident
        model's row, `/v1/models` — must ask this and not read
        `self.context_default` directly, or a backend's override would be
        reported by one and ignored by the other.
        """
        found = self.backends.get(backend_kind)
        if found is None or found.context_default is None:
            return self.context_default
        return found.context_default

    def fingerprint_for(self, backend_kind: str) -> str | None:
        """`<id>@<revision>` for this backend, or None where there is no block.

        None rather than the bare id: a host with no block for this model has no
        revision to name here, and an unpinned fingerprint would be a worse
        record than no fingerprint — it would look like one.
        """
        found = self.backends.get(backend_kind)
        return None if found is None else fingerprint(self.id, found.revision)

    def spec(self, backend_kind: str) -> BackendSpec:
        """The block for `backend_kind`, or a named refusal."""
        found = self.backends.get(backend_kind)
        if found is None:
            raise ManifestError(
                f"model {self.id!r} has no {backend_kind} block; {self.path.name} "
                f"declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "params_b": self.params_b,
            "context_default": self.context_default,
            "modalities": list(self.modalities),
            "defaults": self.defaults.to_dict(),
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


# ------------------------------------------------------------------ locating


def manifests_dir() -> Path:
    """Where `models/*.toml` live on this host. Refuses by name if absent."""
    override = os.environ.get(MODELS_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise ManifestError(
                f"{MODELS_DIR_ENV}={override!r} is not a directory"
            )
        return path
    # models/ sits beside the crucible package in the checkout.
    path = Path(__file__).resolve().parent.parent / "models"
    if not path.is_dir():
        raise ManifestError(
            f"no model manifests at {path}; crucible must run from a checkout "
            f"(pip install -e .) or ${MODELS_DIR_ENV} must point at the manifests"
        )
    return path


# ------------------------------------------------------------------ checking


def check_table(
    where: str,
    table: dict[str, Any],
    required: dict[str, type],
    optional: dict[str, type],
    *,
    error: type[CrucibleError] = ManifestError,
) -> None:
    """Every required key present and correctly typed; no key that is not listed.

    `error` is the exception class to refuse with, because the voice manifests
    (`crucible/voices.py`) are the same kind of file held to the same strictness
    and must refuse in their own vocabulary — a reader told "manifest" about a
    voice file goes looking in `models/`.
    """
    allowed = set(required) | set(optional)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise error(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(allowed)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise error(f"{where}: missing required key(s) {missing}")
    for key, kind in {**required, **optional}.items():
        if key not in table:
            continue
        value = table[key]
        wrong = not isinstance(value, kind)
        # bool is a subclass of int; a bool where an int is wanted is still wrong.
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise error(
                f"{where}: {key} must be {kind.__name__}, got "
                f"{type(value).__name__}"
            )


#: The bounds each `[defaults]` key is checked against, as (test, why). They are
#: the engines' own ranges, and a value outside one is refused here rather than
#: at the first request: a manifest is read at startup and a bad number in it
#: should not become a 400 on somebody's book.
_DEFAULT_BOUNDS: dict[str, tuple[Any, str]] = {
    "temperature": (lambda v: v >= 0.0, "must be zero or more (0 is greedy)"),
    "top_p": (lambda v: 0.0 < v <= 1.0, "must be above 0 and at most 1"),
    "top_k": (
        lambda v: v >= 1,
        "must be at least 1; to leave top-k alone, omit the key rather than "
        "writing a number that means 'off' on one engine and nothing on the other",
    ),
    "max_tokens": (lambda v: v >= 1, "must be at least 1"),
    "repetition_penalty": (lambda v: v > 0.0, "must be above 0 (1.0 is no penalty)"),
    "thinking": (lambda v: True, ""),
}


def _parse_defaults(table: Any, path: Path) -> ModelDefaults:
    """`[defaults]`, validated as strictly as every other table in this file.

    Unknown key: refusal naming it. Wrong type: refusal naming both. Out of
    range: refusal naming the bound. An absent table is `NO_DEFAULTS`, which is
    a statement — "this model states none" — and not an unknown.
    """
    where = f"{path.name} [defaults]"
    if not isinstance(table, dict):
        raise ManifestError(f"{where}: must be a table")
    unknown = sorted(set(table) - set(DEFAULTS_KEYS))
    if unknown:
        raise ManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(DEFAULTS_KEYS)} — the only knobs both vLLM and mlx-lm "
            "honour. A key no engine reads would be a number in a file that "
            "looks like it is doing something"
        )
    if not table:
        raise ManifestError(
            f"{where}: the table is empty. A model that states no defaults says "
            "so by having no [defaults] table at all; an empty one reads as a "
            "decision somebody made and then forgot to write down"
        )
    values: dict[str, Any] = {}
    for key, kind in DEFAULTS_KEYS.items():
        if key not in table:
            continue
        value = table[key]
        if kind is bool:
            if not isinstance(value, bool):
                raise ManifestError(
                    f"{where}: {key} must be bool, got {type(value).__name__}"
                )
        elif kind is int:
            # bool is a subclass of int; a bool where an int is wanted is wrong.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ManifestError(
                    f"{where}: {key} must be int, got {type(value).__name__}"
                )
        else:
            # TOML reads `temperature = 0` as an int, and zero temperature is
            # exactly what a page reader wants, so an int is accepted and
            # stored as a float rather than refused on a technicality.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManifestError(
                    f"{where}: {key} must be a number, got {type(value).__name__}"
                )
            value = float(value)
        test, why = _DEFAULT_BOUNDS[key]
        if not test(value):
            raise ManifestError(f"{where}: {key} is {value!r} and {why}")
        values[key] = value
    return ModelDefaults(**values)


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> ModelManifest:
    unknown = sorted(set(document) - {"model", "backends", "defaults"})
    if unknown:
        raise ManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a manifest has "
            "exactly [model], [backends.<kind>] and an optional [defaults]"
        )
    if "model" not in document:
        raise ManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise ManifestError(f"{path.name}: missing every [backends.<kind>] table")

    model = document["model"]
    if not isinstance(model, dict):
        raise ManifestError(f"{path.name}: [model] must be a table")
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, {})

    model_id = model["id"]
    if not _MODEL_ID.match(model_id):
        raise ManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise ManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if model["params_b"] <= 0:
        raise ManifestError(
            f"{path.name}: model.params_b must be positive, got {model['params_b']}"
        )
    if model["context_default"] <= 0:
        raise ManifestError(
            f"{path.name}: model.context_default must be positive, got "
            f"{model['context_default']}"
        )

    modalities = model["modalities"]
    if not modalities:
        raise ManifestError(
            f"{path.name}: model.modalities is empty; a model that accepts no "
            f"input at all is not a model. It takes one or more of "
            f"{sorted(MODALITIES)}"
        )
    for index, entry in enumerate(modalities):
        if not isinstance(entry, str):
            raise ManifestError(
                f"{path.name}: model.modalities[{index}] must be a string, got "
                f"{type(entry).__name__}"
            )
        if entry not in MODALITIES:
            raise ManifestError(
                f"{path.name}: model.modalities[{index}] is {entry!r}; Crucible "
                f"knows {sorted(MODALITIES)}"
            )
    if len(set(modalities)) != len(modalities):
        raise ManifestError(
            f"{path.name}: model.modalities lists a modality twice: {modalities}"
        )

    backends_table = document["backends"]
    if not isinstance(backends_table, dict):
        raise ManifestError(f"{path.name}: [backends] must hold one table per backend")
    if not backends_table:
        raise ManifestError(
            f"{path.name}: no backend blocks; a model nothing can serve is not a model"
        )

    backends: dict[str, BackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [backends.{kind}]"
        if kind not in BACKEND_ENGINES:
            raise ManifestError(
                f"{where}: {kind!r} is not a Crucible backend; the backends are "
                f"{sorted(BACKEND_ENGINES)}"
            )
        if not isinstance(block, dict):
            raise ManifestError(f"{where}: must be a table")
        check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL)

        engine = block["engine"]
        if engine != BACKEND_ENGINES[kind]:
            raise ManifestError(
                f"{where}: engine {engine!r} does not run on {kind}; that backend's "
                f"engine is {BACKEND_ENGINES[kind]!r}"
            )
        if not _HF_REPO.match(block["hf_repo"]):
            raise ManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
            raise ManifestError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        if block["memory_bytes_estimate"] <= 0:
            raise ManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        engine_args = block.get("engine_args", [])
        for index, argument in enumerate(engine_args):
            if not isinstance(argument, str):
                raise ManifestError(
                    f"{where}: engine_args[{index}] must be a string, got "
                    f"{type(argument).__name__}"
                )
        if "image" in modalities and SKIP_MM_PROFILING in engine_args:
            # The one rule that crosses the two tables, and it crosses them
            # because the fact and the flag live apart: what a model is offered
            # for is `[model]`, how its engine is started is `[backends.<kind>]`.
            # A model advertised for images whose engine was told not to profile
            # for one starts, serves, and then meets a real page with nothing
            # reserved behind it — the kind of failure that arrives as an OOM in
            # the middle of somebody's book rather than at load.
            raise ManifestError(
                f"{where}: engine_args carries {SKIP_MM_PROFILING!r} while "
                f"[model] modalities declares 'image'. That flag stops vLLM "
                f"reserving for an image, so it belongs only to a model this "
                f"server serves text-only; take it out and measure "
                f"--gpu-memory-utilization again with the image profiled in "
                f"(PHASE3-VLM.md section 3)"
            )
        backend_context = block.get("context_default")
        if backend_context is not None and backend_context <= 0:
            raise ManifestError(
                f"{where}: context_default must be positive, got {backend_context}"
            )
        backends[kind] = BackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
            engine_args=tuple(engine_args),
            context_default=backend_context,
        )

    return ModelManifest(
        id=model_id,
        family=model["family"],
        params_b=model["params_b"],
        context_default=model["context_default"],
        modalities=tuple(modalities),
        backends=backends,
        path=path,
        defaults=(
            NO_DEFAULTS
            if "defaults" not in document
            else _parse_defaults(document["defaults"], path)
        ),
    )


# ------------------------------------------------------------------- loading


def parse_manifest(text: str, path: Path, expected_id: str) -> ModelManifest:
    """Parse and validate one manifest's text. Raises ManifestError by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_manifest(model_id: str, directory: Path | None = None) -> ModelManifest:
    """Load `models/<model_id>.toml`. Raises ManifestError if it is not there."""
    root = directory if directory is not None else manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise ManifestError(
            f"no manifest for model {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc
    return parse_manifest(text, path, model_id)


def load_all_manifests(directory: Path | None = None) -> dict[str, ModelManifest]:
    """Every manifest this build ships, by id, in id order."""
    root = directory if directory is not None else manifests_dir()
    manifests: dict[str, ModelManifest] = {}
    # By id — `path.stem` — and not by path. The two orders differ whenever one
    # id is a prefix of another, because the extension gets in the way: as whole
    # paths, `qwen3.8-27b-4bit.toml` sorts BEFORE `qwen3.8-27b.toml` ('-' is
    # 0x2D, '.' is 0x2E), while as ids `qwen3.8-27b` comes first. This function's
    # order is what `/v1/models` lists in, so it is the documented one.
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_manifest(path.stem, root)
    return manifests
